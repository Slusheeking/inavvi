"""
Sentiment analysis model for news and social media data.

Uses state-of-the-art language models to analyze sentiment with entity-specific
tracking and temporal analysis.
"""
import json
import os
import pickle
import subprocess
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from textblob import TextBlob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from src.config.settings import settings
from src.utils.logging import setup_logger, log_execution_time
from src.utils.redis_client import redis_client

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*`resume_download` is deprecated.*")
warnings.filterwarnings("ignore", message=".*The `return_dict` argument.*")
warnings.filterwarnings("ignore", message=".*Some weights of the model checkpoint.*")

# Set up logger
logger = setup_logger("sentiment_model")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() and settings.advanced.use_gpu else "cpu")
logger.info(f"Using device: {device}")

# Default model configurations
DEFAULT_MODELS = {
    "finbert": {
        "name": "ProsusAI/finbert",
        "labels": ["negative", "neutral", "positive"],
        "max_length": 256,
    },
    "financial_roberta": {
        "name": "yiyanghkust/finbert-tone",
        "labels": ["negative", "neutral", "positive"],
        "max_length": 256,
    },
    "sector_bert": {
        "name": "distilbert-base-uncased",
        "labels": ["negative", "neutral", "positive"],
        "max_length": 256,
    },
}


class SentimentDataset(Dataset):
    """Dataset for sentiment analysis."""

    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, tokenizer=None, max_length: int = 256):
        """
        Initialize the dataset.

        Args:
            texts: List of text samples.
            labels: Optional list of labels.
            tokenizer: Tokenizer to use.
            max_length: Maximum sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary with tokenized inputs and optional labels.
        """
        text = self.texts[idx]
        if self.tokenizer:
            encoding = self.tokenizer(
              text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            item = {key: val.squeeze(0) for key, val in encoding.items()}
        else:
            item = {"text": text}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


class EntitySentimentTracker:
    """Track sentiment for specific entities over time."""

    def __init__(self):
        """Initialize the entity sentiment tracker."""
        self.entity_sentiments = defaultdict(list)
        self.entity_mentions = defaultdict(int)
        self.entity_importance = defaultdict(float)
        self.entity_relationships = defaultdict(lambda: defaultdict(float))

    def add_entity_sentiment(
        self, entity: str, sentiment: float, timestamp: str, source: Optional[str] = None, importance: float = 1.0
    ):
        """
        Add a sentiment record for an entity.

        Args:
            entity: Entity name.
            sentiment: Sentiment score (-1 to 1).
            timestamp: Timestamp of the sentiment.
            source: Source of the sentiment (e.g., news article).
            importance: Importance of this sentiment record.
        """
        self.entity_sentiments[entity].append(
            {"timestamp": timestamp, "sentiment": sentiment, "source": source, "importance": importance}
        )
        self.entity_mentions[entity] += 1
        self.entity_importance[entity] += importance

    def add_entity_relationship(self, entity1: str, entity2: str, strength: float = 1.0):
        """
        Add a relationship between two entities.

        Args:
            entity1: First entity.
            entity2: Second entity.
            strength: Strength of the relationship.
        """
        self.entity_relationships[entity1][entity2] += strength
        self.entity_relationships[entity2][entity1] += strength

    def get_entity_sentiment(self, entity: str, time_window: Optional[Tuple[datetime, datetime]] = None) -> float:
        """
        Get the sentiment for an entity, optionally within a time window.

        Args:
            entity: Entity name.
            time_window: Optional time window (start_time, end_time).

        Returns:
            Average sentiment score.
        """
        if entity not in self.entity_sentiments:
            return 0.0
        sentiment_records = self.entity_sentiments[entity]
        if time_window:
            start_time, end_time = time_window
            sentiment_records = [
                record
                for record in sentiment_records
                if start_time <= pd.to_datetime(record["timestamp"]) <= end_time
            ]
        if not sentiment_records:
            return 0.0
        total_sentiment = sum(record["sentiment"] * record["importance"] for record in sentiment_records)
        total_importance = sum(record["importance"] for record in sentiment_records)
        return total_sentiment / total_importance if total_importance > 0 else 0.0

    def get_sentiment_trend(self, entity: str, num_periods: int = 5) -> List[Optional[float]]:
        """
        Get the sentiment trend for an entity.

        Args:
            entity: Entity name.
            num_periods: Number of time periods to divide the data into.

        Returns:
            List of average sentiment values for each period.
        """
        if entity not in self.entity_sentiments or not self.entity_sentiments[entity]:
            return []
        records = sorted(self.entity_sentiments[entity], key=lambda x: pd.to_datetime(x["timestamp"]))
        start_time = pd.to_datetime(records[0]["timestamp"])
        end_time = pd.to_datetime(records[-1]["timestamp"])
        time_range = (end_time - start_time).total_seconds()
        period_size = time_range / num_periods
        trend = []
        for i in range(num_periods):
            period_start = start_time + timedelta(seconds=i * period_size)
            period_end = start_time + timedelta(seconds=(i + 1) * period_size)
            period_records = [
                record
                for record in records
                if period_start <= pd.to_datetime(record["timestamp"]) < period_end
            ]
            if period_records:
                total_sentiment = sum(record["sentiment"] * record["importance"] for record in period_records)
                total_importance = sum(record["importance"] for record in period_records)
                period_sentiment = total_sentiment / total_importance if total_importance > 0 else 0.0
            else:
                period_sentiment = None
            trend.append(period_sentiment)
        return trend

    def get_most_important_entities(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the most important entities based on mentions and importance.

        Args:
            top_n: Number of entities to return.

        Returns:
            List of (entity, importance) tuples.
        """
        entities = sorted(self.entity_importance.items(), key=lambda x: x[1], reverse=True)
        return entities[:top_n]

    def get_entity_network(self, min_relationship_strength: float = 0.5) -> nx.Graph:
        """
        Get a network of entity relationships.

        Args:
            min_relationship_strength: Minimum relationship strength to include.

        Returns:
            NetworkX graph object.
        """
        G = nx.Graph()
        for entity, importance in self.entity_importance.items():
            G.add_node(entity, importance=importance, mentions=self.entity_mentions[entity])
        for entity1, relationships in self.entity_relationships.items():
            for entity2, strength in relationships.items():
                if strength >= min_relationship_strength:
                    G.add_edge(entity1, entity2, weight=strength)
        return G


class FinancialSentimentModel:
    """
    Advanced sentiment analysis model for financial news and social media.

    Supports multiple models, entity-specific sentiment tracking, and temporal analysis.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "finbert",
        use_ensemble: bool = False,
        entity_extraction: bool = True,
        temporal_tracking: bool = True,
        use_cache: bool = True,
    ):
        """
        Initialize the sentiment model.

        Args:
            model_path: Path to a fine-tuned model directory.
            model_name: Pre-trained model name ('finbert', 'financial_roberta', 'sector_bert').
            use_ensemble: Whether to use an ensemble of models.
            entity_extraction: Whether to extract and track entity-specific sentiment.
            temporal_tracking: Whether to track sentiment over time.
            use_cache: Whether to use Redis cache for predictions.
        """
        self.model_name = model_name
        self.use_ensemble = use_ensemble
        self.entity_extraction = entity_extraction
        self.temporal_tracking = temporal_tracking
        self.use_cache = use_cache
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.configs: Dict[str, Any] = {}
        self.model_config = DEFAULT_MODELS.get(model_name, DEFAULT_MODELS["finbert"])
        if model_name not in DEFAULT_MODELS:
            logger.warning(f"Unknown model '{model_name}', defaulting to FinBERT")
        if entity_extraction:
            self.entity_tracker = EntitySentimentTracker()
            try:
                # Ensure NLTK data is downloaded
                nltk.download('punkt')
                nltk.download('averaged_perceptron_tagger')
                nltk.download('maxent_ne_chunker')
                nltk.download('words')
                nltk.download('averaged_perceptron_tagger_eng')
                nltk.download('maxent_ne_chunker_tab')
                logger.info("NLTK resources loaded for NER")
                self.nlp = True  # Just a flag to indicate NLP is available
            except Exception as e:
                logger.error(f"Failed to load NLTK resources for entity extraction: {e}")
                self.nlp = None
        else:
            self.entity_tracker = None
            self.nlp = None
        if use_ensemble:
            self._load_ensemble_models()
        else:
            self._load_model(model_path)
        self.entity_dict = self._load_entity_dictionary()
        self.metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": None,
        }
        self.metadata = {
            "model_name": model_name,
            "ensemble": use_ensemble,
            "entity_extraction": entity_extraction,
            "temporal_tracking": temporal_tracking,
            "labels": self.model_config["labels"],
            "last_trained": None,
            "last_updated": datetime.now().isoformat(),
            "performance": self.metrics,
        }

    def _load_model(self, model_path: Optional[str] = None):
        """
        Load a single sentiment analysis model.

        Args:
            model_path: Path to a fine-tuned model.
        """
        model_name = self.model_config["name"]
        try:
            if model_path:
                if not os.path.isabs(model_path):
                    model_path = os.path.join(settings.models_dir, os.path.basename(model_path))
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    logger.info(f"Loading fine-tuned model from {model_path}")
                    self.models["primary"] = AutoModelForSequenceClassification.from_pretrained(model_path)
                    self.tokenizers["primary"] = AutoTokenizer.from_pretrained(model_path)
                    self.configs["primary"] = AutoConfig.from_pretrained(model_path)
                    labels_path = os.path.join(model_path, "labels.json")
                    if os.path.exists(labels_path):
                        with open(labels_path, "r") as f:
                            self.model_config["labels"] = json.load(f)
                    metadata_path = os.path.join(model_path, "metadata.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                            self.metadata.update(metadata)
                    self.models["primary"].to(device)
                    return
            logger.info(f"Loading pre-trained model {model_name}")
            self.models["primary"] = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.tokenizers["primary"] = AutoTokenizer.from_pretrained(model_name)
            self.configs["primary"] = AutoConfig.from_pretrained(model_name)
            if hasattr(self.configs["primary"], "id2label"):
                self.model_config["labels"] = list(self.configs["primary"].id2label.values())
            self.models["primary"].to(device)
            logger.info(f"Loaded model {model_name} with labels: {self.model_config['labels']}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise

    def _load_ensemble_models(self):
        """Load multiple models for ensemble prediction."""
        models_to_load = ["finbert", "financial_roberta"]
        for model_key in models_to_load:
            model_config = DEFAULT_MODELS[model_key]
            model_name = model_config["name"]
            try:
                logger.info(f"Loading ensemble model {model_name}")
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                config = AutoConfig.from_pretrained(model_name)
                self.models[model_key] = model
                self.tokenizers[model_key] = tokenizer
                self.configs[model_key] = config
                model.to(device)
                logger.info(f"Loaded ensemble model {model_name}")
            except Exception as e:
                logger.error(f"Error loading ensemble model {model_name}: {e}")

    def _load_entity_dictionary(self) -> Dict[str, str]:
        """
        Load custom entity dictionary for financial entities.

        Returns:
            Dictionary of entities and their types.
        """
        entity_dict = {}
        entity_file = os.path.join(settings.data_dir, "entities", "financial_entities.json")
        if os.path.exists(entity_file):
            try:
                with open(entity_file, "r") as f:
                    entity_dict = json.load(f)
                logger.info(f"Loaded {len(entity_dict)} entities from dictionary")
            except Exception as e:
                logger.error(f"Error loading entity dictionary: {e}")
        return entity_dict

    def save_model(self, save_dir: str):
        """
        Save the model to directory.

        Args:
            save_dir: Directory to save the model.
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            if "primary" in self.models:
                self.models["primary"].save_pretrained(save_dir)
                self.tokenizers["primary"].save_pretrained(save_dir)
            with open(os.path.join(save_dir, "labels.json"), "w") as f:
                json.dump(self.model_config["labels"], f)
            self.metadata["last_updated"] = datetime.now().isoformat()
            with open(os.path.join(save_dir, "metadata.json"), "w") as f:
                json.dump(self.metadata, f)
            if self.entity_tracker:
                entity_data = {
                    "entity_sentiments": dict(self.entity_tracker.entity_sentiments),
                    "entity_mentions": dict(self.entity_tracker.entity_mentions),
                    "entity_importance": dict(self.entity_tracker.entity_importance),
                    "entity_relationships": {k: dict(v) for k, v in self.entity_tracker.entity_relationships.items()},
                }
                with open(os.path.join(save_dir, "entity_data.pkl"), "wb") as f:
                    pickle.dump(entity_data, f)
            logger.info(f"Saved model to {save_dir}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def fine_tune(
        self,
        texts: List[str],
        labels: List[int],
        eval_texts: Optional[List[str]] = None,
        eval_labels: Optional[List[int]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_length: Optional[int] = None,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        model_name: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the model on financial data.

        Args:
            texts: List of text samples.
            labels: List of sentiment labels (0=negative, 1=neutral, 2=positive).
            eval_texts: Optional evaluation texts.
            eval_labels: Optional evaluation labels.
            epochs: Number of training epochs.
            batch_size: Batch size.
            learning_rate: Learning rate.
            max_length: Maximum sequence length.
            weight_decay: Weight decay for AdamW optimizer.
            warmup_steps: Number of warmup steps for learning rate scheduler.
            model_name: Optional base model name to use.

        Returns:
            Training history dictionary.
        """
        max_length = max_length or self.model_config["max_length"]
        if model_name:
            logger.info(f"Loading new base model {model_name} for fine-tuning")
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(self.model_config["labels"]))
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            model = self.models["primary"]
            tokenizer = self.tokenizers["primary"]
        model.to(device)
        model.train()
        train_dataset = SentimentDataset(texts, labels, tokenizer, max_length)
        if eval_texts and eval_labels:
            eval_dataset = SentimentDataset(eval_texts, eval_labels, tokenizer, max_length)
        else:
            train_size = int(0.9 * len(train_dataset))
            eval_size = len(train_dataset) - train_size
            train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, [train_size, eval_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        history = {
            "train_loss": [],
            "eval_loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in progress_bar:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "text"}
                optimizer.zero_grad()
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)
            model.eval()
            eval_loss = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in eval_loader:
                    inputs = {k: v.to(device) for k, v in batch.items() if k != "text"}
                    labels = inputs.get("labels", None)
                    outputs = model(**inputs)
                    loss = outputs.loss
                    logits = outputs.logits
                    eval_loss += loss.item()
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    if labels is not None:
                        all_labels.extend(labels.cpu().numpy())
            avg_eval_loss = eval_loss / len(eval_loader)
            history["eval_loss"].append(avg_eval_loss)
            if all_labels:
                accuracy = accuracy_score(all_labels, all_preds)
                precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
                recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
                f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
                history["accuracy"].append(accuracy)
                history["precision"].append(precision)
                history["recall"].append(recall)
                history["f1"].append(f1)
                self.metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
                }
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                    f"Eval Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}"
                )
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")
        self.models["primary"] = model
        self.tokenizers["primary"] = tokenizer
        self.metadata["last_trained"] = datetime.now().isoformat()
        self.metadata["performance"] = self.metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_dir = os.path.join(settings.models_dir, f"sentiment_model_{timestamp}")
        self.save_model(model_save_dir)
        self.save_model(getattr(settings.model, "sentiment_model_path", model_save_dir))
        return history

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text using NLTK.

        Args:
            text: Input text.

        Returns:
            List of entity dictionaries with type and text.
        """
        if not self.nlp:
            logger.warning("NLTK resources not available for entity extraction")
            return []
        try:
            # Tokenize, POS tag, and perform NER
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            ne_tree = ne_chunk(pos_tags)
            
            # Convert the tree to a format we can process
            iob_tags = tree2conlltags(ne_tree)
            
            entities = []
            current_entity = {"text": "", "type": "", "start": 0, "end": 0}
            in_entity = False
            char_index = 0
            
            # Process the IOB tags to extract entities
            for word, pos, tag in iob_tags:
                # Skip punctuation for character index calculation
                if word in ",.;:!?()[]{}\"'":
                    char_index += len(word) + 1  # +1 for space
                    continue
                
                if tag.startswith('B-'):  # Beginning of entity
                    if in_entity:  # If we were already tracking an entity, add it to the list
                        current_entity["end"] = char_index - 1
                        entities.append(current_entity.copy())
                    
                    # Start a new entity
                    entity_type = tag[2:]  # Remove 'B-' prefix
                    mapped_type = {
                        "ORGANIZATION": "company",
                        "PERSON": "person",
                        "GPE": "location",
                        "LOCATION": "location",
                        "PRODUCT": "product",
                        "MONEY": "financial",
                        "PERCENT": "financial",
                    }.get(entity_type, "other")
                    
                    # Check custom entity dictionary
                    if word.lower() in self.entity_dict:
                        mapped_type = self.entity_dict[word.lower()]
                    
                    current_entity = {
                        "text": word,
                        "type": mapped_type,
                        "start": char_index,
                        "end": 0  # Will be set when entity ends
                    }
                    in_entity = True
                
                elif tag.startswith('I-'):  # Inside of entity
                    if in_entity:
                        current_entity["text"] += " " + word
                
                else:  # Outside of entity (O tag)
                    if in_entity:
                        current_entity["end"] = char_index - 1
                        entities.append(current_entity.copy())
                        in_entity = False
                
                char_index += len(word) + 1  # +1 for space
            
            # Add the last entity if we were tracking one
            if in_entity:
                current_entity["end"] = char_index - 1
                entities.append(current_entity)
            
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities with NLTK: {e}")
            return []

    def analyze_sentiment(self, text: str, extract_entities: Optional[bool] = None) -> Dict[str, Any]:
        """
        Analyze the sentiment of a text.

        Args:
            text: Text to analyze.
            extract_entities: Whether to extract entities (overrides instance setting).

        Returns:
            Dictionary of sentiment scores and entities.
        """
        if not text or not isinstance(text, str):
            logger.error("Invalid or empty text input")
            return {"sentiment": {}, "entities": []}
        if self.use_cache:
            cache_key = f"sentiment:{hash(text)}"
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.debug(f"Retrieved sentiment from cache for text: {text[:50]}...")
                return json.loads(cached_result)
        extract_entities = extract_entities if extract_entities is not None else self.entity_extraction
        sentiment_scores = self._get_sentiment_scores(text)
        entities = []
        if extract_entities and self.nlp:
            entities = self.extract_entities(text)
            if entities and self.entity_tracker:
                for entity in entities:
                    entity_sentiment = self._get_entity_sentiment(text, entity)
            if entity_sentiment:
                item.update(entity_sentiment)
            if self.temporal_tracking:
                self.entity_tracker.add_entity_sentiment(
                    entity["text"],
                    entity_sentiment["sentiment_score"],
                    datetime.now().isoformat(),
                    importance=entity_sentiment["relevance_score"],
                )
        result = {"sentiment": sentiment_scores, "entities": entities}
        if extract_entities and len(entities) > 1 and self.entity_tracker:
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i + 1 :]:
                    self.entity_tracker.add_entity_relationship(entity1["text"], entity2["text"], strength=1.0)
        if self.use_cache:
            redis_client.set(cache_key, json.dumps(result), ex=3600)
        return result

    def _get_sentiment_scores(self, text: str) -> Dict[str, float]:
        """
        Get sentiment scores for text from model.

        Args:
            text: Input text.

        Returns:
            Dictionary of sentiment scores.
        """
        text = text[:1024]  # Truncate to avoid tokenizer limits
        if self.use_ensemble:
            ensemble_scores = {}
            for model_key, model in self.models.items():
                tokenizer = self.tokenizers[model_key]
                inputs = tokenizer(
    text,
    truncation=True,
    padding=True,
    max_length=self.model_config["max_length"],
    return_tensors="pt",
).to(device) # Corrected: move .to(device) here
                # inputs = {k: v.to(device) for k, v in inputs.items()} # This line is now redundant
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=1)[0]
                labels = DEFAULT_MODELS[model_key]["labels"] if model_key in DEFAULT_MODELS else self.model_config["labels"]
                scores = {label: float(prob) for label, prob in zip(labels, probs.cpu().numpy())}
                sentiment_score = scores.get("positive", 0) - scores.get("negative", 0)
                ensemble_scores[model_key] = {"scores": scores, "sentiment_score": sentiment_score}
            primary_scores = ensemble_scores.get("finbert", ensemble_scores.get(next(iter(ensemble_scores))))
            combined_scores = primary_scores["scores"].copy()
            ensemble_sentiment = sum(m["sentiment_score"] for m in ensemble_scores.values()) / len(ensemble_scores)
            combined_scores["ensemble_score"] = ensemble_sentiment
            combined_scores["sentiment_score"] = ensemble_sentiment
            return combined_scores
        else:
            model = self.models["primary"]
            tokenizer = self.tokenizers["primary"]
            inputs = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.model_config["max_length"],
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)[0]
            scores = {label: float(prob) for label, prob in zip(self.model_config["labels"], probs.cpu().numpy())}
            sentiment_score = scores.get("positive", 0) - scores.get("negative", 0)
            scores["sentiment_score"] = sentiment_score
            return scores

    def _get_entity_sentiment(self, text: str, entity: Dict) -> Dict[str, Any]:
        """
        Get sentiment specific to an entity.

        Args:
            text: Full text.
            entity: Entity dictionary with text and position.

        Returns:
            Dictionary with entity sentiment scores.
        """
        entity_text = entity["text"]
        start_pos = max(0, entity["start"] - 100)
        end_pos = min(len(text), entity["end"] + 100)
        context = text[start_pos:end_pos]
        relevance_score = 1.0
        if entity["start"] < 50:
            relevance_score *= 1.5
        blob = TextBlob(context)
        sentiment_score = blob.sentiment.polarity
        sentences = [str(sentence) for sentence in blob.sentences if entity_text.lower() in sentence.string.lower()]
        return {
            "sentiment_score": sentiment_score,
            "relevance_score": relevance_score,
            "context": sentences[:2] if sentences else context[:200],
        }

    def analyze_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze the sentiment of multiple texts.

        Args:
            texts: List of texts to analyze.

        Returns:
            List of sentiment analysis results.
        """
        if not texts:
            logger.warning("Empty text list provided")
            return []
        results = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_results = [self.analyze_sentiment(text) for text in batch_texts]
            results.extend(batch_results)
        return results

    def analyze_news_items(self, news_items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Analyze the sentiment of news items.

        Args:
            news_items: List of news items with 'title' and 'summary' fields.

        Returns:
            List of news items with sentiment scores.
        """
        if not news_items:
            logger.warning("Empty news items list provided")
            return []
        texts = []
        for item in news_items:
            title = item.get("title", "")
            summary = item.get("summary", "")
            if title and summary:
                text = f"{title}. {summary}"
            elif title:
                text = title
            elif summary:
                text = summary
            else:
                text = ""
            texts.append(text)
        sentiments = self.analyze_texts(texts)
        results = []
        for item, sentiment in zip(news_items, sentiments):
            result = item.copy()
            result["sentiment"] = sentiment["sentiment"]
            if "entities" in sentiment and sentiment["entities"]:
                result["entities"] = sentiment["entities"]
            result["sentiment_score"] = sentiment["sentiment"].get("sentiment_score", 0)
            results.append(result)
        return results

    def get_overall_sentiment(self, news_items: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate overall sentiment from a list of news items.

        Args:
            news_items: List of news items with sentiment scores.

        Returns:
            Dictionary with overall sentiment scores.
        """
        if not news_items:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0, "overall_score": 0.0}
        sentiments = {label: 0.0 for label in self.model_config["labels"]}
        sentiments["overall_score"] = 0.0
        total_weight = 0.0
        for item in news_items:
            item_sentiment = item.get("sentiment", {})
            if isinstance(item_sentiment, dict) and "scores" in item_sentiment:
                item_sentiment = item_sentiment["scores"]
            weight = item.get("relevance_score", 1.0)
            for label in self.model_config["labels"]:
                sentiments[label] += item_sentiment.get(label, 0.0) * weight
            sentiments["overall_score"] += item.get("sentiment_score", 0.0) * weight
            total_weight += weight
        if total_weight > 0:
            for key in sentiments:
                sentiments[key] /= total_weight
        return sentiments

    def get_entity_sentiments(self, time_window: Optional[Tuple[datetime, datetime]] = None, top_n: int = 10) -> List[Dict]:
        """
        Get sentiment for tracked entities.

        Args:
            time_window: Optional time window (start_time, end_time).
            top_n: Number of top entities to return.

        Returns:
            List of entity sentiment dictionaries.
        """
        if not self.entity_tracker:
            return []
        important_entities = self.entity_tracker.get_most_important_entities(top_n=top_n)
        results = []
        for entity, importance in important_entities:
            sentiment = self.entity_tracker.get_entity_sentiment(entity, time_window)
            trend = self.entity_tracker.get_sentiment_trend(entity)
            mentions = self.entity_tracker.entity_mentions.get(entity, 0)
            results.append(
                {
                    "entity": entity,
                    "sentiment": sentiment,
                    "importance": importance,
                    "mentions": mentions,
                    "trend": trend,
                }
            )
        return results

    def get_entity_network(self, min_relationship_strength: float = 0.5) -> Dict[str, List]:
        """
        Get the entity relationship network.

        Args:
            min_relationship_strength: Minimum relationship strength.

        Returns:
            Dictionary with network data.
        """
        if not self.entity_tracker:
            return {"nodes": [], "edges": []}
        G = self.entity_tracker.get_entity_network(min_relationship_strength)
        nodes = [
            {
                "id": node,
                "importance": attrs.get("importance", 1.0),
                "mentions": attrs.get("mentions", 0),
                "sentiment": self.entity_tracker.get_entity_sentiment(node),
            }
            for node, attrs in G.nodes(data=True)
        ]
        edges = [
            {"source": source, "target": target, "weight": attrs.get("weight", 1.0)}
            for source, target, attrs in G.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}

    def generate_sentiment_report(
        self, news_items: List[Dict], include_entities: bool = True, include_network: bool = True
    ) -> Dict:
        """
        Generate a comprehensive sentiment analysis report.

        Args:
            news_items: List of news items to analyze.
            include_entities: Whether to include entity analysis.
            include_network: Whether to include entity network.

        Returns:
            Report dictionary.
        """
        analyzed_items = self.analyze_news_items(news_items)
        overall_sentiment = self.get_overall_sentiment(analyzed_items)
        positive_items = [item for item in analyzed_items if item.get("sentiment_score", 0) > 0.2]
        negative_items = [item for item in analyzed_items if item.get("sentiment_score", 0) < -0.2]
        neutral_items = [item for item in analyzed_items if -0.2 <= item.get("sentiment_score", 0) <= 0.2]
        report = {
            "overall_sentiment": overall_sentiment,
            "item_count": len(analyzed_items),
            "positive_count": len(positive_items),
            "negative_count": len(negative_items),
            "neutral_count": len(neutral_items),
            "sentiment_distribution": {
                "positive": overall_sentiment.get("positive", 0),
                "neutral": overall_sentiment.get("neutral", 0),
                "negative": overall_sentiment.get("negative", 0),
            },
            "top_positive_items": sorted(positive_items, key=lambda x: x.get("sentiment_score", 0), reverse=True)[:5],
            "top_negative_items": sorted(negative_items, key=lambda x: x.get("sentiment_score", 0))[:5],
        }
        if include_entities and self.entity_tracker:
            top_entities = self.get_entity_sentiments(top_n=10)
            report["entities"] = {"top_entities": top_entities, "entity_count": len(self.entity_tracker.entity_sentiments)}
            if include_network:
                report["entity_network"] = self.get_entity_network()
        timestamps = []
        for item in news_items:
            for key in ["timestamp", "date", "published_at"]:
                if key in item:
                    try:
                        timestamps.append(pd.to_datetime(item[key]))
                        break
                    except:
                        continue
        if timestamps:
            time_sorted_items = [(ts, item) for ts, item in zip(timestamps, analyzed_items)]
            time_sorted_items.sort()
            daily_sentiments = {}
            for ts, item in time_sorted_items:
                day = ts.strftime("%Y-%m-%d")
                if day not in daily_sentiments:
                    daily_sentiments[day] = []
                daily_sentiments[day].append(item)
            sentiment_trend = [
                {
                    "date": day,
                    "sentiment_score": self.get_overall_sentiment(items).get("overall_score", 0),
                    "positive": self.get_overall_sentiment(items).get("positive", 0),
                    "neutral": self.get_overall_sentiment(items).get("neutral", 0),
                    "negative": self.get_overall_sentiment(items).get("negative", 0),
                    "item_count": len(items),
                }
                for day, items in daily_sentiments.items()
            ]
            report["sentiment_trend"] = sentiment_trend
        return report

    def export_model(self, output_dir: Optional[str] = None) -> str:
        """
        Export the model for deployment.

        Args:
            output_dir: Optional output directory.

        Returns:
            Path to exported model.
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(settings.models_dir, "exports", f"sentiment_model_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        self.save_model(output_dir)
        pipeline_dir = os.path.join(output_dir, "pipeline")
        os.makedirs(pipeline_dir, exist_ok=True)
        pipeline_config = {
            "model_name": self.model_config["name"],
            "max_length": self.model_config["max_length"],
            "labels": self.model_config["labels"],
            "metadata": self.metadata,
        }
        with open(os.path.join(pipeline_dir, "config.json"), "w") as f:
            json.dump(pipeline_config, f, indent=2)
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(f"# Financial Sentiment Analysis Model\n\n")
            f.write(f"Exported on: {datetime.now().isoformat()}\n\n")
            f.write(f"Base model: {self.model_config['name']}\n")
            f.write(f"Labels: {', '.join(self.model_config['labels'])}\n\n")
            if self.metrics:
                f.write("## Performance Metrics\n\n")
                f.write(f"- Accuracy: {self.metrics['accuracy']:.4f}\n")
                f.write(f"- Precision: {self.metrics['precision']:.4f}\n")
                f.write(f"- Recall: {self.metrics['recall']:.4f}\n")
                f.write(f"- F1 Score: {self.metrics['f1']:.4f}\n\n")
            f.write("## Usage\n\n")
            f.write("```python\n")
            f.write("from src.models.sentiment import FinancialSentimentModel\n\n")
            f.write("# Load the model\n")
            f.write(f"model = FinancialSentimentModel(model_path='{output_dir}')\n\n")
            f.write("# Analyze text\n")
            f.write("result = model.analyze_sentiment('Apple reported strong earnings.')\n")
            f.write("print(result)\n")
            f.write("```\n")
        logger.info(f"Model exported to {output_dir}")
        return output_dir


sentiment_model = FinancialSentimentModel(
    model_path=getattr(settings.model, "sentiment_model_path", None),
    model_name="finbert",
    use_ensemble=True,
    entity_extraction=True,
    temporal_tracking=True,
)


def analyze_sentiment(text: str, extract_entities: bool = True) -> Dict[str, Any]:
    """Analyze sentiment of a text."""
    if sentiment_model is None:
        logger.error("Sentiment model not available")
        return {"error": "Sentiment model not available"}
    return sentiment_model.analyze_sentiment(text, extract_entities=extract_entities)


def analyze_news_batch(news_items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Analyze sentiment of a batch of news items."""
    if sentiment_model is None:
        logger.error("Sentiment model not available")
        return {"error": "Sentiment model not available"}
    return sentiment_model.analyze_news_items(news_items)


def generate_sentiment_report(news_items: List[Dict]) -> Dict:
    """Generate a comprehensive sentiment report from news items."""
    if sentiment_model is None:
        logger.error("Sentiment model not available")
        return {"error": "Sentiment model not available"}
    return sentiment_model.generate_sentiment_report(news_items)


def get_entity_sentiment_network() -> Dict[str, List]:
    """Get the entity sentiment network."""
    if sentiment_model is None or not sentiment_model.entity_tracker:
        logger.error("Sentiment model or entity tracker not available")
        return {"nodes": [], "edges": []}
    return sentiment_model.get_entity_network()


async def train_sentiment_model(training_data: Optional[Dict] = None, use_default_data: bool = True) -> Dict:
    """
    Train or fine-tune the sentiment model.

    Args:
        training_data: Optional custom training data.
        use_default_data: Whether to use default financial sentiment data.

    Returns:
        Training metrics.
    """
    if sentiment_model is None:
        logger.error("Cannot train: No sentiment model available")
        return {"error": "No sentiment model available"}
    if training_data is None and use_default_data:
        training_data = _load_default_training_data()
    if not training_data or "texts" not in training_data or "labels" not in training_data:
        logger.error("Invalid or missing training data")
        return {"error": "Invalid or missing training data"}
    try:
        logger.info(f"Training sentiment model with {len(training_data['texts'])} samples")
        history = sentiment_model.fine_tune(
            texts=training_data["texts"],
            labels=training_data["labels"],
            eval_texts=training_data.get("eval_texts"),
            eval_labels=training_data.get("eval_labels"),
            epochs=3,
            batch_size=16,
            learning_rate=2e-5,
        )
        sentiment_model.save_model(getattr(settings.model, "sentiment_model_path", None))
        return {"status": "success", "history": history, "metrics": sentiment_model.metrics}
    except Exception as e:
        logger.error(f"Error training sentiment model: {e}")
        return {"error": str(e)}


def _load_default_training_data() -> Dict[str, List]:
    """
    Load default financial sentiment training data.

    Returns:
        Dictionary with texts and labels.
    """
    data_path = os.path.join(settings.data_dir, "sentiment", "financial_sentiment_data.json")
    if os.path.exists(data_path):
        try:
            with open(data_path, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded default sentiment training data with {len(data.get('texts', []))} samples")
            return data
        except Exception as e:
            logger.error(f"Error loading default sentiment data: {e}")
    texts = [
        "The company reported record profits for the quarter.",
        "Shares plummeted after disappointing earnings.",
        "Market remained stable during trading.",
        "Investors are concerned about high debt levels.",
        "New product launch exceeded expectations.",
        "Expansion into new markets announced.",
        "Revenue growth slowed unexpectedly.",
        "Board approved a share buyback program.",
        "Economic indicators suggest a recession.",
        "Merger deal approved by regulators.",
    ]
    labels = [2, 0, 1, 0, 2, 2, 0, 2, 0, 1]  # 0=negative, 1=neutral, 2=positive
    return {"texts": texts, "labels": labels}


async def schedule_sentiment_model_training() -> bool:
    """Schedule sentiment model training to run during off-hours."""
    try:
        current_hour = datetime.now().hour
        is_off_hours = current_hour < 9 or current_hour > 16
        if is_off_hours:
            logger.info("Scheduling sentiment model training during off-hours")
            is_weekend = datetime.now().weekday() >= 5
            if is_weekend:
                logger.info("Running full sentiment model training (weekend schedule)")
                training_data = await _gather_enhanced_training_data()
                if training_data and len(training_data.get("texts", [])) > 100:
                    await train_sentiment_model(training_data, use_default_data=False)
                else:
                    await train_sentiment_model(use_default_data=True)
            else:
                logger.info("Running incremental sentiment model update (weekday schedule)")
                recent_data = await _gather_recent_news_data()
                if recent_data and len(recent_data.get("texts", [])) > 10:
                    try:
                        sentiment_model.fine_tune(
                            texts=recent_data["texts"],
                            labels=recent_data["labels"],
                            epochs=1,
                            batch_size=8,
                        )
                    except Exception as e:
                        logger.error(f"Error in incremental update: {e}")
            return True
        else:
            logger.info("Not in off-hours, skipping scheduled training")
            return False
    except Exception as e:
        logger.error(f"Error in schedule_sentiment_model_training: {e}")
        return False


async def _gather_enhanced_training_data() -> Dict[str, List]:
    """
    Gather enhanced training data from multiple sources.

    Returns:
        Dictionary with texts and labels.
    """
    logger.info("Gathering enhanced training data")
    # In a real implementation, this would fetch data from news APIs, financial datasets, etc.
    data_path = os.path.join(settings.data_dir, "sentiment", "enhanced_sentiment_data.json")
    if os.path.exists(data_path):
        try:
            with open(data_path, "r") as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading enhanced sentiment data: {e}")
    
    # Return simple synthetically generated data if no real data is available
    texts = []
    labels = []
    
    # Generate some positive samples
    for i in range(50):
        texts.append(f"Company reports strong quarterly results, exceeding analyst expectations.")
        labels.append(2)  # positive
    
    # Generate some negative samples
    for i in range(50):
        texts.append(f"Stock drops following earnings miss and lowered guidance.")
        labels.append(0)  # negative
        
    # Generate some neutral samples
    for i in range(50):
        texts.append(f"Market shows mixed trading as investors await economic data.")
        labels.append(1)  # neutral
        
    return {"texts": texts, "labels": labels}


async def _gather_recent_news_data() -> Dict[str, List]:
    """
    Gather recent financial news for incremental model updates.

    Returns:
        Dictionary with texts and labels.
    """
    logger.info("Gathering recent news data")
    # In a real implementation, this would fetch recent news from APIs
    data_path = os.path.join(settings.data_dir, "sentiment", "recent_news_data.json")
    if os.path.exists(data_path):
        try:
            with open(data_path, "r") as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading recent news data: {e}")
    
    # Return minimal synthetic data if no real data is available
    texts = [
        "Market closes higher on positive economic outlook",
        "Tech sector rallies on earnings surprises",
        "Concerns grow over inflation impact on consumer spending",
        "Energy stocks decline amid falling oil prices",
        "Financial sector stable as banks report solid earnings"
    ]
    labels = [2, 2, 0, 0, 1]  # 2=positive, 0=negative, 1=neutral
    
    return {"texts": texts, "labels": labels}
