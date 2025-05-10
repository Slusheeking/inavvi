"""
Sentiment analysis model for news and social media data.

This model uses state-of-the-art language models to analyze sentiment
of financial news and social media content, with multi-modal capabilities
and temporal sentiment tracking.
"""
import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import os
import json
import warnings
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    AutoConfig,
    BertModel,
    BertTokenizer, 
    RobertaModel,
    RobertaTokenizer,
    T5Model,
    T5Tokenizer,
    pipeline
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import networkx as nx
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import redis

from src.config.settings import settings
from src.utils.logging import setup_logger, log_execution_time
from src.utils.redis_client import redis_client

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    logger.warning("Failed to download NLTK resources. Some functionality may be limited.")

# Filter out specific huggingface_hub warnings
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
        "max_length": 256
    },
    "financial_roberta": {
        "name": "yiyanghkust/finbert-tone",
        "labels": ["negative", "neutral", "positive"],
        "max_length": 256
    },
    "sector_bert": {
        "name": "distilbert-base-uncased",  # This would be fine-tuned on sector-specific text
        "labels": ["negative", "neutral", "positive"],
        "max_length": 256
    }
}

class SentimentDataset(Dataset):
    """Dataset for sentiment analysis."""
    
    def __init__(self, texts, labels=None, tokenizer=None, max_length=256):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples
            labels: Optional list of labels
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize the text
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Remove batch dimension
            item = {key: val.squeeze(0) for key, val in encoding.items()}
        else:
            # If no tokenizer, just return the text
            item = {"text": text}
        
        # Add label if available
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        
        return item

class EntitySentimentTracker:
    """
    Track sentiment for specific entities over time.
    """
    
    def __init__(self):
        """Initialize the entity sentiment tracker."""
        self.entity_sentiments = defaultdict(list)
        self.entity_mentions = defaultdict(int)
        self.entity_importance = defaultdict(float)
        self.entity_relationships = defaultdict(lambda: defaultdict(float))
    
    def add_entity_sentiment(self, entity, sentiment, timestamp, source=None, importance=1.0):
        """
        Add a sentiment record for an entity.
        
        Args:
            entity: Entity name
            sentiment: Sentiment score (-1 to 1)
            timestamp: Timestamp of the sentiment
            source: Source of the sentiment (e.g., news article)
            importance: Importance of this sentiment record (1.0 = normal)
        """
        self.entity_sentiments[entity].append({
            "timestamp": timestamp,
            "sentiment": sentiment,
            "source": source,
            "importance": importance
        })
        
        self.entity_mentions[entity] += 1
        self.entity_importance[entity] += importance
    
    def add_entity_relationship(self, entity1, entity2, strength=1.0):
        """
        Add a relationship between two entities.
        
        Args:
            entity1: First entity
            entity2: Second entity
            strength: Strength of the relationship
        """
        self.entity_relationships[entity1][entity2] += strength
        self.entity_relationships[entity2][entity1] += strength
    
    def get_entity_sentiment(self, entity, time_window=None):
        """
        Get the sentiment for an entity, optionally within a time window.
        
        Args:
            entity: Entity name
            time_window: Optional time window (tuple of start_time, end_time)
            
        Returns:
            Average sentiment score
        """
        if entity not in self.entity_sentiments:
            return 0.0
        
        sentiment_records = self.entity_sentiments[entity]
        
        if time_window:
            start_time, end_time = time_window
            sentiment_records = [
                record for record in sentiment_records
                if start_time <= record["timestamp"] <= end_time
            ]
        
        if not sentiment_records:
            return 0.0
        
        # Calculate weighted average
        total_sentiment = sum(record["sentiment"] * record["importance"] for record in sentiment_records)
        total_importance = sum(record["importance"] for record in sentiment_records)
        
        return total_sentiment / total_importance if total_importance > 0 else 0.0
    
    def get_sentiment_trend(self, entity, num_periods=5):
        """
        Get the sentiment trend for an entity.
        
        Args:
            entity: Entity name
            num_periods: Number of time periods to divide the data into
            
        Returns:
            List of average sentiment values for each period
        """
        if entity not in self.entity_sentiments or not self.entity_sentiments[entity]:
            return []
        
        # Sort records by timestamp
        records = sorted(self.entity_sentiments[entity], key=lambda x: x["timestamp"])
        
        # Determine time range
        start_time = records[0]["timestamp"]
        end_time = records[-1]["timestamp"]
        
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        
        time_range = (end_time - start_time).total_seconds()
        period_size = time_range / num_periods
        
        # Calculate sentiment for each period
        trend = []
        
        for i in range(num_periods):
            period_start = start_time + timedelta(seconds=i * period_size)
            period_end = start_time + timedelta(seconds=(i + 1) * period_size)
            
            # Get sentiment records for this period
            period_records = [
                record for record in records
                if period_start <= pd.to_datetime(record["timestamp"]) < period_end
            ]
            
            if period_records:
                # Calculate weighted average
                total_sentiment = sum(record["sentiment"] * record["importance"] for record in period_records)
                total_importance = sum(record["importance"] for record in period_records)
                period_sentiment = total_sentiment / total_importance if total_importance > 0 else 0.0
            else:
                period_sentiment = None  # No data for this period
            
            trend.append(period_sentiment)
        
        return trend
    
    def get_most_important_entities(self, top_n=10):
        """
        Get the most important entities based on mentions and importance.
        
        Args:
            top_n: Number of entities to return
            
        Returns:
            List of (entity, importance) tuples
        """
        entities = sorted(
            self.entity_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return entities[:top_n]
    
    def get_entity_network(self, min_relationship_strength=0.5):
        """
        Get a network of entity relationships.
        
        Args:
            min_relationship_strength: Minimum relationship strength to include
            
        Returns:
            NetworkX graph object
        """
        G = nx.Graph()
        
        # Add nodes (entities)
        for entity, importance in self.entity_importance.items():
            G.add_node(entity, importance=importance, mentions=self.entity_mentions[entity])
        
        # Add edges (relationships)
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
        use_cache: bool = True
    ):
        """
        Initialize the sentiment model.
        
        Args:
            model_path: Path to a fine-tuned model directory
            model_name: Pre-trained model name ('finbert', 'financial_roberta', 'sector_bert')
            use_ensemble: Whether to use an ensemble of models
            entity_extraction: Whether to extract and track entity-specific sentiment
            temporal_tracking: Whether to track sentiment over time
            use_cache: Whether to use Redis cache for predictions
        """
        self.model_name = model_name
        self.use_ensemble = use_ensemble
        self.entity_extraction = entity_extraction
        self.temporal_tracking = temporal_tracking
        self.use_cache = use_cache
        
        # Initialize model registry
        self.models = {}
        self.tokenizers = {}
        self.configs = {}
        
        # Select model configuration
        if model_name in DEFAULT_MODELS:
            self.model_config = DEFAULT_MODELS[model_name]
        else:
            # Default to FinBERT
            self.model_config = DEFAULT_MODELS["finbert"]
            logger.warning(f"Unknown model '{model_name}', defaulting to FinBERT")
        
        # Entity tracker
        if entity_extraction:
            self.entity_tracker = EntitySentimentTracker()
            
            # Load spaCy for NER
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy NER model")
            except:
                logger.warning("Failed to load spaCy. Downloading smaller model...")
                try:
                    # Try to download a smaller model
                    import subprocess
                    subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                    self.nlp = spacy.load("en_core_web_sm")
                except:
                    logger.error("Failed to load spaCy for entity extraction")
                    self.nlp = None
        else:
            self.entity_tracker = None
            self.nlp = None
        
        # Initialize model(s)
        if use_ensemble:
            # Load multiple models for ensemble prediction
            self._load_ensemble_models()
        else:
            # Load single model
            self._load_model(model_path)
        
        # Load custom entity dictionary if available
        self.entity_dict = self._load_entity_dictionary()
        
        # Performance metrics
        self.metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": None
        }
        
        # Model metadata
        self.metadata = {
            "model_name": model_name,
            "ensemble": use_ensemble,
            "entity_extraction": entity_extraction,
            "temporal_tracking": temporal_tracking,
            "labels": self.model_config["labels"],
            "last_trained": None,
            "last_updated": datetime.now().isoformat(),
            "performance": self.metrics
        }
    
    def _load_model(self, model_path=None):
        """
        Load a single sentiment analysis model.
        
        Args:
            model_path: Path to a fine-tuned model
        """
        model_name = self.model_config["name"]
        
        try:
            # If a specific model path is provided, try to load it
            if model_path:
                # Convert to absolute path if it's a relative path
                if not os.path.isabs(model_path):
                    model_path = os.path.join(settings.models_dir, os.path.basename(model_path))
                
                # Check if it's a valid model directory with config.json
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    logger.info(f"Loading fine-tuned model from {model_path}")
                    self.models["primary"] = AutoModelForSequenceClassification.from_pretrained(model_path)
                    self.tokenizers["primary"] = AutoTokenizer.from_pretrained(model_path)
                    self.configs["primary"] = AutoConfig.from_pretrained(model_path)
                    
                    # Load custom labels if available
                    labels_path = os.path.join(model_path, "labels.json")
                    if os.path.exists(labels_path):
                        with open(labels_path, 'r') as f:
                            self.model_config["labels"] = json.load(f)
                    
                    # Load metadata if available
                    metadata_path = os.path.join(model_path, "metadata.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            self.metadata.update(metadata)
                    
                    self.models["primary"].to(device)
                    return
            
            # Load pre-trained model
            logger.info(f"Loading pre-trained model {model_name}")
            self.models["primary"] = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.tokenizers["primary"] = AutoTokenizer.from_pretrained(model_name)
            self.configs["primary"] = AutoConfig.from_pretrained(model_name)
            
            # Check if we need to update labels based on model config
            if hasattr(self.configs["primary"], "id2label"):
                self.model_config["labels"] = list(self.configs["primary"].id2label.values())
            
            # Move model to device
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
                
                # Store in model registry
                self.models[model_key] = model
                self.tokenizers[model_key] = tokenizer
                self.configs[model_key] = config
                
                # Move model to device
                model.to(device)
                
                logger.info(f"Loaded ensemble model {model_name}")
            except Exception as e:
                logger.error(f"Error loading ensemble model {model_name}: {e}")
    
    def _load_entity_dictionary(self):
        """
        Load custom entity dictionary for financial entities.
        
        Returns:
            Dictionary of entities and their types
        """
        entity_dict = {}
        
        # Try to load from file
        entity_file = os.path.join(settings.data_dir, "entities", "financial_entities.json")
        if os.path.exists(entity_file):
            try:
                with open(entity_file, 'r') as f:
                    entity_dict = json.load(f)
                logger.info(f"Loaded {len(entity_dict)} entities from dictionary")
            except Exception as e:
                logger.error(f"Error loading entity dictionary: {e}")
        
        return entity_dict
    
    def save_model(self, save_dir: str):
        """
        Save the model to directory.
        
        Args:
            save_dir: Directory to save the model
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Save primary model and tokenizer
            if "primary" in self.models:
                self.models["primary"].save_pretrained(save_dir)
                self.tokenizers["primary"].save_pretrained(save_dir)
            
            # Save labels
            with open(os.path.join(save_dir, "labels.json"), "w") as f:
                json.dump(self.model_config["labels"], f)
            
            # Save metadata
            self.metadata["last_updated"] = datetime.now().isoformat()
            with open(os.path.join(save_dir, "metadata.json"), "w") as f:
                json.dump(self.metadata, f)
            
            # Save entity tracker if available
            if self.entity_tracker:
                entity_data = {
                    "entity_sentiments": dict(self.entity_tracker.entity_sentiments),
                    "entity_mentions": dict(self.entity_tracker.entity_mentions),
                    "entity_importance": dict(self.entity_tracker.entity_importance),
                    "entity_relationships": {
                        k: dict(v) for k, v in self.entity_tracker.entity_relationships.items()
                    }
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
        max_length: int = None,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        model_name: str = None
    ):
        """
        Fine-tune the model on financial data.
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels (0=negative, 1=neutral, 2=positive)
            eval_texts: Optional evaluation texts
            eval_labels: Optional evaluation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            max_length: Maximum sequence length
            weight_decay: Weight decay for AdamW optimizer
            warmup_steps: Number of warmup steps for learning rate scheduler
            model_name: Optional base model name to use
            
        Returns:
            Training history
        """
        from transformers import AdamW, get_linear_schedule_with_warmup
        
        # Use parameters from config if not provided
        if max_length is None:
            max_length = self.model_config["max_length"]
        
        # If model_name is provided, load a new base model
        if model_name:
            logger.info(f"Loading new base model {model_name} for fine-tuning")
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(self.model_config["labels"]))
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            # Use the existing primary model
            model = self.models["primary"]
            tokenizer = self.tokenizers["primary"]
        
        # Move model to device
        model.to(device)
        
        # Set model to training mode
        model.train()
        
        # Create datasets
        train_dataset = SentimentDataset(texts, labels, tokenizer, max_length)
        
        if eval_texts and eval_labels:
            eval_dataset = SentimentDataset(eval_texts, eval_labels, tokenizer, max_length)
        else:
            # Split training data for evaluation
            from torch.utils.data import random_split
            
            train_size = int(0.9 * len(train_dataset))
            eval_size = len(train_dataset) - train_size
            
            train_dataset, eval_dataset = random_split(
                train_dataset, [train_size, eval_size]
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Initialize training history
        history = {
            'train_loss': [],
            'eval_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            # Progress bar for training
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                # Move batch to device
                inputs = {k: v.to(device) for k, v in batch.items() if k != "text"}
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(**inputs)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                train_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Evaluation
            model.eval()
            eval_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in eval_loader:
                    # Move batch to device
                    inputs = {k: v.to(device) for k, v in batch.items() if k != "text"}
                    
                    # Get labels before they're used by the model
                    labels = inputs.get("labels", None)
                    
                    # Forward pass
                    outputs = model(**inputs)
                    loss = outputs.loss
                    logits = outputs.logits
                    
                    # Update metrics
                    eval_loss += loss.item()
                    
                    # Get predictions
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    
                    # Store predictions and labels
                    all_preds.extend(preds)
                    if labels is not None:
                        all_labels.extend(labels.cpu().numpy())
            
            # Calculate average evaluation loss
            avg_eval_loss = eval_loss / len(eval_loader)
            history['eval_loss'].append(avg_eval_loss)
            
            # Calculate metrics if labels are available
            if all_labels:
                accuracy = accuracy_score(all_labels, all_preds)
                precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
                recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
                f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
                
                history['accuracy'].append(accuracy)
                history['precision'].append(precision)
                history['recall'].append(recall)
                history['f1'].append(f1)
                
                # Update model metrics
                self.metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist()
                }
                
                # Log metrics
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"Train Loss: {avg_train_loss:.4f}, "
                           f"Eval Loss: {avg_eval_loss:.4f}, "
                           f"Accuracy: {accuracy:.4f}, "
                           f"F1: {f1:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"Train Loss: {avg_train_loss:.4f}, "
                           f"Eval Loss: {avg_eval_loss:.4f}")
        
        # Update model and metadata
        self.models["primary"] = model
        self.tokenizers["primary"] = tokenizer
        self.metadata["last_trained"] = datetime.now().isoformat()
        self.metadata["performance"] = self.metrics
        
        # Save model after training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_dir = os.path.join(
            settings.models_dir,
            f"sentiment_model_{timestamp}"
        )
        self.save_model(model_save_dir)
        
        # Also save model to default path
        self.save_model(settings.model.sentiment_model_path)
        
        return history
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries with type and text
        """
        if not self.nlp:
            return []
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            # Entity type mapping (simplify spaCy types)
            entity_type = {
                "ORG": "company",
                "PERSON": "person",
                "GPE": "location",
                "LOC": "location",
                "PRODUCT": "product",
                "MONEY": "financial",
                "PERCENT": "financial"
            }.get(ent.label_, "other")
            
            # Check custom entity dictionary
            if ent.text.lower() in self.entity_dict:
                entity_type = self.entity_dict[ent.text.lower()]
            
            # Add entity
            entities.append({
                "text": ent.text,
                "type": entity_type,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        return entities
    
    def analyze_sentiment(self, text: str, extract_entities: bool = None) -> Dict[str, Any]:
        """
        Analyze the sentiment of a text.
        
        Args:
            text: Text to analyze
            extract_entities: Whether to extract entities (overrides instance setting)
            
        Returns:
            Dictionary of sentiment scores and entities
        """
        # Check cache if enabled
        if self.use_cache:
            cache_key = f"sentiment:{hash(text)}"
            cached_result = redis_client.get(cache_key)
            
            if cached_result:
                logger.debug(f"Retrieved sentiment from cache for text: {text[:50]}...")
                return cached_result
        
        # Use instance setting if not explicitly provided
        if extract_entities is None:
            extract_entities = self.entity_extraction
        
        # Get base sentiment
        sentiment_scores = self._get_sentiment_scores(text)
        
        # Extract entities if enabled
        entities = []
        if extract_entities:
            entities = self.extract_entities(text)
            
            # Extract entity-specific sentiment if entities found
            if entities:
                for entity in entities:
                    # Get entity-specific sentiment
                    entity_sentiment = self._get_entity_sentiment(text, entity)
                    entity.update(entity_sentiment)
                    
                    # Track entity sentiment if temporal tracking is enabled
                    if self.temporal_tracking and self.entity_tracker:
                        self.entity_tracker.add_entity_sentiment(
                            entity["text"],
                            entity_sentiment["sentiment_score"],
                            datetime.now().isoformat(),
                            importance=entity_sentiment["relevance_score"]
                        )
        
        # Create result dictionary
        result = {
            "sentiment": sentiment_scores,
            "entities": entities
        }
        
        # Analyze relationships between entities if multiple entities found
        if extract_entities and len(entities) > 1 and self.entity_tracker:
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    # Add relationship based on co-occurrence
                    self.entity_tracker.add_entity_relationship(
                        entity1["text"],
                        entity2["text"],
                        strength=1.0  # Base relationship strength
                    )
        
        # Cache result if enabled
        if self.use_cache:
            redis_client.set(cache_key, result, expiry=3600)  # Cache for 1 hour
        
        return result
    
    def _get_sentiment_scores(self, text: str) -> Dict[str, float]:
        """
        Get sentiment scores for text from model.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of sentiment scores
        """
        # Truncate long texts to avoid tokenizer limits
        if len(text) > 1024:
            text = text[:1024]
        
        if self.use_ensemble:
            # Use ensemble of models
            ensemble_scores = {}
            
            for model_key, model in self.models.items():
                tokenizer = self.tokenizers[model_key]
                
                # Tokenize text
                inputs = tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=self.model_config["max_length"],
                    return_tensors="pt"
                ).to(device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=1)[0]
                
                # Get labels for this model
                if model_key in DEFAULT_MODELS:
                    labels = DEFAULT_MODELS[model_key]["labels"]
                else:
                    labels = self.model_config["labels"]
                
                # Convert to dictionary
                scores = {label: float(prob) for label, prob in zip(labels, probs.cpu().numpy())}
                
                # Normalize scores to [-1, 1] range
                sentiment_score = scores.get("positive", 0) - scores.get("negative", 0)
                
                # Store model-specific scores
                ensemble_scores[model_key] = {
                    "scores": scores,
                    "sentiment_score": sentiment_score
                }
            
            # Combine ensemble scores
            # Start with the primary model's scores
            primary_scores = ensemble_scores.get("finbert", ensemble_scores.get(next(iter(ensemble_scores))))
            combined_scores = primary_scores["scores"].copy()
            
            # Calculate ensemble sentiment score as weighted average
            ensemble_sentiment = sum(
                m["sentiment_score"] * 1.0 for m in ensemble_scores.values()
            ) / len(ensemble_scores)
            
            # Add ensemble score
            combined_scores["ensemble_score"] = ensemble_sentiment
            
            # Add overall sentiment score
            combined_scores["sentiment_score"] = ensemble_sentiment
            
            return combined_scores
        else:
            # Use single model
            model = self.models["primary"]
            tokenizer = self.tokenizers["primary"]
            
            # Tokenize text
            inputs = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.model_config["max_length"],
                return_tensors="pt"
            ).to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)[0]
            
            # Convert to dictionary
            scores = {label: float(prob) for label, prob in zip(self.model_config["labels"], probs.cpu().numpy())}
            
            # Add overall sentiment score (-1 to 1 range)
            sentiment_score = scores.get("positive", 0) - scores.get("negative", 0)
            scores["sentiment_score"] = sentiment_score
            
            return scores
    
    def _get_entity_sentiment(self, text: str, entity: Dict) -> Dict[str, float]:
        """
        Get sentiment specific to an entity.
        
        Args:
            text: Full text
            entity: Entity dictionary with text and position
            
        Returns:
            Dictionary with entity sentiment scores
        """
        entity_text = entity["text"]
        
        # Extract context window around entity
        start_pos = max(0, entity["start"] - 100)
        end_pos = min(len(text), entity["end"] + 100)
        context = text[start_pos:end_pos]
        
        # Simple relevance score based on entity position and frequency
        relevance_score = 1.0
        
        # Boost relevance if entity is in title position (first 50 chars)
        if entity["start"] < 50:
            relevance_score *= 1.5
        
        # Use TextBlob for entity-specific sentiment
        blob = TextBlob(context)
        sentiment_score = blob.sentiment.polarity
        
        # Try to find sentences containing the entity
        sentences = []
        for sentence in blob.sentences:
            if entity_text.lower() in sentence.string.lower():
                sentences.append(str(sentence))
        
        return {
            "sentiment_score": sentiment_score,
            "relevance_score": relevance_score,
            "context": sentences[:2] if sentences else context[:200]
        }
    
    def analyze_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze the sentiment of multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        # Process in batches to avoid memory issues
        batch_size = 16
        
        for i in range(0, len(texts), batch_size):
            # Get batch
            batch_texts = texts[i:i+batch_size]
            
            # Process each text
            batch_results = []
            for text in batch_texts:
                result = self.analyze_sentiment(text)
                batch_results.append(result)
            
            # Add to results
            results.extend(batch_results)
        
        return results
    
    def analyze_news_items(self, news_items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Analyze the sentiment of news items.
        
        Args:
            news_items: List of news items with 'title' and 'summary' fields
            
        Returns:
            List of news items with sentiment scores
        """
        # Extract texts
        texts = []
        for item in news_items:
            # Combine title and summary
            title = item.get('title', '')
            summary = item.get('summary', '')
            
            if title and summary:
                text = f"{title}. {summary}"
            elif title:
                text = title
            elif summary:
                text = summary
            else:
                text = ''
            
            texts.append(text)
        
        # Analyze sentiments
        sentiments = self.analyze_texts(texts)
        
        # Combine results
        results = []
        for item, sentiment in zip(news_items, sentiments):
            # Create copy of item
            result = item.copy()
            
            # Add sentiment scores
            result['sentiment'] = sentiment["sentiment"]
            
            # Add entities if available
            if "entities" in sentiment and sentiment["entities"]:
                result['entities'] = sentiment["entities"]
            
            # Add overall sentiment score
            result['sentiment_score'] = sentiment["sentiment"].get("sentiment_score", 0)
            
            results.append(result)
        
        return results
    
    def get_overall_sentiment(self, news_items: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate overall sentiment from a list of news items.
        
        Args:
            news_items: List of news items with sentiment scores
            
        Returns:
            Dictionary with overall sentiment scores
        """
        if not news_items:
            return {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0, 'overall_score': 0.0}
        
        # Initialize counters
        sentiments = {label: 0.0 for label in self.model_config["labels"]}
        sentiments["overall_score"] = 0.0
        total_weight = 0.0
        
        # Process each item
        for item in news_items:
            # Get sentiment scores
            item_sentiment = item.get('sentiment', {})
            if isinstance(item_sentiment, dict) and "scores" in item_sentiment:
                item_sentiment = item_sentiment["scores"]
            
            # Get item weight (e.g., relevance)
            weight = item.get('relevance_score', 1.0)
            
            # Update sentiment counts
            for label in self.model_config["labels"]:
                sentiments[label] += item_sentiment.get(label, 0.0) * weight
            
            # Update overall score
            sentiments["overall_score"] += item.get('sentiment_score', 0.0) * weight
            
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            for key in sentiments:
                sentiments[key] /= total_weight
        
        return sentiments
    
    def get_entity_sentiments(self, time_window=None, top_n=10) -> List[Dict]:
        """
        Get sentiment for tracked entities.
        
        Args:
            time_window: Optional time window (tuple of start_time, end_time)
            top_n: Number of top entities to return
            
        Returns:
            List of entity sentiment dictionaries
        """
        if not self.entity_tracker:
            return []
        
        # Get most important entities
        important_entities = self.entity_tracker.get_most_important_entities(top_n=top_n)
        
        results = []
        for entity, importance in important_entities:
            # Get sentiment
            sentiment = self.entity_tracker.get_entity_sentiment(entity, time_window)
            
            # Get sentiment trend
            trend = self.entity_tracker.get_sentiment_trend(entity)
            
            # Get mentions
            mentions = self.entity_tracker.entity_mentions.get(entity, 0)
            
            # Create result
            results.append({
                "entity": entity,
                "sentiment": sentiment,
                "importance": importance,
                "mentions": mentions,
                "trend": trend
            })
        
        return results
    
    def get_entity_network(self, min_relationship_strength=0.5):
        """
        Get the entity relationship network.
        
        Args:
            min_relationship_strength: Minimum relationship strength
            
        Returns:
            Dictionary with network data
        """
        if not self.entity_tracker:
            return {"nodes": [], "edges": []}
        
        # Get network
        G = self.entity_tracker.get_entity_network(min_relationship_strength)
        
        # Convert to dictionary format
        nodes = []
        for node, attrs in G.nodes(data=True):
            sentiment = self.entity_tracker.get_entity_sentiment(node)
            
            nodes.append({
                "id": node,
                "importance": attrs.get("importance", 1.0),
                "mentions": attrs.get("mentions", 0),
                "sentiment": sentiment
            })
        
        edges = []
        for source, target, attrs in G.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "weight": attrs.get("weight", 1.0)
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def generate_sentiment_report(self, 
                                news_items: List[Dict], 
                                include_entities: bool = True, 
                                include_network: bool = True) -> Dict:
        """
        Generate a comprehensive sentiment analysis report.
        
        Args:
            news_items: List of news items to analyze
            include_entities: Whether to include entity analysis
            include_network: Whether to include entity network
            
        Returns:
            Report dictionary
        """
        # Analyze news items
        analyzed_items = self.analyze_news_items(news_items)
        
        # Calculate overall sentiment
        overall_sentiment = self.get_overall_sentiment(analyzed_items)
        
        # Group by sentiment
        positive_items = [item for item in analyzed_items if item.get('sentiment_score', 0) > 0.2]
        negative_items = [item for item in analyzed_items if item.get('sentiment_score', 0) < -0.2]
        neutral_items = [item for item in analyzed_items 
                        if -0.2 <= item.get('sentiment_score', 0) <= 0.2]
        
        # Build report
        report = {
            "overall_sentiment": overall_sentiment,
            "item_count": len(analyzed_items),
            "positive_count": len(positive_items),
            "negative_count": len(negative_items),
            "neutral_count": len(neutral_items),
            "sentiment_distribution": {
                "positive": overall_sentiment.get("positive", 0),
                "neutral": overall_sentiment.get("neutral", 0),
                "negative": overall_sentiment.get("negative", 0)
            },
            "top_positive_items": sorted(positive_items, 
                                        key=lambda x: x.get('sentiment_score', 0), 
                                        reverse=True)[:5],
            "top_negative_items": sorted(negative_items, 
                                        key=lambda x: x.get('sentiment_score', 0))[:5]
        }
        
        # Add entity analysis if enabled
        if include_entities and self.entity_tracker:
            top_entities = self.get_entity_sentiments(top_n=10)
            
            report["entities"] = {
                "top_entities": top_entities,
                "entity_count": len(self.entity_tracker.entity_sentiments)
            }
            
            # Add entity network if enabled
            if include_network:
                report["entity_network"] = self.get_entity_network()
        
        # Add time-based analysis if we have timestamps
        timestamps = []
        for item in news_items:
            if 'timestamp' in item:
                try:
                    timestamps.append(pd.to_datetime(item['timestamp']))
                except:
                    pass
            elif 'date' in item:
                try:
                    timestamps.append(pd.to_datetime(item['date']))
                except:
                    pass
            elif 'published_at' in item:
                try:
                    timestamps.append(pd.to_datetime(item['published_at']))
                except:
                    pass
        
        if timestamps:
            # Sort items by timestamp
            time_sorted_items = [(ts, item) for ts, item in zip(timestamps, analyzed_items)]
            time_sorted_items.sort()
            
            # Group by day
            daily_sentiments = {}
            for ts, item in time_sorted_items:
                day = ts.strftime('%Y-%m-%d')
                if day not in daily_sentiments:
                    daily_sentiments[day] = []
                daily_sentiments[day].append(item)
            
            # Calculate daily sentiment
            sentiment_trend = []
            for day, items in daily_sentiments.items():
                daily_overall = self.get_overall_sentiment(items)
                sentiment_trend.append({
                    "date": day,
                    "sentiment_score": daily_overall.get("overall_score", 0),
                    "positive": daily_overall.get("positive", 0),
                    "neutral": daily_overall.get("neutral", 0),
                    "negative": daily_overall.get("negative", 0),
                    "item_count": len(items)
                })
            
            report["sentiment_trend"] = sentiment_trend
        
        return report
    
    def export_model(self, output_dir=None):
        """
        Export the model for deployment.
        
        Args:
            output_dir: Optional output directory
            
        Returns:
            Path to exported model
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(settings.models_dir, "exports", f"sentiment_model_{timestamp}")
        
        # Create directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model
        self.save_model(output_dir)
        
        # Additional export actions
        # Create a simplified pipeline for inference
        pipeline_dir = os.path.join(output_dir, "pipeline")
        os.makedirs(pipeline_dir, exist_ok=True)
        
        # Save pipeline configuration
        pipeline_config = {
            "model_name": self.model_config["name"],
            "max_length": self.model_config["max_length"],
            "labels": self.model_config["labels"],
            "metadata": self.metadata
        }
        
        with open(os.path.join(pipeline_dir, "config.json"), "w") as f:
            json.dump(pipeline_config, f, indent=2)
        
        # Create a README
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
            f.write("from transformers import pipeline\n")
            f.write("from src.models.sentiment import FinancialSentimentModel\n\n")
            f.write("# Load the model\n")
            f.write(f"model = FinancialSentimentModel(model_path='{output_dir}')\n\n")
            f.write("# Analyze text\n")
            f.write("result = model.analyze_sentiment('Apple reported better than expected earnings.')\n")
            f.write("print(result)\n")
            f.write("```\n")
        
        logger.info(f"Model exported to {output_dir}")
        return output_dir

# Create a global instance of the model
try:
    sentiment_model = FinancialSentimentModel(
        model_path=settings.model.sentiment_model_path,
        model_name="finbert",
        use_ensemble=True,
        entity_extraction=True,
        temporal_tracking=True
    )
    logger.info("Sentiment model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing sentiment model: {e}")
    # Create a fallback model with reduced functionality
    try:
        sentiment_model = FinancialSentimentModel(
            model_name="finbert",
            use_ensemble=False,
            entity_extraction=False,
            temporal_tracking=False
        )
        logger.info("Fallback sentiment model initialized")
    except:
        sentiment_model = None
        logger.critical("Failed to initialize any sentiment model")

# Utility functions
def analyze_sentiment(text, extract_entities=True):
    """Analyze sentiment of a text."""
    if sentiment_model is None:
        return {"error": "Sentiment model not available"}
    
    return sentiment_model.analyze_sentiment(text, extract_entities=extract_entities)

def analyze_news_batch(news_items):
    """Analyze sentiment of a batch of news items."""
    if sentiment_model is None:
        return {"error": "Sentiment model not available"}
    
    return sentiment_model.analyze_news_items(news_items)

def generate_sentiment_report(news_items):
    """Generate a comprehensive sentiment report from news items."""
    if sentiment_model is None:
        return {"error": "Sentiment model not available"}
    
    return sentiment_model.generate_sentiment_report(news_items)

def get_entity_sentiment_network():
    """Get the entity sentiment network."""
    if sentiment_model is None or not sentiment_model.entity_tracker:
        return {"nodes": [], "edges": []}
    
    return sentiment_model.get_entity_network()

# Scheduled training function
async def train_sentiment_model(training_data=None, use_default_data=True):
    """
    Train or fine-tune the sentiment model.
    
    Args:
        training_data: Optional custom training data
        use_default_data: Whether to use default financial sentiment data
        
    Returns:
        Training metrics
    """
    # Check if we have a model to train
    if sentiment_model is None:
        logger.error("Cannot train: No sentiment model available")
        return {"error": "No sentiment model available"}
    
    # Prepare training data
    if training_data is None and use_default_data:
        # Load default financial sentiment data
        training_data = _load_default_training_data()
    
    if not training_data or "texts" not in training_data or "labels" not in training_data:
        logger.error("Invalid or missing training data")
        return {"error": "Invalid or missing training data"}
    
    # Train the model
    try:
        logger.info(f"Training sentiment model with {len(training_data['texts'])} samples")
        
        history = sentiment_model.fine_tune(
            texts=training_data["texts"],
            labels=training_data["labels"],
            eval_texts=training_data.get("eval_texts"),
            eval_labels=training_data.get("eval_labels"),
            epochs=3,
            batch_size=16,
            learning_rate=2e-5
        )
        
        # Save model
        sentiment_model.save_model(settings.model.sentiment_model_path)
        
        return {
            "status": "success",
            "history": history,
            "metrics": sentiment_model.metrics
        }
    except Exception as e:
        logger.error(f"Error training sentiment model: {e}")
        return {"error": str(e)}

def _load_default_training_data():
    """
    Load default financial sentiment training data.
    
    Returns:
        Dictionary with texts and labels
    """
    # Try to load from data directory
    data_path = os.path.join(settings.data_dir, "sentiment", "financial_sentiment_data.json")
    
    if os.path.exists(data_path):
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded default sentiment training data with {len(data.get('texts', []))} samples")
            return data
        except Exception as e:
            logger.error(f"Error loading default sentiment data: {e}")
    
    # If no default data, create a small synthetic dataset
    # This is just for demonstration - a real implementation would use proper labeled data
    texts = [
        "The company reported record profits for the quarter.",
        "Shares plummeted after the disappointing earnings report.",
        "The market remained stable throughout the trading session.",
        "Investors are concerned about the company's high debt levels.",
        "The new product launch exceeded all expectations.",
        "The company announced a significant expansion into new markets.",
        "Revenue growth slowed more than analysts had predicted.",
        "The board approved a new share buyback program.",
        "Economic indicators suggest a potential recession next year.",
        "The merger deal was approved by regulatory authorities."
    ]
    
    labels = [2, 0, 1, 0, 2, 2, 0, 2, 0, 1]  # 0=negative, 1=neutral, 2=positive
    
    return {"texts": texts, "labels": labels}

# Schedule daily model updates
async def schedule_sentiment_model_training():
    """Schedule sentiment model training to run during off-hours."""
    try:
        # Get current hour
        current_hour = datetime.now().hour
        
        # Define off-hours (typically outside of market hours)
        # US market hours are 9:30 AM - 4:00 PM Eastern Time
        is_off_hours = current_hour < 9 or current_hour > 16
        
        if is_off_hours:
            logger.info("Scheduling sentiment model training during off-hours")
            
            # Check if we should run full training (weekly on weekends)
            is_weekend = datetime.now().weekday() >= 5  # 5=Saturday, 6=Sunday
            
            if is_weekend:
                # Run full training on weekends
                logger.info("Running full sentiment model training (weekend schedule)")
                
                # Try to get more comprehensive training data for weekend updates
                training_data = await _gather_enhanced_training_data()
                
                # Train with the enhanced data if available
                if training_data and len(training_data.get("texts", [])) > 100:
                    await train_sentiment_model(training_data, use_default_data=False)
                else:
                    # Fall back to default data
                    await train_sentiment_model(use_default_data=True)
            else:
                # Run lighter fine-tuning on weekdays
                logger.info("Running incremental sentiment model update (weekday schedule)")
                
                # Get recent news for fine-tuning
                recent_data = await _gather_recent_news_data()
                
                if recent_data and len(recent_data.get("texts", [])) > 10:
                    # Fine-tune with recent data (fewer epochs)
                    try:
                        sentiment_model.fine_tune(
                            texts=recent_data["texts"],
                            labels=recent_data["labels"],
                            epochs=1,
                            batch_size=8
                        )
                    except Exception as e:
                        logger.error(f"Error in incremental update: {e}")
        else:
            logger.info("Not in off-hours, skipping scheduled training")
    except Exception as e:
        logger.error(f"Error in schedule_sentiment_model_training: {e}")

async def _gather_enhanced_training_data():
    """
    Gather enhanced training data from multiple sources.
    
    Returns:
        Dictionary with texts and labels
    """
    # In a real implementation, this would gather data from:
    # 1. Stored news articles with manually labeled sentiment
    # 2. Financial social media content
    # 3. Financial reports and filings
    
    # For demonstration, return None to fall back to default data
    return None

async def _gather_recent_news_data():
    """
    Gather recent financial news for incremental model updates.
    
    Returns:
        Dictionary with texts and labels
    """
    # In a real implementation, this would:
    # 1. Fetch recent news articles from the database
    # 2. Use the current model to predict sentiment
    # 3. Filter high-confidence predictions for self-training
    
    # For demonstration, return None to skip incremental updates
    return None