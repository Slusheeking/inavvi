"""
Sentiment analysis model for news and social media data.

This model uses a fine-tuned BERT model to analyze sentiment
of financial news and social media content.
"""
import os
import json
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config.settings import settings
from src.utils.logging import setup_logger

# Filter out specific huggingface_hub warnings
warnings.filterwarnings("ignore", message=".*`resume_download` is deprecated.*")

# Set up logger
logger = setup_logger("sentiment_model")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class FinancialSentimentModel:
    """
    Sentiment analysis model for financial news and social media.
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        model_name: str = "ProsusAI/finbert"
    ):
        """
        Initialize the sentiment model.
        
        Args:
            model_path: Path to the fine-tuned model directory
            model_name: Pre-trained model name
        """
        self.model_name = model_name
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded tokenizer from {model_name}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
        
        # Load model
        try:
            # If fine-tuned model exists, load it
            model_loaded = False
            if model_path:
                # Convert to absolute path if it's a relative path
                if not os.path.isabs(model_path):
                    model_path = os.path.join(settings.models_dir, os.path.basename(model_path))
                
                # Check if it's a valid model directory with config.json
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    logger.info(f"Loaded fine-tuned model from {model_path}")
                    model_loaded = True
            
            # If model wasn't loaded, use pre-trained model
            if not model_loaded:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                logger.info(f"Loaded pre-trained model from {model_name}")
            
            # Move model to device
            self.model.to(device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Load label mapping
        if model_name == "ProsusAI/finbert":
            self.labels = ["negative", "neutral", "positive"]
        else:
            # Try to load labels from config
            if hasattr(self.model.config, "id2label"):
                self.labels = list(self.model.config.id2label.values())
            else:
                self.labels = ["negative", "neutral", "positive"]
    
    def save_model(self, save_dir: str):
        """
        Save the model to directory.
        
        Args:
            save_dir: Directory to save the model
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            
            # Save labels
            with open(os.path.join(save_dir, "labels.json"), "w") as f:
                json.dump(self.labels, f)
            
            logger.info(f"Saved model to {save_dir}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def fine_tune(
        self,
        texts: List[str],
        labels: List[int],
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ):
        """
        Fine-tune the model on financial data.
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels (0=negative, 1=neutral, 2=positive)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        # Set model to training mode
        self.model.train()
        
        # Tokenize data
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        
        # Create dataset
        class SentimentDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)
        
        dataset = SentimentDataset(encodings, labels)
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Initialize training history
        history = {
            'loss': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Initialize metrics
            total_loss = 0
            
            # Train on batches
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
            
            # Calculate average loss
            avg_loss = total_loss / len(dataloader)
            
            # Update history
            history['loss'].append(avg_loss)
            
            # Print progress
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
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
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment of a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of sentiment scores
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)[0]
        
        # Convert to dictionary
        result = {
            label: float(prob) for label, prob in zip(self.labels, probs.cpu().numpy())
        }
        
        return result
    
    def analyze_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze the sentiment of multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment score dictionaries
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize results
        results = []
        
        # Process in batches to avoid memory issues
        batch_size = 16
        
        for i in range(0, len(texts), batch_size):
            # Get batch
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
            
            # Convert to dictionaries
            batch_results = [
                {label: float(prob) for label, prob in zip(self.labels, probs[j].cpu().numpy())}
                for j in range(len(batch_texts))
            ]
            
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
                text = f"{title} {summary}"
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
            result['sentiment'] = sentiment
            
            # Calculate overall sentiment score (-1 to 1)
            neg = sentiment.get('negative', 0)
            pos = sentiment.get('positive', 0)
            result['sentiment_score'] = pos - neg
            
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
        sentiments = {label: 0.0 for label in self.labels}
        total_weight = 0.0
        
        # Process each item
        for item in news_items:
            # Get sentiment
            item_sentiment = item.get('sentiment', {})
            
            # Get item weight (e.g., relevance)
            weight = item.get('relevance_score', 1.0)
            
            # Update counts
            for label in self.labels:
                sentiments[label] += item_sentiment.get(label, 0.0) * weight
            
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            for label in sentiments:
                sentiments[label] /= total_weight
        
        # Calculate overall score (-1 to 1)
        neg = sentiments.get('negative', 0)
        pos = sentiments.get('positive', 0)
        sentiments['overall_score'] = pos - neg
        
        return sentiments

# Create a global instance of the model
try:
    sentiment_model = FinancialSentimentModel(
        model_path=settings.model.sentiment_model_path,
        model_name="ProsusAI/finbert"
    )
    logger.info("Sentiment model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing sentiment model: {e}")
    # Create a fallback model
    sentiment_model = None
