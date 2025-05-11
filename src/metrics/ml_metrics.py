"""
ML Metrics Collection Module

This module provides functionality for tracking and analyzing AI/ML model performance metrics:
- Accuracy metrics (accuracy, precision, recall, F1 score)
- Latency metrics (inference time, request time)
- Output quality metrics (confidence scores, error rates)
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, Optional
import statistics
from collections import defaultdict, deque

import numpy as np

from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("ml_metrics")

# Constants
MAX_HISTORY_SIZE = 1000  # Maximum number of metrics to keep in memory
REDIS_METRICS_KEY_PREFIX = "metrics:ml:"
REDIS_METRICS_EXPIRY = 86400 * 7  # 7 days


class MLMetricsCollector:
    """
    Collector for ML model metrics.
    
    This class provides methods for recording and retrieving metrics for ML models,
    including accuracy, latency, and output quality metrics.
    """
    
    def __init__(self, model_name: str, metrics_history_size: int = MAX_HISTORY_SIZE):
        """
        Initialize the metrics collector.
        
        Args:
            model_name: Name of the model
            metrics_history_size: Maximum number of metrics to keep in memory
        """
        self.model_name = model_name
        self.metrics_history_size = metrics_history_size
        
        # Initialize metrics storage
        self.latency_history = deque(maxlen=metrics_history_size)
        self.accuracy_history = deque(maxlen=metrics_history_size)
        self.confidence_history = deque(maxlen=metrics_history_size)
        self.error_history = deque(maxlen=metrics_history_size)
        
        # Load existing metrics from Redis if available
        self._load_from_redis()
        
        logger.info(f"ML Metrics Collector initialized for model: {model_name}")
    
    def _load_from_redis(self):
        """Load metrics from Redis if available."""
        try:
            # Load latency metrics
            latency_key = f"{REDIS_METRICS_KEY_PREFIX}{self.model_name}:latency"
            latency_data = redis_client.get(latency_key)
            if latency_data:
                self.latency_history.extend(json.loads(latency_data))
            
            # Load accuracy metrics
            accuracy_key = f"{REDIS_METRICS_KEY_PREFIX}{self.model_name}:accuracy"
            accuracy_data = redis_client.get(accuracy_key)
            if accuracy_data:
                self.accuracy_history.extend(json.loads(accuracy_data))
            
            # Load confidence metrics
            confidence_key = f"{REDIS_METRICS_KEY_PREFIX}{self.model_name}:confidence"
            confidence_data = redis_client.get(confidence_key)
            if confidence_data:
                self.confidence_history.extend(json.loads(confidence_data))
            
            # Load error metrics
            error_key = f"{REDIS_METRICS_KEY_PREFIX}{self.model_name}:error"
            error_data = redis_client.get(error_key)
            if error_data:
                self.error_history.extend(json.loads(error_data))
                
            logger.debug(f"Loaded metrics from Redis for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading metrics from Redis: {e}")
    
    def _save_to_redis(self):
        """Save metrics to Redis."""
        try:
            # Save latency metrics
            latency_key = f"{REDIS_METRICS_KEY_PREFIX}{self.model_name}:latency"
            redis_client.set(latency_key, json.dumps(list(self.latency_history)), ex=REDIS_METRICS_EXPIRY)
            
            # Save accuracy metrics
            accuracy_key = f"{REDIS_METRICS_KEY_PREFIX}{self.model_name}:accuracy"
            redis_client.set(accuracy_key, json.dumps(list(self.accuracy_history)), ex=REDIS_METRICS_EXPIRY)
            
            # Save confidence metrics
            confidence_key = f"{REDIS_METRICS_KEY_PREFIX}{self.model_name}:confidence"
            redis_client.set(confidence_key, json.dumps(list(self.confidence_history)), ex=REDIS_METRICS_EXPIRY)
            
            # Save error metrics
            error_key = f"{REDIS_METRICS_KEY_PREFIX}{self.model_name}:error"
            redis_client.set(error_key, json.dumps(list(self.error_history)), ex=REDIS_METRICS_EXPIRY)
            
            logger.debug(f"Saved metrics to Redis for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error saving metrics to Redis: {e}")
    
    def record_latency(self, latency_ms: float, operation: str = "inference"):
        """
        Record latency metric.
        
        Args:
            latency_ms: Latency in milliseconds
            operation: Type of operation (e.g., "inference", "training")
        """
        metric = {
            "timestamp": datetime.now().isoformat(),
            "latency_ms": latency_ms,
            "operation": operation
        }
        self.latency_history.append(metric)
        self._save_to_redis()
        
        # Log if latency is unusually high
        if latency_ms > 1000:  # More than 1 second
            logger.warning(f"High latency detected for {self.model_name}: {latency_ms:.2f}ms")
    
    def record_accuracy(self, 
                        accuracy: Optional[float] = None,
                        precision: Optional[float] = None,
                        recall: Optional[float] = None,
                        f1: Optional[float] = None,
                        true_positive: Optional[int] = None,
                        false_positive: Optional[int] = None,
                        true_negative: Optional[int] = None,
                        false_negative: Optional[int] = None):
        """
        Record accuracy metrics.
        
        Args:
            accuracy: Overall accuracy
            precision: Precision score
            recall: Recall score
            f1: F1 score
            true_positive: Number of true positives
            false_positive: Number of false positives
            true_negative: Number of true negatives
            false_negative: Number of false negatives
        """
        metric = {
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative
        }
        # Filter out None values
        metric = {k: v for k, v in metric.items() if v is not None}
        
        self.accuracy_history.append(metric)
        self._save_to_redis()
    
    def record_confidence(self, confidence: float, correct: bool, prediction_type: str = "classification"):
        """
        Record confidence metric.
        
        Args:
            confidence: Confidence score (0.0 to 1.0)
            correct: Whether the prediction was correct
            prediction_type: Type of prediction
        """
        metric = {
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "correct": correct,
            "prediction_type": prediction_type
        }
        self.confidence_history.append(metric)
        self._save_to_redis()
    
    def record_error(self, error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None):
        """
        Record error metric.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context for the error
        """
        metric = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }
        self.error_history.append(metric)
        self._save_to_redis()
        
        # Log the error
        logger.error(f"ML Error in {self.model_name}: {error_type} - {error_message}")
    
    def get_latency_stats(self, operation: Optional[str] = None, time_window: Optional[int] = None) -> Dict[str, float]:
        """
        Get latency statistics.
        
        Args:
            operation: Filter by operation type
            time_window: Time window in seconds to consider
            
        Returns:
            Dictionary with latency statistics
        """
        # Filter by operation and time window if specified
        filtered_latencies = self.latency_history
        
        if operation:
            filtered_latencies = [m for m in filtered_latencies if m["operation"] == operation]
        
        if time_window:
            cutoff_time = datetime.now().timestamp() - time_window
            filtered_latencies = [
                m for m in filtered_latencies 
                if datetime.fromisoformat(m["timestamp"]).timestamp() > cutoff_time
            ]
        
        # Extract latency values
        latency_values = [m["latency_ms"] for m in filtered_latencies]
        
        if not latency_values:
            return {
                "count": 0,
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "p95": 0,
                "p99": 0
            }
        
        # Calculate statistics
        return {
            "count": len(latency_values),
            "min": min(latency_values),
            "max": max(latency_values),
            "mean": statistics.mean(latency_values),
            "median": statistics.median(latency_values),
            "p95": np.percentile(latency_values, 95),
            "p99": np.percentile(latency_values, 99)
        }
    
    def get_accuracy_stats(self, time_window: Optional[int] = None) -> Dict[str, float]:
        """
        Get accuracy statistics.
        
        Args:
            time_window: Time window in seconds to consider
            
        Returns:
            Dictionary with accuracy statistics
        """
        # Filter by time window if specified
        filtered_metrics = self.accuracy_history
        
        if time_window:
            cutoff_time = datetime.now().timestamp() - time_window
            filtered_metrics = [
                m for m in filtered_metrics 
                if datetime.fromisoformat(m["timestamp"]).timestamp() > cutoff_time
            ]
        
        if not filtered_metrics:
            return {
                "count": 0,
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0
            }
        
        # Calculate statistics
        result = {
            "count": len(filtered_metrics)
        }
        
        # Calculate averages for each metric if available
        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            values = [m[metric_name] for m in filtered_metrics if metric_name in m]
            if values:
                result[metric_name] = statistics.mean(values)
            else:
                result[metric_name] = 0
        
        return result
    
    def get_confidence_stats(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Get confidence statistics.
        
        Args:
            time_window: Time window in seconds to consider
            
        Returns:
            Dictionary with confidence statistics
        """
        # Filter by time window if specified
        filtered_metrics = self.confidence_history
        
        if time_window:
            cutoff_time = datetime.now().timestamp() - time_window
            filtered_metrics = [
                m for m in filtered_metrics 
                if datetime.fromisoformat(m["timestamp"]).timestamp() > cutoff_time
            ]
        
        if not filtered_metrics:
            return {
                "count": 0,
                "mean_confidence": 0,
                "correct_predictions": 0,
                "incorrect_predictions": 0,
                "accuracy": 0,
                "calibration_error": 0
            }
        
        # Extract values
        confidence_values = [m["confidence"] for m in filtered_metrics]
        correct_predictions = [m for m in filtered_metrics if m["correct"]]
        incorrect_predictions = [m for m in filtered_metrics if not m["correct"]]
        
        # Calculate statistics
        result = {
            "count": len(filtered_metrics),
            "mean_confidence": statistics.mean(confidence_values),
            "correct_predictions": len(correct_predictions),
            "incorrect_predictions": len(incorrect_predictions),
            "accuracy": len(correct_predictions) / len(filtered_metrics) if filtered_metrics else 0
        }
        
        # Calculate calibration error (difference between confidence and accuracy)
        result["calibration_error"] = abs(result["mean_confidence"] - result["accuracy"])
        
        return result
    
    def get_error_stats(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Args:
            time_window: Time window in seconds to consider
            
        Returns:
            Dictionary with error statistics
        """
        # Filter by time window if specified
        filtered_metrics = self.error_history
        
        if time_window:
            cutoff_time = datetime.now().timestamp() - time_window
            filtered_metrics = [
                m for m in filtered_metrics 
                if datetime.fromisoformat(m["timestamp"]).timestamp() > cutoff_time
            ]
        
        if not filtered_metrics:
            return {
                "count": 0,
                "error_types": {}
            }
        
        # Count errors by type
        error_types = defaultdict(int)
        for m in filtered_metrics:
            error_types[m["error_type"]] += 1
        
        return {
            "count": len(filtered_metrics),
            "error_types": dict(error_types)
        }
    
    def get_all_metrics(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Get all metrics.
        
        Args:
            time_window: Time window in seconds to consider
            
        Returns:
            Dictionary with all metrics
        """
        return {
            "model_name": self.model_name,
            "latency": self.get_latency_stats(time_window=time_window),
            "accuracy": self.get_accuracy_stats(time_window=time_window),
            "confidence": self.get_confidence_stats(time_window=time_window),
            "errors": self.get_error_stats(time_window=time_window),
            "timestamp": datetime.now().isoformat()
        }


class MLMetricsRegistry:
    """
    Registry for ML metrics collectors.
    
    This class provides a central registry for all ML metrics collectors in the system.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one registry exists."""
        if cls._instance is None:
            cls._instance = super(MLMetricsRegistry, cls).__new__(cls)
            cls._instance.collectors = {}
            cls._instance.logger = setup_logger("ml_metrics_registry")
        return cls._instance
    
    def get_collector(self, model_name: str) -> MLMetricsCollector:
        """
        Get or create a metrics collector for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            MLMetricsCollector for the model
        """
        if model_name not in self.collectors:
            self.collectors[model_name] = MLMetricsCollector(model_name)
            self.logger.info(f"Created new metrics collector for model: {model_name}")
        return self.collectors[model_name]
    
    def get_all_metrics(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Get metrics for all models.
        
        Args:
            time_window: Time window in seconds to consider
            
        Returns:
            Dictionary with metrics for all models
        """
        return {
            model_name: collector.get_all_metrics(time_window=time_window)
            for model_name, collector in self.collectors.items()
        }


# Create global registry instance
metrics_registry = MLMetricsRegistry()


def get_collector(model_name: str) -> MLMetricsCollector:
    """
    Get a metrics collector for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        MLMetricsCollector for the model
    """
    return metrics_registry.get_collector(model_name)


def get_all_metrics(time_window: Optional[int] = None) -> Dict[str, Any]:
    """
    Get metrics for all models.
    
    Args:
        time_window: Time window in seconds to consider
        
    Returns:
        Dictionary with metrics for all models
    """
    return metrics_registry.get_all_metrics(time_window=time_window)


class MetricsTimer:
    """
    Context manager for timing operations and recording metrics.
    
    Example:
        with MetricsTimer("pattern_recognition", "inference") as timer:
            result = model.predict(data)
    """
    
    def __init__(self, model_name: str, operation: str = "inference"):
        """
        Initialize the timer.
        
        Args:
            model_name: Name of the model
            operation: Type of operation
        """
        self.model_name = model_name
        self.operation = operation
        self.collector = get_collector(model_name)
        self.start_time = None
        
    def __enter__(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop the timer and record the metric.
        
        If an exception occurred, also record an error metric.
        """
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            latency_ms = elapsed_time * 1000
            self.collector.record_latency(latency_ms, self.operation)
            
            if exc_type is not None:
                # Record error if an exception occurred
                self.collector.record_error(
                    error_type=exc_type.__name__,
                    error_message=str(exc_val),
                    context={"operation": self.operation}
                )