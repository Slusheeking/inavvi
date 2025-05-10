"""
Multi-factor ranking model for stock screening and selection.

This model ranks stocks based on technical, volume, and momentum features
to identify the best trading opportunities using an ensemble approach.
"""

# Convenience functions for direct import
def rank_opportunities(stock_data, factor_weights=None, model_version="ensemble"):
    """
    Rank trading opportunities based on the multi-factor model.
    
    Args:
        stock_data: Dictionary of stock DataFrames with features
        factor_weights: Optional dictionary of factor weights
        model_version: Model version to use (ensemble, xgboost, lightgbm, catboost, rf)
        
    Returns:
        List of ranked stocks with scores
    """
    model = RankingModel(factor_weights=factor_weights, model_version=model_version)
    return model.rank_stocks(stock_data)

def get_model_weights(model_version="ensemble"):
    """
    Get the current weights used in the ranking model.
    
    Args:
        model_version: Model version to get weights for
        
    Returns:
        Dictionary of model weights
    """
    model = RankingModel(model_version=model_version)
    return model.hyperparams['meta']['weights']
import sys
import os
import asyncio
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from scipy.stats import pearsonr, spearmanr
import optuna
from joblib import dump, load

from src.config.settings import settings
from src.utils.logging import setup_logger, log_execution_time
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("ranking_model")

class RankingModel:
    """
    Multi-factor ranking model for stock selection.
    
    Uses an ensemble of XGBoost, LightGBM, and CatBoost for more robust predictions,
    with time-aware validation and feature importance tracking.
    """
    
    def __init__(self, factor_weights=None, model_version="ensemble"):
        """
        Initialize the ranking model.
        
        Args:
            factor_weights: Optional dictionary of factor weights
            model_version: Model version to use (ensemble, xgboost, lightgbm, catboost, rf)
        """
        self.models = {}
        self.feature_names = []
        self.model_version = model_version
        self.model_dir = os.path.join(settings.models_dir, "ranking")
        self.model_path = settings.model.ranking_model_path
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.feature_correlations = {}
        self.meta_features = {}
        self.last_trained = None
        self.model_metrics = {}
        self.versions_history = []
        
        # Set factor weights from config or use provided weights or defaults
        if factor_weights:
            self.factor_weights = factor_weights
        else:
            try:
                self.factor_weights = settings.model.factor_weights
            except AttributeError:
                # Fallback to default weights if not in config
                self.factor_weights = {
                    'momentum': 0.3,
                    'volume': 0.2,
                    'volatility': 0.2,
                    'trend': 0.2,
                    'value': 0.1
                }
        
        # Default hyperparameters for models
        self.hyperparams = {
            'xgboost': {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'tree_method': 'hist',  # Use CPU-based histogram method
                'device': 'cpu',        # Explicitly use CPU
                'random_state': 42
            },
            'lightgbm': {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbosity': -1,
                'random_state': 42
            },
            'catboost': {
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.05,
                'random_seed': 42,
                'verbose': 0,
                'allow_writing_files': False
            },
            'randomforest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'meta': {
                'method': 'weighted_average',  # weighted_average, stacking
                'weights': {
                    'xgboost': 0.4,
                    'lightgbm': 0.3,
                    'catboost': 0.3
                }
            }
        }
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load model if it exists
        self._load_model()
        
    def _load_model(self) -> bool:
        """
        Load model from disk if it exists.
        
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        try:
            # First try to load the ensemble models
            ensemble_path = os.path.join(self.model_dir, "ensemble")
            os.makedirs(ensemble_path, exist_ok=True)
            
            # Check for metadata file
            metadata_path = os.path.join(ensemble_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.feature_names = metadata.get('feature_names', [])
                self.feature_importance = metadata.get('feature_importance', {})
                self.feature_correlations = metadata.get('feature_correlations', {})
                self.model_metrics = metadata.get('model_metrics', {})
                self.last_trained = metadata.get('last_trained')
                self.versions_history = metadata.get('versions_history', [])
                self.hyperparams = metadata.get('hyperparams', self.hyperparams)
                
                # Check if scaler exists
                scaler_path = os.path.join(ensemble_path, "scaler.joblib")
                if os.path.exists(scaler_path):
                    self.scaler = load(scaler_path)
                
                # Load each model if it exists
                model_loaded = False
                
                # XGBoost model
                xgb_path = os.path.join(ensemble_path, "xgboost.model")
                if os.path.exists(xgb_path):
                    try:
                        self.models['xgboost'] = xgb.Booster()
                        self.models['xgboost'].load_model(xgb_path)
                        model_loaded = True
                    except Exception as e:
                        logger.error(f"Error loading XGBoost model: {e}")
                
                # LightGBM model
                lgb_path = os.path.join(ensemble_path, "lightgbm.txt")
                if os.path.exists(lgb_path):
                    try:
                        self.models['lightgbm'] = lgb.Booster(model_file=lgb_path)
                        model_loaded = True
                    except Exception as e:
                        logger.error(f"Error loading LightGBM model: {e}")
                
                # CatBoost model
                cb_path = os.path.join(ensemble_path, "catboost.cbm")
                if os.path.exists(cb_path):
                    try:
                        self.models['catboost'] = cb.CatBoost()
                        self.models['catboost'].load_model(cb_path)
                        model_loaded = True
                    except Exception as e:
                        logger.error(f"Error loading CatBoost model: {e}")
                
                # Random Forest model
                rf_path = os.path.join(ensemble_path, "randomforest.joblib")
                if os.path.exists(rf_path):
                    try:
                        self.models['randomforest'] = load(rf_path)
                        model_loaded = True
                    except Exception as e:
                        logger.error(f"Error loading Random Forest model: {e}")
                
                if model_loaded:
                    logger.info(f"Models loaded from {ensemble_path}")
                    self._log_model_info()
                    return True
            
            # If ensemble models not found, try to load legacy model
            if os.path.exists(self.model_path):
                try:
                    # Load legacy XGBoost model
                    self.models['xgboost'] = xgb.Booster()
                    self.models['xgboost'].load_model(self.model_path)
                    
                    # Load metadata if exists
                    legacy_metadata_path = Path(self.model_path).with_suffix('.json')
                    if legacy_metadata_path.exists():
                        with open(legacy_metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        self.feature_names = metadata.get('feature_names', [])
                        self.feature_importance = metadata.get('feature_importance', {})
                        self.last_trained = metadata.get('last_trained')
                    
                    logger.info(f"Legacy model loaded from {self.model_path}")
                    self._log_model_info()
                    
                    # Now save in the new format for future
                    self.save_model()
                    return True
                except Exception as e:
                    logger.error(f"Error loading legacy model: {e}")
            
            logger.warning(f"No models found at {ensemble_path} or {self.model_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Set model to None to use simple ranking
            self.models = {}
            return False
    
    def _log_model_info(self):
        """Log information about the loaded model"""
        logger.info(f"Model information:")
        logger.info(f"  Last trained: {self.last_trained}")
        logger.info(f"  Models loaded: {list(self.models.keys())}")
        logger.info(f"  Number of features: {len(self.feature_names)}")
        
        if self.model_metrics:
            logger.info(f"  Model metrics:")
            for model_name, metrics in self.model_metrics.items():
                if isinstance(metrics, dict):
                    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                    logger.info(f"    {model_name}: {metrics_str}")
        
    def save_model(self) -> bool:
        """
        Save model to disk.
        
        Returns:
            bool: True if model was saved successfully, False otherwise
        """
        try:
            if not self.models:
                logger.warning("No models to save")
                return False
            
            # Create directory if it doesn't exist
            ensemble_path = os.path.join(self.model_dir, "ensemble")
            os.makedirs(ensemble_path, exist_ok=True)
            
            # Save each model
            for model_name, model in self.models.items():
                if model_name == 'xgboost' and model is not None:
                    model.save_model(os.path.join(ensemble_path, "xgboost.model"))
                elif model_name == 'lightgbm' and model is not None:
                    model.save_model(os.path.join(ensemble_path, "lightgbm.txt"))
                elif model_name == 'catboost' and model is not None:
                    model.save_model(os.path.join(ensemble_path, "catboost.cbm"))
                elif model_name == 'randomforest' and model is not None:
                    dump(model, os.path.join(ensemble_path, "randomforest.joblib"))
            
            # Save scaler
            dump(self.scaler, os.path.join(ensemble_path, "scaler.joblib"))
            
            # Add current version to history if not empty
            if self.last_trained:
                version_info = {
                    'version': len(self.versions_history) + 1,
                    'timestamp': self.last_trained,
                    'models': list(self.models.keys()),
                    'metrics': self.model_metrics
                }
                self.versions_history.append(version_info)
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'feature_correlations': self.feature_correlations,
                'model_metrics': self.model_metrics,
                'last_trained': datetime.now().isoformat(),
                'versions_history': self.versions_history,
                'hyperparams': self.hyperparams
            }
            
            with open(os.path.join(ensemble_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Also save to legacy path for backward compatibility
            if 'xgboost' in self.models and self.models['xgboost'] is not None:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.models['xgboost'].save_model(self.model_path)
                
                legacy_metadata_path = Path(self.model_path).with_suffix('.json')
                with open(legacy_metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Models saved to {ensemble_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def optimize_hyperparameters(
        self, 
        X_train, 
        y_train, 
        X_val, 
        y_val,
        model_name='xgboost',
        n_trials=30
    ):
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_name: Name of the model to optimize ('xgboost', 'lightgbm', 'catboost')
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'tree_method': 'hist',
                    'device': 'cpu',
                    'random_state': 42
                }
                
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)
                
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=100,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=10,
                    verbose_eval=False
                )
                
                # Predict on validation set
                y_pred = model.predict(dval)
                
            elif model_name == 'lightgbm':
                params = {
                    'objective': 'binary',
                    'metric': 'auc',
                    'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'verbosity': -1,
                    'random_state': 42
                }
                
                # Create dataset
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # Train model
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[val_data],
                    early_stopping_rounds=10,
                    verbose_eval=False
                )
                
                # Predict on validation set
                y_pred = model.predict(X_val)
                
            elif model_name == 'catboost':
                params = {
                    'loss_function': 'Logloss',
                    'eval_metric': 'AUC',
                    'iterations': 100,
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                    'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
                    'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
                    'verbose': 0,
                    'random_seed': 42,
                    'allow_writing_files': False
                }
                
                # Create dataset
                train_data = cb.Pool(X_train, label=y_train)
                val_data = cb.Pool(X_val, label=y_val)
                
                # Train model
                model = cb.CatBoost(params)
                model.fit(
                    train_data,
                    eval_set=val_data,
                    early_stopping_rounds=10,
                    verbose=False
                )
                
                # Predict on validation set
                y_pred = model.predict(X_val, prediction_type='Probability')[:, 1]
            
            # Calculate AUC
            try:
                auc = roc_auc_score(y_val, y_pred)
                return auc
            except:
                # Return a low score if there's an error
                return 0.5
        
        # Create Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        
        # Update our hyperparameters with the best ones
        self.hyperparams[model_name].update(best_params)
        
        logger.info(f"Best {model_name} parameters: {best_params}")
        logger.info(f"Best AUC: {study.best_value:.4f}")
        
        return best_params
    
    @log_execution_time(logger)
    def train(
        self, 
        training_data: pd.DataFrame, 
        target_col: str = 'target',
        optimize_hyperparams: bool = False,
        n_trials: int = 30,
        use_time_series_cv: bool = True,
        test_size: float = 0.2
    ) -> bool:
        """
        Train the ranking model ensemble.
        
        Args:
            training_data: DataFrame with features and target
            target_col: Name of the target column
            optimize_hyperparams: Whether to optimize hyperparameters
            n_trials: Number of hyperparameter optimization trials
            use_time_series_cv: Whether to use time-series cross-validation
            test_size: Test size for validation
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            if training_data.empty:
                logger.error("Empty training data")
                return False
            
            # Separate features and target
            if target_col not in training_data.columns:
                logger.error(f"Target column '{target_col}' not found in training data")
                return False
            
            logger.info(f"Training ranking model on {len(training_data)} samples")
            
            # Store feature names (exclude non-feature columns)
            non_feature_cols = [target_col]
            if 'symbol' in training_data.columns:
                non_feature_cols.append('symbol')
            if 'date' in training_data.columns:
                non_feature_cols.append('date')
            
            X = training_data.drop(columns=non_feature_cols)
            y = training_data[target_col]
            
            self.feature_names = X.columns.tolist()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split into train and validation sets
            if use_time_series_cv and 'date' in training_data.columns:
                # Sort by date
                sorted_indices = training_data['date'].argsort()
                X_sorted = X_scaled[sorted_indices]
                y_sorted = y.iloc[sorted_indices].values
                
                # Use the last test_size portion for validation
                val_size = int(len(X_sorted) * test_size)
                train_size = len(X_sorted) - val_size
                
                X_train = X_sorted[:train_size]
                y_train = y_sorted[:train_size]
                X_val = X_sorted[train_size:]
                y_val = y_sorted[train_size:]
                
                logger.info(f"Using time-series split: {train_size} train, {val_size} validation")
            else:
                # Random split for non-time-series data
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42
                )
                logger.info(f"Using random split: {len(X_train)} train, {len(X_val)} validation")
            
            # Initialize model metrics
            self.model_metrics = {}
            
            # Train XGBoost model
            logger.info("Training XGBoost model...")
            
            if optimize_hyperparams:
                logger.info("Optimizing XGBoost hyperparameters...")
                self.optimize_hyperparameters(X_train, y_train, X_val, y_val, 'xgboost', n_trials)
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            
            # Train XGBoost model
            self.models['xgboost'] = xgb.train(
                self.hyperparams['xgboost'],
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=10,
                verbose_eval=False
            )
            
            # Get XGBoost feature importance
            xgb_importance = self.models['xgboost'].get_score(importance_type='gain')
            total = sum(xgb_importance.values())
            self.feature_importance['xgboost'] = {k: v/total for k, v in xgb_importance.items()}
            
            # Evaluate XGBoost model
            xgb_pred = self.models['xgboost'].predict(dval)
            xgb_pred_binary = (xgb_pred > 0.5).astype(int)
            
            self.model_metrics['xgboost'] = {
                'accuracy': accuracy_score(y_val, xgb_pred_binary),
                'precision': precision_score(y_val, xgb_pred_binary, zero_division=0),
                'recall': recall_score(y_val, xgb_pred_binary, zero_division=0),
                'f1': f1_score(y_val, xgb_pred_binary, zero_division=0),
                'auc': roc_auc_score(y_val, xgb_pred)
            }
            
            # Train LightGBM model
            logger.info("Training LightGBM model...")
            
            if optimize_hyperparams:
                logger.info("Optimizing LightGBM hyperparameters...")
                self.optimize_hyperparameters(X_train, y_train, X_val, y_val, 'lightgbm', n_trials)
            
            # Create dataset for LightGBM
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train LightGBM model
            self.models['lightgbm'] = lgb.train(
                self.hyperparams['lightgbm'],
                train_data,
                num_boost_round=100,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                early_stopping_rounds=10,
                verbose_eval=False
            )
            
            # Get LightGBM feature importance
            lgb_importance = dict(zip(
                self.feature_names,
                self.models['lightgbm'].feature_importance(importance_type='gain')
            ))
            total = sum(lgb_importance.values())
            self.feature_importance['lightgbm'] = {k: v/total for k, v in lgb_importance.items()} if total > 0 else {}
            
            # Evaluate LightGBM model
            lgb_pred = self.models['lightgbm'].predict(X_val)
            lgb_pred_binary = (lgb_pred > 0.5).astype(int)
            
            self.model_metrics['lightgbm'] = {
                'accuracy': accuracy_score(y_val, lgb_pred_binary),
                'precision': precision_score(y_val, lgb_pred_binary, zero_division=0),
                'recall': recall_score(y_val, lgb_pred_binary, zero_division=0),
                'f1': f1_score(y_val, lgb_pred_binary, zero_division=0),
                'auc': roc_auc_score(y_val, lgb_pred)
            }
            
            # Train CatBoost model
            logger.info("Training CatBoost model...")
            
            if optimize_hyperparams:
                logger.info("Optimizing CatBoost hyperparameters...")
                self.optimize_hyperparameters(X_train, y_train, X_val, y_val, 'catboost', n_trials)
            
            # Create dataset for CatBoost
            train_data = cb.Pool(X_train, label=y_train)
            val_data = cb.Pool(X_val, label=y_val)
            
            # Train CatBoost model
            self.models['catboost'] = cb.CatBoost(self.hyperparams['catboost'])
            self.models['catboost'].fit(
                train_data,
                eval_set=val_data,
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Get CatBoost feature importance
            cb_importance = dict(zip(
                self.feature_names,
                self.models['catboost'].get_feature_importance()
            ))
            total = sum(cb_importance.values())
            self.feature_importance['catboost'] = {k: v/total for k, v in cb_importance.items()} if total > 0 else {}
            
            # Evaluate CatBoost model
            cb_pred = self.models['catboost'].predict(X_val, prediction_type='Probability')[:, 1]
            cb_pred_binary = (cb_pred > 0.5).astype(int)
            
            self.model_metrics['catboost'] = {
                'accuracy': accuracy_score(y_val, cb_pred_binary),
                'precision': precision_score(y_val, cb_pred_binary, zero_division=0),
                'recall': recall_score(y_val, cb_pred_binary, zero_division=0),
                'f1': f1_score(y_val, cb_pred_binary, zero_division=0),
                'auc': roc_auc_score(y_val, cb_pred)
            }
            
            # Calculate combined feature importance
            self._combine_feature_importance()
            
            # Calculate feature correlations
            self._calculate_feature_correlations(X)
            
            # Update metadata
            self.last_trained = datetime.now().isoformat()
            
            # Calculate ensemble metrics
            ensemble_pred = self._ensemble_predict(X_val)
            ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
            
            self.model_metrics['ensemble'] = {
                'accuracy': accuracy_score(y_val, ensemble_pred_binary),
                'precision': precision_score(y_val, ensemble_pred_binary, zero_division=0),
                'recall': recall_score(y_val, ensemble_pred_binary, zero_division=0),
                'f1': f1_score(y_val, ensemble_pred_binary, zero_division=0),
                'auc': roc_auc_score(y_val, ensemble_pred)
            }
            
            # Optimize ensemble weights based on validation performance
            self._optimize_ensemble_weights(X_val, y_val)
            
            # Save model
            self.save_model()
            
            # Log training results
            logger.info("Model training completed with:")
            for model_name, metrics in self.model_metrics.items():
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"  {model_name}: {metrics_str}")
            
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def _combine_feature_importance(self):
        """Combine feature importance from all models"""
        # Initialize combined importance
        combined = {}
        
        # Get weights for each model
        weights = self.hyperparams['meta']['weights']
        
        # Combine feature importance
        for feature in self.feature_names:
            combined[feature] = 0
            
            for model_name, importance in self.feature_importance.items():
                if model_name in weights and feature in importance:
                    combined[feature] += importance[feature] * weights[model_name]
        
        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v/total for k, v in combined.items()}
        
        # Store combined importance
        self.feature_importance['combined'] = combined
    
    def _calculate_feature_correlations(self, X: pd.DataFrame):
        """Calculate feature correlations"""
        # Calculate correlation matrix
        corr_matrix = X.corr()
        
        # Store top correlations for each feature
        self.feature_correlations = {}
        
        for feature in self.feature_names:
            # Get correlations for this feature
            correlations = corr_matrix[feature].sort_values(ascending=False)
            
            # Store top 5 correlations (excluding self)
            self.feature_correlations[feature] = {
                other_feature: corr for other_feature, corr in correlations.items()
                if other_feature != feature
            }
    
    def _optimize_ensemble_weights(self, X_val, y_val):
        """Optimize ensemble weights based on validation performance"""
        def objective(trial):
            # Get weights
            xgb_weight = trial.suggest_float('xgb_weight', 0.1, 1.0)
            lgb_weight = trial.suggest_float('lgb_weight', 0.1, 1.0)
            cb_weight = trial.suggest_float('cb_weight', 0.1, 1.0)
            
            # Normalize weights
            total = xgb_weight + lgb_weight + cb_weight
            xgb_weight /= total
            lgb_weight /= total
            cb_weight /= total
            
            # Get predictions
            xgb_pred = self.models['xgboost'].predict(xgb.DMatrix(X_val, feature_names=self.feature_names))
            lgb_pred = self.models['lightgbm'].predict(X_val)
            cb_pred = self.models['catboost'].predict(X_val, prediction_type='Probability')[:, 1]
            
            # Calculate weighted ensemble prediction
            ensemble_pred = (
                xgb_weight * xgb_pred + 
                lgb_weight * lgb_pred + 
                cb_weight * cb_pred
            )
            
            # Calculate AUC
            auc = roc_auc_score(y_val, ensemble_pred)
            
            return auc
        
        # Create Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)
        
        # Get best weights
        best_params = study.best_params
        
        # Normalize weights
        total = sum(best_params.values())
        normalized_weights = {
            'xgboost': best_params['xgb_weight'] / total,
            'lightgbm': best_params['lgb_weight'] / total,
            'catboost': best_params['cb_weight'] / total
        }
        
        # Update weights
        self.hyperparams['meta']['weights'] = normalized_weights
        
        logger.info(f"Optimized ensemble weights: {normalized_weights}")
    
    def _ensemble_predict(self, X):
        """Get prediction from ensemble of models"""
        # Get predictions from individual models
        xgb_pred = self.models['xgboost'].predict(xgb.DMatrix(X, feature_names=self.feature_names))
        lgb_pred = self.models['lightgbm'].predict(X)
        cb_pred = self.models['catboost'].predict(X, prediction_type='Probability')[:, 1]
        
        # Get weights
        weights = self.hyperparams['meta']['weights']
        
        # Weighted average
        ensemble_pred = (
            weights['xgboost'] * xgb_pred + 
            weights['lightgbm'] * lgb_pred + 
            weights['catboost'] * cb_pred
        )
        
        return ensemble_pred
    
    @log_execution_time(logger)
    def rank_stocks(self, stock_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Rank stocks based on the model predictions.
        
        Args:
            stock_data: Dictionary of stock DataFrames with features
            
        Returns:
            List of ranked stocks with scores
        """
        try:
            if not stock_data:
                logger.warning("Empty stock data for ranking")
                return []
            
            # If model is not trained, use simple ranking formula
            if not self.models:
                logger.warning("No models found, using simple ranking formula")
                return self._simple_ranking(stock_data)
            
            # Process each stock
            ranked_stocks = []
            
            for symbol, df in stock_data.items():
                try:
                    # Extract features
                    features = self._extract_features(df)
                    
                    if features is None:
                        continue
                    
                    # Make a copy to avoid modifying the original
                    features_dict = features.copy()
                    
                    # Get features used in the model
                    model_features = {}
                    for feature in self.feature_names:
                        model_features[feature] = features_dict.get(feature, 0.0)
                    
                    # Convert to array
                    features_array = np.array([list(model_features.values())])
                    
                    # Scale features
                    features_scaled = self.scaler.transform(features_array)
                    
                    # Get predictions from each model
                    scores = {}
                    
                    # Get XGBoost prediction
                    if 'xgboost' in self.models and self.models['xgboost'] is not None:
                        dfeatures = xgb.DMatrix(features_scaled, feature_names=self.feature_names)
                        scores['xgboost'] = float(self.models['xgboost'].predict(dfeatures)[0])
                    
                    # Get LightGBM prediction
                    if 'lightgbm' in self.models and self.models['lightgbm'] is not None:
                        scores['lightgbm'] = float(self.models['lightgbm'].predict(features_scaled)[0])
                    
                    # Get CatBoost prediction
                    if 'catboost' in self.models and self.models['catboost'] is not None:
                        scores['catboost'] = float(self.models['catboost'].predict(
                            features_scaled, prediction_type='Probability'
                        )[0, 1])
                    
                    # Get Random Forest prediction
                    if 'randomforest' in self.models and self.models['randomforest'] is not None:
                        scores['randomforest'] = float(self.models['randomforest'].predict_proba(features_scaled)[0, 1])
                    
                    # Calculate ensemble score
                    if self.model_version == 'ensemble' and len(scores) > 1:
                        weights = self.hyperparams['meta']['weights']
                        ensemble_score = 0.0
                        total_weight = 0.0
                        
                        for model_name, score in scores.items():
                            if model_name in weights:
                                ensemble_score += score * weights[model_name]
                                total_weight += weights[model_name]
                        
                        if total_weight > 0:
                            score = ensemble_score / total_weight
                        else:
                            # If no weights, use average
                            score = sum(scores.values()) / len(scores)
                    else:
                        # Use selected model or fallback to first available
                        if self.model_version in scores:
                            score = scores[self.model_version]
                        else:
                            # Use the first available model
                            score = next(iter(scores.values())) if scores else 0.0
                    
                    # Get price information
                    current_price = df['close'].iloc[-1] if 'close' in df.columns else 0
                    
                    # Add to ranked list
                    ranked_stocks.append({
                        'symbol': symbol,
                        'score': score,
                        'model_scores': scores,
                        'price': current_price,
                        'features': features,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Error ranking stock {symbol}: {e}")
                    continue
            
            # Sort by score (descending)
            ranked_stocks.sort(key=lambda x: x['score'], reverse=True)
            
            return ranked_stocks
        except Exception as e:
            logger.error(f"Error ranking stocks: {e}")
            return []
    
    def _simple_ranking(self, stock_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Simple ranking method when no model is available.
        
        Args:
            stock_data: Dictionary of stock DataFrames with features
            
        Returns:
            List of ranked stocks with scores
        """
        try:
            ranked_stocks = []
            
            for symbol, df in stock_data.items():
                try:
                    # Ensure minimum data requirements
                    if len(df) < 20:
                        continue
                    
                    # Extract base metrics
                    if 'close' not in df.columns or 'volume' not in df.columns:
                        continue
                    
                    current_price = df['close'].iloc[-1]
                    previous_price = df['close'].iloc[-2]
                    price_change = current_price - previous_price
                    price_change_pct = (price_change / previous_price) * 100
                    
                    # Calculate daily range
                    if 'high' in df.columns and 'low' in df.columns:
                        daily_high = df['high'].iloc[-1]
                        daily_low = df['low'].iloc[-1]
                        daily_range_pct = ((daily_high - daily_low) / daily_low) * 100
                    else:
                        daily_range_pct = 0
                    
                    # Volume metrics
                    current_volume = df['volume'].iloc[-1]
                    avg_volume = df['volume'].iloc[-20:].mean()
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                    
                    # Calculate volatility
                    volatility = df['close'].pct_change().iloc[-20:].std() * 100
                    
                    # Calculate momentum
                    if len(df) >= 6:
                        momentum_5d = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100
                    else:
                        momentum_5d = 0
                    
                    # Calculate moving averages
                    ma5 = df['close'].rolling(5).mean().iloc[-1] if len(df) >= 5 else df['close'].iloc[-1]
                    ma20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['close'].iloc[-1]
                    
                    # MA crossover
                    ma_crossover = 1 if ma5 > ma20 else 0
                    
                    # RSI
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
                    
                    # Weight factors according to the weight configuration
                    # Momentum factors
                    momentum_score = (
                        abs(price_change_pct) * 0.3 + 
                        momentum_5d * 0.7
                    ) * self.factor_weights.get('momentum', 0.3)
                    
                    # Adjust score based on direction (prefer upward movement)
                    if price_change_pct > 0:
                        momentum_score *= 1.2
                    
                    # Volatility factors
                    volatility_score = (
                        volatility * 0.5 + 
                        daily_range_pct * 0.5
                    ) * self.factor_weights.get('volatility', 0.2)
                    
                    # Volume factors
                    volume_score = (
                        min(volume_ratio, 5) * 1.0
                    ) * self.factor_weights.get('volume', 0.2)
                    
                    # Trend factors
                    trend_score = (
                        (ma_crossover * 0.7) + 
                        ((current_price / ma20 - 1) * 0.3)
                    ) * self.factor_weights.get('trend', 0.2)
                    
                    # Value factors (using RSI as a simple proxy - lower RSI might indicate value)
                    value_score = (
                        (70 - min(max(current_rsi, 30), 70)) / 40
                    ) * self.factor_weights.get('value', 0.1)
                    
                    # Combine scores with weights
                    total_score = momentum_score + volatility_score + volume_score + trend_score + value_score
                    
                    # Extract features for transparency
                    features = {
                        'price_change_pct': float(price_change_pct),
                        'daily_range_pct': float(daily_range_pct),
                        'volume_ratio': float(volume_ratio),
                        'volatility': float(volatility),
                        'momentum_5d': float(momentum_5d),
                        'rsi': float(current_rsi),
                        'ma_crossover': float(ma_crossover),
                        'close_to_ma20': float(current_price / ma20) if ma20 > 0 else 1.0
                    }
                    
                    # Add to ranked list
                    ranked_stocks.append({
                        'symbol': symbol,
                        'score': float(total_score),
                        'price': float(current_price),
                        'features': features,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Error in simple ranking for stock {symbol}: {e}")
                    continue
            
            # Sort by score (descending)
            ranked_stocks.sort(key=lambda x: x['score'], reverse=True)
            
            return ranked_stocks
        except Exception as e:
            logger.error(f"Error in simple ranking: {e}")
            return []
    
    def _extract_features(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Extract features from stock data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of features
        """
        try:
            # Check minimum data requirements
            if df.empty or len(df) < 20:
                return None
            
            features = {}
            
            # Close price features
            if 'close' in df.columns:
                # Calculate price changes
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                
                # Short-term momentum
                features['return_1d'] = df['returns'].iloc[-1]
                features['return_5d'] = df['close'].iloc[-1] / df['close'].iloc[-6] - 1 if len(df) >= 6 else 0
                features['return_10d'] = df['close'].iloc[-1] / df['close'].iloc[-11] - 1 if len(df) >= 11 else 0
                
                # Moving averages
                df['ma5'] = df['close'].rolling(5).mean()
                df['ma10'] = df['close'].rolling(10).mean()
                df['ma20'] = df['close'].rolling(20).mean()
                
                # MA ratios
                features['close_ma5_ratio'] = df['close'].iloc[-1] / df['ma5'].iloc[-1] if df['ma5'].iloc[-1] > 0 else 0
                features['close_ma10_ratio'] = df['close'].iloc[-1] / df['ma10'].iloc[-1] if df['ma10'].iloc[-1] > 0 else 0
                features['close_ma20_ratio'] = df['close'].iloc[-1] / df['ma20'].iloc[-1] if df['ma20'].iloc[-1] > 0 else 0
                features['ma5_ma20_ratio'] = df['ma5'].iloc[-1] / df['ma20'].iloc[-1] if df['ma20'].iloc[-1] > 0 else 0
                
                # Volatility
                features['volatility_5d'] = df['returns'].iloc[-5:].std() if len(df) >= 5 else 0
                features['volatility_10d'] = df['returns'].iloc[-10:].std() if len(df) >= 10 else 0
                features['volatility_20d'] = df['returns'].iloc[-20:].std() if len(df) >= 20 else 0
                
                # Realized volatility - annualized
                features['realized_vol_annual'] = df['returns'].iloc[-20:].std() * np.sqrt(252) if len(df) >= 20 else 0
                
                # 52-week high/low (if we have enough data)
                if len(df) > 252:
                    high_52w = df['close'].iloc[-252:].max()
                    low_52w = df['close'].iloc[-252:].min()
                    features['pct_off_52w_high'] = (df['close'].iloc[-1] / high_52w) - 1
                    features['pct_off_52w_low'] = (df['close'].iloc[-1] / low_52w) - 1
                
                # Enhanced momentum factors
                # Rate of change
                for period in [1, 5, 10, 20]:
                    if len(df) > period:
                        features[f'roc_{period}d'] = df['close'].iloc[-1] / df['close'].iloc[-(period+1)] - 1
                
                # Acceleration - change in rate of change
                if len(df) > 5:
                    roc_1d_now = df['close'].iloc[-1] / df['close'].iloc[-2] - 1
                    roc_1d_prev = df['close'].iloc[-2] / df['close'].iloc[-3] - 1
                    features['momentum_acceleration'] = roc_1d_now - roc_1d_prev
            
            # Volume features
            if 'volume' in df.columns:
                # Volume ratios
                features['volume_1d'] = df['volume'].iloc[-1]
                features['volume_ma5'] = df['volume'].iloc[-5:].mean() if len(df) >= 5 else df['volume'].iloc[-1]
                features['volume_ma10'] = df['volume'].iloc[-10:].mean() if len(df) >= 10 else features['volume_ma5']
                
                features['volume_ratio_5d'] = df['volume'].iloc[-1] / features['volume_ma5'] if features['volume_ma5'] > 0 else 0
                features['volume_ratio_10d'] = df['volume'].iloc[-1] / features['volume_ma10'] if features['volume_ma10'] > 0 else 0
                
                # Volume trend
                volume_trend = df['volume'].iloc[-5:].pct_change().mean() if len(df) >= 5 else 0
                features['volume_trend_5d'] = volume_trend
                
                # Enhanced volume factors
                # On-balance volume (OBV)
                if 'close' in df.columns:
                    df['obv'] = 0
                    df['obv_signal'] = np.where(df['close'] > df['close'].shift(1), df['volume'], 
                                     np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
                    df['obv'] = df['obv_signal'].cumsum()
                    
                    # OBV momentum
                    if len(df) > 5:
                        features['obv_momentum'] = df['obv'].iloc[-1] / df['obv'].iloc[-6] - 1 if df['obv'].iloc[-6] != 0 else 0
                
                # Volume and price relationship
                if 'close' in df.columns and len(df) > 1:
                    # Correlation between volume and absolute price change
                    price_changes = df['close'].pct_change().abs().iloc[-20:]
                    volumes = df['volume'].iloc[-20:]
                    if len(price_changes) > 5 and len(volumes) > 5:
                        try:
                            correlation, _ = pearsonr(price_changes.dropna(), volumes.iloc[1:])
                            features['volume_price_change_correlation'] = correlation if not np.isnan(correlation) else 0
                        except:
                            features['volume_price_change_correlation'] = 0
            
            # Range and candle features
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # Daily range
                df['daily_range'] = (df['high'] - df['low']) / df['low'] * 100
                features['daily_range'] = df['daily_range'].iloc[-1]
                features['daily_range_avg_5d'] = df['daily_range'].iloc[-5:].mean() if len(df) >= 5 else features['daily_range']
                
                # Candle body size
                df['body_size'] = abs(df['close'] - df['open']) / df['open'] * 100
                features['body_size'] = df['body_size'].iloc[-1]
                features['body_size_avg_5d'] = df['body_size'].iloc[-5:].mean() if len(df) >= 5 else features['body_size']
                
                # Upper and lower shadows
                df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df[['open', 'close']].max(axis=1) * 100
                df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df[['open', 'close']].min(axis=1) * 100
                
                features['upper_shadow'] = df['upper_shadow'].iloc[-1]
                features['lower_shadow'] = df['lower_shadow'].iloc[-1]
                
                # Bullish/bearish
                features['is_bullish'] = 1.0 if df['close'].iloc[-1] > df['open'].iloc[-1] else 0.0
                
                # Gap
                if len(df) > 1:
                    prev_close = df['close'].iloc[-2]
                    curr_open = df['open'].iloc[-1]
                    features['gap_pct'] = (curr_open - prev_close) / prev_close * 100
                else:
                    features['gap_pct'] = 0.0
                
                # Enhanced candle pattern features
                # Doji detection
                body_to_range_ratio = features['body_size'] / features['daily_range'] if features['daily_range'] > 0 else 0
                features['is_doji'] = 1.0 if body_to_range_ratio < 0.1 else 0.0
                
                # Hammer/shooting star detection
                if features['lower_shadow'] > 2 * features['body_size'] and features['upper_shadow'] < features['body_size']:
                    features['is_hammer'] = 1.0
                else:
                    features['is_hammer'] = 0.0
                
                if features['upper_shadow'] > 2 * features['body_size'] and features['lower_shadow'] < features['body_size']:
                    features['is_shooting_star'] = 1.0
                else:
                    features['is_shooting_star'] = 0.0
                
                # Engulfing pattern detection
                if len(df) > 1:
                    prev_body_size = abs(df['close'].iloc[-2] - df['open'].iloc[-2])
                    curr_body_size = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
                    
                    # Bullish engulfing
                    if (df['close'].iloc[-1] > df['open'].iloc[-1] and  # Current candle is bullish
                        df['close'].iloc[-2] < df['open'].iloc[-2] and  # Previous candle is bearish
                        df['close'].iloc[-1] > df['open'].iloc[-2] and  # Current close > previous open
                        df['open'].iloc[-1] < df['close'].iloc[-2]):    # Current open < previous close
                        features['is_bullish_engulfing'] = 1.0
                    else:
                        features['is_bullish_engulfing'] = 0.0
                    
                    # Bearish engulfing
                    if (df['close'].iloc[-1] < df['open'].iloc[-1] and  # Current candle is bearish
                        df['close'].iloc[-2] > df['open'].iloc[-2] and  # Previous candle is bullish
                        df['close'].iloc[-1] < df['open'].iloc[-2] and  # Current close < previous open
                        df['open'].iloc[-1] > df['close'].iloc[-2]):    # Current open > previous close
                        features['is_bearish_engulfing'] = 1.0
                    else:
                        features['is_bearish_engulfing'] = 0.0
            
            # Technical indicators
            if 'close' in df.columns:
                # RSI (14)
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                features['rsi'] = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50.0
                
                # RSI trends
                if len(df) > 5:
                    features['rsi_5d_change'] = df['rsi'].iloc[-1] - df['rsi'].iloc[-6] if not pd.isna(df['rsi'].iloc[-6]) else 0
                
                # Overbought/oversold conditions
                features['is_overbought'] = 1.0 if features['rsi'] > 70 else 0.0
                features['is_oversold'] = 1.0 if features['rsi'] < 30 else 0.0
                
                # MACD
                if len(df) >= 26:
                    ema12 = df['close'].ewm(span=12, adjust=False).mean()
                    ema26 = df['close'].ewm(span=26, adjust=False).mean()
                    df['macd'] = ema12 - ema26
                    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']
                    
                    features['macd'] = df['macd'].iloc[-1]
                    features['macd_signal'] = df['macd_signal'].iloc[-1]
                    features['macd_hist'] = df['macd_hist'].iloc[-1]
                    
                    # MACD crossover
                    if len(df) > 1:
                        features['macd_crossover'] = 1.0 if (df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and 
                                                           df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]) else 0.0
                        features['macd_crossunder'] = 1.0 if (df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] and 
                                                            df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]) else 0.0
                
                # Bollinger Bands
                if len(df) >= 20:
                    df['bb_middle'] = df['close'].rolling(20).mean()
                    df['bb_std'] = df['close'].rolling(20).std()
                    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
                    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
                    
                    # Percent B indicator (position within the bands)
                    df['percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                    
                    features['bb_width'] = (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) / df['bb_middle'].iloc[-1]
                    features['percent_b'] = df['percent_b'].iloc[-1]
                    
                    # Band touches/crossovers
                    features['touched_upper_band'] = 1.0 if df['high'].iloc[-1] >= df['bb_upper'].iloc[-1] else 0.0
                    features['touched_lower_band'] = 1.0 if df['low'].iloc[-1] <= df['bb_lower'].iloc[-1] else 0.0
                
                # Commodity Channel Index (CCI)
                if 'high' in df.columns and 'low' in df.columns:
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    moving_avg = typical_price.rolling(window=20).mean()
                    mean_deviation = abs(typical_price - moving_avg).rolling(window=20).mean()
                    df['cci'] = (typical_price - moving_avg) / (0.015 * mean_deviation)
                    
                    features['cci'] = df['cci'].iloc[-1] if not pd.isna(df['cci'].iloc[-1]) else 0
                
                # Average Directional Index (ADX)
                if 'high' in df.columns and 'low' in df.columns and len(df) > 14:
                    high_diff = df['high'].diff()
                    low_diff = df['low'].diff()
                    
                    pos_dm = high_diff.copy()
                    pos_dm[pos_dm < 0] = 0
                    pos_dm[high_diff <= low_diff.abs()] = 0
                    
                    neg_dm = low_diff.abs().copy()
                    neg_dm[neg_dm < 0] = 0
                    neg_dm[low_diff.abs() <= high_diff] = 0
                    
                    tr1 = df['high'] - df['low']
                    tr2 = (df['high'] - df['close'].shift(1)).abs()
                    tr3 = (df['low'] - df['close'].shift(1)).abs()
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    
                    atr = tr.rolling(window=14).mean()
                    
                    pdi = 100 * (pos_dm.rolling(window=14).mean() / atr)
                    ndi = 100 * (neg_dm.rolling(window=14).mean() / atr)
                    
                    dx = 100 * ((pdi - ndi).abs() / (pdi + ndi).abs())
                    adx = dx.rolling(window=14).mean()
                    
                    features['adx'] = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
                    features['pdi'] = pdi.iloc[-1] if not pd.isna(pdi.iloc[-1]) else 0
                    features['ndi'] = ndi.iloc[-1] if not pd.isna(ndi.iloc[-1]) else 0
                    
                    # Trend strength and direction
                    features['trend_strength'] = features['adx'] / 100.0
                    features['trend_direction'] = 1.0 if features['pdi'] > features['ndi'] else -1.0
                
                # Stochastic oscillator
                if 'high' in df.columns and 'low' in df.columns and len(df) >= 14:
                    low_14 = df['low'].rolling(window=14).min()
                    high_14 = df['high'].rolling(window=14).max()
                    
                    # %K calculation
                    k_percent = 100 * ((df['close'] - low_14) / (high_14 - low_14))
                    
                    # %D calculation (3-day SMA of %K)
                    d_percent = k_percent.rolling(window=3).mean()
                    
                    features['stoch_k'] = k_percent.iloc[-1] if not pd.isna(k_percent.iloc[-1]) else 50
                    features['stoch_d'] = d_percent.iloc[-1] if not pd.isna(d_percent.iloc[-1]) else 50
                    
                    # Overbought/oversold signals
                    features['stoch_overbought'] = 1.0 if features['stoch_k'] > 80 else 0.0
                    features['stoch_oversold'] = 1.0 if features['stoch_k'] < 20 else 0.0
                    
                    # Crossover signals
                    if len(df) > 3:
                        features['stoch_crossover'] = 1.0 if (k_percent.iloc[-1] > d_percent.iloc[-1] and 
                                                           k_percent.iloc[-2] <= d_percent.iloc[-2]) else 0.0
                        features['stoch_crossunder'] = 1.0 if (k_percent.iloc[-1] < d_percent.iloc[-1] and 
                                                            k_percent.iloc[-2] >= d_percent.iloc[-2]) else 0.0
                
                # Ichimoku Cloud
                if len(df) >= 52:  # Need enough data for Senkou Span B
                    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
                    nine_period_high = df['high'].rolling(window=9).max()
                    nine_period_low = df['low'].rolling(window=9).min()
                    df['tenkan_sen'] = (nine_period_high + nine_period_low) / 2
                    
                    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
                    period26_high = df['high'].rolling(window=26).max()
                    period26_low = df['low'].rolling(window=26).min()
                    df['kijun_sen'] = (period26_high + period26_low) / 2
                    
                    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
                    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
                    
                    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
                    period52_high = df['high'].rolling(window=52).max()
                    period52_low = df['low'].rolling(window=52).min()
                    df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
                    
                    features['cloud_bullish'] = 1.0 if df['senkou_span_a'].iloc[-1] > df['senkou_span_b'].iloc[-1] else 0.0
                    features['price_above_cloud'] = 1.0 if df['close'].iloc[-1] > max(df['senkou_span_a'].iloc[-1], df['senkou_span_b'].iloc[-1]) else 0.0
                    features['price_below_cloud'] = 1.0 if df['close'].iloc[-1] < min(df['senkou_span_a'].iloc[-1], df['senkou_span_b'].iloc[-1]) else 0.0
                    features['tk_cross'] = 1.0 if (df['tenkan_sen'].iloc[-1] > df['kijun_sen'].iloc[-1] and 
                                               df['tenkan_sen'].iloc[-2] <= df['kijun_sen'].iloc[-2]) else 0.0
                
                # Market regime features
                if len(df) >= 50:
                    # Moving Average trends
                    ma_50 = df['close'].rolling(window=50).mean()
                    ma_200 = df['close'].rolling(window=200).mean() if len(df) >= 200 else None
                    
                    features['ma_50_trend'] = df['close'].iloc[-1] / ma_50.iloc[-1] - 1
                    
                    if ma_200 is not None:
                        features['ma_200_trend'] = df['close'].iloc[-1] / ma_200.iloc[-1] - 1
                        features['golden_cross'] = 1.0 if (ma_50.iloc[-1] > ma_200.iloc[-1] and 
                                                        ma_50.iloc[-2] <= ma_200.iloc[-2]) else 0.0
                        features['death_cross'] = 1.0 if (ma_50.iloc[-1] < ma_200.iloc[-1] and 
                                                       ma_50.iloc[-2] >= ma_200.iloc[-2]) else 0.0
                
                # Volatility analysis
                if len(df) >= 20:
                    # Calculate historical volatility (20-day)
                    log_returns = np.log(df['close'] / df['close'].shift(1))
                    features['historical_volatility'] = log_returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
                    
                    # Volatility trend (increasing or decreasing)
                    if len(df) >= 40:
                        vol_20 = log_returns.rolling(window=20).std().iloc[-1]
                        vol_20_prev = log_returns.rolling(window=20).std().iloc[-20]
                        features['volatility_trend'] = vol_20 / vol_20_prev - 1 if vol_20_prev > 0 else 0
                
                # Mean reversion indicators
                if len(df) >= 20:
                    # Z-score relative to 20-day moving average
                    ma_20 = df['close'].rolling(window=20).mean()
                    std_20 = df['close'].rolling(window=20).std()
                    features['zscore_20d'] = (df['close'].iloc[-1] - ma_20.iloc[-1]) / std_20.iloc[-1] if std_20.iloc[-1] > 0 else 0
                    
                    # Distance from 20-day Bollinger Bands
                    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                        upper_band_distance = (df['bb_upper'].iloc[-1] - df['close'].iloc[-1]) / df['close'].iloc[-1]
                        lower_band_distance = (df['close'].iloc[-1] - df['bb_lower'].iloc[-1]) / df['close'].iloc[-1]
                        features['upper_band_distance'] = upper_band_distance
                        features['lower_band_distance'] = lower_band_distance
                
                # Market breadth features (dummy values for single stock - would use market data in real implementation)
                features['market_breadth'] = 0.5  # Placeholder
                
                # Support and resistance levels
                if 'high' in df.columns and 'low' in df.columns and len(df) >= 20:
                    # Find recent highs and lows
                    recent_highs = df['high'].rolling(window=5, center=True).max()
                    recent_lows = df['low'].rolling(window=5, center=True).min()
                    
                    # Find potential support/resistance levels
                    support_levels = []
                    resistance_levels = []
                    
                    for i in range(5, len(df)-5):
                        # Potential resistance: local high with lower highs on both sides
                        if recent_highs.iloc[i] == df['high'].iloc[i] and \
                           recent_highs.iloc[i] > recent_highs.iloc[i-5:i].max() and \
                           recent_highs.iloc[i] > recent_highs.iloc[i+1:i+6].max():
                            resistance_levels.append(df['high'].iloc[i])
                        
                        # Potential support: local low with higher lows on both sides
                        if recent_lows.iloc[i] == df['low'].iloc[i] and \
                           recent_lows.iloc[i] < recent_lows.iloc[i-5:i].min() and \
                           recent_lows.iloc[i] < recent_lows.iloc[i+1:i+6].min():
                            support_levels.append(df['low'].iloc[i])
                    
                    # Calculate distance to nearest support/resistance
                    if resistance_levels:
                        nearest_resistance = min([r for r in resistance_levels if r > df['close'].iloc[-1]], default=0)
                        if nearest_resistance > 0:
                            features['distance_to_resistance'] = (nearest_resistance - df['close'].iloc[-1]) / df['close'].iloc[-1]
                    
                    if support_levels:
                        nearest_support = max([s for s in support_levels if s < df['close'].iloc[-1]], default=0)
                        if nearest_support > 0:
                            features['distance_to_support'] = (df['close'].iloc[-1] - nearest_support) / df['close'].iloc[-1]
                
                # Seasonality features
                if isinstance(df.index, pd.DatetimeIndex):
                    features['day_of_week'] = df.index[-1].dayofweek / 4.0  # Normalize to [0,1]
                    features['day_of_month'] = (df.index[-1].day - 1) / 30.0  # Normalize to [0,1]
                    features['month_of_year'] = (df.index[-1].month - 1) / 11.0  # Normalize to [0,1]
                    
                    # End of month effect
                    last_day = pd.Timestamp(df.index[-1].year, df.index[-1].month, 1) + pd.offsets.MonthEnd(1)
                    days_to_month_end = (last_day - df.index[-1]).days
                    features['end_of_month'] = 1.0 if days_to_month_end <= 3 else 0.0
                
                # Pattern detection (basic)
                if len(df) >= 5:
                    # Detect breakout pattern
                    if 'high' in df.columns:
                        max_high_4d = df['high'].iloc[-5:-1].max()
                        features['breakout'] = 1.0 if df['close'].iloc[-1] > max_high_4d * 1.02 else 0.0
                    
                    # Detect breakdown pattern
                    if 'low' in df.columns:
                        min_low_4d = df['low'].iloc[-5:-1].min()
                        features['breakdown'] = 1.0 if df['close'].iloc[-1] < min_low_4d * 0.98 else 0.0
                    
                    # Detect consolidation pattern
                    if 'high' in df.columns and 'low' in df.columns:
                        high_range = df['high'].iloc[-5:].max() - df['high'].iloc[-5:].min()
                        low_range = df['low'].iloc[-5:].max() - df['low'].iloc[-5:].min()
                        avg_price = df['close'].iloc[-5:].mean()
                        
                        # Check if the price range is small relative to average price
                        range_pct = max(high_range, abs(low_range)) / avg_price
                        features['consolidation'] = 1.0 if range_pct < 0.03 else 0.0
                    
                    # Detect trend reversal
                    price_direction_old = np.sign(df['close'].iloc[-3] - df['close'].iloc[-5])
                    price_direction_new = np.sign(df['close'].iloc[-1] - df['close'].iloc[-3])
                    features['reversal'] = 1.0 if price_direction_old != price_direction_new and price_direction_old != 0 else 0.0
                
                # Money Flow Index (MFI)
                if 'high' in df.columns and 'low' in df.columns and 'volume' in df.columns and len(df) >= 14:
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    raw_money_flow = typical_price * df['volume']
                    
                    positive_flow = pd.Series(0, index=df.index)
                    negative_flow = pd.Series(0, index=df.index)
                    
                    # Calculate positive and negative money flow
                    for i in range(1, len(df)):
                        if typical_price.iloc[i] > typical_price.iloc[i-1]:
                            positive_flow.iloc[i] = raw_money_flow.iloc[i]
                        else:
                            negative_flow.iloc[i] = raw_money_flow.iloc[i]
                    
                    # Calculate money flow ratio and MFI
                    positive_mf_14 = positive_flow.rolling(window=14).sum()
                    negative_mf_14 = negative_flow.rolling(window=14).sum()
                    
                    money_flow_ratio = positive_mf_14 / negative_mf_14
                    mfi = 100 - (100 / (1 + money_flow_ratio))
                    
                    features['mfi'] = mfi.iloc[-1] if not pd.isna(mfi.iloc[-1]) else 50
                    features['mfi_overbought'] = 1.0 if features['mfi'] > 80 else 0.0
                    features['mfi_oversold'] = 1.0 if features['mfi'] < 20 else 0.0
                
            # Remove any NaN values
            features = {k: float(v) if not pd.isna(v) else 0.0 for k, v in features.items()}
            
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

# Create global instance
ranking_model = RankingModel()
