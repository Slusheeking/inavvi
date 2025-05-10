"""
Multi-factor ranking model for stock screening and selection.

This model ranks stocks based on technical, volume, and momentum features
using an ensemble of XGBoost, LightGBM, and CatBoost.
"""
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import dump, load
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import standard ML libraries (using GPU via standard libraries)
import catboost as cb
import lightgbm as lgb
import optuna
import xgboost as xgb
from src.config.settings import settings
from src.utils.logging import setup_logger, log_execution_time
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("ranking_model")


class RankingModel:
    """
    Multi-factor ranking model for stock selection.

    Uses an ensemble of XGBoost, LightGBM, and CatBoost with time-aware validation
    and feature importance tracking.
    """

    def __init__(self, factor_weights: Optional[Dict[str, float]] = None, model_version: str = "ensemble"):
        """
        Initialize the ranking model.

        Args:
            factor_weights: Optional dictionary of factor weights.
            model_version: Model version to use ('ensemble', 'xgboost', 'lightgbm', 'catboost', 'rf').
        """
        self.models: Dict[str, Any] = {}
        self.feature_names: List[str] = [
            "open", "high", "low", "close", "volume",
            "return_1d", "return_5d", "return_10d",
            "close_ma5_ratio", "close_ma10_ratio", "close_ma20_ratio",
            "ma5_ma20_ratio", "volatility_5d", "volatility_10d", "volatility_20d",
            "volume_1d", "volume_ma5", "volume_ma10",
            "volume_ratio_5d", "volume_ratio_10d", "volume_trend_5d",
            "daily_range", "daily_range_avg_5d", "body_size", "body_size_avg_5d",
            "upper_shadow", "lower_shadow", "is_bullish", "gap_pct",
            "rsi", "bb_width", "percent_b", "macd", "macd_signal", "macd_hist"
        ]
        self.model_version = model_version
        self.model_dir = os.path.join(settings.models_dir, "ranking")
        self.model_path = getattr(settings.model, "ranking_model_path", os.path.join(self.model_dir, "model.pt"))
        self.scaler = StandardScaler()
        self.feature_importance: Dict[str, Dict] = {}
        self.feature_correlations: Dict[str, Dict] = {}
        self.model_metrics: Dict[str, Dict] = {}
        self.versions_history: List[Dict] = []
        self.last_trained: Optional[str] = None
        self.factor_weights = factor_weights or getattr(
            settings.model,
            "factor_weights",
            {
                "momentum": 0.3,
                "volume": 0.2,
                "volatility": 0.2,
                "trend": 0.2,
                "value": 0.1,
            },
        )
        self.hyperparams = {
            "xgboost": {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "tree_method": "hist",
                "device": "cpu",
                "random_state": 42,
            },
            "lightgbm": {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbosity": -1,
                "random_state": 42,
            },
            "catboost": {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "iterations": 100,
                "depth": 6,
                "learning_rate": 0.05,
                "random_seed": 42,
                "verbose": 0,
                "allow_writing_files": False,
            },
            "randomforest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
            "meta": {
                "method": "weighted_average",
                "weights": {"xgboost": 0.4, "lightgbm": 0.3, "catboost": 0.3},
            },
        }
        os.makedirs(self.model_dir, exist_ok=True)
        self._load_model()

    def _load_model(self) -> bool:
        """
        Load model from disk if it exists.

        Returns:
            True if model was loaded successfully, False otherwise.
        """
        try:
            ensemble_path = os.path.join(self.model_dir, "ensemble")
            os.makedirs(ensemble_path, exist_ok=True)
            metadata_path = os.path.join(ensemble_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                self.feature_names = metadata.get("feature_names", [])
                self.feature_importance = metadata.get("feature_importance", {})
                self.feature_correlations = metadata.get("feature_correlations", {})
                self.model_metrics = metadata.get("model_metrics", {})
                self.last_trained = metadata.get("last_trained")
                self.versions_history = metadata.get("versions_history", [])
                self.hyperparams = metadata.get("hyperparams", self.hyperparams)
                scaler_path = os.path.join(ensemble_path, "scaler.joblib")
                if os.path.exists(scaler_path):
                    self.scaler = load(scaler_path)
                model_loaded = False
                xgb_path = os.path.join(ensemble_path, "xgboost.model")
                if os.path.exists(xgb_path):
                    try:
                        self.models["xgboost"] = xgb.Booster()
                        self.models["xgboost"].load_model(xgb_path)
                        model_loaded = True
                    except Exception as e:
                        logger.error(f"Error loading XGBoost model: {e}")
                lgb_path = os.path.join(ensemble_path, "lightgbm.txt")
                if os.path.exists(lgb_path):
                    try:
                        self.models["lightgbm"] = lgb.Booster(model_file=lgb_path)
                        model_loaded = True
                    except Exception as e:
                        logger.error(f"Error loading LightGBM model: {e}")
                cb_path = os.path.join(ensemble_path, "catboost.cbm")
                if os.path.exists(cb_path):
                    try:
                        self.models["catboost"] = cb.CatBoost()
                        self.models["catboost"].load_model(cb_path)
                        model_loaded = True
                    except Exception as e:
                        logger.error(f"Error loading CatBoost model: {e}")
                rf_path = os.path.join(ensemble_path, "randomforest.joblib")
                if os.path.exists(rf_path):
                    try:
                        self.models["randomforest"] = load(rf_path)
                        model_loaded = True
                    except Exception as e:
                        logger.error(f"Error loading Random Forest model: {e}")
                if model_loaded:
                    logger.info(f"Models loaded from {ensemble_path}")
                    self._log_model_info()
                    return True
            if os.path.exists(self.model_path):
                try:
                    self.models["xgboost"] = xgb.Booster()
                    self.models["xgboost"].load_model(self.model_path)
                    legacy_metadata_path = Path(self.model_path).with_suffix(".json")
                    if legacy_metadata_path.exists():
                        with open(legacy_metadata_path, "r") as f:
                            metadata = json.load(f)
                        self.feature_names = metadata.get("feature_names", [])
                        self.feature_importance = metadata.get("feature_importance", {})
                        self.last_trained = metadata.get("last_trained")
                    logger.info(f"Legacy model loaded from {self.model_path}")
                    self._log_model_info()
                    self.save_model()
                    return True
                except Exception as e:
                    logger.error(f"Error loading legacy model: {e}")
            logger.warning(f"No models found at {ensemble_path} or {self.model_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.models = {}
            return False

    def _log_model_info(self):
        """Log information about the loaded model."""
        logger.info("Model information:")
        logger.info(f"  Last trained: {self.last_trained or 'Never'}")
        logger.info(f"  Models loaded: {list(self.models.keys())}")
        logger.info(f"  Number of features: {len(self.feature_names)}")
        if self.model_metrics:
            logger.info("  Model metrics:")
            for model_name, metrics in self.model_metrics.items():
                if isinstance(metrics, dict):
                    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                    logger.info(f"    {model_name}: {metrics_str}")

    def predict(self, data: pd.DataFrame) -> float:
        """
        Predict the score for a given dataset.

        Args:
            data: DataFrame containing the features.

        Returns:
            Predicted score.
        """
        if not self.models:
            logger.warning("No models loaded, returning default score of 0.0")
            return 0.0

        features = self.scaler.transform(data[self.feature_names])
        ensemble_pred = self._ensemble_predict(features)
        return float(ensemble_pred[0])

    def score_multiple(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Score multiple stocks.

        Args:
            stock_data: Dictionary of stock DataFrames with features.

        Returns:
            Dictionary of stock scores.
        """
        scores = {}
        for symbol, data in stock_data.items():
            scores[symbol] = self.predict(data)
        return scores

    def save_model(self) -> bool:
        """
        Save model to disk.
        """
        try:
            if not self.models:
                logger.warning("No models to save")
                return False
            ensemble_path = os.path.join(self.model_dir, "ensemble")
            os.makedirs(ensemble_path, exist_ok=True)
            for model_name, model in self.models.items():
                if model_name == "xgboost" and model is not None:
                    model.save_model(os.path.join(ensemble_path, "xgboost.model"))
                elif model_name == "lightgbm" and model is not None:
                    model.save_model(os.path.join(ensemble_path, "lightgbm.txt"))
                elif model_name == "catboost" and model is not None:
                    model.save_model(os.path.join(ensemble_path, "catboost.cbm"))
                elif model_name == "randomforest" and model is not None:
                    dump(model, os.path.join(ensemble_path, "randomforest.joblib"))
            dump(self.scaler, os.path.join(ensemble_path, "scaler.joblib"))
            if self.last_trained:
                version_info = {
                    "version": len(self.versions_history) + 1,
                    "timestamp": self.last_trained,
                    "models": list(self.models.keys()),
                    "metrics": self.model_metrics,
                }
                self.versions_history.append(version_info)
            metadata = {
                "feature_names": self.feature_names,
                "feature_importance": self.feature_importance,
                "feature_correlations": self.feature_correlations,
                "model_metrics": self.model_metrics,
                "last_trained": datetime.now().isoformat(),
                "versions_history": self.versions_history,
                "hyperparams": self.hyperparams,
            }
            with open(os.path.join(ensemble_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            if "xgboost" in self.models and self.models["xgboost"] is not None:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.models["xgboost"].save_model(self.model_path)
                legacy_metadata_path = Path(self.model_path).with_suffix(".json")
                if os.path.exists(legacy_metadata_path):
                    with open(legacy_metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)
            logger.info(f"Models saved to {ensemble_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str = "xgboost",
        n_trials: int = 30,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            model_name: Model to optimize ('xgboost', 'lightgbm', 'catboost').
            n_trials: Number of optimization trials.

        Returns:
            Best hyperparameters.
        """
        def objective(trial):
            if model_name == "xgboost":
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "tree_method": "hist",
                    "device": "cpu",
                    "random_state": 42,
                }
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)
                model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, "val")], early_stopping_rounds=10, verbose_eval=False)
                y_pred = model.predict(dval)
            elif model_name == "lightgbm":
                params = {
                    "objective": "binary",
                    "metric": "auc",
                    "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
                    "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "verbosity": -1,
                    "random_state": 42,
                }
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
                )
                y_pred = model.predict(X_val)
            elif model_name == "catboost":
                params = {
                    "loss_function": "Logloss",
                    "eval_metric": "AUC",
                    "iterations": 100,
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                    "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
                    "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
                    "verbose": 0,
                    "random_seed": 42,
                    "allow_writing_files": False,
                }
                train_data = cb.Pool(X_train, label=y_train)
                val_data = cb.Pool(X_val, label=y_val)
                model = cb.CatBoost(params)
                model.fit(train_data, eval_set=val_data, early_stopping_rounds=10, verbose=False)
                y_pred = model.predict(X_val, prediction_type="Probability")[:, 1]
            try:
                auc = roc_auc_score(y_val, y_pred)
                return auc
            except:
                return 0.5

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        self.hyperparams[model_name].update(best_params)
        logger.info(f"Best {model_name} parameters: {best_params}")
        logger.info(f"Best AUC: {study.best_value:.4f}")
        return best_params
