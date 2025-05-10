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

# Import RAPIDS libraries if available
try:
    import cudf
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False

import catboost as cb
import lightgbm as lgb
import optuna
import xgboost as xgb
from src.config.settings import settings
from src.utils.logging import setup_logger, log_execution_time
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("ranking_model")

# Enable RAPIDS acceleration for pandas operations where possible
if RAPIDS_AVAILABLE:
    try:
        cudf.pandas_accelerator()
        logger.info("RAPIDS pandas accelerator enabled for ranking_model")
    except Exception as e:
        logger.warning(f"Could not enable RAPIDS pandas accelerator: {e}")


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

        """
        Save model to disk.

        Returns:
            True if model was saved successfully, False otherwise.
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

    @log_execution_time(logger)
    def train(
        self,
        training_data: pd.DataFrame,
        target_col: str = "target",
        optimize_hyperparams: bool = False,
        n_trials: int = 30,
        use_time_series_cv: bool = True,
        test_size: float = 0.2,
        use_gpu: bool = True,
    ) -> bool:
        """
        Train the ranking model ensemble.

        Args:
            training_data: DataFrame with features and target.
            target_col: Name of the target column.
            optimize_hyperparams: Whether to optimize hyperparameters.
            n_trials: Number of hyperparameter optimization trials.
            use_time_series_cv: Whether to use time-series cross-validation.
            test_size: Test size for validation.

        Returns:
            True if training was successful, False otherwise.
        """
        try:
            if training_data.empty:
                logger.error("Empty training data")
                return False
            if target_col not in training_data.columns:
                logger.error(f"Target column '{target_col}' not found in training data")
                return False
                
            logger.info(f"Training ranking model on {len(training_data)} samples")
            non_feature_cols = [target_col]
            if "symbol" in training_data.columns:
                non_feature_cols.append("symbol")
            if "date" in training_data.columns:
                non_feature_cols.append("date")
                
            # Check if we can use GPU acceleration
            use_rapids = use_gpu and RAPIDS_AVAILABLE
            if use_rapids:
                logger.info("Using RAPIDS GPU acceleration for model training")
                return self._train_with_rapids(
                    training_data, 
                    non_feature_cols, 
                    target_col, 
                    optimize_hyperparams, 
                    n_trials, 
                    use_time_series_cv, 
                    test_size
                )
            else:
                logger.info("Using CPU for model training")
                return self._train_with_cpu(
                    training_data, 
                    non_feature_cols, 
                    target_col, 
                    optimize_hyperparams, 
                    n_trials, 
                    use_time_series_cv, 
                    test_size
                )
                
    def _train_with_rapids(
        self,
        training_data: pd.DataFrame,
        non_feature_cols: List[str],
        target_col: str,
        optimize_hyperparams: bool,
        n_trials: int,
        use_time_series_cv: bool,
        test_size: float,
    ) -> bool:
        """Train the model using RAPIDS GPU acceleration."""
        try:
            # Convert to cuDF for GPU acceleration
        except Exception as e:
            logger.error(f"Error training model with RAPIDS: {e}")
            return False
            X = training_data.drop(columns=non_feature_cols)
            y = training_data[target_col]
            self.feature_names = X.columns.tolist()
            
            # Use cuML's StandardScaler
            gpu_scaler = cuStandardScaler()
            X_gpu = cudf.DataFrame.from_pandas(X)
            y_gpu = cudf.Series(y.values)
            
            X_scaled_gpu = gpu_scaler.fit_transform(X_gpu)
            
            # Save the CPU version of the scaler for inference
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            
            # Split data
            if use_time_series_cv and "date" in training_data.columns:
                sorted_indices = training_data["date"].argsort()
                X_sorted = X_scaled_gpu.iloc[sorted_indices]
                y_sorted = y_gpu.iloc[sorted_indices]
                val_size = int(len(X_sorted) * test_size)
                train_size = len(X_sorted) - val_size
                X_train = X_sorted[:train_size]
                y_train = y_sorted[:train_size]
                X_val = X_sorted[train_size:]
                y_val = y_sorted[train_size:]
                logger.info(f"Using time-series split: {train_size} train, {val_size} validation")
            else:
                # Use cuML's train_test_split
                from cuml.model_selection import train_test_split as cu_train_test_split
                X_train, X_val, y_train, y_val = cu_train_test_split(
                    X_scaled_gpu, y_gpu, test_size=test_size, random_state=42
                )
                logger.info(f"Using random split: {len(X_train)} train, {len(X_val)} validation")
            
            self.model_metrics = {}
            
            # Train XGBoost with GPU acceleration
            logger.info("Training XGBoost model with GPU acceleration...")
            if optimize_hyperparams:
                logger.info("Optimizing XGBoost hyperparameters...")
                # Update hyperparams to use GPU
                self.hyperparams["xgboost"]["tree_method"] = "gpu_hist"
                self.hyperparams["xgboost"]["device"] = "cuda"
                self.optimize_hyperparameters(
                    X_train.to_pandas().values, 
                    y_train.to_pandas().values, 
                    X_val.to_pandas().values, 
                    y_val.to_pandas().values, 
                    "xgboost", 
                    n_trials
                )
            
            # Convert to DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train.to_pandas(), label=y_train.to_pandas(), feature_names=self.feature_names)
            dval = xgb.DMatrix(X_val.to_pandas(), label=y_val.to_pandas(), feature_names=self.feature_names)
            
            # Update params for GPU
            xgb_params = self.hyperparams["xgboost"].copy()
            xgb_params["tree_method"] = "gpu_hist"
            xgb_params["device"] = "cuda"
            
            self.models["xgboost"] = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=10,
                verbose_eval=False,
            )
            
            xgb_importance = self.models["xgboost"].get_score(importance_type="gain")
            total = sum(xgb_importance.values()) or 1.0
            self.feature_importance["xgboost"] = {k: v / total for k, v in xgb_importance.items()}
            
            xgb_pred = self.models["xgboost"].predict(dval)
            xgb_pred_binary = (xgb_pred > 0.5).astype(int)
            y_val_np = y_val.to_pandas().values
            
            self.model_metrics["xgboost"] = {
                "accuracy": accuracy_score(y_val_np, xgb_pred_binary),
                "precision": precision_score(y_val_np, xgb_pred_binary, zero_division=0),
                "recall": recall_score(y_val_np, xgb_pred_binary, zero_division=0),
                "f1": f1_score(y_val_np, xgb_pred_binary, zero_division=0),
                "auc": roc_auc_score(y_val_np, xgb_pred),
            }
            
            # Train LightGBM with GPU acceleration
            logger.info("Training LightGBM model with GPU acceleration...")
            if optimize_hyperparams:
                logger.info("Optimizing LightGBM hyperparameters...")
                # Update hyperparams to use GPU
                self.hyperparams["lightgbm"]["device"] = "gpu"
                self.optimize_hyperparameters(
                    X_train.to_pandas().values, 
                    y_train.to_pandas().values, 
                    X_val.to_pandas().values, 
                    y_val.to_pandas().values, 
                    "lightgbm", 
                    n_trials
                )
            
            # Convert to Dataset for LightGBM
            train_data = lgb.Dataset(X_train.to_pandas(), label=y_train.to_pandas(), feature_name=self.feature_names)
            val_data = lgb.Dataset(X_val.to_pandas(), label=y_val.to_pandas(), reference=train_data)
            
            # Update params for GPU
            lgb_params = self.hyperparams["lightgbm"].copy()
            lgb_params["device"] = "gpu"
            
            self.models["lightgbm"] = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data, val_data],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
            )
            
            lgb_importance = dict(zip(self.feature_names, self.models["lightgbm"].feature_importance(importance_type="gain")))
            total = sum(lgb_importance.values()) or 1.0
            self.feature_importance["lightgbm"] = {k: v / total for k, v in lgb_importance.items()}
            
            lgb_pred = self.models["lightgbm"].predict(X_val.to_pandas())
            lgb_pred_binary = (lgb_pred > 0.5).astype(int)
            
            self.model_metrics["lightgbm"] = {
                "accuracy": accuracy_score(y_val_np, lgb_pred_binary),
                "precision": precision_score(y_val_np, lgb_pred_binary, zero_division=0),
                "recall": recall_score(y_val_np, lgb_pred_binary, zero_division=0),
                "f1": f1_score(y_val_np, lgb_pred_binary, zero_division=0),
                "auc": roc_auc_score(y_val_np, lgb_pred),
            }
            
            # Train CatBoost with GPU acceleration
            logger.info("Training CatBoost model with GPU acceleration...")
            if optimize_hyperparams:
                logger.info("Optimizing CatBoost hyperparameters...")
                # Update hyperparams to use GPU
                self.hyperparams["catboost"]["task_type"] = "GPU"
                self.optimize_hyperparameters(
                    X_train.to_pandas().values, 
                    y_train.to_pandas().values, 
                    X_val.to_pandas().values, 
                    y_val.to_pandas().values, 
                    "catboost", 
                    n_trials
                )
            
            # Convert to Pool for CatBoost
            train_data = cb.Pool(X_train.to_pandas(), label=y_train.to_pandas())
            val_data = cb.Pool(X_val.to_pandas(), label=y_val.to_pandas())
            
            # Update params for GPU
            cb_params = self.hyperparams["catboost"].copy()
            cb_params["task_type"] = "GPU"
            
            self.models["catboost"] = cb.CatBoost(cb_params)
            self.models["catboost"].fit(train_data, eval_set=val_data, early_stopping_rounds=10, verbose=False)
            
            cb_importance = dict(zip(self.feature_names, self.models["catboost"].get_feature_importance()))
            total = sum(cb_importance.values()) or 1.0
            self.feature_importance["catboost"] = {k: v / total for k, v in cb_importance.items()}
            
            cb_pred = self.models["catboost"].predict(X_val.to_pandas(), prediction_type="Probability")[:, 1]
            cb_pred_binary = (cb_pred > 0.5).astype(int)
            
            self.model_metrics["catboost"] = {
                "accuracy": accuracy_score(y_val_np, cb_pred_binary),
                "precision": precision_score(y_val_np, cb_pred_binary, zero_division=0),
                "recall": recall_score(y_val_np, cb_pred_binary, zero_division=0),
                "f1": f1_score(y_val_np, cb_pred_binary, zero_division=0),
                "auc": roc_auc_score(y_val_np, cb_pred),
            }
            
            # Train RandomForest with cuML
            logger.info("Training RandomForest model with cuML...")
            rf_params = self.hyperparams["randomforest"].copy()
            
            # Create and train cuML RandomForest
            rf_model = cuRF(
                n_estimators=rf_params["n_estimators"],
                max_depth=rf_params["max_depth"],
                max_features=0.8,  # cuML specific parameter
                n_bins=256,        # cuML specific parameter for faster training
                random_state=rf_params["random_state"],
            )
            
            rf_model.fit(X_train, y_train)
            
            # Save the model
            self.models["randomforest"] = rf_model
            
            # Get feature importances
            rf_importance = dict(zip(self.feature_names, rf_model.feature_importances_))
            total = sum(rf_importance.values()) or 1.0
            self.feature_importance["randomforest"] = {k: v / total for k, v in rf_importance.items()}
            
            # Get predictions
            rf_pred_proba = rf_model.predict_proba(X_val)
            rf_pred = rf_pred_proba[:, 1]
            rf_pred_binary = (rf_pred > 0.5).astype(int)
            
            self.model_metrics["randomforest"] = {
                "accuracy": accuracy_score(y_val_np, rf_pred_binary),
                "precision": precision_score(y_val_np, rf_pred_binary, zero_division=0),
                "recall": recall_score(y_val_np, rf_pred_binary, zero_division=0),
                "f1": f1_score(y_val_np, rf_pred_binary, zero_division=0),
                "auc": roc_auc_score(y_val_np, rf_pred),
            }
            
            # Combine feature importance and calculate correlations
            self._combine_feature_importance()
            self._calculate_feature_correlations(X)
            self.last_trained = datetime.now().isoformat()
            
            # Get ensemble predictions
            X_val_np = X_val.to_pandas().values
            ensemble_pred = self._ensemble_predict(X_val_np)
            ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
            
            self.model_metrics["ensemble"] = {
                "accuracy": accuracy_score(y_val_np, ensemble_pred_binary),
                "precision": precision_score(y_val_np, ensemble_pred_binary, zero_division=0),
                "recall": recall_score(y_val_np, ensemble_pred_binary, zero_division=0),
                "f1": f1_score(y_val_np, ensemble_pred_binary, zero_division=0),
                "auc": roc_auc_score(y_val_np, ensemble_pred),
            }
            
            # Optimize ensemble weights
            self._optimize_ensemble_weights(X_val_np, y_val_np)
            
            # Save the model
            self.save_model()
            
            logger.info("Model training completed with:")
            for model_name, metrics in self.model_metrics.items():
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"  {model_name}: {metrics_str}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error training model with RAPIDS: {e}")
            logger.info("Falling back to CPU training")
            return self._train_with_cpu(
                training_data, 
                non_feature_cols, 
                target_col, 
                optimize_hyperparams, 
                n_trials, 
                use_time_series_cv, 
                test_size
            )
    
    def _train_with_cpu(
        self,
        training_data: pd.DataFrame,
        non_feature_cols: List[str],
        target_col: str,
        optimize_hyperparams: bool,
        n_trials: int,
        use_time_series_cv: bool,
        test_size: float,
    ) -> bool:
        """Train the model using CPU."""
        try:
            X = training_data.drop(columns=non_feature_cols)
            y = training_data[target_col]
            self.feature_names = X.columns.tolist()
            X_scaled = self.scaler.fit_transform(X)
            
            if use_time_series_cv and "date" in training_data.columns:
                sorted_indices = training_data["date"].argsort()
                X_sorted = X_scaled[sorted_indices]
                y_sorted = y.iloc[sorted_indices].values
                val_size = int(len(X_sorted) * test_size)
                train_size = len(X_sorted) - val_size
                X_train = X_sorted[:train_size]
                y_train = y_sorted[:train_size]
                X_val = X_sorted[train_size:]
                y_val = y_sorted[train_size:]
                logger.info(f"Using time-series split: {train_size} train, {val_size} validation")
            else:
                X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
                logger.info(f"Using random split: {len(X_train)} train, {len(X_val)} validation")
                
            self.model_metrics = {}
            
            # Train XGBoost
            logger.info("Training XGBoost model...")
            if optimize_hyperparams:
                logger.info("Optimizing XGBoost hyperparameters...")
                self.optimize_hyperparameters(X_train, y_train, X_val, y_val, "xgboost", n_trials)
                
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            
            self.models["xgboost"] = xgb.train(
                self.hyperparams["xgboost"],
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=10,
                verbose_eval=False,
            )
            
            xgb_importance = self.models["xgboost"].get_score(importance_type="gain")
            total = sum(xgb_importance.values()) or 1.0
            self.feature_importance["xgboost"] = {k: v / total for k, v in xgb_importance.items()}
            
            xgb_pred = self.models["xgboost"].predict(dval)
            xgb_pred_binary = (xgb_pred > 0.5).astype(int)
            
            self.model_metrics["xgboost"] = {
                "accuracy": accuracy_score(y_val, xgb_pred_binary),
                "precision": precision_score(y_val, xgb_pred_binary, zero_division=0),
                "recall": recall_score(y_val, xgb_pred_binary, zero_division=0),
                "f1": f1_score(y_val, xgb_pred_binary, zero_division=0),
                "auc": roc_auc_score(y_val, xgb_pred),
            }
            
            # Train LightGBM
            logger.info("Training LightGBM model...")
            if optimize_hyperparams:
                logger.info("Optimizing LightGBM hyperparameters...")
                self.optimize_hyperparameters(X_train, y_train, X_val, y_val, "lightgbm", n_trials)
                
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            self.models["lightgbm"] = lgb.train(
                self.hyperparams["lightgbm"],
                train_data,
                num_boost_round=100,
                valid_sets=[train_data, val_data],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
            )
            
            lgb_importance = dict(zip(self.feature_names, self.models["lightgbm"].feature_importance(importance_type="gain")))
            total = sum(lgb_importance.values()) or 1.0
            self.feature_importance["lightgbm"] = {k: v / total for k, v in lgb_importance.items()}
            
            lgb_pred = self.models["lightgbm"].predict(X_val)
            lgb_pred_binary = (lgb_pred > 0.5).astype(int)
            
            self.model_metrics["lightgbm"] = {
                "accuracy": accuracy_score(y_val, lgb_pred_binary),
                "precision": precision_score(y_val, lgb_pred_binary, zero_division=0),
                "recall": recall_score(y_val, lgb_pred_binary, zero_division=0),
                "f1": f1_score(y_val, lgb_pred_binary, zero_division=0),
                "auc": roc_auc_score(y_val, lgb_pred),
            }
            
            # Train CatBoost
            logger.info("Training CatBoost model...")
            if optimize_hyperparams:
                logger.info("Optimizing CatBoost hyperparameters...")
                self.optimize_hyperparameters(X_train, y_train, X_val, y_val, "catboost", n_trials)
                
            train_data = cb.Pool(X_train, label=y_train)
            val_data = cb.Pool(X_val, label=y_val)
            
            self.models["catboost"] = cb.CatBoost(self.hyperparams["catboost"])
            self.models["catboost"].fit(train_data, eval_set=val_data, early_stopping_rounds=10, verbose=False)
            
            cb_importance = dict(zip(self.feature_names, self.models["catboost"].get_feature_importance()))
            total = sum(cb_importance.values()) or 1.0
            self.feature_importance["catboost"] = {k: v / total for k, v in cb_importance.items()}
            
            cb_pred = self.models["catboost"].predict(X_val, prediction_type="Probability")[:, 1]
            cb_pred_binary = (cb_pred > 0.5).astype(int)
            
            self.model_metrics["catboost"] = {
                "accuracy": accuracy_score(y_val, cb_pred_binary),
                "precision": precision_score(y_val, cb_pred_binary, zero_division=0),
                "recall": recall_score(y_val, cb_pred_binary, zero_division=0),
                "f1": f1_score(y_val, cb_pred_binary, zero_division=0),
                "auc": roc_auc_score(y_val, cb_pred),
            }
            
            # Train RandomForest with scikit-learn
            logger.info("Training RandomForest model with scikit-learn...")
            rf_params = self.hyperparams["randomforest"].copy()
            
            rf_model = RandomForestClassifier(
                n_estimators=rf_params["n_estimators"],
                max_depth=rf_params["max_depth"],
                min_samples_split=rf_params["min_samples_split"],
                min_samples_leaf=rf_params["min_samples_leaf"],
                random_state=rf_params["random_state"],
            )
            
            rf_model.fit(X_train, y_train)
            
            # Save the model
            self.models["randomforest"] = rf_model
            
            # Get feature importances
            rf_importance = dict(zip(self.feature_names, rf_model.feature_importances_))
            total = sum(rf_importance.values()) or 1.0
            self.feature_importance["randomforest"] = {k: v / total for k, v in rf_importance.items()}
            
            # Get predictions
            rf_pred = rf_model.predict_proba(X_val)[:, 1]
            rf_pred_binary = (rf_pred > 0.5).astype(int)
            
            self.model_metrics["randomforest"] = {
                "accuracy": accuracy_score(y_val, rf_pred_binary),
                "precision": precision_score(y_val, rf_pred_binary, zero_division=0),
                "recall": recall_score(y_val, rf_pred_binary, zero_division=0),
                "f1": f1_score(y_val, rf_pred_binary, zero_division=0),
                "auc": roc_auc_score(y_val, rf_pred),
            }
            
            # Combine feature importance and calculate correlations
            self._combine_feature_importance()
            self._calculate_feature_correlations(X)
            self.last_trained = datetime.now().isoformat()
            
            # Get ensemble predictions
            ensemble_pred = self._ensemble_predict(X_val)
            ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
            
            self.model_metrics["ensemble"] = {
                "accuracy": accuracy_score(y_val, ensemble_pred_binary),
                "precision": precision_score(y_val, ensemble_pred_binary, zero_division=0),
                "recall": recall_score(y_val, ensemble_pred_binary, zero_division=0),
                "f1": f1_score(y_val, ensemble_pred_binary, zero_division=0),
                "auc": roc_auc_score(y_val, ensemble_pred),
            }
            
            # Optimize ensemble weights
            self._optimize_ensemble_weights(X_val, y_val)
            
            # Save the model
            self.save_model()
            
            logger.info("Model training completed with:")
            for model_name, metrics in self.model_metrics.items():
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"  {model_name}: {metrics_str}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error training model with CPU: {e}")
            return False
            X = training_data.drop(columns=non_feature_cols)
            y = training_data[target_col]
            self.feature_names = X.columns.tolist()
            X_scaled = self.scaler.fit_transform(X)
            if use_time_series_cv and "date" in training_data.columns:
                sorted_indices = training_data["date"].argsort()
                X_sorted = X_scaled[sorted_indices]
                y_sorted = y.iloc[sorted_indices].values
                val_size = int(len(X_sorted) * test_size)
                train_size = len(X_sorted) - val_size
                X_train = X_sorted[:train_size]
                y_train = y_sorted[:train_size]
                X_val = X_sorted[train_size:]
                y_val = y_sorted[train_size:]
                logger.info(f"Using time-series split: {train_size} train, {val_size} validation")
            else:
                X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
                logger.info(f"Using random split: {len(X_train)} train, {len(X_val)} validation")
            self.model_metrics = {}
            logger.info("Training XGBoost model...")
            if optimize_hyperparams:
                logger.info("Optimizing XGBoost hyperparameters...")
                self.optimize_hyperparameters(X_train, y_train, X_val, y_val, "xgboost", n_trials)
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            self.models["xgboost"] = xgb.train(
                self.hyperparams["xgboost"],
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=10,
                verbose_eval=False,
            )
            xgb_importance = self.models["xgboost"].get_score(importance_type="gain")
            total = sum(xgb_importance.values()) or 1.0
            self.feature_importance["xgboost"] = {k: v / total for k, v in xgb_importance.items()}
            xgb_pred = self.models["xgboost"].predict(dval)
            xgb_pred_binary = (xgb_pred > 0.5).astype(int)
            self.model_metrics["xgboost"] = {
                "accuracy": accuracy_score(y_val, xgb_pred_binary),
                "precision": precision_score(y_val, xgb_pred_binary, zero_division=0),
                "recall": recall_score(y_val, xgb_pred_binary, zero_division=0),
                "f1": f1_score(y_val, xgb_pred_binary, zero_division=0),
                "auc": roc_auc_score(y_val, xgb_pred),
            }
            logger.info("Training LightGBM model...")
            if optimize_hyperparams:
                logger.info("Optimizing LightGBM hyperparameters...")
                self.optimize_hyperparameters(X_train, y_train, X_val, y_val, "lightgbm", n_trials)
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            self.models["lightgbm"] = lgb.train(
                self.hyperparams["lightgbm"],
                train_data,
                num_boost_round=100,
                valid_sets=[train_data, val_data],
                valid_names=["train", "valid"],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
            )
            lgb_importance = dict(zip(self.feature_names, self.models["lightgbm"].feature_importance(importance_type="gain")))
            total = sum(lgb_importance.values()) or 1.0
            self.feature_importance["lightgbm"] = {k: v / total for k, v in lgb_importance.items()}
            lgb_pred = self.models["lightgbm"].predict(X_val)
            lgb_pred_binary = (lgb_pred > 0.5).astype(int)
            self.model_metrics["lightgbm"] = {
                "accuracy": accuracy_score(y_val, lgb_pred_binary),
                "precision": precision_score(y_val, lgb_pred_binary, zero_division=0),
                "recall": recall_score(y_val, lgb_pred_binary, zero_division=0),
                "f1": f1_score(y_val, lgb_pred_binary, zero_division=0),
                "auc": roc_auc_score(y_val, lgb_pred),
            }
            logger.info("Training CatBoost model...")
            if optimize_hyperparams:
                logger.info("Optimizing CatBoost hyperparameters...")
                self.optimize_hyperparameters(X_train, y_train, X_val, y_val, "catboost", n_trials)
            train_data = cb.Pool(X_train, label=y_train)
            val_data = cb.Pool(X_val, label=y_val)
            self.models["catboost"] = cb.CatBoost(self.hyperparams["catboost"])
            self.models["catboost"].fit(train_data, eval_set=val_data, early_stopping_rounds=10, verbose=False)
            cb_importance = dict(zip(self.feature_names, self.models["catboost"].get_feature_importance()))
            total = sum(cb_importance.values()) or 1.0
            self.feature_importance["catboost"] = {k: v / total for k, v in cb_importance.items()}
            cb_pred = self.models["catboost"].predict(X_val, prediction_type="Probability")[:, 1]
            cb_pred_binary = (cb_pred > 0.5).astype(int)
            self.model_metrics["catboost"] = {
                "accuracy": accuracy_score(y_val, cb_pred_binary),
                "precision": precision_score(y_val, cb_pred_binary, zero_division=0),
                "recall": recall_score(y_val, cb_pred_binary, zero_division=0),
                "f1": f1_score(y_val, cb_pred_binary, zero_division=0),
                "auc": roc_auc_score(y_val, cb_pred),
            }
            self._combine_feature_importance()
            self._calculate_feature_correlations(X)
            self.last_trained = datetime.now().isoformat()
            ensemble_pred = self._ensemble_predict(X_val)
            ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
            self.model_metrics["ensemble"] = {
                "accuracy": accuracy_score(y_val, ensemble_pred_binary),
                "precision": precision_score(y_val, ensemble_pred_binary, zero_division=0),
                "recall": recall_score(y_val, ensemble_pred_binary, zero_division=0),
                "f1": f1_score(y_val, ensemble_pred_binary, zero_division=0),
                "auc": roc_auc_score(y_val, ensemble_pred),
            }
            self._optimize_ensemble_weights(X_val, y_val)
            self.save_model()
            logger.info("Model training completed with:")
            for model_name, metrics in self.model_metrics.items():
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"  {model_name}: {metrics_str}")
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def _combine_feature_importance(self):
        """Combine feature importance from all models."""
        combined = {feature: 0 for feature in self.feature_names}
        weights = self.hyperparams["meta"]["weights"]
        for feature in self.feature_names:
            for model_name, importance in self.feature_importance.items():
                if model_name in weights and feature in importance:
                    combined[feature] += importance[feature] * weights[model_name]
        total = sum(combined.values()) or 1.0
        combined = {k: v / total for k, v in combined.items()}
        self.feature_importance["combined"] = combined

    def _calculate_feature_correlations(self, X: pd.DataFrame):
        """Calculate feature correlations."""
        corr_matrix = X.corr()
        self.feature_correlations = {}
        for feature in self.feature_names:
            correlations = corr_matrix[feature].sort_values(ascending=False)
            self.feature_correlations[feature] = {
                other_feature: corr
                for other_feature, corr in correlations.items()
                if other_feature != feature and not pd.isna(corr)
            }

    def _optimize_ensemble_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """Optimize ensemble weights based on validation performance."""
        def objective(trial):
            xgb_weight = trial.suggest_float("xgb_weight", 0.1, 1.0)
            lgb_weight = trial.suggest_float("lgb_weight", 0.1, 1.0)
            cb_weight = trial.suggest_float("cb_weight", 0.1, 1.0)
            total = xgb_weight + lgb_weight + cb_weight
            xgb_weight /= total
            lgb_weight /= total
            cb_weight /= total
            xgb_pred = self.models["xgboost"].predict(xgb.DMatrix(X_val, feature_names=self.feature_names))
            lgb_pred = self.models["lightgbm"].predict(X_val)
            cb_pred = self.models["catboost"].predict(X_val, prediction_type="Probability")[:, 1]
            ensemble_pred = xgb_weight * xgb_pred + lgb_weight * lgb_pred + cb_weight * cb_pred
            auc = roc_auc_score(y_val, ensemble_pred)
            return auc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)
        best_params = study.best_params
        total = sum(best_params.values())
        normalized_weights = {
            "xgboost": best_params["xgb_weight"] / total,
            "lightgbm": best_params["lgb_weight"] / total,
            "catboost": best_params["cb_weight"] / total,
        }
        self.hyperparams["meta"]["weights"] = normalized_weights
        logger.info(f"Optimized ensemble weights: {normalized_weights}")

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Get prediction from ensemble of models."""
        xgb_pred = self.models["xgboost"].predict(xgb.DMatrix(X, feature_names=self.feature_names))
        lgb_pred = self.models["lightgbm"].predict(X)
        cb_pred = self.models["catboost"].predict(X, prediction_type="Probability")[:, 1]
        weights = self.hyperparams["meta"]["weights"]
        ensemble_pred = weights["xgboost"] * xgb_pred + weights["lightgbm"] * lgb_pred + weights["catboost"] * cb_pred
        return ensemble_pred

    @log_execution_time(logger)
    def rank_stocks(self, stock_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Rank stocks based on model predictions.

        Args:
            stock_data: Dictionary of stock DataFrames with features.

        Returns:
            List of ranked stocks with scores.
        """
        try:
            if not stock_data:
                logger.warning("Empty stock data for ranking")
                return []
            if not self.models:
                logger.warning("No models found, using simple ranking formula")
                return self._simple_ranking(stock_data)
            ranked_stocks = []
            for symbol, df in stock_data.items():
                try:
                    features = self._extract_features(df)
                    if features is None:
                        continue
                    features_dict = features.copy()
                    model_features = {feature: features_dict.get(feature, 0.0) for feature in self.feature_names}
                    features_array = np.array([list(model_features.values())])
                    features_scaled = self.scaler.transform(features_array)
                    scores = {}
                    if "xgboost" in self.models and self.models["xgboost"] is not None:
                        dfeatures = xgb.DMatrix(features_scaled, feature_names=self.feature_names)
                        scores["xgboost"] = float(self.models["xgboost"].predict(dfeatures)[0])
                    if "lightgbm" in self.models and self.models["lightgbm"] is not None:
                        scores["lightgbm"] = float(self.models["lightgbm"].predict(features_scaled)[0])
                    if "catboost" in self.models and self.models["catboost"] is not None:
                        scores["catboost"] = float(
                            self.models["catboost"].predict(features_scaled, prediction_type="Probability")[0, 1]
                        )
                    if "randomforest" in self.models and self.models["randomforest"] is not None:
                        scores["randomforest"] = float(self.models["randomforest"].predict_proba(features_scaled)[0, 1])
                    if self.model_version == "ensemble" and len(scores) > 1:
                        weights = self.hyperparams["meta"]["weights"]
                        ensemble_score = 0.0
                        total_weight = 0.0
                        for model_name, score in scores.items():
                            if model_name in weights:
                                ensemble_score += score * weights[model_name]
                                total_weight += weights[model_name]
                        score = ensemble_score / total_weight if total_weight > 0 else sum(scores.values()) / len(scores)
                    else:
                        score = scores.get(self.model_version, next(iter(scores.values()), 0.0))
                    current_price = df["close"].iloc[-1] if "close" in df.columns else 0
                    ranked_stocks.append(
                        {
                            "symbol": symbol,
                            "score": score,
                            "model_scores": scores,
                            "price": current_price,
                            "features": features,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                except Exception as e:
                    logger.error(f"Error ranking stock {symbol}: {e}")
                    continue
            ranked_stocks.sort(key=lambda x: x["score"], reverse=True)
            return ranked_stocks
        except Exception as e:
            logger.error(f"Error ranking stocks: {e}")
            return []

    def _simple_ranking(self, stock_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Simple ranking method when no model is available.

        Args:
            stock_data: Dictionary of stock DataFrames with features.

        Returns:
            List of ranked stocks with scores.
        """
        try:
            ranked_stocks = []
            for symbol, df in stock_data.items():
                try:
                    if len(df) < 20:
                        continue
                    if "close" not in df.columns or "volume" not in df.columns:
                        continue
                    current_price = df["close"].iloc[-1]
                    previous_price = df["close"].iloc[-2]
                    price_change = current_price - previous_price
                    price_change_pct = (price_change / previous_price) * 100
                    daily_range_pct = (
                        ((df["high"].iloc[-1] - df["low"].iloc[-1]) / df["low"].iloc[-1]) * 100
                        if "high" in df.columns and "low" in df.columns
                        else 0
                    )
                    current_volume = df["volume"].iloc[-1]
                    avg_volume = df["volume"].iloc[-20:].mean()
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                    volatility = df["close"].pct_change().iloc[-20:].std() * 100
                    momentum_5d = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100 if len(df) >= 6 else 0
                    ma5 = df["close"].rolling(5).mean().iloc[-1] if len(df) >= 5 else df["close"].iloc[-1]
                    ma20 = df["close"].rolling(20).mean().iloc[-1] if len(df) >= 20 else df["close"].iloc[-1]
                    ma_crossover = 1 if ma5 > ma20 else 0
                    delta = df["close"].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    current_rsi = float(rs.iloc[-1]) if not pd.isna(rs.iloc[-1]) else 50
                    momentum_score = (abs(price_change_pct) * 0.3 + momentum_5d * 0.7) * self.factor_weights.get("momentum", 0.3)
                    if price_change_pct > 0:
                        momentum_score *= 1.2
                    volatility_score = (volatility * 0.5 + daily_range_pct * 0.5) * self.factor_weights.get("volatility", 0.2)
                    volume_score = min(volume_ratio, 5) * self.factor_weights.get("volume", 0.2)
                    trend_score = (ma_crossover * 0.7 + (current_price / ma20 - 1) * 0.3) * self.factor_weights.get("trend", 0.2)
                    value_score = ((70 - min(max(current_rsi, 30), 70)) / 40) * self.factor_weights.get("value", 0.1)
                    total_score = momentum_score + volatility_score + volume_score + trend_score + value_score
                    features = {
                        "price_change_pct": float(price_change_pct),
                        "daily_range_pct": float(daily_range_pct),
                        "volume_ratio": float(volume_ratio),
                        "volatility": float(volatility),
                        "momentum_5d": float(momentum_5d),
                        "rsi": float(current_rsi),
                        "ma_crossover": float(ma_crossover),
                        "close_to_ma20": float(current_price / ma20) if ma20 > 0 else 1.0,
                    }
                    ranked_stocks.append(
                        {
                            "symbol": symbol,
                            "score": float(total_score),
                            "price": float(current_price),
                            "features": features,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                except Exception as e:
                    logger.error(f"Error in simple ranking for stock {symbol}: {e}")
                    continue
            ranked_stocks.sort(key=lambda x: x["score"], reverse=True)
            return ranked_stocks
        except Exception as e:
            logger.error(f"Error in simple ranking: {e}")
            return []

    def _extract_features(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Extract features from stock data.

        Args:
            df: DataFrame with OHLCV data.

        Returns:
            Dictionary of features, or None if extraction fails.
        """
        try:
            if df.empty or len(df) < 20:
                logger.warning("Insufficient data for feature extraction")
                return None
            features = {}
            if "close" in df.columns:
                df["returns"] = df["close"].pct_change()
                df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
                features["return_1d"] = df["returns"].iloc[-1]
                features["return_5d"] = df["close"].iloc[-1] / df["close"].iloc[-6] - 1 if len(df) >= 6 else 0
                features["return_10d"] = df["close"].iloc[-1] / df["close"].iloc[-11] - 1 if len(df) >= 11 else 0
                df["ma5"] = df["close"].rolling(5).mean()
                df["ma10"] = df["close"].rolling(10).mean()
                df["ma20"] = df["close"].rolling(20).mean()
                features["close_ma5_ratio"] = df["close"].iloc[-1] / df["ma5"].iloc[-1] if df["ma5"].iloc[-1] > 0 else 0
                features["close_ma10_ratio"] = df["close"].iloc[-1] / df["ma10"].iloc[-1] if df["ma10"].iloc[-1] > 0 else 0
                features["close_ma20_ratio"] = df["close"].iloc[-1] / df["ma20"].iloc[-1] if df["ma20"].iloc[-1] > 0 else 0
                features["ma5_ma20_ratio"] = df["ma5"].iloc[-1] / df["ma20"].iloc[-1] if df["ma20"].iloc[-1] > 0 else 0
                features["volatility_5d"] = df["returns"].iloc[-5:].std() if len(df) >= 5 else 0
                features["volatility_10d"] = df["returns"].iloc[-10:].std() if len(df) >= 10 else 0
                features["volatility_20d"] = df["returns"].iloc[-20:].std() if len(df) >= 20 else 0
                features["realized_vol_annual"] = df["returns"].iloc[-20:].std() * np.sqrt(252) if len(df) >= 20 else 0
                if len(df) > 252:
                    high_52w = df["close"].iloc[-252:].max()
                    low_52w = df["close"].iloc[-252:].min()
                    features["pct_off_52w_high"] = (df["close"].iloc[-1] / high_52w) - 1
                    features["pct_off_52w_low"] = (df["close"].iloc[-1] / low_52w) - 1
                for period in [1, 5, 10, 20]:
                    if len(df) > period:
                        features[f"roc_{period}d"] = df["close"].iloc[-1] / df["close"].iloc[-(period + 1)] - 1
                if len(df) > 5:
                    roc_1d_now = df["close"].iloc[-1] / df["close"].iloc[-2] - 1
                    roc_1d_prev = df["close"].iloc[-2] / df["close"].iloc[-3] - 1
                    features["momentum_acceleration"] = roc_1d_now - roc_1d_prev
            if "volume" in df.columns:
                features["volume_1d"] = df["volume"].iloc[-1]
                features["volume_ma5"] = df["volume"].iloc[-5:].mean() if len(df) >= 5 else df["volume"].iloc[-1]
                features["volume_ma10"] = df["volume"].iloc[-10:].mean() if len(df) >= 10 else features["volume_ma5"]
                features["volume_ratio_5d"] = (
                    df["volume"].iloc[-1] / features["volume_ma5"] if features["volume_ma5"] > 0 else 0
                )
                features["volume_ratio_10d"] = (
                    df["volume"].iloc[-1] / features["volume_ma10"] if features["volume_ma10"] > 0 else 0
                )
                volume_trend = df["volume"].iloc[-5:].pct_change().mean() if len(df) >= 5 else 0
                features["volume_trend_5d"] = volume_trend
                if "close" in df.columns:
                    df["obv"] = 0
                    df["obv_signal"] = np.where(
                        df["close"] > df["close"].shift(1),
                        df["volume"],
                        np.where(df["close"] < df["close"].shift(1), -df["volume"], 0),
                    )
                    df["obv"] = df["obv_signal"].cumsum()
                    if len(df) > 5:
                        features["obv_momentum"] = (
                            df["obv"].iloc[-1] / df["obv"].iloc[-6] - 1 if df["obv"].iloc[-6] != 0 else 0
                        )
                if "close" in df.columns and len(df) > 1:
                    price_changes = df["close"].pct_change().abs().iloc[-20:]
                    volumes = df["volume"].iloc[-20:]
                    if len(price_changes) > 5 and len(volumes) > 5:
                        try:
                            correlation, _ = pearsonr(price_changes.dropna(), volumes.iloc[1:])
                            features["volume_price_change_correlation"] = correlation if not np.isnan(correlation) else 0
                        except:
                            features["volume_price_change_correlation"] = 0
            if all(col in df.columns for col in ["open", "high", "low", "close"]):
                df["daily_range"] = (df["high"] - df["low"]) / df["low"] * 100
                features["daily_range"] = df["daily_range"].iloc[-1]
                features["daily_range_avg_5d"] = (
                    df["daily_range"].iloc[-5:].mean() if len(df) >= 5 else features["daily_range"]
                )
                df["body_size"] = abs(df["close"] - df["open"]) / df["open"] * 100
                features["body_size"] = df["body_size"].iloc[-1]
                features["body_size_avg_5d"] = df["body_size"].iloc[-5:].mean() if len(df) >= 5 else features["body_size"]
                df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df[["open", "close"]].max(axis=1) * 100
                df["lower_shadow"] = (
                    df[["open", "close"]].min(axis=1) - df["low"]
                ) / df[["open", "close"]].min(axis=1) * 100
                features["upper_shadow"] = df["upper_shadow"].iloc[-1]
                features["lower_shadow"] = df["lower_shadow"].iloc[-1]
                features["is_bullish"] = 1.0 if df["close"].iloc[-1] > df["open"].iloc[-1] else 0.0
                if len(df) > 1:
                    prev_close = df["close"].iloc[-2]
                    curr_open = df["open"].iloc[-1]
                    features["gap_pct"] = (curr_open - prev_close) / prev_close * 100
                else:
                    features["gap_pct"] = 0.0
                body_to_range_ratio = (
                    features["body_size"] / features["daily_range"] if features["daily_range"] > 0 else 0
                )
                features["is_doji"] = 1.0 if body_to_range_ratio < 0.1 else 0.0
                if features["lower_shadow"] > 2 * features["body_size"] and features["upper_shadow"] < features["body_size"]:
                    features["is_hammer"] = 1.0
                else:
                    features["is_hammer"] = 0.0
                if features["upper_shadow"] > 2 * features["body_size"] and features["lower_shadow"] < features["body_size"]:
                    features["is_shooting_star"] = 1.0
                else:
                    features["is_shooting_star"] = 0.0
                if len(df) > 1:
                    prev_body_size = abs(df["close"].iloc[-2] - df["open"].iloc[-2])
                    curr_body_size = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
                    if (
                        df["close"].iloc[-1] > df["open"].iloc[-1]
                        and df["close"].iloc[-2] < df["open"].iloc[-2]
                        and df["close"].iloc[-1] > df["open"].iloc[-2]
                        and df["open"].iloc[-1] < df["close"].iloc[-2]
                    ):
                        features["is_bullish_engulfing"] = 1.0
                    else:
                        features["is_bullish_engulfing"] = 0.0
                    if (
                        df["close"].iloc[-1] < df["open"].iloc[-1]
                        and df["close"].iloc[-2] > df["open"].iloc[-2]
                        and df["close"].iloc[-1] < df["open"].iloc[-2]
                        and df["open"].iloc[-1] > df["close"].iloc[-2]
                    ):
                        features["is_bearish_engulfing"] = 1.0
                    else:
                        features["is_bearish_engulfing"] = 0.0
            if "close" in df.columns:
                delta = df["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df["rsi"] = 100 - (100 / (1 + rs))
                features["rsi"] = df["rsi"].iloc[-1] if not pd.isna(df["rsi"].iloc[-1]) else 50.0
                if len(df) > 5:
                    features["rsi_5d_change"] = (
                        df["rsi"].iloc[-1] - df["rsi"].iloc[-6] if not pd.isna(df["rsi"].iloc[-6]) else 0
                    )
                features["is_overbought"] = 1.0 if features["rsi"] > 70 else 0.0
                features["is_oversold"] = 1.0 if features["rsi"] < 30 else 0.0
                if len(df) >= 26:
                    ema12 = df["close"].ewm(span=12, adjust=False).mean()
                    ema26 = df["close"].ewm(span=26, adjust=False).mean()
                    df["macd"] = ema12 - ema26
                    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
                    df["macd_hist"] = df["macd"] - df["macd_signal"]
                    features["macd"] = df["macd"].iloc[-1]
                    features["macd_signal"] = df["macd_signal"].iloc[-1]
                    features["macd_hist"] = df["macd_hist"].iloc[-1]
                    if len(df) > 1:
                        features["macd_crossover"] = (
                            1.0
                            if df["macd"].iloc[-1] > df["macd_signal"].iloc[-1]
                            and df["macd"].iloc[-2] <= df["macd_signal"].iloc[-2]
                            else 0.0
                        )
                        features["macd_crossunder"] = (
                            1.0
                            if df["macd"].iloc[-1] < df["macd_signal"].iloc[-1]
                            and df["macd"].iloc[-2] >= df["macd_signal"].iloc[-2]
                            else 0.0
                        )
                if len(df) >= 20:
                    df["bb_middle"] = df["close"].rolling(20).mean()
                    df["bb_std"] = df["close"].rolling(20).std()
                    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
                    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
                    df["percent_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
                    features["bb_width"] = (
                        (df["bb_upper"].iloc[-1] - df["bb_lower"].iloc[-1]) / df["bb_middle"].iloc[-1]
                    )
                    features["percent_b"] = df["percent_b"].iloc[-1]
                    features["touched_upper_band"] = 1.0 if df["high"].iloc[-1] >= df["bb_upper"].iloc[-1] else 0.0
                    features["touched_lower_band"] = 1.0 if df["low"].iloc[-1] <= df["bb_lower"].iloc[-1] else 0.0
                if "high" in df.columns and "low" in df.columns:
                    typical_price = (df["high"] + df["low"] + df["close"]) / 3
                    moving_avg = typical_price.rolling(window=20).mean()
                    mean_deviation = abs(typical_price - moving_avg).rolling(window=20).mean()
                    df["cci"] = (typical_price - moving_avg) / (0.015 * mean_deviation)
                    features["cci"] = df["cci"].iloc[-1] if not pd.isna(df["cci"].iloc[-1]) else 0
                if "high" in df.columns and "low" in df.columns and len(df) > 14:
                    high_diff = df["high"].diff()
                    low_diff = df["low"].diff()
                    pos_dm = high_diff.copy()
                    pos_dm[pos_dm < 0] = 0
                    pos_dm[high_diff <= low_diff.abs()] = 0
                    neg_dm = low_diff.abs().copy()
                    neg_dm[neg_dm < 0] = 0
                    neg_dm[low_diff.abs() <= high_diff] = 0
                    tr1 = df["high"] - df["low"]
                    tr2 = (df["high"] - df["close"].shift(1)).abs()
                    tr3 = (df["low"] - df["close"].shift(1)).abs()
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = tr.rolling(window=14).mean()
                    pdi = 100 * (pos_dm.rolling(window=14).mean() / atr)
                    ndi = 100 * (neg_dm.rolling(window=14).mean() / atr)
                    dx = 100 * ((pdi - ndi).abs() / (pdi + ndi).abs())
                    adx = dx.rolling(window=14).mean()
                    features["adx"] = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
                    features["pdi"] = pdi.iloc[-1] if not pd.isna(pdi.iloc[-1]) else 0
                    features["ndi"] = ndi.iloc[-1] if not pd.isna(ndi.iloc[-1]) else 0
                    features["trend_strength"] = features["adx"] / 100.0
                    features["trend_direction"] = 1.0 if features["pdi"] > features["ndi"] else -1.0
                if "high" in df.columns and "low" in df.columns and len(df) >= 14:
                    low_14 = df["low"].rolling(window=14).min()
                    high_14 = df["high"].rolling(window=14).max()
                    k_percent = 100 * ((df["close"] - low_14) / (high_14 - low_14))
                    d_percent = k_percent.rolling(window=3).mean()
                    features["stoch_k"] = k_percent.iloc[-1] if not pd.isna(k_percent.iloc[-1]) else 50
                    features["stoch_d"] = d_percent.iloc[-1] if not pd.isna(d_percent.iloc[-1]) else 50
                    features["stoch_overbought"] = 1.0 if features["stoch_k"] > 80 else 0.0
                    features["stoch_oversold"] = 1.0 if features["stoch_k"] < 20 else 0.0
                    if len(df) > 3:
                        features["stoch_crossover"] = (
                            1.0
                            if k_percent.iloc[-1] > d_percent.iloc[-1] and k_percent.iloc[-2] <= d_percent.iloc[-2]
                            else 0.0
                        )
                        features["stoch_crossunder"] = (
                            1.0
                            if k_percent.iloc[-1] < d_percent.iloc[-1] and k_percent.iloc[-2] >= d_percent.iloc[-2]
                            else 0.0
                        )
                if len(df) >= 52:
                    nine_period_high = df["high"].rolling(window=9).max()
                    nine_period_low = df["low"].rolling(window=9).min()
                    df["tenkan_sen"] = (nine_period_high + nine_period_low) / 2
                    period26_high = df["high"].rolling(window=26).max()
                    period26_low = df["low"].rolling(window=26).min()
                    df["kijun_sen"] = (period26_high + period26_low) / 2
                    df["senkou_span_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(26)
                    period52_high = df["high"].rolling(window=52).max()
                    period52_low = df["low"].rolling(window=52).min()
                    df["senkou_span_b"] = ((period52_high + period52_low) / 2).shift(26)
                    features["cloud_bullish"] = (
                        1.0 if df["senkou_span_a"].iloc[-1] > df["senkou_span_b"].iloc[-1] else 0.0
                    )
                    features["price_above_cloud"] = (
                        1.0
                        if df["close"].iloc[-1] > max(df["senkou_span_a"].iloc[-1], df["senkou_span_b"].iloc[-1])
                        else 0.0
                    )
                    features["price_below_cloud"] = (
                        1.0
                        if df["close"].iloc[-1] < min(df["senkou_span_a"].iloc[-1], df["senkou_span_b"].iloc[-1])
                        else 0.0
                    )
                    features["tk_cross"] = (
                        1.0
                        if df["tenkan_sen"].iloc[-1] > df["kijun_sen"].iloc[-1]
                        and df["tenkan_sen"].iloc[-2] <= df["kijun_sen"].iloc[-2]
                        else 0.0
                    )
                if len(df) >= 50:
                    ma_50 = df["close"].rolling(window=50).mean()
                    ma_200 = df["close"].rolling(window=200).mean() if len(df) >= 200 else None
                    features["ma_50_trend"] = df["close"].iloc[-1] / ma_50.iloc[-1] - 1
                    if ma_200 is not None:
                        features["ma_200_trend"] = df["close"].iloc[-1] / ma_200.iloc[-1] - 1
                        features["golden_cross"] = (
                            1.0 if ma_50.iloc[-1] > ma_200.iloc[-1] and ma_50.iloc[-2] <= ma_200.iloc[-2] else 0.0
                        )
                        features["death_cross"] = (
                            1.0 if ma_50.iloc[-1] < ma_200.iloc[-1] and ma_50.iloc[-2] >= ma_200.iloc[-2] else 0.0
                        )
                if len(df) >= 20:
                    log_returns = np.log(df["close"] / df["close"].shift(1))
                    features["historical_volatility"] = log_returns.rolling(window=20).std() * np.sqrt(252)
                    if len(df) >= 40:
                        vol_20 = log_returns.rolling(window=20).std().iloc[-1]
                        vol_20_prev = log_returns.rolling(window=20).std().iloc[-20]
                        features["volatility_trend"] = vol_20 / vol_20_prev - 1 if vol_20_prev > 0 else 0
                if len(df) >= 20:
                    ma_20 = df["close"].rolling(window=20).mean()
                    std_20 = df["close"].rolling(window=20).std()
                    features["zscore_20d"] = (
                        (df["close"].iloc[-1] - ma_20.iloc[-1]) / std_20.iloc[-1] if std_20.iloc[-1] > 0 else 0
                    )
                    if "bb_upper" in df.columns and "bb_lower" in df.columns:
                        upper_band_distance = (df["bb_upper"].iloc[-1] - df["close"].iloc[-1]) / df["close"].iloc[-1]
                        lower_band_distance = (df["close"].iloc[-1] - df["bb_lower"].iloc[-1]) / df["close"].iloc[-1]
                        features["upper_band_distance"] = upper_band_distance
                        features["lower_band_distance"] = lower_band_distance
                features["market_breadth"] = 0.5
                if "high" in df.columns and "low" in df.columns and len(df) >= 20:
                    recent_highs = df["high"].rolling(window=5, center=True).max()
                    recent_lows = df["low"].rolling(window=5, center=True).min()
                    support_levels = []
                    resistance_levels = []
                    for i in range(5, len(df) - 5):
                        if (
                            recent_highs.iloc[i] == df["high"].iloc[i]
                            and recent_highs.iloc[i] > recent_highs.iloc[i - 5 : i].max()
                            and recent_highs.iloc[i] > recent_highs.iloc[i + 1 : i + 6].max()
                        ):
                            resistance_levels.append(df["high"].iloc[i])
                        if (
                            recent_lows.iloc[i] == df["low"].iloc[i]
                            and recent_lows.iloc[i] < recent_lows.iloc[i - 5 : i].min()
                            and recent_lows.iloc[i] < recent_lows.iloc[i + 1 : i + 6].min()
                        ):
                            support_levels.append(df["low"].iloc[i])
                    if resistance_levels:
                        nearest_resistance = min([r for r in resistance_levels if r > df["close"].iloc[-1]], default=0)
                        if nearest_resistance > 0:
                            features["distance_to_resistance"] = (
                                nearest_resistance - df["close"].iloc[-1]
                            ) / df["close"].iloc[-1]
                    if support_levels:
                        nearest_support = max([s for s in support_levels if s < df["close"].iloc[-1]], default=0)
                        if nearest_support > 0:
                            features["distance_to_support"] = (
                                df["close"].iloc[-1] - nearest_support
                            ) / df["close"].iloc[-1]
                if isinstance(df.index, pd.DatetimeIndex):
                    features["day_of_week"] = df.index[-1].dayofweek / 4.0
                    features["day_of_month"] = (df.index[-1].day - 1) / 30.0
                    features["month_of_year"] = (df.index[-1].month - 1) / 11.0
                    last_day = pd.Timestamp(df.index[-1].year, df.index[-1].month, 1) + pd.offsets.MonthEnd(1)
                    days_to_month_end = (last_day - df.index[-1]).days
                    features["end_of_month"] = 1.0 if days_to_month_end <= 3 else 0.0
                if len(df) >= 5:
                    if "high" in df.columns:
                        max_high_4d = df["high"].iloc[-5:-1].max()
                        features["breakout"] = 1.0 if df["close"].iloc[-1] > max_high_4d * 1.02 else 0.0
                    if "low" in df.columns:
                        min_low_4d = df["low"].iloc[-5:-1].min()
                        features["breakdown"] = 1.0 if df["close"].iloc[-1] < min_low_4d * 0.98 else 0.0
                    if "high" in df.columns and "low" in df.columns:
                        high_range = df["high"].iloc[-5:].max() - df["high"].iloc[-5:].min()
                        low_range = df["low"].iloc[-5:].max() - df["low"].iloc[-5:].min()
                        avg_price = df["close"].iloc[-5:].mean()
                        range_pct = max(high_range, abs(low_range)) / avg_price
                        features["consolidation"] = 1.0 if range_pct < 0.03 else 0.0
                    price_direction_old = np.sign(df["close"].iloc[-3] - df["close"].iloc[-5])
                    price_direction_new = np.sign(df["close"].iloc[-1] - df["close"].iloc[-3])
                    features["reversal"] = 1.0 if price_direction_old != price_direction_new and price_direction_old != 0 else 0.0
                if "high" in df.columns and "low" in df.columns and "volume" in df.columns and len(df) >= 14:
                    typical_price = (df["high"] + df["low"] + df["close"]) / 3
                    raw_money_flow = typical_price * df["volume"]
                    positive_flow = pd.Series(0, index=df.index)
                    negative_flow = pd.Series(0, index=df.index)
                    for i in range(1, len(df)):
                        if typical_price.iloc[i] > typical_price.iloc[i - 1]:
                            positive_flow.iloc[i] = raw_money_flow.iloc[i]
                        else:
                            negative_flow.iloc[i] = raw_money_flow.iloc[i]
                    positive_mf_14 = positive_flow.rolling(window=14).sum()
                    negative_mf_14 = negative_flow.rolling(window=14).sum()
                    money_flow_ratio = positive_mf_14 / negative_mf_14
                    mfi = 100 - (100 / (1 + money_flow_ratio))
                    features["mfi"] = mfi.iloc[-1] if not pd.isna(mfi.iloc[-1]) else 50
                    features["mfi_overbought"] = 1.0 if features["mfi"] > 80 else 0.0
                    features["mfi_oversold"] = 1.0 if features["mfi"] < 20 else 0.0
            features = {k: float(v) if not pd.isna(v) else 0.0 for k, v in features.items()}
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None


ranking_model = RankingModel()


def rank_opportunities(
    stock_data: Dict[str, pd.DataFrame], factor_weights: Optional[Dict[str, float]] = None, model_version: str = "ensemble"
) -> List[Dict[str, Any]]:
    """
    Rank trading opportunities based on the multi-factor model.

    Args:
        stock_data: Dictionary of stock DataFrames with features.
        factor_weights: Optional dictionary of factor weights.
        model_version: Model version to use ('ensemble', 'xgboost', 'lightgbm', 'catboost', 'rf').

    Returns:
        List of ranked stocks with scores.
    """
    model = RankingModel(factor_weights=factor_weights, model_version=model_version)
    return model.rank_stocks(stock_data)


def get_model_weights(model_version: str = "ensemble") -> Dict[str, float]:
    """
    Get the current weights used in the ranking model.

    Args:
        model_version: Model version to get weights for.

    Returns:
        Dictionary of model weights.
    """
    model = RankingModel(model_version=model_version)
    return model.hyperparams["meta"]["weights"]
