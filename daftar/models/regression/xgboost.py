"""XGBoost regression model implementation for DAFTAR-ML."""

import numpy as np
import optuna
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import shap

from daftar.models.base import BaseRegressionModel
from daftar.core.callbacks import RelativeEarlyStoppingCallback


class XGBoostRegressionModel(BaseRegressionModel):
    """XGBoost regression implementation."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit XGBoost regression model with hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        # Suppress ALL XGBoost warnings for a cleaner output
        warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True),
                'random_state': self.seed,
                'n_jobs': self.n_jobs,
                'use_label_encoder': False  # Add this to suppress warnings
            }
            
            model = XGBRegressor(**params)
            model.fit(X, y)
            y_pred = model.predict(X)
            
            if self.metric == 'mse':
                return ((y - y_pred) ** 2).mean()
            elif self.metric == 'rmse':
                return np.sqrt(((y - y_pred) ** 2).mean())
            elif self.metric == 'mae':
                return np.abs(y - y_pred).mean()
            else:  # r2
                return -1 * (1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
                
            return score

        # Set environment variables to help callbacks know which metrics to display properly
        import os
        os.environ['DAFTAR-ML_PROBLEM_TYPE'] = 'regression'
        os.environ['DAFTAR-ML_METRIC'] = self.metric
        
        # Create early stopping callback (now that env vars are set)
        early_stopping = RelativeEarlyStoppingCallback(
            patience=self.patience,
            relative_threshold=self.relative_threshold
        )
        
        # Replace Optuna's logging to show correct metric signs
        # Since we're negating metrics like r2 for optimization, we need to fix the display values
        import logging
        logging.getLogger("optuna").setLevel(logging.ERROR)  # Suppress default output
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        
        # Custom callback to intercept and reformat Optuna's output
        def log_trial_callback(study, trial):
            # For r2 metric, use absolute values for display
            if self.metric == 'r2':
                value_to_display = abs(trial.value)
                best_value_to_display = abs(study.best_value)
            else:
                value_to_display = trial.value
                best_value_to_display = study.best_value
            
            if trial.number == 0:
                print(f"[I {self._get_timestamp()}] Trial {trial.number} finished with value: {value_to_display} " + 
                      f"and parameters: {trial.params}. Best is trial {trial.number} with value: {value_to_display}.")
            else:
                print(f"[I {self._get_timestamp()}] Trial {trial.number} finished with value: {value_to_display} " + 
                      f"and parameters: {trial.params}. Best is trial {study.best_trial.number} with value: {best_value_to_display}.")
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            callbacks=[early_stopping, log_trial_callback]
        )
        
        # Store study for visualization
        self.study = study
        
        self.model = XGBRegressor(**study.best_params)
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        return self.model.predict(X)
        
    @property
    def feature_importances_(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_
        
    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Get SHAP values.
        
        Args:
            X: Feature matrix
            
        Returns:
            SHAP values
        """
        explainer = shap.TreeExplainer(self.model)
        # Create shap values - this matches the original implementation
        # and ensures the output is compatible with the viz code
        return explainer.shap_values(X)
        
    def _get_timestamp(self):
        """Get current timestamp in format used by Optuna."""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
