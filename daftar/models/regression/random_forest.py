"""Random Forest regression model implementation for DAFTAR-ML."""

import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap

from daftar.models.base import BaseRegressionModel
from daftar.core.callbacks import RelativeEarlyStoppingCallback


class RandomForestRegressionModel(BaseRegressionModel):
    """Random Forest regression implementation."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Random Forest regression model with hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                'random_state': self.seed,
                'n_jobs': self.n_jobs
            }
            
            model = RandomForestRegressor(**params)
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
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(
            objective,
            n_trials=self.n_trials,
            callbacks=[early_stopping]
        )
        
        # Store study for visualization
        self.study = study
        
        self.model = RandomForestRegressor(**study.best_params)
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
        return explainer.shap_values(X)
