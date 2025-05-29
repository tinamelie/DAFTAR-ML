"""Random Forest regression model implementation for DAFTAR-ML."""

import numpy as np
import optuna
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap

from daftar.models.base import BaseRegressionModel
from daftar.core.callbacks import RelativeEarlyStoppingCallback
from daftar.models.hyperparams import get_hyperparameter_space


class RandomForestRegressionModel(BaseRegressionModel):
    """Random Forest regression implementation."""
    
    def _get_timestamp(self):
        """Format timestamp for logging."""
        return datetime.now().strftime("%H:%M:%S")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Random Forest regression model with hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        def objective(trial):
            # Get hyperparameters from the centralized configuration
            params = get_hyperparameter_space(trial, 'random_forest', 'regression')
            
            # Add model-specific parameters that aren't part of the search space
            params['random_state'] = self.seed
            params['n_jobs'] = self.n_jobs
            
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
        
        # Override Optuna's default callbacks to use our custom reporting
        def callback(study, trial):
            # For metrics that should naturally be positive (r2), convert negatives to positives
            # For metrics that should be minimized (mse, rmse, mae), preserve the original sign
            
            # Get raw values
            raw_trial_value = trial.value
            raw_best_value = study.best_value
            
            # Convert to display values with correct sign
            if self.metric == 'r2':
                # For r2, show the POSITIVE value (flip the negative optimization value)
                trial_display_value = -raw_trial_value
                best_display_value = -raw_best_value
            else:
                # For MSE, RMSE, MAE - use actual values (already positive)
                trial_display_value = raw_trial_value
                best_display_value = raw_best_value
            
            # Format timestamp for logging
            timestamp = self._get_timestamp()
            
            # Print appropriate message - Make sure values are positive for display
            if trial.number == study.best_trial.number:
                # Make sure we're not showing negative values for minimization metrics (MSE, RMSE, MAE)
                display_value = abs(trial_display_value) if self.metric in ['mse', 'rmse', 'mae'] else trial_display_value
                print(f"[I {timestamp}] Trial {trial.number} finished with value: {display_value} " +
                      f"and parameters: {trial.params}. Best is trial {trial.number} with value: {display_value}.")
            else:
                # Make sure we're not showing negative values for minimization metrics (MSE, RMSE, MAE)
                display_value = abs(trial_display_value) if self.metric in ['mse', 'rmse', 'mae'] else trial_display_value
                display_best = abs(best_display_value) if self.metric in ['mse', 'rmse', 'mae'] else best_display_value
                print(f"[I {timestamp}] Trial {trial.number} finished with value: {display_value} " +
                      f"and parameters: {trial.params}. Best is trial {study.best_trial.number} with value: {display_best}.")
                      
            # Call the early stopping callback
            early_stopping(study, trial)
                      
        study.optimize(
            objective,
            n_trials=self.n_trials,
            callbacks=[callback]
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
        
    def shap_interaction_values(self, X: np.ndarray) -> np.ndarray:
        """Get SHAP interaction values.
        
        Args:
            X: Feature matrix
            
        Returns:
            SHAP interaction values matrix with shape (n_samples, n_features, n_features)
        """
        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_interaction_values(X)
