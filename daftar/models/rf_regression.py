"""Random Forest regression model implementation for DAFTAR-ML."""

import numpy as np
import optuna
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import shap

from daftar.models.base import BaseRegressionModel
from daftar.core.callbacks import RelativeEarlyStoppingCallback
from daftar.models.hyperparams import get_hyperparameter_space


class RandomForestRegressionModel(BaseRegressionModel):
    """Random Forest regression implementation."""
    
    def _get_timestamp(self):
        """Format timestamp for logging."""
        return datetime.now().strftime("%H:%M:%S")
    
    def fit(self, X: np.ndarray, y: np.ndarray, inner_cv=None) -> None:
        """Fit Random Forest regression model with hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target vector
            inner_cv: Cross-validation splitter for inner folds (optional)
        """
        def objective_func(trial):
            # Get hyperparameters from the centralized configuration
            params = get_hyperparameter_space(trial, 'random_forest', 'regression')
            
            # Add model-specific parameters that aren't part of the search space
            params['random_state'] = self.seed
            
            # If inner CV is provided, use it to evaluate hyperparameters
            if inner_cv is not None:
                # Evaluate on ALL inner folds and average
                train_scores = []
                val_scores = []
                for train_idx, val_idx in inner_cv.split(X, y):
                    # Split data into training and validation sets using inner CV indices
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train on inner training set
                    model = RandomForestRegressor(**params)
                    model.fit(X_train, y_train)
                    
                    # Predict on both training and validation sets
                    y_train_pred = model.predict(X_train)
                    y_val_pred = model.predict(X_val)
                    
                    # Calculate separate training and validation metrics
                    if self.metric == 'mse':
                        train_scores.append(((y_train - y_train_pred) ** 2).mean())
                        val_scores.append(((y_val - y_val_pred) ** 2).mean())
                    elif self.metric == 'rmse':
                        train_scores.append(np.sqrt(((y_train - y_train_pred) ** 2).mean()))
                        val_scores.append(np.sqrt(((y_val - y_val_pred) ** 2).mean()))
                    elif self.metric == 'mae':
                        train_scores.append(np.abs(y_train - y_train_pred).mean())
                        val_scores.append(np.abs(y_val - y_val_pred).mean())
                    else:  # r2
                        # Calculate R2 scores (higher is better) - return positive values for maximize
                        train_r2 = 1 - ((y_train - y_train_pred) ** 2).sum() / ((y_train - y_train.mean()) ** 2).sum()
                        val_r2 = 1 - ((y_val - y_val_pred) ** 2).sum() / ((y_val - y_val.mean()) ** 2).sum()
                        train_scores.append(train_r2)
                        val_scores.append(val_r2)
                mean_train = float(np.mean(train_scores))
                mean_val = float(np.mean(val_scores))
                trial.set_user_attr('train_metric', mean_train)
                return mean_val

        # Create early stopping callback
        early_stopping = RelativeEarlyStoppingCallback(
            patience=self.patience,
            relative_threshold=self.relative_threshold
        )
        
        # Run optimization with proper direction
        if self.metric == 'r2':
            study = optuna.create_study(direction='maximize')
        else:  # mse, rmse, mae
            study = optuna.create_study(direction='minimize')
        
        # Simple callback without convoluted display value conversions
        def callback(study, trial):
            timestamp = self._get_timestamp()
            # Call the early stopping callback
            early_stopping(study, trial)
                      
        study.optimize(
            objective_func,
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
        
