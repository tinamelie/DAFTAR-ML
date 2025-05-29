"""XGBoost classification model implementation for DAFTAR-ML."""

import numpy as np
import optuna
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import shap

from daftar.models.base import BaseClassificationModel
from daftar.core.callbacks import RelativeEarlyStoppingCallback
from daftar.utils.warnings import suppress_xgboost_warnings
from daftar.models.hyperparams import get_hyperparameter_space

# Suppress specific XGBoost warnings globally
suppress_xgboost_warnings()


class XGBoostClassificationModel(BaseClassificationModel):
    """XGBoost classification implementation."""
    
    def _get_timestamp(self):
        """Format timestamp for logging."""
        return datetime.now().strftime("%H:%M:%S")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit XGBoost classification model with hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        # Convert to XGBoost's expected format
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        
        # Create label encoder to transform classes to integers
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        def objective_func(trial):
            # Get hyperparameters from the centralized configuration
            params = get_hyperparameter_space(trial, 'xgboost', 'classification')
            
            # Add model-specific parameters that aren't part of the search space
            params['random_state'] = self.seed
            
            model = xgb.XGBClassifier(**params)
            model.fit(X, y_encoded)
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)
            
            # Choose the metric based on the optimization criterion
            if self.metric == 'accuracy':
                return -accuracy_score(y_encoded, y_pred)  # Negative for minimization
            elif self.metric == 'f1':
                if len(np.unique(y)) > 2:  # Multi-class
                    return -f1_score(y_encoded, y_pred, average='weighted')  # Negative for minimization
                else:  # Binary
                    return -f1_score(y_encoded, y_pred)  # Negative for minimization
            elif self.metric == 'roc_auc':
                if len(np.unique(y)) > 2:  # Multi-class
                    return -roc_auc_score(y_encoded, y_pred_proba, multi_class='ovr')  # Negative for minimization
                else:  # Binary
                    return -roc_auc_score(y_encoded, y_pred_proba[:, 1])  # Negative for minimization
            
            return 0  # Fallback - should never reach here
        
        # Set environment variables to help callbacks know which metrics to display properly
        import os
        os.environ['DAFTAR-ML_PROBLEM_TYPE'] = 'classification' 
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
            # For metrics that should naturally be positive (accuracy, f1, roc_auc), convert negatives to positives
            # For other metrics, preserve the original sign
            metrics_to_flip = ['accuracy', 'f1', 'roc_auc']
            
            # Get raw values
            raw_trial_value = trial.value
            raw_best_value = study.best_value
            
            # Convert to display values with correct sign
            if self.metric in metrics_to_flip:
                # For these metrics, show the POSITIVE value (flip the negative optimization value)
                trial_display_value = -raw_trial_value
                best_display_value = -raw_best_value
            else:
                # For all other metrics, preserve the original sign
                trial_display_value = raw_trial_value
                best_display_value = raw_best_value
            
            # Format timestamp for logging
            timestamp = self._get_timestamp()
            
            # Print appropriate message
            if trial.number == study.best_trial.number:
                # Always show positive values for classification metrics (they're always maximized)
                display_value = abs(trial_display_value) if trial_display_value < 0 else trial_display_value
                print(f"[I {timestamp}] Trial {trial.number} finished with value: {display_value} " +
                      f"and parameters: {trial.params}. Best is trial {trial.number} with value: {display_value}.")
            else:
                # Always show positive values for classification metrics (they're always maximized)
                display_value = abs(trial_display_value) if trial_display_value < 0 else trial_display_value
                display_best = abs(best_display_value) if best_display_value < 0 else best_display_value
                print(f"[I {timestamp}] Trial {trial.number} finished with value: {display_value} " +
                      f"and parameters: {trial.params}. Best is trial {study.best_trial.number} with value: {display_best}.")
                      
            # Call the early stopping callback
            early_stopping(study, trial)
                      
        study.optimize(
            objective_func,
            n_trials=self.n_trials,
            callbacks=[callback]
        )
        
        # Store study for visualization
        self.study = study
        
        # Get best params and fit final model
        best_params = study.best_params.copy()
        self.model = xgb.XGBClassifier(**best_params)
        self.model.fit(X, y_encoded)
        self.label_encoder = le
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted classes
        """
        y_pred_numeric = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_numeric)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        return self.model.predict_proba(X)
        
    @property
    def feature_importances_(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_
    
    @property  
    def classes_(self) -> np.ndarray:
        """Get class labels."""
        return self.label_encoder.classes_
        
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
            SHAP interaction values matrix with shape (n_classes, n_samples, n_features, n_features)
            For binary classification, this will be a list with two elements.
        """
        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_interaction_values(X)
