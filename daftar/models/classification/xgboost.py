"""XGBoost classification model implementation for DAFTAR-ML."""

import numpy as np
import optuna
import warnings
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import shap

from daftar.models.base import BaseClassificationModel
from daftar.core.callbacks import RelativeEarlyStoppingCallback


class XGBoostClassificationModel(BaseClassificationModel):
    """XGBoost classification implementation."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit XGBoost classification model with hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        # Suppress ALL XGBoost warnings for a cleaner output
        warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
        
        # Handle string labels by encoding them
        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y)
        
        # Determine if this is a binary or multi-class problem
        n_classes = len(np.unique(y))
        if n_classes <= 2:
            xgb_objective = 'binary:logistic'
        else:
            xgb_objective = 'multi:softprob'
            
        def objective_func(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True),
                'objective': xgb_objective,
                'eval_metric': 'logloss',
                'random_state': self.seed,
                'use_label_encoder': False,
                'n_jobs': self.n_jobs
            }
            
            # Add num_class parameter for multi-class problems
            if xgb_objective == 'multi:softprob':
                params['num_class'] = n_classes
            
            model = XGBClassifier(**params)
            model.fit(X, y_encoded)
            y_pred = model.predict(X)
            
            if self.metric == 'accuracy':
                return -accuracy_score(y_encoded, y_pred)  # Negative for minimization
            elif self.metric == 'f1':
                return -f1_score(y_encoded, y_pred)  # Negative for minimization
            elif self.metric == 'roc_auc':
                if xgb_objective == 'binary:logistic':
                    y_pred_proba = model.predict_proba(X)[:, 1]
                    return -roc_auc_score(y_encoded, y_pred_proba)  # Negative for minimization
                else:
                    # For multi-class, we can't use regular ROC AUC
                    return -accuracy_score(y_encoded, y_pred)  # Fall back to accuracy
            else:
                return -accuracy_score(y_encoded, y_pred)  # Default to accuracy
                
            return score

        # Set environment variables to help callbacks know which metrics to display properly
        import os
        os.environ['DAFTAR-ML_PROBLEM_TYPE'] = 'classification'
        os.environ['DAFTAR-ML_METRIC'] = self.metric
        
        # Create early stopping callback
        early_stopping = RelativeEarlyStoppingCallback(
            patience=self.patience,
            relative_threshold=self.relative_threshold
        )
        
        # Replace Optuna's logging to show correct metric signs
        # Since we're negating the metrics for optimization, we need to fix the display values
        import logging
        logging.getLogger("optuna").setLevel(logging.ERROR)  # Suppress default output
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        
        # Custom callback to intercept and reformat Optuna's output
        def log_trial_callback(study, trial):
            # For classification metrics, use absolute values for display
            value_to_display = abs(trial.value)  # Use absolute value for display
            best_value_to_display = abs(study.best_value)
            
            if trial.number == 0:
                print(f"[I {self._get_timestamp()}] Trial {trial.number} finished with value: {value_to_display} " + 
                      f"and parameters: {trial.params}. Best is trial {trial.number} with value: {value_to_display}.")
            else:
                print(f"[I {self._get_timestamp()}] Trial {trial.number} finished with value: {value_to_display} " + 
                      f"and parameters: {trial.params}. Best is trial {study.best_trial.number} with value: {best_value_to_display}.")
        
        study.optimize(
            objective_func,
            n_trials=self.n_trials,
            callbacks=[early_stopping, log_trial_callback]
        )
        
        # Store study for visualization
        self.study = study
        
        best_params = study.best_params
        best_params.update({
            'objective': xgb_objective,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': self.seed
        })
        
        # Add num_class parameter for multi-class problems
        if xgb_objective == 'multi:softprob':
            best_params['num_class'] = n_classes
        
        self.model = XGBClassifier(**best_params)
        self.model.fit(X, y_encoded)
        
        # Store the LabelEncoder for later use
        self.classes_ = self.le.classes_
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted classes
        """
        # Get numeric predictions and convert back to original labels
        y_pred_num = self.model.predict(X)
        return self.le.inverse_transform(y_pred_num.astype(int))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
        """
        # Return probabilities without modifying them
        # The probabilities already match the order of self.le.classes_
        return self.model.predict_proba(X)
        
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
        
    def _get_timestamp(self):
        """Get current timestamp in format used by Optuna."""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
