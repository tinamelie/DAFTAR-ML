"""Random Forest classification model implementation for DAFTAR-ML."""

import numpy as np
import optuna
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import shap

from daftar.models.base import BaseClassificationModel
from daftar.core.callbacks import RelativeEarlyStoppingCallback
from daftar.models.hyperparams import get_hyperparameter_space


class RandomForestClassificationModel(BaseClassificationModel):
    """Random Forest classification implementation."""
    
    def _get_timestamp(self):
        """Format timestamp for logging."""
        return datetime.now().strftime("%H:%M:%S")
    
    def fit(self, X: np.ndarray, y: np.ndarray, inner_cv=None) -> None:
        """Fit Random Forest classification model with hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target vector
            inner_cv: Cross-validation splitter for inner folds (optional)
        """
        def objective_func(trial):
            # Get hyperparameters from the centralized configuration
            params = get_hyperparameter_space(trial, 'random_forest', 'classification')
            
            # Add model-specific parameters that aren't part of the search space
            params['random_state'] = self.seed
            params['n_jobs'] = self.n_jobs
            
            # If inner CV is provided, use it to evaluate hyperparameters
            if inner_cv is not None:
                # Evaluate hyperparameters on ALL inner folds and average
                train_scores = []
                val_scores = []
                for train_idx, val_idx in inner_cv.split(X, y):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    model = RandomForestClassifier(**params)
                    model.fit(X_train, y_train)
                    
                    y_train_pred = model.predict(X_train)
                    y_val_pred = model.predict(X_val)
                    y_train_proba = model.predict_proba(X_train)
                    y_val_proba = model.predict_proba(X_val)
                    
                    if self.metric == 'accuracy':
                        train_scores.append(accuracy_score(y_train, y_train_pred))
                        val_scores.append(accuracy_score(y_val, y_val_pred))
                    elif self.metric == 'f1':
                        if len(np.unique(y)) > 2:
                            train_scores.append(f1_score(y_train, y_train_pred, average='weighted'))
                            val_scores.append(f1_score(y_val, y_val_pred, average='weighted'))
                        else:
                            train_scores.append(f1_score(y_train, y_train_pred))
                            val_scores.append(f1_score(y_val, y_val_pred))
                    elif self.metric == 'roc_auc':
                        if len(np.unique(y)) > 2:
                            train_scores.append(roc_auc_score(y_train, y_train_proba, multi_class='ovr'))
                            val_scores.append(roc_auc_score(y_val, y_val_proba, multi_class='ovr'))
                        else:
                            train_scores.append(roc_auc_score(y_train, y_train_proba[:, 1]))
                            val_scores.append(roc_auc_score(y_val, y_val_proba[:, 1]))
                
                mean_train = float(np.mean(train_scores)) if train_scores else 0.0
                mean_val = float(np.mean(val_scores)) if val_scores else 0.0
                trial.set_user_attr('train_metric', mean_train)
                return mean_val

        # Create early stopping callback
        early_stopping = RelativeEarlyStoppingCallback(
            patience=self.patience,
            relative_threshold=self.relative_threshold
        )
        
        # Run optimization with proper direction - accuracy, f1, roc_auc should be maximized
        study = optuna.create_study(direction='maximize')
        
        # Simple callback without any convoluted conversions
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
        
        # Get best params and fit final model
        self.model = RandomForestClassifier(**study.best_params)
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted classes
        """
        return self.model.predict(X)
        
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
        return self.model.classes_
        
    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Get SHAP values.
        
        Args:
            X: Feature matrix
            
        Returns:
            SHAP values
        """
        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_values(X)
    