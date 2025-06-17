"""Base model interfaces for DAFTAR-ML.

This module provides abstract base classes for regression and classification models
to ensure a consistent interface across all model implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import optuna
from daftar.core.callbacks import RelativeEarlyStoppingCallback


class BaseModel(ABC):
    """Base abstract class for all DAFTAR-ML models."""
    
    def __init__(self, metric: str, n_trials: int, n_jobs: int,
                 patience: int = 50, relative_threshold: float = 1e-6,
                 seed: Optional[int] = None):
        """Initialize model.
        
        Args:
            metric: Metric to optimize
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs
            patience: Number of trials to wait without improvement
            relative_threshold: Relative improvement threshold
            seed: Random seed
        """
        self.metric = metric
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.patience = patience
        self.relative_threshold = relative_threshold
        self.seed = seed
        self.model = None
        self.study = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit model to data."""
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
        
    @property
    @abstractmethod
    def feature_importances_(self) -> np.ndarray:
        """Get feature importance."""
        pass
        
    @abstractmethod
    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Get SHAP values."""
        pass


class BaseRegressionModel(BaseModel):
    """Base class for all regression models."""
    
    def __init__(self, metric: str, n_trials: int, n_jobs: int,
                 patience: int = 50, relative_threshold: float = 1e-6,
                 seed: Optional[int] = None):
        """Initialize regression model.
        
        Args:
            metric: Metric to optimize ('mse', 'rmse', 'mae', 'r2')
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs
            patience: Number of trials to wait without improvement
            relative_threshold: Relative improvement threshold
            seed: Random seed
        """
        super().__init__(
            metric=metric, n_trials=n_trials, n_jobs=n_jobs,
            patience=patience, relative_threshold=relative_threshold,
            seed=seed
        )


class BaseClassificationModel(BaseModel):
    """Base class for all classification models."""
    
    def __init__(self, metric: str, n_trials: int, n_jobs: int,
                 patience: int = 50, relative_threshold: float = 1e-6,
                 seed: Optional[int] = None):
        """Initialize classification model.
        
        Args:
            metric: Metric to optimize ('accuracy', 'f1', 'roc_auc')
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs
            patience: Number of trials to wait without improvement
            relative_threshold: Relative improvement threshold
            seed: Random seed
        """
        super().__init__(
            metric=metric, n_trials=n_trials, n_jobs=n_jobs,
            patience=patience, relative_threshold=relative_threshold,
            seed=seed
        )
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities for classification models."""
        pass
