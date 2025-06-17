"""Model implementations for DAFTAR-ML."""

# Import base models
from daftar.models.base import BaseModel, BaseClassificationModel, BaseRegressionModel

# Import model implementations from flattened structure
from daftar.models.xgboost_classification import XGBoostClassificationModel
from daftar.models.xgboost_regression import XGBoostRegressionModel
from daftar.models.rf_classification import RandomForestClassificationModel
from daftar.models.rf_regression import RandomForestRegressionModel

# Make these classes available at the package level
__all__ = [
    'BaseModel',
    'BaseClassificationModel',
    'BaseRegressionModel',
    'XGBoostClassificationModel',
    'XGBoostRegressionModel',
    'RandomForestClassificationModel',
    'RandomForestRegressionModel',
]
