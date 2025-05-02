"""Utility module for handling warnings."""

import warnings

def suppress_xgboost_warnings():
    """Suppress all XGBoost warnings globally."""
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
    # This specifically targets the "use_label_encoder" warning
    warnings.filterwarnings("ignore", message=".*Parameters: { \"use_label_encoder\" }.*")
