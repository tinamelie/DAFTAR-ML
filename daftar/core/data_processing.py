"""Data loading and preparation functions for DAFTAR-ML."""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def prepare_data(config) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and prepare data for analysis.
    
    Args:
        config: Configuration object with input_file
        
    Returns:
        Tuple of (feature matrix, target vector, feature names)
    """
    # Load data
    logger.info(f"Loading data from {config.input_file}")
    try:
        data = pd.read_csv(config.input_file)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    
    # Initialize dataset
    return init_dataset(data, config)


def init_dataset(
    data: pd.DataFrame, config, feature_names: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Initialize the dataset.
    
    Args:
        data: Input DataFrame with features and target
        config: Configuration object with target and id_column information
        feature_names: Optional list of feature names
        
    Returns:
        Tuple of (X, y, feature_names), and updates original_data in the calling context
    """
    # Get target and ID column
    if config.target not in data.columns:
        raise ValueError(f"Target column '{config.target}' not found in data")
    
    original_data = None
    # Store original data with IDs for later reference
    if config.id_column and config.id_column in data.columns:
        # Make a copy of the original data with index set to match the dataset
        original_data = data.copy().reset_index(drop=True)
        # Remove ID column from working data
        data = data.drop(columns=[config.id_column])
    
    # Split features and target
    y = data[config.target].values
    X = data.drop(columns=[config.target])
    
    # Get feature names
    if feature_names is None:
        feature_names = X.columns.tolist()
    
    # Convert to numpy arrays
    X = X.values
    
    # Note: All transformations are now applied during preprocessing
    
    return X, y, feature_names, original_data
