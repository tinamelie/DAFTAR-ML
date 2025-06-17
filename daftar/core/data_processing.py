"""Data loading and preparation functions for DAFTAR-ML."""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

def prepare_data(config) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and prepare data for analysis.
    
    Args:
        config: Configuration object with input_file
        
    Returns:
        Tuple of (feature matrix, target vector, feature names)
    """
    # Load data
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
    
    # Store original data with IDs for later reference (BEFORE any modifications)
    original_data = None
    if config.id_column and config.id_column in data.columns:
        original_data = data.copy().reset_index(drop=True)
        # Remove ID column from working data
        data = data.drop(columns=[config.id_column])
    
    # Split features and target
    y = data[config.target].values.copy()  # Make a copy to avoid modifying original
    X = data.drop(columns=[config.target])
    
    # Apply label encoding for classification tasks
    config.label_encoder = None  # Initialize to None
    if hasattr(config, 'problem_type') and config.problem_type == 'classification':
        # Check if target contains non-numeric values OR string-like numeric values
        if (not pd.api.types.is_numeric_dtype(data[config.target]) or 
            data[config.target].dtype == 'object' or
            data[config.target].dtype.name in ['string', 'category']):
            
            logger.info(f"Encoding labels for classification task")
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Store label encoder for later use
            config.label_encoder = label_encoder
            print(f"Classes: {list(label_encoder.classes_)}")
            
            # Use encoded values for training
            y = y_encoded
    
    # Get feature names
    if feature_names is None:
        feature_names = X.columns.tolist()
    
    # Convert to numpy arrays
    X = X.values
    
    return X, y, feature_names, original_data
