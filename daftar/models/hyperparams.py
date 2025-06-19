#!/usr/bin/env python
"""DAFTAR-ML Hyperparameter Configuration.

This module provides utilities for loading and managing hyperparameter search spaces
for all models in DAFTAR-ML. It centralizes all hyperparameter configuration into a
single module, allowing users to customize the search spaces through a YAML file.

Usage:
    # Import hyperparameter utility:
    from daftar.models.hyperparams import get_hyperparameter_space
    
    # Get hyperparameter space for a specific model:
    params = get_hyperparameter_space(trial, 'random_forest', 'regression')
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


# Default configuration file path
DEFAULT_CONFIG_PATH = Path(__file__).parent / 'hyperparams.yaml'

# Cache for loaded hyperparameters
_hyperparams_cache = None


def load_hyperparams(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load hyperparameters from the configuration file.
    
    Args:
        config_path: Path to the configuration file. If None, uses the default path.
        
    Returns:
        Dictionary with all hyperparameter configurations
    """
    global _hyperparams_cache
    
    # Return cached configuration if available
    if _hyperparams_cache is not None:
        return _hyperparams_cache
    
    # Use default path if none specified
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    # Load from file if it exists, otherwise use defaults
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                _hyperparams_cache = yaml.safe_load(f)
            return _hyperparams_cache
        except Exception as e:
            print(f"Error loading hyperparameters configuration: {e}")
            print(f"Falling back to default hyperparameters.")
    
    # Fallback to default hyperparameters
    _hyperparams_cache = _get_default_hyperparams()
    return _hyperparams_cache


def _get_default_hyperparams() -> Dict[str, Any]:
    """
    Get default hardcoded hyperparameters as a fallback.
    
    Returns:
        Dictionary with default hyperparameters
    """
    return {
        'random_forest': {
            'n_estimators': {'min': 100, 'max': 1000},
            'max_depth': {'min': 3, 'max': 20},
            'min_samples_leaf': {'min': 1, 'max': 10},
            'max_features': {'min': 0.1, 'max': 0.8},
            'min_samples_split': {'min': 2, 'max': 10},
            'criterion': {
                'classification': ['gini', 'entropy'],
                'regression': ['squared_error', 'absolute_error']
            }
        },
        'xgboost': {
            'n_estimators': {'min': 100, 'max': 1000},
            'max_depth': {'min': 3, 'max': 20},
            'learning_rate': {'min': 0.01, 'max': 0.4},
            'subsample': {'min': 0.5, 'max': 1.0},
            'colsample_bytree': {'min': 0.5, 'max': 1.0},
            'min_child_weight': {'min': 1, 'max': 10},
            'gamma': {'min': 0, 'max': 0.5}
        }
    }


def get_hyperparameter_space(trial, model_type: str, task_type: str = None) -> Dict[str, Any]:
    """
    Get hyperparameter space for a specific model type.
    
    Args:
        trial: Optuna trial object
        model_type: Model type ('random_forest' or 'xgboost')
        task_type: Task type ('regression' or 'classification') - used for task-specific parameters
        
    Returns:
        Dictionary with hyperparameters for Optuna trials
    """
    # Load hyperparameters
    hyperparams = load_hyperparams()
    
    # Get specific hyperparameter space
    try:
        space = hyperparams[model_type]
    except KeyError:
        raise ValueError(f"No hyperparameter space found for {model_type}")
    
    # Create parameter dictionary for trial
    params = {}
    
    # Map hyperparameters to trial suggestions
    for param_name, param_range in space.items():
        if param_name in ['n_estimators', 'max_depth', 'min_samples_split',
                        'min_samples_leaf', 'min_child_weight']:
            # Integer parameters
            params[param_name] = trial.suggest_int(
                param_name, 
                param_range['min'], 
                param_range['max']
            )
        elif param_name == 'bootstrap':
            # Boolean categorical parameter
            params[param_name] = trial.suggest_categorical(
                param_name,
                param_range['choices']
            )
        elif param_name == 'criterion':
            # Task-specific categorical parameter
            if task_type and task_type in param_range:
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_range[task_type]
                )
            # If task_type is not specified or not found, skip this parameter
        else:
            # Float parameters
            params[param_name] = trial.suggest_float(
                param_name, 
                param_range['min'], 
                param_range['max']
            )
    
    return params