"""Core visualization utilities for DAFTAR-ML."""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from daftar.viz.color_definitions import CONFUSION_MATRIX_CMAP, REGRESSION_MEAN_LINE_COLOR


def create_shap_summary(shap_values: List[np.ndarray], 
                      feature_names: List[str],
                      output_path: Path) -> None:
    """Create SHAP summary plot.
    
    Args:
        shap_values: List of SHAP values arrays for each fold
        feature_names: List of feature names
        output_path: Path to save the plot
    """
    # Combine SHAP values from all folds
    combined_values = np.concatenate(shap_values)
    
    # Create feature matrix with proper names
    X = pd.DataFrame(np.zeros((combined_values.shape[0], len(feature_names))), 
                   columns=feature_names)
    
    # Create and save plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(combined_values, X, show=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def create_feature_importance(importances: np.ndarray,
                            feature_names: List[str],
                            output_path: Path) -> None:
    """Create feature importance plot.
    
    Args:
        importances: Feature importance array (folds x features)
        feature_names: List of feature names
        output_path: Path to save the plot
    """
    # Calculate mean and std of importance across folds
    mean_importance = np.mean(importances, axis=0)
    std_importance = np.std(importances, axis=0)
    
    # Sort by mean importance
    indices = np.argsort(mean_importance)[::-1]
    
    # Select top 25 features
    top_n = min(25, len(indices))
    indices = indices[:top_n]
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': mean_importance[indices],
        'StdDev': std_importance[indices]
    })
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'],
           xerr=importance_df['StdDev'], capsize=5)
    plt.xlabel('Mean Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance (Mean ± Std)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_prediction_analysis(y_true: List,
                             y_pred: List,
                             output_path: Path) -> None:
    """Create prediction analysis plots.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        output_path: Path to save the plot
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check if values are numeric or strings (for classification)
    is_numeric = np.issubdtype(y_true.dtype, np.number) and np.issubdtype(y_pred.dtype, np.number)
    
    if is_numeric:
        # For regression tasks
        # Calculate residuals
        residuals = y_pred - y_true
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Predicted vs. Actual
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title('Predicted vs. Actual')
        
        # Plot 2: Residuals
        axes[1].scatter(y_true, residuals, alpha=0.5)
        axes[1].axhline(y=0, color=REGRESSION_MEAN_LINE_COLOR, linestyle='--')
        axes[1].set_xlabel('Actual')
        axes[1].set_ylabel('Residual (Predicted - Actual)')
        axes[1].set_title('Residual Plot')
    else:
        # For classification tasks
        from sklearn.metrics import confusion_matrix
        
        # Get unique classes
        classes = np.unique(np.concatenate((y_true, y_pred)))
        
        # Create confusion matrix plot
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        # Single plot for classification
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap=CONFUSION_MATRIX_CMAP, cbar=False,
                   xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix - Overall Performance')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
