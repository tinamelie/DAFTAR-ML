"""Classification-specific visualizations for DAFTAR-ML."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from daftar.viz.common import set_plot_style, save_plot, create_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc
from daftar.viz.color_definitions import CONFUSION_MATRIX_CMAP, CONFUSION_MATRIX_LINEWIDTH, CONFUSION_MATRIX_LINECOLOR


def generate_confusion_matrix(true_values, predictions, output_path, title="Confusion Matrix", metric=None):
    """Generate a confusion matrix plot.
    
    Args:
        true_values: True labels
        predictions: Predicted labels
        output_path: Path to save the plot
        title: Title of the plot
        metric: Optional metric to display (name and value)
    """
    # This would be extracted from the predictions.py module's generate_confusion_matrix function
    set_plot_style()
    fig, ax = create_subplots(figsize=(10, 8))
    
    # Get unique classes preserving the original order
    classes = np.unique(np.concatenate((true_values, predictions)))
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_values, predictions, labels=classes)
    
    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap=CONFUSION_MATRIX_CMAP, cbar=False,
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 16}, ax=ax,
                linewidths=CONFUSION_MATRIX_LINEWIDTH, linecolor=CONFUSION_MATRIX_LINECOLOR)
    
    # Increase font sizes for better readability
    plt.title(title, fontsize=18, pad=20)
    plt.ylabel("True Label", fontsize=16, labelpad=15)
    plt.xlabel("Predicted Label", fontsize=16, labelpad=15)
    
    # Set tick size
    ax.tick_params(labelsize=14)
    
    # Add metric if provided
    if metric is not None:
        metric_name, metric_value = metric
        metric_text = f"{metric_name}: {metric_value:.4f}"
        plt.figtext(0.02, 0.02, metric_text, fontsize=14, 
                    bbox={"facecolor": "white", "alpha": 0.9, "pad": 10, "edgecolor": "#cccccc"})
    
    # Save the plot
    save_plot(fig, output_path)


def create_roc_curve(true_values, probabilities, output_path, classes=None):
    """Create ROC curve for classification predictions.
    
    Args:
        true_values: Array of true target values
        probabilities: Array of predicted probabilities
        output_path: Path to save the plot
        classes: List of class names
    """
    # This would be implemented based on sklearn's ROC curve functionality
    pass 