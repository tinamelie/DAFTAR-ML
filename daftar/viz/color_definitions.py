"""Centralized color definitions for DAFTAR-ML visualizations.

This module provides all color definitions used throughout DAFTAR-ML
to ensure consistency across all visualization components.
"""

from typing import List, Union
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Binary classification colors (alternating)
CLASS_BAR_COLOR0 = "#225ea8"          # Primary bar chart color for binary classification
CLASS_BAR_COLOR1 = "#c7e9b4"          # Secondary bar chart color for binary classification
BINARY_CLASSIFICATION_COLORS = [CLASS_BAR_COLOR0, CLASS_BAR_COLOR1]

# Multiclass classification colors (for more than 2 classes)
# More distinctive colors for better differentiation
MULTICLASS_COLORS = [
    "#0c2c84",
    "#225ea8",
    "#1d91c0",
    "#41b6c4",
    "#7fcdbb",
    "#c7e9b4",
    "#edf8b1",
    "#ffffd9",
]  # Add more colors as needed for more classes

# Regression visualization colors
REGRESSION_COLOR = "#0c2c84"           # Color for regression visualizations and histograms
REGRESSION_MEAN_LINE_COLOR = "#8a994e"       # Mean line color
REGRESSION_HIST_ALPHA = 0.6            # Histogram transparency

# Compare train/test visualization colors
TRAIN_HIST_COLOR = "#70e4ef"           # Train set histogram color
TEST_HIST_COLOR = "#dfdf20"            # Test set histogram color 
HIST_ALPHA = 0.8                       # Transparency for compare histograms

# Background colors for all plots
HISTOGRAM_BG_COLOR = "#F8F8F8"         # Background color for regression histogram plots
CLASSIFICATION_BAR_BG_COLOR = "#F8F8F8"   # Background color for classification bar charts
DENSITY_PLOT_BG_COLOR = "#FAFAFA"      # Background color for density plots
CORRELATION_BAR_BG = "#F7F7F7"         # Background color for correlation bar plots
FEATURE_IMPORTANCE_BAR_BG = "#FAFAFA"     # Background color for feature importance plots
SHAP_BG_COLOR = "#FAFAFA"              # Background color for SHAP plots

# Feature importance colors
FEATURE_IMPORTANCE_BAR_COLOR = "#018f9c"  # Color for feature importance bars

# SHAP plot colors
SHAP_POSITIVE_COLOR = "#691a4c"        # Color for positive SHAP values (increases prediction)
SHAP_NEGATIVE_COLOR = "#32748E"        # Color for negative SHAP values (decreases prediction)

# Density plot colors (previously hardcoded in predictions.py)
DENSITY_ACTUAL_COLOR = "#00BFC4"       # Color for actual values in density plots
DENSITY_PREDICTED_COLOR = "#F8766D"    # Color for predicted values in density plots
DENSITY_ALPHA = 0.5                    # Transparency for density plots

# Confusion matrix colors
CONFUSION_MATRIX_CMAP = "Blues"        # Default colormap for confusion matrices
CONFUSION_MATRIX_LINEWIDTH = 2       # Line width for confusion matrix grid
CONFUSION_MATRIX_LINECOLOR = "black"   # Line color for confusion matrix grid

# Correlation plot colors
CORRELATION_CMAP = plt.cm.twilight_shifted     # Colormap for correlation plots

def get_color_palette(is_classification: bool, class_count: int = None) -> Union[str, List[str]]:
    """Get appropriate color palette based on task type and class count.
    
    Args:
        is_classification: Whether the task is classification
        class_count: Number of classes for classification tasks
        
    Returns:
        Color palette (seaborn palette name or list of colors)
    """
    if is_classification:
        if class_count is None or class_count <= 2:
            # Binary classification - use our predefined colors
            return BINARY_CLASSIFICATION_COLORS
        else:
            # Multiclass classification - use our predefined multiclass colors
            return MULTICLASS_COLORS[:min(class_count, len(MULTICLASS_COLORS))]
    else:
        # Regression - use our predefined regression color
        return REGRESSION_COLOR

def get_train_test_colors():
    """Get colors for train/test split visualizations.
    
    Returns:
        Dictionary with train and test colors
    """
    return {
        "Train": TRAIN_HIST_COLOR,
        "Test": TEST_HIST_COLOR
    } 