"""Centralized color definitions for DAFTAR-ML visualizations.

This module provides all color definitions used throughout DAFTAR-ML
to ensure consistency across all visualization components.
"""

from typing import List, Union
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Binary classification colors (alternating)
CLASS_BAR_COLOR0 = "#1C0F13"          # Primary bar chart color for binary classification
CLASS_BAR_COLOR1 = "#6E7E85"          # Secondary bar chart color for binary classification
BINARY_CLASSIFICATION_COLORS = [CLASS_BAR_COLOR0, CLASS_BAR_COLOR1]

# Multiclass classification colors (for more than 2 classes)
# More distinctive colors for better differentiation
MULTICLASS_COLORS = [
    "#03045E",  
    "#023E8A",    
    "#0077B6",    
    "#0096C7",    
    "#00B4D8",    
    "#48CAE4",    
    "#90E0EF",    
    "#ADE8F4",  
    "#CAF0F8", 
    "#316395"   
]  # Add more colors as needed for more classes

# Regression visualization colors
REGRESSION_COLOR = "#1C0F13"           # Single color for regression histograms
REGRESSION_HIST_COLOR = "#1C0F13"      # Histogram fill color
REGRESSION_HIST_ALPHA = 0.8            # Histogram transparency
REGRESSION_MEAN_LINE_COLOR = "r"       # Mean line color

# Compare train/test visualization colors
TRAIN_HIST_COLOR = "#70e4ef"           # Train set histogram color
TEST_HIST_COLOR = "#dfdf20"            # Test set histogram color 
HIST_ALPHA = 0.8                       # Transparency for compare histograms

# Feature importance colors
FEATURE_IMPORTANCE_BAR_COLOR = "#968FF3"  # Color for feature importance bars
FEATURE_IMPORTANCE_BAR_BG = "#E6E6E6"     # Background color for feature importance plots

# SHAP plot colors
SHAP_POSITIVE_COLOR = "#AB0264"        # Color for positive SHAP values (increases prediction)
SHAP_NEGATIVE_COLOR = "#3E95B5"        # Color for negative SHAP values (decreases prediction)
SHAP_BG_COLOR = "#F0F0F0"              # Background color for SHAP plots
SHAP_FEATURE_COLORS = ["#1E88E5", "#ffffff", "#ff0d57"]  # Colors for SHAP beeswarm plot

# Density plot colors (previously hardcoded in predictions.py)
DENSITY_ACTUAL_COLOR = "#00BFC4"       # Color for actual values in density plots
DENSITY_PREDICTED_COLOR = "#F8766D"    # Color for predicted values in density plots
DENSITY_ALPHA = 0.5                    # Transparency for density plots

# Confusion matrix colors
CONFUSION_MATRIX_CMAP = "Blues"        # Default colormap for confusion matrices

# General plot cycling colors (used for axes.prop_cycle)
PLOT_CYCLE_COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']

# Get a pre-defined SHAP feature colormap
def get_shap_feature_cmap():
    """Get the colormap used for SHAP feature plots."""
    return LinearSegmentedColormap.from_list("feature_cmap", SHAP_FEATURE_COLORS)

# Correlation plot colors - using coolwarm for regression correlation plots
CORRELATION_CMAP = plt.cm.coolwarm     # Colormap for correlation plots

# Train/Test split colors (using seaborn pastel palette indices)
TRAIN_COLOR_IDX = 0  # First color in pastel palette
TEST_COLOR_IDX = 1   # Second color in pastel palette

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