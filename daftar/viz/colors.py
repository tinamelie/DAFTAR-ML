#!/usr/bin/env python
"""DAFTAR-ML Color System.

This module provides all color definitions and utilities for DAFTAR-ML visualizations.
It centralizes all color-related functionality into a single module, including:
- Loading colors from YAML configuration
- Color constants for all visualizations
- Utility functions for color management
- Visualization tools for color display

Usage:
    # Import constants directly:
    from daftar.viz.colors import SHAP_POSITIVE_COLOR, SHAP_NEGATIVE_COLOR
    
    # Import utility functions:
    from daftar.viz.colors import get_color_palette, get_correlation_cmap
"""

import os
import sys
import yaml
import argparse
import inspect
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Union, Optional, Any

# ============================================================================
# Color Configuration Loading
# ============================================================================

# Default configuration file path
DEFAULT_CONFIG_PATH = Path(__file__).parent / 'colors.yaml'

# Cache for loaded colors
_color_cache = None

def load_colors(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load colors from the configuration file.
    
    Args:
        config_path: Path to the configuration file. If None, uses the default path.
        
    Returns:
        Dictionary with all color configurations
    """
    global _color_cache
    
    # Return cached colors if available
    if _color_cache is not None:
        return _color_cache
    
    # Use default path if not specified
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    # Check if file exists
    if not config_path.exists():
        # Fall back to hardcoded defaults if file doesn't exist
        return _get_default_colors()
    
    # Load colors from file
    try:
        with open(config_path, 'r') as f:
            colors = yaml.safe_load(f)
        
        # Cache the loaded colors
        _color_cache = colors
        return colors
    except Exception as e:
        print(f"Error loading colors from {config_path}: {e}")
        # Fall back to hardcoded defaults
        return _get_default_colors()

def _get_default_colors() -> Dict[str, Any]:
    """
    Get default hardcoded colors as a fallback.
    
    Returns:
        Dictionary with default colors
    """
    return {
        "classification": {
            "binary": {
                "class0": "#225ea8",
                "class1": "#c7e9b4"
            },
            "multiclass": [
                "#0c2c84", "#225ea8", "#1d91c0", "#41b6c4",
                "#7fcdbb", "#c7e9b4", "#edf8b1", "#ffffd9"
            ],
            "background": "#F8F8F8"
        },
        "regression": {
            "main": "#0c2c84",
            "mean_line": "#8a994e",
            "hist_alpha": 0.6,
            "background": "#F8F8F8"
        },
        "train_test": {
            "train": "#70e4ef",
            "test": "#dfdf20",
            "alpha": 0.8
        },
        "density": {
            "actual": "#00BFC4",
            "predicted": "#F8766D",
            "alpha": 0.5,
            "background": "#FAFAFA"
        },
        "feature_importance": {
            "bar": "#018f9c",
            "background": "#FAFAFA"
        },
        "shap": {
            "positive": "#691a4c",
            "negative": "#32748E",
            "background": "#FAFAFA",
            "interactions": {
                "link_color": "#1E88E5",
                "network_node_color": "#1E88E5",
                "network_edge_color": "#999999"
            }
        },
        "correlation": {
            "positive": "#A93226",
            "negative": "#2471A3",
            "cmap_type": "diverging",
            "background": "#F7F7F7"
        },
        "confusion_matrix": {
            "cmap": "Blues",
            "linewidth": 0.5,
            "linecolor": "#cccccc"
        },
        "per_class_shap": {
            "positive": "#1b9e77",
            "negative": "#d95f02",
            "class_colors": [
                "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
                "#66a61e", "#e6ab02", "#a6761d", "#666666"
            ]
        }
    }

# ============================================================================
# Color Utility Functions
# ============================================================================

def get_color_palette(is_classification: bool, class_count: int = None) -> Union[str, List[str]]:
    """
    Get appropriate color palette based on task type and class count.
    
    Args:
        is_classification: Whether the task is classification
        class_count: Number of classes for classification tasks
        
    Returns:
        Color palette (seaborn palette name or list of colors)
    """
    colors = load_colors()
    
    if is_classification:
        if class_count is None or class_count <= 0:
            raise ValueError("Must provide positive class_count for classification tasks")
            
        if class_count == 2:
            # Binary classification
            return [
                colors["classification"]["binary"]["class0"],
                colors["classification"]["binary"]["class1"]
            ]
        else:
            # Multiclass classification
            multiclass_colors = colors["classification"]["multiclass"]
            if class_count <= len(multiclass_colors):
                return multiclass_colors[:class_count]
            else:
                # If we need more colors than defined, use a built-in palette
                return sns.color_palette("husl", class_count)
    else:
        # Regression task - just use the main regression color
        return colors["regression"]["main"]

def get_train_test_colors() -> Dict[str, str]:
    """
    Get colors for train/test split visualizations.
    
    Returns:
        Dictionary with train and test colors
    """
    colors = load_colors()
    return {
        "Train": colors["train_test"]["train"],
        "Test": colors["train_test"]["test"]
    }

def get_correlation_cmap() -> LinearSegmentedColormap:
    """
    Get colormap for correlation plots.
    
    Returns:
        Matplotlib colormap
    """
    colors = load_colors()
    cmap_name = colors["correlation"]["cmap"]
    
    # Handle custom colormaps vs. built-in ones
    if cmap_name == "twilight_shifted":
        return plt.cm.twilight_shifted
    else:
        return plt.get_cmap(cmap_name)

def get_binary_classification_colors() -> List[str]:
    """Get colors for binary classification."""
    colors = load_colors()
    return [colors["classification"]["binary"]["class0"], 
            colors["classification"]["binary"]["class1"]]

# Note: We don't need a separate get_multiclass_colors function since 
# get_color_palette already handles this case when is_classification=True and class_count>2

def get_shap_colors() -> Dict[str, str]:
    """Get colors for SHAP visualizations."""
    colors = load_colors()
    return {
        "positive": colors["shap"]["positive"],
        "negative": colors["shap"]["negative"],
        "background": colors["shap"]["background"]
    }

def get_per_class_shap_colors() -> Dict[str, Any]:
    """Get colors for per-class SHAP visualizations."""
    colors = load_colors()
    return colors["per_class_shap"]

# ============================================================================
# Color Constants
# ============================================================================

# Load all colors from the configuration
_colors = load_colors()

# Binary classification colors (alternating)
CLASS_BAR_COLOR0 = _colors["classification"]["binary"]["class0"]  # Primary bar chart color for binary classification
CLASS_BAR_COLOR1 = _colors["classification"]["binary"]["class1"]  # Secondary bar chart color for binary classification
BINARY_CLASSIFICATION_COLORS = [CLASS_BAR_COLOR0, CLASS_BAR_COLOR1]
CLASSIFICATION_BAR_COLOR = CLASS_BAR_COLOR0  # Default color for classification bar charts

# Multiclass classification colors (for more than 2 classes)
MULTICLASS_COLORS = _colors["classification"]["multiclass"]

# Regression visualization colors
REGRESSION_COLOR = _colors["regression"]["main"]  # Color for regression visualizations and histograms
REGRESSION_MEAN_LINE_COLOR = _colors["regression"]["mean_line"]  # Mean line color
REGRESSION_HIST_ALPHA = _colors["regression"]["hist_alpha"]  # Histogram transparency

# Compare train/test visualization colors
TRAIN_HIST_COLOR = _colors["train_test"]["train"]  # Train set histogram color
TEST_HIST_COLOR = _colors["train_test"]["test"]  # Test set histogram color 
HIST_ALPHA = _colors["train_test"]["alpha"]  # Transparency for compare histograms

# Background colors for all plots
HISTOGRAM_BG_COLOR = _colors["regression"]["background"]  # Background color for regression histogram plots
CLASSIFICATION_BAR_BG_COLOR = _colors["classification"]["background"]  # Background color for classification bar charts
DENSITY_PLOT_BG_COLOR = _colors["density"]["background"]  # Background color for density plots
CORRELATION_BAR_BG = _colors["correlation"]["background"]  # Background color for correlation bar plots
FEATURE_IMPORTANCE_BAR_BG = _colors["feature_importance"]["background"]  # Background color for feature importance plots
SHAP_BG_COLOR = _colors["shap"]["background"]  # Background color for SHAP plots

# Feature importance colors
FEATURE_IMPORTANCE_BAR_COLOR = _colors["feature_importance"]["bar"]  # Color for feature importance bars

# SHAP plot colors
SHAP_POSITIVE_COLOR = _colors["shap"]["positive"]  # Color for positive SHAP values (increases prediction)
SHAP_NEGATIVE_COLOR = _colors["shap"]["negative"]  # Color for negative SHAP values (decreases prediction)

# SHAP configuration
SHAP_TOP_N_FEATURES = 15  # Default number of top features to display in SHAP visualizations

# SHAP interaction network colors
SHAP_NETWORK_NODE_COLOR = _colors["shap"]["interactions"]["network_node_color"]  # Node color for network plot
SHAP_NETWORK_EDGE_COLOR = _colors["shap"]["interactions"]["network_edge_color"]  # Edge color for network plot
SHAP_NETWORK_LINK_COLOR = _colors["shap"]["interactions"]["link_color"]  # Link color for network plot
TOP_BOTTOM_POSITIVE_COLOR = _colors["shap"]["positive"]  # Positive impact color for top-bottom network
TOP_BOTTOM_NEGATIVE_COLOR = _colors["shap"]["negative"]  # Negative impact color for top-bottom network
SHAP_NETWORK_EDGE_COLOR_DEFAULT = _colors["shap"]["interactions"]["network_edge_color"]  # Default edge color

# Density plot colors (previously hardcoded in predictions.py)
DENSITY_ACTUAL_COLOR = _colors["density"]["actual"]  # Color for actual values in density plots
DENSITY_PREDICTED_COLOR = _colors["density"]["predicted"]  # Color for predicted values in density plots
DENSITY_ALPHA = _colors["density"]["alpha"]  # Transparency for density plots

# Confusion matrix colors
CONFUSION_MATRIX_CMAP = _colors["confusion_matrix"]["cmap"]  # Default colormap for confusion matrices
CONFUSION_MATRIX_LINEWIDTH = _colors["confusion_matrix"]["linewidth"]  # Line width for confusion matrix grid
CONFUSION_MATRIX_LINECOLOR = _colors["confusion_matrix"]["linecolor"]  # Line color for confusion matrix grid

# Correlation plot colors
# Use get_correlation_cmap function instead
CORRELATION_CMAP = get_correlation_cmap()  # Colormap for correlation plots

# ============================================================================
# Color Visualization and CLI Utilities
# ============================================================================

def is_dark_color(color):
    """
    Determine if a color is dark (for text contrast).
    
    Args:
        color: Color hex code or RGB tuple
        
    Returns:
        True if the color is dark
    """
    if isinstance(color, str) and color.startswith('#'):
        # Convert hex to RGB
        color = color.lstrip('#')
        if len(color) == 3:
            color = ''.join(c+c for c in color)
        r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        r, g, b = r/255.0, g/255.0, b/255.0
    elif isinstance(color, (list, tuple)) and len(color) >= 3:
        # RGB tuple/list
        r, g, b = color[:3]
    else:
        # Try to let matplotlib convert it
        try:
            rgba = np.array(plt.matplotlib.colors.to_rgba(color))
            r, g, b = rgba[0], rgba[1], rgba[2]
        except:
            # If all else fails, assume it's not dark
            return False
    
    # Calculate perceived brightness
    # Using the formula from W3C: https://www.w3.org/TR/AERT/#color-contrast
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness < 0.5

def plot_color_swatches(colors, title, axis=None, justify='center', alphas=None, background_color=None):
    """
    Plot color swatches to visualize each color.
    
    Args:
        colors: List of colors or single color
        title: Title for the swatches
        axis: Matplotlib axis to plot on
        justify: Justification of swatches ('left', 'right', or 'center')
        alphas: List of alpha values or single alpha value (optional)
        background_color: Optional background color for the entire swatch area
    """
    # Convert single color to list
    if not isinstance(colors, (list, tuple)) or (
            isinstance(colors, str) and not colors.startswith('[')):
        colors = [colors]
    
    # Handle optional axis
    if axis is None:
        fig, axis = plt.subplots(figsize=(8, 2))
    
    # Handle background color
    if background_color:
        axis.set_facecolor(background_color)
    
    n = len(colors)
    rect_height = 0.6
    
    # Set positions based on justification
    if justify == 'left':
        positions = np.arange(n)
    elif justify == 'right':
        positions = np.arange(n) - n + 1
    else:  # center
        positions = np.arange(n) - (n-1)/2
    
    # If alphas provided, ensure it's a list of same length
    if alphas is not None:
        if not isinstance(alphas, (list, tuple)):
            alphas = [alphas] * n
        elif len(alphas) != n:
            alphas = alphas * n if len(alphas) < n else alphas[:n]
    else:
        alphas = [1.0] * n
    
    # Plot each color as a rectangle
    for i, (color, alpha) in enumerate(zip(colors, alphas)):
        rect = plt.Rectangle((positions[i] - 0.4, -rect_height/2), 0.8, rect_height, 
                          color=color, alpha=alpha, ec='black')
        axis.add_patch(rect)
        
        # Add color code as text
        if isinstance(color, str):
            label = color
        else:
            label = str(color)
        
        # Determine text color based on background
        text_color = 'white' if is_dark_color(color) else 'black'
        axis.text(positions[i], 0, label, ha='center', va='center', 
                 color=text_color, fontsize=8)
    
    # Set axis properties
    axis.set_xlim(positions[0] - 0.7, positions[-1] + 0.7)
    axis.set_ylim(-0.5, 0.5)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_frame_on(False)

def plot_colormap_gradient(cmap_name, title, axis):
    """
    Plot a gradient for a named colormap.
    
    Args:
        cmap_name: Name of the matplotlib/seaborn colormap
        title: Title for the gradient
        axis: Matplotlib axis to plot on
    """
    # Get the colormap
    if isinstance(cmap_name, str):
        try:
            cmap = plt.cm.get_cmap(cmap_name)
        except:
            try:
                cmap = sns.color_palette(cmap_name, as_cmap=True)
            except:
                print(f"Warning: colormap {cmap_name} not found")
                cmap = plt.cm.viridis
    else:
        cmap = cmap_name
    
    # Create a gradient
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    
    # Plot the gradient
    axis.imshow(gradient, aspect='auto', cmap=cmap)
    axis.set_title(title)
    axis.set_xticks([0, 128, 255])
    axis.set_xticklabels(['0', '0.5', '1'])
    axis.set_yticks([])

def get_colors_path():
    """
    Get the path to the colors.py file.
    
    Returns:
        Path to the colors.py file
    """
    # Get the path to this file
    file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    return file_path

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Display all colors defined in DAFTAR-ML color palettes"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        metavar="PATH",
        help="Output directory (default: current directory)"
    )
    
    parser.add_argument(
        "--path", "-p",
        action="store_true",
        help="Show the path to the colors.py file and exit"
    )
    
    return parser.parse_args()

def main():
    """Main function to display all colors."""
    # Parse command-line arguments
    args = parse_args()
    
    # If the user just wants to know the path to the colors file
    if args.path:
        # Check if the YAML config exists
        yaml_path = DEFAULT_CONFIG_PATH
        if yaml_path.exists():
            print(f"DAFTAR-ML colors are defined in:\n  {yaml_path}")
            print("\nTo modify colors, edit this YAML file and restart your application.")
        else:
            # Fallback to showing the old path
            colors_path = get_colors_path()
            print(f"DAFTAR-ML colors are defined in:\n  {colors_path}")
        return
    
    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure for color swatches
    plt.figure(figsize=(12, 16))
    plt.suptitle("DAFTAR-ML Color Palette", fontsize=16, y=0.99)
    plt.subplots_adjust(hspace=0.5)
    
    # Create grid of subplots
    n_rows = 9  # Adjust based on number of color groups
    
    # 1. Binary Classification Colors
    ax1 = plt.subplot(n_rows, 2, 1)
    plot_color_swatches(BINARY_CLASSIFICATION_COLORS, "Binary Classification Colors", ax1)
    
    # 2. Multiclass Classification Colors
    ax2 = plt.subplot(n_rows, 2, 2)
    plot_color_swatches(MULTICLASS_COLORS, "Multiclass Colors", ax2)
    
    # 3. Regression Colors
    ax3 = plt.subplot(n_rows, 2, 3)
    plot_color_swatches([REGRESSION_COLOR, REGRESSION_MEAN_LINE_COLOR], 
                      "Regression Colors", ax3, 
                      alphas=[1.0, 1.0])
    
    # 4. Train/Test Colors
    ax4 = plt.subplot(n_rows, 2, 4)
    plot_color_swatches([TRAIN_HIST_COLOR, TEST_HIST_COLOR], 
                       "Train/Test Colors", ax4, 
                       alphas=[HIST_ALPHA, HIST_ALPHA])
    
    # 5. Feature Importance Colors
    ax5 = plt.subplot(n_rows, 2, 5)
    plot_color_swatches([FEATURE_IMPORTANCE_BAR_COLOR], 
                       "Feature Importance Bar", ax5, 
                       background_color=FEATURE_IMPORTANCE_BAR_BG)
    
    # 6. SHAP Colors
    ax6 = plt.subplot(n_rows, 2, 6)
    plot_color_swatches([SHAP_POSITIVE_COLOR, SHAP_NEGATIVE_COLOR], 
                       "SHAP Colors (Positive/Negative)", ax6, 
                       background_color=SHAP_BG_COLOR)
    
    # 7. Density Plot Colors
    ax7 = plt.subplot(n_rows, 2, 7)
    plot_color_swatches([DENSITY_ACTUAL_COLOR, DENSITY_PREDICTED_COLOR], 
                       "Density Plot (Actual/Predicted)", ax7, 
                       alphas=[DENSITY_ALPHA, DENSITY_ALPHA], 
                       background_color=DENSITY_PLOT_BG_COLOR)
    
    # 8. SHAP Network Colors
    ax8 = plt.subplot(n_rows, 2, 8)
    plot_color_swatches([SHAP_NETWORK_NODE_COLOR, SHAP_NETWORK_EDGE_COLOR, SHAP_NETWORK_LINK_COLOR], 
                       "SHAP Network Colors (Node/Edge/Link)", ax8)
    
    # 9. Confusion Matrix Colormap
    ax9 = plt.subplot(n_rows, 2, 9)
    plot_colormap_gradient(CONFUSION_MATRIX_CMAP, "Confusion Matrix Colormap", ax9)
    
    # 10. Correlation Colormap
    ax10 = plt.subplot(n_rows, 2, 10)
    plot_colormap_gradient(CORRELATION_CMAP, "Correlation Colormap", ax10)
    
    # 11. Background Colors
    ax11 = plt.subplot(n_rows, 2, 11)
    plot_color_swatches([
        HISTOGRAM_BG_COLOR, 
        CLASSIFICATION_BAR_BG_COLOR,
        DENSITY_PLOT_BG_COLOR,
        CORRELATION_BAR_BG,
        FEATURE_IMPORTANCE_BAR_BG,
        SHAP_BG_COLOR
    ], "Background Colors", ax11)
    
    # 12. Top-Bottom Network Colors
    ax12 = plt.subplot(n_rows, 2, 12)
    plot_color_swatches([
        TOP_BOTTOM_POSITIVE_COLOR,
        TOP_BOTTOM_NEGATIVE_COLOR
    ], "Top-Bottom Network Colors (Pos/Neg)", ax12)
    
    # 13. Per-Class SHAP Colors
    ax13 = plt.subplot(n_rows, 2, 13)
    per_class_colors = get_per_class_shap_colors()
    plot_color_swatches([
        per_class_colors["positive"],
        per_class_colors["negative"]
    ], "Per-Class SHAP Colors (Pos/Neg)", ax13)
    
    # 14. Per-Class SHAP Class Colors
    ax14 = plt.subplot(n_rows, 2, 14)
    plot_color_swatches(
        per_class_colors["class_colors"],
        "Per-Class SHAP Class Colors", ax14
    )
    
    # Save the figure
    output_path = output_dir / "daftar_colors.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Color palette saved to: {output_path}")
    
    # Show color configuration path
    yaml_path = DEFAULT_CONFIG_PATH
    if yaml_path.exists():
        print(f"\nDaFTAR-ML colors are defined in: {yaml_path}")
    
    plt.close()

def run_main():
    """Wrapper function to call main() directly to avoid import issues."""
    main()

# Run the main function if called directly
if __name__ == "__main__":
    main()
