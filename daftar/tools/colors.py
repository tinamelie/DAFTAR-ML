#!/usr/bin/env python
"""DAFTAR-ML Color Visualization Tool.

This utility creates visualizations to display all the colors defined in the 
DAFTAR-ML color palette. It helps ensure consistent visual styling across all 
visualizations and makes it easy to reference available colors.

Usage:
    daftar-colors --output_dir OUTPUT_DIR
    daftar-colors --path
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.patches as mpatches
import inspect
import matplotlib.cm as cm

# Direct import from colors.py file
from daftar.viz.colors import (
    BINARY_CLASSIFICATION_COLORS,
    REGRESSION_COLOR,
    REGRESSION_HIST_COLOR,
    REGRESSION_MEAN_LINE_COLOR,
    FEATURE_IMPORTANCE_BAR_COLOR,
    FEATURE_IMPORTANCE_BAR_BG,
    TRAIN_HIST_COLOR,
    TEST_HIST_COLOR,
    HIST_ALPHA,
    SHAP_POSITIVE_COLOR,
    SHAP_NEGATIVE_COLOR,
    SHAP_BG_COLOR,
    SHAP_FEATURE_COLORS,
    MULTICLASS_COLORS,
    DENSITY_ACTUAL_COLOR,
    DENSITY_PREDICTED_COLOR,
    DENSITY_ALPHA,
    CONFUSION_MATRIX_CMAP,
    get_color_palette,
    get_train_test_colors,
    get_shap_feature_cmap
)


def plot_color_swatches(colors, title, axis=None):
    """Plot color swatches to visualize each color.
    
    Args:
        colors: List of colors or single color
        title: Title for the swatches
        axis: Matplotlib axis to plot on
    """
    if axis is None:
        plt.figure(figsize=(10, 2))
        axis = plt.gca()
    
    # Handle single color vs. list of colors
    if isinstance(colors, str):
        colors = [colors]
    
    for i, color in enumerate(colors):
        rect = plt.Rectangle((i, 0), 0.8, 1, color=color)
        axis.add_patch(rect)
        # Add hex code text
        axis.text(i + 0.4, 0.5, color, ha='center', va='center', 
                 color='white' if is_dark_color(color) else 'black',
                 fontsize=9, fontweight='bold')
    
    axis.set_xlim(0, len(colors))
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title)
    

def is_dark_color(color):
    """Determine if a color is dark (for text contrast).
    
    Args:
        color: Color hex code or RGB tuple
        
    Returns:
        True if the color is dark
    """
    # Handle RGB tuples from seaborn
    if isinstance(color, tuple) and len(color) == 3:
        r, g, b = [int(c * 255) for c in color]
    # Convert hex to RGB
    elif isinstance(color, str) and color.startswith('#'):
        color = color[1:]
        if len(color) == 3:  # Handle shorthand hex
            color = ''.join(c + c for c in color)
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    else:
        # Default to black if color format is unknown
        return True
    
    # Calculate luminance - a common approach to determine text color
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    # Return True if the color is dark (luminance is low)
    return luminance < 0.5


def plot_linear_gradient(colors, title, axis):
    """Plot a linear gradient using the given colors.
    
    Args:
        colors: List of colors for gradient
        title: Title for the gradient
        axis: Matplotlib axis to plot on
    """
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    
    cmap = get_shap_feature_cmap()
    axis.imshow(gradient, aspect='auto', cmap=cmap)
    axis.set_title(title)
    axis.set_yticks([])
    axis.set_xticks([0, 127, 255])
    axis.set_xticklabels(['Negative', 'Neutral', 'Positive'])


def plot_colormap_gradient(cmap_name, title, axis):
    """Plot a gradient for a named colormap.
    
    Args:
        cmap_name: Name of the matplotlib/seaborn colormap
        title: Title for the gradient
        axis: Matplotlib axis to plot on
    """
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    
    cmap = plt.get_cmap(cmap_name)
    axis.imshow(gradient, aspect='auto', cmap=cmap)
    axis.set_title(title)
    axis.set_yticks([])
    axis.set_xticks([0, 127, 255])
    axis.set_xticklabels(['Low', 'Medium', 'High'])


def plot_hist_example(axis1, axis2):
    """Plot example histograms with the histogram colors.
    
    Args:
        axis1: Matplotlib axis for regression histogram
        axis2: Matplotlib axis for train/test comparison
    """
    # Sample data
    np.random.seed(42)
    reg_data = np.random.normal(0, 1, 1000)
    
    # Regression histogram
    axis1.hist(reg_data, bins=30, color=REGRESSION_HIST_COLOR, alpha=HIST_ALPHA)
    axis1.axvline(np.mean(reg_data), color=REGRESSION_MEAN_LINE_COLOR, linestyle='dashed')
    axis1.set_title("Regression Histogram")
    
    # Train/Test comparison
    train_data = np.random.normal(0, 1, 1000)
    test_data = np.random.normal(0.5, 1, 300)
    
    axis2.hist(train_data, bins=20, alpha=HIST_ALPHA, color=TRAIN_HIST_COLOR, label='Train')
    axis2.hist(test_data, bins=20, alpha=HIST_ALPHA, color=TEST_HIST_COLOR, label='Test')
    axis2.legend()
    axis2.set_title("Train/Test Comparison")


def plot_density_example(axis):
    """Plot an example density plot with the density colors.
    
    Args:
        axis: Matplotlib axis to plot on
    """
    # Sample data
    np.random.seed(42)
    actual_data = np.random.normal(0, 1, 1000)
    predicted_data = np.random.normal(0.3, 1.2, 1000)
    
    # Create density plots
    sns.kdeplot(actual_data, label='Actual', fill=True, alpha=DENSITY_ALPHA, color=DENSITY_ACTUAL_COLOR, ax=axis)
    sns.kdeplot(predicted_data, label='Predicted', fill=True, alpha=DENSITY_ALPHA, color=DENSITY_PREDICTED_COLOR, ax=axis)
    
    # Add labels and legend
    axis.set_title("Density Plot Example")
    axis.set_xlabel("Value")
    axis.set_ylabel("Density")
    axis.legend()


def plot_confusion_matrix_example(axis):
    """Plot an example confusion matrix with the default colormap.
    
    Args:
        axis: Matplotlib axis to plot on
    """
    # Create a sample confusion matrix
    confusion_matrix = np.array([
        [85, 15],
        [10, 90]
    ])
    
    # Plot the confusion matrix
    cmap = plt.get_cmap(CONFUSION_MATRIX_CMAP)
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap=cmap, cbar=False,
               xticklabels=['Predicted 0', 'Predicted 1'], 
               yticklabels=['Actual 0', 'Actual 1'], 
               annot_kws={"size": 14}, ax=axis)
    
    # Set title
    axis.set_title(f"Confusion Matrix ({CONFUSION_MATRIX_CMAP} colormap)")


def plot_shap_bar_example(axis):
    """Plot an example SHAP bar chart.
    
    Args:
        axis: Matplotlib axis to plot on
    """
    # Sample data
    features = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']
    values = [0.5, -0.3, 0.2, -0.4, 0.1]
    
    # Plot bars
    colors = [SHAP_POSITIVE_COLOR if v > 0 else SHAP_NEGATIVE_COLOR for v in values]
    axis.barh(features, values, color=colors)
    axis.axvline(0, color='gray', linestyle='--')
    axis.set_facecolor(SHAP_BG_COLOR)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=SHAP_POSITIVE_COLOR, label='Increases prediction'),
        mpatches.Patch(facecolor=SHAP_NEGATIVE_COLOR, label='Decreases prediction')
    ]
    axis.legend(handles=legend_elements, loc='lower right')
    axis.set_title("SHAP Bar Chart Example")


def get_colors_path():
    """Get the path to the colors.py file.
    
    Returns:
        Path to the colors.py file
    """
    # Get the file path of the colors module
    import daftar.viz.colors
    colors_path = inspect.getfile(daftar.viz.colors)
    return colors_path


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="DAFTAR-ML color test visualization tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
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
        colors_path = get_colors_path()
        print(f"DAFTAR-ML colors are defined in:")
        print(f"  {colors_path}")
        print("\nTo modify colors, edit this file and restart your application.")
        return
    
    # Set the output directory and create it if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the output file path in the specified directory
    output_file = output_dir / "color_palette.png"
    
    # Create a figure with subplots for each category of colors
    fig = plt.figure(figsize=(15, 32))  # Increased height for additional plots
    
    # Calculate grid size to fit all our visualization types
    grid_size = (12, 2)  # Increased rows for confusion matrix colors and example
    
    # Set up layout with a title on top
    fig.suptitle("DAFTAR-ML Color Palette", fontsize=16, y=0.98)
    
    # Binary classification colors
    ax1 = plt.subplot2grid(grid_size, (0, 0), colspan=2)
    plot_color_swatches(BINARY_CLASSIFICATION_COLORS, "Binary Classification Colors", ax1)
    
    # Multiclass colors
    ax2 = plt.subplot2grid(grid_size, (1, 0), colspan=2)
    plot_color_swatches(MULTICLASS_COLORS, "Multiclass Classification Colors", ax2)
    
    # Regression color
    ax3 = plt.subplot2grid(grid_size, (2, 0), colspan=1)
    plot_color_swatches(REGRESSION_COLOR, "Regression Color", ax3)
    
    # Mean line color
    ax4 = plt.subplot2grid(grid_size, (2, 1), colspan=1)
    plot_color_swatches(REGRESSION_MEAN_LINE_COLOR, "Regression Mean Line Color", ax4)
    
    # Feature importance colors
    ax5 = plt.subplot2grid(grid_size, (3, 0), colspan=2)
    plot_color_swatches([FEATURE_IMPORTANCE_BAR_COLOR, FEATURE_IMPORTANCE_BAR_BG], 
                      "Feature Importance Colors", ax5)
    
    # Train/test histogram colors
    ax6 = plt.subplot2grid(grid_size, (4, 0), colspan=2)
    plot_color_swatches([TRAIN_HIST_COLOR, TEST_HIST_COLOR], 
                      "Train/Test Histogram Colors", ax6)
    
    # Density plot colors
    ax7 = plt.subplot2grid(grid_size, (5, 0), colspan=2)
    plot_color_swatches([DENSITY_ACTUAL_COLOR, DENSITY_PREDICTED_COLOR], 
                      "Density Plot Colors (Actual/Predicted)", ax7)
    
    # Confusion Matrix Colormap
    ax8 = plt.subplot2grid(grid_size, (6, 0), colspan=2)
    plot_colormap_gradient(CONFUSION_MATRIX_CMAP, f"Confusion Matrix Colormap ({CONFUSION_MATRIX_CMAP})", ax8)
    
    # SHAP colors
    ax9 = plt.subplot2grid(grid_size, (7, 0), colspan=2)
    plot_color_swatches([SHAP_POSITIVE_COLOR, SHAP_NEGATIVE_COLOR, SHAP_BG_COLOR], 
                      "SHAP Chart Colors", ax9)
    
    # SHAP feature colors (linear gradient)
    ax10 = plt.subplot2grid(grid_size, (8, 0), colspan=2)
    plot_linear_gradient(SHAP_FEATURE_COLORS, "SHAP Feature Colormap", ax10)
    
    # Histogram examples
    ax11 = plt.subplot2grid(grid_size, (9, 0), colspan=1)
    ax12 = plt.subplot2grid(grid_size, (9, 1), colspan=1)
    plot_hist_example(ax11, ax12)
    
    # Density plot and confusion matrix examples
    ax13 = plt.subplot2grid(grid_size, (10, 0), colspan=1)
    ax14 = plt.subplot2grid(grid_size, (10, 1), colspan=1)
    plot_density_example(ax13)
    plot_confusion_matrix_example(ax14)
    
    # SHAP bar example
    ax15 = plt.subplot2grid(grid_size, (11, 0), colspan=2)
    plot_shap_bar_example(ax15)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for suptitle
    
    # Save the color palette
    plt.savefig(output_file, dpi=100, bbox_inches="tight")
    plt.close()
    
    # Print messages in the exact format requested
    colors_path = get_colors_path()
    print(f"Color palette saved to {output_file}")
    print()
    print(f"DAFTAR-ML colors are defined in:")
    print(f"  {colors_path}")
    print()
    print("To modify colors, edit this file and restart your application.")


def run_main():
    """Wrapper function to call main() directly to avoid import issues."""
    main()


if __name__ == "__main__":
    main() 