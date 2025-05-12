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

# Direct import from color_definitions.py file
from daftar.viz.color_definitions import (
    BINARY_CLASSIFICATION_COLORS,
    REGRESSION_COLOR,
    REGRESSION_MEAN_LINE_COLOR,
    REGRESSION_HIST_ALPHA,
    FEATURE_IMPORTANCE_BAR_COLOR,
    FEATURE_IMPORTANCE_BAR_BG,
    TRAIN_HIST_COLOR,
    TEST_HIST_COLOR,
    HIST_ALPHA,
    SHAP_POSITIVE_COLOR,
    SHAP_NEGATIVE_COLOR,
    SHAP_BG_COLOR,
    MULTICLASS_COLORS,
    DENSITY_ACTUAL_COLOR,
    DENSITY_PREDICTED_COLOR,
    DENSITY_ALPHA,
    CONFUSION_MATRIX_CMAP,
    CONFUSION_MATRIX_LINEWIDTH,
    CONFUSION_MATRIX_LINECOLOR,
    get_color_palette,
    get_train_test_colors,
    CORRELATION_CMAP,
    HISTOGRAM_BG_COLOR,
    DENSITY_PLOT_BG_COLOR,
    CLASSIFICATION_BAR_BG_COLOR,
    CORRELATION_BAR_BG
)


def plot_color_swatches(colors, title, axis=None, justify='center', alphas=None, background_color=None):
    """Plot color swatches to visualize each color.
    
    Args:
        colors: List of colors or single color
        title: Title for the swatches
        axis: Matplotlib axis to plot on
        justify: Justification of swatches ('left', 'right', or 'center')
        alphas: List of alpha values or single alpha value (optional)
        background_color: Optional background color for the entire swatch area
    """
    if axis is None:
        plt.figure(figsize=(10, 2))
        axis = plt.gca()
    
    # Set background color if provided
    if background_color:
        axis.set_facecolor(background_color)
    
    # Handle single color vs. list of colors
    if isinstance(colors, str):
        colors = [colors]
    
    # Handle alpha values
    if alphas is None:
        alphas = [1.0] * len(colors)
    elif isinstance(alphas, (int, float)):
        alphas = [alphas] * len(colors)
    
    # Ensure alphas and colors have the same length
    if len(alphas) != len(colors):
        alphas = alphas[:len(colors)] if len(alphas) > len(colors) else alphas + [1.0] * (len(colors) - len(alphas))
    
    for i, (color, alpha) in enumerate(zip(colors, alphas)):
        rect = plt.Rectangle((i, 0), 0.8, 1, color=color, alpha=alpha)
        axis.add_patch(rect)
        
        # Add hex code text and alpha if not 1.0
        if alpha < 1.0:
            text = f"{color}\nα={alpha:.1f}"
        else:
            text = color
            
        axis.text(i + 0.4, 0.5, text, ha='center', va='center', 
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
    axis1.hist(reg_data, bins=30, color=REGRESSION_COLOR, alpha=REGRESSION_HIST_ALPHA)
    axis1.axvline(np.mean(reg_data), color=REGRESSION_MEAN_LINE_COLOR, linestyle='dashed')
    axis1.set_title("Overall Feature Distribution")
    axis1.set_facecolor(HISTOGRAM_BG_COLOR)
    
    # Train/Test comparison
    train_data = np.random.normal(0, 1, 1000)
    test_data = np.random.normal(0.5, 1, 300)
    
    axis2.hist(train_data, bins=20, alpha=HIST_ALPHA, color=TRAIN_HIST_COLOR, label='Train')
    axis2.hist(test_data, bins=20, alpha=HIST_ALPHA, color=TEST_HIST_COLOR, label='Test')
    axis2.legend()
    axis2.set_title("Train/Test Comparison")
    axis2.set_facecolor(HISTOGRAM_BG_COLOR)


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
    
    # Set background color
    axis.set_facecolor(DENSITY_PLOT_BG_COLOR)


def plot_confusion_matrix_example(axis):
    """Plot an example confusion matrix with the default colormap.
    
    Args:
        axis: Matplotlib axis to plot on
    """
    # Create a sample confusion matrix
    confusion_matrix = np.array([
        [25, 35],
        [10, 60]
    ])
    
    # Plot the confusion matrix
    cmap = plt.get_cmap(CONFUSION_MATRIX_CMAP)
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap=cmap, cbar=False,
               xticklabels=['Predicted 0', 'Predicted 1'], 
               yticklabels=['Actual 0', 'Actual 1'], 
               annot_kws={"size": 14}, ax=axis,
               linewidths=CONFUSION_MATRIX_LINEWIDTH, linecolor=CONFUSION_MATRIX_LINECOLOR)
    
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


def plot_feature_importance_example(axis):
    """Plot an example of feature importance visualization.
    
    Args:
        axis: Matplotlib axis to plot on
    """
    # Sample data
    features = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E', 
               'Feature F', 'Feature G', 'Feature H']
    importances = [0.35, 0.25, 0.15, 0.10, 0.08, 0.04, 0.02, 0.01]
    
    # Sort features by importance
    sorted_indices = np.argsort(importances)[::-1]
    features = [features[i] for i in sorted_indices]
    importances = [importances[i] for i in sorted_indices]
    
    # Plot bars
    bars = axis.barh(np.arange(len(features)), importances, 
                    color=FEATURE_IMPORTANCE_BAR_COLOR, alpha=0.8,
                    height=0.7)
    
    # Set background color
    axis.set_facecolor(FEATURE_IMPORTANCE_BAR_BG)
    
    # Add labels and title
    axis.set_yticks(np.arange(len(features)))
    axis.set_yticklabels(features)
    axis.set_xlabel('Importance')
    axis.set_title('Feature Importance Example')
    
    # Add grid lines
    axis.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add value labels to bars
    for i, bar in enumerate(bars):
        axis.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f"{importances[i]:.2f}", va='center')


def get_colors_path():
    """Get the path to the colors.py file.
    
    Returns:
        Path to the colors.py file
    """
    # Get the file path of the colors module
    import daftar.viz.color_definitions
    colors_path = inspect.getfile(daftar.viz.color_definitions)
    return colors_path


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="DAFTAR-ML color test visualization tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="daftar-colors [-h] [--output_dir PATH] [--path]"
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
    fig = plt.figure(figsize=(15, 45))  # Increased height for feature importance example
    
    # Calculating grid size: back to 15 rows since we'll integrate backgrounds
    grid_size = (15, 2)
    
    # Set up layout with a title on top
    fig.suptitle("DAFTAR-ML Color Palette", fontsize=16, y=0.98)
    
    # Binary classification colors
    ax1 = plt.subplot2grid(grid_size, (0, 0), colspan=2)
    plot_color_swatches(BINARY_CLASSIFICATION_COLORS, "Binary Classification Colors", ax1)
    
    # Multiclass colors - keep the 9 colors but revert justification
    ax2 = plt.subplot2grid(grid_size, (1, 0), colspan=2)
    plot_color_swatches(MULTICLASS_COLORS[:9], "Multiclass Classification Colors", ax2)
    
    # Regression color
    ax3 = plt.subplot2grid(grid_size, (2, 0), colspan=1)
    plot_color_swatches(REGRESSION_COLOR, "Regression Color", ax3, alphas=REGRESSION_HIST_ALPHA)
    
    # Mean line color
    ax4 = plt.subplot2grid(grid_size, (2, 1), colspan=1)
    plot_color_swatches(REGRESSION_MEAN_LINE_COLOR, "Regression Mean Line Color", ax4)
    
    # Feature importance colors with alpha
    ax5 = plt.subplot2grid(grid_size, (3, 0), colspan=2)
    plot_color_swatches([FEATURE_IMPORTANCE_BAR_COLOR], 
                      "Feature Importance Colors", ax5, alphas=[0.8], 
                      background_color=FEATURE_IMPORTANCE_BAR_BG)
    
    # Train/test histogram colors with alpha
    ax6 = plt.subplot2grid(grid_size, (4, 0), colspan=2)
    plot_color_swatches([TRAIN_HIST_COLOR, TEST_HIST_COLOR], 
                      "Train/Test Histogram Colors", ax6, alphas=HIST_ALPHA,
                      background_color=HISTOGRAM_BG_COLOR)
    
    # Density plot colors with alpha
    ax7 = plt.subplot2grid(grid_size, (5, 0), colspan=2)
    plot_color_swatches([DENSITY_ACTUAL_COLOR, DENSITY_PREDICTED_COLOR], 
                      "Density Plot Colors (Actual/Predicted)", ax7, alphas=DENSITY_ALPHA,
                      background_color=DENSITY_PLOT_BG_COLOR)
    
    # Confusion Matrix Colormap
    ax8 = plt.subplot2grid(grid_size, (6, 0), colspan=2)
    plot_colormap_gradient(CONFUSION_MATRIX_CMAP, f"Confusion Matrix Colormap ({CONFUSION_MATRIX_CMAP})", ax8)
    
    # Correlation Colormap
    ax8b = plt.subplot2grid(grid_size, (7, 0), colspan=2)
    if isinstance(CORRELATION_CMAP, str):
        plot_colormap_gradient(CORRELATION_CMAP, f"Correlation Colormap ({CORRELATION_CMAP})", ax8b)
    else:
        # For matplotlib colormap objects
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        ax8b.imshow(gradient, aspect='auto', cmap=CORRELATION_CMAP)
        # Use the actual name of the colormap from matplotlib
        cmap_name = CORRELATION_CMAP.name if hasattr(CORRELATION_CMAP, 'name') else "Custom Colormap"
        ax8b.set_title(f"Correlation Colormap ({cmap_name})")
        ax8b.set_yticks([])
        ax8b.set_xticks([0, 127, 255])
        ax8b.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    
    # SHAP colors
    ax9 = plt.subplot2grid(grid_size, (8, 0), colspan=2)
    plot_color_swatches([SHAP_POSITIVE_COLOR, SHAP_NEGATIVE_COLOR], 
                      "SHAP Bar Colors", ax9, background_color=SHAP_BG_COLOR)
    
    # Histogram examples
    ax10 = plt.subplot2grid(grid_size, (9, 0), colspan=1)
    ax11 = plt.subplot2grid(grid_size, (9, 1), colspan=1)
    plot_hist_example(ax10, ax11)
    
    # Density plot and confusion matrix examples
    ax12 = plt.subplot2grid(grid_size, (10, 0), colspan=1)
    ax13 = plt.subplot2grid(grid_size, (10, 1), colspan=1)
    plot_density_example(ax12)
    plot_confusion_matrix_example(ax13)
    
    # Binary classification example (half width)
    ax14 = plt.subplot2grid(grid_size, (11, 0), colspan=1)
    plot_binary_classification_example(ax14)
    
    # Multiclass classification example (half width)
    ax15 = plt.subplot2grid(grid_size, (11, 1), colspan=1)
    plot_multiclass_classification_example(ax15)
    
    # SHAP bar example
    ax16 = plt.subplot2grid(grid_size, (12, 0), colspan=2)
    plot_shap_bar_example(ax16)
    
    # Feature importance example
    ax17 = plt.subplot2grid(grid_size, (13, 0), colspan=2)
    plot_feature_importance_example(ax17)
    
    # Correlation example
    ax18 = plt.subplot2grid(grid_size, (14, 0), colspan=2)
    plot_correlation_example(ax18)
    
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


def plot_correlation_example(axis):
    """Plot example of feature correlation with target showing color gradient.
    
    Args:
        axis: Matplotlib axis to plot on
    """
    # Simplified feature names
    top_features = [f'Feature {chr(65+i)}' for i in range(10)]  # A, B, C, etc.
    bottom_features = [f'Feature {chr(90-i)}' for i in range(10)]  # Z, Y, X, etc.
    features = top_features + bottom_features
    
    # Generate correlations: top 10 positive (descending), bottom 10 negative (ascending)
    top_correlations = np.linspace(0.9, 0.4, 10)  # Top 10 from 0.9 down to 0.4
    bottom_correlations = np.linspace(-0.9, -0.4, 10)  # Bottom 10 from -0.9 up to -0.4
    correlations = np.concatenate([top_correlations, bottom_correlations])
    
    # Plot the correlations as horizontal bars with the defined correlation colormap
    norm = plt.Normalize(-1, 1)
    
    if isinstance(CORRELATION_CMAP, str):
        cmap = plt.get_cmap(CORRELATION_CMAP)
    else:
        cmap = CORRELATION_CMAP
        
    colors = cmap(norm(correlations))
    
    # Plot bars - smaller bars to show gradient
    y_pos = np.arange(len(features))
    axis.barh(y_pos, correlations, color=colors, height=0.7)
    
    # Add zero reference line
    axis.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set y-ticks to feature names
    axis.set_yticks(y_pos)
    axis.set_yticklabels(features, fontsize=8)
    
    # Set labels and title
    axis.set_title('Feature-Target Correlation Plot')
    axis.set_xlabel('Correlation Coefficient')
    axis.set_xlim(-1, 1)
    
    # Reverse the y-axis to show top features at the top
    axis.invert_yaxis()
    axis.set_facecolor(CORRELATION_BAR_BG)


def plot_binary_classification_example(axis):
    """Plot example of binary feature distribution with side-by-side bars.
    
    Args:
        axis: Matplotlib axis to plot on
    """
    # Sample data - just one feature
    feature = 'Feature'
    class0_count = 65
    class1_count = 35
    
    # Create a grouped bar chart (side by side)
    x = np.arange(1)  # Just one feature
    width = 0.35
    
    # Use the correct binary classification colors
    bar1 = axis.bar(x - width/2, [class0_count], width, label='Class 0', color=BINARY_CLASSIFICATION_COLORS[0])
    bar2 = axis.bar(x + width/2, [class1_count], width, label='Class 1', color=BINARY_CLASSIFICATION_COLORS[1])
    
    # Add labels and values on top of bars
    axis.bar_label(bar1, padding=3)
    axis.bar_label(bar2, padding=3)
    
    # Set x-axis ticks and labels
    axis.set_xticks(x)
    axis.set_xticklabels([feature])
    
    # Add labels and legend
    axis.set_ylabel('Count')
    axis.set_title('Binary Feature Distribution')
    axis.legend()
    
    # Adjust layout
    axis.set_ylim(0, max(class0_count, class1_count) * 1.2)
    axis.set_facecolor(HISTOGRAM_BG_COLOR)


def plot_multiclass_classification_example(axis):
    """Plot example of multiclass feature distribution with one feature.
    
    Args:
        axis: Matplotlib axis to plot on
    """
    # Sample data - 8 classes
    n_classes = 8
    class_counts = [40, 35, 30, 25, 20, 15, 10, 8, 5]  # Distribution across classes
    width = 0.8
    for i in range(n_classes):
        axis.bar(i, class_counts[i], width=width, 
                label=f'Class {i}', color=MULTICLASS_COLORS[i % len(MULTICLASS_COLORS)])
    axis.set_xticks(range(n_classes))
    axis.set_xticklabels([f'Class {i}' for i in range(n_classes)], rotation=45, fontsize=8)
    axis.set_ylabel('Count')
    axis.set_title('Multiclass Feature Distribution')
    axis.set_ylim(0, max(class_counts) * 1.2)
    axis.set_facecolor(HISTOGRAM_BG_COLOR)


if __name__ == "__main__":
    main() 