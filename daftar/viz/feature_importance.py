"""Feature importance visualization utilities for DAFTAR-ML.

This module provides utilities for generating feature importance visualizations
that work for both regression and classification models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

from daftar.viz.colors import FEATURE_IMPORTANCE_BAR_COLOR, FEATURE_IMPORTANCE_BAR_BG, SHAP_TOP_N_FEATURES, format_with_superscripts
from daftar.viz.common import save_plot


def process_fold_importances(fold_importances):
    """Process importance values.
    
    Args:
        fold_importances: List of importance dictionaries or DataFrames
        
    Returns:
        DataFrame with processed importance statistics
    """
    
    # If the input is already a DataFrame or can be converted to one
    if isinstance(sample_importances, pd.DataFrame):
        # Calculate statistics directly
        return pd.DataFrame({
            "Mean": sample_importances.mean(),
            "Std": sample_importances.std()
        })
    
    # If it's a list of dictionaries
    elif isinstance(sample_importances, list) and all(isinstance(x, dict) for x in sample_importances):
        # Convert list of dicts to DataFrame
        return pd.DataFrame({
            "Mean": pd.DataFrame(sample_importances).mean(),
            "Std": pd.DataFrame(sample_importances).std()
        })
    
    # If we couldn't process it, return an empty DataFrame
    return pd.DataFrame(columns=["Mean", "Std"])


def plot_feature_importance_bar(feature_importance_df, output_dir, top_n=None, 
                       bar_color=FEATURE_IMPORTANCE_BAR_COLOR, 
                       bar_opacity=1.0, 
                       bg_color=FEATURE_IMPORTANCE_BAR_BG):
    """Create feature importance bar plot with error bars.
    
    Args:
        feature_importance_df: DataFrame with feature importance values
        output_dir: Output directory path
        top_n: Number of top features to show (default: from color_definitions)
        bar_color: Color of the bars (hex code or matplotlib color name)
        bar_opacity: Opacity of the bars (0.0 to 1.0)
        bg_color: Background color of the plot (hex code or matplotlib color name)
    """
    # Sort by mean importance and take top N
    df = feature_importance_df.sort_values("Mean", ascending=False).head(top_n)
    
    # Reverse order for descending visualization (highest at top)
    df = df.iloc[::-1]
    
    # Create figure with white/transparent background, but colored plot area
    fig = plt.figure(figsize=(10, max(5, len(df) * 0.4)), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor(bg_color) 
    
    # Plot error bars (whiskers) behind the bars
    plt.errorbar(df["Mean"], range(len(df)), xerr=df["Std"],
                 fmt='none', ecolor='black', capsize=5, alpha=0.7, zorder=1)
                 
    # Then plot the actual bars on top
    plt.barh(df.index, df["Mean"], color=bar_color, alpha=bar_opacity, zorder=2)
    
    # Set explicit y-axis limits to remove extra space above and below bars
    if len(df) > 0:
        plt.ylim(-0.5, len(df) - 0.5)  # Tight fit around actual bars
    
    # Adjust xlim to ensure we have space for labels using 10% padding like SHAP plots
    # Get current x limits
    xmin, xmax = plt.xlim()
    
    # Calculate the maximum extent needed (value + error bar)
    max_extent = max(df["Mean"] + df["Std"])
    # Use 10% padding like in SHAP plots
    padding = max_extent * 0.1
    
    # Set limit with minimal balanced padding
    plt.xlim(0, max_extent + padding)
    
    # Recalculate limits after adjustment
    xmin, xmax = plt.xlim()
    
    epsilon = (xmax - xmin) * 0.03  # Increased spacing from whiskers
    
    # Position labels with consistent spacing after error bars
    for i, (v, std) in enumerate(zip(df["Mean"], df["Std"])):
        x_text = v + std + epsilon  # After error bar
        
        # Ensure label is inside plot boundaries
        plt.text(x_text, i,
                format_with_superscripts(v),
                va="center", ha="left", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, pad=1, edgecolor='gray', linewidth=0.5, boxstyle="round,pad=0.5", linestyle='-'),
                zorder=3)
    
    # Set labels and title
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Features by Raw Importance")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    # Create feature_importance directory if it doesn't exist
    feature_importance_dir = os.path.join(output_dir, "feature_importance")
    os.makedirs(feature_importance_dir, exist_ok=True)
    
    # Save plot to feature_importance directory
    plot_path = os.path.join(feature_importance_dir, "feature_importance_bar.png")
    fig = plt.gcf()  # Get current figure
    save_plot(fig, plot_path, tight_layout=False)  # We already called tight_layout

def save_feature_importance_values(fold_results, output_dir, in_fold_dirs=True):
    """Save feature importance values from model for each fold with a consolidated view.
    
    Args:
        fold_results: List of fold results
        output_dir: Output directory path
        in_fold_dirs: Whether to store individual fold results in their respective fold directories
        
    Returns:
        DataFrame with feature importance values
    """
    # Create a directory for feature importance files
    feature_importance_dir = os.path.join(output_dir, "feature_importance")
    os.makedirs(feature_importance_dir, exist_ok=True)
    
    # Save feature importance for each fold
    fold_feature_importances = []
    
    for fold in fold_results:
        fold_idx = fold["fold_index"]
        importances = fold["feature_importances"]

        # Add to fold list
        fold_feature_importances.append(importances)
    
    # Create feature importance DataFrame
    fold_df = pd.DataFrame({
        name: values for name, values in zip(
            ["Fold" + str(i+1) for i in range(len(fold_feature_importances))],
            fold_feature_importances
        )
    })
    
    # Calculate mean and std (averaging fold importance values)
    feature_importance_df = pd.DataFrame({
        "Mean": fold_df.mean(axis=1),
        "Std": fold_df.std(axis=1)
    })
    
    # Sort by mean importance
    feature_importance_df = feature_importance_df.sort_values("Mean", ascending=False)
    
    # Save to CSV
    csv_path = os.path.join(feature_importance_dir, "feature_importance_values.csv")
    feature_importance_df.to_csv(csv_path, index_label="Feature")

    
    return feature_importance_df
