"""Feature importance visualization utilities for DAFTAR-ML.

This module provides utilities for generating feature importance visualizations
that work for both regression and classification models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

from daftar.viz.color_definitions import FEATURE_IMPORTANCE_BAR_COLOR, FEATURE_IMPORTANCE_BAR_BG


def process_sample_importances(sample_importances):
    """Process sample-level importance values.
    
    Args:
        sample_importances: List of sample-level importance dictionaries or DataFrames
        
    Returns:
        DataFrame with processed sample-level importance statistics
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


def plot_feature_importance_bar(feature_importance_df, output_dir, top_n=25, 
                       bar_color=FEATURE_IMPORTANCE_BAR_COLOR, 
                       bar_opacity=1.0, 
                       bg_color=FEATURE_IMPORTANCE_BAR_BG):
    """Create feature importance bar plot with error bars.
    
    Args:
        feature_importance_df: DataFrame with feature importance values
        output_dir: Output directory path
        top_n: Number of top features to show
        bar_color: Color of the bars (hex code or matplotlib color name)
        bar_opacity: Opacity of the bars (0.0 to 1.0)
        bg_color: Background color of the plot (hex code or matplotlib color name)
    """
    print(f"Generating feature importance bar plot (top {top_n} features)...")
    
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
    
    # Adjust xlim to ensure we have space for labels
    # Get current x limits
    xmin, xmax = plt.xlim()
    # Extend xmax to ensure we have enough space for labels
    adjusted_xmax = max(df["Mean"]) * 1.3 + max(df["Std"]) * 2  # Add extra space for error bars
    plt.xlim(0, adjusted_xmax)
    
    # Recalculate limits after adjustment
    xmin, xmax = plt.xlim()
    
    epsilon = (xmax - xmin) * 0.01
    
    # Position labels with consistent spacing after error bars
    for i, (v, std) in enumerate(zip(df["Mean"], df["Std"])):
        x_text = v + std + epsilon  # After error bar
        
        # Ensure label is inside plot boundaries
        plt.text(x_text, i,
                f"{v:.3f}",
                va="center", ha="left", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, pad=1),
                zorder=3)
    
    # Set labels and title
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Features by Importance")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    # Create feature_importance directory if it doesn't exist
    feature_importance_dir = os.path.join(output_dir, "feature_importance")
    os.makedirs(feature_importance_dir, exist_ok=True)
    
    # Save plot to feature_importance directory
    plot_path = os.path.join(feature_importance_dir, "feature_importance_bar.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    print(f"Feature importance bar plot saved at {plot_path}")


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
        
        # Determine destination path for fold results
        if in_fold_dirs:
            # Save in respective fold directory
            fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)
            csv_path = os.path.join(fold_dir, f"feature_importance_fold_{fold_idx}.csv")
        else:
            csv_path = os.path.join(feature_importance_dir, f"feature_importance_fold_{fold_idx}.csv")
        
        # Save to CSV
        importances.to_frame("Importance").to_csv(csv_path, index_label="Feature")
        print(f"[Fold {fold_idx}] Feature importance values saved to {csv_path}")
        
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
    print(f"Feature importance saved to {csv_path}")
    
    return feature_importance_df
