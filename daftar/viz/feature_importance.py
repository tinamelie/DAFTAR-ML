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


def plot_feature_importance_bar(feature_importance_df, output_dir, top_n=25, bar_color="#968FF3", bar_opacity=1.0, bg_color="#F0F0F0"):
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
    
    # First plot error bars (whiskers) behind the bars
    plt.errorbar(df["Mean"], range(len(df)), xerr=df["Std"],
                 fmt='none', ecolor='black', capsize=5, alpha=0.7, zorder=1)
                 
    # Then plot the actual bars on top
    plt.barh(df.index, df["Mean"], color=bar_color, alpha=bar_opacity, zorder=2)
    
    # Adjust xlim to ensure we have space for labels
    # Get current x limits
    xmin, xmax = plt.xlim()
    # Extend xmax to ensure we have enough space for labels
    adjusted_xmax = max(df["Mean"]) * 1.3 + max(df["Std"]) * 2
    plt.xlim(0, adjusted_xmax)
    
    # Recalculate limits after adjustment
    xmin, xmax = plt.xlim()
    
    epsilon = (xmax - xmin) * 0.01
    
    # Position labels with consistent spacing after error bars
    for i, (v, std) in enumerate(zip(df["Mean"], df["Std"])):
        x_text = v + std + epsilon
        
        # Ensure label is inside plot boundaries
        plt.text(x_text, i,
                f"{v:.3f}",
                va="center", ha="left", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, pad=1),
                zorder=3)
    
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Features by Importance")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"feature_imp_bar_top{top_n}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    print(f"Feature importance bar plot saved at {plot_path}")


def save_feature_importance_values(fold_results, output_dir, in_fold_dirs=True):
    """Save feature importance values from model for each fold.
    
    Args:
        fold_results: List of fold results
        output_dir: Output directory path
        in_fold_dirs: Whether to store individual fold results in their respective fold directories
        
    Returns:
        DataFrame with overall feature importance values
    """
    # Save feature importance for each fold
    feature_importances = []
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
            importance_dir = os.path.join(output_dir, "feature_importance")
            os.makedirs(importance_dir, exist_ok=True)
            csv_path = os.path.join(importance_dir, f"feature_importance_fold_{fold_idx}.csv")
        
        # Save to CSV
        importances.to_frame("Importance").to_csv(csv_path, index_label="Feature")
        print(f"[Fold {fold_idx}] Feature importance values saved to {csv_path}")
        
        # Add to list
        feature_importances.append(importances)
    
    # Calculate overall statistics
    feature_importance_df = pd.DataFrame({
        name: values for name, values in zip(
            ["Fold" + str(i+1) for i in range(len(feature_importances))],
            feature_importances
        )
    })
    
    # Calculate mean and std
    overall_df = pd.DataFrame({
        "Mean": feature_importance_df.mean(axis=1),
        "Std": feature_importance_df.std(axis=1)
    })
    
    # Sort by mean importance
    overall_df = overall_df.sort_values("Mean", ascending=False)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "feature_importance_overall.csv")
    overall_df.to_csv(csv_path, index_label="Feature")
    print(f"Overall: Aggregated feature importance values saved to {csv_path}")
    
    return overall_df
