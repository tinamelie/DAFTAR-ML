"""SHAP visualization and analysis utilities."""

import os
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from daftar.viz.common import save_plot
from daftar.viz.colors import (
    SHAP_POSITIVE_COLOR,
    SHAP_NEGATIVE_COLOR,
    SHAP_BG_COLOR,
    SHAP_TOP_N_FEATURES,
    format_with_superscripts
)
from daftar.utils.file_utils import save_figures_explanation

# SHAP visualization and analysis utilities


def format_scientific(val, cutoff=0.01, sig=4, sci_sig=1):
    """Format a number using scientific notation if below cutoff.
    
    Args:
        val: Value to format
        cutoff: Threshold for using scientific notation
        sig: Significant digits for normal format
        sci_sig: Significant digits for scientific notation
        
    Returns:
        Formatted string
    """
    # Handle zero or extremely small values
    if val == 0 or abs(val) < 1e-323:  # Smallest positive value in double precision
        return "0.0"
        
    if abs(val) >= cutoff:
        return f"{val:.{sig}f}"
        
    # Safe log calculation for scientific notation
    try:
        exp = int(np.floor(np.log10(abs(val))))
        return f"{val/10**exp:.{sci_sig}f}e{exp}"
    except (ValueError, OverflowError):
        # Fallback for any unexpected numerical issues
        return f"{val:.{sig}e}"


def save_mean_shap_analysis(fold_results, main_output_dir, prefix="Mean", problem_type="regression", top_n=None):
    """Consolidate SHAP values across folds and create final summary plots & CSV.
    
    Args:
        fold_results: List of fold results
        main_output_dir: Output directory path
        prefix: Prefix for output files
        problem_type: Type of problem ('regression' or 'classification')
        top_n: Number of top features to include in visualizations (default: from color_definitions)
        
    Returns:
        DataFrame with SHAP analysis results
    """
    # Use default from color_definitions if not specified
    # SHAP_TOP_N_FEATURES is imported at the top of the file
    if top_n is None:
        top_n = SHAP_TOP_N_FEATURES
    
    # Define custom linear colormap
    # Create color map with centralized colors (blue to white to red)
    colors = [SHAP_NEGATIVE_COLOR, SHAP_BG_COLOR, SHAP_POSITIVE_COLOR]
    feature_cmap = LinearSegmentedColormap.from_list("feature_cmap", colors)

    # Handle different SHAP structures for classification vs regression
    # For binary classification, shap_values often have shape (samples, features, 2)
    # For regression, they have shape (samples, features)
    is_classification = problem_type == "classification"
    
    # -------------------------------------------------------------------------
    # === Build per‑fold mean SHAP matrix with proper feature handling =======
    # -------------------------------------------------------------------------
    # Track which features are present in which folds to avoid penalizing absence
    all_features = set()
    fold_feature_means = {}  # fold_idx -> {feature_name -> mean_shap_value}
    
    # First pass: collect all unique features across all folds
    for fold_idx, fold in enumerate(fold_results):
        fold_X = fold['shap_data'][1]
        all_features.update(fold_X.columns)
    
    # Convert to sorted list for consistent ordering
    all_features = sorted(list(all_features))
    
    # Second pass: calculate mean SHAP values per fold, handling missing features properly
    for fold_idx, fold in enumerate(fold_results):
        fold_shap_values = fold['shap_data'][0]
        fold_X = fold['shap_data'][1]
        
        # For classification with class dimension, take class 1 (positive class)
        if is_classification and len(fold_shap_values.shape) > 2:
            fold_shap_values = fold_shap_values[:, :, 1]  # Use positive class (index 1)
        
        # Calculate mean SHAP value for each feature in this fold
        fold_means = {}
        for feature_idx, feature_name in enumerate(fold_X.columns):
            fold_means[feature_name] = fold_shap_values[:, feature_idx].mean()
        
        # For features not present in this fold, we don't record a value
        # (rather than recording 0, which would unfairly penalize the feature)
        fold_feature_means[fold_idx] = fold_means
    
    # Calculate statistics across folds, only using folds where each feature is present
    feature_statistics = {}
    
    for feature in all_features:
        # Collect mean SHAP values from folds where this feature is present
        feature_values = []
        for fold_idx in range(len(fold_results)):
            if feature in fold_feature_means[fold_idx]:
                feature_values.append(fold_feature_means[fold_idx][feature])
        
        if len(feature_values) > 0:
            # Calculate statistics only from folds where feature is present
            mean_shap = np.mean(feature_values)
            std_across = np.std(feature_values, ddof=1) if len(feature_values) > 1 else 0.0
            
            # Calculate fold consistency (direction agreement across folds)
            signs = np.sign(feature_values)
            global_sign = np.sign(mean_shap)
            direction_consistency = np.sum(signs == global_sign) / len(signs) if len(signs) > 0 else 0.0
            
            feature_statistics[feature] = {
                'Mean_SHAP': mean_shap,
                'Magnitude': np.abs(mean_shap),  # Keep it simple - magnitude = absolute value
                'SHAP_StdDev': std_across,
                'Direction_Consistency': direction_consistency,
                'Direction': "Positive" if mean_shap > 0 else "Negative" if mean_shap < 0 else "Neutral"
            }
    
    # Create DataFrame from feature statistics
    shap_values_df = pd.DataFrame.from_dict(feature_statistics, orient='index')
    
    # Combine SHAP results from all folds for visualization
    shap_values_list = [fold['shap_data'][0] for fold in fold_results]
    X_test_list = [fold['shap_data'][1] for fold in fold_results]
    
    # Process the overall SHAP values for visualizations
    # For classification, handle the extra dimension
    if is_classification and len(shap_values_list[0].shape) > 2:
        # Concatenate and then select positive class (index 1)
        overall_shap_values = np.concatenate([s[:, :, 1] for s in shap_values_list])
    else:
        overall_shap_values = np.concatenate(shap_values_list)
    
    overall_X_test = pd.concat(X_test_list)

    # Features are sorted by magnitude in the dataframe
    
    # ----------------------------
    # Global beeswarm based on fold-aggregated values
    # Features are ranked by their magnitude across folds where they're present
    # ----------------------------
    
    # Create beeswarm plot using fold-aggregated data
    plt.figure(figsize=(12, 8))
    
    # Show top N features in the title, using min to handle case where top_n is None or exceeds available features
    n_features = min(top_n or len(shap_values_df), len(shap_values_df))
    
    # Create a sorted index based on Magnitude
    sorted_features = shap_values_df.sort_values('Magnitude', ascending=False).index.tolist()
    
    # Use this to reorder the overall matrices (only way to control feature order in beeswarm plot)
    reordered_columns = [col for col in sorted_features if col in overall_X_test.columns]
    remaining_columns = [col for col in overall_X_test.columns if col not in reordered_columns]
    reordered_indices = [overall_X_test.columns.get_loc(col) for col in reordered_columns + remaining_columns]
    
    # Reorder the data for the plot
    reordered_shap = overall_shap_values[:, reordered_indices]
    reordered_X = overall_X_test.iloc[:, reordered_indices]
    
    # Generate the beeswarm plot with reordered features
    # Use top_n parameter to control number of features shown
    shap.summary_plot(reordered_shap, reordered_X,
                     show=False, plot_type="dot", color_bar=True,
                     max_display=top_n, sort=False)  # Disable auto-sort to maintain our custom ranking
    
    # Add title after generating the plot to ensure it's not overwritten
    plt.title(f"SHAP Values Across All Folds (Top {n_features} Features by SHAP Magnitude)", 
              pad=20, fontsize=14, y=1.05)
    plt.tight_layout()
    
    # Save the figure with title
    fig = plt.gcf()
    save_plot(fig, os.path.join(main_output_dir, "top_shap_beeswarm_plot.png"), tight_layout=False)

    # Define variables needed for visualizations
    pos_color = SHAP_POSITIVE_COLOR
    neg_color = SHAP_NEGATIVE_COLOR
    cap_width = 0.2  # Width of error bar caps in the SHAP bar plot

    # -------------------------------------------------------------------------
    # Bar plot of top features by SHAP values
    # -------------------------------------------------------------------------
    # Get features with their mean SHAP values across all folds
    fold_df = shap_values_df.copy()
    
    # Sort by shap value and separate positive/negative
    neg_fold = fold_df[fold_df["Mean_SHAP"] < 0].copy()
    # Sort negative features by raw value (most negative first)
    neg_fold = neg_fold.sort_values("Mean_SHAP", ascending=True).head(top_n)
    # Reverse the order so the least negative is first
    neg_fold = neg_fold.iloc[::-1]

    pos_fold = fold_df[fold_df["Mean_SHAP"] > 0].copy()
    # Sort positive features by raw value (largest positive first)
    pos_fold = pos_fold.sort_values("Mean_SHAP", ascending=False).head(top_n)
    
    # Create the dataframe with positives on top and negatives on bottom
    bar_fold_df = pd.concat([pos_fold, neg_fold])
    
    # Reverse the order for matplotlib (it places first item at the bottom of the plot)
    bar_fold_df = bar_fold_df.iloc[::-1]
    
    ys = np.arange(len(bar_fold_df))

    # Create figure with appropriate size and colors
    # SHAP_BG_COLOR is imported at the top of the file
    fig, ax = plt.subplots(figsize=(10, max(4, len(bar_fold_df) * 0.4)))
    ax.set_facecolor(SHAP_BG_COLOR)

    # Draw error bars with caps to show cross-fold variation
    for y, (_, row) in zip(ys, bar_fold_df.iterrows()):
        v = row["Mean_SHAP"]
        e = row["SHAP_StdDev"]
        ax.plot([v - e, v + e], [y, y], color="black", lw=1, alpha=0.7, zorder=1)
        ax.vlines([v - e, v + e], y - cap_width, y + cap_width,
                  colors="black", lw=1, alpha=0.7, zorder=1)

    # Use colors that match the legend - positive and negative values from color_definitions
    colors = [pos_color if x > 0 else neg_color for x in bar_fold_df["Mean_SHAP"]]
    ax.barh(ys, bar_fold_df["Mean_SHAP"], height=0.7, color=colors, alpha=1, zorder=2)

    # Set feature names as y-axis labels
    ax.set_yticks(ys)
    ax.set_yticklabels(bar_fold_df.index)
    
    # Set explicit y-axis limits to remove extra space above and below bars
    if len(ys) > 0:
        ax.set_ylim(ys.min() - 0.5, ys.max() + 0.5)  # Tight fit around actual bars

    # Add zero reference line
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7, zorder=0)

    # Label each bar with its value
    xmin, xmax = ax.get_xlim()
    x_range = xmax - xmin
    epsilon = (x_range) * 0.03  # Increased spacing from whiskers
    for y, (_, row) in zip(ys, bar_fold_df.iterrows()):
        v = row["Mean_SHAP"]
        e = row["SHAP_StdDev"]
        label = format_with_superscripts(v)
        if v < 0:
            x_text = v - e - epsilon
            ha = "right"
        else:
            x_text = v + e + epsilon
            ha = "left"
        ax.text(x_text, y, label,
                va="center", ha=ha, fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, pad=1, edgecolor="gray", linewidth=0.5, boxstyle="round,pad=0.5", linestyle="-"),
                zorder=3)

    # Adjust x-axis limits to ensure labels fit with balanced padding
    xmin, xmax = ax.get_xlim()
    x_range = xmax - xmin
    ax.set_xlim(xmin - 0.1 * x_range, xmax + 0.1 * x_range)  # 10% padding on both sides

    # Set labels and grid
    ax.set_xlabel("SHAP Value")
    ax.set_ylabel("Feature")
    
    # Update title to clearly indicate top/bottom features if both are present
    if len(neg_fold) > 0 and len(pos_fold) > 0:
        ax.set_title(f"Top {min(top_n, len(pos_fold))} Positive & Top {min(top_n, len(neg_fold))} Negative Features by SHAP Magnitude")
    else:
        ax.set_title(f"Top {top_n} Features by SHAP Magnitude")
    
    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    ax.grid(axis="y", visible=False)

    # Add divider between negative and positive features
    if len(neg_fold) and len(pos_fold):
        # Divider between positive and negative groups
        ax.axhline(len(neg_fold) - 0.5, color="gray", lw=1, alpha=0.5, zorder=0)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=pos_color, label='Increases prediction'),
        Patch(facecolor=neg_color, label='Decreases prediction')
    ]

    # Create space for the legend below the plot (with less vertical space)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Add legend below the plot
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=2, frameon=True, fontsize=9, framealpha=0.9)

    # Save plot with informative filename
    fig = plt.gcf()
    save_plot(fig, os.path.join(main_output_dir, "top_shap_bar_pos_neg.png"), tight_layout=False)
    
    # Create an additional bar plot sorted by absolute SHAP value
    create_absolute_sorted_bar_plot(fold_df, main_output_dir, pos_color, neg_color, top_n)
    

    # -------------------------------------------------------------------------
    # Save detailed CSV
    # -------------------------------------------------------------------------
    # Create output dataframe for CSV saving
    output_df = shap_values_df.copy()
    
    # Organize columns: Group by measurement type
    cols = list(output_df.columns)
    # Identify column groups
    fold_cols = [col for col in output_df.columns if col.startswith('Fold_')]
    other_cols = [col for col in output_df.columns if not col.startswith('Fold_')]

    # Reorder columns to put fold metrics after other metrics
    fold_cols = [col for col in output_df.columns if col.startswith('Fold_')]
    other_cols = [col for col in output_df.columns if not col.startswith('Fold_')]
    
    # Order: other cols, fold cols
    reordered_cols = other_cols + fold_cols 
    output_df = output_df[reordered_cols]
    
    output_df.to_csv(os.path.join(main_output_dir, "shap_means_overall.csv"), index_label="Feature")
    
    column_order = [
        'Mean_SHAP', 'Magnitude', 'SHAP_StdDev', 'Direction_Consistency', 'Direction'
    ]
    # Add any additional columns that might exist
    for col in shap_values_df.columns:
        if col not in column_order:
            column_order.append(col)
    shap_values_df = shap_values_df[column_order]
    
    # Save the raw SHAP value matrix for all samples with proper IDs
    # First check if we have sample IDs in the fold results
    sample_ids = []
    for fold in fold_results:
        if 'original_ids' in fold and fold['original_ids'] is not None:
            sample_ids.extend(fold['original_ids'])
        elif 'ids_test' in fold and fold['ids_test'] is not None:
            sample_ids.extend(fold['ids_test'])
        else:
            # Add placeholder IDs if none available
            sample_ids.extend([f'Sample_{i}' for i in range(len(fold.get('y_test', [])))])
    
    # Create DataFrame with sample IDs and fold information
    shap_df = pd.DataFrame(overall_shap_values, columns=overall_X_test.columns)
    
    # Add fold information and target values
    fold_ids = []
    target_values = []
    
    for fold_idx, fold in enumerate(fold_results, 1):
        num_samples = len(fold.get('y_test', []))
        fold_ids.extend([fold_idx] * num_samples)
        target_values.extend(fold.get('y_test', [np.nan] * num_samples))
    
    # Add ID column if available
    if len(sample_ids) == len(shap_df):
        shap_df.insert(0, 'ID', sample_ids)  # Add ID column as first column
    
    # Add fold and target info
    shap_df.insert(1, 'Fold', fold_ids)
    shap_df.insert(2, 'Target', target_values)
    
    # Save to comprehensive CSV (this will be the single source of truth)
    shap_df.to_csv(os.path.join(main_output_dir, "shap_raw_all_folds.csv"), index=False)
    

    # -------------------------------------------------------------------------

    # Make sure to return the dataframe so it can be used for summary generation
    return shap_values_df.copy()

def create_absolute_sorted_bar_plot(fold_df, main_output_dir, pos_color, neg_color, top_n=None):
    """
    Create a bar plot of features sorted by absolute SHAP value, while maintaining directionality.
    
    Args:
        fold_df: DataFrame with mean SHAP values and statistics for each feature
        main_output_dir: Output directory path
        pos_color: Color for positive SHAP values
        neg_color: Color for negative SHAP values
        top_n: Number of top features to include
    """
    # Use default from color_definitions if not specified
    # SHAP_TOP_N_FEATURES is imported at the top of the file
    if top_n is None:
        top_n = SHAP_TOP_N_FEATURES
    
    # Sort by Magnitude (absolute Mean_SHAP) which is the main importance metric
    top_features = fold_df.sort_values('Magnitude', ascending=False).index.tolist()
    
    # Filter to top_n features and ensure they're in fold_df
    top_features = [f for f in top_features if f in fold_df.index][:top_n]
    
    # Select those features from fold_df
    abs_fold_df = fold_df.loc[top_features].copy()
    
    # Sort by absolute SHAP value in ascending order (smallest absolute values first)
    abs_fold_df = abs_fold_df.sort_values('Magnitude', ascending=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(abs_fold_df) * 0.25)))
    
    # Set background color to match top_shap_bar_pos_neg.png
    ax.set_facecolor(SHAP_BG_COLOR)
    
    # Set up y-coordinates
    ys = np.arange(len(abs_fold_df))
    
    # Draw error bars with caps to show cross-fold variation
    for y, (_, row) in zip(ys, abs_fold_df.iterrows()):
        v = row["Mean_SHAP"]
        e = row["SHAP_StdDev"]
        ax.plot([v - e, v + e], [y, y], color="black", lw=1, alpha=0.7, zorder=1)
        ax.vlines([v - e, v + e], y - 0.2, y + 0.2,
                  colors="black", lw=1, alpha=0.7, zorder=1)
    
    # Draw bars with color based on directionality
    colors = [pos_color if x > 0 else neg_color for x in abs_fold_df["Mean_SHAP"]]
    ax.barh(ys, abs_fold_df["Mean_SHAP"], height=0.7, color=colors, alpha=1, zorder=2)
    
    
    # Set feature names as y-axis labels
    ax.set_yticks(ys)
    ax.set_yticklabels(abs_fold_df.index)
    
    # Set explicit y-axis limits
    if len(ys) > 0:
        ax.set_ylim(ys.min() - 0.5, ys.max() + 0.5)  # Tight fit
    
    # Add zero reference line
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7, zorder=0)
    
    # Label each bar with its value
    xmin, xmax = ax.get_xlim()
    x_range = xmax - xmin
    epsilon = (x_range) * 0.03  # Increased spacing from whiskers
    for y, (_, row) in zip(ys, abs_fold_df.iterrows()):
        v = row["Mean_SHAP"]
        e = row["SHAP_StdDev"]
        label = format_with_superscripts(v)
        if v < 0:
            x_text = v - e - epsilon
            ha = "right"
        else:
            x_text = v + e + epsilon
            ha = "left"
        ax.text(x_text, y, label,
                va="center", ha=ha, fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, pad=1, edgecolor="gray", linewidth=0.5, boxstyle="round,pad=0.5", linestyle="-"),
                zorder=3)
    
    # Adjust x-axis limits to ensure labels fit with balanced padding
    xmin, xmax = ax.get_xlim()
    x_range = xmax - xmin
    ax.set_xlim(xmin - 0.1 * x_range, xmax + 0.1 * x_range)  # 10% padding on both sides
    
    # Set labels and grid
    ax.set_xlabel("SHAP Value")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} Features by SHAP Magnitude")
    
    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    ax.grid(axis="y", visible=False)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=pos_color, label='Increases prediction'),
        Patch(facecolor=neg_color, label='Decreases prediction')
    ]
    
    # Create space for the legend below the plot
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Add legend below the plot
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=True, fontsize=9, framealpha=0.9)
    
    # Save plot with informative filename
    fig = plt.gcf()
    save_plot(fig, os.path.join(main_output_dir, "top_shap_bar_plot.png"), tight_layout=False)