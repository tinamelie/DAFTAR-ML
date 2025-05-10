"""SHAP visualization and analysis utilities."""

import os
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from matplotlib.colors import LinearSegmentedColormap
from daftar.viz.color_definitions import (
    SHAP_POSITIVE_COLOR,
    SHAP_NEGATIVE_COLOR,
    SHAP_BG_COLOR,
    SHAP_FEATURE_COLORS
)

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
    if abs(val) >= cutoff:
        return f"{val:.{sig}f}"
    exp = int(np.floor(np.log10(abs(val))))
    return f"{val/10**exp:.{sci_sig}f}e{exp}"


def save_mean_shap_analysis(fold_results, main_output_dir, prefix="Mean", problem_type="regression", top_n=25):
    """Consolidate SHAP values across folds and create final summary plots & CSV.
    
    Args:
        fold_results: List of fold results
        main_output_dir: Output directory path
        prefix: Prefix for output files
        problem_type: Type of problem ('regression' or 'classification')
        top_n: Number of top features to include in visualizations
        
    Returns:
        DataFrame with SHAP analysis results
    """
    # Analyze feature impacts using SHAP values
    print(f"Starting {prefix} SHAP analysis with focus on positive and negative impact...")

    # Define custom linear colormap
    # Replacement for: colors = ["#1E88E5", "#ffffff", "#ff0d57"]
    colors = SHAP_FEATURE_COLORS
    feature_cmap = LinearSegmentedColormap.from_list("feature_cmap", colors)

    # Combine SHAP results from all folds
    shap_values_list = [fold['shap_data'][0] for fold in fold_results]
    X_test_list = [fold['shap_data'][1] for fold in fold_results]
    
    # Handle different SHAP structures for classification vs regression
    # For binary classification, shap_values often have shape (samples, features, 2)
    # For regression, they have shape (samples, features)
    is_classification = problem_type == "classification"
    
    # -------------------------------------------------------------------------
    # === Build per‑fold mean SHAP matrix =====================================
    # -------------------------------------------------------------------------
    # For each fold, calculate the mean SHAP value for each feature
    # This creates a matrix of (n_folds x n_features)
    per_fold_means = []
    for fold in fold_results:
        fold_shap_values = fold['shap_data'][0]
        # For classification with class dimension, take class 1 (positive class)
        if is_classification and len(fold_shap_values.shape) > 2:
            fold_shap_values = fold_shap_values[:, :, 1]  # Use positive class (index 1)
        fold_mean = fold_shap_values.mean(axis=0)  # mean per feature in THIS fold
        per_fold_means.append(fold_mean)
    per_fold_means = np.vstack(per_fold_means)  # shape: (n_folds, n_features)
    
    # Process the overall SHAP values for visualizations
    # For classification, handle the extra dimension
    if is_classification and len(shap_values_list[0].shape) > 2:
        # Concatenate and then select positive class (index 1)
        overall_shap_values = np.concatenate([s[:, :, 1] for s in shap_values_list])
    else:
        overall_shap_values = np.concatenate(shap_values_list)
    
    overall_X_test = pd.concat(X_test_list)

    # Calculate statistics across folds
    mean_signed = per_fold_means.mean(axis=0)              # global mean SHAP value (signed) for each feature
    std_across = per_fold_means.std(axis=0, ddof=1)       # across‑fold standard deviation of mean SHAP
    
    # Analyze consistency of feature impact direction across folds
    sign_matrix = np.sign(per_fold_means)                  # Sign (+/-) of impact in each fold
    global_sign = np.sign(mean_signed)                     # Overall sign of impact
    # Fold_consistency measures how often a feature has the same directional impact 
    # across different folds (1.0 = perfectly consistent direction)
    fold_consistency = (sign_matrix == global_sign).sum(axis=0) / sign_matrix.shape[0]

    shap_signed_df = pd.DataFrame({
        "Mean_Signed": mean_signed,
        "Std_MeanAcrossFolds": std_across,
        "Impact_Magnitude": np.abs(mean_signed),
        "Impact_Direction": ["Positive" if v > 0 else "Negative" if v < 0 else "Neutral"
                            for v in mean_signed]
        # Fold_Consistency removed as requested
    }, index=overall_X_test.columns)

    # ----------------------------
    # SAMPLE-LEVEL: Global beeswarm colored by actual feature values
    # This uses the raw SHAP values from all samples across all folds
    # Features are ranked by their mean absolute SHAP value across all samples
    # Does not penalize features for being absent in some folds
    # ----------------------------
    print("Creating sample-level SHAP beeswarm plot...")
    plt.figure(figsize=(12, 8))
    plt.title("Sample-Level SHAP Beeswarm (All Features, Colored by Feature Value)")
    
    # Calculate the sample-level feature importance directly from raw SHAP values
    sample_level_importance = np.abs(overall_shap_values).mean(axis=0)
    sample_level_importance_df = pd.DataFrame({
        "Sample_Level_Impact": sample_level_importance
    }, index=overall_X_test.columns)
    
    # Create sample-level beeswarm plot with explicit sample-level ordering
    # Sort features by their sample-level impact
    sample_sorted_features = sample_level_importance_df.sort_values('Sample_Level_Impact', ascending=False).index.tolist()
    
    # Use this to reorder the overall matrices
    sample_reordered_columns = [col for col in sample_sorted_features if col in overall_X_test.columns]
    sample_remaining_columns = [col for col in overall_X_test.columns if col not in sample_reordered_columns]
    sample_reordered_indices = [overall_X_test.columns.get_loc(col) for col in sample_reordered_columns + sample_remaining_columns]
    
    # Reorder the data for the plot
    sample_reordered_shap = overall_shap_values[:, sample_reordered_indices]
    sample_reordered_X = overall_X_test.iloc[:, sample_reordered_indices]
    
    # Generate the beeswarm plot with reordered features
    shap.summary_plot(sample_reordered_shap, sample_reordered_X,
                      show=False, plot_type="dot", color_bar=True,
                      max_display=30, sort=False)  # Disable auto-sort to keep sample-level ranking
    plt.savefig(os.path.join(main_output_dir, "shap_beeswarm_sample.png"),
                bbox_inches="tight")
    plt.close()
    
    # ----------------------------
    # FOLD-LEVEL: Global beeswarm based on fold-aggregated values
    # This uses the mean SHAP values calculated per fold first
    # Features are ranked by their mean absolute SHAP value across folds
    # ----------------------------
    print("Creating fold-level SHAP beeswarm plot...")
    
    # Create modified fold-level analysis that doesn't penalize absent features
    # We'll track which features are present in which folds
    feature_counts = {}
    fold_means_dict = {}
    
    # Process each fold individually
    for i, fold in enumerate(fold_results):
        fold_shap_values = fold['shap_data'][0]
        fold_X = fold['shap_data'][1]
        
        # For classification, handle the extra dimension
        if is_classification and len(fold_shap_values.shape) > 2:
            fold_shap_values = fold_shap_values[:, :, 1]  # Use positive class
            
        # Process each feature in this fold
        for j, feature in enumerate(fold_X.columns):
            if feature not in feature_counts:
                feature_counts[feature] = 0
                fold_means_dict[feature] = []
                
            # Record that this feature was present in this fold
            feature_counts[feature] += 1
            
            # Store the mean SHAP value for this feature in this fold
            fold_means_dict[feature].append(fold_shap_values[:, j].mean())
    
    # Calculate the fold-level average importance (only for folds where feature exists)
    fold_level_impact = {}
    for feature, means in fold_means_dict.items():
        if means:  # Only if we have data for this feature
            fold_level_impact[feature] = np.mean(np.abs(means))
    
    # Create a DataFrame with fold-level impact
    fold_level_importance_df = pd.DataFrame({
        "Fold_Level_Impact": fold_level_impact
        # Removed Fold_Presence as it's not relevant to the analysis
    })
    
    # Add the fold-level impact to the main dataframe
    for feature in shap_signed_df.index:
        if feature in fold_level_importance_df.index:
            shap_signed_df.loc[feature, "Fold_Level_Impact"] = fold_level_importance_df.loc[feature, "Fold_Level_Impact"]
        else:
            shap_signed_df.loc[feature, "Fold_Level_Impact"] = 0
    
    # Also add sample-level impact to main dataframe
    for feature in shap_signed_df.index:
        if feature in sample_level_importance_df.index:
            shap_signed_df.loc[feature, "Sample_Level_Impact"] = sample_level_importance_df.loc[feature, "Sample_Level_Impact"]
    
    # Create fold-level beeswarm plot using fold-aggregated data
    plt.figure(figsize=(12, 8))
    plt.title("Fold-Level SHAP Beeswarm (Features Ranked by Cross-Fold Consistency)")
    
    # Create a sorted index based on fold-level impact
    sorted_features = fold_level_importance_df.sort_values('Fold_Level_Impact', ascending=False).index.tolist()
    
    # Use this to reorder the overall matrices (only way to control feature order in beeswarm plot)
    reordered_columns = [col for col in sorted_features if col in overall_X_test.columns]
    remaining_columns = [col for col in overall_X_test.columns if col not in reordered_columns]
    reordered_indices = [overall_X_test.columns.get_loc(col) for col in reordered_columns + remaining_columns]
    
    # Reorder the data for the plot
    reordered_shap = overall_shap_values[:, reordered_indices]
    reordered_X = overall_X_test.iloc[:, reordered_indices]
    
    # Generate the beeswarm plot with reordered features
    shap.summary_plot(reordered_shap, reordered_X,
                     show=False, plot_type="dot", color_bar=True,
                     max_display=30, sort=False)  # Disable auto-sort to keep fold-level ranking
    
    plt.savefig(os.path.join(main_output_dir, "shap_beeswarm_fold.png"),
                bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # SAMPLE LEVEL - Bar plot of top features by raw sample impact
    # -------------------------------------------------------------------------
    print("Generating sample-level SHAP bar plot...")

    # Replacement for:
    # pos_color = "#AB0264"  # Red for positive impact
    # neg_color = "#3E95B5"  # Blue for negative impact
    pos_color = SHAP_POSITIVE_COLOR  # Red for positive impact
    neg_color = SHAP_NEGATIVE_COLOR  # Blue for negative impact
    cap_width = 0.2

    # ----- SAMPLE LEVEL BAR CHART -----
    
    # First make the sample-level chart using Sample_Level_Impact
    # Get features with highest absolute sample-level impact
    sample_df = shap_signed_df.copy()
    
    # Sort by sample-level impact and separate positive/negative
    neg_sample = sample_df[sample_df["Mean_Signed"] < 0].copy()
    neg_sample["abs_val"] = neg_sample["Sample_Level_Impact"]
    neg_sample = neg_sample.nlargest(top_n, "abs_val")

    pos_sample = sample_df[sample_df["Mean_Signed"] > 0].copy()
    pos_sample = pos_sample.nlargest(top_n, "Sample_Level_Impact")
    pos_sample = pos_sample.iloc[::-1]  # So largest positive is at top

    # Create the combined dataframe for plotting
    bar_sample_df = pd.concat([neg_sample, pos_sample])
    ys = np.arange(len(bar_sample_df))

    # Create figure with appropriate size and colors
    fig, ax = plt.subplots(figsize=(10, max(4, len(bar_sample_df) * 0.4)))
    ax.set_facecolor('#F0F0F0')

    # Use colors that match the legend
    # #AB0264 (dark pink/red) for positive values, #3E95B5 (blue) for negative values
    colors = [pos_color if x > 0 else neg_color for x in bar_sample_df["Mean_Signed"]]
    ax.barh(ys, bar_sample_df["Mean_Signed"], height=0.7, color=colors, alpha=1, zorder=2)

    # Set feature names as y-axis labels
    ax.set_yticks(ys)
    ax.set_yticklabels(bar_sample_df.index)

    # Add zero reference line
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7, zorder=0)

    # Label each bar with its value
    xmin, xmax = ax.get_xlim()
    epsilon = (xmax - xmin) * 0.01
    for y, (_, row) in zip(ys, bar_sample_df.iterrows()):
        v = row["Mean_Signed"]
        label = format_scientific(v, cutoff=0.01, sig=4, sci_sig=1)
        if v < 0:
            x_text = v - epsilon
            ha = "right"
        else:
            x_text = v + epsilon
            ha = "left"
        ax.text(x_text, y, label,
                va="center", ha=ha, fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, pad=1),
                zorder=3)

    # Set labels and grid
    ax.set_xlabel("SHAP Impact (Sample-Level)")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top 25 Features by Sample-Level Impact")
    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    ax.grid(axis="y", visible=False)

    # Add divider between negative and positive features
    if len(neg_sample) and len(pos_sample):
        ax.axhline(len(neg_sample) - 0.5, color="gray", lw=1, alpha=0.5, zorder=0)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=pos_color, label='Increases prediction'),
        Patch(facecolor=neg_color, label='Decreases prediction')
    ]

    plt.tight_layout()
    
    # Add the legend at the bottom with minimal space
    ax.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False, fontsize=9)
    
    ax.margins(x=0.1)

    # Save plot
    plt.savefig(os.path.join(main_output_dir, "shap_bar_sample.png"),
                bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print("Sample-level SHAP bar plot saved.")

    # -------------------------------------------------------------------------
    # FOLD LEVEL - Bar plot of top features by fold-level consistency
    # -------------------------------------------------------------------------
    print("Generating fold-level SHAP bar plot with cross-validation whiskers...")

    # Get features with highest fold-level impact 
    fold_df = shap_signed_df.copy()
    
    # Sort by fold-level impact (not presence!) and separate positive/negative
    neg_fold = fold_df[fold_df["Mean_Signed"] < 0].copy()
    neg_fold["abs_val"] = neg_fold["Fold_Level_Impact"]
    neg_fold = neg_fold.nlargest(top_n, "abs_val")

    pos_fold = fold_df[fold_df["Mean_Signed"] > 0].copy()
    pos_fold = pos_fold.nlargest(top_n, "Fold_Level_Impact")
    pos_fold = pos_fold.iloc[::-1]  # So largest positive is at top

    # Create the combined dataframe for plotting
    bar_fold_df = pd.concat([neg_fold, pos_fold])
    ys = np.arange(len(bar_fold_df))

    # Create figure with appropriate size and colors
    fig, ax = plt.subplots(figsize=(10, max(4, len(bar_fold_df) * 0.4)))
    ax.set_facecolor('#F0F0F0')

    # Draw error bars with caps to show cross-fold variation
    for y, (_, row) in zip(ys, bar_fold_df.iterrows()):
        v = row["Mean_Signed"]
        e = row["Std_MeanAcrossFolds"]
        ax.plot([v - e, v + e], [y, y], color="black", lw=1, alpha=0.7, zorder=1)
        ax.vlines([v - e, v + e], y - cap_width, y + cap_width,
                  colors="black", lw=1, alpha=0.7, zorder=1)

    # Use colors that match the legend
    # #AB0264 (dark pink/red) for positive values, #3E95B5 (blue) for negative values
    colors = [pos_color if x > 0 else neg_color for x in bar_fold_df["Mean_Signed"]]
    ax.barh(ys, bar_fold_df["Mean_Signed"], height=0.7, color=colors, alpha=1, zorder=2)

    # Set feature names as y-axis labels
    ax.set_yticks(ys)
    ax.set_yticklabels(bar_fold_df.index)

    # Add zero reference line
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7, zorder=0)

    # Label each bar with its value
    xmin, xmax = ax.get_xlim()
    epsilon = (xmax - xmin) * 0.01
    for y, (_, row) in zip(ys, bar_fold_df.iterrows()):
        v = row["Mean_Signed"]
        e = row["Std_MeanAcrossFolds"]
        label = format_scientific(v, cutoff=0.01, sig=4, sci_sig=1)
        if v < 0:
            x_text = v - e - epsilon
            ha = "right"
        else:
            x_text = v + e + epsilon
            ha = "left"
        ax.text(x_text, y, label,
                va="center", ha=ha, fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, pad=1),
                zorder=3)

    # Set labels and grid
    ax.set_xlabel("SHAP Impact (Fold-Level)")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top 25 Features by Fold-Level Impact (with cross-fold variation)")
    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    ax.grid(axis="y", visible=False)

    # Add divider between negative and positive features
    if len(neg_fold) and len(pos_fold):
        ax.axhline(len(neg_fold) - 0.5, color="gray", lw=1, alpha=0.5, zorder=0)

    # Add legend
    legend_elements = [
        Patch(facecolor=pos_color, label='Increases prediction'),
        Patch(facecolor=neg_color, label='Decreases prediction')
    ]

    plt.tight_layout()
    
    # Add the legend at the bottom with minimal space
    ax.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False, fontsize=9)
    
    ax.margins(x=0.1)

    # Save plot
    plt.savefig(os.path.join(main_output_dir, "shap_bar_fold.png"),
                bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print("Fold-level SHAP bar plot saved.")

    # Calculate correlation between SHAP values and feature values
    print("Calculating SHAP‑target correlations...")
    signed_corrs = {}
    
    if problem_type == "classification":
        # Set correlations to 0 for classification - not meaningful in this context
        for feature in overall_X_test.columns:
            signed_corrs[feature] = 0
    else:
        # Calculate actual correlations for regression
        for feature in overall_X_test.columns:
            fold_corrs = []
            for fold in fold_results:
                shap_vals, X_f, y_f = fold['shap_data']
                if feature in X_f.columns:
                    col_idx = X_f.columns.get_loc(feature)
                    # Safely calculate correlation with error handling
                    try:
                        # Handle divide by zero warnings
                        with np.errstate(divide='ignore', invalid='ignore'):
                            corr = np.corrcoef(shap_vals[:, col_idx], y_f)[0, 1]
                            if np.isnan(corr):
                                corr = 0
                    except:
                        corr = 0
                    fold_corrs.append(corr)
            signed_corrs[feature] = np.mean(fold_corrs) if fold_corrs else 0

    # Add correlation column to main dataframe
    shap_signed_df["Target_Correlation"] = pd.Series(signed_corrs)

    # -------------------------------------------------------------------------
    # Generate correlation plots for regression problems
    # -------------------------------------------------------------------------
    if problem_type == "regression":
        print("Generating SHAP-target correlation plots for regression...")
        
        # Sample-level correlation plot
        plt.figure(figsize=(10, max(6, len(shap_signed_df) * 0.2)))
        
        # Sort by absolute correlation value
        corr_df = shap_signed_df.sort_values("Target_Correlation", key=abs, ascending=False).head(25)
        
        # Red for positive correlation, blue for negative
        colors = [pos_color if x > 0 else neg_color for x in corr_df["Target_Correlation"]]
        plt.barh(corr_df.index, corr_df["Target_Correlation"], color=colors)
        plt.axvline(0, color="black", linestyle="-", linewidth=0.5)
        plt.xlabel("Correlation between SHAP values and target")
        plt.title("Features by SHAP-Target Correlation (Sample Level)")
        plt.tight_layout()
        plt.savefig(os.path.join(main_output_dir, "shap_corr_bar_sample.png"), 
                   bbox_inches="tight")
        plt.close()
        
        # Fold-level correlation plot - uses same data but with error bars
        fig, ax = plt.subplots(figsize=(10, max(6, len(corr_df) * 0.2)))
        
        # Use the same sorting as above
        ys = range(len(corr_df))
        
        # Draw bars
        ax.barh(ys, corr_df["Target_Correlation"], color=colors)
        ax.set_yticks(ys)
        ax.set_yticklabels(corr_df.index)
        ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Correlation between SHAP values and target")
        ax.set_title("Features by SHAP-Target Correlation (Fold Level)")
        
        # Add error bars for standard deviation
        for i, (feat, row) in enumerate(corr_df.iterrows()):
            std = row.get("Std_MeanAcrossFolds", 0)
            ax.plot([row["Target_Correlation"] - std, row["Target_Correlation"] + std], 
                   [i, i], color="black", linewidth=1, zorder=3)
            ax.plot([row["Target_Correlation"] - std, row["Target_Correlation"] - std], 
                   [i - 0.2, i + 0.2], color="black", linewidth=1, zorder=3)
            ax.plot([row["Target_Correlation"] + std, row["Target_Correlation"] + std], 
                   [i - 0.2, i + 0.2], color="black", linewidth=1, zorder=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(main_output_dir, "shap_corr_bar_fold.png"), 
                   bbox_inches="tight")
        plt.close()
        print("SHAP correlation plots saved successfully.")

    # -------------------------------------------------------------------------
    # Save detailed CSV
    # -------------------------------------------------------------------------
    # For classification problems, remove the Target_Correlation column
    output_df = shap_signed_df.copy()
    
    if problem_type == "classification":
        if "Target_Correlation" in output_df.columns:
            output_df = output_df.drop(columns=["Target_Correlation"])
    
    print("Saving SHAP analysis to CSV...")
    # Use output_df which has Target_Correlation removed for classification
    output_df.to_csv(os.path.join(main_output_dir, "shap_feature_metrics.csv"), index_label="Feature")
    
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
    shap_df.to_csv(os.path.join(main_output_dir, "shap_values_all_folds.csv"), index=False)
    
    # Rankings content is now consolidated into shap_features_summary.txt (created by save_top_features_summary)
    # We no longer generate a separate shap_feature_rankings.txt file
    
    # -------------------------------------------------------------------------
    # Create correlation bar plots with enhanced visualization (for regression only)
    # -------------------------------------------------------------------------
    # Initialize correlation dataframe for both regression and classification
    # For classification, it will remain empty
    corr_df = pd.DataFrame(columns=['Correlation_With_Target'])
    corr_df.index.name = 'Feature'
    
    if problem_type == "regression":
        # Define visualization parameters
        top_n = 25  # Top N features to show
        feature_cmap = plt.cm.coolwarm  # Use a diverging colormap
        
        # Add correlation ranks if not already present
        if "Rank_by_Correlation" not in shap_signed_df.columns:
            shap_signed_df["Rank_by_Magnitude"] = shap_signed_df["Impact_Magnitude"].rank(ascending=False)
            shap_signed_df["Rank_by_Correlation"] = shap_signed_df["Target_Correlation"].rank(ascending=False, na_option='bottom')
        
        # Calculate sample-level correlations (individual samples across all folds)
        # This uses all individual samples across folds directly
        sample_level_corrs = {}
        
        # First, collect all SHAP values and target values across all folds
        all_shap_values = {}
        all_targets = []
        
        for fold in fold_results:
            if 'shap_data' not in fold or fold['shap_data'] is None:
                continue
                
            shap_vals, X_test, y_test = fold['shap_data']
            if shap_vals is None or X_test is None or y_test is None:
                continue
                
            # For each feature, collect all SHAP values from this fold
            for j, feature in enumerate(X_test.columns):
                if j >= shap_vals.shape[1]:
                    continue
                if feature not in all_shap_values:
                    all_shap_values[feature] = []
                    
                all_shap_values[feature].extend(shap_vals[:, j].tolist())
            
            # Add target values from this fold
            all_targets.extend(y_test.tolist())
        
        # Now calculate correlations for each feature using all samples
        for feature, shap_vals in all_shap_values.items():
            if len(shap_vals) == len(all_targets) and len(shap_vals) > 0:
                try:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        corr = np.corrcoef(shap_vals, all_targets)[0, 1]
                        if np.isnan(corr):
                            corr = 0
                except:
                    corr = 0
                    
                sample_level_corrs[feature] = corr
        
        # Create DataFrame from dict and sort by correlation value
        corr_df = pd.DataFrame.from_dict(sample_level_corrs, orient='index', columns=['Correlation_With_Target'])
        corr_df.index.name = 'Feature'
        corr_df.sort_values('Correlation_With_Target', ascending=False, inplace=True)
        # Only save correlation CSV for regression tasks
        if problem_type == "regression":
            corr_df.to_csv(os.path.join(main_output_dir, f'shap_corr_sample.csv'))
        
        print("Creating SHAP‑target correlation plots...")
        
        # Get top and bottom features by correlation using the same data as the CSV
        top_corr = corr_df.nlargest(top_n, "Correlation_With_Target")
        bottom_corr = corr_df.nsmallest(top_n, "Correlation_With_Target")
        corr_df_viz = pd.concat([top_corr, bottom_corr]).sort_values("Correlation_With_Target", ascending=False)
        
        # Create colormap for better visualization
        vmax = corr_df_viz["Correlation_With_Target"].abs().max()
        norm = plt.Normalize(-vmax, vmax)
        colors = [feature_cmap(norm(v)) for v in corr_df_viz["Correlation_With_Target"]]
        
        # Create figure and plot
        fig, axc = plt.subplots(figsize=(10, max(5, len(corr_df_viz) * 0.4)))
        axc.set_title(f"SHAP‑Target Correlation (Sample-Level, Top {top_n} & Bottom {top_n})")
        
        sns.barplot(
            x="Correlation_With_Target",
            y=corr_df_viz.index,
            data=corr_df_viz,
            hue=corr_df_viz.index,
            palette=colors,
            legend=False,  # Hide the legend
            ax=axc
        )
        
        # Set appropriate axis limits with padding
        pad = vmax * 0.15  # 15% of the largest absolute bar
        if (corr_df_viz["Correlation_With_Target"] < 0).any():
            axc.set_xlim(-vmax - pad, vmax + pad)
        else:
            axc.set_xlim(0, vmax + pad)
        
        axc.set_xlabel("Correlation")
        axc.set_ylabel("Feature")
        axc.grid(axis="x", ls="--", alpha=0.3)
        axc.grid(axis="y", visible=False)
        
        # Add value annotations on bars
        offset = pad * 0.25 
        for i, v in enumerate(corr_df_viz["Correlation_With_Target"]):
            x_text = v + (offset if v >= 0 else -offset)
            ha = "left" if v >= 0 else "right"
            axc.text(x_text, i,
                    f"{v:.3f}",
                    va="center", ha=ha, fontsize=8)
        
        # Add styling elements
        axc.axvline(0, color="grey", ls="--", lw=1)
        
        # Save and close
        fig.tight_layout()
        fig.savefig(os.path.join(main_output_dir, f'shap_corr_bar_sample.png'), bbox_inches="tight")
        plt.close(fig)
        print(f"SHAP‑target correlation plot (sample-level) saved.")
    
    # Now create a fold-level correlation plot
    # Calculate fold-level correlations if possible
    fold_corrs = {}
    
    if len(fold_results) > 0:
        # Try to calculate fold-level correlations
        for fold in fold_results:
            if 'shap_data' not in fold or fold['shap_data'] is None:
                continue
                
            shap_vals, X_test, y_test = fold['shap_data']
            if shap_vals is None or X_test is None or y_test is None:
                continue
                
            # For each feature, calculate correlation between its SHAP values and target
            for j, feature in enumerate(X_test.columns):
                if j >= shap_vals.shape[1]:
                    continue
                    
                # Calculate correlation for this feature in this fold
                try:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        corr = np.corrcoef(shap_vals[:, j], y_test)[0, 1]
                        if np.isnan(corr):
                            corr = 0
                except:
                    corr = 0
                
                # Add to or update the dict
                if feature not in fold_corrs:
                    fold_corrs[feature] = []
                fold_corrs[feature].append(corr)
        
        # Average the correlations across folds
        fold_level_corrs = {}
        for feature, correlations in fold_corrs.items():
            if correlations:
                fold_level_corrs[feature] = np.mean(correlations)
        
        # Create DataFrame from dict and sort by correlation value
        fold_corr_df = pd.DataFrame.from_dict(fold_level_corrs, orient='index', columns=['Correlation_With_Target'])
        fold_corr_df.index.name = 'Feature'
        fold_corr_df.sort_values('Correlation_With_Target', ascending=False, inplace=True)
        
        # Only save correlation CSV for regression tasks
        if problem_type == "regression":
            fold_corr_df.to_csv(os.path.join(main_output_dir, f'shap_corr_fold.csv'))
        
        # Verify the fold-level correlations are different from sample-level
        # This helps ensure we're actually using different data for each plot
        has_differences = True
        if not fold_corr_df.empty and not corr_df.empty:
            if fold_corr_df.index.equals(corr_df.index):
                # Check if all values are the same
                max_diff = np.max(np.abs(fold_corr_df['Correlation_With_Target'].values - 
                                        corr_df.loc[fold_corr_df.index, 'Correlation_With_Target'].values))
                if max_diff < 1e-10:  # If practically identical
                    print("WARNING: Fold-level correlations are nearly identical to sample-level.")
                    has_differences = False
        
        if problem_type == "regression":
            # Get top and bottom features by correlation for fold-level
            top_fold_corr = fold_corr_df.nlargest(top_n, "Correlation_With_Target")
            bottom_fold_corr = fold_corr_df.nsmallest(top_n, "Correlation_With_Target")
            fold_corr_viz = pd.concat([top_fold_corr, bottom_fold_corr]).sort_values("Correlation_With_Target", ascending=False)
            
            # Create colormap for better visualization
            vmax_fold = fold_corr_viz["Correlation_With_Target"].abs().max()
            norm_fold = plt.Normalize(-vmax_fold, vmax_fold)
            colors_fold = [feature_cmap(norm_fold(v)) for v in fold_corr_viz["Correlation_With_Target"]]
            
            # Create figure and plot for fold-level
            fig_fold, axf = plt.subplots(figsize=(10, max(5, len(fold_corr_viz) * 0.4)))
            axf.set_title(f"SHAP‑Target Correlation (Fold-Level, Top {top_n} & Bottom {top_n})")
            
            # Create the bar plot for fold-level using seaborn
            sns.barplot(
                x="Correlation_With_Target",
                y=fold_corr_viz.index,
                data=fold_corr_viz,
                hue=fold_corr_viz.index,
                palette=colors_fold,
                legend=False,  # Hide the legend
                ax=axf
            )
            
            # Set appropriate axis limits with padding for fold-level
            pad_fold = vmax_fold * 0.15
            if (fold_corr_viz["Correlation_With_Target"] < 0).any():
                axf.set_xlim(-vmax_fold - pad_fold, vmax_fold + pad_fold)
            else:
                axf.set_xlim(0, vmax_fold + pad_fold)
            
            axf.set_xlabel("Correlation")
            axf.set_ylabel("Feature")
            axf.grid(axis="x", ls="--", alpha=0.3)
            axf.grid(axis="y", visible=False)
            
            # Add value annotations on bars for fold-level
            offset_fold = pad_fold * 0.25
            for i, v in enumerate(fold_corr_viz["Correlation_With_Target"]):
                x_text = v + (offset_fold if v >= 0 else -offset_fold)
                ha = "left" if v >= 0 else "right"
                axf.text(x_text, i, f"{v:.3f}", va="center", ha=ha, fontsize=8)
            
            # Add styling elements for fold-level
            axf.axvline(0, color="grey", ls="--", lw=1)
            
            # Save and close
            fig_fold.tight_layout()
            fig_fold.savefig(os.path.join(main_output_dir, f"shap_corr_bar_fold.png"), bbox_inches="tight")
            plt.close(fig_fold)
            print(f"SHAP‑target correlation plot (fold-level) saved.")
    
    # Make sure to return the dataframe so it can be used for summary generation
    return shap_signed_df.copy()
