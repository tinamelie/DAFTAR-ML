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

    # Create a consistent colormap (blue-white-red) for SHAP plots
    # Blue = negative impact (decreases predictions)
    # Red = positive impact (increases predictions)
    colors = ["#1E88E5", "#ffffff", "#ff0d57"]
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
                            for v in mean_signed],
        "Fold_Consistency": fold_consistency
    }, index=overall_X_test.columns)

    # ----------------------------
    # Global beeswarm colored by actual feature values
    # ----------------------------
    print("Creating global SHAP beeswarm plot...")
    plt.figure(figsize=(12, 8))
    plt.title("SHAP Beeswarm (All Features, Coloured by Feature Value)")
    shap.summary_plot(overall_shap_values, overall_X_test,
                      show=False, plot_type="dot", color_bar=True)
    plt.savefig(os.path.join(main_output_dir, "shap_beeswarm_colored_global.png"),
                bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Mean‑SHAP bar plot with whiskers + labels
    # -------------------------------------------------------------------------
    print("Generating mean‑SHAP bar plot with whiskers and labels...")

    # make sure these are set:
    pos_color = "#AB0264"
    neg_color = "#3E95B5"
    cap_width = 0.2
    title = f"Top {top_n} Positive & Negative Mean SHAP Values"

    # Get top negative and positive features
    neg = shap_signed_df[shap_signed_df["Mean_Signed"] < 0].copy()
    neg["abs_val"] = neg["Mean_Signed"].abs()
    neg = neg.nlargest(top_n, "abs_val")

    pos = shap_signed_df[shap_signed_df["Mean_Signed"] > 0].nlargest(top_n, "Mean_Signed")
    pos = pos.iloc[::-1]  # So largest positive is at top

    bar_df = pd.concat([neg, pos])
    ys = np.arange(len(bar_df))

    # Create figure with appropriate size and colors
    fig, ax = plt.subplots(figsize=(10, max(4, len(bar_df) * 0.4)))
    ax.set_facecolor('#E6E6E6')

    # Draw error bars with caps
    for y, (_, row) in zip(ys, bar_df.iterrows()):
        v = row["Mean_Signed"]
        e = row["Std_MeanAcrossFolds"]
        ax.plot([v - e, v + e], [y, y], color="black", lw=1, alpha=0.7, zorder=1)
        ax.vlines([v - e, v + e], y - cap_width, y + cap_width,
                  colors="black", lw=1, alpha=0.7, zorder=1)

    # Add the bars
    colors = [neg_color if v < 0 else pos_color for v in bar_df["Mean_Signed"]]
    ax.barh(ys, bar_df["Mean_Signed"], height=0.7, color=colors, alpha=1, zorder=2)

    # Set feature names as y-axis labels
    ax.set_yticks(ys)
    ax.set_yticklabels(bar_df.index)

    # Add zero reference line
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7, zorder=0)

    # Label each bar with its value
    xmin, xmax = ax.get_xlim()
    epsilon = (xmax - xmin) * 0.01
    for y, (_, row) in zip(ys, bar_df.iterrows()):
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
    ax.set_xlabel("Mean Signed SHAP Value")
    ax.set_ylabel("Feature")
    if title:
        ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    ax.grid(axis="y", visible=False)

    # Add divider between negative and positive features
    if len(neg) and len(pos):
        ax.axhline(len(neg) - 0.5, color="gray", lw=1, alpha=0.5, zorder=0)

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
    plt.savefig(os.path.join(
        main_output_dir,
        f"shap_bar_top{top_n}pos_top{top_n}neg.png"),
        bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print("Mean‑SHAP bar plot saved.")

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
    # Save detailed CSV
    # -------------------------------------------------------------------------
    csv_path = os.path.join(main_output_dir, "shap_feature_impact_analysis.csv")
    shap_signed_df.sort_values("Impact_Magnitude", ascending=False).to_csv(csv_path, index_label="Feature")
    print(f"SHAP feature impact analysis CSV saved to {csv_path}")

    # -------------------------------------------------------------------------
    # Create correlation bar plot with enhanced visualization (for regression only)
    # -------------------------------------------------------------------------
    if problem_type == "regression":
        print("Creating SHAP‑target correlation bar plot...")
        
        # Add correlation ranks if not already present
        if "Rank_by_Correlation" not in shap_signed_df.columns:
            shap_signed_df["Rank_by_Magnitude"] = shap_signed_df["Impact_Magnitude"].rank(ascending=False)
            shap_signed_df["Rank_by_Correlation"] = shap_signed_df["Target_Correlation"].rank(ascending=False, na_option='bottom')
        
        # Get top and bottom features by correlation
        df_signed_corr = pd.DataFrame(shap_signed_df["Target_Correlation"])
        top_corr = df_signed_corr.nlargest(top_n, "Target_Correlation")
        bottom_corr = df_signed_corr.nsmallest(top_n, "Target_Correlation")
        corr_df = pd.concat([top_corr, bottom_corr]).sort_values("Target_Correlation", ascending=False)
        
        # Create colormap for better visualization
        vmax = corr_df["Target_Correlation"].abs().max()
        norm = plt.Normalize(-vmax, vmax)
        colors = [feature_cmap(norm(v)) for v in corr_df["Target_Correlation"]]
        
        # Create figure and plot
        fig, axc = plt.subplots(figsize=(10, max(5, len(corr_df) * 0.4)))
        axc.set_title(f"SHAP‑Target Correlation (Top {top_n} & Bottom {top_n})")
        
        sns.barplot(
            x="Target_Correlation",
            y=corr_df.index,
            data=corr_df,
            hue=corr_df.index,
            palette=colors,
            legend=False,  # Hide the legend
            ax=axc
        )
        
        # Set appropriate axis limits with padding
        pad = vmax * 0.15  # 15% of the largest absolute bar
        if (corr_df["Target_Correlation"] < 0).any():
            axc.set_xlim(-vmax - pad, vmax + pad)
        else:
            axc.set_xlim(0, vmax + pad)
        
        axc.set_xlabel("Correlation")
        axc.set_ylabel("Feature")
        axc.grid(axis="x", ls="--", alpha=0.3)
        axc.grid(axis="y", visible=False)
        
        # Add value annotations on bars
        offset = pad * 0.25 
        for i, v in enumerate(corr_df["Target_Correlation"]):
            x_text = v + (offset if v >= 0 else -offset)
            ha = "left" if v >= 0 else "right"
            axc.text(x_text, i,
                    f"{v:.3f}",
                    va="center", ha=ha, fontsize=8)
        
        # Add styling elements
        axc.axvline(0, color="grey", ls="--", lw=1)
        
        # Save and close
        fig.tight_layout()
        corr_path = os.path.join(main_output_dir, f"shap_corr_bar_top{top_n}pos_top{top_n}neg.png")
        fig.savefig(corr_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Enhanced SHAP‑target correlation plot saved to {corr_path}")

    return shap_signed_df
