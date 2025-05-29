"""DAFTAR-ML Feature Explanation Module.

This module provides functions for generating explanatory files and summaries
about feature importance and SHAP analysis results.
"""

import os
import numpy as np
import pandas as pd
from daftar.viz.colors import SHAP_TOP_N_FEATURES

def save_top_features_summary(shap_df, output_dir, prefix="", problem_type="regression", 
                         top_n=None, filename="shap_features_summary.txt"):
    """Save a summary of top features by SHAP value to a text file.
    
    Args:
        shap_df: DataFrame with SHAP analysis
        output_dir: Directory to save output
        prefix: Prefix for the output file
        problem_type: Type of problem ('regression' or 'classification')
        top_n: Number of top features to display (default: from color_definitions)
        filename: Name of the output file
    """
    # Use default from colors module if not specified
    if top_n is None:
        top_n = SHAP_TOP_N_FEATURES
    # Create output path
    output_path = os.path.join(output_dir, filename)
    
    # Open file and write header
    with open(output_path, "w") as f:
        # Get the dataset/target name from the directory name
        dir_name = os.path.basename(output_dir)
        parts = dir_name.split("_")
        dataset_name = parts[0] if len(parts) > 0 else "Unknown"
        
        # Write main header
        f.write("="*80 + "\n")
        f.write(f"SHAP SUMMARY FOR {dataset_name} PREDICTION\n")
        f.write("="*80 + "\n\n")
        
        # Write model info if available in directory name
        if len(parts) > 1:
            f.write(f"Model Type: {parts[1]}\n")
        if len(parts) > 2:
            metric_part = parts[2] if not parts[2].startswith("cv") else "N/A"
            f.write(f"Optimization Metric: {metric_part}\n")
        
        # -----------------------------------------------------------------
        # EXPLANATION OF METRICS
        # -----------------------------------------------------------------
        f.write("\n" + "="*80 + "\n")
        f.write("EXPLANATION OF SHAP METRICS\n")
        f.write("="*80 + "\n\n")
        f.write("DAFTAR-ML provides multiple metrics for each feature:\n\n")
        f.write("Fold_Mean_SHAP: Average fold-level SHAP value (mean of fold means)\n")
        f.write("   * Positive values indicate the feature increases predictions\n")
        f.write("   * Negative values indicate the feature decreases predictions\n")
        f.write("Fold_SHAP_StdDev: Standard deviation of fold-level SHAP values\n")
        f.write("   * Lower values indicate more consistent impact across folds\n")
        f.write("Fold_Level_Impact: Consistency-weighted average absolute SHAP impact\n")
        f.write("   * Used for ranking features in the beeswarm plot\n")
        f.write("   * Combines magnitude and consistency across folds\n")
        f.write("Fold_Impact_StdDev: Standard deviation of fold-level SHAP impacts\n")
        f.write("Fold_Presence: Fraction of folds where this feature was present\n")
        
        if problem_type == "regression":
            f.write("Fold_Level_Correlation: Correlation between feature's fold-level SHAP values and target\n")
            f.write("   * Shows how the feature's impact relates to the target value\n")
        
        # -----------------------------------------------------------------
        # SECTION 1: TOP FEATURES BY DIRECTIONAL IMPACT (POSITIVE & NEGATIVE)
        # -----------------------------------------------------------------
        # First show the positive and negative impact features at fold level
        f.write("\n" + "="*80 + "\n")
        f.write(f"TOP {top_n} FEATURES WITH POSITIVE IMPACT (FOLD LEVEL)\n")
        f.write("="*80 + "\n")
        f.write(f"Column: Fold_Mean_SHAP > 0 [shap_bar_fold.png]\n\n")
        
        pos_features = shap_df[shap_df["Fold_Mean_SHAP"] > 0].sort_values("Fold_Mean_SHAP", ascending=False).head(top_n)
        for i, (feature, row) in enumerate(pos_features.iterrows(), 1):
            value = row["Fold_Mean_SHAP"]
            stddev = row["Fold_SHAP_StdDev"]
            f.write(f"{i}. {feature}: {value:.6f} (±{stddev:.6f})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"TOP {top_n} FEATURES WITH NEGATIVE IMPACT (FOLD LEVEL)\n")
        f.write("="*80 + "\n")
        f.write(f"Column: Fold_Mean_SHAP < 0 [shap_bar_fold.png]\n\n")
        
        neg_features = shap_df[shap_df["Fold_Mean_SHAP"] < 0].sort_values("Fold_Mean_SHAP", ascending=True).head(top_n)
        for i, (feature, row) in enumerate(neg_features.iterrows(), 1):
            value = row["Fold_Mean_SHAP"]
            stddev = row["Fold_SHAP_StdDev"]
            f.write(f"{i}. {feature}: {value:.6f} (±{stddev:.6f})\n")
        
        # End of fold-level impact features section
            
        # -----------------------------------------------------------------
        # SECTION 3: SHAP-TARGET CORRELATIONS (REGRESSION ONLY)
        # -----------------------------------------------------------------
        if problem_type == "regression" and "Fold_Level_Correlation" in shap_df.columns:
            # Top positive correlations at fold level
            f.write("\n" + "="*80 + "\n")
            f.write(f"TOP {top_n} FEATURES BY POSITIVE SHAP-TARGET CORRELATION (FOLD LEVEL)\n")
            f.write("="*80 + "\n")
            f.write(f"Column: Fold_Level_Correlation > 0 [shap_corr_bar_fold.png]\n\n")
            
            # Get top features with positive correlation
            pos_fold_corr_features = shap_df[shap_df["Fold_Level_Correlation"] > 0].sort_values("Fold_Level_Correlation", ascending=False).head(top_n)
            for i, (feature, row) in enumerate(pos_fold_corr_features.iterrows(), 1):
                value = row["Fold_Level_Correlation"]
                f.write(f"{i}. {feature}: {value:.6f}\n")
            
            # Top negative correlations at fold level
            f.write("\n" + "="*80 + "\n")
            f.write(f"TOP {top_n} FEATURES BY NEGATIVE SHAP-TARGET CORRELATION (FOLD LEVEL)\n")
            f.write("="*80 + "\n")
            f.write(f"Column: Fold_Level_Correlation < 0 [shap_corr_bar_fold.png]\n\n")
            
            # Get top features with negative correlation
            neg_fold_corr_features = shap_df[shap_df["Fold_Level_Correlation"] < 0].sort_values("Fold_Level_Correlation", ascending=True).head(top_n)
            for i, (feature, row) in enumerate(neg_fold_corr_features.iterrows(), 1):
                value = row["Fold_Level_Correlation"]
                f.write(f"{i}. {feature}: {value:.6f}\n")
            
        # -----------------------------------------------------------------
        # INTERPRETATION GUIDE
        # -----------------------------------------------------------------
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("="*80 + "\n\n")
        
        f.write("SHAP values reveal how each feature impacts model predictions:\n\n")
        f.write("1. Directionality: Positive SHAP = feature pushes prediction higher\n")
        f.write("                   Negative SHAP = feature pushes prediction lower\n\n")
        f.write("2. Magnitude: Higher absolute SHAP = stronger impact on prediction\n\n")
        f.write("3. Consistency: Standard deviation (±) shows impact variability across folds\n\n")
        
        if problem_type == "regression":
            f.write("4. Correlation: Shows how consistent SHAP values are with the target\n")
            f.write("                High correlation means the feature's impact aligns with target values\n\n")
        
        f.write("VISUALIZATION GUIDE:\n\n")
        f.write("1. Beeswarm Plot (shap_beeswarm_impact.png):\n")
        f.write("   * Features ranked by Fold_Level_Impact (directional importance)\n")
        f.write("   * Colors show feature values (blue=low, red=high)\n")
        f.write("   * Each dot represents a sample's SHAP value\n\n")
        
        f.write("2. Bar Plots (shap_bar_impact.png, shap_bar_fold.png):\n")
        f.write("   * Use the same features as the beeswarm plot but ordered differently:\n")
        f.write("   * shap_bar_impact.png: Sorted by absolute impact (regardless of direction)\n")
        f.write("   * shap_bar_fold.png: Sorted by directional impact (matching this summary)\n")
        f.write("   * Colors indicate direction (red=positive impact, blue=negative impact)\n\n")
        
        f.write("3. Interaction Plots:\n")
        f.write("   * Network plots show how features interact with each other\n")
        f.write("   * Color scale indicates SHAP values (red=positive, blue=negative)\n")
        f.write("   * Top-bottom network separates connected features in center and lists\n")
        f.write("     unconnected features on the sides\n")
    
    return output_path
