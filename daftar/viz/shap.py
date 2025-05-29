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
from daftar.viz.colors import (
    SHAP_POSITIVE_COLOR,
    SHAP_NEGATIVE_COLOR,
    SHAP_BG_COLOR,
    CORRELATION_CMAP,
    SHAP_TOP_N_FEATURES
)
from daftar.viz.feature_explanation import save_top_features_summary

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
    # Analyze feature impacts using SHAP values
    print(f"Starting {prefix} SHAP analysis with focus on positive and negative impact...")
    
    # Interactions are now computed and saved at the fold level during pipeline execution.
    # Disable redundant (and memory-heavy) interaction computation here.
    if False:  # previously: if problem_type == "regression":
        print("Calculating SHAP interactions for all folds...")
        successful_interactions = 0

        # Process each fold separately
        for i, fold in enumerate(fold_results, 1):
            # Check for SHAP data
            if 'shap_data' not in fold or not fold['shap_data']:
                print(f"Skipping fold {i}: missing SHAP data")
                continue
                
            # Get or load the model
            model = None
            if 'model' in fold and fold['model'] is not None:
                model = fold['model']
                print(f"Using in-memory model for fold {i}")
            else:
                # Try to load model from disk
                import joblib
                model_path = Path(main_output_dir) / f"fold_{i}" / f"best_model_fold_{i}.pkl"
                try:
                    model = joblib.load(model_path)
                    print(f"Successfully loaded model from {model_path}")
                except Exception as e:
                    print(f"Error loading model for fold {i}: {str(e)}")
                    continue
                    
            if model is None:
                print(f"Skipping fold {i}: couldn't get a valid model")
                continue
                
            # Get test data (model already loaded above)
            X_test = fold.get('X_test')
            
            # If X_test is missing, try to get it from shap_data
            if X_test is None and len(fold['shap_data']) > 1:
                X_test = fold['shap_data'][1]
                
            # Skip if no test data
            if X_test is None:
                print(f"Skipping fold {i}: missing X_test data")
                continue
                
            # Get SHAP values and feature names
            shap_values = fold['shap_data'][0]
            
            # Get feature names - try multiple sources
            feature_names = None
            
            # Source 1: Check if X_test is already a DataFrame with column names
            if isinstance(X_test, pd.DataFrame):
                feature_names = list(X_test.columns)
                print(f"Using feature names from X_test DataFrame for fold {i}")
            
            # Source 2: Check if feature_names are explicitly stored in fold
            elif fold.get('feature_names') is not None:
                feature_names = fold.get('feature_names')
                print(f"Using feature_names from fold data for fold {i}")
            
            # Source 3: Extract from SHAP data - this is most reliable
            elif 'shap_data' in fold and len(fold['shap_data']) > 1:
                # If SHAP data has a DataFrame as second element
                shap_df = fold['shap_data'][1]
                if isinstance(shap_df, pd.DataFrame):
                    feature_names = list(shap_df.columns)
                    print(f"Using feature names from SHAP data DataFrame for fold {i}")
                # If not a DataFrame but we have X_test numpy array, try to create feature names
                elif hasattr(X_test, 'shape') and hasattr(fold['shap_data'][0], 'shape'):
                    # Match X_test with SHAP values shape
                    num_features = fold['shap_data'][0].shape[1] if fold['shap_data'][0].ndim > 1 else X_test.shape[1]
                    # Get feature names from existing CSV files
                    fold_dir = Path(main_output_dir) / f"fold_{i}"
                    try:
                        # Try to load feature names from feature importance file
                        fi_file = fold_dir / f"feature_importance_fold_{i}.csv"
                        if fi_file.exists():
                            fi_df = pd.read_csv(fi_file)
                            if 'Feature' in fi_df.columns:
                                feature_names = fi_df['Feature'].tolist()
                                print(f"Loaded {len(feature_names)} feature names from feature importance file for fold {i}")
                    except Exception as e:
                        print(f"Error loading feature names from CSV for fold {i}: {str(e)}")
            
            # Convert X_test to DataFrame if we have feature names
            if feature_names:
                if not isinstance(X_test, pd.DataFrame):
                    X_test = pd.DataFrame(X_test, columns=feature_names)
            else:
                print(f"Skipping fold {i}: could not find feature names from any source")
                continue
                    
            # Get top features for this fold based on SHAP values
            # Use 4*top_n to ensure coverage of both positive and negative features
            max_features = 4 * top_n
            
            # Calculate mean absolute SHAP values and get top indices
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(-mean_abs_shap)[:max_features]
            top_features = [feature_names[i] for i in top_indices]
            
            # Filter X_test to only include top features to speed up interaction calculation
            X_filtered = X_test[top_features]
            
            print(f"Calculating interactions for fold {i} using top {len(top_features)} features...")
            
            try:
                # Create fold directory if it doesn't exist
                fold_dir = Path(main_output_dir) / f"fold_{i}"
                fold_dir.mkdir(exist_ok=True)
                
                # Extract the underlying model if needed
                if hasattr(model, 'model'):
                    underlying_model = model.model
                else:
                    underlying_model = model
                    
                # For XGBoost models, extract the booster
                if 'xgboost' in str(type(underlying_model)).lower() and hasattr(underlying_model, 'get_booster'):
                    underlying_model = underlying_model.get_booster()
                
                # Calculate SHAP interactions
                explainer = shap.TreeExplainer(underlying_model)
                interaction_values = explainer.shap_interaction_values(X_filtered)
                
                # If interaction_values is a list (for multi-output models), use the first element
                if isinstance(interaction_values, list):
                    interaction_values = interaction_values[0]
                    
                # Average across samples to get feature-feature interactions
                interaction_matrix = np.mean(np.abs(interaction_values), axis=0)
                
                # Create tidy DataFrame with feature1, feature2, interaction_strength
                interactions = []
                for i1, f1 in enumerate(top_features):
                    for i2, f2 in enumerate(top_features):
                        if i1 <= i2:  # Include diagonal and upper triangle
                            strength = interaction_matrix[i1, i2]
                            interactions.append({"feature1": f1, "feature2": f2, "interaction_strength": strength})
                
                interaction_df = pd.DataFrame(interactions)
                
                # Save to CSV
                csv_path = fold_dir / f"fold_{i}_interactions.csv"
                interaction_df.to_csv(csv_path, index=False)
                
                print(f"Successfully saved interactions for fold {i} to {csv_path}")
                successful_interactions += 1
                
            except Exception as e:
                print(f"Error calculating interactions for fold {i}: {str(e)}")
        
        # Verify that at least one fold had successful interactions
        if successful_interactions == 0:
            print("WARNING: No fold interactions were successfully calculated. Network visualizations will fail.")
            print("Check for model compatibility with SHAP interactions.")
        else:
            print(f"Successfully calculated interactions for {successful_interactions} fold(s).")

    # Define custom linear colormap
    # Create color map with centralized colors (blue to white to red)
    colors = [SHAP_NEGATIVE_COLOR, SHAP_BG_COLOR, SHAP_POSITIVE_COLOR]
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

    # Calculate fold-level metrics
    shap_signed_df = pd.DataFrame({
        "Fold_Mean_SHAP": mean_signed,
        "Fold_SHAP_StdDev": std_across,
        "Fold_Impact_Direction": ["Positive" if v > 0 else "Negative" if v < 0 else "Neutral"
                            for v in mean_signed],
    }, index=overall_X_test.columns)

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
    })
    
    # Add the fold-level impact to the main dataframe
    for feature in shap_signed_df.index:
        if feature in fold_level_importance_df.index:
            shap_signed_df.loc[feature, "Fold_Level_Impact"] = fold_level_importance_df.loc[feature, "Fold_Level_Impact"]
        else:
            shap_signed_df.loc[feature, "Fold_Level_Impact"] = 0
    
    # Sample-level impact references removed
    
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
    # Use top_n parameter to control number of features shown
    shap.summary_plot(reordered_shap, reordered_X,
                     show=False, plot_type="dot", color_bar=True,
                     max_display=top_n, sort=False)  # Disable auto-sort to keep fold-level ranking
    
    plt.savefig(os.path.join(main_output_dir, "shap_beeswarm_impact.png"),
                bbox_inches="tight")
    plt.close()

    # Sample-level bar plot code removed
    # Define variables needed for fold-level visualizations
    pos_color = SHAP_POSITIVE_COLOR
    neg_color = SHAP_NEGATIVE_COLOR
    cap_width = 0.2  # Used in fold-level error bars

    # -------------------------------------------------------------------------
    # FOLD LEVEL - Bar plot of top features by fold-level consistency
    # -------------------------------------------------------------------------
    print("Generating fold-level SHAP bar plot with cross-validation whiskers...")

    # Get features with highest fold-level impact 
    fold_df = shap_signed_df.copy()
    
    # Sort by fold-level impact (not presence!) and separate positive/negative
    neg_fold = fold_df[fold_df["Fold_Mean_SHAP"] < 0].copy()
    # Sort negative features by raw value (most negative first)
    neg_fold = neg_fold.sort_values("Fold_Mean_SHAP", ascending=True).head(top_n)
    # Reverse the order so the least negative is first
    neg_fold = neg_fold.iloc[::-1]

    pos_fold = fold_df[fold_df["Fold_Mean_SHAP"] > 0].copy()
    # Sort positive features by raw value (largest positive first)
    pos_fold = pos_fold.sort_values("Fold_Mean_SHAP", ascending=False).head(top_n)
    
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
        v = row["Fold_Mean_SHAP"]
        e = row["Fold_SHAP_StdDev"]
        ax.plot([v - e, v + e], [y, y], color="black", lw=1, alpha=0.7, zorder=1)
        ax.vlines([v - e, v + e], y - cap_width, y + cap_width,
                  colors="black", lw=1, alpha=0.7, zorder=1)

    # Use colors that match the legend - positive and negative values from color_definitions
    colors = [pos_color if x > 0 else neg_color for x in bar_fold_df["Fold_Mean_SHAP"]]
    ax.barh(ys, bar_fold_df["Fold_Mean_SHAP"], height=0.7, color=colors, alpha=1, zorder=2)

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
    epsilon = (xmax - xmin) * 0.01
    for y, (_, row) in zip(ys, bar_fold_df.iterrows()):
        v = row["Fold_Mean_SHAP"]
        e = row["Fold_SHAP_StdDev"]
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

    # Adjust x-axis limits to ensure labels fit
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin*1.2, xmax*1.2)  # Add 20% padding on both sides

    # Set labels and grid
    ax.set_xlabel("SHAP Impact (Fold-Level)")
    ax.set_ylabel("Feature")
    
    # Update title to clearly indicate top/bottom features if both are present
    if len(neg_fold) > 0 and len(pos_fold) > 0:
        ax.set_title(f"Fold-Level SHAP Impact (Top {min(top_n, len(pos_fold))} Positive & Top {min(top_n, len(neg_fold))} Negative Features)")
    else:
        ax.set_title(f"Fold-Level SHAP Impact (Top {top_n} Features)")
    
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

    plt.tight_layout()
    
    # Move legend to the bottom left corner as requested
    ax.legend(handles=legend_elements, loc='lower left', 
              frameon=True, fontsize=9, framealpha=0.9)

    # Save plot with informative filename
    plt.savefig(os.path.join(main_output_dir, "shap_bar_pos_neg_impact.png"),
                bbox_inches="tight", pad_inches=0.1)
    plt.close()
    
    # Create an additional bar plot sorted by absolute SHAP value
    # We need to pass fold_level_importance_df to ensure consistent feature selection with beeswarm plot
    create_absolute_sorted_bar_plot(fold_df, fold_level_importance_df, main_output_dir, pos_color, neg_color, top_n)
    # Calculate correlation between SHAP values and feature values
    fold_level_corrs = {}
    
    if problem_type == "classification":
        # Set correlations to 0 for classification - not meaningful in this context
        for feature in overall_X_test.columns:
            fold_level_corrs[feature] = 0
    else:
        # Calculate fold-level correlations for regression
        for feature in overall_X_test.columns:
            feature_corrs = []
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
                    
                    feature_corrs.append(corr)
            
            # Store the average correlation for this feature
            fold_level_corrs[feature] = np.mean(feature_corrs) if feature_corrs else 0
            
    # Add correlation columns to main dataframe - only for regression problems
    if problem_type != "classification":
        shap_signed_df["Fold_Level_Correlation"] = pd.Series(fold_level_corrs)
    
    # Add fold-level rank columns only
    # Removed ranking columns as requested

    # -------------------------------------------------------------------------
    # Generate correlation plots for regression problems
    # -------------------------------------------------------------------------
    if problem_type == "regression":
        
        # Only fold-level correlation plots are used
        
        # Fold-level correlation plot
        corr_fold_df = shap_signed_df.sort_values("Fold_Level_Correlation", key=abs, ascending=False).head(25)
        fig, ax = plt.subplots(figsize=(10, max(6, len(corr_fold_df) * 0.2)))
        
        # Create the colors based on fold-level correlations
        colors_fold = [pos_color if x > 0 else neg_color for x in corr_fold_df["Fold_Level_Correlation"]]
        
        # Draw bars
        ax.barh(range(len(corr_fold_df)), corr_fold_df["Fold_Level_Correlation"], color=colors_fold)
        ax.set_yticks(range(len(corr_fold_df)))
        ax.set_yticklabels(corr_fold_df.index)
        ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Correlation between SHAP values and target")
        ax.set_title("Features by SHAP-Target Correlation (Fold Level)")
        
        # Add error bars for standard deviation
        for i, (feat, row) in enumerate(corr_fold_df.iterrows()):
            std = row.get("Fold_SHAP_StdDev", 0)
            ax.plot([row["Fold_Level_Correlation"] - std, row["Fold_Level_Correlation"] + std], 
                   [i, i], color="black", linewidth=1, zorder=3)
            ax.plot([row["Fold_Level_Correlation"] - std, row["Fold_Level_Correlation"] - std], 
                   [i - 0.2, i + 0.2], color="black", linewidth=1, zorder=3)
            ax.plot([row["Fold_Level_Correlation"] + std, row["Fold_Level_Correlation"] + std], 
                   [i - 0.2, i + 0.2], color="black", linewidth=1, zorder=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(main_output_dir, "shap_corr_bar_fold.png"), 
                   bbox_inches="tight")
        plt.close()
        # Removed excessive success message

    # -------------------------------------------------------------------------
    # Save detailed CSV
    # -------------------------------------------------------------------------
    # For classification problems, remove the correlation columns
    output_df = shap_signed_df.copy()
    
    # For classification, drop correlation columns
    if problem_type == "classification":
        if "Fold_Level_Correlation" in output_df.columns:
            output_df = output_df.drop(columns=["Fold_Level_Correlation"])
    
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
    
    print("Saving SHAP analysis to CSV...")
    # Use output_df which has columns removed as needed
    output_df.to_csv(os.path.join(main_output_dir, "shap_features_analysis.csv"), index_label="Feature")
    
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
        print("Creating SHAP‑target correlation plots...")
        
        # Create fold-level correlation DataFrame
        corr_df = pd.DataFrame.from_dict(fold_level_corrs, orient='index', columns=['Correlation_With_Target'])
        corr_df.index.name = 'Feature'
        corr_df.sort_values('Correlation_With_Target', ascending=False, inplace=True)
        
        # Get top and bottom features by correlation for visualization
        top_corr = corr_df.nlargest(top_n, "Correlation_With_Target")
        bottom_corr = corr_df.nsmallest(top_n, "Correlation_With_Target")
        corr_df_viz = pd.concat([top_corr, bottom_corr]).sort_values("Correlation_With_Target", ascending=False)
        
        # Create colormap for better visualization
        vmax = corr_df_viz["Correlation_With_Target"].abs().max()
        norm = plt.Normalize(-vmax, vmax)
        colors = [feature_cmap(norm(v)) for v in corr_df_viz["Correlation_With_Target"]]
        # Only save correlation CSV for regression tasks
        if problem_type == "regression":
            # Sample-level CSV removed
            pass
        
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
        
        # Add value annotations on bars for fold-level
        # Fix variable naming to use corr_df_viz instead of fold_corr_viz
        # Calculate padding based on the data range
        pad_fold = max(0.05, (vmax - (-vmax)) * 0.05)  # 5% of the data range
        offset_fold = pad_fold * 0.25
        for i, v in enumerate(corr_df_viz["Correlation_With_Target"]):
            x_text = v + (offset_fold if v >= 0 else -offset_fold)
            ha = "left" if v >= 0 else "right"
            axc.text(x_text, i, f"{v:.3f}", va="center", ha=ha, fontsize=8)
        
        # Add styling elements for fold-level
        axc.axvline(0, color="grey", ls="--", lw=1)
        
        # Save and close
        fig.tight_layout()
        fig.savefig(os.path.join(main_output_dir, f"shap_corr_bar_fold.png"), bbox_inches="tight")
        plt.close(fig)
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
                
                # Individual feature correlations are already aggregated during the main fold-level correlation calculations
        
        # Create DataFrame from dict and sort by correlation value
        fold_corr_df = pd.DataFrame.from_dict(fold_level_corrs, orient='index', columns=['Correlation_With_Target'])
        fold_corr_df.index.name = 'Feature'
        fold_corr_df.sort_values('Correlation_With_Target', ascending=False, inplace=True)
        
        # Only save correlation CSV for regression tasks
        if problem_type == "regression":
            fold_corr_df.to_csv(os.path.join(main_output_dir, f'shap_corr_fold.csv'))
        
        # Prepare fold-level correlation visualization
        has_differences = True
        
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
    
    # Make sure to return the dataframe so it can be used for summary generation
    return shap_signed_df.copy()

def create_absolute_sorted_bar_plot(fold_df, fold_level_importance_df, main_output_dir, pos_color, neg_color, top_n=None):
    # Use default from color_definitions if not specified
    # SHAP_TOP_N_FEATURES is imported at the top of the file
    if top_n is None:
        top_n = SHAP_TOP_N_FEATURES
    """
    Create a bar plot of features sorted by absolute SHAP value, while maintaining directionality.
    
    Args:
        fold_df: DataFrame with fold-level SHAP values
        fold_level_importance_df: DataFrame with features sorted by fold-level impact 
        main_output_dir: Output directory path
        pos_color: Color for positive SHAP values
        neg_color: Color for negative SHAP values
        top_n: Number of top features to include
    """
    # Use the same feature selection as the beeswarm plot for consistency,
    # but sort properly by absolute SHAP values
    
    # First get the same features used in the beeswarm plot
    top_features = fold_level_importance_df.sort_values('Fold_Level_Impact', ascending=False).index.tolist()
    
    # Filter to top_n features and ensure they're in fold_df
    top_features = [f for f in top_features if f in fold_df.index][:top_n]
    
    # Select those features from fold_df
    abs_fold_df = fold_df.loc[top_features].copy()
    
    # Now create a temporary column for absolute values and sort properly
    abs_fold_df['abs_shap'] = abs_fold_df['Fold_Mean_SHAP'].abs()
    
    # Sort by absolute SHAP value in ascending order (smallest absolute values first)
    abs_fold_df = abs_fold_df.sort_values('abs_shap', ascending=True)
    
    # Drop the temporary column
    abs_fold_df = abs_fold_df.drop(columns=['abs_shap'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(abs_fold_df) * 0.25)))
    
    # Set up y-coordinates
    ys = np.arange(len(abs_fold_df))
    
    # Draw bars with color based on directionality
    colors = [pos_color if x > 0 else neg_color for x in abs_fold_df["Fold_Mean_SHAP"]]
    ax.barh(ys, abs_fold_df["Fold_Mean_SHAP"], height=0.7, color=colors, alpha=1, zorder=2)
    
    # Add error bars
    for i, (_, row) in enumerate(abs_fold_df.iterrows()):
        ax.errorbar(row["Fold_Mean_SHAP"], i, 
                   xerr=row["Fold_SHAP_StdDev"], 
                   fmt="none", ecolor="black", capsize=3, capthick=1, zorder=3)
    
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
    epsilon = (xmax - xmin) * 0.01
    for y, (_, row) in zip(ys, abs_fold_df.iterrows()):
        v = row["Fold_Mean_SHAP"]
        e = row["Fold_SHAP_StdDev"]
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
    
    # Adjust x-axis limits to ensure labels fit
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin*1.2, xmax*1.2)  # Add 20% padding on both sides
    
    # Set labels and grid
    ax.set_xlabel("SHAP Impact (Fold-Level)")
    ax.set_ylabel("Feature")
    ax.set_title(f"Fold-Level SHAP Impact (Top {top_n} Features by Absolute Value)")
    
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
    plt.savefig(os.path.join(main_output_dir, "shap_bar_impact.png"),
                bbox_inches="tight", pad_inches=0.1)
    plt.close()

# Function save_top_features_summary is now in feature_explanation.py
