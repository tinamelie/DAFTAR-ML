"""
Per-class SHAP explainer for classification models.

This module provides functionality to show which features contribute most to
specific classes in binary and multiclass classification models, including:
- Feature importance strength per class
- Direction of influence (towards or away from each class)
- Visualizations for easier interpretation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import logging
from daftar.viz.colors import MULTICLASS_COLORS, get_per_class_shap_colors

# Configure logging
LOGGER = logging.getLogger(__name__)

def _get_shap_values(model, X):
    """
    Extract per-class SHAP values independent of SHAP version.
    
    Args:
        model: Trained classifier model
        X: Data to explain (DataFrame or ndarray)
        
    Returns:
        tuple: (shap_values, base_values)
            - shap_values: array of shape (n_classes, n_samples, n_features)
            - base_values: Expected values per class
    """
    # Create explainer - use model_output="probability" for per-class values
    explainer = shap.TreeExplainer(model, model_output="probability")
    
    # Handle different SHAP API versions
    shap_out = explainer(X)  # SHAP ≥ 0.41 returns object
    
    if hasattr(shap_out, "values"):  # New SHAP API
        # Shape (n_samples, n_classes, n_features) → (n_classes, n_samples, n_features)
        S = shap_out.values
        S = np.moveaxis(S, 1, 0)
    else:  # Old SHAP API returns list of length n_classes
        S = np.asarray(explainer.shap_values(X))  # (n_classes, n_samples, n_features)
    
    # Get base/expected values
    base = explainer.expected_value
    
    return S, base

def _per_class_tables(S, feature_names):
    """
    Create tables of per-class SHAP statistics.
    
    Args:
        S: SHAP values array (n_classes, n_samples, n_features)
        feature_names: List of feature names
        
    Returns:
        list: DataFrames, one per class with columns:
            - feature: Feature name
            - mean_|SHAP|: Importance magnitude
            - mean_SHAP: Signed importance (direction)
            - rank_by_|SHAP|: Rank by absolute importance
            - rank_by_SHAP: Rank by signed importance
    """
    tables = []
    for c, s in enumerate(S):  # s → (n_samples, n_features)
        # Compute mean absolute and signed SHAP values
        mean_abs = np.mean(np.abs(s), axis=0)
        mean_signed = np.mean(s, axis=0)
        
        # Create DataFrame with results
        df = pd.DataFrame({
            "feature": feature_names,
            "mean_|SHAP|": mean_abs,
            "mean_SHAP": mean_signed
        })
        
        # Add ranking columns
        df["rank_by_|SHAP|"] = df["mean_|SHAP|"].rank(ascending=False)
        df["rank_by_SHAP"] = df["mean_SHAP"].rank(ascending=False)
        
        # Sort by importance and append to result list
        tables.append(df.sort_values("mean_|SHAP|", ascending=False))
    
    return tables

def _create_barplot(df, class_id, max_display, out_dir):
    """
    Create bar plot showing top positive and negative SHAP impact features for a class.
    
    Args:
        df: DataFrame with per-class SHAP statistics
        class_id: Class identifier (label)
        max_display: Maximum number of features to display (will show max_display/2 positive and negative)
        out_dir: Output directory
    """
    # Sort by signed SHAP values to get positive and negative impact features
    pos_features = df.sort_values("mean_SHAP", ascending=False)
    neg_features = df.sort_values("mean_SHAP", ascending=True)
    
    # Get colors from the consolidated color module
    colors_config = get_per_class_shap_colors()
    
    # Get top positive and negative features
    n_each = max(2, max_display // 2)  # At least 2 features each way
    top_pos = pos_features.head(n_each)
    top_neg = neg_features.head(n_each)
    
    # Combine and sort by absolute impact
    combined = pd.concat([top_pos, top_neg])
    combined = combined.drop_duplicates().sort_values("mean_|SHAP|", ascending=True)
    
    # Create plot
    plt.figure(figsize=(10, combined.shape[0] * 0.3 + 1.5))
    
    # Plot bars colored by direction
    bars = plt.barh(combined["feature"], combined["mean_SHAP"])
    
    # Color bars by direction
    for i, bar in enumerate(bars):
        value = combined.iloc[i]["mean_SHAP"]
        bar.set_color(colors_config["positive"] if value > 0 else colors_config["negative"])
    
    # Add vertical line at zero
    plt.axvline(0, color="black", lw=1)
    
    # Add legend
    import matplotlib.patches as mpatches
    pos_patch = mpatches.Patch(color=colors_config["positive"], label="Positive impact")
    neg_patch = mpatches.Patch(color=colors_config["negative"], label="Negative impact")
    plt.legend(handles=[pos_patch, neg_patch])
    
    # Add labels and title
    plt.title(f"Class {class_id}: Top SHAP Impact Features")
    plt.xlabel("SHAP Value (impact on model output)")
    plt.tight_layout()
    
    # Save and close
    plt.savefig(Path(out_dir) / f"class_{class_id}_shap_impact.png", dpi=300)
    plt.close()

# Summary plot function removed as requested

def report(model, X, *, feature_names=None, max_display=20, out_dir="shap_class_report", 
           include_summary_plot=True):
    """
    Generate per-class SHAP reports for classification models.
    
    For each class, this function:
    - Computes class-specific SHAP values
    - Creates CSV files with feature importance and direction
    - Generates visualization plots
    - Prints a summary of top features per class
    
    Args:
        model: Trained classifier model
        X: Data to explain (DataFrame or ndarray)
        feature_names: Optional list of feature names (default: derived from X)
        max_display: Maximum number of features to show in plots (default: 20)
        out_dir: Directory for output files (default: "shap_class_report")
        include_summary_plot: Whether to include SHAP summary plots (default: True)
        
    Returns:
        list: DataFrames with per-class SHAP statistics
    """
    # Create output directory
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    
    # Get feature names if not provided
    if feature_names is None:
        if hasattr(X, "columns"):  # DataFrame
            feature_names = list(X.columns)
        else:  # ndarray
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Compute SHAP values for all classes
    LOGGER.info("Computing per-class SHAP values...")
    S, base = _get_shap_values(model, X)
    n_classes = len(S)
    
    # Generate per-class statistics
    LOGGER.info(f"Analyzing SHAP values for {n_classes} classes...")
    tables = _per_class_tables(S, feature_names)
    
    # Create output files and visualizations for each class
    for cls_idx, df in enumerate(tables):
        # Save CSV with statistics
        csv_path = Path(out_dir) / f"class{cls_idx}_shap_stats.csv"
        df.to_csv(csv_path, index=False)
        
        # Create bar plot
        _create_barplot(df, cls_idx, max_display, out_dir)
        
        # Create summary plot if requested
        if include_summary_plot:
            _create_summary_plot(S[cls_idx], X, cls_idx, out_dir)
    
    # Print summary of top features per class
    LOGGER.info("Top features per class (by |SHAP|):")
    for cls_idx, df in enumerate(tables):
        top_features = ", ".join(df.head(5)["feature"].tolist())
        LOGGER.info(f"  Class {cls_idx}: {top_features}")
    
    return tables

def explain_binary_model(model, X, feature_names=None, max_display=20, out_dir="shap_binary_report"):
    """
    Wrapper for binary classification that shows results for positive class (class 1).
    
    Args:
        model: Trained binary classifier
        X: Data to explain (DataFrame or ndarray)
        feature_names: Optional list of feature names
        max_display: Maximum number of features to show
        out_dir: Output directory
        
    Returns:
        DataFrame: SHAP statistics for positive class
    """
    tables = report(model, X, feature_names=feature_names, 
                    max_display=max_display, out_dir=out_dir)
    
    # For binary classification, typically class 1 (positive class) is of most interest
    LOGGER.info("For binary classification, focus on Class 1 (positive class) results")
    return tables[1]  # Return positive class table

def explain_multiclass_model(model, X, feature_names=None, max_display=15, out_dir="shap_multiclass_report"):
    """
    Wrapper for multiclass classification that focuses on feature contributions across all classes.
    
    Args:
        model: Trained multiclass classifier
        X: Data to explain (DataFrame or ndarray)
        feature_names: Optional list of feature names
        max_display: Maximum number of features to show
        out_dir: Output directory
        
    Returns:
        list: DataFrames with per-class SHAP statistics
    """
    tables = report(model, X, feature_names=feature_names, 
                    max_display=max_display, out_dir=out_dir)
    
    # For multiclass, create an additional plot showing top features across all classes
    _create_multiclass_comparison(tables, max_display, out_dir)
    
    return tables

def _create_multiclass_comparison(tables, class_labels, max_display, out_dir):
    """
    Create a comparison plot of top features across all classes.
    
    Args:
        tables: List of per-class SHAP statistic DataFrames
        class_labels: List of class labels
        max_display: Maximum number of features to display
        out_dir: Output directory
    """
    # Get global feature importance by combining across classes
    all_features = set()
    for df in tables:
        all_features.update(df["feature"].tolist())
    
    # Calculate global importance
    feature_to_importance = {}
    for feature in all_features:
        importance = 0
        for df in tables:
            if feature in df["feature"].values:
                importance += df.loc[df["feature"] == feature, "mean_|SHAP|"].values[0]
        feature_to_importance[feature] = importance
    
    # Sort features by importance
    sorted_features = sorted(feature_to_importance.items(), 
                           key=lambda x: x[1], reverse=True)[:max_display]
    
    # Get the top features for the plot
    top_features = [feature for feature, _ in sorted_features]
    
    # Create a single unified plot
    plt.figure(figsize=(10, max_display * 0.4 + 1.5))
    
    # Use multiclass colors from the consolidated color module
    class_colors = MULTICLASS_COLORS
    # Extend if more classes than colors
    while len(class_colors) < len(tables):
        class_colors.extend(class_colors)
    
    # Create data for directional bars (showing positive and negative values)
    pos_data = {cls_idx: [] for cls_idx in range(len(tables))}
    neg_data = {cls_idx: [] for cls_idx in range(len(tables))}
    
    # Collect positive and negative values for each feature and class
    for feature in top_features:
        for cls_idx, df in enumerate(tables):
            if feature in df["feature"].values:
                # Get actual value (with sign) to show direction
                value = df.loc[df["feature"] == feature, "mean_SHAP"].values[0]
                if value >= 0:
                    pos_data[cls_idx].append(value)
                    neg_data[cls_idx].append(0)
                else:
                    pos_data[cls_idx].append(0)
                    neg_data[cls_idx].append(abs(value))  # Store as positive for plotting
            else:
                pos_data[cls_idx].append(0)
                neg_data[cls_idx].append(0)
    
    # Create figure with vertical axis centered at 0
    plt.figure(figsize=(12, max_display * 0.4 + 1.5))
    
    # Plot bars for each class - positive values to the right, negative to the left
    for cls_idx, label in enumerate(class_labels):
        # Plot positive values (to the right)
        plt.barh(top_features, pos_data[cls_idx], color=class_colors[cls_idx],
                 label=f"Class {label}", alpha=0.8)
        
        # Plot negative values (to the left) - use negative numbers for left side
        plt.barh(top_features, [-x for x in neg_data[cls_idx]], color=class_colors[cls_idx],
                 alpha=0.8)  # No label to avoid duplicates in legend
    
    plt.title("SHAP Impact Across Classes")
    plt.xlabel("SHAP Value (negative ← 0 → positive)")
    # Add a vertical line at x=0 to mark the center
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    # Move legend to bottom left
    plt.legend(title="Class Contribution", loc='lower left')
    plt.tight_layout()
    
    # Save and close
    plt.savefig(Path(out_dir) / "multiclass_comparison.png", dpi=300)
    plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# Aggregator across folds
# ──────────────────────────────────────────────────────────────────────────────

def _standardize_shap_values(shap_vals, n_features):
    """Convert various possible shapes/lists returned by SHAP into
    np.ndarray with shape (n_classes, n_samples, n_features).
    """
    import numpy as np

    # Case 1 – list length = n_classes
    if isinstance(shap_vals, list):
        # Each element shape (n_samples, n_features)
        return np.stack([np.asarray(sv) for sv in shap_vals], axis=0)

    # Case 2 – ndarray 3-D
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        arr = shap_vals
        # Potential shapes:
        # (n_classes, n_samples, n_features)  → already good if last dim == n_features
        if arr.shape[2] == n_features:
            return arr
        # (n_samples, n_classes, n_features) → moveaxis 1 -> 0
        if arr.shape[1] == n_features:
            # unlikely but safeguard
            return np.moveaxis(arr, 2, 0)
        # (n_samples, n_classes, n_features) common for new SHAP
        return np.moveaxis(arr, 1, 0)

    raise ValueError("Unsupported shap_values format for standardisation")


def save_per_class_shap_analysis(
    fold_results: list,
    output_dir,
    *,
    feature_names: list | None = None,
    top_n: int = 20,
) -> dict:
    """Aggregate per-class SHAP statistics across folds and save artefacts.

    Args:
        fold_results: Output from pipeline with per-fold dictionaries.
        output_dir: Directory to put results in.
        feature_names: Optional list of feature names.
        top_n: Max number of features to display in bar plots.

    Returns:
        dict mapping artefact names → file paths.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd

    out_dir = Path(output_dir) / "per_class_shap"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- gather data
    all_S_list   = []   # per-fold SHAP arrays (c, n, f)
    all_X_list   = []   # per-fold X_test arrays (n, f)
    all_y_list   = []   # per-fold target values
    class_labels = None

    for fr in fold_results:
        shap_vals = fr.get("shap_values")
        X_test    = fr.get("X_test")
        y_test    = fr.get("y_test")
        
        if shap_vals is None or X_test is None:
            continue

        if feature_names is None:
            # Try to infer from this fold
            fnames = fr.get("feature_names") if fr.get("feature_names") else None
            if fnames:
                feature_names = fnames
        
        # Try to get class labels if not already found
        if class_labels is None and y_test is not None:
            try:
                # For classification problems we expect y_test to have the class labels
                unique_classes = np.unique(y_test)
                class_labels = [str(cl) for cl in unique_classes]
                LOGGER.info(f"Found class labels: {class_labels}")
            except Exception as e:
                LOGGER.warning(f"Could not extract class labels: {e}")

        try:
            S = _standardize_shap_values(shap_vals, X_test.shape[1])
        except Exception as e:
            LOGGER.warning(f"Skipping fold due to SHAP format error: {e}")
            continue

        all_S_list.append(S)
        all_X_list.append(X_test)
        if y_test is not None:
            all_y_list.append(y_test)

    if not all_S_list:
        raise RuntimeError("No SHAP values found in fold results for per-class analysis")

    # If we couldn't find class labels, use indices
    if class_labels is None or len(class_labels) != all_S_list[0].shape[0]:
        n_classes = all_S_list[0].shape[0]
        class_labels = [f"Class {i}" for i in range(n_classes)]
        LOGGER.info(f"Using default class labels: {class_labels}")

    # Concatenate along sample axis
    combined_S = np.concatenate(all_S_list, axis=1)  # shape (c, N_total, f)
    combined_X = np.concatenate(all_X_list, axis=0)   # shape (N_total, f)
    
    if all_y_list:
        combined_y = np.concatenate(all_y_list, axis=0)
    else:
        combined_y = None

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(combined_S.shape[2])]

    # ---------------------------------------------------------------- stats + plots
    tables = _per_class_tables(combined_S, feature_names)

    artefacts: dict[str, str] = {}
    
    # Create a single consolidated CSV with features as rows and classes as columns
    # First, create a dictionary to hold all feature data
    all_features = set()
    for df in tables:
        all_features.update(df['feature'].tolist())
    
    # Create a DataFrame with features as index
    consolidated_df = pd.DataFrame(index=sorted(all_features))
    
    # Add columns for each class's mean SHAP value
    for cls_idx, df in enumerate(tables):
        class_label = class_labels[cls_idx]
        # Convert to dictionary for easy mapping to features
        class_values = dict(zip(df['feature'], df['mean_SHAP']))
        # Add column with class mean SHAP values
        consolidated_df[f'Class_{class_label}_Mean_SHAP'] = consolidated_df.index.map(lambda x: class_values.get(x, 0))
    
    # Also add absolute value columns for easy reference
    for cls_idx, df in enumerate(tables):
        class_label = class_labels[cls_idx]
        # Convert to dictionary for easy mapping to features
        class_abs_values = dict(zip(df['feature'], df['mean_|SHAP|']))
        # Add column with class mean absolute SHAP values
        consolidated_df[f'Class_{class_label}_Mean_Abs_SHAP'] = consolidated_df.index.map(lambda x: class_abs_values.get(x, 0))
    
    # Save consolidated CSV
    consolidated_csv_path = out_dir / "all_classes_shap_stats.csv"
    consolidated_df.to_csv(consolidated_csv_path)
    artefacts["all_classes_stats"] = str(consolidated_csv_path)
    
    # Still create individual plots for each class
    for cls_idx, df in enumerate(tables):
        class_label = class_labels[cls_idx]
        # Bar plot
        _create_barplot(df, class_label, top_n, out_dir)
        artefacts[f"class_{class_label}_bar"] = str(out_dir / f"class_{class_label}_shap_impact.png")

    # Multiclass comparison if >1 class
    if len(tables) > 1:
        _create_multiclass_comparison(tables, class_labels, top_n, out_dir)
        artefacts["multiclass_comparison"] = str(out_dir / "multiclass_comparison.png")

    artefacts["per_class_shap_dir"] = str(out_dir)
    return artefacts
