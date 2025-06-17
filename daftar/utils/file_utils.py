"""File handling utilities for managing outputs and documentation."""

import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union


def combine_metrics_for_fold(fold_dir: Path, fold_idx: int, test_metrics_df: pd.DataFrame) -> Path:
    """
    Create a combined metrics file for a specific fold that includes both test metrics
    and hyperparameter tuning metrics in the same format as metrics_all_folds.csv.
    
    Args:
        fold_dir: Path to the fold directory
        fold_idx: Fold index
        test_metrics_df: DataFrame containing test metrics for this fold
        
    Returns:
        Path to the created combined metrics file
    """
    # Output file path
    metrics_file_path = fold_dir / f"metrics_fold_{fold_idx}.csv"
    
    # Create the new combined DataFrame with the same format as metrics_all_folds.csv
    headers = ['Fold', 'TEST METRICS BY FOLD', '', '', '', 'HYPERPARAMETER TUNING METRICS', '', '', '']
    combined_df = pd.DataFrame(columns=headers)
    
    # Add subheaders
    subheaders = ['', 'mse', 'rmse', 'mae', 'r2', 'Training MSE', 'Validation MSE', 'Gap', 'Hyperparameters']
    combined_df.loc[0] = subheaders
    
    # Check if we have hyperparameter tuning data for this fold
    hp_tuning_file = fold_dir / f"hyperparam_tuning_fold_{fold_idx}.txt"
    training_mse = ''
    validation_mse = ''
    gap = ''
    hyperparameters = ''
    
    if hp_tuning_file.exists():
        # Extract parameters from the hyperparameter tuning summary file
        try:
            with open(hp_tuning_file, 'r') as f:
                content = f.read()
                
                # Extract training and validation metrics and gap
                training_match = re.search(r'Training Metric:\s*([\d\.e\-]+)', content)
                validation_match = re.search(r'Validation Metric:\s*([\d\.e\-]+)', content)
                gap_match = re.search(r'Training/Val Gap:\s*([\d\.e\-]+)', content)
                
                if training_match:
                    training_mse = float(training_match.group(1))
                
                if validation_match:
                    validation_mse = float(validation_match.group(1))
                
                if gap_match:
                    gap = float(gap_match.group(1))
                elif training_match and validation_match:
                    # Calculate gap if not directly available
                    try:
                        gap = abs(float(validation_match.group(1)) - float(training_match.group(1)))
                    except:
                        gap = ''
                
                # Extract best hyperparameters
                best_params_section = re.search(r'Best Hyperparameters:[\s\-]+(.+?)(?:\n\n|$)', content, re.DOTALL)
                if best_params_section:
                    params_text = best_params_section.group(1).strip()
                    param_lines = [line.strip() for line in params_text.split('\n') if line.strip()]
                    
                    # Convert the list of parameter lines to a formatted string
                    param_pairs = []
                    for line in param_lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            param_pairs.append(f"{key.strip()}={value.strip()}")
                    
                    hyperparameters = ", ".join(param_pairs)
        except Exception as e:
            print(f"Error reading hyperparameter tuning file: {e}")
    
    # Get test metrics values
    mse = test_metrics_df['mse'].values[0] if 'mse' in test_metrics_df.columns else ''
    rmse = test_metrics_df['rmse'].values[0] if 'rmse' in test_metrics_df.columns else ''
    mae = test_metrics_df['mae'].values[0] if 'mae' in test_metrics_df.columns else ''
    r2 = test_metrics_df['r2'].values[0] if 'r2' in test_metrics_df.columns else ''
    
    # Add fold data row
    new_row = [fold_idx, mse, rmse, mae, r2, training_mse, validation_mse, gap, hyperparameters]
    combined_df.loc[1] = new_row
    
    # Save to file
    combined_df.to_csv(metrics_file_path, index=False)
    
    return metrics_file_path


def combine_metrics_files(output_dir: Path, test_metrics_df=None) -> Path:
    """
    Combine hyperparameter tuning metrics and test metrics into a single horizontal table.
    This file is saved as metrics_all_folds.csv.
    
    Args:
        output_dir: Path to the directory containing the metrics files
        test_metrics_df: Optional DataFrame containing test metrics (if provided, no file is read)
        
    Returns:
        Path to the created combined metrics file
    """
    # File paths
    metrics_file_path = output_dir / "metrics_all_folds.csv"
    old_tuning_metrics_path = output_dir / "hyperparameter_tuning_metrics_summary.csv"
    
    # Use provided test_metrics_df or return None if not available
    if test_metrics_df is None:
        return None
    else:
        test_df = test_metrics_df
    
    # Read tuning metrics from individual fold metrics files
    tuning_df = pd.DataFrame(columns=['Fold', 'Training MSE', 'Validation MSE', 'Gap', 'Hyperparameters'])
    
    # Look for fold metrics files which contain the hyperparameter tuning data
    for fold_idx in range(1, 21):  # Check a reasonable range of folds
        fold_metrics_path = output_dir / f"fold_{fold_idx}" / f"metrics_fold_{fold_idx}.csv"
        
        if fold_metrics_path.exists():
            try:
                # Read fold metrics file
                fold_df = pd.read_csv(fold_metrics_path)
                
                # Skip if it's just a header row or empty
                if len(fold_df) < 2:
                    continue
                    
                # Extract hyperparameter tuning metrics - should be in row 1, columns 5-8
                if len(fold_df.columns) >= 9:  # Make sure we have enough columns
                    # Get training MSE, validation MSE, gap, and hyperparameters
                    tuning_data = {
                        'Fold': fold_idx,
                        'Training MSE': fold_df.iloc[1, 5],  # 6th column
                        'Validation MSE': fold_df.iloc[1, 6],  # 7th column
                        'Gap': fold_df.iloc[1, 7],            # 8th column
                        'Hyperparameters': fold_df.iloc[1, 8]  # 9th column
                    }
                    tuning_df = pd.concat([tuning_df, pd.DataFrame([tuning_data])], ignore_index=True)
            except Exception as e:
                print(f"Error reading fold metrics file {fold_metrics_path}: {e}")
                # Continue without this fold's tuning metrics
    
    # Filter out the summary rows from test_df
    # First check if fold column can be converted to numeric
    try:
        test_df['fold_numeric'] = pd.to_numeric(test_df['fold'], errors='coerce')
        fold_rows = test_df[test_df['fold_numeric'].notna()].copy()
        summary_rows = test_df[test_df['fold_numeric'].isna()].copy()
    except Exception:
        # Fallback to string-based filtering
        is_fold = test_df['fold'].apply(lambda x: 
                                        isinstance(x, (int, float)) and not pd.isna(x) or 
                                        (isinstance(x, str) and x.isdigit()))
        fold_rows = test_df[is_fold].copy()
        summary_rows = test_df[~is_fold].copy()
    
    # Create the new combined DataFrame with TEST METRICS first
    headers = ['Fold', 'TEST METRICS BY FOLD', '', '', '', 'HYPERPARAMETER TUNING METRICS', '', '', '']
    combined_df = pd.DataFrame(columns=headers)
    
    # Add subheaders
    # Determine if we're dealing with regression or classification based on metrics present
    is_regression = 'mse' in fold_rows.columns
    
    if is_regression:
        subheaders = ['', 'mse', 'rmse', 'mae', 'r2', 'Training MSE', 'Validation MSE', 'Gap', 'Hyperparameters']
    else:  # Classification metrics
        subheaders = ['', 'accuracy', 'f1', 'roc_auc', '', 'Training Accuracy', 'Validation Accuracy', 'Gap', 'Hyperparameters']
    
    combined_df.loc[0] = subheaders
    
    # Add data rows
    for idx, row in fold_rows.iterrows():
        # Safely convert fold to integer
        if 'fold_numeric' in fold_rows.columns:
            fold_num = int(row['fold_numeric'])
        else:
            # Try to convert to int safely
            try:
                fold_num = int(float(row['fold']))
            except (ValueError, TypeError):
                fold_num = str(row['fold'])
        
        # Get metrics based on regression or classification        
        if is_regression:
            new_row = [fold_num, row['mse'], row['rmse'], row['mae'], row['r2']]
        else:  # Classification
            new_row = [fold_num]
            # Get classification metrics with fallbacks
            for metric in ['accuracy', 'f1', 'roc_auc']:
                new_row.append(row.get(metric, ''))
            new_row.append('')  # Empty cell for alignment
        
        # Add hyperparameter tuning data if available
        if tuning_df is not None and not tuning_df.empty:
            tuning_row = tuning_df[tuning_df['Fold'] == fold_num]
            if not tuning_row.empty:
                if is_regression:
                    new_row.extend([
                        tuning_row['Training MSE'].values[0], 
                        tuning_row['Validation MSE'].values[0], 
                        tuning_row['Gap'].values[0], 
                        tuning_row['Hyperparameters'].values[0]])
                else:
                    # For classification, use different metrics if available
                    # Safely handle potential string or Series values
                    train_metric = tuning_row.get('Training Accuracy', tuning_row.get('Training MSE', '').values[0] if hasattr(tuning_row.get('Training MSE', ''), 'values') else '')
                    train_metric = train_metric.values[0] if hasattr(train_metric, 'values') else train_metric
                    
                    val_metric = tuning_row.get('Validation Accuracy', tuning_row.get('Validation MSE', '').values[0] if hasattr(tuning_row.get('Validation MSE', ''), 'values') else '')
                    val_metric = val_metric.values[0] if hasattr(val_metric, 'values') else val_metric
                    
                    gap = tuning_row['Gap'].values[0]
                    hyperparams = tuning_row['Hyperparameters'].values[0]
                    
                    new_row.extend([train_metric, val_metric, gap, hyperparams])
            else:
                new_row.extend(['', '', '', ''])
        else:
            new_row.extend(['', '', '', ''])
            
        combined_df.loc[len(combined_df)] = new_row
    
    # Add summary rows (mean, std, min, max)
    for idx, row in summary_rows.iterrows():
        summary_label = row['fold']
        
        # Get metrics based on regression or classification
        if is_regression:
            new_row = [summary_label, row['mse'], row['rmse'], row['mae'], row['r2'], '', '', '', '']
        else:  # Classification
            new_row = [summary_label]
            # Get classification metrics with fallbacks
            for metric in ['accuracy', 'f1', 'roc_auc']:
                new_row.append(row.get(metric, ''))
            # Add remaining empty cells
            new_row.extend(['', '', '', '', ''])
            
        combined_df.loc[len(combined_df)] = new_row
    
    # Save to file with the new name
    combined_df.to_csv(metrics_file_path, index=False)
    
    return metrics_file_path


def save_figures_explanation(shap_df=None, output_dir=None, prefix="", problem_type="regression", 
                           top_n=15, filename=None):
    """Save a summary of top features by SHAP value to a text file, with detailed explanation
    of SHAP metrics and visualizations.
    
    Args:
        shap_df: DataFrame with SHAP analysis (optional if only creating general guide)
        output_dir: Directory to save output
        prefix: Prefix for the output file
        problem_type: Type of problem ('regression' or 'classification') - only used for legacy compatibility
        top_n: Number of top features to display
        filename: Name of the output file (defaults to different filenames based on content)
        
    Returns:
        Path to the created explanation file
    """
    # Determine appropriate filename based on content
    if filename is None:
        if shap_df is not None:
            filename = "shap_top_features_summary.txt"
        else:
            filename = "output_files_explanation.txt"
    
    # Create output path
    output_path = Path(output_dir) / filename
    
    # Open file and write header
    with open(output_path, "w") as f:
        # For explanation files with SHAP data, show top features
        if shap_df is not None and "shap_top_features" in filename:
            f.write("="*80 + "\n")
            f.write(f"TOP {top_n} FEATURES WITH POSITIVE EFFECT\n")
            f.write("="*80 + "\n\n")
            pos_features = shap_df[shap_df["Mean_SHAP"] > 0].sort_values("Mean_SHAP", ascending=False).head(top_n)
            for i, (feature, row) in enumerate(pos_features.iterrows(), 1):
                value = row["Mean_SHAP"]
                stddev = row["SHAP_StdDev"]
                f.write(f"{i}. {feature}: {value:.6f} ± {stddev:.6f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"TOP {top_n} FEATURES WITH NEGATIVE EFFECT\n")
            f.write("="*80 + "\n\n")
            neg_features = shap_df[shap_df["Mean_SHAP"] < 0].sort_values("Mean_SHAP", ascending=True).head(top_n)
            for i, (feature, row) in enumerate(neg_features.iterrows(), 1):
                value = row["Mean_SHAP"]
                stddev = row["SHAP_StdDev"]
                f.write(f"{i}. {feature}: {value:.6f} ± {stddev:.6f}\n")
            
            f.write("\n\n")
        
        # For output_files_explanation.txt, include comprehensive file structure guide
        elif "output_files_explanation" in filename:
            f.write("="*80 + "\n")
            f.write("For quick results reporting, use top_shap_bar_plot.png\n")
            f.write("="*80 + "\n")
            f.write("OUTPUT FILES STRUCTURE\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. OUTPUT DIRECTORIES\n")
            f.write("-" * 40 + "\n")
            f.write("- Main directory: Contains overall metrics and summary files\n")
            f.write("- feature_importance: Contains feature importance data and visualizations\n")
            f.write("- fold_X: Contains results specific to each cross-validation fold\n")
            f.write("- fold_X/optuna_plots: Hyperparameter optimization visualizations\n")
            f.write("- shap_feature_interactions: Feature interaction analysis (XGBoost regression)\n")
            f.write("- per_class_shap: Class-specific SHAP analysis (classification only)\n\n")
            
            # Model-specific visualizations section
            f.write("2. MODEL-SPECIFIC VISUALIZATIONS\n")
            f.write("-" * 40 + "\n")
            f.write("Regression-specific outputs:\n")
            f.write("- density_plot_overall.png: Actual vs predicted value density plot\n")
            f.write("- fold_X/density_plot_fold_X.png: Fold-specific density plots\n\n")
            
            f.write("Classification-specific outputs:\n")
            f.write("- per_class_shap/class_X_top_shap_bar_plot.png: SHAP magnitude plots per class\n")
            f.write("- per_class_shap/multiclass_comparison.png: Cross-class comparison\n")
            f.write("- confusion_matrix_global.png: Overall confusion matrix\n")
            f.write("- fold_X/confusion_matrix_fold_X.png: Fold-specific confusion matrices\n\n")
            
            f.write("3. SHAP VISUALIZATIONS\n")
            f.write("-" * 40 + "\n")
            f.write("- top_shap_beeswarm_plot.png: SHAP values for all features and samples\n")
            f.write("- top_shap_bar_plot.png: Features ranked by SHAP magnitude - good for reporting results\n")
            f.write("- top_shap_bar_pos_neg.png: Features split by positive/negative SHAP values\n")
            f.write("- shap_raw_all_folds.csv: Complete SHAP values for all features across folds\n")
            f.write("- shap_feature_interactions/: Detailed analysis of feature interactions (XGBoost regression)\n")
            f.write("  - interaction_heatmap.png: Heatmap of strongest feature interactions\n")
            f.write("  - interaction_network.png: Network visualization of strongest feature interactions\n")
            f.write("  - top_bottom_network.png: Network of top/bottom features with interactions\n\n")
            
            f.write("4. KEY TEXT FILES\n")
            f.write("-" * 40 + "\n")
            f.write("- performance.txt: Summary of model performance metrics\n")
            f.write("- shap_top_features_summary.txt: Summary of top positive/negative SHAP-scored features\n")
            f.write("- metrics_all_folds.csv: Complete metrics for all folds\n")
            f.write("- predictions_vs_actual_overall.csv: Predictions across all folds and the actual values\n")
            f.write("- fold_X/feature_importance_fold_X.csv: Per-fold feature importance values\n")
            f.write("- fold_X/predictions_vs_actual_fold_X.csv: Per-fold predictions\n")
            f.write("- fold_X/test_train_splits_fold_X.csv: Sample IDs in each fold\n")
            f.write("- fold_X/shap_interactions_fold_X.csv: Per-fold feature interactions (XGBoost)\n")
        
    return output_path

def save_shap_values(fold_results: List[Dict[str, Any]], output_dir: Path) -> None:
    """Save SHAP values from all folds to CSV files.
    
    Args:
        fold_results: List of fold results
        output_dir: Output directory path
    """
    import numpy as np
    import pandas as pd
    
    # Concatenate SHAP values from all folds
    all_samples_shap_values = []
    
    for fold_idx, fold_result in enumerate(fold_results):
        shap_values = fold_result['shap_values']
        X_test = fold_result['X_test']
        y_test = fold_result['y_test']
        feature_names = fold_result['feature_importances'].index.tolist()
        
        # Use original IDs if available, otherwise use test IDs
        if 'original_ids' in fold_result and fold_result['original_ids'] is not None:
            sample_ids = fold_result['original_ids']
        else:
            sample_ids = fold_result['ids_test']
        
        # Skip if no SHAP values (shouldn't happen)
        if shap_values is None or len(shap_values) == 0:
            continue
            
        # Handle different SHAP value shapes based on problem type
        # Classification models can have multiple formats based on the SHAP explainer
        if len(shap_values.shape) == 3:  # shape: (n_classes, n_samples, n_features)
            # For multiclass, we'll just use the values for class 1 (positive class)
            shap_values = shap_values[1]  # Select positive class SHAP values
        
        # Make sure we don't exceed the bounds of arrays
        n_samples = min(len(shap_values), len(sample_ids), len(y_test))
        
        # For each sample, collect all feature SHAP values
        for i in range(n_samples):
            sample_id = sample_ids[i]
            target_value = y_test[i]
            sample_shap = shap_values[i]
            
            # Handle feature dimension safely
            n_features = min(len(sample_shap), len(feature_names))
            
            # Store all SHAP values for this sample
            sample_shap_dict = {feature_names[j]: sample_shap[j] for j in range(n_features)}
            
            # Add metadata
            sample_shap_dict['ID'] = sample_id  # Use a consistent ID column name
            sample_shap_dict['Fold'] = fold_idx + 1  # 1-indexed folds
            sample_shap_dict['Target'] = target_value  # Add target value
            
            # Append to the list of samples
            all_samples_shap_values.append(sample_shap_dict)
    
    # Create DataFrame with one row per sample, features as columns
    shap_df = pd.DataFrame(all_samples_shap_values)
    
    # Collect all possible feature names from all folds
    all_feature_names = set()
    for fold_result in fold_results:
        if 'shap_data' in fold_result and fold_result['shap_data'] is not None:
            # Get feature names from X_test in shap_data
            fold_X_test = fold_result['shap_data'][1]
            if fold_X_test is not None and hasattr(fold_X_test, 'columns'):
                all_feature_names.update(fold_X_test.columns)
    
    # Ensure all features are present in the dataframe - avoid fragmentation
    missing_features = list(all_feature_names - set(shap_df.columns))
    if missing_features:
        # Create a DataFrame with zeros for missing features
        missing_df = pd.DataFrame(0.0, index=shap_df.index, columns=missing_features)
        # Concat with original DataFrame to avoid fragmentation
        shap_df = pd.concat([shap_df, missing_df], axis=1)
    
    # Reorder columns to put ID, Fold, and Target first
    metadata_cols = ['ID', 'Fold', 'Target']
    other_cols = sorted([col for col in shap_df.columns if col not in metadata_cols])
    ordered_cols = metadata_cols + other_cols
    shap_df = shap_df[ordered_cols]
    
    # Save to CSV
    shap_df.to_csv(output_dir / 'shap_values_all_folds.csv', index=False)
    
    # Save per-fold SHAP values
    for fold_idx, fold_result in enumerate(fold_results):
        fold_dir = output_dir / f"fold_{fold_idx+1}"
        fold_dir.mkdir(exist_ok=True)
        
        # Filter for this fold
        fold_shap_df = shap_df[shap_df['Fold'] == fold_idx + 1].copy()
        # Use the same column ordering as the main SHAP CSV
        fold_shap_df.to_csv(fold_dir / f"shap_values_fold_{fold_idx+1}.csv", index=False)
