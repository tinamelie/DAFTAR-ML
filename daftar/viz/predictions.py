"""Prediction visualization utilities for DAFTAR-ML."""

import os
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from daftar.viz.color_definitions import (
    DENSITY_ACTUAL_COLOR,
    DENSITY_PREDICTED_COLOR,
    DENSITY_ALPHA,
    CONFUSION_MATRIX_CMAP,
    CONFUSION_MATRIX_LINEWIDTH,
    CONFUSION_MATRIX_LINECOLOR
)


def determine_task_type(y: pd.Series) -> Tuple[bool, str]:
    """
    Decide if *y* is classification or regression.

    Criteria:
    1. Any pandas.Categorical dtype        -> classification
    2. Any non‑numeric / object dtype      -> classification
    3. Binary data (exactly 0/1 or True/False) -> classification
    4. Integer data with only a few values  -> classification
    5. Otherwise                           -> regression
    
    Args:
        y: Series containing target values
        
    Returns:
        Tuple of (is_classification, task_type_string)
    """
    if pd.api.types.is_categorical_dtype(y):
        return True, "classification"
    if not pd.api.types.is_numeric_dtype(y):
        return True, "classification"
    
    # Check for binary classification
    unique_values = set(y.unique())
    if unique_values == {0, 1} or unique_values == {False, True}:
        return True, "classification"
    
    # Check if data appears to be multi-class (integers with small range)
    if pd.api.types.is_integer_dtype(y):
        min_val, max_val = y.min(), y.max()
        # If all values are integers in a small range and there aren't too many unique values
        if 0 <= min_val and max_val <= 3 and len(unique_values) <= (max_val - min_val + 1):
            return True, "classification"
        
    return False, "regression"


def save_fold_predictions_vs_actual(fold_idx, ids, y_pred, y_test, main_output_dir, original_ids=None, problem_type=None):
    """Save a CSV listing the predicted vs. actual target values for each sample in this fold.
    
    Args:
        fold_idx: Index of current fold
        ids: Sample IDs (indices)
        y_pred: Predicted values
        y_test: True values
        main_output_dir: Output directory path
        original_ids: Original IDs from input file, if available
    """
    fold_dir = os.path.join(main_output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    
    # Convert to numpy arrays if they are lists
    y_pred_array = np.array(y_pred)
    y_test_array = np.array(y_test)
    
    # Use original IDs if available, otherwise use provided ids or create placeholders
    if original_ids is not None:
        display_ids = original_ids
    elif ids is not None:
        display_ids = ids
    else:
        # Create placeholder IDs if none are provided
        display_ids = [f"Sample_{i+1}" for i in range(len(y_pred))]
    
    # Determine problem type - use explicit parameter if provided, otherwise use smart detection
    if problem_type is None:
        # Convert arrays to pandas Series for smart detection
        y_test_series = pd.Series(y_test)
        is_classification, detected_type = determine_task_type(y_test_series)
        is_regression = not is_classification
    else:
        # Use the explicitly provided problem type
        is_regression = problem_type.lower() == 'regression'
    
    # Create DataFrame with base columns
    data_dict = {
        'ID': display_ids,  # These should be original IDs from the dataset when available
        'Predicted': y_pred,
        'Actual': y_test
    }
    
    # Add a 'Correct' column only for classification problems
    if not is_regression:
        # For classification, compare predicted and actual classes
        data_dict['Correct'] = ['TRUE' if y_pred[i] == y_test[i] else 'FALSE' for i in range(len(y_pred))]
    
    # Add residual columns only for regression problems
    if is_regression:
        residuals = y_test_array - y_pred_array
        data_dict['Residual'] = residuals
        data_dict['Abs_Residual'] = np.abs(residuals)
        
    df = pd.DataFrame(data_dict)
    
    # Sort by appropriate column for easier analysis
    if is_regression:
        # For regression, sort by absolute residual
        df = df.sort_values('Abs_Residual', ascending=False)
    else:        # Just keep original order or sort by ID for consistency
        df = df.sort_values('ID')
    
    # Save to CSV, ensuring NA values are preserved
    csv_path = os.path.join(fold_dir, f"predictions_vs_actual_fold_{fold_idx}.csv")
    df.to_csv(csv_path, index=False, na_rep='NA')
    print(f"[Fold {fold_idx}] Predictions vs Actual CSV saved at {csv_path}")


def evaluate_predictions(y_true, y_pred):
    """Calculate performance metrics for predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dict of metrics
    """
    # Convert to numpy arrays if they are lists
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def generate_confusion_matrix(y_true, y_pred, output_path, title="Confusion Matrix", metric=None, cmap=None):
    """Generate a confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
        title: Title of the plot
        metric: Primary metric to display (default: accuracy)
        cmap: Colormap for the confusion matrix (default: defined in colors.py)
    """
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
    import os
    
    # Get unique classes preserving the original order
    classes = np.unique(np.concatenate((y_true, y_pred)))
    class_indices = {cls: i for i, cls in enumerate(classes)}
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Create plot with larger size and higher DPI for clarity
    plt.figure(figsize=(10, 8), dpi=150)
    
    # Use a more distinct colormap and larger font sizes
    # Use the colormap from parameters, or the default from colors.py if not specified
    colormap = cmap if cmap is not None else CONFUSION_MATRIX_CMAP
    
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap=colormap, cbar=False,
               xticklabels=classes, yticklabels=classes, annot_kws={"size": 16},
               linewidths=CONFUSION_MATRIX_LINEWIDTH, linecolor=CONFUSION_MATRIX_LINECOLOR)
    
    # Increase font sizes for better readability
    plt.title(title, fontsize=18, pad=20)
    plt.ylabel("True Label", fontsize=16, labelpad=15)
    plt.xlabel("Predicted Label", fontsize=16, labelpad=15)
    
    # Set tick size
    ax.tick_params(labelsize=14)
    
    # Get appropriate metric based on what was selected or available
    # Default to accuracy if not specified
    if not metric:
        metric = os.environ.get('DAFTAR-ML_METRIC', 'accuracy')
    
    # Calculate the selected metric
    if metric == 'accuracy':
        metric_value = accuracy_score(y_true, y_pred)
        metric_text = f"Accuracy: {metric_value:.4f}"
    elif metric == 'f1':
        metric_value = f1_score(y_true, y_pred, average='weighted')
        metric_text = f"F1 Score: {metric_value:.4f}"
    elif metric == 'precision':
        metric_value = precision_score(y_true, y_pred, average='weighted')
        metric_text = f"Precision: {metric_value:.4f}"
    elif metric == 'recall':
        metric_value = recall_score(y_true, y_pred, average='weighted')
        metric_text = f"Recall: {metric_value:.4f}"
    elif metric == 'roc_auc' and len(classes) == 2:
        try:
            metric_value = roc_auc_score(y_true, y_pred)
            metric_text = f"ROC AUC: {metric_value:.4f}"
        except:
            metric_value = accuracy_score(y_true, y_pred)
            metric_text = f"Accuracy: {metric_value:.4f}"
    else:
        # Fall back to accuracy
        metric_value = accuracy_score(y_true, y_pred)
        metric_text = f"Accuracy: {metric_value:.4f}"
    
    plt.figtext(0.02, 0.02, metric_text, fontsize=14, 
                bbox={"facecolor": "white", "alpha": 0.9, "pad": 10, "edgecolor": "#cccccc"})
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved at {output_path}")


def generate_density_plots(fold_results, all_true_values, all_predictions, output_dir, target_name, problem_type="regression", cmap=None):
    """Generate density plots for the entire set of predictions.
    
    Args:
        fold_results: Results from each fold
        all_true_values: All true values
        all_predictions: All predicted values
        output_dir: Output directory path
        target_name: Name of target variable
        problem_type: Type of problem ('regression' or 'classification')
        cmap: Optional colormap for confusion matrices
    """
    # For each fold, save a CSV of predicted vs actual
    for fold in fold_results:
        fold_idx = fold['fold_index']
        y_pred = fold['y_pred']
        y_test = fold['y_test']
        ids_test = fold['ids_test']
        
        # Get original IDs if available
        original_ids = fold.get('original_ids', None)
        
        # Call with original IDs when available
        save_fold_predictions_vs_actual(fold_idx, ids_test, y_pred, y_test, output_dir, original_ids=original_ids)
        
        # Create fold directory if it doesn't exist
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Generate confusion matrix for classification problems (per fold)
        if problem_type == "classification":
            confusion_path = os.path.join(fold_dir, f"confusion_matrix_fold_{fold_idx}.png")
            generate_confusion_matrix(
                y_test, y_pred, 
                confusion_path, 
                title=f"Confusion Matrix - Fold {fold_idx}",
                metric=fold.get('metric', None),
                cmap=cmap
            )
        # Generate density plots for regression problems (per fold)
        elif problem_type == "regression":
            # Create per-fold density plot
            plt.figure(figsize=(10, 6))
            sns.kdeplot(y_test, label='Actual', fill=True, alpha=DENSITY_ALPHA, color=DENSITY_ACTUAL_COLOR)
            sns.kdeplot(y_pred, label='Predicted', fill=True, alpha=DENSITY_ALPHA, color=DENSITY_PREDICTED_COLOR)
            plt.title(f"Density Plot - Fold {fold_idx}")
            plt.xlabel(target_name)
            plt.ylabel('Density')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Add evaluation metrics as text annotation if available
            if 'metric_value' in fold and 'metric_name' in fold:
                metric_name = fold['metric_name']
                metric_value = fold['metric_value']
                plt.annotate(f"{metric_name}: {metric_value:.4f}", 
                             xy=(0.05, 0.95), xycoords='axes fraction',
                             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
            
            density_path = os.path.join(fold_dir, f"density_plot_fold_{fold_idx}.png")
            plt.savefig(density_path, bbox_inches='tight')
            plt.close()
            print(f"Fold {fold_idx} density plot saved at {density_path}")

    # Convert lists to numpy arrays if needed
    all_true_values_array = np.array(all_true_values)
    all_predictions_array = np.array(all_predictions)
    
    # Create global visualizations based on problem type
    if problem_type == "classification":
        # For classification, create a global confusion matrix for all folds combined
        confusion_path = os.path.join(output_dir, "confusion_matrix_global.png")

        # Get the metric from the first fold (should be consistent across folds)
        metric = None
        if fold_results and len(fold_results) > 0:
            metric = fold_results[0].get('metric', None)
            
        # Generate the global confusion matrix
        generate_confusion_matrix(
            all_true_values_array, all_predictions_array,
            confusion_path,
            title="Global Confusion Matrix (All Folds)",
            metric=metric,
            cmap=cmap
        )
        print(f"Global confusion matrix saved at {confusion_path}")
        
        # Build a simplified overall predictions DataFrame (ID, Fold, Actual, Predicted, Correct)
        csv_path = os.path.join(output_dir, "predictions_vs_actual_overall.csv")
        
        all_ids = []
        all_fold_indices = []
        all_actual_values = []
        all_predicted_values = []
        correct_flags = []
        
        # Gather data from each fold
        for fold_idx, fold in enumerate(fold_results):
            fold_num = fold_idx + 1
            
            # Prefer original IDs when available
            if 'original_ids' in fold and fold['original_ids'] is not None:
                sample_ids = fold['original_ids']
            elif 'ids_test' in fold and fold['ids_test'] is not None:
                sample_ids = fold['ids_test']
            else:
                sample_ids = [f"Sample_{i+1}" for i in range(len(fold['y_test']))]
            
            y_true = fold['y_test']
            y_pred = fold['y_pred']
            
            for i in range(len(y_true)):
                all_ids.append(sample_ids[i])
                all_fold_indices.append(fold_num)
                all_actual_values.append(y_true[i])
                all_predicted_values.append(y_pred[i])
                correct_flags.append(y_true[i] == y_pred[i])
        
        # Create DataFrame with raw data from all folds
        raw_df = pd.DataFrame({
            'ID': all_ids,
            'Fold': all_fold_indices,
            'Actual': all_actual_values,
            'Predicted': all_predicted_values,
            'Correct': ['TRUE' if flag else 'FALSE' for flag in correct_flags]
        })
        
        # Get unique IDs to create a summary by ID
        unique_ids = sorted(list(set(all_ids)))
        
        # Get unique classes from both actual and predicted values
        all_classes = sorted(list(set(list(set(all_actual_values)) + list(set(all_predicted_values)))))
        
        # Prepare summary data
        summary_data = {
            'ID': []
        }
        
        # Add a column for each class to count predictions
        for cls in all_classes:
            summary_data[f'Predicted_{cls}'] = []
        
        summary_data['Actual'] = []
        summary_data['Overall_Prediction'] = []
        summary_data['Correct'] = []
        
        # Process each unique ID
        for id_val in unique_ids:
            id_rows = raw_df[raw_df['ID'] == id_val]
            
            if len(id_rows) == 0:
                continue
                
            summary_data['ID'].append(id_val)
            
            # Most common actual value (should be the same for all folds)
            actual_value = id_rows['Actual'].value_counts().index[0]
            summary_data['Actual'].append(actual_value)
            
            # Count predictions for each class
            pred_counts = id_rows['Predicted'].value_counts().to_dict()
            for cls in all_classes:
                summary_data[f'Predicted_{cls}'].append(pred_counts.get(cls, 0))
            
            # Find the most predicted class(es)
            max_count = max(pred_counts.values()) if pred_counts else 0
            max_classes = [cls for cls, count in pred_counts.items() if count == max_count]
            
            # Handle potential ties
            if len(max_classes) == 1:
                overall_pred = max_classes[0]
            else:
                # If there's a tie, list all tied classes
                overall_pred = '/'.join(str(cls) for cls in max_classes)
            
            summary_data['Overall_Prediction'].append(overall_pred)
            
            # Check if the overall prediction matches the actual value
            # For tied predictions, use a special 'TIE' label when the actual value is among the tied predictions
            # Always convert to string before checking for '/'
            overall_pred_str = str(overall_pred)
            if '/' in overall_pred_str:  # This indicates a tie
                is_correct = 'TIE' if str(actual_value) in overall_pred_str.split('/') else 'FALSE'
            else:  # Single prediction
                is_correct = 'TRUE' if overall_pred == actual_value or str(overall_pred) == str(actual_value) else 'FALSE'
            
            summary_data['Correct'].append(is_correct)
        
        # Create the summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save the summary CSV
        summary_df.to_csv(csv_path, index=False)
        print(f"Overall predictions vs actual CSV saved at {csv_path}")
        return
    else:
        # For regression, use density plots as before
        plt.figure(figsize=(12, 6))
        sns.kdeplot(all_true_values_array, label='Actual', fill=True, alpha=DENSITY_ALPHA, color=DENSITY_ACTUAL_COLOR)
        sns.kdeplot(all_predictions_array, label='Predicted', fill=True, alpha=DENSITY_ALPHA, color=DENSITY_PREDICTED_COLOR)
        plt.title(f"Global Density Plot of Actual vs. Predicted for {target_name}")
        plt.xlabel(f"{target_name} (Target Value)")
        plt.ylabel('Density')
        
        # Get the user-selected metric for regression
        scores = evaluate_predictions(all_true_values_array, all_predictions_array)
        
        # Get selected metric from any fold (should be the same for all)
        metric = None
        if fold_results and len(fold_results) > 0:
            metric = fold_results[0].get('metric', None)
        
        if not metric:
            metric = os.environ.get('DAFTAR-ML_METRIC', 'rmse').lower()
        
        # Display the selected metric
        if metric.lower() == 'rmse':
            metrics_text = f"RMSE: {scores['RMSE']:.7f}"
        elif metric.lower() == 'mse':
            metrics_text = f"MSE: {scores['MSE']:.7f}"
        elif metric.lower() == 'mae':
            metrics_text = f"MAE: {scores['MAE']:.7f}"
        elif metric.lower() == 'r2':
            metrics_text = f"R²: {scores['R2']:.7f}"
        else:
            # Default to RMSE if metric is not recognized
            metrics_text = f"RMSE: {scores['RMSE']:.7f}"
        
        # Add small margin on the right side
        plt.tight_layout(rect=[0, 0, 0.85, 1])  
        
        # Simple text box with minimal styling
        plt.gca().text(1.02, 0.5, metrics_text, transform=plt.gca().transAxes,
                       ha='left', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    density_plot_path = os.path.join(output_dir, "density_actual_vs_pred_global.png")
    plt.savefig(density_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Global density plot saved at {density_plot_path}")

    # Save overall predictions vs. actual targets to a CSV
    csv_path = os.path.join(output_dir, "predictions_vs_actual_overall.csv")
    
    # Collect all IDs from fold results
    all_ids = []
    for fold in fold_results:
        if 'original_ids' in fold and fold['original_ids'] is not None:
            all_ids.extend(fold['original_ids'])
        elif 'ids_test' in fold:
            all_ids.extend([f'Sample_{idx}' for idx in fold['ids_test']])
        else:
            # Add placeholder IDs if none available
            all_ids.extend([f'Sample_{i}' for i in range(len(fold.get('y_test', [])))])
    
    df_overall = pd.DataFrame({
        'ID': all_ids,
        'Predicted': all_predictions,
        'Actual': all_true_values
    })
    
    # Determine if this is a regression problem based on problem_type or smart detection
    if problem_type is None:
        # Convert arrays to pandas Series for smart detection
        y_true_series = pd.Series(all_true_values)
        is_classification, detected_type = determine_task_type(y_true_series)
        is_regression = not is_classification
    else:
        # Use the explicitly provided problem type
        is_regression = problem_type.lower() == 'regression'
    
    # Calculate residuals only for regression problems
    if is_regression:  # Only for regression
        df_overall['Residual'] = np.array(df_overall['Predicted']) - np.array(df_overall['Actual'])
        df_overall['Abs_Residual'] = np.abs(df_overall['Residual'])
    
    # Only calculate prediction stats for regression problems
    if is_regression:  # Only for regression
        # Calculate min and max predictions for each actual value
        prediction_stats = df_overall.groupby('Actual')['Predicted'].agg(['min', 'max']).reset_index()
        prediction_stats.columns = ['Actual', 'Min_Prediction', 'Max_Prediction']
        
        # For each row in df_overall, add the min and max prediction for its actual value
        actual_to_min = dict(zip(prediction_stats['Actual'], prediction_stats['Min_Prediction']))
        actual_to_max = dict(zip(prediction_stats['Actual'], prediction_stats['Max_Prediction']))
        
        # Add min/max columns based on the actual value for each row
        df_overall['min_pred'] = df_overall['Actual'].map(actual_to_min)
        df_overall['max_pred'] = df_overall['Actual'].map(actual_to_max)
    
    # Save the enhanced file (without creating additional files)
    df_overall.to_csv(csv_path, index=False)
    print(f"Overall predictions vs actual CSV saved at {csv_path}")


def save_top_features_summary(feature_impact_df, main_output_dir, config):
    """Create a text file summarizing the most important features based on SHAP stats.
    
    Args:
        feature_impact_df: DataFrame with SHAP impact statistics
        main_output_dir: Output directory path
        config: Configuration object
    """
    # Create a text file summarizing the most important features based on SHAP stats.
    summary_path = os.path.join(main_output_dir, "shap_features_summary.txt")
    
    # Check if feature_impact_df is None
    if feature_impact_df is None:
        print("Important features summary cannot be generated: no SHAP data available")
        return
        
    # Create a default target if missing from config
    target = getattr(config, 'target', 'Target')
    metric = getattr(config, 'metric', 'accuracy')
    model_type = getattr(config, 'model', 'model')
    problem_type = getattr(config, 'problem_type', 'regression')
    
    # Get configuration values
    top_n = config.top_n
    transform_x = getattr(config, 'transform_features', False)
    transform_y = getattr(config, 'transform_target', False)
    
    # Create feature data organized by different metrics - reflecting only main columns in CSV
    sections = {
        "Fold_Mean_SHAP": feature_impact_df.sort_values("Fold_Mean_SHAP", ascending=False),
        "Sample_Mean_SHAP": feature_impact_df.sort_values("Sample_Mean_SHAP", ascending=False),
        "Fold_Level_Impact": feature_impact_df.sort_values("Fold_Level_Impact", ascending=False),
        "Sample_Level_Impact": feature_impact_df.sort_values("Sample_Level_Impact", ascending=False),
    }
    
    # Add correlation sections only for regression problems
    if problem_type == 'regression':
        if 'Fold_Level_Correlation' in feature_impact_df.columns:
            sections["Fold_Level_Correlation"] = feature_impact_df.sort_values("Fold_Level_Correlation", ascending=False)
        if 'Sample_Level_Correlation' in feature_impact_df.columns:
            sections["Sample_Level_Correlation"] = feature_impact_df.sort_values("Sample_Level_Correlation", ascending=False)
    
    # Also add positive/negative impact sections based on Fold_Mean_SHAP
    positive_features = feature_impact_df[feature_impact_df["Fold_Mean_SHAP"] > 0].sort_values("Fold_Mean_SHAP", ascending=False)
    negative_features = feature_impact_df[feature_impact_df["Fold_Mean_SHAP"] < 0].sort_values("Fold_Mean_SHAP", ascending=True)
    
    # Write out a concise summary of feature impacts
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"FEATURE IMPORTANCE SUMMARY FOR {target} PREDICTION\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Optimization Metric: {metric}\n")
        if transform_x:
            f.write("Feature Transformation: Log1p\n")
        if transform_y:
            f.write("Target Transformation: Log1p\n")
        f.write("\n")
        
        # Explanation of different columns
        f.write("="*80 + "\n")
        f.write("EXPLANATION OF FEATURE METRICS\n")
        f.write("="*80 + "\n\n")
        f.write("DAFTAR-ML provides multiple metrics for each feature:\n\n")
        
        f.write("Fold_Mean_SHAP: Average SHAP value across folds (positive = increases prediction, negative = decreases)\n")
        f.write("Sample_Mean_SHAP: Average SHAP value across all samples\n")
        f.write("Fold_Level_Impact: Average absolute SHAP impact across folds\n")
        f.write("Sample_Level_Impact: Average absolute SHAP impact across all samples\n")
        
        if problem_type == 'regression':
            f.write("Fold_Level_Correlation: Correlation between SHAP values and target (fold-level)\n")
            f.write("Sample_Level_Correlation: Correlation between SHAP values and target (sample-level)\n")
            f.write("\n")
        
        # Generate sections for each metric
        for metric_name, metric_df in sections.items():
            f.write("="*80 + "\n")
            if metric_name.startswith("Fold_"):
                f.write(f"TOP {top_n} FEATURES BY {metric_name} (Fold-Level, Column: {metric_name})\n")
            elif metric_name.startswith("Sample_"):
                f.write(f"TOP {top_n} FEATURES BY {metric_name} (Sample-Level, Column: {metric_name})\n")
            else:
                f.write(f"TOP {top_n} FEATURES BY {metric_name} (Column: {metric_name})\n")
            f.write("="*80 + "\n\n")
            
            for i, (feature, row) in enumerate(metric_df.head(top_n).iterrows(), 1):
                f.write(f"{i}. {feature}: {row[metric_name]: .6f}")
                
                # Add standard deviation only for SHAP values (not impacts or correlations)
                if metric_name == "Fold_Mean_SHAP" and "Fold_SHAP_StdDev" in row:
                    f.write(f" (±{row['Fold_SHAP_StdDev']:.6f})")
                elif metric_name == "Sample_Mean_SHAP" and "Sample_SHAP_StdDev" in row:
                    f.write(f" (±{row['Sample_SHAP_StdDev']:.6f})")
                
                f.write("\n")
        
            # Add extra space after the last item in the list
            f.write("\n")
        
        # Special sections for positive and negative impacts
        f.write("="*80 + "\n")
        f.write(f"TOP {top_n} FEATURES WITH POSITIVE IMPACT (Fold-Level, Column: Fold_Mean_SHAP > 0)\n")
        f.write("="*80 + "\n\n")
        
        for i, (feature, row) in enumerate(positive_features.head(top_n).iterrows(), 1):
            f.write(f"{i}. {feature}: {row['Fold_Mean_SHAP']:.6f}")
            if "Fold_SHAP_StdDev" in row:
                f.write(f" (±{row['Fold_SHAP_StdDev']:.6f})")
            f.write("\n")
        
        # Add extra space after the list
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write(f"TOP {top_n} FEATURES WITH NEGATIVE IMPACT (Fold-Level, Column: Fold_Mean_SHAP < 0)\n")
        f.write("="*80 + "\n\n")
        
        for i, (feature, row) in enumerate(negative_features.head(top_n).iterrows(), 1):
            f.write(f"{i}. {feature}: {abs(row['Fold_Mean_SHAP']):.6f}")
            if "Fold_SHAP_StdDev" in row:
                f.write(f" (±{row['Fold_SHAP_StdDev']:.6f})")
            f.write("\n")
        
        # Add extra space after the list
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("="*80 + "\n\n")
        f.write("SHAP values represent the impact of each feature on model predictions.\n")
        f.write("- Positive values mean the feature pushes predictions higher\n")
        f.write("- Negative values mean the feature pushes predictions lower\n")
        f.write("- The magnitude indicates how strong the effect is\n")
        f.write("- Standard deviation (±) shows the variability of this effect\n\n")
        f.write("These values are calculated based on SHAP (SHapley Additive exPlanations),\n")
        f.write("which analyzes each feature's contribution to predictions across the dataset.\n")
    
    print(f"Important features summary saved to {summary_path}")
