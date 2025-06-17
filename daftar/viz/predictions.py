"""Prediction visualization utilities for DAFTAR-ML."""

import os
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from daftar.viz.common import save_plot

from daftar.viz.colors import (
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


def save_fold_predictions_vs_actual(fold_idx, ids, y_pred, y_test, main_output_dir, original_ids=None, problem_type=None, config=None):
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
    
    # Convert encoded labels to display labels if needed
    display_y_pred = y_pred
    display_y_test = y_test
    if (config and hasattr(config, 'label_encoder') and 
        config.label_encoder is not None):
        try:
            # Convert to numpy arrays and ensure integer type for inverse transform
            y_pred_array = np.array(y_pred, dtype=int)
            y_test_array = np.array(y_test, dtype=int)
            
            display_y_pred = config.label_encoder.inverse_transform(y_pred_array)
            display_y_test = config.label_encoder.inverse_transform(y_test_array)
        except Exception as e:
            print(f"Warning: Could not decode labels for fold {fold_idx} CSV: {e}")
    
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
    
    # Create DataFrame with base columns using display labels
    data_dict = {
        'ID': display_ids,  # These should be original IDs from the dataset when available
        'Predicted': display_y_pred,
        'Actual': display_y_test
    }
    
    # Add a 'Correct' column only for classification problems (use original encoded values for comparison)
    if not is_regression:
        # For classification, compare predicted and actual classes using original values
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


def evaluate_predictions(y_true, y_pred):
    """Calculate performance metrics for predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dict of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Convert to numpy arrays if they are lists
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Use sklearn metrics for consistency
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
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
    save_plot(plt.gcf(), output_path, tight_layout=True)


def generate_classification_outputs(fold_results, all_true_values, all_predictions, output_dir, config=None):
    """Generate classification-specific outputs including confusion matrix and CSV.
    
    Args:
        fold_results: Results from each fold
        all_true_values: All true values (encoded)
        all_predictions: All predicted values (encoded)
        output_dir: Output directory path
        config: Configuration object with label encoder
    """
    # Convert lists to numpy arrays if needed
    all_true_values_array = np.array(all_true_values)
    all_predictions_array = np.array(all_predictions)
    
    # Prepare display versions with original class names
    display_true_values = all_true_values_array
    display_predictions = all_predictions_array
    if (config and hasattr(config, 'label_encoder') and 
        config.label_encoder is not None):
        try:
            display_true_values = config.label_encoder.inverse_transform(all_true_values_array)
            display_predictions = config.label_encoder.inverse_transform(all_predictions_array)
        except Exception as e:
            print(f"Warning: Could not decode labels for display: {e}")
    
    # Generate global confusion matrix
    confusion_path = os.path.join(output_dir, "confusion_matrix_global.png")
    metric = fold_results[0].get('metric', None) if fold_results else None
    
    generate_confusion_matrix(
        display_true_values, display_predictions,
        confusion_path,
        title="Global Confusion Matrix (All Folds)",
        metric=metric
    )
    
    # Generate overall predictions CSV
    _generate_classification_csv(fold_results, output_dir, config)


def generate_regression_outputs(fold_results, all_true_values, all_predictions, output_dir, target_name, config=None):
    """Generate regression outputs including density plots and CSV.
    
    Args:
        fold_results: Results from each fold
        all_true_values: All true values
        all_predictions: All predicted values  
        output_dir: Output directory path
        target_name: Name of target variable
        config: Configuration object
    """
    # Convert lists to numpy arrays if needed
    all_true_values_array = np.array(all_true_values)
    all_predictions_array = np.array(all_predictions)
    
    # Generate density plots for regression
    _generate_regression_density_plot(all_true_values_array, all_predictions_array, output_dir, target_name, config)
    
    # Generate overall predictions CSV for regression
    _generate_regression_csv(fold_results, output_dir)


def generate_density_plots(fold_results, all_true_values, all_predictions, output_dir, target_name, problem_type="regression", cmap=None, config=None):
    """Generate density plots for the entire set of predictions.
    
    Args:
        fold_results: Results from each fold
        all_true_values: All true values
        all_predictions: All predicted values
        output_dir: Output directory path
        target_name: Name of target variable
        problem_type: Type of problem ('regression' or 'classification')
        cmap: Optional colormap for confusion matrices
        config: Configuration object
    """
    if problem_type == "classification":
        generate_classification_outputs(fold_results, all_true_values, all_predictions, output_dir, config)
    else:
        generate_regression_outputs(fold_results, all_true_values, all_predictions, output_dir, target_name, config)


def _generate_classification_csv(fold_results, output_dir, config=None):
    """Generate overall predictions CSV for classification."""
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
        
        # Convert encoded labels to display labels for CSV if needed
        display_y_true = y_true
        display_y_pred = y_pred
        if (config and hasattr(config, 'label_encoder') and 
            config.label_encoder is not None):
            try:
                display_y_true = config.label_encoder.inverse_transform(y_true)
                display_y_pred = config.label_encoder.inverse_transform(y_pred) 
            except Exception as e:
                print(f"Warning: Could not decode labels for CSV display: {e}")
        
        for i in range(len(y_true)):
            all_ids.append(sample_ids[i])
            all_fold_indices.append(fold_num)
            all_actual_values.append(display_y_true[i])
            all_predicted_values.append(display_y_pred[i])
            correct_flags.append(y_true[i] == y_pred[i])
    
    # Create DataFrame and save
    df_overall = pd.DataFrame({
        'ID': all_ids,
        'Fold': all_fold_indices,
        'Actual': all_actual_values,
        'Predicted': all_predicted_values,
        'Correct': correct_flags
    })
    
    df_overall.to_csv(csv_path, index=False, na_rep='NA')


def _generate_regression_density_plot(all_true_values, all_predictions, output_dir, target_name, config=None):
    """Generate density plot for regression predictions."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import sklearn.metrics
    import os
    from daftar.viz.common import save_plot
    from daftar.viz.colors import DENSITY_ACTUAL_COLOR, DENSITY_PREDICTED_COLOR, DENSITY_ALPHA
    
    # Convert to numpy arrays if needed
    all_true_values = np.array(all_true_values)
    all_predictions = np.array(all_predictions)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(all_true_values, label="Actual", fill=True, alpha=DENSITY_ALPHA, color=DENSITY_ACTUAL_COLOR)
    sns.kdeplot(all_predictions, label="Predicted", fill=True, alpha=DENSITY_ALPHA, color=DENSITY_PREDICTED_COLOR)
    plt.title("Global Density Plot - Actual vs Predicted")
    plt.xlabel(target_name)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Calculate metrics
    scores = {}
    scores['MSE'] = sklearn.metrics.mean_squared_error(all_true_values, all_predictions)
    scores['RMSE'] = np.sqrt(scores['MSE'])
    scores['MAE'] = sklearn.metrics.mean_absolute_error(all_true_values, all_predictions)
    scores['R2'] = sklearn.metrics.r2_score(all_true_values, all_predictions)
    
    # Get the selected metric from config
    metric = config.metric.lower() if config and hasattr(config, 'metric') else 'mse'
    
    # Create the metric text with appropriate precision for the selected metric only
    if metric == 'rmse':
        metrics_text = f"RMSE: {scores['RMSE']:.7f}"
    elif metric == 'mse':
        metrics_text = f"MSE: {scores['MSE']:.7f}"
    elif metric == 'mae':
        metrics_text = f"MAE: {scores['MAE']:.7f}"
    elif metric == 'r2':
        metrics_text = f"R²: {scores['R2']:.7f}"
    else:
        # Default to MSE if metric not recognized
        metrics_text = f"MSE: {scores['MSE']:.7f}"
    
    # Get number of folds from config
    n_folds = config.outer_folds if config and hasattr(config, 'outer_folds') else 5  # Default to 5 if not available
    
    # Add metrics box with updated legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.gca().text(1.02, 0.5, f"Average Test Metric\nAcross {n_folds} Folds:\n{metrics_text}", 
                  transform=plt.gca().transAxes,
                  ha='left', va='center', 
                  bbox=dict(facecolor='white', alpha=0.7))
    
    # Save the plot with the expected filename
    output_path = os.path.join(output_dir, "density_plot_overall.png")
    save_plot(plt.gcf(), output_path, tight_layout=True)


def _generate_regression_csv(fold_results, output_dir):
    """Generate overall predictions CSV for regression."""
    import pandas as pd
    import os
    
    csv_path = os.path.join(output_dir, "predictions_vs_actual_overall.csv")
    
    all_ids = []
    all_fold_indices = []
    all_actual_values = []
    all_predicted_values = []
    all_residuals = []
    
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
        
        # Calculate residuals (actual - predicted)
        residuals = y_true - y_pred if hasattr(y_true, '__sub__') else [true - pred for true, pred in zip(y_true, y_pred)]
        
        for i in range(len(y_true)):
            all_ids.append(sample_ids[i])
            all_fold_indices.append(fold_num)
            all_actual_values.append(y_true[i] if hasattr(y_true, '__getitem__') else y_true.iloc[i])
            all_predicted_values.append(y_pred[i])
            all_residuals.append(residuals[i] if hasattr(residuals, '__getitem__') else residuals.iloc[i])
    
    # Create DataFrame and save
    df_overall = pd.DataFrame({
        'ID': all_ids,
        'Fold': all_fold_indices,
        'Actual': all_actual_values,
        'Predicted': all_predicted_values,
        'Residual': all_residuals
    })
    
    df_overall.to_csv(csv_path, index=False, na_rep='NA')
