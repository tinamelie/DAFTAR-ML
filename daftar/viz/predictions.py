"""Prediction visualization utilities for DAFTAR-ML."""

import os
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    
    # Add residual columns only for regression problems
    if is_regression:
        residuals = y_test_array - y_pred_array
        data_dict['Residual'] = residuals
        data_dict['Abs_Residual'] = np.abs(residuals)
    # No additional columns for classification problems
        
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


def generate_confusion_matrix(y_true, y_pred, output_path, title="Confusion Matrix", metric=None):
    """Generate a confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
        title: Title of the plot
        metric: Primary metric to display (default: accuracy)
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
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
               xticklabels=classes, yticklabels=classes, annot_kws={"size": 16})
    
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


def generate_density_plots(fold_results, all_true_values, all_predictions, output_dir, target_name, problem_type="regression"):
    """Generate density plots for the entire set of predictions.
    
    Args:
        fold_results: Results from each fold
        all_true_values: All true values
        all_predictions: All predicted values
        output_dir: Output directory path
        target_name: Name of target variable
        problem_type: Type of problem ('regression' or 'classification')
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
        
        # Generate confusion matrix for classification problems (per fold)
        if problem_type == "classification":
            fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)
            confusion_path = os.path.join(fold_dir, f"confusion_matrix_fold_{fold_idx}.png")
            generate_confusion_matrix(
                y_test, y_pred, 
                confusion_path, 
                title=f"Confusion Matrix - Fold {fold_idx}",
                metric=fold.get('metric', None)
            )

    # Convert lists to numpy arrays if needed
    all_true_values_array = np.array(all_true_values)
    all_predictions_array = np.array(all_predictions)
    
    # Handle density plots differently for classification and regression
    if problem_type == "classification":
        # Skip density plot creation for classification problems
        # Only create global confusion matrix for all folds combined
        confusion_path = os.path.join(output_dir, "confusion_matrix_global.png")

        metric = None
        if fold_results and len(fold_results) > 0:
            metric = fold_results[0].get('metric', None)
            
        generate_confusion_matrix(
            all_true_values_array, all_predictions_array,
            confusion_path,
            title="Global Confusion Matrix (All Folds)",
            metric=metric
        )
        # Skip the rest of the function for classification problems
        # Save overall predictions vs. actual targets to a CSV (
        csv_path = os.path.join(output_dir, "predictions_vs_actual_overall.csv")
        
        # Extract all sample data with fold information
        all_ids = []
        all_fold_indices = []
        all_actual_values = []  # Recreate to match the fold indices order
        all_predicted_values = []  # Recreate to match the fold indices order
        
        for fold_idx, fold in enumerate(fold_results):
            fold_num = fold_idx + 1
            
            # Get IDs preferring original IDs if available
            if 'original_ids' in fold and fold['original_ids'] is not None:
                sample_ids = fold['original_ids']
            elif 'ids_test' in fold and fold['ids_test'] is not None:
                sample_ids = fold['ids_test']
            else:
                sample_ids = [f"Sample_{i+1}" for i in range(len(fold['y_test']))]
            
            # Get true values and predictions for this fold
            y_true = fold['y_test']
            y_pred = fold['y_pred']
            
            # Collect all data with fold information
            for i in range(len(y_true)):
                all_ids.append(sample_ids[i])
                all_fold_indices.append(fold_num)
                all_actual_values.append(y_true[i])
                all_predicted_values.append(y_pred[i])
        
        # Create main DataFrame with predictions, IDs, and fold
        df = pd.DataFrame({
            'ID': all_ids,
            'Fold': all_fold_indices,
            'Actual': all_actual_values,
            'Predicted': all_predicted_values
            # No 'Match' column for classification problems
        })
        
        # Get unique class values
        unique_classes = np.unique(df['Actual'].values)
        
        # Calculate global statistics instead of row-specific ones
        class_stats = {}

        # Get counts for each combination of actual and predicted classes
        stats_df = pd.DataFrame()
        
        # Calculate stats for all samples combined
        for actual_class in unique_classes:
            # Calculate total for this actual class
            actual_mask = df['Actual'] == actual_class
            total_this_class = len(df.loc[actual_mask])
            
            # For each predicted class, calculate count and percentage
            for pred_class in unique_classes:
                pred_mask = df['Predicted'] == pred_class
                # Count samples that match both conditions
                count = len(df.loc[actual_mask & pred_mask])
                # Calculate percentage
                pct = (count / total_this_class) * 100 if total_this_class > 0 else 0
                
                # Add these stats as columns to the dataframe
                col_name = f"count_{actual_class}_pred_{pred_class}"
                pct_name = f"pct_{actual_class}_pred_{pred_class}"
                df[col_name] = count
                df[pct_name] = pct
        
        # Save to the single output file
        df.to_csv(csv_path, index=False)
        print(f"Overall predictions vs actual CSV saved at {csv_path}")
        return
    else:
        # For regression, use density plots as before
        plt.figure(figsize=(12, 6))
        sns.kdeplot(all_true_values_array, label='Actual', fill=True, alpha=0.5, color='#00BFC4')
        sns.kdeplot(all_predictions_array, label='Predicted', fill=True, alpha=0.5, color='#F8766D')
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
    
    positive_features = feature_impact_df[feature_impact_df["Mean_Signed"] > 0].sort_values("Mean_Signed", ascending=False)
    negative_features = feature_impact_df[feature_impact_df["Mean_Signed"] < 0].sort_values("Mean_Signed", ascending=True)
    magnitude_features = feature_impact_df.sort_values("Impact_Magnitude", ascending=False)
    
    # Get configuration values
    target = config.target
    model_type = config.model
    metric = config.metric
    top_n = config.top_n
    transform_x = getattr(config, 'transform_features', False)
    transform_y = getattr(config, 'transform_target', False)
    
    # Write out a thorough summary of feature impacts
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
        
        f.write("="*80 + "\n")
        f.write(f"TOP {top_n} FEATURES BY OVERALL IMPACT MAGNITUDE\n")
        f.write("="*80 + "\n\n")
        f.write("These features have the strongest overall effect on predictions (regardless of direction).\n\n")
        
        for i, (feature, row) in enumerate(magnitude_features.head(top_n).iterrows(), 1):
            direction = "Increases" if row["Mean_Signed"] > 0 else "Decreases"
            magnitude = abs(row["Mean_Signed"])
            std = row["Std_MeanAcrossFolds"]
            f.write(f"{i}. {feature}\n")
            f.write(f"   {direction} prediction by {magnitude:.6f} (±{std:.6f})\n")
            # Only show correlation information for regression problems
            if config.problem_type == 'regression' and 'Target_Correlation' in row and not pd.isna(row['Target_Correlation']):
                f.write(f"   Correlation with target: {row['Target_Correlation']:.6f}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write(f"TOP {top_n} FEATURES WITH POSITIVE IMPACT (INCREASE PREDICTIONS)\n")
        f.write("="*80 + "\n\n")
        f.write("These features tend to increase the predicted value when their values increase.\n\n")
        
        for i, (feature, row) in enumerate(positive_features.head(top_n).iterrows(), 1):
            magnitude = row["Mean_Signed"]
            std = row["Std_MeanAcrossFolds"]
            f.write(f"{i}. {feature}\n")
            f.write(f"   Increases prediction by {magnitude:.6f} (±{std:.6f})\n")
            # Only show correlation information for regression problems
            if config.problem_type == 'regression' and 'Target_Correlation' in row and not pd.isna(row['Target_Correlation']):
                f.write(f"   Correlation with target: {row['Target_Correlation']:.6f}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write(f"TOP {top_n} FEATURES WITH NEGATIVE IMPACT (DECREASE PREDICTIONS)\n")
        f.write("="*80 + "\n\n")
        f.write("These features tend to decrease the predicted value when their values increase.\n\n")
        
        for i, (feature, row) in enumerate(negative_features.head(top_n).iterrows(), 1):
            magnitude = abs(row["Mean_Signed"])
            std = row["Std_MeanAcrossFolds"]
            f.write(f"{i}. {feature}\n")
            f.write(f"   Decreases prediction by {magnitude:.6f} (±{std:.6f})\n")
            # Only show correlation information for regression problems
            if config.problem_type == 'regression' and 'Target_Correlation' in row and not pd.isna(row['Target_Correlation']):
                f.write(f"   Correlation with target: {row['Target_Correlation']:.6f}\n")
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
