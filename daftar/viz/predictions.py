"""Prediction visualization utilities for DAFTAR-ML."""

import os
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_fold_predictions_vs_actual(fold_idx, ids, y_pred, y_test, main_output_dir, original_ids=None):
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
    
    # Calculate residuals
    residuals = y_test_array - y_pred_array
    df = pd.DataFrame({
        'ID': display_ids,  # These should be original IDs from the dataset when available
        'Predicted': y_pred,
        'Actual': y_test,
        'Residual': residuals,
        'Abs_Residual': np.abs(residuals)
    })
    
    # Sort by absolute residual for easier analysis
    df = df.sort_values('Abs_Residual', ascending=False)
    
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


def generate_confusion_matrix(y_true, y_pred, output_path, title="Confusion Matrix"):
    """Generate a confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
        title: Title of the plot
    """
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # Add metrics as text
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    metrics_text = f"Accuracy: {accuracy:.4f}\nF1 Score: {f1:.4f}"
    
    # Try to add ROC AUC if binary classification
    classes = np.unique(y_true)
    if len(classes) == 2:
        try:
            # Only calculate ROC AUC for binary classification
            roc_auc = roc_auc_score(y_true, y_pred)
            metrics_text += f"\nROC AUC: {roc_auc:.4f}"
        except:
            pass
    
    plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})
    
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
                title=f"Confusion Matrix - Fold {fold_idx}"
            )

    # Convert lists to numpy arrays if needed
    all_true_values_array = np.array(all_true_values)
    all_predictions_array = np.array(all_predictions)
    
    # Handle density plots differently for classification and regression
    if problem_type == "classification":
        # Skip density plot creation for classification problems
        # Only create global confusion matrix for all folds combined
        confusion_path = os.path.join(output_dir, "confusion_matrix_global.png")
        generate_confusion_matrix(
            all_true_values_array, all_predictions_array,
            confusion_path,
            title="Global Confusion Matrix (All Folds)"
        )
        # Skip the rest of the function for classification problems
        # Save overall predictions vs. actual targets to a CSV (moved from below)
        csv_path = os.path.join(output_dir, "predictions_vs_actual_overall.csv")
        
        # Create main DataFrame with predictions
        df = pd.DataFrame({
            'actual': all_true_values,
            'predicted': all_predictions
        })
        
        # Calculate percentage of predicted 0s and 1s for each actual class
        # We'll include this directly in the main output file
        # First, get unique actual values
        unique_actual = np.unique(all_true_values_array)
        
        # For each sample, calculate counts and percentages for each actual class
        counts_0s = {}
        counts_1s = {}
        pct_0s = {}
        pct_1s = {}
        for actual_val in unique_actual:
            mask = np.array(all_true_values) == actual_val
            preds_for_this_actual = np.array(all_predictions)[mask]
            counts_0s[actual_val] = (preds_for_this_actual == 0).sum()
            counts_1s[actual_val] = (preds_for_this_actual == 1).sum()
            pct_0s[actual_val] = (preds_for_this_actual == 0).mean() * 100
            pct_1s[actual_val] = (preds_for_this_actual == 1).mean() * 100
        
        # Add counts and percentages to each row based on its actual value
        df['count_pred_0'] = [counts_0s[val] for val in df['actual']]
        df['count_pred_1'] = [counts_1s[val] for val in df['actual']]
        df['pct_pred_0'] = [pct_0s[val] for val in df['actual']]
        df['pct_pred_1'] = [pct_1s[val] for val in df['actual']]
        
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
        
        # Add only the optimization metric (RMSE) for regression
        scores = evaluate_predictions(all_true_values_array, all_predictions_array)
        metrics_text = f"RMSE: {scores['RMSE']:.7f}"
        
        # Add small margin on the right side
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Small horizontal padding
        
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
    # Calculate residuals using numpy arrays to avoid list subtraction
    df_overall['Residual'] = np.array(df_overall['Predicted']) - np.array(df_overall['Actual'])
    df_overall['Abs_Residual'] = np.abs(df_overall['Residual'])
    
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
            if 'Target_Correlation' in row and not pd.isna(row['Target_Correlation']):
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
            if 'Target_Correlation' in row and not pd.isna(row['Target_Correlation']):
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
            if 'Target_Correlation' in row and not pd.isna(row['Target_Correlation']):
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
