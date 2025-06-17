"""Evaluation utilities for model performance assessment."""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
import os
from pathlib import Path


def calculate_overall_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Calculate overall metrics across all folds.
    
    Args:
        fold_metrics: List of metrics dictionaries from each fold
        
    Returns:
        Dictionary of overall metrics
    """
    # Initialize overall metrics
    if not fold_metrics:
        return {}
    
    overall_metrics = {}
    
    # Get all metric names from first fold
    metric_names = fold_metrics[0].keys()
    
    # Calculate mean and std for each metric
    for metric in metric_names:
        metric_values = [fold[metric] for fold in fold_metrics]
        overall_metrics[metric] = np.mean(metric_values)
        overall_metrics[f"{metric}_std"] = np.std(metric_values)
    
    return overall_metrics


def save_metrics(metrics: Dict, fold_metrics: List[Dict], output_dir: Path) -> str:
    """
    Save overall and per-fold metrics to text and CSV files.
    
    Args:
        metrics: Overall metrics dictionary
        fold_metrics: List of per-fold metrics dictionaries
        output_dir: Output directory path
        
    Returns:
        Path to the created metrics text file
    """
    # Create performance.txt with readable format
    txt_file = output_dir / "performance.txt"
    with open(txt_file, "w") as f:
        f.write("===== DAFTAR-ML Performance Metrics =====\n\n")
        
        # Overall metrics
        f.write("Overall Performance Metrics\n")
        f.write("------------------------\n")
        f.write("These values represent the AVERAGE performance across all CV folds.\n")
        f.write("Calculation method: Each metric is calculated for individual folds, then averaged.\n\n")
        
        # Overall metrics values
        if 'mse' in metrics:
            f.write(f"MSE:  {metrics['mse']:.7f}  (Mean Squared Error)\n")
        if 'rmse' in metrics:
            f.write(f"RMSE: {metrics['rmse']:.7f}  (Root Mean Squared Error)\n")
        if 'mae' in metrics:
            f.write(f"MAE:  {metrics['mae']:.7f}  (Mean Absolute Error)\n")
        if 'r2' in metrics:
            f.write(f"R2:   {metrics['r2']:.7f}  (Coefficient of Determination)\n")
        if 'accuracy' in metrics:
            f.write(f"Accuracy: {metrics['accuracy']:.7f}\n")
        if 'f1' in metrics:
            f.write(f"F1 Score: {metrics['f1']:.7f}\n")
        if 'roc_auc' in metrics:
            f.write(f"ROC AUC:  {metrics['roc_auc']:.7f}\n")
        f.write("\n")
        
        # Per-fold metrics
        f.write("Per-Fold Metrics\n")
        f.write("---------------\n")
        f.write("These values are calculated for each fold independently.\n\n")
        
        for i, fold_metric in enumerate(fold_metrics):
            f.write(f"Fold {i+1}:\n")
            for metric_name, metric_value in fold_metric.items():
                f.write(f"  {metric_name}:  {metric_value:.7f}\n")
            f.write("\n")
    
    # Also add metrics to overall predictions CSV if it exists using pandas
    try:
        predictions_csv = output_dir / "predictions_vs_actual_overall.csv"
        if predictions_csv.exists():
            # Read the CSV into a DataFrame
            df = pd.read_csv(predictions_csv)
            
            # Create header with overall metrics
            metric_header = "# DAFTAR-ML Overall Performance Metrics: "
            if 'r2' in metrics:
                metric_header += f"R² = {metrics['r2']:.6f}, "
            if 'rmse' in metrics:
                metric_header += f"RMSE = {metrics['rmse']:.6f}, "
            if 'mae' in metrics:
                metric_header += f"MAE = {metrics['mae']:.6f}, "
            if 'accuracy' in metrics:
                metric_header += f"Accuracy = {metrics['accuracy']:.6f}, "
            if 'f1' in metrics:
                metric_header += f"F1 = {metrics['f1']:.6f}, "
            if 'roc_auc' in metrics:
                metric_header += f"ROC AUC = {metrics['roc_auc']:.6f}, "
            
            # Remove trailing comma and space if present
            if metric_header.endswith(", "):
                metric_header = metric_header[:-2]
            
            # Write the header as a comment and then the DataFrame
            with open(predictions_csv, 'w') as f:
                f.write(f"{metric_header}\n")
                df.to_csv(f, index=False)
                
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Could not add metrics to overall predictions CSV: {e}")
    
    return str(txt_file)


def make_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Serializable object
    """
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        return obj
