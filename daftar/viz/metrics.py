"""Metrics visualization utilities for DAFTAR-ML.

This module provides utilities for generating metrics visualizations
for both regression and classification models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, 
    auc, PrecisionRecallDisplay, RocCurveDisplay
)
from scipy import stats

from daftar.viz.color_definitions import (
    CONFUSION_MATRIX_CMAP,
    REGRESSION_MEAN_LINE_COLOR,
    DENSITY_ACTUAL_COLOR,
    DENSITY_PREDICTED_COLOR
)


def plot_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Predicted vs Actual Values",
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """Plot regression metrics visualization.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        output_path: Path to save the plot
        title: Plot title
        metrics: Dictionary of metric names and values
    """
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot of predicted vs actual values
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    # Add metric values as text if provided
    if metrics:
        metric_text = "\n".join([f"{name}: {value:.4f}" for name, value in metrics.items()])
        plt.annotate(
            metric_text, 
            xy=(0.05, 0.95), 
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
        )
    
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save or display the plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_regression_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Residual Plot"
) -> None:
    """Plot regression residuals visualization.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        output_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Create residual plot
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color=REGRESSION_MEAN_LINE_COLOR, linestyle='--', label='Zero residual')
    
    # Add smoothed trend line
    try:
        sns.regplot(x=y_pred, y=residuals, scatter=False, color='blue', line_kws={"color": "blue", "alpha": 0.7, "lw": 2})
    except Exception:
        # If regplot fails, skip it
        pass
    
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save or display the plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_regression_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Distribution of Actual vs Predicted Values"
) -> None:
    """Plot distribution of actual vs predicted values.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        output_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Create distribution plot
    sns.kdeplot(y_true, label='Actual', fill=True, alpha=0.3)
    sns.kdeplot(y_pred, label='Predicted', fill=True, alpha=0.3)
    
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save or display the plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    class_names: Optional[List[str]] = None
) -> None:
    """Plot confusion matrix for classification.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        output_path: Path to save the plot
        title: Plot title
        normalize: Whether to normalize the confusion matrix
        class_names: List of class names
    """
    plt.figure(figsize=(10, 8))
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Generate class names if not provided
    if class_names is None:
        class_values = sorted(np.unique(np.concatenate([y_true, y_pred])))
        class_names = [f"Class {val}" for val in class_values]
    
    # Plot confusion matrix
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap=CONFUSION_MATRIX_CMAP,
        xticklabels=class_names, yticklabels=class_names
    )
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    
    # Save or display the plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "ROC Curve",
    class_names: Optional[List[str]] = None
) -> None:
    """Plot ROC curve for binary or multiclass classification.
    
    Args:
        y_true: True target values
        y_prob: Predicted probabilities
        output_path: Path to save the plot
        title: Plot title
        class_names: List of class names
    """
    plt.figure(figsize=(10, 8))
    
    # Handle both binary and multiclass cases
    if len(y_prob.shape) == 1 or y_prob.shape[1] == 2:
        # Binary classification
        if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
            # Take probability of the positive class
            y_prob = y_prob[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(
            fpr, tpr, lw=2, 
            label=f'ROC curve (area = {roc_auc:.2f})'
        )
    else:
        # Multiclass classification
        n_classes = y_prob.shape[1]
        
        # Generate class names if not provided
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]
        
        # Binarize the true labels for each class
        y_true_bin = np.zeros((len(y_true), n_classes))
        for i in range(n_classes):
            y_true_bin[:, i] = (y_true == i).astype(int)
        
        # Compute ROC curve and ROC area for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr, lw=2,
                label=f'{class_names[i]} (area = {roc_auc:.2f})'
            )
    
    # Plot diagonal line for random performance
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save or display the plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Precision-Recall Curve",
    class_names: Optional[List[str]] = None
) -> None:
    """Plot precision-recall curve for binary or multiclass classification.
    
    Args:
        y_true: True target values
        y_prob: Predicted probabilities
        output_path: Path to save the plot
        title: Plot title
        class_names: List of class names
    """
    plt.figure(figsize=(10, 8))
    
    # Handle both binary and multiclass cases
    if len(y_prob.shape) == 1 or y_prob.shape[1] == 2:
        # Binary classification
        if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
            # Take probability of the positive class
            y_prob = y_prob[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        # Plot precision-recall curve
        plt.plot(
            recall, precision, lw=2,
            label=f'Precision-Recall curve (area = {pr_auc:.2f})'
        )
    else:
        # Multiclass classification
        n_classes = y_prob.shape[1]
        
        # Generate class names if not provided
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]
        
        # Binarize the true labels for each class
        y_true_bin = np.zeros((len(y_true), n_classes))
        for i in range(n_classes):
            y_true_bin[:, i] = (y_true == i).astype(int)
        
        # Compute precision-recall curve for each class
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            pr_auc = auc(recall, precision)
            
            plt.plot(
                recall, precision, lw=2,
                label=f'{class_names[i]} (area = {pr_auc:.2f})'
            )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    # Save or display the plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_regression_visualizations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    output_dir: Path
) -> Dict[str, Path]:
    """Generate a comprehensive set of regression visualizations.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        metrics: Dictionary of metric names and values
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store plot paths
    plot_paths = {}
    
    # Generate predicted vs actual plot
    pred_vs_actual_path = output_dir / "predicted_vs_actual.png"
    plot_regression_metrics(
        y_true, y_pred, pred_vs_actual_path,
        metrics=metrics, title="Predicted vs Actual Values"
    )
    plot_paths['predicted_vs_actual'] = pred_vs_actual_path
    
    # Generate residuals plot
    residuals_path = output_dir / "residuals.png"
    plot_regression_residuals(
        y_true, y_pred, residuals_path,
        title="Residual Plot"
    )
    plot_paths['residuals'] = residuals_path
    
    # Generate distribution plot
    distribution_path = output_dir / "distribution.png"
    plot_regression_distribution(
        y_true, y_pred, distribution_path,
        title="Distribution of Actual vs Predicted Values"
    )
    plot_paths['distribution'] = distribution_path
    
    return plot_paths


def generate_classification_visualizations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metrics: Dict[str, float],
    output_dir: Path,
    class_names: Optional[List[str]] = None
) -> Dict[str, Path]:
    """Generate a comprehensive set of classification visualizations.
    
    Args:
        y_true: True target values
        y_pred: Predicted class labels
        y_prob: Predicted probabilities
        metrics: Dictionary of metric names and values
        output_dir: Directory to save visualizations
        class_names: List of class names
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store plot paths
    plot_paths = {}
    
    # Generate confusion matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        y_true, y_pred, cm_path,
        title="Confusion Matrix",
        normalize=True,
        class_names=class_names
    )
    plot_paths['confusion_matrix'] = cm_path
    
    # Generate non-normalized confusion matrix
    cm_raw_path = output_dir / "confusion_matrix_raw.png"
    plot_confusion_matrix(
        y_true, y_pred, cm_raw_path,
        title="Confusion Matrix (Raw Counts)",
        normalize=False,
        class_names=class_names
    )
    plot_paths['confusion_matrix_raw'] = cm_raw_path
    
    # Generate ROC curve and precision-recall curve if probabilities available
    if y_prob is not None:
        roc_path = output_dir / "roc_curve.png"
        plot_roc_curve(
            y_true, y_prob, roc_path,
            title="ROC Curve",
            class_names=class_names
        )
        plot_paths['roc_curve'] = roc_path
        
        pr_path = output_dir / "precision_recall_curve.png"
        plot_precision_recall_curve(
            y_true, y_prob, pr_path,
            title="Precision-Recall Curve",
            class_names=class_names
        )
        plot_paths['precision_recall_curve'] = pr_path
    
    return plot_paths


def generate_fold_metrics_visualization(
    fold_metrics: List[Dict[str, float]],
    output_path: Optional[Path] = None,
    title: str = "Metrics Across Folds",
    problem_type: str = 'regression'
) -> None:
    """Generate visualization of metrics across folds.
    
    Args:
        fold_metrics: List of metric dictionaries from each fold
        output_path: Path to save the plot
        title: Plot title
        problem_type: Type of problem ('regression' or 'classification')
    """
    plt.figure(figsize=(12, 8))
    
    # Convert fold metrics to DataFrame
    metrics_df = pd.DataFrame(fold_metrics)
    
    # Calculate summary statistics
    means = metrics_df.mean()
    stds = metrics_df.std()
    
    # Plot metrics
    x = np.arange(len(means))
    width = 0.35
    
    plt.bar(x, means, width, label='Mean', yerr=stds, capsize=10, alpha=0.7)
    
    # Add individual fold markers
    for i, metric in enumerate(means.index):
        fold_values = metrics_df[metric].values
        plt.scatter([i] * len(fold_values), fold_values, color='red', alpha=0.7, label='_nolegend_')
    
    # Add text labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.annotate(
            f'{mean:.4f} ± {std:.4f}',
            xy=(i, mean + std + 0.05 * mean),
            ha='center',
            fontsize=9
        )
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title(title)
    plt.xticks(x, means.index)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Different y-axis limits based on problem type
    if problem_type == 'regression':
        # For regression, R² is between 0 and 1, but errors can be large
        if 'r2' in means.index:
            plt.ylim(
                min(0, metrics_df['r2'].min() - 0.1),
                max(1, metrics_df['r2'].max() + 0.1)
            )
    else:
        # For classification, most metrics are between 0 and 1
        plt.ylim(0, 1.1)
    
    # Save or display the plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
