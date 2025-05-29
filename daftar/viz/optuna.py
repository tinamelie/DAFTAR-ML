"""Optuna visualization utilities for DAFTAR-ML."""

import os
import warnings
from pathlib import Path
from typing import Any, Optional

import optuna
import plotly.io as pio


def save_optuna_visualizations(study: Any, fold_idx: int, output_dir: Path, config: Optional[Any] = None) -> None:
    """Generate and save Optuna plots for a given fold.
    
    Args:
        study: Optuna study object
        fold_idx: Index of current fold
        output_dir: Output directory path
        config: Configuration object, used to determine if metric is maximized
    """
    # Create fold directory
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(exist_ok=True)
    
    # Create optuna plots directory
    optuna_dir = fold_dir / "optuna_plots"
    optuna_dir.mkdir(exist_ok=True)
    
    # Save hyperparameter tuning details
    tuning_summary_path = fold_dir / f"hyperparameter_tuning_summary_fold{fold_idx}.txt"
    with open(tuning_summary_path, "w") as f:
        f.write(f"Hyperparameter Tuning Summary for Fold {fold_idx}\n")
        f.write("=" * 60 + "\n\n")
        
        # Add study information
        best_trial = study.best_trial
        f.write(f"Total Trials: {len(study.trials)}\n")
        f.write(f"Best Trial: {best_trial.number}\n")
        f.write(f"Best Value: {best_trial.value:.8f}\n\n")
        
        # Track patience pattern
        f.write("Early Stopping Pattern:\n")
        f.write("-" * 60 + "\n")
        
        # Count trials with no improvement to recreate patience counter
        best_value_by_trial = float('inf')
        patience_counter = 0
        best_trial_idx = 0
        
        for i, trial in enumerate(study.trials):
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
                
            trial_value = trial.value if trial.value is not None else float('inf')
            did_improve = False
            
            if trial_value < best_value_by_trial:
                best_value_by_trial = trial_value
                best_trial_idx = i
                patience_counter = 0
                did_improve = True
            else:
                patience_counter += 1
            
            improvement_marker = "✓" if did_improve else "×"
            f.write(f"Trial {i:3d}: Value = {trial_value:.8f} | Patience = {patience_counter:2d} | Improvement: {improvement_marker}\n")
            
            # Add hyperparameters for this trial
            f.write("  Hyperparameters:\n")
            for param_name, param_value in trial.params.items():
                f.write(f"    {param_name}: {param_value}\n")
            f.write("\n")
        
        # Add best hyperparameters
        f.write("\nBest Hyperparameters:\n")
        f.write("-" * 60 + "\n")
        
        for param_name, param_value in best_trial.params.items():
            f.write(f"{param_name}: {param_value}\n")
    
    # Determine if we need to show positive values (for maximized metrics)
    is_maximization_metric = False
    if config and hasattr(config, 'metric') and config.metric:
        maximization_metrics = ['accuracy', 'f1', 'roc_auc', 'r2']
        is_maximization_metric = config.metric in maximization_metrics
    
    # Save optimization history plot
    try:
        # Get the base figure from Optuna
        fig = optuna.visualization.plot_optimization_history(study)
        
        if fig.data:
            # Update the plot aesthetics
            for trace in fig.data:
                if hasattr(trace, "marker") and trace.marker is not None:
                    trace.marker.colorscale = "jet"
            
            # If it's a maximization metric, adjust the y-axis values and labels
            if is_maximization_metric:
                # For the main figure (trace 0 is usually the history line)
                if len(fig.data) > 0 and hasattr(fig.data[0], 'y'):
                    # Convert negative values to positive for maximization metrics
                    fig.data[0].y = [-y for y in fig.data[0].y]
                
                # For the best value trace (usually trace 1)
                if len(fig.data) > 1 and hasattr(fig.data[1], 'y'):
                    fig.data[1].y = [-y for y in fig.data[1].y]
                
                # Update the y-axis title to reflect that these are positive values
                if hasattr(fig.layout, 'yaxis') and hasattr(fig.layout.yaxis, 'title'):
                    original_title = fig.layout.yaxis.title.text
                    if original_title:
                        # Replace "Objective Value" with the actual metric name
                        new_title = original_title.replace("Objective Value", f"{config.metric.upper()}")
                        fig.layout.yaxis.title.text = new_title
        
        # Save the modified figure
        pio.write_html(fig, optuna_dir / f"optuna_history_fold{fold_idx+1}.html")
    except Exception as e:
        warnings.warn(f"Could not generate optimization history plot for fold_{fold_idx+1}: {e}")

    # Save parallel coordinate plot
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        if fig.data:
            for trace in fig.data:
                trace.line.colorscale = "jet"
                
            # If it's a maximization metric, update the color mapping and add a note
            if is_maximization_metric and fig.layout and hasattr(fig.layout, 'title'):
                # Add a note about the metric being maximized
                if hasattr(fig.layout.title, 'text') and fig.layout.title.text:
                    # Change title to indicate inverted scale for maximization metrics
                    metric_name = config.metric.upper() if config and hasattr(config, 'metric') else 'Metric'
                    fig.layout.title.text = fig.layout.title.text + f" ({metric_name} - Higher is better)"
                
                # Change colorscale direction for maximization metrics
                for trace in fig.data:
                    if hasattr(trace, 'line') and hasattr(trace.line, 'color'):
                        # Invert the color mapping - not the values themselves
                        # This way higher values (more negative) will get better colors
                        if hasattr(trace.line, 'colorscale'):
                            # Reverse the colorscale to make higher values (more negative) get warmer colors
                            trace.line.reversescale = True
                
        pio.write_html(fig, optuna_dir / f"optuna_parallel_fold{fold_idx+1}.html")
    except Exception as e:
        warnings.warn(f"Could not generate parallel coordinate plot for fold_{fold_idx+1}: {e}")

    # Save slice plot
    try:
        fig = optuna.visualization.plot_slice(study)
        if fig.data:
            for trace in fig.data:
                if hasattr(trace, "marker") and trace.marker is not None:
                    trace.marker.colorscale = "jet"
            
            # If it's a maximization metric, adjust the y-axis values and labels
            if is_maximization_metric:
                for trace in fig.data:
                    if hasattr(trace, 'y'):
                        # Convert negative values to positive for maximization metrics
                        trace.y = [-y for y in trace.y]
                
                # Update the y-axis title to reflect that these are positive values
                if hasattr(fig.layout, 'yaxis') and hasattr(fig.layout.yaxis, 'title'):
                    original_title = fig.layout.yaxis.title.text
                    if original_title:
                        metric_name = config.metric.upper() if config and hasattr(config, 'metric') else 'Metric'
                        new_title = original_title.replace("Objective Value", metric_name)
                        fig.layout.yaxis.title.text = new_title
                
                if hasattr(fig.layout, 'title') and hasattr(fig.layout.title, 'text'):
                    title_text = fig.layout.title.text
                    if title_text:
                        fig.layout.title.text = title_text + f" ({config.metric.upper()} - Higher is better)"
        
        pio.write_html(fig, optuna_dir / f"optuna_slice_fold{fold_idx+1}.html")
    except Exception as e:
        warnings.warn(f"Could not generate slice plot for fold_{fold_idx+1}: {e}")
