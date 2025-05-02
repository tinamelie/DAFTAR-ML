"""Optuna visualization utilities for DAFTAR-ML."""

import os
import warnings
from pathlib import Path
from typing import Any, Optional

import optuna
import plotly.io as pio


def save_optuna_visualizations(study: Any, fold_idx: int, output_dir: Path) -> None:
    """Generate and save Optuna plots for a given fold.
    
    Args:
        study: Optuna study object
        fold_idx: Index of current fold
        output_dir: Output directory path
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
    
    # Save optimization history plot
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        if fig.data:
            for trace in fig.data:
                if hasattr(trace, "marker") and trace.marker is not None:
                    trace.marker.colorscale = "jet"
        pio.write_html(fig, optuna_dir / f"optuna_history_fold{fold_idx+1}.html")
    except Exception as e:
        warnings.warn(f"Could not generate optimization history plot for fold_{fold_idx+1}: {e}")

    # Save parallel coordinate plot
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        if fig.data:
            for trace in fig.data:
                trace.line.colorscale = "jet"
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
        pio.write_html(fig, optuna_dir / f"optuna_slice_fold{fold_idx+1}.html")
    except Exception as e:
        warnings.warn(f"Could not generate slice plot for fold_{fold_idx+1}: {e}")
