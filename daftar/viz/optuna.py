"""Optuna visualization utilities for DAFTAR-ML."""

import os
import warnings
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import optuna
import plotly.io as pio


def save_optuna_visualizations(study: Any, fold_idx: int, output_dir: Path, config: Optional[Any] = None) -> None:
    """Generate and save Optuna plots for a given fold.
    
    Args:
        study: Optuna study object
        fold_idx: Fold index
        output_dir: Output directory for plots
        config: Configuration object (optional)
    """
    # Create fold directory
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(exist_ok=True)
    
    # Create optuna plots directory
    optuna_dir = fold_dir / "optuna_plots"
    optuna_dir.mkdir(exist_ok=True)
    
    # Clean up any existing Optuna plot files
    for old_file in optuna_dir.glob("optuna_*.html"):
        try:
            old_file.unlink()
        except Exception as e:
            warnings.warn(f"Could not remove old Optuna plot file {old_file}: {e}")
    
    # Save hyperparameter tuning summary in main fold directory (for metrics combination)
    with open(fold_dir / f"hyperparam_tuning_fold_{fold_idx}.txt", "w") as f:
        f.write(f"Hyperparameter Tuning Summary for Fold {fold_idx}\n")
        f.write("=" * 60 + "\n\n")
        
        # Add study information
        best_trial = study.best_trial
        
        # Display metrics at the top
        val_metric = best_trial.value
        
        # Get training metric and gap if available
        summary_section = ""
        if 'train_metric' in best_trial.user_attrs:
            train_metric = best_trial.user_attrs['train_metric']
                
            # Calculate gap (training - validation for all metrics)
            gap = train_metric - val_metric
                
            summary_section = f"METRICS SUMMARY:\n"
            summary_section += f"Training Metric:     {train_metric:.8f}\n"
            summary_section += f"Validation Metric:   {val_metric:.8f}\n"
            summary_section += f"Training/Val Gap:    {gap:.8f}\n\n"
        else:
            summary_section = f"METRICS SUMMARY:\n"
            summary_section += f"Validation Metric:   {val_metric:.8f}\n\n"
        
        f.write(summary_section)
        f.write(f"Total Trials: {len(study.trials)}\n")
        f.write(f"Best Trial: {best_trial.number}\n")
        
        f.write("\n")
        
        # Add best hyperparameters
        f.write("Best Hyperparameters:\n")
        f.write("-" * 60 + "\n")
        
        for param_name, param_value in best_trial.params.items():
            f.write(f"{param_name}: {param_value}\n")

    # Save detailed hyperparameter tuning summary in optuna plots directory
    with open(optuna_dir / f"optuna_summary_fold_{fold_idx}.txt", "w") as f:
        f.write(f"Hyperparameter Tuning Summary for Fold {fold_idx}\n")
        f.write("=" * 60 + "\n\n")
        
        # Add study information
        best_trial = study.best_trial
            
        # Display metrics at the top
        val_metric = best_trial.value
        
        # Get training metric and gap if available
        summary_section = ""
        if 'train_metric' in best_trial.user_attrs:
            train_metric = best_trial.user_attrs['train_metric']
                
            # Calculate gap (training - validation for all metrics)
            gap = train_metric - val_metric
                
            summary_section = f"METRICS SUMMARY:\n"
            summary_section += f"Training Metric:     {train_metric:.8f}\n"
            summary_section += f"Validation Metric:   {val_metric:.8f}\n"
            summary_section += f"Training/Val Gap:    {gap:.8f}\n\n"
        else:
            summary_section = f"METRICS SUMMARY:\n"
            summary_section += f"Validation Metric:   {val_metric:.8f}\n\n"
        
        f.write(summary_section)
        f.write(f"Total Trials: {len(study.trials)}\n")
        f.write(f"Best Trial: {best_trial.number}\n")
        
        f.write("\n")
        
        # Track patience pattern
        f.write("Early Stopping Pattern:\n")
        f.write("-" * 60 + "\n")
        
        # Count trials with improvement based on study direction
        best_value_by_trial = float('-inf') if study.direction == optuna.study.StudyDirection.MAXIMIZE else float('inf')
        patience_counter = 0
        best_trial_idx = 0
        
        for i, trial in enumerate(study.trials):
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
                
            trial_value = trial.value if trial.value is not None else (float('-inf') if study.direction == optuna.study.StudyDirection.MAXIMIZE else float('inf'))
            did_improve = False
            
            # Check improvement based on study direction
            if study.direction == optuna.study.StudyDirection.MAXIMIZE:
                if trial_value > best_value_by_trial:
                    best_value_by_trial = trial_value
                    best_trial_idx = i
                    patience_counter = 0
                    did_improve = True
                else:
                    patience_counter += 1
            else:  # MINIMIZE
                if trial_value < best_value_by_trial:
                    best_value_by_trial = trial_value
                    best_trial_idx = i
                    patience_counter = 0
                    did_improve = True
                else:
                    patience_counter += 1
                
            # Get training metric if available
            train_metric = None
            if 'train_metric' in trial.user_attrs:
                train_metric = trial.user_attrs['train_metric']
            
            # Calculate gap if both metrics are available
            gap = None
            if train_metric is not None:
                gap = train_metric - trial_value
            
            improvement_marker = "✓" if did_improve else "×"
            
            # Write metrics with training metrics and gap if available
            if train_metric is not None and gap is not None:
                f.write(f"Trial {i:3d}: Val = {trial_value:.8f} | Train = {train_metric:.8f} | Gap = {gap:.8f} | ")
                f.write(f"Patience = {patience_counter:2d} | Improvement: {improvement_marker}\n")
            else:
                f.write(f"Trial {i:3d}: Val = {trial_value:.8f} | Patience = {patience_counter:2d} | Improvement: {improvement_marker}\n")
            
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
    
    # Save trials data as CSV
    trials_csv_path = fold_dir / f"optuna_trials_fold_{fold_idx}.csv"
    with open(trials_csv_path, 'w', newline='') as csvfile:
        if study.trials:
            # Get all parameter names from all trials
            all_param_names = set()
            for trial in study.trials:
                all_param_names.update(trial.params.keys())
            all_param_names = sorted(list(all_param_names))
            
            # Create header
            header = ['trial_number', 'value', 'state'] + all_param_names
            if study.trials and 'train_metric' in study.trials[0].user_attrs:
                header.append('train_metric')
            
            writer = csv.writer(csvfile)
            writer.writerow(header)
            
            # Write trial data
            for trial in study.trials:
                row = [trial.number, trial.value, trial.state.name]
                
                # Add parameter values
                for param_name in all_param_names:
                    row.append(trial.params.get(param_name, ''))
                
                # Add training metric if available
                if 'train_metric' in trial.user_attrs:
                    row.append(trial.user_attrs['train_metric'])
                elif 'train_metric' in header:
                    row.append('')
                
                writer.writerow(row)
    
    # Save optimization history plot
    try:
        # Get the base figure from Optuna
        fig = optuna.visualization.plot_optimization_history(study)
        
        if fig.data:
            # Update the plot aesthetics
            for trace in fig.data:
                if hasattr(trace, "marker") and trace.marker is not None:
                    trace.marker.colorscale = "jet"
            
            # Update the y-axis title to show the actual metric name
            if config and hasattr(config, 'metric') and hasattr(fig.layout, 'yaxis') and hasattr(fig.layout.yaxis, 'title'):
                    original_title = fig.layout.yaxis.title.text
                    if original_title:
                        # Replace "Objective Value" with the actual metric name
                        new_title = original_title.replace("Objective Value", f"{config.metric.upper()}")
                        fig.layout.yaxis.title.text = new_title
        
        # Save the figure
        pio.write_html(fig, optuna_dir / f"optuna_history_fold_{fold_idx}.html")
    except Exception as e:
        warnings.warn(f"Could not generate optimization history plot for fold_{fold_idx}: {e}")

    # Save parallel coordinate plot
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        if fig.data:
            for trace in fig.data:
                trace.line.colorscale = "jet"
                
        pio.write_html(fig, optuna_dir / f"optuna_parallel_fold_{fold_idx}.html")
    except Exception as e:
        warnings.warn(f"Could not generate parallel coordinate plot for fold_{fold_idx}: {e}")

    # Save slice plot
    try:
        fig = optuna.visualization.plot_slice(study)
        if fig.data:
            for trace in fig.data:
                if hasattr(trace, "marker") and trace.marker is not None:
                    trace.marker.colorscale = "jet"
            
            # Update the y-axis title to show the actual metric name
            if config and hasattr(config, 'metric') and hasattr(fig.layout, 'yaxis') and hasattr(fig.layout.yaxis, 'title'):
                    original_title = fig.layout.yaxis.title.text
                    if original_title:
                        metric_name = config.metric.upper() if config and hasattr(config, 'metric') else 'Metric'
                        new_title = original_title.replace("Objective Value", metric_name)
                        fig.layout.yaxis.title.text = new_title
        
        pio.write_html(fig, optuna_dir / f"optuna_slice_fold_{fold_idx}.html")
    except Exception as e:
        warnings.warn(f"Could not generate slice plot for fold_{fold_idx}: {e}")
