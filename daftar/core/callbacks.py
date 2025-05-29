"""Callback implementations for DAFTAR-ML optimization.

This module provides callback functions used during hyperparameter optimization.
"""

from typing import Dict, Any, List

import numpy as np
import optuna


class RelativeEarlyStoppingCallback:
    """Early stopping callback based on relative improvement.
    
    This callback stops the optimization when no improvement exceeding
    the relative threshold is observed for more than patience trials.
    """
    
    def __init__(self, patience: int, relative_threshold: float):
        """Initialize the callback.
        
        Args:
            patience: Number of trials to wait without improvement
            relative_threshold: Minimum relative improvement to consider
        """
        self.patience = patience
        self.relative_threshold = relative_threshold
        self.best_value = None
        self.best_trial = None
        self.stagnation_count = 0
        
        # Get the problem type and metric from the environment if possible
        self.problem_type = None
        self.metric = None
        try:
            import os
            # These are set by the models when fitting
            self.problem_type = os.environ.get('DAFTAR-ML_PROBLEM_TYPE', None)
            self.metric = os.environ.get('DAFTAR-ML_METRIC', None)
        except:
            pass
        
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Called after each trial.
        
        Args:
            study: Optuna study object
            trial: Current trial object
        """
        current_value = study.best_value
        current_trial_num = trial.number
        
        # For display, convert the optimization value to natural metric direction
        # Only for metrics that should naturally be maximized
        def to_display_value(value):
            # Only negate maximizing metrics (accuracy, f1, roc_auc, r2) 
            # since they're already negated for optuna minimization
            if (self.metric in ['accuracy', 'f1', 'roc_auc'] or 
                (self.problem_type == 'regression' and self.metric == 'r2')):
                return -value  # Convert back to positive by negating
            return value
            
        display_current_value = to_display_value(current_value)
        display_trial_value = to_display_value(trial.value)
        
        # Helper to format metric values based on magnitude for readability
        def fmt_metric(value: float) -> str:
            """Format metric value for display with appropriate precision."""
            if abs(value) >= 0.1:
                return f"{value:.6f}"
            elif abs(value) >= 0.01:
                return f"{value:.8f}"
            elif abs(value) >= 0.001:
                return f"{value:.10f}"
            else:
                # For extremely small values, use scientific notation
                return f"{value:.2e}"
        
        # First trial
        if self.best_value is None:
            self.best_value = current_value
            self.best_trial = study.best_trial.number
            
            # Display with proper sign for metrics that are naturally maximized
            is_max_metric = (
                (self.metric in ['accuracy', 'f1', 'roc_auc']) or
                (self.problem_type == 'regression' and self.metric == 'r2')
            )
            
            if is_max_metric:
                # For maximization metrics, flip the sign (since we internally negate them)
                display_value = abs(current_value) if current_value < 0 else current_value  # Always show positive values
                print(f"[Trial {current_trial_num}] First trial: Score = {fmt_metric(display_value)}, patience counter = 0/{self.patience}")
            else:
                # For minimization metrics, use the value as is but ensure it's positive for display
                display_value = abs(current_value) if self.metric in ['mse', 'rmse', 'mae'] else current_value
                print(f"[Trial {current_trial_num}] First trial: Score = {fmt_metric(display_value)}, patience counter = 0/{self.patience}")
            return
        
        # For minimization (default in Optuna)
        if study.direction == optuna.study.StudyDirection.MINIMIZE:
            # Safe division to avoid division by zero
            if abs(self.best_value) < 1e-10:  # Near zero
                improvement = float('inf') if current_value < self.best_value else 0.0
            else:
                improvement = (self.best_value - current_value) / abs(self.best_value)
            is_better = current_value < self.best_value
            compare_symbol = '<'
        # For maximization
        else:
            # Safe division to avoid division by zero
            if abs(self.best_value) < 1e-10:  # Near zero
                improvement = float('inf') if current_value > self.best_value else 0.0
            else:
                improvement = (current_value - self.best_value) / abs(self.best_value)
            is_better = current_value > self.best_value
            compare_symbol = '>'            
            
        # Check if this is a metric that should be displayed with flipped sign
        should_flip_sign = (
            (self.metric in ['accuracy', 'f1', 'roc_auc']) or
            (self.problem_type == 'regression' and self.metric == 'r2')
        )
        
        # Format values for display with adaptive precision
        rel_improvement = improvement * 100  # Convert to percentage
        threshold_pct = self.relative_threshold * 100
        
        # Determine appropriate precision based on threshold magnitude
        if threshold_pct < 0.0001:  # Very small threshold (< 0.0001%)
            decimals = 8
        elif threshold_pct < 0.01:  # Small threshold (< 0.01%)
            decimals = 6
        elif threshold_pct < 0.1:  # Medium threshold (< 0.1%)
            decimals = 4
        elif threshold_pct < 1.0:  # Large threshold (< 1%)
            decimals = 3
        else:  # Very large threshold (≥ 1%)
            decimals = 2
        
        # If there's a significant improvement
        if is_better and improvement > self.relative_threshold:
            prev_value = self.best_value
            self.best_value = current_value
            self.best_trial = study.best_trial.number
            self.stagnation_count = 0
            
            # Display values with correct sign for metrics
            if should_flip_sign:
                # For maximization metrics (like accuracy), display as positive values
                display_value = abs(current_value) if current_value < 0 else current_value 
                display_prev = abs(prev_value) if prev_value < 0 else prev_value
                # For maximizing metrics, the comparison should be > not <
                print(f"[Trial {current_trial_num}] New best: {fmt_metric(display_value)} > {fmt_metric(display_prev)}, "
                      f"patience counter reset = 0/{self.patience}")
            else:
                # For minimization metrics (like MSE), ensure positive display values
                display_value = abs(current_value) if self.metric in ['mse', 'rmse', 'mae'] else current_value
                display_prev = abs(prev_value) if self.metric in ['mse', 'rmse', 'mae'] else prev_value
                print(f"[Trial {current_trial_num}] New best: {fmt_metric(display_value)} {compare_symbol} {fmt_metric(display_prev)}, "
                      f"patience counter reset = 0/{self.patience}")
        else:
            self.stagnation_count += 1
            # Distinguish between identical values and actual worse values
            if study.best_trial.number == current_trial_num:
                # This is a new best value but improvement isn't significant enough
                if should_flip_sign:
                    # For maximization metrics (like accuracy), display as positive values
                    display_value = abs(current_value) if current_value < 0 else current_value
                    display_best = abs(self.best_value) if self.best_value < 0 else self.best_value
                    # For maximizing metrics, the comparison should always be >
                    print(f"[Trial {current_trial_num}] Marginal improvement: {fmt_metric(display_value)} > {fmt_metric(display_best)}, "
                          f"patience counter = {self.stagnation_count}/{self.patience}")
                else:
                    # For minimization metrics (like MSE), ensure positive display values
                    display_value = abs(current_value) if self.metric in ['mse', 'rmse', 'mae'] else current_value
                    display_best = abs(self.best_value) if self.metric in ['mse', 'rmse', 'mae'] else self.best_value
                    print(f"[Trial {current_trial_num}] Marginal improvement: {fmt_metric(display_value)} {compare_symbol} {fmt_metric(display_best)}, "
                          f"patience counter = {self.stagnation_count}/{self.patience}")
            else:
                # This trial is worse than the best so far
                if should_flip_sign:
                    # For maximization metrics (like accuracy), display as positive values
                    display_trial = abs(trial.value) if trial.value < 0 else trial.value
                    display_best = abs(self.best_value) if self.best_value < 0 else self.best_value
                    print(f"[Trial {current_trial_num}] No improvement: Current = {fmt_metric(display_trial)}, Best = {fmt_metric(display_best)}, "
                          f"patience counter = {self.stagnation_count}/{self.patience}")
                else:
                    # For minimization metrics (like MSE), ensure positive display values
                    display_trial = abs(trial.value) if self.metric in ['mse', 'rmse', 'mae'] else trial.value
                    display_best = abs(self.best_value) if self.metric in ['mse', 'rmse', 'mae'] else self.best_value
                    print(f"[Trial {current_trial_num}] No improvement: Current = {fmt_metric(display_trial)}, Best = {fmt_metric(display_best)}, "
                          f"patience counter = {self.stagnation_count}/{self.patience}")
            
        # If we've exceeded patience, stop the optimization
        if self.stagnation_count >= self.patience:
            study.stop()
            print(f"[Trial {current_trial_num}] Early stopping triggered: no significant improvement for {self.stagnation_count} consecutive trials")
