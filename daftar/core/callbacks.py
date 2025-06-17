"""Callback implementations for DAFTAR-ML optimization.

This module provides callback functions used during hyperparameter optimization.
"""

# Terminal color codes
YELLOW = '\033[93m'
CYAN = '\033[96m'
GREEN = '\033[92m'
BRIGHT_GREEN = '\033[92;1m'
BOLD = '\033[1m'
PINK = '\033[95m'
RESET = '\033[0m'

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
        
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Called after each trial.
        
        Args:
            study: Optuna study object
            trial: Current trial object
        """
        current_value = study.best_value
        current_trial_num = trial.number
        
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
            
            print(
                f"[Trial {current_trial_num}] First trial: Score = {fmt_metric(current_value)}, "
                f"patience counter = 0/{self.patience}")
            return
        
        # For minimization
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
        
        # If there's a significant improvement
        if is_better and improvement > self.relative_threshold:
            prev_value = self.best_value
            self.best_value = current_value
            self.best_trial = study.best_trial.number
            self.stagnation_count = 0
            
            print(f"{GREEN}[Trial {current_trial_num}] New best: {fmt_metric(current_value)} {compare_symbol} {fmt_metric(prev_value)}, "
                f"patience counter reset = 0/{self.patience}{RESET}")
        else:
            self.stagnation_count += 1
            
            # Check if there was some improvement (but not enough to reset counter)
            if study.best_trial.number == current_trial_num:
                print(
                    f"[Trial {current_trial_num}] Improvement not significant: {fmt_metric(current_value)} {compare_symbol} {fmt_metric(self.best_value)}, "
                    f"patience counter = {self.stagnation_count}/{self.patience}")
            else:
                # This trial is worse than the best so far
                print(
                    f"[Trial {current_trial_num}] No improvement: Current = {fmt_metric(current_value)}, "
                    f"Best = {fmt_metric(self.best_value)}, patience counter = {self.stagnation_count}/{self.patience}")
            
        # If we've exceeded patience, stop the optimization
        if self.stagnation_count >= self.patience:
            study.stop()
            print(f"[Trial {current_trial_num}] Early stopping triggered: no significant improvement for {self.stagnation_count} consecutive trials")
