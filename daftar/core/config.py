"""Configuration management for DAFTAR-ML.

This module provides the configuration class that holds settings for DAFTAR-ML.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal, Union

# Default values
DEFAULT_PATIENCE = 50  # Higher default for more thorough optimization
DEFAULT_RELATIVE_THRESHOLD = 1e-6
DEFAULT_TOP_N = 25

# Metric-specific default relative thresholds
METRIC_DEFAULT_RELATIVE_THRESHOLDS = {
    "mse": 1e-6,        # very small improvements due to squared error scale
    "rmse": 1e-4,       # 0.01% relative improvement assuming moderate RMSE values
    "mae": 1e-4,        # similar to RMSE
    "r2": 1e-3,         # require 0.1% improvement
    "accuracy": 1e-3,   # 0.1% improvement in accuracy
    "f1": 1e-3,         # 0.1% improvement
    "roc_auc": 1e-3,    # 0.1% improvement
}


@dataclass
class Config:
    """Configuration for DAFTAR-ML pipeline.
    
    Attributes:
        input_file: Path to input CSV file
        target: Name of target column to predict
        problem_type: Type of problem ('regression' or 'classification')
        model: Model type to use ('xgboost' or 'random_forest')
        id_column: Column with sample IDs
        metric: Metric to optimize
        transform_features: Whether to apply log1p to features
        transform_target: Whether to apply log1p to target
        inner_folds: Number of inner CV folds
        outer_folds: Number of outer CV folds
        repeats: Number of CV repetitions
        trials: Number of hyperparameter optimization trials
        patience: Number of trials to wait without improvement before early stopping
        relative_threshold: Minimum improvement threshold for early stopping
        output_dir: Custom output directory name (will be created under results_root)
        results_root: Root directory for all results
        cores: Number of CPU cores to use
        seed: Random seed
        top_n: Number of top features to include in visualizations
        verbose: Whether to enable verbose console output
        force_overwrite: Whether to overwrite existing output directory without asking
        original_command: The original command that was used to run the pipeline
        use_stratified: Whether to use stratified splitting for classification (default: True)
        confusion_cmap: Colormap to use for confusion matrices (default: 'Blues')
    """
    # Required parameters
    input_file: str
    target: str
    problem_type: Literal['regression', 'classification']
    model: Literal['xgb', 'rf']
    
    # Required parameters (continued)
    id_column: str
    metric: Optional[str] = None  # Will be set based on problem_type if None
        
    # Cross-validation parameters
    inner_folds: int = 3
    outer_folds: int = 5
    repeats: int = 5
    
    # Optimization parameters
    trials: int = 1000  # High default, optimization will be controlled by patience/early stopping
    patience: int = DEFAULT_PATIENCE
    relative_threshold: float = DEFAULT_RELATIVE_THRESHOLD
    
    # Output parameters
    output_dir: Optional[str] = None
    results_root: Optional[str] = None
    
    # Execution parameters
    cores: int = -1  # -1 means use all cores
    seed: int = 42
    top_n: int = DEFAULT_TOP_N
    verbose: bool = False
    force_overwrite: bool = False
    
    # Visualization parameters
    confusion_cmap: Optional[str] = None
    
    # New parameter
    use_stratified: bool = True
    
    def __post_init__(self):
        """Validate and set derived attributes after initialization."""
        # Set default metric based on problem type if not specified
        if self.metric is None:
            if self.problem_type == 'regression':
                self.metric = 'mse'
            else:  # classification
                self.metric = 'accuracy'
        
        # Validate metric based on problem type
        self._validate_metric()

        # Override relative_threshold with metric-specific default when not provided
        if self.relative_threshold == DEFAULT_RELATIVE_THRESHOLD:
            self.relative_threshold = METRIC_DEFAULT_RELATIVE_THRESHOLDS.get(
                self.metric, DEFAULT_RELATIVE_THRESHOLD
            )
        
        # Convert input_file to Path
        if isinstance(self.input_file, str):
            self.input_file = Path(self.input_file)
    
    def _validate_metric(self):
        """Validate that the metric is appropriate for the problem type."""
        regression_metrics = ['mse', 'rmse', 'mae', 'r2']
        classification_metrics = ['accuracy', 'f1', 'roc_auc']
        
        if self.problem_type == 'regression' and self.metric not in regression_metrics:
            raise ValueError(
                f"Invalid metric '{self.metric}' for regression. "
                f"Choose from: {', '.join(regression_metrics)}"
            )
        elif self.problem_type == 'classification' and self.metric not in classification_metrics:
            raise ValueError(
                f"Invalid metric '{self.metric}' for classification. "
                f"Choose from: {', '.join(classification_metrics)}"
            )
    
    def get_auto_name(self) -> str:
        """Generate automatic name for the output directory based on configuration.
        
        Returns:
            Auto-generated directory name
        """
        # Map short model names to their full names for directory naming
        model_name_mapping = {
            'xgb': 'xgboost',
            'rf': 'random_forest'
        }
        
        # Get full model name from the mapping or use the original if not found
        full_model_name = model_name_mapping.get(self.model, self.model)
        
        # Create auto-generated name that includes problem type
        components = ["DAFTAR-ML", self.target, full_model_name, self.problem_type]
        
        # Transformation info now in preprocessed filename
            
        # Add CV info
        components.append(f"cv{self.outer_folds}x{self.inner_folds}x{self.repeats}")
        
        return "_".join(components)
    
    def get_output_dir(self) -> Path:
        """Get the output directory for saving results.
        
        Returns:
        
        Returns:
            Path object for the output directory (which may or may not exist yet)
        """
        # Generate auto name if needed
        auto_name = self.get_auto_name()
        
        if self.output_dir:
            # Use user-provided directory with auto-generated name
            root_dir = Path(self.output_dir)
            output_path = root_dir / auto_name
        else:
            # Use default root with auto name
            root_dir = Path(self.results_root or os.getenv("DAFTAR-ML_RESULTS_DIR", Path.cwd()))
            output_path = root_dir / auto_name
        
        return output_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization.
        
        Returns:
            Dictionary representation of config
        """
        return {
            "input_file": str(self.input_file),
            "target": self.target,
            "problem_type": self.problem_type,
            "model": self.model,
            "id_column": self.id_column,
            "metric": self.metric,
            "transform_features": self.transform_features,
            "transform_target": self.transform_target,
            "inner_folds": self.inner_folds,
            "outer_folds": self.outer_folds,
            "repeats": self.repeats,
            "trials": self.trials,
            "patience": self.patience,
            "relative_threshold": self.relative_threshold,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "results_root": str(self.results_root) if self.results_root else None,
            "cores": self.cores,
            "seed": self.seed,
            "top_n": self.top_n,
            "force_overwrite": self.force_overwrite,
            "use_stratified": self.use_stratified,
            "confusion_cmap": self.confusion_cmap
        }
