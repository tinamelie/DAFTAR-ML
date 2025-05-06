"""Pipeline implementation for DAFTAR-ML.

This module provides the main pipeline implementation that orchestrates the
model training, evaluation, and analysis process.
"""

import logging
import os
import gc
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score
)

from daftar.core.config import Config
from daftar.models.base import BaseModel, BaseRegressionModel, BaseClassificationModel
from daftar.models.regression.xgboost import XGBoostRegressionModel
from daftar.models.regression.random_forest import RandomForestRegressionModel
from daftar.models.classification.xgboost import XGBoostClassificationModel
from daftar.models.classification.random_forest import RandomForestClassificationModel

# Import visualization functions from original DAFTAR-ML modules
from daftar.viz.core_plots import create_shap_summary, create_feature_importance, create_prediction_analysis
from daftar.viz.optuna import save_optuna_visualizations
from daftar.viz.shap import save_mean_shap_analysis
from daftar.viz.feature_importance import plot_feature_importance_bar, save_feature_importance_values
from daftar.viz.predictions import generate_density_plots, save_fold_predictions_vs_actual, save_top_features_summary


def setup_logging(output_dir=None):
    """Set up logging configuration.
    
    Args:
        output_dir: Optional directory to write log file to
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if output directory is provided
    if output_dir:
        try:
            output_dir = os.path.abspath(output_dir)
            # IMPORTANT: Do NOT create the directory here with os.makedirs
            # This directory must have been validated and created before
            # we get here, or else we risk creating directories we don't want
            log_file = os.path.join(output_dir, 'DAFTAR-ML_run.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.warning(f"Failed to set up file logging: {e}")
    
    return logger


class Pipeline:
    """Main DAFTAR-ML pipeline implementation."""
    
    def __init__(self, config: Config):
        """Initialize pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # First set up ONLY console logging without any file output
        self.logger = setup_logging(None)
        
        # Calculate expected output path WITHOUT creating anything yet
        auto_name = config.get_auto_name()
        if config.output_dir:
            # Use user-provided directory with auto-generated name
            root_dir = Path(config.output_dir)
            output_path = root_dir / auto_name
        else:
            # Use default root with auto name
            root_dir = Path(config.results_root or os.getenv("DAFTAR-ML_RESULTS_DIR", Path.cwd()))
            output_path = root_dir / auto_name
            
        # Check if directory exists and contains files BEFORE creating anything
        if output_path.exists():
            # First make a list to avoid any dir creation during checking
            entries = []
            try:
                if output_path.is_dir():
                    entries = list(output_path.iterdir())
            except Exception:
                pass  # Just continue if we can't check
                
            if entries and not config.force_overwrite:
                error_msg = f"Output directory already exists and contains files: {output_path}\n"
                error_msg += f"Use --force flag to overwrite existing files.\n\n"
                # Only include required parameters in the example
                error_msg += f"Example: daftar --input {config.input_file} --target {config.target} --id {config.id_column} --model {config.model}"
                
                # Only add output_dir to example if it was explicitly specified
                if config.output_dir:
                    error_msg += f" --output_dir {config.output_dir}"
                    
                error_msg += " --force"
                
                raise FileExistsError(error_msg)
        
        # Now it's safe to create the directory
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_path
        
        # Now we can safely add file logging since we've validated the directory
        self.logger = setup_logging(self.output_dir)
        self.model = None
        
    def run(self) -> Dict[str, Any]:
        """Run the pipeline.
        
        Returns:
            Dictionary with results and analysis
        """
        self.logger.info("Starting DAFTAR-ML pipeline")
        
        # Log the command that was used to run the pipeline
        if hasattr(self.config, 'original_command'):
            self.logger.info(f"Command: {self.config.original_command}")
        
        # Load and validate data
        X, y, feature_names = self._prepare_data()
        
        # Run nested cross-validation
        cv_results = self._run_nested_cv(X, y, feature_names)
        
        # Analyze results and generate visualizations
        analysis_results = self._analyze_results(cv_results, X, y, feature_names)
        
        self.logger.info("Pipeline completed successfully")
        
        return {**cv_results, **analysis_results}
    
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and prepare data for analysis.
        
        Returns:
            Tuple of (feature matrix, target vector, feature names)
        """
        # Load data
        self.logger.info(f"Loading data from {self.config.input_file}")
        try:
            data = pd.read_csv(self.config.input_file)
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
        
        # Initialize dataset
        return self._init_dataset(data)
    
    def _init_dataset(
        self, data: pd.DataFrame, feature_names: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Initialize the dataset.
        
        Args:
            data: Input DataFrame with features and target
            feature_names: Optional list of feature names
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Get target and ID column
        if self.config.target not in data.columns:
            raise ValueError(f"Target column '{self.config.target}' not found in data")
        
        # Store original data with IDs for later reference
        if self.config.id_column and self.config.id_column in data.columns:
            # Make a copy of the original data with index set to match the dataset
            self.original_data = data.copy().reset_index(drop=True)
            # Remove ID column from working data
            data = data.drop(columns=[self.config.id_column])
        
        # Split features and target
        y = data[self.config.target].values
        X = data.drop(columns=[self.config.target])
        
        # Get feature names
        if feature_names is None:
            feature_names = X.columns.tolist()
        
        # Convert to numpy arrays
        X = X.values
        
        # Note: All transformations are now applied during preprocessing
        
        return X, y, feature_names
    
    def _create_model(self) -> BaseModel:
        """Create model based on configuration.
        
        Returns:
            Model instance
        """
        if self.config.problem_type == 'regression':
            if self.config.model == 'xgb':
                return XGBoostRegressionModel(
                    metric=self.config.metric,
                    n_trials=self.config.trials,
                    n_jobs=self.config.cores,
                    patience=self.config.patience,
                    relative_threshold=self.config.relative_threshold,
                    seed=self.config.seed
                )
            elif self.config.model == 'rf':
                return RandomForestRegressionModel(
                    metric=self.config.metric,
                    n_trials=self.config.trials,
                    n_jobs=self.config.cores,
                    patience=self.config.patience,
                    relative_threshold=self.config.relative_threshold,
                    seed=self.config.seed
                )
            else:
                raise ValueError(f"Unsupported regression model: {self.config.model}")
        elif self.config.problem_type == 'classification':
            if self.config.model == 'xgb':
                return XGBoostClassificationModel(
                    metric=self.config.metric,
                    n_trials=self.config.trials,
                    n_jobs=self.config.cores,
                    patience=self.config.patience,
                    relative_threshold=self.config.relative_threshold,
                    seed=self.config.seed
                )
            elif self.config.model == 'rf':
                return RandomForestClassificationModel(
                    metric=self.config.metric,
                    n_trials=self.config.trials,
                    n_jobs=self.config.cores,
                    patience=self.config.patience,
                    relative_threshold=self.config.relative_threshold,
                    seed=self.config.seed
                )
            else:
                raise ValueError(f"Unsupported classification model: {self.config.model}")
        else:
            raise ValueError(f"Unsupported problem type: {self.config.problem_type}")
    
    def _run_nested_cv(
        self, X: np.ndarray, y: np.ndarray, feature_names: list
    ) -> Dict[str, Any]:
        """Run nested cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            
        Returns:
            Dictionary containing results
        """
        # Initialize variables to store results
        fold_results = []
        fold_metrics = []
        fold_idx = 0
        
        # Total number of folds = outer_folds * repeats
        total_folds = self.config.outer_folds * self.config.repeats
        
        # Configure outer cross-validation
        cv = RepeatedKFold(
            n_splits=self.config.outer_folds,
            n_repeats=self.config.repeats,
            random_state=self.config.seed
        )
        
        # Loop through outer folds
        for train_idx, test_idx in cv.split(X):
            fold_idx += 1
            self.logger.info(f"Processing fold {fold_idx}/{total_folds}")
            
            # Process fold
            fold_result = self._process_fold(fold_idx, X, y, train_idx, test_idx, feature_names)
            fold_results.append(fold_result)
            fold_metrics.append(fold_result['metrics'])
            
            self.logger.info(f"Completed fold {fold_idx}")
            
            # Clean up to reduce memory usage
            gc.collect()
        
        # Calculate overall metrics
        metrics = self._calculate_overall_metrics(fold_metrics)
        
        return {
            'fold_results': fold_results,
            'fold_metrics': fold_metrics,
            'metrics': metrics,
            'feature_names': feature_names
        }
    
    def _process_fold(
        self, fold_idx: int, X: np.ndarray, y: np.ndarray,
        train_idx: np.ndarray, test_idx: np.ndarray,
        feature_names: list
    ) -> Dict[str, Any]:
        """Process a single cross-validation fold.
        
        Args:
            fold_idx: Index of current fold
            X: Feature matrix
            y: Target vector
            train_idx: Training set indices
            test_idx: Test set indices
            feature_names: List of feature names
            
        Returns:
            Dictionary containing fold results
        """
        # Extract training and test sets
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create fold directory inside the main output directory
        output_dir = self.config.get_output_dir()
        output_dir.mkdir(exist_ok=True, parents=True)
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True, parents=True)
        
        # Create and train model
        model = self._create_model()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probability predictions for classification problems
        y_prob = None
        if self.config.problem_type == 'classification' and hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
        
        # Calculate metrics based on problem type
        if self.config.problem_type == 'regression':
            metrics = {
                'mse': ((y_test - y_pred) ** 2).mean(),
                'rmse': np.sqrt(((y_test - y_pred) ** 2).mean()),
                'mae': np.abs(y_test - y_pred).mean(),
                'r2': 1 - ((y_test - y_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
            }
        else:  # classification
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            # Add ROC AUC if model supports predict_proba
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                if y_prob.shape[1] == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_test, y_prob[:, 1])
        
        # Get feature importance
        feature_importances = model.feature_importances_
        
        # Calculate SHAP values
        shap_values = model.shap_values(X_test)
        
        # Convert to DataFrame for SHAP analysis
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Save Optuna visualizations
        if hasattr(model, 'study'):
            save_optuna_visualizations(model.study, fold_idx, output_dir, self.config)
        
        # Save the best model to the fold directory
        model_path = fold_dir / f"best_model_fold_{fold_idx}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"[Fold {fold_idx}] Best model saved to {model_path}")
        
        # Log completion
        self.logger.info(f"Completed fold {fold_idx}")
        
        # Create IDs (we don't have real IDs, so use indices)
        ids = test_idx.tolist()
        
        # Get original IDs if available (from input dataset)
        original_ids = None
        if hasattr(self, 'original_data') and self.config.id_column and self.config.id_column in getattr(self, 'original_data', pd.DataFrame()).columns:
            # Map indices back to original IDs
            original_ids = [self.original_data.loc[idx, self.config.id_column] 
                          if idx in self.original_data.index else f'Sample_{idx}' 
                          for idx in test_idx]
        
        # Create fold-wise predictions vs actual CSV
        out_csv = fold_dir / f"predictions_vs_actual_fold_{fold_idx}.csv"
        save_fold_predictions_vs_actual(fold_idx, ids, y_pred, y_test, output_dir, original_ids=original_ids)
        
        # Return results
        result = {
            'fold_index': fold_idx,
            'predictions': y_pred.tolist(),
            'true_values': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_test': y_test.tolist(),
            'X_test': X_test,
            'ids_test': ids,
            'original_ids': original_ids,
            'feature_importances': pd.Series(feature_importances, index=feature_names),
            'shap_values': shap_values,
            'metrics': metrics,
            'metric': self.config.metric,  # Add selected metric from config
            'study': model.study if hasattr(model, 'study') else None,
            'shap_data': (shap_values, X_test_df, y_test)
        }
        
        # Add probability predictions if available
        if y_prob is not None:
            result['y_prob'] = y_prob
            
        return result
    
    def _calculate_overall_metrics(self, fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
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
    
    def _analyze_results(
        self, results: Dict[str, Any], X: np.ndarray,
        y: np.ndarray, feature_names: list
    ) -> Dict[str, Any]:
        """Analyze results and generate visualizations.
        
        Args:
            results: Pipeline results
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            
        Returns:
            Dictionary with analysis results and paths to generated files
        """
        output_dir = self.config.get_output_dir()
        self.logger.info(f"Saving results to {output_dir}")
        
        # Store output paths for reporting
        output_files = {}
        
        # Calculate overall metrics
        overall_metrics = {}
        for metric in results['fold_metrics'][0].keys():
            overall_metrics[metric] = np.mean([m[metric] for m in results['fold_metrics']])
        
        # Save performance metrics
        metrics_file = self._save_metrics(overall_metrics, results['fold_metrics'], output_dir)
        output_files['metrics'] = metrics_file
        
        # Generate basic visualizations
        self.logger.info("Generating basic visualizations...")
        
        # Don't create SHAP summary plot anymore as requested - but track file
        shap_summary_path = output_dir / 'shap_summary.png'
        output_files['shap_summary'] = str(shap_summary_path)
        
        # Don't create feature importance plot anymore as requested - but track file
        feat_imp_path = output_dir / 'feature_importance.png'
        output_files['feature_importance'] = str(feat_imp_path)
        
        # Collect all true values and predictions from fold results
        all_true_values = []
        all_predictions = []
        for fold_result in results['fold_results']:
            all_true_values.extend(fold_result['y_test'].tolist() if isinstance(fold_result['y_test'], np.ndarray) else fold_result['y_test'])
            all_predictions.extend(fold_result['y_pred'])
            
        
        # SHAP analysis / visualization
        self.logger.info("Performing SHAP analysis (signed, focusing on positive/negative impact)...")
        shap_df = save_mean_shap_analysis(results['fold_results'], output_dir, problem_type=self.config.problem_type, top_n=self.config.top_n)
        output_files['shap_analysis'] = str(output_dir / 'shap_feature_impact_analysis.csv')
        
        # Save top features summary based on SHAP
        self.logger.info("Saving SHAP-based features summary...")
        summary_path = output_dir / 'shap_features_summary.txt'
        save_top_features_summary(shap_df, output_dir, self.config)
        output_files['features_summary'] = str(summary_path)
        
        # Feature importance
        self.logger.info("Saving feature importance values...")
        # Save per-fold feature importance values in their respective fold directories
        # And save consolidated overall values in main output dir
        overall_df = save_feature_importance_values(results['fold_results'], output_dir, in_fold_dirs=True)
        output_files['feature_importance_values'] = str(output_dir / 'feature_importance_overall.csv')
        
        # Save per-fold and concatenated SHAP values as CSV files
        self._save_shap_values(results['fold_results'], output_dir)
        
        # Create feature importance bar plot
        imp_bar_path = output_dir / 'feature_importance_bar.png'
        plot_feature_importance_bar(overall_df, output_dir, self.config.top_n, bar_color="#968FF3", bar_opacity=1.0, bg_color="#E6E6E6")
        output_files['feature_importance_bar'] = str(imp_bar_path)
        
        # Generate predictions vs actual and density plots (density only for regression)
        if self.config.problem_type == "regression":
            self.logger.info("Generating density plots...")
            density_path = output_dir / "density_actual_vs_pred_global.png"
            output_files['density_plot'] = str(density_path)
        else:
            self.logger.info("Generating predictions and confusion matrices...")
            
        # Call function for both classification and regression
        # For classification, it will only generate confusion matrices
        # For regression, it will generate density plots
        generate_density_plots(
            fold_results=results['fold_results'],
            all_true_values=all_true_values,
            all_predictions=all_predictions,
            output_dir=output_dir,
            target_name=self.config.target,
            problem_type=self.config.problem_type
        )
        
        # Create figures explanation file
        explanation_path = output_dir / "figures_explanation.txt"
        self._create_figures_explanation(output_dir)
        output_files['figures_explanation'] = str(explanation_path)
        
        # Save analysis summary
        self.logger.info("Analysis complete.")
        
        # Return overall results
        return {
            'metrics': overall_metrics,
            'output_files': output_files,
            'shap_analysis': shap_df.to_dict(),
            'feature_importance': overall_df.to_dict()
        }
        
    def _save_shap_values(self, fold_results: List[Dict[str, Any]], output_dir: Path) -> None:
        """Save SHAP values from all folds to CSV files.
        
        Args:
            fold_results: List of fold results
            output_dir: Output directory path
        """
        # Concatenate SHAP values from all folds
        all_samples_shap_values = []
        
        for fold_idx, fold_result in enumerate(fold_results):
            shap_values = fold_result['shap_values']
            X_test = fold_result['X_test']
            y_test = fold_result['y_test']
            feature_names = fold_result['feature_importances'].index.tolist()
            
            # Use original IDs if available, otherwise use test IDs
            if 'original_ids' in fold_result and fold_result['original_ids'] is not None:
                sample_ids = fold_result['original_ids']
            else:
                sample_ids = fold_result['ids_test']
            
            # Skip if no SHAP values (shouldn't happen)
            if shap_values is None or len(shap_values) == 0:
                continue
                
            # Handle different SHAP value shapes based on problem type
            # Classification models can have multiple formats based on the SHAP explainer
            if len(shap_values.shape) == 3:  # shape: (n_classes, n_samples, n_features)
                # For multiclass, we'll just use the values for class 1 (positive class)
                shap_values = shap_values[1]  # Select positive class SHAP values
            
            # Make sure we don't exceed the bounds of arrays
            n_samples = min(len(shap_values), len(sample_ids), len(y_test))
            
            # For each sample, collect all feature SHAP values
            for i in range(n_samples):
                sample_id = sample_ids[i]
                target_value = y_test[i]
                sample_shap = shap_values[i]
                
                # Handle feature dimension safely
                n_features = min(len(sample_shap), len(feature_names))
                
                # Store all SHAP values for this sample
                sample_shap_dict = {feature_names[j]: sample_shap[j] for j in range(n_features)}
                
                # Add metadata
                sample_shap_dict['ID'] = sample_id  # Use a consistent ID column name
                sample_shap_dict['Fold'] = fold_idx + 1  # 1-indexed folds
                sample_shap_dict['Target'] = target_value  # Add target value
                
                # Append to the list of samples
                all_samples_shap_values.append(sample_shap_dict)
        
        # Create DataFrame with one row per sample, features as columns
        shap_df = pd.DataFrame(all_samples_shap_values)
        
        # Reorder columns to put ID, Fold, and Target first
        metadata_cols = ['ID', 'Fold', 'Target']
        other_cols = [col for col in shap_df.columns if col not in metadata_cols]
        ordered_cols = metadata_cols + other_cols
        shap_df = shap_df[ordered_cols]
        
        # Save to CSV
        shap_df.to_csv(output_dir / 'shap_values_all_folds.csv', index=False)
        
        # Save per-fold SHAP values
        for fold_idx, fold_result in enumerate(fold_results):
            fold_dir = output_dir / f"fold_{fold_idx+1}"
            fold_dir.mkdir(exist_ok=True)
            
            # Filter for this fold
            fold_shap_df = shap_df[shap_df['Fold'] == fold_idx + 1].copy()
            # Use the same column ordering as the main SHAP CSV
            fold_shap_df.to_csv(fold_dir / f"shap_values_fold_{fold_idx+1}.csv", index=False)
    
    def _create_figures_explanation(self, output_dir: Path) -> None:
        """Create explanation file for all figures.
        
        Args:
            output_dir: Output directory path
        """
        explanation_text = [
            "FIGURES EXPLANATION",
            "=" * 50,
            "",
            "This document explains the various figures generated by DAFTAR-ML.",
            "",
            "1. SHAP Summary Plot (shap_summary.png)",
            "-" * 50,
            "The SHAP summary plot shows the impact of each feature on the model output.",
            "Features are ranked by their mean absolute SHAP value.",
            "Each point represents a sample, with color indicating the feature value (red=high, blue=low).",
            "Points to the right indicate positive impact on the prediction, left indicates negative impact.",
            "",
            "2. Feature Importance Plot (feature_importance.png)",
            "-" * 50,
            "This bar chart shows the importance of each feature according to the model.",
            "Features are ranked by their mean importance across all cross-validation folds.",
            "Error bars indicate the standard deviation of importance across folds.",
            "",

            "4. Global Density Plot (density_actual_vs_pred_global.png)",
            "-" * 50,
            "Distribution of actual vs. predicted values across all folds.",
            "Includes metrics (MAE, MSE, RMSE, R2) in the right margin.",
            "",
            "5. SHAP Beeswarm Plot (shap_beeswarm_colored_global.png)",
            "-" * 50,
            "This plot shows the distribution of SHAP values for each feature.",
            "Each point represents a sample, with color indicating the feature value.",
            "Features are ordered by their mean absolute SHAP value.",
            "",
            "6. SHAP Bar Plots (shap_bar_top25pos_top25neg.png)",
            "-" * 50,
            "This bar chart shows the top 25 features with positive impact and top 25 with negative impact.",
            "Features are ranked by their mean SHAP value across all samples.",
            "Error bars indicate the standard deviation of SHAP values across folds.",
            "",
            "7. SHAP-Target Correlation (shap_corr_bar_top25pos_top25neg.png) - Regression Only",
            "-" * 50,
            "This bar chart shows the correlation between SHAP values and the target variable (generated only for regression problems).",
            "Features with high positive correlation have SHAP values that align well with the target.",
            "Features with high negative correlation have SHAP values that inversely align with the target.",
            "",
            "8. Feature Importance Bar (feature_imp_bar_top25.png)",
            "-" * 50,
            "This bar chart shows the top 25 features by importance.",
            "Features are ranked by their mean importance across all folds.",
            "Error bars indicate the standard deviation of importance across folds.",
            "",
            "9. Optuna Plots (in fold_X/optuna_plots/)",
            "-" * 50,
            "- optuna_history_foldX.html: Optimization history showing how the objective value improved over trials.",
            "- optuna_parallel_foldX.html: Parallel coordinate plot showing the relationship between hyperparameters and objective value.",
            "- optuna_slice_foldX.html: Slice plot showing the effect of each hyperparameter on the objective value.",
            "",
            "10. CSV Files",
            "-" * 50,
            "- predictions_vs_actual_overall.csv: Contains all predictions, actual values, and residuals.",
            "- fold_X/predictions_vs_actual_fold_X.csv: Contains predictions for each fold.",
            "- feature_importance_overall.csv: Contains mean and std of feature importance across folds.",
            "- shap_feature_impact_analysis.csv: Contains comprehensive SHAP statistics for all features.",
        ]
        
        with open(output_dir / "figures_explanation.txt", "w") as f:
            f.write("\n".join(explanation_text))
    
    def _save_metrics(self, metrics: Dict, fold_metrics: List[Dict], output_dir: Path) -> str:
        """
        Save overall and per-fold metrics to a text file.
        
        Args:
            metrics: Overall metrics dictionary
            fold_metrics: List of per-fold metrics dictionaries
            output_dir: Output directory path
            
        Returns:
            Path to the created metrics text file
        """
        # Save metrics to JSON
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({"overall": metrics, "folds": fold_metrics}, f, indent=2)
            
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
            
        return str(metrics_file)
        
    def _make_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
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
