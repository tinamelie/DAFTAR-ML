"""Pipeline implementation for DAFTAR-ML.

This module provides the main pipeline implementation that runs the
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
from sklearn.model_selection import RepeatedKFold, KFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score
)

from daftar.core.config import Config
from daftar.core.logging_utils import setup_logging
from daftar.core.data_processing import prepare_data, init_dataset
from daftar.core.evaluation import calculate_overall_metrics, save_metrics, make_serializable
from daftar.utils.file_utils import create_figures_explanation, save_shap_values

from daftar.models.base import BaseModel, BaseRegressionModel, BaseClassificationModel
from daftar.models.regression.xgboost import XGBoostRegressionModel
from daftar.models.regression.random_forest import RandomForestRegressionModel
from daftar.models.classification.xgboost import XGBoostClassificationModel
from daftar.models.classification.random_forest import RandomForestClassificationModel

# Import visualization functions
from daftar.viz.core_plots import create_shap_summary, create_feature_importance, create_prediction_analysis
from daftar.viz.optuna import save_optuna_visualizations
from daftar.viz.shap import save_mean_shap_analysis
from daftar.viz.feature_importance import plot_feature_importance_bar, save_feature_importance_values
from daftar.viz.predictions import generate_density_plots, save_fold_predictions_vs_actual, save_top_features_summary
from daftar.viz.color_definitions import FEATURE_IMPORTANCE_BAR_COLOR, FEATURE_IMPORTANCE_BAR_BG
from daftar.viz.color_definitions import get_train_test_colors


class Pipeline:
    """Main DAFTAR-ML pipeline implementation."""
    
    def __init__(self, config: Config):
        """Initialize pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Configure Optuna logging to ensure it's captured properly
        import logging
        optuna_logger = logging.getLogger("optuna")
        optuna_logger.setLevel(logging.INFO)
        optuna_logger.propagate = True
        
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
                # First set up simple console logging for error reporting
                self.logger = setup_logging(None, config.verbose if hasattr(config, 'verbose') else False)
                
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
        
        # Set up logging with file output now that we have a valid directory
        self.logger = setup_logging(self.output_dir, config.verbose if hasattr(config, 'verbose') else False)
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
        # Use the prepare_data function from data_processing module
        X, y, feature_names, original_data = prepare_data(self.config)
        
        # Store original data if available
        if original_data is not None:
            self.original_data = original_data
        
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
        use_stratified = self.config.problem_type == 'classification' and self.config.use_stratified
        
        if use_stratified:
            self.logger.info("Using StratifiedKFold for classification task")
            cv = RepeatedStratifiedKFold(
                n_splits=self.config.outer_folds,
                n_repeats=self.config.repeats,
                random_state=self.config.seed
            )
        else:
            self.logger.info(f"Using KFold for {'classification' if self.config.problem_type == 'classification' else 'regression'} task")
            cv = RepeatedKFold(
                n_splits=self.config.outer_folds,
                n_repeats=self.config.repeats,
                random_state=self.config.seed
            )
        
        # Loop through outer folds
        for train_idx, test_idx in cv.split(X, y):
            fold_idx += 1
            self.logger.info(f"Processing fold {fold_idx}/{total_folds}")
            
            # Process fold
            fold_result = self._process_fold(fold_idx, X, y, train_idx, test_idx, feature_names)
            fold_results.append(fold_result)
            fold_metrics.append(fold_result['metrics'])
            
            # Only log fold completion once
            self.logger.info(f"Completed fold {fold_idx}")
            
            # Clean up to reduce memory usage
            gc.collect()
        
        # Calculate overall metrics using the imported function
        metrics = calculate_overall_metrics(fold_metrics)
        
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
                try:
                    # For multiclass problems, use 'ovr' (one-vs-rest) strategy
                    if len(np.unique(y_test)) > 2:
                        # For multiclass, use OVR approach
                        metrics['roc_auc'] = roc_auc_score(y_test, y_prob, multi_class='ovr')
                    else:
                        # For binary classification
                        y_prob_positive = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.ravel()
                        metrics['roc_auc'] = roc_auc_score(y_test, y_prob_positive)
                except Exception as e:
                    # Still catch any other errors that might occur
                    self.logger.warning(f"Could not calculate ROC AUC: {e}")
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
        else:
            # Some models might not have this attribute, provide zeros as fallback
            feature_importances = np.zeros(len(feature_names))
        
        # Get SHAP values if model supports it
        shap_values = None
        if hasattr(model, 'shap_values'):
            try:
                # Create DataFrame for SHAP values calculation (for better visualization)
                X_test_df = pd.DataFrame(X_test, columns=feature_names)
                shap_values = model.shap_values(X_test)
            except Exception as e:
                self.logger.warning(f"Could not calculate SHAP values for fold {fold_idx}: {e}")
                X_test_df = None
        else:
            X_test_df = None
        
        # Create plots for this fold
        if hasattr(model, 'study'):
            save_optuna_visualizations(model.study, fold_idx, output_dir, self.config)
        
        # Save the best model to the fold directory
        model_path = fold_dir / f"best_model_fold_{fold_idx}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"[Fold {fold_idx}] Best model saved to {model_path}")
        
        # Log completion is already done in the main loop
        # so we don't need to duplicate it here
        
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
        
        # NEW: Generate fold distribution histograms
        self._create_fold_distribution_histogram(fold_idx, y_train, y_test, fold_dir)
        
        # NEW: Save text file with samples in this fold
        self._save_fold_samples_list(fold_idx, train_idx, test_idx, y, fold_dir, original_ids)
        
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
    
    def _create_fold_distribution_histogram(self, fold_idx: int, y_train: np.ndarray, y_test: np.ndarray, fold_dir: Path) -> None:
        """Create histogram visualizations showing train/test data distribution for a fold.
        
        Args:
            fold_idx: Index of current fold
            y_train: Training set target values
            y_test: Test set target values
            fold_dir: Directory to save the histogram
        """
        # Convert numpy arrays to pandas Series for easier handling
        train_series = pd.Series(y_train)
        test_series = pd.Series(y_test)
        
        # Determine if this is a classification or regression problem
        is_classification = self.config.problem_type == 'classification'
        
        # Create a figure for the combined histogram
        plt.figure(figsize=(10, 6))
        
        # Get colors for train/test from the central color management
        colors = get_train_test_colors()
        
        if is_classification:
            # For classification, combine train and test data
            combined_data = pd.concat([
                pd.DataFrame({"y": train_series, "Set": "Train"}),
                pd.DataFrame({"y": test_series, "Set": "Test"})
            ])
            
            # Create countplot with our colors
            ax = sns.countplot(x="y", hue="Set", data=combined_data, palette=colors)
            
            # Remove "Set" from legend - just show Train and Test
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles, ["Train", "Test"])
            
            plt.xlabel("Class")
            plt.ylabel("Count")
        else:
            # For regression, calculate optimal bins based on combined data
            combined_data = pd.concat([train_series, test_series])
            
            # Calculate better binning for continuous data
            data_range = combined_data.max() - combined_data.min()
            n_samples = len(combined_data)
            # Use Freedman-Diaconis rule as a starting point
            bin_width = 2 * (combined_data.quantile(0.75) - combined_data.quantile(0.25)) / (n_samples**(1/3))
            if bin_width > 0:
                n_bins = max(10, min(50, int(data_range / bin_width)))
            else:
                n_bins = 20  # Fallback if bin_width calculation fails
            
            # Create histograms with calculated number of bins
            plt.hist(train_series, bins=n_bins, alpha=0.7, label="Train", color=colors["Train"])
            plt.hist(test_series, bins=n_bins, alpha=0.7, label="Test", color=colors["Test"])
            
            plt.xlabel("Target Value")
            plt.ylabel("Frequency")
            plt.legend()
        
        plt.title(f"Fold {fold_idx} - Train/Test Distribution")
        plt.tight_layout()
        
        # Save the plot to the fold directory
        hist_path = fold_dir / f"fold_{fold_idx}_distribution.png"
        plt.savefig(hist_path)
        plt.close()
        
        print(f"[Fold {fold_idx}] Distribution histogram saved to {hist_path}")
    
    def _save_fold_samples_list(self, fold_idx: int, train_idx: np.ndarray, test_idx: np.ndarray, 
                                y: np.ndarray, fold_dir: Path, original_ids=None) -> None:
        """Save a CSV file listing which samples were included in the fold.
        
        Args:
            fold_idx: Index of current fold
            train_idx: Indices of training samples
            test_idx: Indices of test samples
            y: Target values
            fold_dir: Directory to save the file
            original_ids: Original sample IDs if available
        """
        # Get original sample IDs if available
        all_ids = []
        if hasattr(self, 'original_data') and self.config.id_column in getattr(self, 'original_data', pd.DataFrame()).columns:
            # Use original IDs from dataset
            all_ids = self.original_data[self.config.id_column].tolist()
        else:
            # Create sequential IDs if none available
            all_ids = [f"Sample_{i}" for i in range(len(y))]
        
        # Create DataFrames for training and test sets
        train_data = []
        test_data = []
        
        for idx in train_idx:
            sample_id = all_ids[idx] if idx < len(all_ids) else f"Sample_{idx}"
            target_val = y[idx]
            train_data.append({
                'ID': sample_id,
                'Target_Value': target_val,
                'Set': 'Train'
            })
            
        for idx in test_idx:
            sample_id = all_ids[idx] if idx < len(all_ids) else f"Sample_{idx}"
            target_val = y[idx]
            test_data.append({
                'ID': sample_id,
                'Target_Value': target_val,
                'Set': 'Test'
            })
        
        # Combine into a single DataFrame
        fold_samples = pd.DataFrame(train_data + test_data)
        
        # Save to CSV
        samples_file = fold_dir / f"fold_{fold_idx}_samples.csv"
        fold_samples.to_csv(samples_file, index=False)
        
        print(f"[Fold {fold_idx}] Samples list saved to {samples_file}")
    
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
        
        # Save performance metrics using the imported function
        metrics_file = save_metrics(overall_metrics, results['fold_metrics'], output_dir)
        output_files['metrics'] = metrics_file
        
        # Generate basic visualizations
        self.logger.info("Generating basic visualizations...")
        
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
        # Creates both fold-level and sample-level versions in feature_importance dir
        fold_level_df, sample_level_df = save_feature_importance_values(results['fold_results'], output_dir, in_fold_dirs=True)
        
        # Track output files
        feature_importance_dir = output_dir / "feature_importance"
        output_files['feature_importance_values_fold'] = str(feature_importance_dir / 'feature_importance_values_fold.csv')
        output_files['feature_importance_values_sample'] = str(feature_importance_dir / 'feature_importance_values_sample.csv')
        output_files['feature_importance_values'] = str(output_dir / 'feature_importance_overall.csv')  # Legacy path
        
        # Save per-fold and concatenated SHAP values as CSV files
        save_shap_values(results['fold_results'], output_dir)
        
        # Create fold-level feature importance bar plot
        self.logger.info("Creating fold-level feature importance bar plot...")
        plot_feature_importance_bar(fold_level_df, output_dir, self.config.top_n, 
                                   bar_color=FEATURE_IMPORTANCE_BAR_COLOR, bar_opacity=1.0, bg_color=FEATURE_IMPORTANCE_BAR_BG, 
                                   plot_type="fold")
        
        # Create sample-level feature importance bar plot
        self.logger.info("Creating sample-level feature importance bar plot...")
        plot_feature_importance_bar(sample_level_df, output_dir, self.config.top_n, 
                                   bar_color=FEATURE_IMPORTANCE_BAR_COLOR, bar_opacity=1.0, bg_color=FEATURE_IMPORTANCE_BAR_BG, 
                                   plot_type="sample")
        
        # Track all feature importance bar plots
        output_files['feature_importance_bar'] = str(output_dir / 'feature_importance_bar.png')  # Legacy path
        output_files['feature_importance_bar_fold'] = str(feature_importance_dir / 'feature_importance_bar_fold.png')
        output_files['feature_importance_bar_sample'] = str(feature_importance_dir / 'feature_importance_bar_sample.png')
        
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
        
        # Get confusion matrix colormap from config if available
        confusion_cmap = getattr(self.config, 'confusion_cmap', None)
        
        generate_density_plots(
            fold_results=results['fold_results'],
            all_true_values=all_true_values,
            all_predictions=all_predictions,
            output_dir=output_dir,
            target_name=self.config.target,
            problem_type=self.config.problem_type,
            cmap=confusion_cmap
        )
        
        # Create figures explanation file
        explanation_path = output_dir / "figures_explanation.txt"
        create_figures_explanation(output_dir)
        output_files['figures_explanation'] = str(explanation_path)
        
        # Save analysis summary
        self.logger.info("Analysis complete.")
        
        # Return overall results
        return {
            'metrics': overall_metrics,
            'output_files': output_files,
            'shap_analysis': shap_df.to_dict() if shap_df is not None else {},
            'feature_importance': fold_level_df.to_dict() if 'fold_level_df' in locals() and fold_level_df is not None else {}
        }
