"""
Pipeline implementation for DAFTAR-ML.

This module provides the main pipeline implementation that runs the
model training, evaluation, and analysis process.
"""

import logging
import os
import gc
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import daftar  # Import for version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from daftar.viz.common import save_plot
from sklearn.model_selection import (
    RepeatedKFold, KFold, RepeatedStratifiedKFold, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score
)

from daftar.core.config import Config
from daftar.core.logging_utils import setup_logging
from daftar.core.data_processing import prepare_data, init_dataset
from daftar.core.evaluation import (
    calculate_overall_metrics, save_metrics, make_serializable
)

from daftar.utils.file_utils import (
    combine_metrics_files, combine_metrics_for_fold, save_shap_values,
    save_figures_explanation
)

from daftar.models.base import BaseModel, BaseRegressionModel, BaseClassificationModel
from daftar.models.xgboost_regression import XGBoostRegressionModel
from daftar.models.rf_regression import RandomForestRegressionModel
from daftar.models.xgboost_classification import XGBoostClassificationModel
from daftar.models.rf_classification import RandomForestClassificationModel

# visualisation helpers --------------------------------------------------------
from daftar.viz.core_plots import (
    create_shap_summary, create_feature_importance,
    create_prediction_analysis
)
from daftar.viz.optuna import save_optuna_visualizations
from daftar.viz.shap import (
    save_mean_shap_analysis
)
from daftar.viz.feature_importance import (
    plot_feature_importance_bar, save_feature_importance_values
)
from daftar.viz.predictions import (
    generate_density_plots, save_fold_predictions_vs_actual
)
from daftar.viz.colors import (
    FEATURE_IMPORTANCE_BAR_COLOR,
    FEATURE_IMPORTANCE_BAR_BG,
    HISTOGRAM_BG_COLOR,
    CLASSIFICATION_BAR_BG_COLOR,
    get_train_test_colors
)

# ──────────────────────────────────────────────────────────────────────────────
# NEW – SHAP interactions
# ──────────────────────────────────────────────────────────────────────────────
from daftar.viz.shap_interaction_utils import save_shap_interactions_analysis


class Pipeline:
    """Main DAFTAR-ML pipeline implementation."""
    
    # Terminal color codes
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    BRIGHT_GREEN = '\033[92;1m'
    BOLD = '\033[1m'
    PINK = '\033[95m'
    RESET = '\033[0m'
    
    def __init__(self, config: Config):
        """Initialize pipeline."""
        self.config = config
        
        # ------------------------------------------------------------------ log
        optuna_logger = logging.getLogger("optuna")
        optuna_logger.setLevel(logging.INFO)
        optuna_logger.propagate = True
        
        # ---------------------------------------------------------------- output
        auto_name = config.get_auto_name()
        root_dir = Path(config.output_dir) if config.output_dir else Path(
            config.results_root or os.getenv("DAFTAR-ML_RESULTS_DIR", Path.cwd())
        )
        output_path = root_dir / auto_name
            
        # Create output directory (--force flag handles overwrite behavior elsewhere)
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_path
        
        self.logger = setup_logging(
            self.output_dir, getattr(config, "verbose", False)
        )
        self.model = None
        
    @staticmethod
    def _format_metric_name(name: str) -> str:
        """Format metric name for display (e.g., r2 -> R²)."""
        if name.lower() == 'r2':
            return 'R²'
        return name.upper()
    
    # ────────────────────────────────────────────────────────────────────────
    # public entry
    # ────────────────────────────────────────────────────────────────────────

    
    def run(self) -> Dict[str, Any]:
        """Run the pipeline."""
        # Log to file but don't print to console
        if hasattr(self.config, "original_command"):
            print(f"Command: {self.config.original_command}")
        
        X, y, feature_names = self._prepare_data()
        cv_results          = self._run_nested_cv(X, y, feature_names)
        analysis_results    = self._analyze_results(
            cv_results, X, y, feature_names
        )
        # Add timestamp for when the pipeline completes
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Pipeline completed successfully - {end_time}")
        return {**cv_results, **analysis_results}
    
    # ────────────────────────────────────────────────────────────────────────
    # data preparation
    # ────────────────────────────────────────────────────────────────────────
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        X, y, feature_names, original_data = prepare_data(self.config)
        if original_data is not None:
            self.original_data = original_data
        return X, y, feature_names
    
    # ────────────────────────────────────────────────────────────────────────
    # model factory
    # ────────────────────────────────────────────────────────────────────────
    def _create_model(self) -> BaseModel:
        model_params = {
            "metric": self.config.metric,
            "n_trials": self.config.trials,
            "n_jobs": self.config.cores,
            "patience": self.config.patience,
            "relative_threshold": self.config.relative_threshold,
            "seed": self.config.seed,
        }
        
        model_map = {
            ("regression", "xgb"): XGBoostRegressionModel,
            ("regression", "rf"): RandomForestRegressionModel,
            ("classification", "xgb"): XGBoostClassificationModel,
            ("classification", "rf"): RandomForestClassificationModel,
        }
        
        model_key = (self.config.problem_type, self.config.model)
        if model_key not in model_map:
            raise ValueError(f"Unsupported {self.config.problem_type} model: {self.config.model}")
        
        return model_map[model_key](**model_params)
    
    # ────────────────────────────────────────────────────────────────────────
    # nested CV
    # ────────────────────────────────────────────────────────────────────────
    def _run_nested_cv(
        self, X: np.ndarray, y: np.ndarray, feature_names: list
    ) -> Dict[str, Any]:
        fold_results: list = []
        fold_metrics: list = []
        fold_studies: dict = {}
        fold_idx      = 0
        total_folds   = self.config.outer_folds * self.config.repeats
        
        use_strat = (
            self.config.problem_type == "classification"
            and self.config.use_stratified
        )
        if use_strat:
            cv = RepeatedStratifiedKFold(
                n_splits=self.config.outer_folds,
                n_repeats=self.config.repeats,
                random_state=self.config.seed,
            )
        else:
            cv = RepeatedKFold(
                n_splits=self.config.outer_folds,
                n_repeats=self.config.repeats,
                random_state=self.config.seed,
            )
        
        # -------------------------------------------------------------- loop
        VERSION = daftar.__version__  # Use centralized version from package
        problem_type = self.config.problem_type.upper() if hasattr(self.config, 'problem_type') else 'UNKNOWN'
        metric = self.config.metric if hasattr(self.config, 'metric') else ''
        dataset = self.config.input_file if hasattr(self.config, 'input_file') else ''
        target = self.config.target if hasattr(self.config, 'target') else ''
        model = self.config.model if hasattr(self.config, 'model') else ''
        model_friendly = {'xgb': 'Gradient boosting', 'rf': 'Random Forest'}.get(model, model.capitalize() if model else '')
        kfold_type = ("StratifiedKFold" if (self.config.problem_type == "classification" and getattr(self.config, "use_stratified", False)) else "KFold")

        HEADER = f"{self.CYAN}==============================\n  DAFTAR-ML PIPELINE SUMMARY\n=============================={self.RESET}"
        print(f"\n{HEADER}")
        print(f"Detected task  : {problem_type}")
        print(f"Metric         : {metric.upper() if metric.lower() != 'r2' else 'R²'}")
        print(f"CV Split       : {kfold_type}")
        print(f"Dataset        : {dataset}")
        print(f"Target         : {target}")
        print(f"Model          : {model_friendly}")
        
        # Add timestamp for when the pipeline starts
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Time started   : {start_time}\n")

        for train_idx, test_idx in cv.split(X, y):
            fold_idx += 1
            print(f"{self.YELLOW}----------- Fold {fold_idx}/{total_folds} -----------{self.RESET}")
            fold_res = self._process_fold(
                fold_idx, X, y, train_idx, test_idx, feature_names
            )
            fold_results.append(fold_res)
            fold_metrics.append(fold_res["metrics"])
            print(f"{self.GREEN}  Fold {fold_idx}/{total_folds} complete.{self.RESET}")
            print()
            gc.collect()

        metrics = calculate_overall_metrics(fold_metrics)
        
        try:
            all_folds_metrics = []
            for i, fold_metric in enumerate(fold_metrics, 1):
                fold_data = {'fold': i}
                fold_data.update(fold_metric)
                all_folds_metrics.append(fold_data)
            
            if all_folds_metrics:
                all_metrics_df = pd.DataFrame(all_folds_metrics)
                
                metric_cols = [col for col in all_metrics_df.columns if col != 'fold']
                summary_data = {
                    'fold': ['mean', 'std', 'min', 'max']
                }
                
                for metric in metric_cols:
                    values = all_metrics_df[metric].values
                    summary_data[metric] = [
                        np.mean(values),
                        np.std(values),
                        np.min(values),
                        np.max(values)
                    ]
                
                summary_df = pd.DataFrame(summary_data)
                
                empty_row = pd.DataFrame(
                    {col: [""] if col == "fold" else [np.nan] for col in all_metrics_df.columns}, 
                    index=[0]
                )
                
                combined_df = pd.concat([all_metrics_df, empty_row, summary_df], ignore_index=True)
                
                self.test_metrics_df = combined_df
                
                print(f"{self.CYAN}=============================================={self.RESET}")
                print(f" AVERAGE TEST METRICS ACROSS FOLDS{self.RESET}")
                print(f"{self.CYAN}=============================================={self.RESET}")
                for metric in metric_cols:
                    mean_val = summary_data[metric][0]
                    std_val = summary_data[metric][1]
                    
                    # Format the metric name (R2 -> R²)
                    metric_label = self._format_metric_name(metric)
                    
                    if metric.lower() == self.config.metric.lower():
                        # Highlight the optimization metric
                        print(f"  {self.GREEN}{metric_label:<8}: {mean_val:.6f} ± {std_val:.6f}{self.RESET}")
                    else:
                        print(f"  {metric_label:<8}: {mean_val:.6f} ± {std_val:.6f}")
                print(f"{self.CYAN}=============================={self.RESET}")
                
        except Exception as e:
            self.logger.warning(f"Error creating test metrics summary: {e}")
        
        return dict(
            fold_results=fold_results,
            fold_metrics=fold_metrics,
            metrics=metrics,
            feature_names=feature_names,
            fold_studies=fold_studies,
        )
    
    # ────────────────────────────────────────────────────────────────────────
    # one fold
    # ────────────────────────────────────────────────────────────────────────
    def _process_fold(
        self,
        fold_idx: int,
        X: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        feature_names: list,
    ) -> Dict[str, Any]:
        # Initialize fold results dictionary
        fold_res = {"fold_studies": {}}
        
        # Create fold directory
        fold_dir = self.output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True, parents=True)
        
        # Print and log fold info
        print(f"\n{self.CYAN}Processing fold {fold_idx}...{self.RESET}")
        self.logger.info(f"Processing fold {fold_idx}")
        
        # ------------------------------------------------ data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create inner CV for hyperparameter tuning
        if self.config.problem_type == "classification" and self.config.use_stratified:
            inner_cv = StratifiedKFold(
                n_splits=self.config.inner_folds, shuffle=True, random_state=self.config.seed
            )
        else:
            inner_cv = KFold(
                n_splits=self.config.inner_folds, shuffle=True, random_state=self.config.seed
            )
        
        # ------------------------------------------------ model
        print(f"{self.CYAN}  Training model with hyperparameter optimization...{self.RESET}")
        model = self._create_model()
        model.fit(X_train, y_train, inner_cv)
        
        # ------------------------------------------------ predictions
        y_pred = model.predict(X_test)
        
        # For classification, also get probabilities if available
        y_prob = None
        if self.config.problem_type == "classification" and hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
            except Exception as e:
                self.logger.warning(f"Could not get prediction probabilities: {e}")
        
        # ------------------------------------------------ metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_prob, fold_idx, fold_dir)
        
        # Print test metrics immediately after calculation
        print(f"\n{self.CYAN}  Fold {fold_idx} test performance:{self.RESET}")
        for k, v in metrics.items():
            metric_label = self._format_metric_name(k)
            if k.lower() == (self.config.metric.lower() if hasattr(self.config, 'metric') else ''):
                print(f"    {self.GREEN}{metric_label:<8}: {v:.6f}{self.RESET}")
            else:
                print(f"    {metric_label:<8}: {v:.6f}")
        print()  # Add space after metrics
        
        # ------------------------------------------------ feature importance
        feature_importances = getattr(model, "feature_importances_", np.zeros(len(feature_names)))
        
        # Save per-fold feature importance immediately (top features will be trimmed later if needed)
        try:
            fi_series = pd.Series(feature_importances, index=feature_names, name="Importance")
            fi_csv = fold_dir / f"feature_importance_fold_{fold_idx}.csv"
            fi_series.to_csv(fi_csv, header=True, index_label="Feature")
    
        except Exception as e:
            self.logger.warning(f"Could not save per-fold feature importance: {e}")
        
        # ------------------------------------------------ ids / predictions csv
        # Save predictions immediately after computing them
        ids = test_idx.tolist()
        original_ids = self._get_original_ids(test_idx)
        
        save_fold_predictions_vs_actual(
            fold_idx,
            ids,
            y_pred,
            y_test,
            self.output_dir,
            original_ids=original_ids,
            problem_type=self.config.problem_type,
            config=self.config,
        )
        
        # ------------------------------------------------ persist model
        # Save model immediately after training
        model_path = fold_dir / f"best_model_fold_{fold_idx}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # ------------------------------------------------ Optuna figures
        # Save Optuna visualizations immediately
        if hasattr(model, "study"):
            # Save study object for later summary
            fold_res["fold_studies"][fold_idx] = model.study
            
            # Generate individual fold visualizations
            save_optuna_visualizations(
                model.study, fold_idx, self.output_dir, self.config
            )
        
        # ------------------------------------------------ basic visualizations
        # Create basic visualizations immediately
        print(f"{self.CYAN}  Creating basic visualization plots...{self.RESET}")
        
        # ------------------------------------------------ density plot (regression only)
        if self.config.problem_type == "regression":
            try:
                self._create_fold_density_plot(fold_idx, y_test, y_pred, fold_dir, self.config.target)
            except Exception as e:
                self.logger.warning(f"Failed to create density plot for fold {fold_idx}: {e}")
        
        # ------------------------------------------------ helper visuals
        self._create_fold_distribution_histogram(
            fold_idx, y_train, y_test, fold_dir
        )
        self._save_fold_samples_list(
            fold_idx, train_idx, test_idx, y, fold_dir, original_ids
        )
        
        # ------------------------------------------------ SHAP
        print(f"{self.CYAN}  Analyzing feature importance with SHAP values...{self.RESET}")
        shap_values = None
        X_test_df   = None
        if hasattr(model, "shap_values"):
            try:
                X_test_df   = pd.DataFrame(X_test, columns=feature_names)
                shap_values = model.shap_values(X_test)
                print(f"SHAP feature analysis complete.")
            except Exception as e:
                self.logger.warning(
                    f"Could not calculate SHAP values for fold {fold_idx}: {e}"
                )
        
        # ------------------------------------------------ SHAP Interactions  
        # Compute and save SHAP interactions for regression models only (TreeExplainer compatibility issues with classification)
        if (self.config.problem_type == "regression" and X_test_df is not None and not self.config.skip_interaction):
            try:
                print(f"{self.CYAN}  Calculating SHAP interaction values. This may take a moment...{self.RESET}")
                from daftar.viz.shap_interaction_utils import compute_fold_shap_interactions
                
                # Create temporary fold result for interaction computation
                temp_fold_result = {
                    "model": model,
                    "shap_data": (shap_values, X_test_df, y_test)
                }
                
                # Compute interactions for this fold
                interaction_result = compute_fold_shap_interactions(
                    temp_fold_result, fold_idx, self.output_dir
                )
                
                if interaction_result is not None:
                    interactions_df, interaction_matrix, interaction_features = interaction_result
                    
                    # Save interactions to CSV in fold directory
                    interactions_csv = fold_dir / f"shap_interactions_fold_{fold_idx}.csv"
                    interactions_df.to_csv(interactions_csv, index=False)
                    print(f"SHAP interaction analysis complete.")
                else:
                    self.logger.warning(f"[Fold {fold_idx}] Failed to compute SHAP interactions")
                    
            except Exception as e:
                self.logger.warning(f"[Fold {fold_idx}] SHAP interactions computation failed: {e}")
                # Continue without interactions - don't fail the entire fold
                
        # ------------------------------------------------ result dict
        result = dict(
            fold_index=fold_idx,
            predictions=y_pred.tolist(),
            true_values=y_test.tolist(),
            y_pred=y_pred.tolist(),
            y_test=y_test.tolist(),
            X_test=X_test,
            ids_test=ids,
            original_ids=original_ids,
            feature_importances=pd.Series(feature_importances, index=feature_names),
            shap_values=shap_values,
            metrics=metrics,
            metric=self.config.metric,
            study=model.study if hasattr(model, "study") else None,
            shap_data=(shap_values, X_test_df, y_test),
        )
        if y_prob is not None:
            result["y_prob"] = y_prob
        return result
    
    def _calculate_metrics(self, y_test, y_pred, y_prob, fold_idx, fold_dir):
        """Calculate and save metrics for the current fold."""
        if self.config.problem_type == "regression":
            # Calculate test metrics using sklearn for consistency
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store in metrics dictionary
            metrics = dict(mse=mse, rmse=rmse, mae=mae, r2=r2)
            
            # Save per-fold test metrics to CSV
            metrics_df = pd.DataFrame({
                'fold': [fold_idx],
                'mse': [mse],
                'rmse': [rmse],
                'mae': [mae],
                'r2': [r2]
            })
        else:
            # Calculate classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            
            metrics = dict(accuracy=accuracy, f1=f1)
            
            roc_auc = None
            if y_prob is not None:
                try:
                    if len(np.unique(y_test)) > 2:
                        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
                    else:
                        pos = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.ravel()
                        roc_auc = roc_auc_score(y_test, pos)
                    metrics["roc_auc"] = roc_auc
                except Exception as e:
                    self.logger.warning(f"Could not calculate ROC AUC: {e}")
            
            # Save metrics to CSV
            metrics_dict = {
                'fold': [fold_idx],
                'accuracy': [accuracy],
                'f1': [f1]
            }
            if roc_auc is not None:
                metrics_dict['roc_auc'] = [roc_auc]
                
            metrics_df = pd.DataFrame(metrics_dict)
        
        # Create combined metrics file for this fold
        combine_metrics_for_fold(fold_dir, fold_idx, metrics_df, self.config)
        
        # Add metrics to predictions CSV for better integration
        self._add_metrics_to_predictions_csv(fold_idx, fold_dir, metrics)
        
        return metrics
    
    def _add_metrics_to_predictions_csv(self, fold_idx, fold_dir, metrics):
        """Add metrics as metadata to predictions CSV."""
        pred_csv = fold_dir / f"predictions_vs_actual_fold_{fold_idx}.csv"
        if pred_csv.exists():
            try:
                # Read existing predictions CSV
                with open(pred_csv, 'r') as f:
                    pred_content = f.read()
                
                # Create metric header
                if self.config.problem_type == "regression":
                    metric_header = f"# Fold {fold_idx} Test Metrics: R² = {metrics['r2']:.6f}, RMSE = {metrics['rmse']:.6f}, MAE = {metrics['mae']:.6f}"
                else:
                    metric_header = f"# Fold {fold_idx} Test Metrics: "
                    if self.config.metric == 'accuracy':
                        metric_header += f"Accuracy = {metrics['accuracy']:.6f}"
                    elif self.config.metric == 'f1':
                        metric_header += f"F1 = {metrics['f1']:.6f}"
                    elif self.config.metric == 'roc_auc' and 'roc_auc' in metrics:
                        metric_header += f"ROC AUC = {metrics['roc_auc']:.6f}"
                
                # Write header and content
                with open(pred_csv, 'w') as f:
                    f.write(f"{metric_header}\n{pred_content}")
            except Exception as e:
                self.logger.warning(f"Could not add metrics to predictions CSV: {e}")
    
    def _get_original_ids(self, test_idx):
        """Get original IDs for test samples."""
        if (
            hasattr(self, "original_data")
            and self.config.id_column
            and self.config.id_column in getattr(self, "original_data", pd.DataFrame()).columns
        ):
            return [
                self.original_data.loc[idx, self.config.id_column]
                if idx in self.original_data.index
                else f"Sample_{idx}"
                for idx in test_idx
            ]
        return None
    
    # ────────────────────────────────────────────────────────────────────────
    # helper visuals per fold
    # ────────────────────────────────────────────────────────────────────────
    def _create_fold_distribution_histogram(
        self,
        fold_idx: int,
        y_train: np.ndarray,
        y_test: np.ndarray,
        fold_dir: Path,
    ) -> None:
        train_series = pd.Series(y_train)
        test_series  = pd.Series(y_test)
        is_cls = self.config.problem_type == "classification"
        
        plt.figure(figsize=(10, 6))
        colors = get_train_test_colors()
        
        if is_cls:
            # Convert encoded labels to display labels for visualization
            display_train = train_series
            display_test = test_series
            if (hasattr(self.config, 'label_encoder') and 
                self.config.label_encoder is not None):
                try:
                    display_train = pd.Series(self.config.label_encoder.inverse_transform(train_series))
                    display_test = pd.Series(self.config.label_encoder.inverse_transform(test_series))
                except Exception as e:
                    print(f"Warning: Could not decode labels for histogram display: {e}")
            
            combined = pd.concat([
                pd.DataFrame({"y": display_train, "Set": "Train"}),
                pd.DataFrame({"y": display_test,  "Set": "Test"}),
            ])
            ax = sns.countplot(x="y", hue="Set", data=combined, palette=colors)
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles, ["Train", "Test"])
            plt.xlabel("Class")
            plt.ylabel("Count")
            ax.set_facecolor(CLASSIFICATION_BAR_BG_COLOR)
        else:
            combined = pd.concat([train_series, test_series])
            data_range = combined.max() - combined.min()
            n_samples  = len(combined)
            bin_width  = 2 * (
                combined.quantile(0.75) - combined.quantile(0.25)
            ) / (n_samples ** (1 / 3))
            n_bins = (
                max(10, min(50, int(data_range / bin_width)))
                if bin_width > 0
                else 20
            )
            plt.hist(
                train_series,
                bins=n_bins,
                alpha=0.7,
                label="Train",
                color=colors["Train"],
            )
            plt.hist(
                test_series,
                bins=n_bins,
                alpha=0.7,
                label="Test",
                color=colors["Test"],
            )
            plt.xlabel("Target Value")
            plt.ylabel("Frequency")
            plt.legend()
            plt.gca().set_facecolor(HISTOGRAM_BG_COLOR)
        
        plt.title(f"Fold {fold_idx} - Train/Test Distribution")
        plt.tight_layout()
        hist_path = fold_dir / f"test_train_splits_fold_{fold_idx}.png"
        save_plot(plt.gcf(), hist_path, tight_layout=True)
    
    def _save_fold_samples_list(
        self,
        fold_idx: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        y: np.ndarray,
        fold_dir: Path,
        original_ids=None,
    ) -> None:
        all_ids = []
        if (
            hasattr(self, "original_data")
            and self.config.id_column
            in getattr(self, "original_data", pd.DataFrame()).columns
        ):
            all_ids = self.original_data[self.config.id_column].tolist()
        else:
            all_ids = [f"Sample_{i}" for i in range(len(y))]
        
        rows = []
        for idx in train_idx:
            # Convert encoded label to display label if needed
            display_value = y[idx]
            if (hasattr(self.config, 'label_encoder') and 
                self.config.label_encoder is not None):
                try:
                    display_value = self.config.label_encoder.inverse_transform([y[idx]])[0]
                except Exception:
                    pass  # Use original value if conversion fails
            
            rows.append(
                dict(ID=all_ids[idx], Target_Value=display_value, Set="Train")
            )
        for idx in test_idx:
            # Convert encoded label to display label if needed
            display_value = y[idx]
            if (hasattr(self.config, 'label_encoder') and 
                self.config.label_encoder is not None):
                try:
                    display_value = self.config.label_encoder.inverse_transform([y[idx]])[0]
                except Exception:
                    pass  # Use original value if conversion fails
            
            rows.append(
                dict(ID=all_ids[idx], Target_Value=display_value, Set="Test")
            )
        csv_path = fold_dir / f"test_train_splits_fold_{fold_idx}.csv"
        pd.DataFrame(rows).to_csv(
            csv_path, index=False
        )
    
    # ────────────────────────────────────────────────────────────────────────
    # helper: per-fold density plot (regression)
    # ────────────────────────────────────────────────────────────────────────
    def _create_fold_density_plot(
        self,
        fold_idx: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fold_dir: Path,
        target_name: str,
    ) -> None:
        """Save KDE density plot of actual vs predicted for one fold (regression)."""
        plt.figure(figsize=(10, 6))
        sns.kdeplot(y_true, label="Actual", fill=True, alpha=0.4, color="#00BFC4")
        sns.kdeplot(y_pred, label="Predicted", fill=True, alpha=0.4, color="#F8766D")
        plt.title(f"Density Plot – Fold {fold_idx}")
        plt.xlabel(target_name)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Calculate metrics for the fold
        scores = {}
        scores['MSE'] = sklearn.metrics.mean_squared_error(y_true, y_pred)
        scores['RMSE'] = np.sqrt(scores['MSE'])
        scores['MAE'] = sklearn.metrics.mean_absolute_error(y_true, y_pred)
        scores['R2'] = sklearn.metrics.r2_score(y_true, y_pred)
        
        # Get the selected metric (consistent with overall plots)
        metric = self.config.metric.lower() if hasattr(self.config, 'metric') else 'mse'
        
        # Create the metric text with appropriate precision
        if metric == 'rmse':
            metrics_text = f"RMSE: {scores['RMSE']:.7f}"
        elif metric == 'mse':
            metrics_text = f"MSE: {scores['MSE']:.7f}"
        elif metric == 'mae':
            metrics_text = f"MAE: {scores['MAE']:.7f}"
        elif metric == 'r2':
            metrics_text = f"R²: {scores['R2']:.7f}"
        else:
            # Default to MSE if metric not recognized
            metrics_text = f"MSE: {scores['MSE']:.7f}"
        
        # Add small margin on right side and add a labeled metrics box
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        metrics_box_text = f"Test Metric:\n{metrics_text}"
        plt.gca().text(1.02, 0.5, metrics_box_text, transform=plt.gca().transAxes,
                      ha='left', va='center', bbox=dict(facecolor='white', alpha=0.7))
        
        fold_dir.mkdir(exist_ok=True, parents=True)
        plot_path = fold_dir / f"density_plot_fold_{fold_idx}.png"
        save_plot(plt.gcf(), plot_path, tight_layout=True)
    
    # ────────────────────────────────────────────────────────────────────────
    # final analysis / visualisation
    # ────────────────────────────────────────────────────────────────────────
    def _analyze_results(
        self,
        results: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
    ) -> Dict[str, Any]:
        output_dir = self.config.get_output_dir()        
        output_files: dict[str, str] = {}
        
        overall_metrics = {
            m: np.mean([fm[m] for fm in results["fold_metrics"]])
            for m in results["fold_metrics"][0].keys()
        }
        metrics_file = save_metrics(
            overall_metrics, results["fold_metrics"], output_dir
        )
        output_files["metrics"] = metrics_file
        
        # -------------------------------------------- SHAP (main-effect)
        shap_df = save_mean_shap_analysis(
            results["fold_results"],
            output_dir,
            problem_type=self.config.problem_type,
            top_n=self.config.top_n
        )
        output_files["shap_analysis"] = str(
            output_dir / "shap_feature_analysis.csv"
        )
        save_figures_explanation(
            shap_df=shap_df,
            output_dir=output_dir,
            problem_type=self.config.problem_type,
            top_n=self.config.top_n
        )
        output_files["features_summary"] = str(
            output_dir / "shap_top_features_summary.txt"
        )
                
        # -------------------------------------------- feature importance
        fi_df = save_feature_importance_values(
            results["fold_results"], output_dir, in_fold_dirs=True
        )
        fi_dir = output_dir / "feature_importance"
        output_files["feature_importance_values"] = str(
            fi_dir / "feature_importance_values.csv"
        )
        plot_feature_importance_bar(
            fi_df,
            output_dir,
            self.config.top_n,
            bar_color=FEATURE_IMPORTANCE_BAR_COLOR,
            bar_opacity=1.0,
            bg_color=FEATURE_IMPORTANCE_BAR_BG,
        )
        output_files["feature_importance_bar"] = str(
            fi_dir / "feature_importance_bar.png"
        )
        
        # -------------------------------------------- density / confusion
        all_true = []
        all_pred = []
        for fr in results["fold_results"]:
            all_true.extend(
                fr["y_test"].tolist()
                if isinstance(fr["y_test"], np.ndarray)
                else fr["y_test"]
            )
            all_pred.extend(fr["y_pred"])
        
        confusion_cmap = getattr(self.config, "confusion_cmap", None)
        generate_density_plots(
            fold_results=results["fold_results"],
            all_true_values=all_true,
            all_predictions=all_pred,
            output_dir=output_dir,
            target_name=self.config.target,
            problem_type=self.config.problem_type,
            cmap=confusion_cmap,
            config=self.config,
        )
        if self.config.problem_type == "regression":
            output_files["density_plot"] = str(
                output_dir / "density_plot_overall.png"
            )
        
        # SHAP feature interactions for regression models only (TreeExplainer compatibility issues with classification)
        if self.config.problem_type == "regression" and not self.config.skip_interaction:
            try:
                # NEVER recalculate, only use pre-calculated interactions from the SHAP phase
                inter_files = save_shap_interactions_analysis(
                    results["fold_results"],
                    output_dir,
                    top_n_interactions=self.config.top_n,
                    shap_df=shap_df
                )
                output_files.update(inter_files)
                print(f"Generated interaction visualizations for {self.config.model} {self.config.problem_type}")
            except RuntimeError as e:
                self.logger.error(f"Failed to generate interaction visualizations: {e}")
                self.logger.error("SHAP interactions must be calculated during the initial SHAP analysis phase.")
                self.logger.error("Check if each fold is properly computing and saving interaction values.")
                # Don't raise error for interactions - continue with other analyses
                self.logger.warning("Continuing without interaction analysis...")
            except Exception as e:
                self.logger.warning(f"Interaction analysis failed: {e}")
        elif self.config.skip_interaction:
            print(f"Skipping SHAP interaction analysis (--skip_interaction flag set)")
        else:
            print(f"SHAP interaction analysis not supported for {self.config.problem_type} models")
        
        # Additional classification-specific analysis
        if self.config.problem_type == "classification":
            # Aggregate per-class SHAP statistics that were computed during each fold
            try:
                print("Aggregating per-class SHAP analysis across folds ...")
                from daftar.viz.per_class_shap import save_per_class_shap_analysis
                
                per_class_files = save_per_class_shap_analysis(
                    results["fold_results"],
                    output_dir,
                    feature_names=feature_names,
                    top_n=self.config.top_n,
                    config=self.config,
                )
                output_files.update(per_class_files)
            except Exception as e:
                self.logger.warning(f"Per-class SHAP analysis failed: {e}")
        
        # -------------------------------------------- combine metrics files
        # Create combined metrics table with test metrics and hyperparameter tuning metrics
        if hasattr(self, 'test_metrics_df'):
            combined_metrics_path = combine_metrics_files(output_dir, self.test_metrics_df, self.config)
            if combined_metrics_path:
                output_files["metrics_all"] = str(combined_metrics_path)
        
        # -------------------------------------------- figures explanation
        # Create a general figures explanation file
        save_figures_explanation(shap_df=None, output_dir=output_dir, 
                               problem_type="classification" if self.config.problem_type == "classification" else "regression",
                               filename="output_files_explanation.txt")
        output_files["output_files_explanation"] = str(
            output_dir / "output_files_explanation.txt"
        )
        
        return dict(
            metrics=overall_metrics,
            output_files=output_files,
            shap_analysis=shap_df.to_dict() if shap_df is not None else {},
            feature_importance=fi_df.to_dict() if fi_df is not None else {},
        )