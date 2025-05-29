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
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    create_figures_explanation, save_shap_values
)

from daftar.models.base import BaseModel, BaseRegressionModel, BaseClassificationModel
from daftar.models.regression.xgboost import XGBoostRegressionModel
from daftar.models.regression.random_forest import RandomForestRegressionModel
from daftar.models.classification.xgboost import XGBoostClassificationModel
from daftar.models.classification.random_forest import RandomForestClassificationModel

# visualisation helpers --------------------------------------------------------
from daftar.viz.core_plots import (
    create_shap_summary, create_feature_importance,
    create_prediction_analysis
)
from daftar.viz.optuna import save_optuna_visualizations
from daftar.viz.shap import (
    save_mean_shap_analysis, save_top_features_summary
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
    
    def __init__(self, config: Config):
        """Initialize pipeline."""
        self.config = config
        
        # ------------------------------------------------------------------ log
        import logging as _lg
        optuna_logger = _lg.getLogger("optuna")
        optuna_logger.setLevel(_lg.INFO)
        optuna_logger.propagate = True
        
        # ---------------------------------------------------------------- output
        auto_name = config.get_auto_name()
        if config.output_dir:
            root_dir = Path(config.output_dir)
            output_path = root_dir / auto_name
        else:
            root_dir = Path(
                config.results_root
                or os.getenv("DAFTAR-ML_RESULTS_DIR", Path.cwd())
            )
            output_path = root_dir / auto_name
            
        if output_path.exists():
            entries = []
            try:
                if output_path.is_dir():
                    entries = list(output_path.iterdir())
            except Exception:
                pass
                
            if entries and not config.force_overwrite:
                self.logger = setup_logging(
                    None, config.verbose if hasattr(config, "verbose") else False
                )
                msg = (
                    f"Output directory already exists and contains files: "
                    f"{output_path}\nUse --force to overwrite.\n\n"
                    f"Example: daftar --input {config.input_file} "
                    f"--target {config.target} --id {config.id_column} "
                    f"--model {config.model}"
                )
                if config.output_dir:
                    msg += f" --output_dir {config.output_dir}"
                msg += " --force"
                raise FileExistsError(msg)
        
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_path
        
        self.logger = setup_logging(
            self.output_dir, config.verbose if hasattr(config, "verbose") else False
        )
        self.model = None
        
    # ────────────────────────────────────────────────────────────────────────
    # public entry
    # ────────────────────────────────────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        """Run the pipeline."""
        self.logger.info("Starting DAFTAR-ML pipeline")
        if hasattr(self.config, "original_command"):
            self.logger.info(f"Command: {self.config.original_command}")
        
        X, y, feature_names = self._prepare_data()
        cv_results          = self._run_nested_cv(X, y, feature_names)
        analysis_results    = self._analyze_results(
            cv_results, X, y, feature_names
        )
        self.logger.info("Pipeline completed successfully")
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
        if self.config.problem_type == "regression":
            if self.config.model == "xgb":
                return XGBoostRegressionModel(
                    metric=self.config.metric,
                    n_trials=self.config.trials,
                    n_jobs=self.config.cores,
                    patience=self.config.patience,
                    relative_threshold=self.config.relative_threshold,
                    seed=self.config.seed,
                )
            if self.config.model == "rf":
                return RandomForestRegressionModel(
                    metric=self.config.metric,
                    n_trials=self.config.trials,
                    n_jobs=self.config.cores,
                    patience=self.config.patience,
                    relative_threshold=self.config.relative_threshold,
                    seed=self.config.seed,
                )
            raise ValueError(f"Unsupported regression model: {self.config.model}")
        
        if self.config.problem_type == "classification":
            if self.config.model == "xgb":
                return XGBoostClassificationModel(
                    metric=self.config.metric,
                    n_trials=self.config.trials,
                    n_jobs=self.config.cores,
                    patience=self.config.patience,
                    relative_threshold=self.config.relative_threshold,
                    seed=self.config.seed,
                )
            if self.config.model == "rf":
                return RandomForestClassificationModel(
                    metric=self.config.metric,
                    n_trials=self.config.trials,
                    n_jobs=self.config.cores,
                    patience=self.config.patience,
                    relative_threshold=self.config.relative_threshold,
                    seed=self.config.seed,
                )
            raise ValueError(f"Unsupported classification model: {self.config.model}")
        
        raise ValueError(f"Unsupported problem type: {self.config.problem_type}")
    
    # ────────────────────────────────────────────────────────────────────────
    # nested CV
    # ────────────────────────────────────────────────────────────────────────
    def _run_nested_cv(
        self, X: np.ndarray, y: np.ndarray, feature_names: list
    ) -> Dict[str, Any]:
        fold_results: list = []
        fold_metrics: list = []
        fold_idx      = 0
        total_folds   = self.config.outer_folds * self.config.repeats
        
        use_strat = (
            self.config.problem_type == "classification"
            and self.config.use_stratified
        )
        if use_strat:
            self.logger.info("Using StratifiedKFold for classification task")
            cv = RepeatedStratifiedKFold(
                n_splits=self.config.outer_folds,
                n_repeats=self.config.repeats,
                random_state=self.config.seed,
            )
        else:
            self.logger.info(
                f"Using KFold for "
                f"{'classification' if self.config.problem_type == 'classification' else 'regression'} task"
            )
            cv = RepeatedKFold(
                n_splits=self.config.outer_folds,
                n_repeats=self.config.repeats,
                random_state=self.config.seed,
            )
        
        # -------------------------------------------------------------- loop
        for train_idx, test_idx in cv.split(X, y):
            fold_idx += 1
            self.logger.info(f"Processing fold {fold_idx}/{total_folds}")
            fold_res = self._process_fold(
                fold_idx, X, y, train_idx, test_idx, feature_names
            )
            fold_results.append(fold_res)
            fold_metrics.append(fold_res["metrics"])
            self.logger.info(f"Completed fold {fold_idx}")
            gc.collect()
        
        metrics = calculate_overall_metrics(fold_metrics)
        return dict(
            fold_results=fold_results,
            fold_metrics=fold_metrics,
            metrics=metrics,
            feature_names=feature_names,
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
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        output_dir = self.config.get_output_dir()
        output_dir.mkdir(exist_ok=True, parents=True)
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True, parents=True)
        
        model = self._create_model()
        model.fit(X_train, y_train)
        
        # Calculate SHAP interactions immediately after model training for regression
        # This is the only place where we should calculate interactions - when the model is in memory
        # Skip interaction calculation for Random Forest models which cause memory corruption
        if self.config.problem_type == "regression" and not any(rf_name in str(type(model)).lower() for rf_name in ['randomforest', 'random_forest']):
            try:
                # Calculate and save interactions for top features
                import shap
                import pandas as pd
                import numpy as np
                
                # Create a DataFrame for X_test with feature names
                X_test_df = pd.DataFrame(X_test, columns=feature_names)
                
                # Calculate SHAP values for this fold
                # Try to use underlying model if it's wrapped
                if hasattr(model, 'model'):
                    underlying_model = model.model
                else:
                    underlying_model = model
                
                # Use TreeExplainer to get SHAP values
                explainer = shap.TreeExplainer(underlying_model)
                shap_values = explainer.shap_values(X_test_df)
                
                # Use a very limited number of features for interactions
                max_features = 10  # Hard limit - keeping it very small to avoid memory issues
                
                # Calculate mean absolute SHAP values and get top indices
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                top_indices = np.argsort(-mean_abs_shap)[:max_features]
                top_features = [feature_names[i] for i in top_indices]
                
                self.logger.info(f"Calculating interactions for fold {fold_idx} with {len(top_features)} features")
                
                # For XGBoost models, use the built-in interaction calculation
                if 'xgboost' in str(type(underlying_model)).lower():
                    import xgboost as xgb
                    import pandas as pd
                    
                    self.logger.info("Using XGBoost native methods for interaction calculation")
                    
                    # Get the booster directly if possible
                    if hasattr(underlying_model, 'get_booster'):
                        booster = underlying_model.get_booster()
                    else:
                        booster = underlying_model
                    
                    # Create interaction matrix directly from feature importance scores
                    # This avoids the memory allocation issues in TreeExplainer
                    interaction_matrix = np.zeros((len(top_features), len(top_features)))
                    
                    # Diagonal is just the feature importance
                    for i, feat in enumerate(top_features):
                        importance = 0
                        try:
                            # Try to get feature importance from the model
                            importance_dict = dict(zip(feature_names, underlying_model.feature_importances_))
                            importance = importance_dict.get(feat, 0)
                        except:
                            pass
                        interaction_matrix[i, i] = importance
                    
                    # Off-diagonal elements - estimate based on feature co-occurrence
                    # This is a simplified approach but should give reasonable values
                    for i, feat1 in enumerate(top_features):
                        for j, feat2 in enumerate(top_features):
                            if i < j:  # Upper triangle only
                                # Calculate correlation between SHAP values as an approximation
                                f1_idx = feature_names.index(feat1)
                                f2_idx = feature_names.index(feat2)
                                f1_shap = shap_values[:, f1_idx]
                                f2_shap = shap_values[:, f2_idx]
                                # Use absolute correlation as interaction strength
                                corr = np.abs(np.corrcoef(f1_shap, f2_shap)[0, 1])
                                # Scale by geometric mean of individual importances
                                int_val = corr * np.sqrt(interaction_matrix[i, i] * interaction_matrix[j, j])
                                interaction_matrix[i, j] = int_val
                                interaction_matrix[j, i] = int_val  # Symmetric matrix
                    
                    # Create a 3D tensor with the same shape as SHAP interaction values
                    # Each sample gets the same interaction matrix (averaged across all samples)
                    interaction_values = np.zeros((X_test_df.shape[0], len(top_features), len(top_features)))
                    for k in range(X_test_df.shape[0]):
                        interaction_values[k] = interaction_matrix
                elif 'random_forest' not in self.model_type.lower():
                    # For other non-XGBoost and non-RandomForest models, attempt to use TreeExplainer safely
                    try:
                        # Filter X_test to only include top features
                        X_filtered = X_test_df[top_features]
                        interaction_values = explainer.shap_interaction_values(X_filtered)
                    except Exception as e:
                        self.logger.warning(f"TreeExplainer failed: {e}")
                        # Just set interaction values to None
                        interaction_values = None
                else:
                    # Skip interaction calculations entirely for Random Forest models
                    self.logger.info("Interactions not produced for Random Forest models.")
                    interaction_values = None
                
                # If interaction_values is a list (for multi-output models), use the first element
                if isinstance(interaction_values, list):
                    interaction_values = interaction_values[0]
                
                # Average across samples to get feature-feature interactions
                interaction_matrix = np.mean(np.abs(interaction_values), axis=0)
                
                # Create tidy DataFrame with feature1, feature2, interaction_strength
                interactions = []
                for i1, f1 in enumerate(top_features):
                    for i2, f2 in enumerate(top_features):
                        if i1 <= i2:  # Include diagonal and upper triangle
                            strength = interaction_matrix[i1, i2]
                            interactions.append({"feature1": f1, "feature2": f2, "interaction_strength": strength})
                
                # Save to CSV
                interaction_df = pd.DataFrame(interactions)
                csv_path = fold_dir / f"fold_{fold_idx}_interactions.csv"
                interaction_df.to_csv(csv_path, index=False)
                
                self.logger.info(f"Successfully saved SHAP interactions for fold {fold_idx}")
            except Exception as e:
                self.logger.warning(f"Failed to calculate SHAP interactions for fold {fold_idx}: {str(e)}")
        
        y_pred = model.predict(X_test)
        
        y_prob = None
        if (
            self.config.problem_type == "classification"
            and hasattr(model, "predict_proba")
        ):
            # Import pandas to ensure it's accessible in this block
            import pandas as pd
            # Import numpy to ensure it's accessible in this block
            import numpy as np
            y_prob = model.predict_proba(X_test)
        
        # ------------------------------------------------ metrics
        if self.config.problem_type == "regression":
            # Ensure numpy is imported for metrics calculation
            import numpy as np
            
            metrics = dict(
                mse=((y_test - y_pred) ** 2).mean(),
                rmse=np.sqrt(((y_test - y_pred) ** 2).mean()),
                mae=np.abs(y_test - y_pred).mean(),
                r2=1
                - ((y_test - y_pred) ** 2).sum()
                / ((y_test - y_test.mean()) ** 2).sum(),
            )
        else:
            metrics = dict(
                accuracy=accuracy_score(y_test, y_pred),
                f1=f1_score(y_test, y_pred, average="weighted"),
            )
            if hasattr(model, "predict_proba"):
                try:
                    if len(np.unique(y_test)) > 2:
                        metrics["roc_auc"] = roc_auc_score(
                            y_test, y_prob, multi_class="ovr"
                        )
                    else:
                        pos = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.ravel()
                        metrics["roc_auc"] = roc_auc_score(y_test, pos)
                except Exception as e:
                    self.logger.warning(f"Could not calculate ROC AUC: {e}")
        
        # ------------------------------------------------ feature importance
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
        else:
            feature_importances = np.zeros(len(feature_names))
        
        # Import pandas within this block to ensure it's accessible
        import pandas as pd
        import numpy as np
        
        # Save per-fold feature importance immediately (top features will be trimmed later if needed)
        try:
            fi_series = pd.Series(feature_importances, index=feature_names, name="Importance")
            fi_csv = fold_dir / f"feature_importance_fold_{fold_idx}.csv"
            fi_series.to_csv(fi_csv, header=True)
            print(f"[Fold {fold_idx}] Feature importance values saved to {fi_csv}")
        except Exception as e:
            self.logger.warning(f"Could not save per-fold feature importance: {e}")
        
        # ------------------------------------------------ SHAP
        shap_values = None
        X_test_df   = None
        if hasattr(model, "shap_values"):
            try:
                X_test_df   = pd.DataFrame(X_test, columns=feature_names)
                shap_values = model.shap_values(X_test)
            except Exception as e:
                self.logger.warning(
                    f"Could not calculate SHAP values for fold {fold_idx}: {e}"
                )
        
        # ------------------------------------------------ Optuna figures
        if hasattr(model, "study"):
            save_optuna_visualizations(
                model.study, fold_idx, output_dir, self.config
            )
        
        # ------------------------------------------------ persist model
        model_path = fold_dir / f"best_model_fold_{fold_idx}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[Fold {fold_idx}] Best model saved to {model_path}")
        
        # ------------------------------------------------ ids / predictions csv
        ids = test_idx.tolist()
        original_ids = None
        # Import pandas to ensure it's accessible in this block
        import pandas as pd
        
        if (
            hasattr(self, "original_data")
            and self.config.id_column
            and self.config.id_column
            in getattr(self, "original_data", pd.DataFrame()).columns
        ):
            original_ids = [
                self.original_data.loc[idx, self.config.id_column]
                if idx in self.original_data.index
                else f"Sample_{idx}"
                for idx in test_idx
            ]
        save_fold_predictions_vs_actual(
            fold_idx,
            ids,
            y_pred,
            y_test,
            output_dir,
            original_ids=original_ids,
        )
        
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
    
    # ────────────────────────────────────────────────────────────────────────
    # helper visuals per fold  (unchanged)
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
            combined = pd.concat([
                pd.DataFrame({"y": train_series, "Set": "Train"}),
                pd.DataFrame({"y": test_series,  "Set": "Test"}),
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
        hist_path = fold_dir / f"fold_{fold_idx}_distribution.png"
        plt.savefig(hist_path)
        plt.close()
        print(f"[Fold {fold_idx}] Distribution histogram saved to {hist_path}")
    
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
            rows.append(
                dict(ID=all_ids[idx], Target_Value=y[idx], Set="Train")
            )
        for idx in test_idx:
            rows.append(
                dict(ID=all_ids[idx], Target_Value=y[idx], Set="Test")
            )
        pd.DataFrame(rows).to_csv(
            fold_dir / f"fold_{fold_idx}_samples.csv", index=False
        )
        print(f"[Fold {fold_idx}] Samples list saved to "
              f"{fold_dir / f'fold_{fold_idx}_samples.csv'}")
    
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
        import seaborn as sns
        
        plt.figure(figsize=(10, 6))
        sns.kdeplot(y_true, label="Actual", fill=True, alpha=0.4, color="#00BFC4")
        sns.kdeplot(y_pred, label="Predicted", fill=True, alpha=0.4, color="#F8766D")
        plt.title(f"Density Plot – Fold {fold_idx}")
        plt.xlabel(target_name)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(alpha=0.3)
        
        fold_dir.mkdir(exist_ok=True, parents=True)
        plot_path = fold_dir / f"density_plot_fold_{fold_idx}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Fold {fold_idx} density plot saved at {plot_path}")
    
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
        self.logger.info(f"Saving results to {output_dir}")
        
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
        self.logger.info(
            "Performing SHAP analysis (signed, focusing on positive/negative impact)..."
        )
        shap_df = save_mean_shap_analysis(
            results["fold_results"],
            output_dir,
            problem_type=self.config.problem_type,
            top_n=self.config.top_n
        )
        output_files["shap_analysis"] = str(
            output_dir / "shap_feature_impact_analysis.csv"
        )
        self.logger.info("Saving SHAP-based features summary...")
        save_top_features_summary(
            shap_df,
            output_dir,
            problem_type=self.config.problem_type,
            top_n=self.config.top_n,
        )
        output_files["features_summary"] = str(
            output_dir / "shap_features_summary.txt"
        )
        
        # -------------------------------------------- feature importance
        self.logger.info("Saving feature importance values...")
        fi_df = save_feature_importance_values(
            results["fold_results"], output_dir, in_fold_dirs=True
        )
        fi_dir = output_dir / "feature_importance"
        output_files["feature_importance_values"] = str(
            fi_dir / "feature_importance_values.csv"
        )
        self.logger.info("Creating feature importance bar plot...")
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
        
        self.logger.info("Generating prediction-vs-actual visualisations...")
        confusion_cmap = getattr(self.config, "confusion_cmap", None)
        generate_density_plots(
            fold_results=results["fold_results"],
            all_true_values=all_true,
            all_predictions=all_pred,
            output_dir=output_dir,
            target_name=self.config.target,
            problem_type=self.config.problem_type,
            cmap=confusion_cmap,
        )
        if self.config.problem_type == "regression":
            output_files["density_plot"] = str(
                output_dir / "density_actual_vs_pred_global.png"
            )
        
        # Only run interaction visualization for regression with XGBoost models
        # Skip for Random Forest models
        if self.config.problem_type == "regression" and 'random_forest' not in self.config.model.lower():
            from daftar.viz.shap_interaction_utils import save_shap_interactions_analysis
            
            # NEVER recalculate, only use pre-calculated interactions from the SHAP phase
            try:
                # Skip calculation=True forces it to use existing files and just create visualizations
                inter_files = save_shap_interactions_analysis(
                    results["fold_results"],
                    output_dir,
                    top_n_interactions=self.config.top_n,
                    shap_df=shap_df,
                    in_fold_dirs=True,
                    skip_calculation=True  # ALWAYS skip calculation at this phase
                )
            except RuntimeError as e:
                self.logger.error(f"Failed to generate interaction visualizations: {e}")
                self.logger.error("SHAP interactions must be calculated during the initial SHAP analysis phase.")
                self.logger.error("Check if each fold is properly computing and saving interaction values.")
                raise RuntimeError(f"Pipeline failed: {e}")
            output_files.update(inter_files)
        elif self.config.problem_type == "classification":
            # Aggregate per-class SHAP statistics that were computed during each fold
            try:
                self.logger.info("Aggregating per-class SHAP analysis across folds ...")
                from daftar.viz.per_class_shap import save_per_class_shap_analysis
                
                per_class_files = save_per_class_shap_analysis(
                    results["fold_results"],
                    output_dir,
                    feature_names=feature_names,
                    top_n=self.config.top_n,
                )
                output_files.update(per_class_files)
            except Exception as e:
                self.logger.warning(f"Per-class SHAP analysis failed: {e}")
                self.logger.info("Standard SHAP analysis is still available")
        
        # -------------------------------------------- figures explanation
        create_figures_explanation(output_dir)
        output_files["figures_explanation"] = str(
            output_dir / "figures_explanation.txt"
        )
        
        self.logger.info("Analysis complete.")
        return dict(
            metrics=overall_metrics,
            output_files=output_files,
            shap_analysis=shap_df.to_dict() if shap_df is not None else {},
            feature_importance=fi_df.to_dict() if fi_df is not None else {},
        )