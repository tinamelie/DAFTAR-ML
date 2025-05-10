"""Visualization components for DAFTAR-ML."""

# Expose key visualization functions
from .shap import save_mean_shap_analysis
from .feature_importance import plot_feature_importance_bar, save_feature_importance_values
from .predictions import generate_density_plots, save_fold_predictions_vs_actual, save_top_features_summary
from .optuna import save_optuna_visualizations

# Import color module
from .color_definitions import get_color_palette, get_train_test_colors

# Expose problem-specific visualizations
from .regression import generate_density_plots as regression_density_plots, create_residual_plot
from .classification import generate_confusion_matrix, create_roc_curve
