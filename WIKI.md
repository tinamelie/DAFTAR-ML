# DAFTAR-ML Wiki

## **D**ata **A**gnostic **F**eature-**T**arget **A**nalysis & **R**anking **M**achine **L**earning Pipeline


DAFTAR-ML identifies which features in your dataset have the strongest relationship to your target variable using machine learning and SHAP (SHapley Additive exPlanations) analysis. Unlike standard ML pipelines that focus on prediction accuracy, DAFTAR-ML focuses on feature ranking and interpretation.

**Pipeline Features**: Data preprocessing, support for regression and classification, automated hyperparameter optimization, nested cross-validation, SHAP (SHapley Additive exPlanations) feature analysis, and publication-quality visualizations.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Input Data Format](#input-data-format)
3. [Workflow Overview](#workflow-overview)
4. [Detailed Workflow Steps](#detailed-workflow-steps)
   - [1. Data Preprocessing](#1-data-preprocessing)
   - [2. Cross-Validation Configuration Calculator](#2-cross-validation-configuration-calculator)
   - [3. Running DAFTAR-ML](#3-running-daftar-ml)
5. [Results and Output Explanation](#results-and-output-explanation)
6. [Understanding SHAP Analysis](#understanding-shap-analysis)
7. [Model-Specific Outputs](#model-and-problem-type-specific-outputs)
8. [Code Structure](#code-structure)
9. [Installation](#installation)
10. [Advanced Features](#advanced-features)

---

## Quick Start

Installation:
```bash
pip install git+https://github.com/tinamelie/DAFTAR-ML.git
```
Preprocess and run:

```bash
daftar-preprocess --input raw.csv --target Growth_rate --sample Species

daftar --input preprocessed.csv --target Growth_rate --sample Species --model xgb
```

## Input Data Format

DAFTAR-ML expects a comma-separated file (csv) with:

- **Sample identifier** (--sample)
- **Target variable** (--target; continuous or binary)
- One or more **feature** columns (numeric)
 
### Input Data Example:

&nbsp;&nbsp;Sample&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Target column&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Feature columns
| Species | **Growth_rate** | Gene_cluster1 | Gene_cluster2 | Gene_cluster3 |
|----------|:---------------:|:-------------:|:-------------:|:-------------:|
| Species1 |     **0.34**    |       10      |       0       |       15      |
| Species2 |      **0**      |       8       |       1       |       6       |
| Species3 |     **0.01**    |       0       |       4       |       3       |

**Typical uses:** Discovering which genes affect a phenotype, which biomarkers predict disease, or which variables drive any outcome of interest.

### Detailed Input Data Requirements

DAFTAR-ML expects a comma-separated (.csv) matrix with the following columns:

- **Identifier**: Unique sample identifier (species, strain, isolate, etc.). Specify with `--sample COLUMN`.
- **Target**: Response variable to predict (e.g., growth rate, yield, etc.). May be continuous (regression) or categorical (binary or multiclass classification). Specify with `--target COLUMN`.     
- **Features**: Predictor columns (e.g., orthologous gene counts, CAI values, expression profiles).

 **Target Variable Requirements:**  
 - Each dataset must have exactly **one target column**.
 - For classification problems, target values can be **binary** (e.g., 0/1, True/False, Yes/No) or **multiclass** (e.g., categories like 'low', 'medium', 'high').

Examples provided in [`test_data/`](test_data):

* `Test_data_binary.csv` – binary classification example (0/1 targets)
* `Test_data_continuous.csv` – regression example with continuous targets

---

## Workflow Overview

### Workflow Steps 
A typical DAFTAR-ML workflow consists of three steps:
1. **Data Preprocessing**: Clean and prepare your data by selecting the most informative features
2. **Cross-Validation Calculator**: Determine the optimal CV configuration for your dataset size
3. **Model Training & Analysis**: Train models with nested CV and analyze features

### 1. Data Preprocessing (Recommended)
Filters features by mutual information, reducing dataset size for improved model performance

```bash
daftar-preprocess --input raw.csv --target Growth_Rate --sample Species
```

### 2. Cross-Validation Configuration (Optional)
Assists in visualizing and optimizing your CV splits before training.

```bash
daftar-cv --input preprocessed.csv --target Growth_Rate --sample Species 
```

### 3. Main Pipeline
Executes the pipeline
```bash
daftar --input preprocessed.csv --target Growth_Rate --sample Species --model xgb
```

### Example Results:
| Rank |    Feature    |  SHAP Score |
|------|:-------------:|:------:|
| 1    | Gene_cluster2 |  0.556 |
| 2    | Gene_cluster1 |  0.461 |
| 3    | Gene_cluster3 |  0.321 |

In these results, Gene_cluster2 has the highest SHAP score, indicating it contributes most to a higher growth rate.

### Interpreting Your Results

For most applications, **top_shap_bar_plot.png** provides the best balance between interpretability and statistical robustness, indicating which features are important and how they influence predictions.

---

## Detailed Workflow Steps

## 1. Data Preprocessing

Before running DAFTAR-ML, prepare your data using the preprocessing script to filter your data. This step is optional but recommended for better performance and more accurate results:

```bash
daftar-preprocess --input PATH --target COLUMN --sample COLUMN --output_dir PATH
```
The preprocessing module uses mutual information to select features with the strongest relationship to the target variable, without assuming linearity. It selects the top-k features with highest MI scores, reducing dimensionality while preserving predictive power. Results include a summary report of transformations and selected features. It produces the following files:

**Preprocessed Data:**
- `[filename]_MI500_[problemtype].csv`: The preprocessed dataset with filtered features and (optionally) transformed values.

**Feature Importance Rankings:**
- `[filename]_MI500_[problemtype]_feature_scores.csv`: A CSV file containing the mutual information (MI) scores for each feature.

**Summary Report:**
- `[filename]_MI500_[problemtype]_report.txt`: A text file containing a summary of the preprocessing steps and selected features.

### Required Parameters:
* `--input PATH`: Path to input CSV file
* `--target COLUMN`: Target column name to predict
* `--sample COLUMN`: Name of the identifier column (e.g., species identifiers, sample names)

### Optional Parameters:

#### Output Configuration:
* `--output_dir PATH`: Directory where output files will be saved. If not specified, files will be saved in the input file's directory.
* `--force`: Force overwrite if output file already exists.

#### Analysis Configuration:
* `--task_type {regression,classification}`: Problem type. Optional and not recommended - dataset type will be auto-detected.
* `--k INTEGER`: Number of top features to select based on mutual information scores. Higher values retain more features but may include less informative ones and slow processing. Lower values provide a more focused feature set (default: 500).

#### Data Transformations:
* `--trans_feat {log1p,standard,minmax}`: Transformation to apply to feature columns:
  - `log1p`: Natural log(x+1) transform, good for skewed data
  - `standard`: Standardize to mean=0, std=1 (z-scores)
  - `minmax`: Scale to range [0,1]
* `--trans_target {log1p,standard,minmax}`: Transformation for target values (regression only):
  - `log1p`: Natural log(x+1) transform, often used for right-skewed values
  - `standard`: Standardize to mean=0, std=1 (z-scores)
  - `minmax`: Scale to range [0,1]

#### Processing Options:
* `--jobs INTEGER`: Number of parallel processing jobs for feature selection. -1 uses all cores. (default: -1)
* `--keep_na`: Keep rows with missing values (NaN). By default, such rows are removed. Not recommended as some models are sensitive to missing data.
* `--keep_constant`: Keep features with zero variance. By default, constant features are removed. Not recommended as some models are sensitive to constant features.
* `--no_rename`: Disable automatic renaming of duplicate column names. By default, duplicates are renamed with numeric suffixes.
* `--keep_zero_mi`: Keep features with zero mutual information with target. By default, these are removed. Not recommended as they may negatively impact model performance.

#### Output Options:
* `--quiet`: Suppress detailed console output during processing.
* `--no_report`: Skip generating the detailed text report and feature importance CSV files (these are produced by default).

---

## 2. Cross-Validation Configuration Calculator

The CV calculator is a planning tool for visualizing and analyzing your cross-validation splits. It provides detailed visualizations of target value distributions across CV splits, statistical validation of fold quality, and comprehensive reporting to help you evaluate your CV strategy before running the main DAFTAR-ML pipeline.

Note: This does not produce any modified or processed data from your input. This is simply a tool to help you select your parameters. It can be skipped if you already have a configuration in mind, have balanced classes/distributions, or prefer the defaults.

### Nested Cross-Validation Approach

#### CV Calculator Usage

```bash
daftar-cv --input PATH --target COLUMN --sample COLUMN --outer INTEGER --inner INTEGER --repeats INTEGER --output_dir PATH
```

### CV Calculator Parameters

#### Required Parameters:
* `--input PATH`: Path to the preprocessed CSV file containing your feature data
* `--target COLUMN`: Name of the target column to predict
* `--sample COLUMN`: Name of the identifier column (e.g., species, sample names)

### Optional Parameters:

#### Cross-Validation Configuration:
* `--outer INTEGER`: Number of partitions for the outer CV loop, which evaluates model performance (default: 5)
* `--inner INTEGER`: Number of partitions for the inner CV loop, which optimizes hyperparameters (default: 3)
* `--repeats INTEGER`: Number of times to repeat the entire CV process, which reduces performance variance (default: 3)

**Note**: If you specify any of the CV parameters (outer, inner, repeats), you must specify all three.

* `--seed INTEGER`: Random seed for reproducibility. Using the same seed ensures identical fold splits

**Note**: Using the same seed value in both the CV calculator and main pipeline ensures identical fold distributions. This allows you to validate fold quality with the CV calculator and then apply the same validated folds in your main analysis.

#### Output Configuration:
* `--output_dir PATH`: Directory where output files and visualizations will be saved
* `--force`: Force overwrite if output files already exist in the results directory

### Generated Output Files:

#### CSV Exports:
* `CV_[target]_[task-type]_cv[outer]x[inner]x[repeats]_splits_basic.csv`: Sample assignments to train/test for each outer fold
* `CV_[target]_[task-type]_cv[outer]x[inner]x[repeats]_splits_granular.csv`: Detailed dataset showing all sample assignments across all folds and repeats
* `fold_[N]_samples.csv`: Per-fold CSV file listing all samples with their ID, target value, and assignment (Train/Test)

#### Visualizations:
* `CV_[target]_[task-type]_cv[outer]x[inner]x[repeats]_overall_distribution.png`: Histogram/density plot of the overall target distribution with automatically optimized bin sizes
* `CV_[target]_[task-type]_cv[outer]x[inner]x[repeats]_histograms.png`: Multi-panel visualization comparing train/test distributions for each fold with automatically optimized bin sizes
* `fold_[N]_distribution.png`: Individual fold histograms showing train/test distribution for each fold
* `svg/` subdirectory: Contains SVG versions of all plots for high-quality vector graphics

#### Reports:
* `CV_[target]_[task-type]_cv[outer]x[inner]x[repeats]_fold_report.txt`: Statistical assessment of fold quality with p-value tests
  - For classification tasks: Chi-square test of independence comparing class distributions
  - For regression tasks: Kolmogorov-Smirnov test comparing value distributions
  - p-value ≥ 0.05 indicates train/test sets have similar distributions (good fold quality)
  - p-value < 0.05 indicates statistically significant differences between train/test distributions (potential fold quality issue)
Note: Larger p-values are better for fold quality.

### Cross-Validation Guidelines

#### Important Notes:
- DAFTAR-ML's cross-validation implementation **always uses shuffling by default** to ensure more representative data distribution across splits.
- **Stratified sampling** is used by default for classification tasks but can be disabled with `--stratify false`.
- For regression tasks, standard (non-stratified) sampling is used by default but can be enabled with `--stratify true` if your target values benefit from balanced distribution.
- All fold indexing is **zero-based** (starting from 0). For example, with 5 outer folds, they are numbered 0 through 4 in all output files and visualizations.

#### Rules of Thumb:

1. Smaller datasets benefit from **more repeats** to reduce variance in performance estimates
2. Larger datasets can use **more folds** for more reliable model evaluation
3. Balance computational cost and statistical reliability based on your resources
4. For high-dimensional data (many features), ensure training sets contain enough samples
5. For imbalanced classification tasks, use the default stratification to maintain class proportions
6. For regression with unusual distributions (bimodal, highly skewed), consider using `--stratify true`

---

## 3. Running DAFTAR-ML

After preprocessing your data and planning your cross-validation strategy, run the main DAFTAR-ML pipeline to train models, analyze feature importance, and generate visualizations:

```bash
daftar --input PATH --target COLUMN --sample COLUMN --model {xgb,rf} --output_dir PATH
```

### DAFTAR-ML Pipeline

1. **Nested Cross-Validation**: Implements the CV structure you designed in step 2
2. **Hyperparameter Optimization**: Uses Optuna to efficiently tune model parameters
3. **Model Training**: Trains optimized models for each fold
4. **Performance Evaluation**: Calculates metrics across all CV folds
5. **SHAP Value Analysis**: Generates SHAP-based feature rankings
6. **Visualization**: Produces figures and tables

### DAFTAR-ML Parameters

#### Required Parameters:
* `--input PATH`: Path to the preprocessed CSV file containing features and target variable
* `--target COLUMN`: Name of the target column to predict in the input file
* `--sample COLUMN`: Name of the identifier column (e.g., species, sample names)
* `--model {xgb,rf}`: Machine learning algorithm to use (xgb=XGBoost, rf=Random Forest)

#### Optional Parameters:

##### Analysis Configuration:
* `--task_type {regression,classification}`: Problem type (regression or classification). Auto-detected if not specified
* `--metric {mse,rmse,mae,r2,accuracy,f1,roc_auc}`: Performance metric to optimize. For regression: 'mse', 'rmse', 'mae', 'r2'; for classification: 'accuracy', 'f1', 'roc_auc' (default: 'mse' for regression, 'accuracy' for classification)

##### Cross-Validation Configuration:
* `--outer INTEGER`: Number of partitions for the outer CV loop, which evaluates model performance (default: 5)
* `--inner INTEGER`: Number of partitions for the inner CV loop, which optimizes hyperparameters (default: 3)
* `--repeats INTEGER`: Number of times to repeat the entire CV process, which reduces performance variance (default: 3)

##### Optimization Configuration:
* `--patience INTEGER`: Number of trials to wait without improvement before stopping hyperparameter optimization (default: 50)
  - 50 is a good balance between exploration and computational efficiency
  - Small datasets (< 1000 samples): 30-50 trials is usually sufficient
  - Medium datasets (1000-10000 samples): 50-100 trials allows for thorough search
  - Large/complex datasets: 100+ can find better hyperparameters but with diminishing returns
* `--threshold FLOAT`: Minimum improvement to consider a new trial better than previous best. Defaults: 1e-6 (MSE), 1e-4 (RMSE/MAE), 1e-3 (R²/accuracy/f1/roc_auc)

##### Execution Configuration:
* `--cores INTEGER`: Number of CPU cores to use. Set to -1 to use all available cores (default: -1)
* `--seed INTEGER`: Random seed for reproducible results. Using the same seed ensures identical fold splits (default: 42)

##### Visualization Configuration:
* `--top_n INTEGER`: Number of top features to include in visualizations (default: 15)
* `--skip_interaction`: Skip SHAP interaction calculations for faster execution (regression only)

##### Output Configuration:
* `--output_dir PATH`: Directory where output files will be saved. If not specified, an auto-generated directory name will be created
* `--force`: Force overwrite if output files already exist
* `--verbose`: Enable detailed logging output

---

## Results and Output Explanation

Each run creates a folder in either the current directory or the directory specified by `--output_dir`

### Visualizations and File Formats

All visualizations in DAFTAR-ML are provided in two formats:

* **PNG format**: Standard bitmap images saved in their original locations
* **SVG format**: Vector graphics to use in graphical editors like Inkscape or Adobe Illustrator, stored in `svg/` subdirectories

---

## Understanding SHAP Analysis

DAFTAR-ML uses SHAP (SHapley Additive exPlanations) to provide interpretable feature importance analysis. SHAP values quantify how much each feature contributes to individual predictions.

### SHAP Value Terminology:

* **SHAP Value:** The actual signed value showing direction of influence (positive increases prediction, negative decreases prediction)
* **Magnitude:** The absolute SHAP value used for ranking feature importance. Features are typically ordered by magnitude in visualizations.

### Interpreting SHAP Visualizations

#### Beeswarm Plots
These plots show detailed impact of features on individual predictions:
* Each dot represents one sample in your dataset
* Features are ordered by SHAP magnitude (top to bottom)
* Horizontal position shows impact on prediction:
  - Right side: Increases prediction
  - Left side: Decreases prediction
* Color indicates the feature value:
  - Red (default): High feature values
  - Blue (default): Low feature values
* Linear relationships: Consistent color gradient from left to right
* Non-linear effects: Same colors appearing on both sides

#### Bar Plots
Features are ordered by magnitude (absolute SHAP value)
* Bar direction shows SHAP value (positive increases prediction, negative decreases prediction)
* Error bars show variation across cross-validation folds

---

## Model and Problem Type Specific Outputs

Depending on which model type (XGBoost/Random Forest) and problem type (regression/classification) you use, DAFTAR-ML produces different specialized visualizations and analyses:

| Problem Type | Model Type | Specialized Outputs |
|--------------|------------|---------------------|
| **Regression** | XGBoost/Random Forest | • Feature interaction analysis<br>• SHAP interaction network visualization<br>• Density plots |
| **Classification** | XGBoost/Random Forest | • Per-class feature importance analysis<br>• Class-specific SHAP plots<br>• Confusion matrices<br>• Multiclass comparison (for >2 classes) |

### Feature Interactions (Regression Only)
When using tree-based regression models (XGBoost and Random Forest regression), DAFTAR-ML performs an additional analysis of how features interact with each other using SHAP TreeExplainer:

**Note:** This analysis can be skipped using the `--skip_interaction` flag for faster execution.

**Technical Implementation:**
- Only regression models support SHAP interaction computation for simplicity
- Uses SHAP TreeExplainer directly for reliable interaction computation

**Output Files:**
* `interaction_network.png`: Network of top 20 strongest feature interactions
* `interaction_.png`: Heatmap showing the top 20 features by interaction strength
* `top_bottom_network.png`: Network showing interactions between the 10 most positive and 10 most negative SHAP-scored features
* `interaction_matrix.csv`: Full numerical interaction matrix for all computed interactions

### Per-Class Feature Analysis (Classification Only)
For classification problems (both XGBoost and Random Forest), DAFTAR-ML analyzes which features are most important for predicting each specific class:

* `all_classes_shap_stats.csv`: Consolidated statistics about features' importance to each class
* `class_X_shap_impact.png`: Individual bar plots showing feature importance for each class
* `multiclass_comparison.png`: Visual comparison of feature importance across classes

### Output Structure Overview

```
DAFTAR-ML_GrowthRate_random_forest_regression_cv5x3x3/
├── DAFTAR-ML_run.log                     # Combined console + file log
├── performance.txt                       # Summary metrics across folds
├── metrics_all_folds.csv                 # Detailed metrics for all folds
├── feature_importance/                   # Feature importance directory
│   ├── feature_importance_values.csv        # Consolidated feature importance (mean ± std)
│   ├── feature_importance_bar.png           # Bar visualization of top features
│   └── svg/                                 # SVG versions of plot files
│       └── feature_importance_bar.svg       # SVG version for high-quality vector graphics
├── shap_beeswarm_impact.png               # SHAP impact beeswarm plot
├── shap_bar_impact.png                    # Feature impact by absolute SHAP magnitude
├── shap_bar_pos_neg_impact.png            # Features with positive and negative impacts
├── svg/                                    # SVG versions of plot files
│   ├── shap_beeswarm_impact.svg           # SVG version of SHAP impact beeswarm plot
│   ├── shap_bar_impact.svg                # SVG version of feature impact plot
│   └── shap_bar_pos_neg_impact.svg        # SVG version of positive/negative impact plot
├── shap_feature_analysis.csv              # Complete feature metrics with all statistics
├── shap_top_features_summary.txt          # Comprehensive feature analysis and rankings
├── shap_values_all_folds.csv              # Combined SHAP values from all folds
├── predictions_vs_actual_overall.csv      # Combined predictions from all folds
├── density_plot_overall.png               # Regression density plot (regression only)
├── output_files_explanation.txt           # Detailed explanations of all output visualizations
├── config.json                            # Record of all settings used in the analysis
├── shap_feature_interactions/             # Feature interaction visualizations (regression only)
│   ├── interaction_network.png            # Network graph of feature interactions
│   ├── interaction_heatmap.png            # Heatmap of interaction strengths
│   ├── all_interactions.csv               # Matrix of interaction strengths
│   └── interaction_summary.txt            # Summary of interaction analysis
├── per_class_shap/                        # Per-class feature analysis (classification only)
│   ├── all_classes_shap_stats.csv         # Features important for each class
│   ├── [class_name]_shap_impact.png       # Impact plots for each class
│   └── multiclass_comparison.png          # Comparison across classes (if >1 class)
└── fold_1/ … fold_N/                      # Individual fold directories
    ├── best_model_fold_N.pkl             # Trained model for this fold
    ├── test_indices_fold_N.csv           # Sample indices used in test set
    ├── predictions_vs_actual_fold_N.csv  # Test set predictions for this fold
    ├── test_train_splits_fold_N.csv      # List of samples with Train/Test sets
    ├── test_train_splits_fold_N.png      # Train/test target distribution histogram
    ├── density_plot_fold_N.png           # Fold-specific density plot (regression)
    ├── confusion_matrix_fold_N.png       # Confusion matrix (classification)
    ├── shap_values_fold_N.csv            # SHAP values for this fold
    ├── shap_interactions_fold_N.csv      # SHAP interaction values (regression)
    ├── feature_importance_fold_N.csv     # Feature importance rankings
    ├── metrics_fold_N.csv                # Fold performance metrics
    ├── optuna_trials_fold_N.csv          # All hyperparameter combinations tested
    ├── hyperparam_tuning_fold_N.txt      # Hyperparameter tuning summary
    └── optuna_plots/                      # Nested directory for optuna visualizations
        ├── optuna_summary_fold_N.txt      # Optuna optimization summary
        ├── optuna_history_fold_N.html     # Optimization history plot
        ├── optuna_parallel_fold_N.html    # Parallel coordinates plot
        └── optuna_slice_fold_N.html       # Slice plot for parameter analysis
```

---

## Code Structure

```
daftar/                      # Main library package
├── core/                       # Core pipeline components
│   ├── pipeline.py             # Main pipeline
│   ├── data_processing.py      # Data loading and preparation
│   ├── evaluation.py           # Model evaluation logic
│   ├── config.py               # Configuration management
│   ├── callbacks.py            # Hyperparameter optimization callbacks
│   └── logging_utils.py        # Logging utilities
├── models/                     # Model implementations
│   ├── base.py                 # Base model classes
│   ├── hyperparams.py          # Hyperparameter management
│   ├── xgboost_regression.py   # XGBoost regression implementation
│   ├── rf_regression.py        # Random Forest regression implementation
│   ├── xgboost_classification.py # XGBoost classification implementation
│   ├── rf_classification.py    # Random Forest classification implementation
│   └── hyperparams.yaml        # Hyperparameter search spaces
├── viz/                        # Visualization modules
│   ├── common.py               # Common visualization utilities
│   ├── predictions.py          # Prediction visualization
│   ├── feature_importance.py   # Feature importance plots
│   ├── shap.py                 # SHAP visualizations 
│   ├── shap_interaction_utils.py # SHAP interaction utilities
│   ├── per_class_shap.py       # Per-class SHAP analysis
│   ├── core_plots.py           # Core plotting functions
│   ├── metrics.py              # Metrics visualization
│   ├── optuna.py               # Hyperparameter optimization plots
│   ├── colors.py               # Color utility functions
│   └── colors.yaml             # Centralized color definitions
├── utils/                      # General utilities
│   ├── validation.py           # Data validation
│   ├── file_utils.py           # File handling 
│   └── warnings.py             # Warning utilities
└── cli.py                      # Command-line interface

# Root level files
setup.py                        # Package installation 
requirements.txt                # Package dependencies
LICENSE                         # License information
README.md                       # This documentation
```

---

## Installation

### Basic Installation

You can install DAFTAR-ML directly from GitHub:

```bash
pip install git+https://github.com/tinamelie/DAFTAR-ML.git
```

### Upgrade to Latest Version

To upgrade an existing installation to the latest version:

```bash
pip install --upgrade git+https://github.com/tinamelie/DAFTAR-ML.git
```

If you encounter conflicts during upgrade, you can try a clean reinstall:

```bash
pip uninstall -y daftar-ml
pip install git+https://github.com/tinamelie/DAFTAR-ML.git
```

---

## Advanced Features

### YAML Configuration (Optional)

Instead of passing a long list of CLI flags you can store them in a YAML file:

```yaml
input: PATH
sample: COLUMN
target: COLUMN
model: xgb
outer: 5
inner: 5
repeats: 3
cores: 8
seed: 42
metric: rmse
```

Run DAFTAR-ML with:

```bash
daftar --config config.yml
```

Anything specified on the CLI will override YAML values.

### Color Visualization Tool

DAFTAR-ML includes a utility to display all the color palettes used in its visualizations. Running the colors tool will show you the current color configuration and example plots. 

```bash
daftar-colors --output_dir PATH
```

Color management is consolidated into:
- `daftar/viz/colors.py`: Contains all color definitions, utilities, and visualization tools
- `daftar/viz/colors.yaml`: Centralized color configuration file

To customize colors, modify the `colors.yaml` configuration file. 

---

## Complete Usage Examples

### 1. Preprocess Raw Data (Optional)
```bash
daftar-preprocess --input input.csv --target TARGET_COLUMN --sample SAMPLE_COLUMN
```

### 2. Visualize CV Splits (Optional)
```bash
daftar-cv --input preprocessed_input.csv --target TARGET_COLUMN --sample SAMPLE_COLUMN
```

### 3. Run the Main Pipeline
```bash
daftar --input preprocessed_input.csv --target TARGET_COLUMN --sample SAMPLE_COLUMN --model {xgb,rf}
```

---

## Citing DAFTAR-ML

If you use DAFTAR-ML in academic work, please cite:

```
@software{daftar2025,
  author  = {Melie, Tina},
  title   = {DAFTAR-ML},
  year    = {2025},
  url     = {https://github.com/tinamelie/DAFTAR-ML},
  version = {v0.2.4}
}
```

---

For questions, feature requests or bug reports please open an issue on GitHub. 