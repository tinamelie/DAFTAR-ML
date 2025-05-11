## **D**ata **A**gnostic **F**eature-**T**arget **A**nalysis & **R**anking **M**achine **L**earning Pipeline
 
DAFTAR-ML is a specialized machine learning pipeline that identifies relevant **features** based on their relationship to a **target** variable. It supports both regression and classification tasks. DAFTAR-ML expects a single CSV file with one target column and any number of feature columns.

### Use Cases

- **Gene-Phenotype Relationships**: Discover which genes correlate with specific phenotypes (e.g., growth on a substrate)
- **Metabolite–Disease Biomarker Discovery**: Rank metabolites that distinguish healthy vs. diseased states or track treatment response.
- **Drug-Response Prediction**: Identify the gene, transcript, or compound features that explain drug sensitivity or resistance.
- **Non-biological examples here** 

### Functionality Highlights:

- Automates data preprocessing and feature selection via mutual information
- Trains models with nested cross-validation 
- Optimizes hyperparameters with Optuna
- Scores features using SHAPley Additive exPlanations (SHAP)
- Generates publication-quality visualizations

## Installation

You can install DAFTAR-ML directly from GitHub:

```bash
pip install git+https://github.com/tinamelie/DAFTAR-ML.git
```

## Quick Start

DAFTAR‑ML expects a comma‑separated file (csv) with:

- id_column: Unique sample identifier
- target_column: Continuous or categorical response
- Features for prediction
 

 id_column&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_column&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Feature columns
| Species | **Growth_on_Galactose** | Gene_cluster1 | Gene_cluster2 | Gene_cluster3 |
|----------|:---------------:|:-------------:|:-------------:|:-------------:|
| Species1 |     **0.34**    |       10      |       0       |       15      |
| Species2 |      **0**      |       8       |       1       |       6       |
| Species3 |     **0.01**    |       0       |       4       |       3       |

In this gene-phenotype example, we use gene clusters OrthoFinder as our **features**. The **target** column, Growth_on_Galactose, is growth rates of species in galactose medium. The aim is to identify which gene clusters (**features**) are important to this **target**, i.e. which gene clusters are relevant to growth on galactose. 
### Results example (summary):
| Rank |    Feature    |  SHAP Score |
|------|:-------------:|:------:|
| 1    | Gene_cluster1 |  0.556 |
| 2    | Gene_cluster2 |  0.461 |
| 3    | Gene_cluster3 |  0.321 |

Running DAFTAR-ML on the dataset will score and rank these **features**.
(Actual output will provide SHAP summary plots, cross‑validation performance metrics, and visualizations.)

### Workflow steps 
A typical DAFTAR-ML workflow consists of three steps:
1. **Data Preprocessing**: Clean and prepare your data by selecting the most informative features
2. **Cross-Validation Calculator**: Determine the optimal CV configuration for your dataset size
3. **Model Training & Analysis**: Train models with nested CV and analyze features

### 1. Preprocess raw data (optional)
```bash
daftar-preprocess --input PATH --target COLUMN --id COLUMN
```

### 2. Visualize CV splits (optional)
```bash
daftar-cv --input PATH --target COLUMN --id COLUMN
```

### 3. Run the main pipeline
```
daftar --input PATH --target COLUMN --id COLUMN --model {xgb,rf}
```
## Input Data

DAFTAR-ML expects a comma-separated (.csv) matrix with the following columns:

- **ID**: Unique sample identifier (species, strain, isolate, etc.). Specify with `--id COLUMN`.
- **Target**: Response variable to predict (e.g., growth rate, yield, etc.). May be continuous (regression) or categorical (binary or multiclass classification). Specify with `--target COLUMN`.     
- **Features**: Predictor columns (e.g., orthologous gene counts, CAI values, expression profiles).


 **Target Variable Requirements:**  
 - Each dataset must have exactly **one target column**.
 - For classification problems, target values can be **binary** (e.g., 0/1, True/False, Yes/No) or **multiclass** (e.g., categories like 'low', 'medium', 'high').

Examples provided in [`test_data/`](test_data):

* `Test_data_binary.csv` – binary classification example (0/1 targets)
* `Test_data_continuous.csv` – regression example with continuous targets

### 1. Data Preprocessing

Before running DAFTAR-ML, prepare your data using the preprocessing script to filter your data. This step is optional but recommended for better performance and more accurate results:

```bash
daftar-preprocess --input PATH --target COLUMN --id COLUMN --output_dir PATH
```
The preprocessing module uses mutual information to select features with the strongest relationship to the target variable, without assuming linearity. It selects the top-k features with highest MI scores, reducing dimensionality while preserving predictive power. Results include a summary report of transformations and selected features. Produces the following files:

**Preprocessed Data:**
- `[filename]_MI500_classif.csv`: The preprocessed dataset with selected features and transformed values.

**Feature Importance Rankings:**
- `[filename]_MI500_classif_feature_scores.csv`: A CSV file containing the mutual information scores for each feature.

**Summary Report:**
- `[filename]_MI500_classif_report.txt`: A text file containing a summary of the preprocessing steps and selected features.

#### Required Parameters:
* `--input PATH`: Path to input CSV file
* `--target COLUMN`: Target column name to predict
* `--id COLUMN`: Name of the ID column (e.g., species identifiers, sample names)

#### Optional Parameters:

##### Output Configuration:
* `--output_dir PATH`: Directory where output files will be saved. If not specified, files will be saved in the input file's directory.
* `--force`: Force overwrite if output file already exists.

##### Analysis Configuration:
* `--task {regression,classification}`: Problem type. Optional and not recommended - dataset type will be auto-detected.
* `--k INTEGER`: Number of top features to select based on mutual information scores. Higher values retain more features but may include less informative ones and slow processing. Lower values provide a more focused feature set (default: 500).

##### Data Transformations:
* `--trans_feat {log1p,standard,minmax}`: Transformation to apply to feature columns:
  - `log1p`: Natural log(x+1) transform, good for skewed data
  - `standard`: Standardize to mean=0, std=1 (z-scores)
  - `minmax`: Scale to range [0,1]
* `--trans_target {log1p,standard,minmax}`: Transformation for target values (regression only):
  - `log1p`: Natural log(x+1) transform, often used for right-skewed values
  - `standard`: Standardize to mean=0, std=1 (z-scores)
  - `minmax`: Scale to range [0,1]

##### Processing Options:
* `--jobs INTEGER`: Number of parallel processing jobs for feature selection. -1 uses all cores. (default: -1)
* `--keep_na`: Keep rows with missing values (NaN). By default, such rows are removed. Not recommended as some models are sensitive to missing data.
* `--keep_constant`: Keep features with zero variance. By default, constant features are removed. Not recommended as some models are sensitive to constant features.
* `--no_rename`: Disable automatic renaming of duplicate column names. By default, duplicates are renamed with numeric suffixes.
* `--keep_zero_mi`: Keep features with zero mutual information with target. By default, these are removed. Not recommended as they may negatively impact model performance.

##### Output Options:
* `--quiet`: Suppress detailed console output during processing.
* `--no_report`: Skip generating the detailed text report and feature importance CSV files (these are produced by default).

### 2. Cross-Validation Configuration Calculator

The CV calculator is a planning tool for visualizing and analyzing your cross-validation splits. It provides detailed visualizations of target value distributions across CV splits, statistical validation of fold quality, and comprehensive reporting to help you evaluate your CV strategy before running the main DAFTAR-ML pipeline.

Note: This does not produce any modified or processed data from your input. This is simply a tool to help you select your parameters. It can be skipped if you already have a configuration in mind, have balanced classes/distributions, or prefer the defaults.

### Nested Cross-Validation Approach

##### CV Calculator Usage

```bash
daftar-cv --input PATH --target COLUMN --id COLUMN --outer INTEGER --inner INTEGER --repeats INTEGER --output_dir PATH
```

#### CV Calculator Parameters

#### Required Parameters:
* `--input PATH`: Path to the preprocessed CSV file containing your feature data
* `--target COLUMN`: Name of the target column to predict
* `--id COLUMN`: Name of the ID column (e.g., species identifiers, sample names)

#### Optional Parameters:

##### Cross-validation Configuration:
* `--outer INTEGER`: Number of partitions for the outer CV loop, which evaluates model performance (default: 5)
* `--inner INTEGER`: Number of partitions for the inner CV loop, which optimizes hyperparameters (default: 3)
* `--repeats INTEGER`: Number of times to repeat the entire CV process, which reduces performance variance (default: 3)

**Note**: If you specify any of the CV parameters (outer, inner, repeats), you must specify all three.

* `--seed INTEGER`: Random seed for reproducibility. Using the same seed ensures identical fold splits

**Note**: Using the same seed value in both the CV calculator and main pipeline ensures identical fold distributions. This allows you to validate fold quality with the CV calculator and then apply the same validated folds in your main analysis.

##### Output Configuration:
* `--output_dir PATH`: Directory where output files and visualizations will be saved
* `--force`: Force overwrite if output files already exist in the results directory

#### Generated Output Files:

##### CSV Exports:
* `CV_[target]_[task-type]_cv[outer]x[inner]x[repeats]_splits_basic.csv`: Sample assignments to train/test for each outer fold
* `CV_[target]_[task-type]_cv[outer]x[inner]x[repeats]_splits_granular.csv`: Detailed dataset showing all sample assignments across all folds and repeats
* `fold_[N]_samples.csv`: Per-fold CSV file listing all samples with their ID, target value, and assignment (Train/Test)

##### Visualizations:
* `CV_[target]_[task-type]_cv[outer]x[inner]x[repeats]_overall_distribution.png/pdf`: Histogram/density plot of the overall target distribution with automatically optimized bin sizes
* `CV_[target]_[task-type]_cv[outer]x[inner]x[repeats]_histograms.png/pdf`: Multi-panel visualization comparing train/test distributions for each fold with automatically optimized bin sizes
* `fold_[N]_distribution.png`: Individual fold histograms showing train/test distribution for each fold

##### Reports:
* `CV_[target]_[task-type]_cv[outer]x[inner]x[repeats]_fold_report.txt`: Statistical assessment of fold quality with p-value tests
  - For classification tasks: Chi-square test of independence comparing class distributions
  - For regression tasks: Kolmogorov-Smirnov test comparing value distributions
  - p-value ≥ 0.05 indicates train/test sets have similar distributions (good fold quality)
  - p-value < 0.05 indicates statistically significant differences between train/test distributions (potential fold quality issue)
Note: Larger p-values are better for fold quality.

#### Cross-Validation Guidelines

#### Important Notes:
- DAFTAR-ML's cross-validation implementation **always uses shuffling by default** to ensure more representative data distribution across splits.
- **Stratified sampling** is used by default for classification tasks but can be disabled with `--stratify false`.
- For regression tasks, standard (non-stratified) sampling is used by default but can be enabled with `--stratify true` if your target values benefit from balanced distribution.

##### Rules of thumb:

1. Smaller datasets benefit from **more repeats** to reduce variance in performance estimates
2. Larger datasets can use **more folds** for more reliable model evaluation
3. Balance computational cost and statistical reliability based on your resources
4. For high-dimensional data (many features), ensure training sets contain enough samples
5. For imbalanced classification tasks, use the default stratification to maintain class proportions
6. For regression with unusual distributions (bimodal, highly skewed), consider using `--stratify true`

### 3. Running DAFTAR-ML

After preprocessing your data and planning your cross-validation strategy, run the main DAFTAR-ML pipeline to train models, analyze feature importance, and generate visualizations:

```bash
daftar --input PATH --target COLUMN --id COLUMN --model {xgb,rf} --output_dir PATH
```
#### DAFTAR-ML Pipeline

1. **Nested Cross-Validation**: Implements the CV structure you designed in step 2
2. **Hyperparameter Optimization**: Uses Optuna to efficiently tune model parameters
3. **Model Training**: Trains optimized models for each fold
4. **Performance Evaluation**: Calculates metrics across all CV folds
5. **Feature Importance Analysis**: Generates SHAP-based feature rankings
6. **Visualization**: Produces figures and tables

#### DAFTAR-ML Parameters

#### Required Parameters:
* `--input PATH`: Path to the preprocessed CSV file containing features and target variable
* `--target COLUMN`: Name of the target column to predict in the input file
* `--id COLUMN`: Name of the ID column (e.g., species identifiers, sample names)
* `--model {xgb,rf}`: Machine learning algorithm to use (xgb=XGBoost, rf=Random Forest)

#### Optional Parameters:

##### Analysis Configuration:
* `--task {regression,classification}`: Problem type (regression or classification). Auto-detected if not specified
* `--metric NAME`: Performance metric to optimize. For regression: 'mse', 'rmse', 'mae', 'r2'; for classification: 'accuracy', 'f1', 'roc_auc' (default: 'mse' for regression, 'accuracy' for classification)

##### Cross-validation Configuration:
* `--outer INTEGER`: Number of partitions for the outer CV loop, which evaluates model performance (default: 5)
* `--inner INTEGER`: Number of partitions for the inner CV loop, which optimizes hyperparameters (default: 3)
* `--repeats INTEGER`: Number of times to repeat the entire CV process, which reduces performance variance (default: 3)

##### Optimization Configuration:
* `--patience INTEGER`: Number of trials to wait without improvement before stopping hyperparameter optimization (default: 50)
* `--threshold FLOAT`: Minimum improvement to consider a new trial better than previous best. Applies differently based on metric direction: higher metrics (r2, accuracy) must exceed by this factor; lower metrics (mse, rmse) must decrease by this factor. Defaults: 1e-6 (MSE), 1e-4 (RMSE/MAE), 1e-3 (R²/accuracy/f1/roc_auc)

##### Execution Configuration:
* `--cores INTEGER`: Number of CPU cores to use. Set to -1 to use all available cores (default: -1)
* `--seed INTEGER`: Random seed for reproducible results. Using the same seed ensures identical fold splits (default: 42)

##### Output Configuration:
* `--output_dir PATH`: Directory where output files will be saved. If not specified, an auto-generated directory name will be created
* `--force`: Force overwrite if output files already exist

## Results and Output Explanation

Each run creates a folder in either the current directory or the directory specified by `--output_dir`


### Output Structure Overview

```
DAFTAR-ML_GrowthRate_random_forest_regression_cv5x3x3/
├── DAFTAR-ML_run.log                     # Combined console + file log
├── metrics_overall.csv                   # Mean scores across folds
├── feature_importance/                   # Feature importance directory
│   ├── feature_importance_values_fold.csv  # Fold-level feature importance
│   ├── feature_importance_values_sample.csv # Sample-level feature importance 
│   ├── feature_importance_bar_fold.png    # Fold-level importance visualization
│   └── feature_importance_bar_sample.png  # Sample-level importance visualization
├── shap_*.png                            # SHAP summary visualizations
├── shap_feature_metrics.csv              # Feature statistics with both calculation methods
├── shap_features_summary.txt             # Comprehensive feature analysis and rankings
├── shap_values_all_folds.csv             # Combined SHAP values from all folds
├── predictions_vs_actual_overall.csv     # Combined predictions from all folds
├── density_actual_vs_pred_global.png     # Regression density plot
├── figures_explanation.txt               # Detailed explanations of all output visualizations
├── config.json                           # Record of all settings used in the analysis
└── fold_1/ … fold_N/                     # Individual fold directories
    ├── best_model_fold_N.pkl             # Trained model for this fold
    ├── test_indices_fold_N.csv           # Sample indices used in test set
    ├── predictions_vs_actual_fold_N.csv  # Test set predictions for this fold
    ├── fold_N_samples.csv                # List of samples with Train/Test sets
    ├── fold_N_distribution.png           # Train/test target distribution histogram
    ├── confusion_matrix_fold_N.png       # Confusion matrix (classification)
    ├── shap_values_fold_N.csv            # SHAP values for this fold
    ├── feature_importance_fold_N.csv     # Feature importance rankings
    ├── optuna_trials_fold_N.csv          # All hyperparameter combinations tested
    ├── optuna_importance_fold_N.png      # Parameter importance visualization
    └── optuna_parallel_coordinate_fold_N.png # Parallel coordinates plot
```

### Understanding SHAP Analysis

DAFTAR-ML uses SHAP (SHapley Additive exPlanations) to provide interpretable feature importance analysis. SHAP values quantify how much each feature contributes to individual predictions.

#### Two Types of Feature Importance Results

1. **Sample-level calculations**: Based on raw SHAP values across all samples combined
   * Advantages: Captures full feature impact across the entire dataset
   * Limitations: May be influenced by outliers or specific data contexts

2. **Fold-level calculations**: First calculates mean SHAP values within each fold, then averages across folds
   * Advantages: More robust to outliers and better identifies consistently important features
   * Limitations: May undervalue features that are important in specific contexts only

#### Interpreting SHAP Visualizations

##### Beeswarm Plots
These plots show detailed impact of features on individual predictions:
* Each dot represents one sample in your dataset
* Features are ordered by importance (top to bottom)
* Horizontal position shows impact on prediction:
  - Right side (red): Increases prediction
  - Left side (blue): Decreases prediction
* Color indicates the feature value:
  - Red: High feature values
  - Blue: Low feature values

Patterns to look for:
* Linear relationships: Consistent color gradient from left to right
* Non-linear effects: Same colors appearing on both sides
* Interactions: Unexpected patterns in dot distribution

##### Bar Plots
The summary bar plots show average magnitude of feature impacts:
* For classification: Red/blue bars show direction of impact on different classes
* For regression: Direction shows whether feature increases/decreases predictions

##### Correlation Plots (Regression Only)
For regression problems, correlation plots show relationship between feature SHAP values and the target:
* Red bars (positive correlation): Higher feature values → higher predictions
* Blue bars (negative correlation): Higher feature values → lower predictions

#### Comparing Different Rankings

These plots help identify patterns such as:
* Linear relationships: Consistent color gradient from left to right
* Non-linear effects: Same colors appearing on both sides of the zero line
* Interactions: Unexpected patterns in the distribution of dots

### Performance Evaluation Metrics

DAFTAR-ML supports various metrics for evaluating model performance:

- **Regression metrics:** mse, rmse, mae, r2  
  _Default threshold values_: 1e-6 (MSE), 1e-4 (RMSE/MAE), 1e-3 (R²)
- **Classification metrics:** accuracy, f1, roc_auc  
  _Default threshold value_: 1e-3

These thresholds determine when to stop hyperparameter optimization. You can customize them with the `--threshold` parameter.

## Advanced Features

### YAML Configuration (optional)

Instead of passing a long list of CLI flags you can store them in a YAML file:

```yaml
input: PATH
id: COLUMN
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

Colors are managed in `daftar/viz/color_definitions.py` - make any changes to the palette here. 


## Code Structure

```
daftar/                   # Main library package
├── core/                    # Core pipeline components
│   ├── pipeline.py          # Main pipeline
│   ├── data_processing.py   # Data loading and preparation
│   ├── evaluation.py        # Generic evaluation logic
│   ├── config.py            # Configuration handling
│   ├── callbacks.py         # Optimization callbacks
│   └── logging_utils.py     # Logging utilities
├── models/                  # All model implementations
│   ├── base.py              # Base model classes
│   ├── regression/          # Regression models
│   └── classification/      # Classification models
├── analysis/                # Analysis by problem type
│   ├── __init__.py
│   ├── regression.py        # Regression-specific analysis
│   └── classification.py    # Classification-specific analysis
├── viz/                     # All visualizations
│   ├── __init__.py
│   ├── common.py            # Common visualization utilities
│   ├── regression.py        # Regression visualizations
│   ├── classification.py    # Classification visualizations
│   ├── shap.py              # SHAP visualizations (simplified)
│   ├── feature_importance.py # Feature importance visualizations
│   ├── color_definitions.py  # Centralized color definitions
│   └── optuna.py            # Hyperparameter tuning visualizations
├── utils/                   # General utilities
│   ├── __init__.py
│   ├── validation.py        # Data validation
│   ├── file_utils.py        # File handling
│   └── warnings.py          # Warning management
├── tools/                   # Command-line tools
│   ├── __init__.py
│   ├── preprocess.py        # Preprocessing script
│   ├── cv_calculator.py     # CV calculation script
│   ├── colors.py            # Color visualization tool
│   └── run_daftar.py        # Main entry point script
└── cli.py                   # Command-line interface

# Root level files
setup.py                     # Package installation configuration
requirements.txt             # Package dependencies
example_config.yaml          # Example YAML configuration
test_data/                   # Example datasets for testing

```


## Citing DAFTAR-ML

If you use DAFTAR-ML in academic work, please cite:

```
@software{daftar2025,
  author  = {Melie, Tina},
  title   = {DAFTAR-ML},
  year    = {2025},
  url     = {https://github.com/tinamelie/DAFTAR-ML},
  version = {v0.1.2}
}
```

---

For questions, feature requests or bug reports please open an issue on GitHub.
