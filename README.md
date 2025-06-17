## **D**ata **A**gnostic **F**eature-**T**arget **A**nalysis & **R**anking **M**achine **L**earning Pipeline
 
DAFTAR-ML is a specialized machine learning pipeline that identifies relevant **features** based on their relationship to a **target** variable. While typical workflows fit models that predict targets from features, DAFTAR-ML instead evaluates each feature's relationship to the target and produces a ranked list of the most impactful features.

It supports both regression and classification tasks. DAFTAR-ML expects a single CSV file with one target column and any number of feature columns.



### Use Cases:

- **Gene-Phenotype Relationships**: Discover which genes correlate with specific phenotypes (e.g., growth on a substrate)
- **Metabolite–Disease Biomarker Discovery**: Rank metabolites that distinguish healthy vs. diseased states or track treatment response.
- **Drug-Response Prediction**: Identify the gene, transcript, or compound features that explain drug sensitivity or resistance.
- **Non-biological examples here** 

### Functionality Highlights:

- Automates data preprocessing and feature selection via mutual information
- Trains models with nested cross-validation 
- Optimizes hyperparameters with Optuna
- Scores features using SHAP (SHapley Additive exPlanations) 
- Generates publication-quality visualizations

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
## Overview

DAFTAR‑ML expects a comma‑separated file (csv) with:

- Unique sample identifier (--sample)
- Continuous or categorical response (--target)
- Features for prediction
 
### **input.csv example:**

&nbsp;&nbsp;Sample&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Target column&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Feature columns
| Species | **Xylose_growth** | Gene_cluster1 | Gene_cluster2 | Gene_cluster3 |
|----------|:---------------:|:-------------:|:-------------:|:-------------:|
| Species1 |     **0.34**    |       10      |       0       |       15      |
| Species2 |      **0**      |       8       |       1       |       6       |
| Species3 |     **0.01**    |       0       |       4       |       3       |

In this gene–phenotype example, OrthoFinder gene clusters are our **features**, and the **target** column (Xylose_growth) records each species' growth rate in xylose. DAFTAR-ML will reveal which **features** most strongly drive that target.

### Results example:
| Rank |    Feature    |  SHAP Score |
|------|:-------------:|:------:|
| 1    | Gene_cluster2 |  0.556 |
| 2    | Gene_cluster1 |  0.461 |
| 3    | Gene_cluster3 |  0.321 |

In these results, Gene_cluster2 has the highest SHAP score, indicating it contributes most to increased growth on Xylose.

### Workflow steps 
A typical DAFTAR-ML workflow consists of three steps:
1. **Data Preprocessing**: Clean and prepare your data by selecting the most informative features
2. **Cross-Validation Calculator**: Determine the optimal CV configuration for your dataset size
3. **Model Training & Analysis**: Train models with nested CV and analyze features

## Quick Start

### 1. Preprocess raw data (optional)
```bash
daftar-preprocess --input input.csv --target Xylose_growth --sample Species
```

### 2. Visualize CV splits (optional)
```bash
daftar-cv --input preprocessed_input.csv --target Xylose_growth --sample Species
```

### 3. Run the main pipeline
```
daftar --input preprocessed_input.csv --target Xylose_growth --sample Species --model {xgb,rf} --seed 893200
```

### Interpreting your results

For most applications, **top_shap_bar_plot.png** provides the best balance between interpretability and statistical robustness, indicating which features are important and how they influence predictions.

## Input Data

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

#### Required Parameters:
* `--input PATH`: Path to input CSV file
* `--target COLUMN`: Target column name to predict
* `--sample COLUMN`: Name of the identifier column (e.g., species identifiers, sample names)

#### Optional Parameters:

##### Output Configuration:
* `--output_dir PATH`: Directory where output files will be saved. If not specified, files will be saved in the input file's directory.
* `--force`: Force overwrite if output file already exists.

##### Analysis Configuration:
* `--task_type {regression,classification}`: Problem type. Optional and not recommended - dataset type will be auto-detected.
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

## 2. Cross-Validation Configuration Calculator

The CV calculator is a planning tool for visualizing and analyzing your cross-validation splits. It provides detailed visualizations of target value distributions across CV splits, statistical validation of fold quality, and comprehensive reporting to help you evaluate your CV strategy before running the main DAFTAR-ML pipeline.

Note: This does not produce any modified or processed data from your input. This is simply a tool to help you select your parameters. It can be skipped if you already have a configuration in mind, have balanced classes/distributions, or prefer the defaults.

### Nested Cross-Validation Approach

##### CV Calculator Usage

```bash
daftar-cv --input PATH --target COLUMN --sample COLUMN --outer INTEGER --inner INTEGER --repeats INTEGER --output_dir PATH
```

#### CV Calculator Parameters

##### Required Parameters:
* `--input PATH`: Path to the preprocessed CSV file containing your feature data
* `--target COLUMN`: Name of the target column to predict
* `--sample COLUMN`: Name of the identifier column (e.g., species, sample names)

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
* `CV_[target]_[task-type]_cv[outer]x[inner]x[repeats]_overall_distribution.png`: Histogram/density plot of the overall target distribution with automatically optimized bin sizes
* `CV_[target]_[task-type]_cv[outer]x[inner]x[repeats]_histograms.png`: Multi-panel visualization comparing train/test distributions for each fold with automatically optimized bin sizes
* `fold_[N]_distribution.png`: Individual fold histograms showing train/test distribution for each fold
* `svg/` subdirectory: Contains SVG versions of all plots for high-quality vector graphics

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
- All fold indexing is **zero-based** (starting from 0). For example, with 5 outer folds, they are numbered 0 through 4 in all output files and visualizations.

##### Rules of thumb:

1. Smaller datasets benefit from **more repeats** to reduce variance in performance estimates
2. Larger datasets can use **more folds** for more reliable model evaluation
3. Balance computational cost and statistical reliability based on your resources
4. For high-dimensional data (many features), ensure training sets contain enough samples
5. For imbalanced classification tasks, use the default stratification to maintain class proportions
6. For regression with unusual distributions (bimodal, highly skewed), consider using `--stratify true`

## 3. Running DAFTAR-ML

After preprocessing your data and planning your cross-validation strategy, run the main DAFTAR-ML pipeline to train models, analyze feature importance, and generate visualizations:

```bash
daftar --input PATH --target COLUMN --sample COLUMN --model {xgb,rf} --output_dir PATH
```
#### DAFTAR-ML Pipeline

1. **Nested Cross-Validation**: Implements the CV structure you designed in step 2
2. **Hyperparameter Optimization**: Uses Optuna to efficiently tune model parameters
3. **Model Training**: Trains optimized models for each fold
4. **Performance Evaluation**: Calculates metrics across all CV folds
5. **SHAP Value Analysis**: Generates SHAP-based feature rankings
6. **Visualization**: Produces figures and tables

#### DAFTAR-ML Parameters

#### Required Parameters:
* `--input PATH`: Path to the preprocessed CSV file containing features and target variable
* `--target COLUMN`: Name of the target column to predict in the input file
* `--sample COLUMN`: Name of the identifier column (e.g., species, sample names)
* `--model {xgb,rf}`: Machine learning algorithm to use (xgb=XGBoost, rf=Random Forest)

#### Optional Parameters:

##### Analysis Configuration:
* `--task_type {regression,classification}`: Problem type (regression or classification). Auto-detected if not specified
* `--metric {mse,rmse,mae,r2,accuracy,f1,roc_auc}`: Performance metric to optimize. For regression: 'mse', 'rmse', 'mae', 'r2'; for classification: 'accuracy', 'f1', 'roc_auc' (default: 'mse' for regression, 'accuracy' for classification)

##### Cross-validation Configuration:
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
* `--skip_interaction`: Skip SHAP interaction calculations for faster execution (regression only). When enabled, feature interaction analysis will be bypassed, reducing computation time at the cost of interaction insights.

##### Output Configuration:
* `--output_dir PATH`: Directory where output files will be saved. If not specified, an auto-generated directory name will be created
* `--force`: Force overwrite if output files already exist
* `--verbose`: Enable detailed logging output

## Results and Output Explanation

Each run creates a folder in either the current directory or the directory specified by `--output_dir`

### Visualizations and File Formats

All visualizations in DAFTAR-ML are provided in two formats:

* **PNG format**: Standard bitmap images saved in their original locations
* **SVG format**: Vector graphics to use in graphical editors like Inkscape or Adobe Illustrator, stored in `svg/` subdirectories

### Understanding SHAP Analysis

DAFTAR-ML uses SHAP (SHapley Additive exPlanations) to provide interpretable feature importance analysis. SHAP values quantify how much each feature contributes to individual predictions.

#### SHAP Value Terminology:

* **SHAP Value:** The actual signed value showing direction of influence (positive increases prediction, negative decreases prediction)
* **Magnitude:** The absolute SHAP value used for ranking feature importance. Features are typically ordered by magnitude in visualizations.

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

##### Bar Plots
Features are ordered by magnitude (absolute SHAP value)
* Bar direction shows SHAP value (positive increases prediction, negative decreases prediction)
* Colors indicate direction: red for positive, blue for negative
* Error bars show variation across cross-validation folds

### Model and Problem Type Specific Outputs

Depending on which model type (XGBoost/Random Forest) and problem type (regression/classification) you use, DAFTAR-ML produces different specialized visualizations and analyses:

| Problem Type | Model Type | Specialized Outputs |
|--------------|------------|---------------------|
| **Regression** | XGBoost/Random Forest | • Feature interaction analysis<br>• SHAP interaction network visualization |
| **Classification** | XGBoost/Random Forest | • Per-class feature importance analysis<br>• Class-specific SHAP plots<br>• Confusion matrices<br>• Multiclass comparison (for >2 classes) |

#### Feature Interactions (Regression Only)
When using tree-based regression models (XGBoost and Random Forest regression), DAFTAR-ML performs an additional analysis of how features interact with each other using SHAP TreeExplainer:

**Note:** This analysis can be skipped using the `--skip_interaction` flag for faster execution.

**Technical Implementation:**
- Only regression models support SHAP interaction computation for simplicity
- Uses SHAP TreeExplainer directly for reliable interaction computation
- Compatible with XGBoost and Random Forest regression models only

**Output Files:**
* `interaction_network.png`: Network of top 20 strongest feature interactions
* `interaction_heatmap.png`: Heatmap showing the top 20 features by interaction strength
* `top_bottom_network.png`: Network showing interactions between the 10 most positive and 10 most negative SHAP-scored features
* `interaction_matrix.csv`: Full numerical interaction matrix for all computed interactions
* `interaction_strength.csv`: Ranked list of all feature interactions by average magnitude

**Interaction Calculation Process:**
1. For each fold, SHAP interaction values are computed using TreeExplainer directly on the test data
2. Sample-level: interactions are averaged across test samples to create fold-specific matrices
3. Cross-fold: per-fold interaction matrices are aggregated preserving feature union (no penalty for absent features)
4. Visualization: strongest interactions are displayed in network and heatmap plots

**Note:** Classification models do not yet support feature interactions due to TreeExplainer shape complexity with multi-class outputs.

#### Per-Class Feature Analysis (Classification Only)
For classification problems (both XGBoost and Random Forest), DAFTAR-ML analyzes which features are most important for predicting each specific class:

* `all_classes_shap_stats.csv`: Consolidated statistics about features' importance to each class
* `class_X_shap_impact.png`: Individual bar plots showing feature importance for each class
* `multiclass_comparison.png`: Visual comparison of feature importance across classes


### Output Structure Overview

```
DAFTAR-ML_GrowthRate_random_forest_regression_cv5x3x3/
├── DAFTAR-ML_run.log                     # Combined console + file log
├── model_performance.txt                       # Summary metrics across folds
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
├── shap_features_analysis.csv             # Complete feature metrics with all statistics
├── shap_features_summary.txt              # Comprehensive feature analysis and rankings
├── shap_values_all_folds.csv              # Combined SHAP values from all folds
├── predictions_vs_actual_overall.csv      # Combined predictions from all folds
├── density_plot_overall.png                # Regression density plot (regression only)
├── figures_explanation.txt                # Detailed explanations of all output visualizations
├── config.json                            # Record of all settings used in the analysis
├── shap_feature_interactions/             # Feature interaction visualizations (regression only)
│   ├── interaction_network.png            # Network graph of feature interactions
│   ├── interaction_heatmap.png            # Heatmap of interaction strengths
│   ├── interaction_matrix.csv             # Matrix of interaction strengths
│   └── interaction_strength.csv           # Ranked interaction strengths
├── per_class_shap/                        # Per-class feature analysis (classification only)
│   ├── all_classes_shap_stats.csv         # Features important for each class
│   ├── class_X_shap_impact.png            # Impact plots for each class
│   └── multiclass_comparison.png          # Comparison across classes (if >1 class)
└── fold_1/ … fold_N/                      # Individual fold directories
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

## Advanced Features

### YAML Configuration (optional)

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
│   ├── xgb_base.py             # Base XGBoost model
│   ├── rf_base.py              # Base Random Forest model
│   ├── xgb_regression.py       # XGBoost regression implementation
│   ├── rf_regression.py        # Random Forest regression implementation
│   ├── xgb_classification.py   # XGBoost classification implementation
│   ├── rf_classification.py    # Random Forest classification implementation
│   └── hyperparams.yaml        # Hyperparameter search spaces
├── viz/                        # Visualization modules
│   ├── common.py               # Common visualization utilities
│   ├── predictions.py          # Prediction visualization
│   ├── feature_importance.py   # Feature importance plots
│   ├── shap.py                 # SHAP visualizations 
│   ├── shap_interaction_utils.py # SHAP interaction utilities
│   ├── colors.py               # Color utility functions
│   ├── colors.yaml             # Centralized color definitions
│   └── plotting_utils.py       # General plotting utilities
├── utils/                      # General utilities
│   ├── validation.py           # Data validation
│   ├── file_utils.py           # File handling 
│   ├── stats.py                # Statistical utilities
│   └── timer.py                # Timing utilities
└── cli.py                      # Command-line interface

# Root level files
setup.py                        # Package installation 
requirements.txt                # Package dependencies
LICENSE                         # License information
README.md                       # This documentation
```


## Citing DAFTAR-ML

If you use DAFTAR-ML in academic work, please cite:

```
@software{daftar2025,
  author  = {Melie, Tina},
  title   = {DAFTAR-ML},
  year    = {2025},
  url     = {https://github.com/tinamelie/DAFTAR-ML},
  version = {v0.1.5}
}
```

---

For questions, feature requests or bug reports please open an issue on GitHub.