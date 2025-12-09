# DAFTAR-ML

## **D**ata **A**gnostic **F**eature-**T**arget **A**nalysis & **R**anking **M**achine **L**earning Pipeline

DAFTAR-ML identifies which features in your dataset have the strongest relationship to your target variable using machine learning and SHAP analysis. Unlike standard ML pipelines that focus on prediction accuracy, DAFTAR-ML focuses on feature ranking and interpretation.

**Pipeline Features**: Data preprocessing, support for regression and classification, automated hyperparameter optimization, nested cross-validation, SHAP (SHapley Additive exPlanations) feature analysis, and publication-quality visualizations.

**For comprehensive documentation, tutorials, and advanced usage examples, see the [DAFTAR-ML Wiki](WIKI.md).**

## Quick Start (Minimal Example)
Installation:
```bash
pip install git+https://github.com/tinamelie/DAFTAR-ML.git

```
Preprocess and run: 
```bash
daftar-preprocess --input raw.csv --target Growth_rate --sample Species
```
```bash
daftar --input preprocessed.csv --target Growth_rate --sample Species --model xgb
```

## Input Data Format

DAFTAR-ML expects a comma-separated file (csv) with:

- **Sample identifier** (--sample)
- **Target variable** (--target; continuous or binary)
- One or more **feature** columns (numeric)
 
### Input data example:

&nbsp;&nbsp;Sample&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Target column&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Feature columns
| Species | **Growth_rate** | Gene_cluster1 | Gene_cluster2 | Gene_cluster3 |
|----------|:---------------:|:-------------:|:-------------:|:-------------:|
| Species1 |     **0.34**    |       10      |       0       |       15      |
| Species2 |      **0**      |       8       |       1       |       6       |
| Species3 |     **0.01**    |       0       |       4       |       3       |

**Typical uses:** Discovering which genes affect a phenotype, which biomarkers predict disease, or which variables drive any outcome of interest.

## Workflow Overview

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
Executes the pipeline (with XGBoost)
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

## Interpreting Your Results

A good overview of your results can be found in **top_shap_bar_plot.png**. This plot shows the most important features (by SHAP value) and their influence on predictions. 

![top_shap_bar_plot.png](https://github.com/tinamelie/DAFTAR-ML/blob/main/misc/top_shap_bar_plot.png "top_shap_bar_plot.png")

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

## Citing DAFTAR-ML

If you use DAFTAR-ML in academic work, please cite:

```
@software{daftar2025,
  author  = {Melie, Tina},
  title   = {DAFTAR-ML},
  year    = {2025},
  url     = {https://github.com/tinamelie/DAFTAR-ML},
  version = {v0.2.7}
}
```
## Copyright Notice

Data Agnostic Feature-Target Analysis & Ranking Machine Learning Pipeline (DAFTAR-ML) Copyright (c) 2025, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.

---

For questions, feature requests or bug reports please open an issue on GitHub.
