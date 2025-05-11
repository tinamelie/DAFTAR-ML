#!/usr/bin/env python
"""
Nested CV target-distribution visualiser.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold

from daftar.viz.color_definitions import get_color_palette, get_train_test_colors


# Visualization helpers
def calculate_optimal_bins(data: pd.Series, max_bins: int = 50) -> int:
    """Calculate optimal number of bins using Freedman-Diaconis rule.
    
    Args:
        data: Data series to calculate bins for
        max_bins: Maximum number of bins to return
        
    Returns:
        Optimal number of bins
    """
    n = len(data)
    if n <= 1:
        return 10
        
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    if iqr <= 0:
        return 10
        
    bin_width = 2 * iqr / (n ** (1/3))
    data_range = data.max() - data.min()
    
    if data_range <= 0 or bin_width <= 0:
        return 10
        
    bins = max(int(np.ceil(data_range / bin_width)), 5)
    return min(bins, max_bins)


def determine_task_type(y: pd.Series) -> Tuple[bool, str]:
    """
    Decide if *y* is classification or regression.

    Criteria:
    1. Any pandas.Categorical dtype        -> classification
    2. Any non‑numeric / object dtype      -> classification
    3. Binary data (exactly 0/1 or True/False) -> classification
    4. Integer data with only a few values  -> classification
    5. Otherwise                           -> regression
    
    Args:
        y: Series containing target values
        
    Returns:
        Tuple of (is_classification, task_type_string)
    """
    if pd.api.types.is_categorical_dtype(y):
        return True, "classification"
    if not pd.api.types.is_numeric_dtype(y):
        return True, "classification"
    
    # Check for binary classification
    unique_values = set(y.unique())
    if unique_values == {0, 1} or unique_values == {False, True}:
        return True, "classification"
    
    # Check if data appears to be multi-class (integers with small range)
    if pd.api.types.is_integer_dtype(y):
        min_val, max_val = y.min(), y.max()
        # If all values are integers in a small range and there aren't too many unique values
        if 0 <= min_val and max_val <= 10 and len(unique_values) <= (max_val - min_val + 1):
            return True, "classification"
        
    return False, "regression"


def pct_bar(pct: float, full_len: int = 50, char: str = "▒") -> str:
    """Return a fixed‑width bar filled proportionally to *pct*.
    
    Args:
        pct: Percentage value (0-100)
        full_len: Full length of the progress bar
        char: Character to use for the bar
        
    Returns:
        String representation of the progress bar
    """
    filled = math.ceil(full_len * pct / 100)
    return char * filled


def get_cv_splitter(task_type: str, n_splits: int, n_repeats: int, use_stratification: bool = True, random_state: Optional[int] = None) -> Union[RepeatedKFold, RepeatedStratifiedKFold]:
    """Get appropriate CV splitter based on task type.
    
    Args:
        task_type: "classification" or "regression"
        n_splits: Number of splits for CV
        n_repeats: Number of repeats for CV
        use_stratification: Whether to use stratification for classification tasks
        random_state: Random seed for reproducibility
        
    Returns:
        CV splitter object
    """
    if task_type == "classification" and use_stratification:
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    else:
        return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


def get_inner_cv_splitter(task_type: str, n_splits: int, use_stratification: bool = True, random_state: Optional[int] = None) -> Union[KFold, StratifiedKFold]:
    """Get appropriate inner CV splitter based on task type.
    
    Args:
        task_type: "classification" or "regression"
        n_splits: Number of splits for CV
        use_stratification: Whether to use stratification for classification tasks
        random_state: Random seed for reproducibility
        
    Returns:
        CV splitter object
    """
    if task_type == "classification" and use_stratification:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# Split-helper routines
def _outer_splits(
    cv: Union[RepeatedKFold, RepeatedStratifiedKFold],
    y: pd.Series,
    repeats: int,
) -> Iterable[Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Yield (repeat, fold, train_idx, test_idx).

    Centralises outer‑split generation so it uses exactly the same ordering.
    
    Args:
        cv: Cross-validation splitter object
        y: Target variable Series
        repeats: Number of repeats for CV
        
    Yields:
        Tuples of (repeat_number, fold_number, train_indices, test_indices)
    """
    n_per_repeat = cv.get_n_splits() // repeats
    for split_no, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        rep = (split_no - 1) // n_per_repeat + 1
        fold = (split_no - 1) % n_per_repeat + 1
        yield rep, fold, tr, te


def get_base_name(
    target: str,
    is_classification: bool,
    inner: int,
    outer: int,
    repeats: int,
    seed: Optional[int] = None
) -> str:
    """Generate standardized base name for directory and files.
    
    Args:
        target: Target variable name
        is_classification: Whether the task is classification or regression
        inner: Number of inner folds
        outer: Number of outer folds
        repeats: Number of repeats
        seed: Optional random seed to include in the name
        
    Returns:
        Standardized base name string for files and directories
    """
    problem_type = "classif" if is_classification else "regression"
    base = f"CV_{target}_{problem_type}_cv{outer}x{inner}x{repeats}"
    if seed is not None:
        return f"{base}_seed{seed}"
    return base


def save_splits_csv(
    csv_name: str,
    rows: list[dict],
    output_dir: Path,
    base_filename: str = "",
) -> None:
    """Helper for writing CSVs.
    
    Args:
        csv_name: Name of the CSV file (without the base filename)
        rows: List of dictionaries to save to CSV
        output_dir: Directory to save the CSV to
        base_filename: Optional base filename prefix
    """
    filename = f"{base_filename}_{csv_name}" if base_filename else csv_name
    pd.DataFrame(rows).to_csv(output_dir / filename, index=False)


def build_split_rows(
    ids: pd.Series,
    y: pd.Series,
    outer_cv: Union[RepeatedKFold, RepeatedStratifiedKFold],
    repeats: int,
    inner_folds: Optional[int] = None,
    seed: Optional[int] = None,
    task_type: str = "regression",
    use_stratification: bool = True,
) -> Tuple[list[dict], list[dict]]:
    """Build data structures for CV splits.
    
    Args:
        ids: Series of sample IDs
        y: Series of target values
        outer_cv: Outer cross-validation splitter
        repeats: Number of repeats
        inner_folds: Number of inner folds (None for no inner CV)
        seed: Random seed for reproducibility
        task_type: "classification" or "regression"
        use_stratification: Whether to use stratification for classification
        
    Returns:
        Tuple of (basic_rows, granular_rows) for CSV output
    """
    basic_rows: list[dict] = []
    granular_rows: list[dict] = []

    inner_cv = None
    if inner_folds is not None:
        inner_cv = get_inner_cv_splitter(task_type, inner_folds, use_stratification, seed)

    for rep, fold, tr_idx, te_idx in _outer_splits(outer_cv, y, repeats):
        # Basic (outer only)
        for kind, idxs in (("Train", tr_idx), ("Test", te_idx)):
            basic_rows.extend(
                {
                    "Repeat": rep,
                    "Fold": fold,
                    "Type": kind,
                    "ID": ids.iloc[i],
                    "Target": y.iloc[i],
                }
                for i in idxs
            )

        # Granular (outer + inner)
        granular_rows.extend(
            {
                "Repeat": rep,
                "OuterFold": fold,
                "InnerFold": "NA",
                "SplitType": "Test",
                "ID": ids.iloc[i],
                "Target": y.iloc[i],
            }
            for i in te_idx
        )

        if inner_cv is None:
            continue

        # Use appropriate inner split method based on task type
        X_dummy = np.zeros(len(tr_idx))
        y_inner = y.iloc[tr_idx]
        
        for inner_no, (in_tr, in_val) in enumerate(inner_cv.split(X_dummy, y_inner), start=1):
            tr_orig = tr_idx[in_tr]
            val_orig = tr_idx[in_val]

            granular_rows.extend(
                {
                    "Repeat": rep,
                    "OuterFold": fold,
                    "InnerFold": inner_no,
                    "SplitType": "Train",
                    "ID": ids.iloc[i],
                    "Target": y.iloc[i],
                }
                for i in tr_orig
            )
            granular_rows.extend(
                {
                    "Repeat": rep,
                    "OuterFold": fold,
                    "InnerFold": inner_no,
                    "SplitType": "Validation",
                    "ID": ids.iloc[i],
                    "Target": y.iloc[i],
                }
                for i in val_orig
            )

    return basic_rows, granular_rows


def plot_hist_or_bar(
    series: pd.Series,
    ax: plt.Axes,
    title: str,
    is_classification: bool,
    bins: Optional[int] = None,
) -> None:
    """Draw a histogram (regression) or bar chart (classification).
    
    Args:
        series: Data series to plot
        ax: Matplotlib axes to draw on
        title: Plot title
        is_classification: Whether to use a bar chart (True) or histogram (False)
        bins: Number of bins for histogram (regression only)
    """
    import matplotlib.patches as mpatches
    if is_classification:
        # For classification, get the value counts and sort by index
        counts = series.value_counts().sort_index()
        class_values = counts.index.tolist()
        num_classes = len(class_values)
        
        # Get appropriate color palette
        palette = get_color_palette(is_classification=True, class_count=num_classes)
        
        # Create the bar chart
        bars = ax.bar(range(len(class_values)), counts.values, color=palette)
        
        # Set proper x-axis ticks and labels
        ax.set_xticks(range(len(class_values)))
        ax.set_xticklabels([str(val) for val in class_values])
        
        ax.set_xlabel("Class")
        ax.set_ylabel("Frequency")
        
        # Create legend with actual class values
        patches = []
        labels = []
        for i, val in enumerate(class_values):
            color = palette[i % len(palette)]
            patches.append(mpatches.Patch(color=color))
            labels.append(f"Class {val}")
            
        if patches:  # Only create legend if we have classes
            ax.legend(patches, labels)
    else:
        # Calculate optimal bins if not specified
        if bins is None:
            bins = calculate_optimal_bins(series)
            
        # Get regression color
        hist_color = get_color_palette(is_classification=False)
        
        # Create histogram with computed bins
        sns.histplot(series, kde=False, bins=bins, color=hist_color, alpha=0.8, ax=ax)
        ax.set_xlabel("Target Value")
        ax.set_ylabel("Frequency")
        mean = series.mean()
        ax.axvline(mean, color="r", linestyle="--", label=f"Mean: {mean:.2f}")
        ax.legend()
    ax.set_title(title)


def plot_fold_comparison(
    train_series: pd.Series,
    test_series: pd.Series,
    ax: plt.Axes,
    is_classification: bool,
    title: str,
    p_value: float = None,
    alpha: float = 0.05
) -> None:
    """Plot train/test comparison for a fold.
    
    Args:
        train_series: Training data series
        test_series: Test data series
        ax: Matplotlib axes to draw on
        is_classification: Whether to use a bar chart (True) or histogram (False)
        title: Plot title
        p_value: Optional p-value to show on the plot
        alpha: Significance threshold for p-value coloring
    """
    # Get colors from centralized color definitions
    train_test_colors = get_train_test_colors()
    
    if is_classification:
        # Combine train and test data for classification
        comb = pd.concat(
            [
                pd.DataFrame({"y": train_series, "Set": "Train"}),
                pd.DataFrame({"y": test_series, "Set": "Test"}),
            ]
        )
        # Use colors from our centralized definitions
        sns.countplot(x="y", hue="Set", data=comb, ax=ax, 
                    palette=train_test_colors)
        ax.set_xlabel("Class")
        
        # Remove "Set" header from legend
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ["Train", "Test"])
    else:
        # Calculate optimal bins based on combined data
        data_combined = pd.concat([train_series, test_series])
        bins = calculate_optimal_bins(data_combined, max_bins=40)
        
        # Create histograms with computed bins and our centralized colors
        ax.hist(train_series, alpha=0.8, color=train_test_colors["Train"], bins=bins, label="Train")
        ax.hist(test_series, alpha=0.8, color=train_test_colors["Test"], bins=bins, label="Test")
        ax.set_xlabel("Target Value")
        ax.legend()
    
    # Create title with p-value if provided
    if p_value is not None:
        # Determine if p-value is "bad" (less than alpha)
        is_bad = p_value < alpha
        
        # Create the title with p-value
        p_text = f" (p={p_value:.4f})"
        
        # Set the title with p-value
        ax.set_title(f"{title}{p_text}", color='red' if is_bad else 'black')
    else:
        # Set the title without p-value
        ax.set_title(title)


def evaluate_fold_quality(
    y: pd.Series,
    cv_splitter: Union[RepeatedKFold, RepeatedStratifiedKFold],
    is_classification: bool,
    alpha: float = 0.05,
    repeats: int = 1
) -> Tuple[List[float], List[str], List[str]]:
    """Evaluate the quality of CV folds using statistical tests.
    
    Args:
        y: Target variable Series
        cv_splitter: CV splitter object
        is_classification: Whether the task is classification
        alpha: Significance level for statistical tests
        repeats: Number of repeats to evaluate
        
    Returns:
        Tuple of (p_values, marks, quality_lines)
    """
    pvals: List[float] = []
    marks: List[str] = []
    quality_lines: List[str] = []
    
    quality_lines.append(f"[✓] p-value ≥ {alpha:.2f}: Good fold balance ({('Chi-square' if is_classification else 'Kolmogorov-Smirnov')} test)")
    quality_lines.append(f"[✗] p-value < {alpha:.2f}: Potential fold imbalance")
    quality_lines.append("")
    
    # Evaluate all repeats up to the specified number
    all_splits = []
    for rep, fold, tr, te in _outer_splits(cv_splitter, y, repeats=repeats):
        if rep <= repeats:
            all_splits.append((rep, fold, tr, te))
    
    for rep, fold, tr, te in all_splits:
        tr_y, te_y = y.iloc[tr], y.iloc[te]
        if is_classification:
            all_classes = sorted(set(tr_y) | set(te_y))
            cont = np.vstack(
                [
                    [np.sum(tr_y == c) for c in all_classes],
                    [np.sum(te_y == c) for c in all_classes],
                ]
            )
            if (cont == 0).any() or len(all_classes) == 1:
                pval = 1.0
            else:
                pval = stats.chi2_contingency(cont)[1]
        else:
            pval = stats.ks_2samp(tr_y, te_y)[1]
        
        pvals.append(pval)
        mark = "✓" if pval >= alpha else "✗"
        marks.append(mark)
        # Calculate overall fold number for consistent reference with fold files
        overall_fold_num = (rep - 1) * (len(all_splits) // repeats) + fold
        quality_lines.append(f"  [{mark}] Repeat {rep}, Fold {fold} (overall fold {overall_fold_num}): p‑value={pval:.4f}")
    
    return pvals, marks, quality_lines


def build_reproduction_command(args: argparse.Namespace, seed: int, is_classification: bool, use_stratification: bool) -> str:
    """Build a reproduction command string with the current parameters.
    
    Args:
        args: Command line arguments
        seed: Random seed used
        is_classification: Whether the task is classification
        use_stratification: Whether stratification is being used
        
    Returns:
        Command string for reproducing the results
    """
    cmd = ["daftar-cv", f"--input {args.input}", f"--target {args.target}", f"--id {args.id}"]
    
    # Add optional arguments if they differ from defaults
    if args.outer != 5:
        cmd.append(f"--outer {args.outer}")
    if args.inner != 3:
        cmd.append(f"--inner {args.inner}")
    if args.repeats != 3:
        cmd.append(f"--repeats {args.repeats}")
    if args.seed is not None:
        cmd.append(f"--seed {args.seed}")
    if args.task_type is not None:
        cmd.append(f"--task_type {args.task_type}")
    
    # Add stratify only if it differs from the default behavior
    # Default: stratify for classification, don't stratify for regression
    default_stratification = is_classification
    if use_stratification != default_stratification:
        cmd.append(f"--stratify {'true' if use_stratification else 'false'}")
    
    if args.output_dir is not None:
        cmd.append(f"--output_dir {args.output_dir}")
    if args.alpha != 0.05:
        cmd.append(f"--alpha {args.alpha}")
    if args.granular:
        cmd.append("--granular")
    if args.force:
        cmd.append("--force")
        
    return " ".join(cmd)


def build_daftar_command(args: argparse.Namespace, seed: int, is_classification: bool = None, use_stratification: bool = None) -> str:
    """Build a DAFTAR command string with the current parameters.
    
    Args:
        args: Command line arguments
        seed: Random seed used
        is_classification: Whether the task is classification
        use_stratification: Whether stratification is being used
        
    Returns:
        Command string for running DAFTAR with these parameters
    """
    cmd = [
        "daftar",
        f"--input {args.input}",
        f"--target {args.target}",
        f"--id {args.id}",
        f"--outer {args.outer}",
        f"--inner {args.inner}",
        f"--repeats {args.repeats}",
        f"--seed {seed}",
        "--model [xgb|rf]"
    ]
    
    # Add stratify flag if this is classification and we need to override the default
    if is_classification is not None and use_stratification is not None:
        default_stratification = is_classification  # Default is to stratify for classification
        if use_stratification != default_stratification:
            cmd.append(f"--stratify {'true' if use_stratification else 'false'}")
    
    if args.output_dir:
        cmd.append(f"--output_dir {args.output_dir}")
        
    return " ".join(cmd)


# Command-line arguments
def get_args() -> argparse.Namespace:
    """Parse command line arguments and return the parsed namespace object."""
    p = argparse.ArgumentParser(
        description="Visualise target values across nested CV splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )

    req = p.add_argument_group("Required")
    cv = p.add_argument_group("Cross‑validation")
    task = p.add_argument_group("Task Type")
    out = p.add_argument_group("Output")
    viz = p.add_argument_group("Visualization")

    req.add_argument("--input", required=True, help="Path to CSV data file")
    req.add_argument("--target", required=True, help="Target column name")
    req.add_argument("--id", required=True, help="ID column name")

    cv.add_argument("--outer", type=int, default=5, help="Outer folds (default=5)")
    cv.add_argument("--inner", type=int, default=3, help="Inner folds (default=3)")
    cv.add_argument("--repeats", type=int, default=3, help="Repeats (default=3)")
    cv.add_argument("--seed", type=int, help="Random seed for reproducibility")
    cv.add_argument("--stratify", type=str, choices=["true", "false"], 
                  help="Whether to use stratified splitting for classification tasks (default: true for classification, false for regression)")
    
    task.add_argument(
        "--task_type",
        choices=["classification", "regression"],
        help="Force specific task type (default=auto-detect)"
    )

    out.add_argument(
        "--output_dir",
        help="Directory for all outputs (defaults to current directory)"
    )
    out.add_argument(
        "--force", 
        action="store_true",
        help="Force overwrite if output files already exist"
    )
    out.add_argument(
        "--granular",
        action="store_true",
        help="Generate additional detailed output files (may be large)"
    )
    
    viz.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Statistical significance threshold for fold balance tests (default=0.05)"
    )

    return p.parse_args()


# Main program
def main() -> None:
    """Main program execution function that orchestrates the CV calculation and visualization."""
    args = get_args()

    # Set up output dir
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"[ERROR] Input file not found: {input_path}")

    output_dir = Path(args.output_dir or Path.cwd()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read data & task type
    df = pd.read_csv(input_path)
    if args.target not in df.columns or args.id not in df.columns:
        sys.exit("[ERROR] target or id column missing in data")

    y = df[args.target]
    ids = df[args.id]
    
    # Determine task type (auto-detect or user override)
    if args.task_type is None:
        is_cls, task_type = determine_task_type(y)
        print(f"Auto-detected {task_type.upper()} task (override with --task_type flag if needed)")
    else:
        # User manually specified the task type
        is_cls = args.task_type == "classification"
        task_type = args.task_type
        print(f"User-specified {task_type.upper()} task")
    
    # Determine stratification based on task type and user input
    if args.stratify is None:
        # Default behavior: stratify for classification, don't stratify for regression
        use_stratification = is_cls
    else:
        # User explicitly specified stratification
        use_stratification = args.stratify == "true"
    
    # Print information about CV splitter
    if is_cls:
        if use_stratification:
            print(f"Using StratifiedKFold for classification (default, override with --stratify false)")
        else:
            print(f"Using KFold for classification (as requested with --stratify false)")
    else:
        print(f"Using KFold for regression (default for regression tasks)")

    # Use appropriate CV splitter based on task
    seed = args.seed or np.random.randint(0, 2**31 - 1)
    outer_cv = get_cv_splitter(task_type, args.outer, args.repeats, use_stratification, seed)

    # Generate base filename
    base_name = get_base_name(
        args.target, is_cls, args.inner, args.outer, args.repeats, seed
    )
    
    # Create a results directory
    results_dir = output_dir / f"{base_name}_results"
    
    # Check if the results directory already exists
    if results_dir.exists() and not args.force:
        print(f"\nERROR: Results directory already exists: {results_dir}")
        print(f"To overwrite existing files, add the --force flag:")
        
        # Build command for the user to run with --force
        cmd = build_reproduction_command(args, seed, is_cls, use_stratification) + " --force"
        print(f"\n===\n{cmd}\n===\n")
        return 1
        
    # Create the directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Split CSVs - only create granular ones if requested
    basic_rows, granular_rows = build_split_rows(
        ids, y, outer_cv, args.repeats, args.inner, seed, task_type, use_stratification
    )
    save_splits_csv("splits_basic.csv", basic_rows, results_dir, base_name)
    
    if args.granular:
        save_splits_csv("splits_granular.csv", granular_rows, results_dir, base_name)
    
    # Set up matplotlib style for better plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Overall histogram
    plt.figure(figsize=(12, 8))
    plot_hist_or_bar(
        y, plt.gca(), "Overall Target Distribution", is_classification=is_cls
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        filename = f"{base_name}_overall_distribution.{ext}"
        plt.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()

    # Gather all fold/repeat combinations for quality assessment
    all_fold_data = []
    for rep, fold, tr_idx, te_idx in _outer_splits(outer_cv, y, args.repeats):
        # We'll collect the data for every fold and repeat
        all_fold_data.append((rep, fold, tr_idx, te_idx))
    
    # Group fold data by repeat
    fold_data_by_repeat = {}
    for rep, fold, tr_idx, te_idx in all_fold_data:
        if rep not in fold_data_by_repeat:
            fold_data_by_repeat[rep] = []
        fold_data_by_repeat[rep].append((fold, tr_idx, te_idx))
        
    # Calculate p-values for each repeat separately
    pval_map = {}
    quality_lines_by_repeat = {}
    
    for rep, fold_data in fold_data_by_repeat.items():
        # Extract fold info for this repeat only
        rep_fold_idxs = [(fold, tr, te) for fold, tr, te in fold_data]
        
        # Calculate p-values for this repeat
        pvals = []
        marks = []
        for fold, tr, te in rep_fold_idxs:
            tr_y, te_y = y.iloc[tr], y.iloc[te]
            if is_cls:
                all_classes = sorted(set(tr_y) | set(te_y))
                cont = np.vstack(
                    [
                        [np.sum(tr_y == c) for c in all_classes],
                        [np.sum(te_y == c) for c in all_classes],
                    ]
                )
                if (cont == 0).any() or len(all_classes) == 1:
                    pval = 1.0
                else:
                    pval = stats.chi2_contingency(cont)[1]
            else:
                pval = stats.ks_2samp(tr_y, te_y)[1]
            
            pvals.append(pval)
            pval_map[(rep, fold)] = pval
            mark = "✓" if pval >= args.alpha else "✗"
            marks.append(mark)
        
        # Store quality lines for this repeat
        quality_lines = []
        for (fold, _, _), pval, mark in zip(rep_fold_idxs, pvals, marks):
            # Calculate overall fold number for consistent reference with fold files
            overall_fold_num = (rep - 1) * len(fold_data) + fold
            quality_lines.append(f"  [{mark}] Repeat {rep}, Fold {fold} (overall fold {overall_fold_num}): p‑value={pval:.4f}")
        
        quality_lines_by_repeat[rep] = quality_lines
    
    # Calculate p-values for fold quality assessment for reporting purposes
    # This includes all repeats
    report_pvals, report_marks, report_quality_lines = evaluate_fold_quality(y, outer_cv, is_cls, args.alpha, args.repeats)

    # Fold-wise histograms
    n_splits = args.outer
    fig, axes = plt.subplots(
        n_splits, args.repeats, figsize=(args.repeats * 4, n_splits * 4), squeeze=False
    )
    
    for rep, fold, tr_idx, te_idx in all_fold_data:
        ax = axes[fold - 1, rep - 1]
        tr_y, te_y = y.iloc[tr_idx], y.iloc[te_idx]
        
        # Get p-value from our map - now we have p-values for ALL repeats
        p_value = pval_map.get((rep, fold), None)
        
        # Use the specialized comparison plot function
        plot_fold_comparison(tr_y, te_y, ax, is_cls, f"Fold {fold}, Repeat {rep}", p_value, args.alpha)
    
    plt.tight_layout()
    for ext in ("png", "pdf"):
        filename = f"{base_name}_histograms.{ext}"
        plt.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()

    # Nested-CV ASCII diagram & stats report
    report_lines: list[str] = []
    console_lines: list[str] = []

    # HEADER
    header_lines = [
        "=" * 80,
        "CROSS‑VALIDATION CONFIGURATION",
        "=" * 80,
        "",
        "DATASET SUMMARY",
        "-" * 40,
        f"- Total samples: {len(y)}",
        f"- TASK TYPE: {task_type.upper()}",
        f"- TARGET COLUMN: {args.target}",
        f"- ID COLUMN: {args.id}",
        "",
        "CROSS‑VALIDATION SETUP",
        "-" * 40,
        f"Folds:    {args.outer} outer / {args.inner} inner",
        f"Repeats:  {args.repeats}",
        f"Seed:     {seed}",
    ]
    report_lines.extend(header_lines)
    console_lines.extend(header_lines)

    # KEY
    key_lines = [
        "",
        "KEY",
        "-" * 40,
        "  ░ Full dataset",
        "  ▒ Training   → Model learning",
        "  ▓ Validation → Hyperparameter tuning",
        "  █ Testing    → Final evaluation",
        "",
        "NESTED PROCESS WITH VISUALISATION",
        "=" * 80,
    ]
    report_lines.extend(key_lines)
    console_lines.extend(key_lines)

    # ASCII – full dataset bar (both report and console)
    dataset_bar = f"Full dataset: {len(y)} samples (100%)  | {'░' * 50}"
    report_lines.append(dataset_bar)
    console_lines.append(dataset_bar)
    report_lines.append("")
    console_lines.append("")

    # ASCII – repeats & folds for REPORT FILE (complete)
    for rep in range(1, args.repeats + 1):
        report_lines.append(f"REPEAT {rep}/{args.repeats}")
        for fold in range(1, args.outer + 1):
            # Calculate percentages
            tr_pct = 100 * (args.outer - 1) / args.outer
            te_pct = 100 / args.outer
            
            # Calculate actual sample counts
            train_count = int(len(y) * tr_pct / 100)
            test_count = len(y) - train_count
            
            report_lines.append(f"  └─ OUTER FOLD {fold}/{args.outer}")
            report_lines.append(
                f"       • Training:    {pct_bar(tr_pct)} ({tr_pct:.0f}%, {train_count} samples)"
            )
            # Show inner fold details for all folds in the first repeat
            if rep == 1:
                # Calculate inner fold percentages
                in_tr_pct = tr_pct * (args.inner - 1) / args.inner
                in_val_pct = tr_pct / args.inner
                
                # Calculate inner fold sample counts
                inner_train_count = int(train_count * (args.inner - 1) / args.inner)
                inner_val_count = train_count - inner_train_count
                
                # Show all inner folds
                for inner_fold in range(1, args.inner + 1):
                    report_lines.append(
                        f"            └─ INNER FOLD {inner_fold}/{args.inner}"
                    )
                    report_lines.append(
                        f"               • Training:   {pct_bar(in_tr_pct)} ({in_tr_pct:.0f}%, {inner_train_count} samples)"
                    )
                    report_lines.append(
                        f"               • Validation: {pct_bar(in_val_pct, char='▓')} ({in_val_pct:.0f}%, {inner_val_count} samples)"
                    )
            else:
                # For subsequent repeats, just show a collapsed message
                report_lines.append(
                    f"            └─ INNER FOLDS: pattern identical to first repeat"
                )
            report_lines.append(
                f"       • Testing:     {pct_bar(te_pct, char='█')} ({te_pct:.0f}%, {test_count} samples)"
            )
        report_lines.append("")
        
    # ASCII – repeats & folds for CONSOLE (collapsed)
    for rep in range(1, args.repeats + 1):
        console_lines.append(f"REPEAT {rep}/{args.repeats}")
        if rep == 1:
            # For first repeat, show detailed first fold
            for fold in range(1, args.outer + 1):
                # Calculate percentages
                tr_pct = 100 * (args.outer - 1) / args.outer
                te_pct = 100 / args.outer
                
                # Calculate actual sample counts
                train_count = int(len(y) * tr_pct / 100)
                test_count = len(y) - train_count
                
                console_lines.append(f"  └─ OUTER FOLD {fold}/{args.outer}")
                console_lines.append(
                    f"       • Training:    {pct_bar(tr_pct)} ({tr_pct:.0f}%, {train_count} samples)"
                )
                # Always show inner fold details for all folds in the first repeat
                # Calculate inner fold percentages
                in_tr_pct = tr_pct * (args.inner - 1) / args.inner
                in_val_pct = tr_pct / args.inner
                
                # Calculate inner fold sample counts
                inner_train_count = int(train_count * (args.inner - 1) / args.inner)
                inner_val_count = train_count - inner_train_count
                
                # Show all inner folds
                for inner_fold in range(1, args.inner + 1):
                    console_lines.append(
                        f"            └─ INNER FOLD {inner_fold}/{args.inner}"
                    )
                    console_lines.append(
                        f"               • Training:   {pct_bar(in_tr_pct)} ({in_tr_pct:.0f}%, {inner_train_count} samples)"
                    )
                    console_lines.append(
                        f"               • Validation: {pct_bar(in_val_pct, char='▓')} ({in_val_pct:.0f}%, {inner_val_count} samples)"
                    )
                console_lines.append(
                    f"       • Testing:     {pct_bar(te_pct, char='█')} ({te_pct:.0f}%, {test_count} samples)"
                )
        else:
            # For subsequent repeats, just show a collapsed message
            console_lines.append(f"  └─ Identical structure to Repeat 1 (with different random splits)")
        console_lines.append("")

    # Fold quality assessment section (both report and console)
    quality_header = [
        "=" * 80,
        "FOLD QUALITY ASSESSMENT",
        "=" * 80,
        "",
    ]
    report_lines.extend(quality_header)
    console_lines.extend(quality_header)
    report_lines.extend(report_quality_lines)
    console_lines.extend(report_quality_lines)  # Use the same quality lines for console output

    # Summary section
    good = sum(1 for p in report_pvals if p >= args.alpha)
    total_folds = len(report_pvals)
    summary_lines = [
        "",
        "SUMMARY",
        "-" * 40,
        f"{good}/{total_folds} folds have sufficiently similar distributions.",
    ]
    if good < 0.8 * total_folds:
        summary_lines += [
            "",
            "⚠️  WARNING: Too many imbalanced folds. Consider a different seed.",
        ]
        
    report_lines.extend(summary_lines)
    console_lines.extend(summary_lines)
    
    # Files generated section (both report and console)
    files_list = [
        "",
        "FILES GENERATED",
        "-" * 40,
        f"- Directory: {results_dir.name}/",
        f"    {base_name}_splits_basic.csv",
    ]
    
    if args.granular:
        files_list.append(f"    {base_name}_splits_granular.csv")
        
    files_list.extend([
        f"    {base_name}_histograms.png/pdf",
        f"    {base_name}_overall_distribution.png/pdf", 
        f"    {base_name}_fold_report.txt",
    ])
    
    report_lines.extend(files_list)
    console_lines.extend(files_list)
    
    # Build the DAFTAR command for next steps
    daftar_cmd = build_daftar_command(args, seed, is_cls, use_stratification)
    
    # Next steps section (both report and console)
    next_steps = [
        "",
        "=" * 80,
        "NEXT STEPS",
        "=" * 80,
        "Run DAFTAR‑ML with these CV parameters:",
        "",
        daftar_cmd,
        "",
    ]
    report_lines.extend(next_steps)
    console_lines.extend(next_steps)
    
    # Write report (full)
    report_filename = f"{base_name}_fold_report.txt"
    with open(results_dir / report_filename, "w") as f:
        f.write("\n".join(report_lines))

    # Console echo (collapsed)
    for line in console_lines:
        print(line)


if __name__ == "__main__":
    main()