#!/usr/bin/env python
"""
Nested CV target-distribution visualiser.
"""

from __future__ import annotations

# Customizable visualization colors
# Classification visualization colors
CLASS_BAR_COLOR0 = "#1C0F13"          # Primary bar chart color for classification
CLASS_BAR_COLOR1 = "#6E7E85"       # Secondary bar chart color for classification (alternating)

# Regression visualization colors
REGRESSION_HIST_COLOR = "#1C0F13"       # Histogram fill color
REGRESSION_HIST_ALPHA = 0.8          # Histogram transparency
REGRESSION_MEAN_LINE_COLOR = "r"     # Mean line color

# Compare train/test visualization colors
TRAIN_HIST_COLOR = "#70e4ef"            # Train set histogram color
TEST_HIST_COLOR = "#dfdf20"           # Test set histogram color 
HIST_ALPHA = 0.8                     # Transparency for compare histograms

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.model_selection import KFold, RepeatedKFold

# Global state (computed once in main())
_IS_CLASSIFICATION: bool | None = None
_TASK_TYPE: str | None = None        # "classification" | "regression"


# Utilities
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
        if 0 <= min_val and max_val <= 3 and len(unique_values) <= (max_val - min_val + 1):
            return True, "classification"
        
    return False, "regression"


def set_task_type(y: pd.Series) -> Tuple[bool, str]:
    """Initialise global flags exactly once.
    
    Args:
        y: Target variable Series
        
    Returns:
        Tuple of (is_classification, task_type_string)
    """
    global _IS_CLASSIFICATION, _TASK_TYPE
    _IS_CLASSIFICATION, _TASK_TYPE = determine_task_type(y)
    return _IS_CLASSIFICATION, _TASK_TYPE


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


# Split-helper routines
def _outer_splits(
    cv: RepeatedKFold | KFold,
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
    for split_no, (tr, te) in enumerate(cv.split(y), start=1):
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
    outer_cv: RepeatedKFold,
    repeats: int,
    inner_folds: int | None = None,
    seed: int | None = None,
) -> Tuple[list[dict], list[dict]]:
    """Build data structures for CV splits.
    
    Args:
        ids: Series of sample IDs
        y: Series of target values
        outer_cv: Outer cross-validation splitter
        repeats: Number of repeats
        inner_folds: Number of inner folds (None for no inner CV)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (basic_rows, granular_rows) for CSV output
    """
    basic_rows: list[dict] = []
    granular_rows: list[dict] = []

    inner_cv = (
        None
        if inner_folds is None
        else KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    )

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

        for inner_no, (in_tr, in_val) in enumerate(inner_cv.split(tr_idx), start=1):
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


# Visualization helpers
def plot_hist_or_bar(
    series: pd.Series,
    ax: plt.Axes,
    title: str,
    classification: bool,
    bins: int = None,
) -> None:
    """Draw a histogram (regression) or bar chart (classification).
    
    Args:
        series: Data series to plot
        ax: Matplotlib axes to draw on
        title: Plot title
        classification: Whether to use a bar chart (True) or histogram (False)
        bins: Number of bins for histogram (regression only)
    """
    import matplotlib.patches as mpatches
    if classification:
        # For classification, use alternating colors for the bars
        counts = series.value_counts().sort_index()
        class_values = counts.index.tolist()
        
        # Use alternating colors for the bars
        colors = [CLASS_BAR_COLOR0 if i % 2 == 0 else CLASS_BAR_COLOR1 for i in range(len(counts))]
        bars = ax.bar(range(len(class_values)), counts.values, color=colors)
        
        # Set proper x-axis ticks and labels
        ax.set_xticks(range(len(class_values)))
        ax.set_xticklabels([str(val) for val in class_values])
        
        ax.set_xlabel("Class")
        ax.set_ylabel("Frequency")
        
        # Create legend with actual class values
        patches = []
        labels = []
        for i, val in enumerate(class_values):
            color = CLASS_BAR_COLOR0 if i % 2 == 0 else CLASS_BAR_COLOR1
            patches.append(mpatches.Patch(color=color))
            labels.append(f"Class {val}")
            
        if patches:  # Only create legend if we have classes
            ax.legend(patches, labels)
    else:
        # Automatically calculate the number of bins if not specified
        if bins is None:
            # Use Freedman-Diaconis rule to determine optimal bin width
            n = len(series)
            if n > 0:
                iqr = np.percentile(series, 75) - np.percentile(series, 25)
                if iqr > 0:
                    bin_width = 2 * iqr / (n ** (1/3))
                    data_range = series.max() - series.min()
                    if data_range > 0 and bin_width > 0:
                        bins = max(int(np.ceil(data_range / bin_width)), 5)
                    else:
                        bins = 10
                else:
                    bins = 10
            else:
                bins = 10
            
            # Cap number of bins to avoid overly detailed histograms
            bins = min(bins, 50)
            
        # Create histogram with computed bins
        sns.histplot(series, kde=False, bins=bins, color=REGRESSION_HIST_COLOR, alpha=REGRESSION_HIST_ALPHA, ax=ax)
        ax.set_xlabel("Target Value")
        ax.set_ylabel("Frequency")
        mean = series.mean()
        ax.axvline(mean, color=REGRESSION_MEAN_LINE_COLOR, linestyle="--", label=f"Mean: {mean:.2f}")
        ax.legend()
    ax.set_title(title)


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

    req.add_argument("--input", required=True, help="Path to CSV data file")
    req.add_argument("--target", required=True, help="Target column name")
    req.add_argument("--id", required=True, help="ID column name")

    cv.add_argument("--outer", type=int, default=5, help="Outer folds (default=5)")
    cv.add_argument("--inner", type=int, default=3, help="Inner folds (default=3)")
    cv.add_argument("--repeats", type=int, default=3, help="Repeats (default=3)")
    cv.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    task.add_argument(
        "--task_type",
        choices=["classification", "regression"],
        help="Force specific task type (default=auto-detect)"
    )

    out.add_argument(
        "--output_dir",
        help="Directory for all outputs (defaults to current directory)",
    )
    out.add_argument(
        "--force", 
        action="store_true",
        help="Force overwrite if output files already exist"
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
    global _IS_CLASSIFICATION, _TASK_TYPE
    
    if args.task_type is None:
        is_cls, task_type = set_task_type(y)  # should set globals, but let's make sure
        _IS_CLASSIFICATION, _TASK_TYPE = is_cls, task_type  # explicitly set globals
        print(f"Auto-detected {task_type.upper()} task")
    else:
        # User manually specified the task type
        is_cls = args.task_type == "classification"
        task_type = args.task_type
        _IS_CLASSIFICATION, _TASK_TYPE = is_cls, task_type  # set globals
        print(f"User-specified {task_type.upper()} task")

    # Repeated-KFold
    seed = args.seed or np.random.randint(0, 2**31 - 1)
    outer_cv = RepeatedKFold(
        n_splits=args.outer, n_repeats=args.repeats, random_state=seed
    )

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
        cmd = f"daftar-cv --input {args.input}"
        if args.output_dir:
            cmd += f" --output_dir {args.output_dir}"
        cmd += f" --target {args.target} --id {args.id}"
        
        if args.outer != 5:
            cmd += f" --outer {args.outer}"
        if args.inner != 3:
            cmd += f" --inner {args.inner}"
        if args.repeats != 3:
            cmd += f" --repeats {args.repeats}"
        if args.seed:
            cmd += f" --seed {args.seed}"
        if args.task_type:
            cmd += f" --task_type {args.task_type}"
            
        cmd += f" --force"
        print(cmd)
        return 1
        
    # Create the directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Split CSVs
    basic_rows, granular_rows = build_split_rows(
        ids, y, outer_cv, args.repeats, args.inner, seed
    )
    save_splits_csv("splits_basic.csv", basic_rows, results_dir, base_name)
    save_splits_csv("splits_granular.csv", granular_rows, results_dir, base_name)
    
    # Overall histogram
    plt.figure(figsize=(12, 8))
    plot_hist_or_bar(
        y, plt.gca(), "Overall Target Distribution", classification=is_cls
    )
    for ext in ("png", "pdf"):
        filename = f"{base_name}_overall_distribution.{ext}"
        plt.savefig(results_dir / filename, dpi=300)
    plt.close()

    # Fold-wise histograms
    n_splits = args.outer
    fig, axes = plt.subplots(
        n_splits, args.repeats, figsize=(args.repeats * 4, n_splits * 4), squeeze=False
    )
    for rep, fold, tr_idx, te_idx in _outer_splits(outer_cv, y, args.repeats):
        ax = axes[fold - 1, rep - 1]
        tr_y, te_y = y.iloc[tr_idx], y.iloc[te_idx]
        if is_cls:
            comb = pd.concat(
                [
                    pd.DataFrame({"y": tr_y, "Set": "Train"}),
                    pd.DataFrame({"y": te_y, "Set": "Test"}),
                ]
            )
            # Use custom colors for train and test
            sns.countplot(x="y", hue="Set", data=comb, ax=ax, 
                        palette={"Train": TRAIN_HIST_COLOR, "Test": TEST_HIST_COLOR})
            ax.set_xlabel("Class")
        else:
            # Use matplotlib histograms with customized colors and transparency
            # Calculate optimal number of bins based on data
            data_combined = pd.concat([tr_y, te_y])
            n = len(data_combined)
            if n > 0:
                iqr = np.percentile(data_combined, 75) - np.percentile(data_combined, 25)
                if iqr > 0:
                    bin_width = 2 * iqr / (n ** (1/3))
                    data_range = data_combined.max() - data_combined.min()
                    if data_range > 0 and bin_width > 0:
                        bins = max(int(np.ceil(data_range / bin_width)), 5)
                    else:
                        bins = 15
                else:
                    bins = 15
            else:
                bins = 15
            
            # Cap number of bins to avoid overly detailed histograms
            bins = min(bins, 40)
            
            # Create histograms with computed bins
            ax.hist(tr_y, alpha=HIST_ALPHA, color=TRAIN_HIST_COLOR, bins=bins, label="Train")
            ax.hist(te_y, alpha=HIST_ALPHA, color=TEST_HIST_COLOR, bins=bins, label="Test")
            ax.set_xlabel("Target Value")
            ax.legend()
        ax.set_title(f"Fold {fold}")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        filename = f"{base_name}_histograms.{ext}"
        plt.savefig(results_dir / filename, dpi=300)
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
        f"- TASK TYPE: {_TASK_TYPE.upper()}",
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
            tr_pct = 100 * (args.outer - 1) / args.outer
            te_pct = 100 / args.outer
            report_lines.append(f"  └─ OUTER FOLD {fold}/{args.outer}")
            report_lines.append(
                f"       • Training:    {pct_bar(tr_pct)} ({tr_pct:.0f}%)"
            )
            if fold == 1 and rep == 1:
                report_lines.append(f"         └─ INNER CV ({args.inner} folds)")
                in_tr_pct = tr_pct * (args.inner - 1) / args.inner
                in_val_pct = tr_pct / args.inner
                report_lines.append(
                    f"            └─ INNER FOLD 1/{args.inner}"
                )
                report_lines.append(
                    f"               • Training:   {pct_bar(in_tr_pct)} ({in_tr_pct:.0f}%)"
                )
                report_lines.append(
                    f"               • Validation: {pct_bar(in_val_pct, char='▓')} ({in_val_pct:.0f}%)"
                )
            else:
                report_lines.append(
                    f"         └─ INNER CV: {args.inner} folds (pattern identical to first fold)"
                )
            report_lines.append(
                f"       • Testing:     {pct_bar(te_pct, char='█')} ({te_pct:.0f}%)"
            )
        report_lines.append("")
        
    # ASCII – repeats & folds for CONSOLE (collapsed)
    for rep in range(1, args.repeats + 1):
        console_lines.append(f"REPEAT {rep}/{args.repeats}")
        if rep == 1:
            # For first repeat, show detailed first fold
            for fold in range(1, args.outer + 1):
                tr_pct = 100 * (args.outer - 1) / args.outer
                te_pct = 100 / args.outer
                console_lines.append(f"  └─ OUTER FOLD {fold}/{args.outer}")
                console_lines.append(
                    f"       • Training:    {pct_bar(tr_pct)} ({tr_pct:.0f}%)"
                )
                if fold == 1:
                    console_lines.append(f"         └─ INNER CV ({args.inner} folds)")
                    in_tr_pct = tr_pct * (args.inner - 1) / args.inner
                    in_val_pct = tr_pct / args.inner
                    console_lines.append(
                        f"            └─ INNER FOLD 1/{args.inner}"
                    )
                    console_lines.append(
                        f"               • Training:   {pct_bar(in_tr_pct)} ({in_tr_pct:.0f}%)"
                    )
                    console_lines.append(
                        f"               • Validation: {pct_bar(in_val_pct, char='▓')} ({in_val_pct:.0f}%)"
                    )
                else:
                    console_lines.append(
                        f"         └─ INNER CV: {args.inner} folds (pattern identical to first fold)"
                    )
                console_lines.append(
                    f"       • Testing:     {pct_bar(te_pct, char='█')} ({te_pct:.0f}%)"
                )
        else:
            # For subsequent repeats, just show a collapsed message
            console_lines.append(f"  └─ Identical structure to Repeat 1 (with different random splits)")
        console_lines.append("")

    # Prepare variables needed for reports
    alpha = 0.05
    pvals: list[float] = []
    
    daftar_cmd = (
        f"  daftar --input {args.input} --target {args.target} --id {args.id} "
        f"--outer {args.outer} --inner {args.inner} --repeats {args.repeats} "
        f"--seed {seed} --model [xgb|rf]"
    )
    if args.output_dir:
        daftar_cmd += f" --output_dir {args.output_dir}"

    # Fold quality assessment section (both report and console)
    quality_lines = [
        "=" * 80,
        "FOLD QUALITY ASSESSMENT",
        "=" * 80,
        "",
        f"[✓] p-value ≥ 0.05: Good fold balance ({('Chi-square' if is_cls else 'Kolmogorov-Smirnov')} test)",
        f"[✗] p-value < 0.05: Potential fold imbalance",
        "",
    ]
    for idx, (tr, te) in enumerate(outer_cv.split(y), start=1):
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
        mark = "✓" if pval >= alpha else "✗"
        quality_lines.append(f"  [{mark}] Fold {idx}: p‑value={pval:.4f}")

    good = sum(p >= alpha for p in pvals)
    quality_lines += [
        "",
        "SUMMARY",
        "-" * 40,
        f"{good}/{len(pvals)} folds have sufficiently similar distributions.",
    ]
    if good < 0.8 * len(pvals):
        quality_lines += [
            "",
            "⚠️  WARNING: Too many imbalanced folds. Consider a different seed.",
        ]
        
    report_lines.extend(quality_lines)
    console_lines.extend(quality_lines)
    
    # Files generated section (both report and console)
    files_list = [
        "",
        "FILES GENERATED",
        "-" * 40,
        f"- Directory: {results_dir.name}/",
        f"    {base_name}_splits_basic.csv",
        f"    {base_name}_splits_granular.csv",
        f"    {base_name}_histograms.png/pdf",
        f"    {base_name}_overall_distribution.png/pdf", 
        f"    {base_name}_fold_report.txt",
    ]
    report_lines.extend(files_list)
    console_lines.extend(files_list)
    
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