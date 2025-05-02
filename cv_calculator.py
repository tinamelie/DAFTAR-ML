#!/usr/bin/env python
"""
Cross-validation Configuration Calculator

This script analyzes a dataset and calculates the sample sizes for different
cross-validation configurations, helping users understand how their data
will be split during nested cross-validation. Outputs a configuration summary
and provides instructions for running DAFTAR-ML.
"""

import argparse
import os
import sys
import math
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate nested cross-validation configuration details.",
        allow_abbrev=False
    )
    parser.add_argument(
        "--input", required=True, 
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--target", required=True, 
        help="Name of the target column to predict"
    )
    parser.add_argument(
        "--id", required=True, 
        help="Name of the ID column (e.g., species identifiers, sample names)"
    )
    parser.add_argument(
        "--outer", type=int, default=5, 
        help="Number of outer folds (default: 5)"
    )
    parser.add_argument(
        "--inner", type=int, default=3, 
        help="Number of inner folds (default: 3)"
    )
    parser.add_argument(
        "--repeats", type=int, default=3, 
        help="Number of times to repeat the nested cross-validation (inner + outer loops)(default: 3)"
    )
    parser.add_argument(
        "--output_dir",
        help="Directory where output files will be saved. If specified, visualizations will be saved to this directory"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite if output files already exist"
    )
    # Customize help message to capitalize 'Show'
    parser._optionals.title = "options"
    parser._actions[0].help = "Show this help message and exit"
    
    return parser.parse_args()


def generate_visualization_bars(percentage, length=50, char='▒'):
    """Generate visualization bars for the output."""
    filled = math.ceil(percentage * length / 100)
    return char * filled


def main():
    """Main function."""
    args = parse_args()
    
    # Extract file name from input path
    filename = os.path.basename(args.input)
    
    # Load dataset and validate columns
    try:
        data = pd.read_csv(args.input)
        n_samples = len(data)
        
        # Validate required columns
        if args.target not in data.columns:
            print(f"Error: Target column '{args.target}' not found in dataset")
            return
        if args.id not in data.columns:
            print(f"Error: ID column '{args.id}' not found in dataset")
            return
            
        # Determine task type (regression or classification)
        task_type = "classification"
        target_data = data[args.target].dropna()
        unique_values = target_data.unique()
        
        # If only 2 unique values or boolean values, it's classification
        if len(unique_values) <= 2 or target_data.dtype == bool:
            task_type = "classification"
        else:
            # If more than 2 unique values and numeric, it's likely regression
            if pd.api.types.is_numeric_dtype(target_data):
                task_type = "regression"
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    # Store parameters for template
    k_outer = args.outer
    k_inner = args.inner
    n_repeats = args.repeats
    
    # Check if any fold parameters were specified
    outer_specified = "--outer" in sys.argv
    inner_specified = "--inner" in sys.argv
    repeats_specified = "--repeats" in sys.argv
    
    # Force all-or-nothing approach for fold parameters
    if outer_specified or inner_specified or repeats_specified:
        if not (outer_specified and inner_specified and repeats_specified):
            print("ERROR: You must either specify ALL fold parameters (--outer, --inner, --repeats)")
            print("       or NONE of them to use the defaults.")
            return
    else:
        # Using all defaults, show the message
        print()
        print("NOTE: Using default CV configuration: 5 outer folds, 3 inner folds, 3 repeats.")
        print("      Consider optimizing these values for your specific dataset size.")
        print("      For guidance, see the 'Cross-Validation Guidelines' section in the README.")
        print()
    
    # Calculate metrics
    outer_train_ct = round(n_samples * (k_outer - 1) / k_outer)
    outer_test_ct = round(n_samples / k_outer)
    outer_train_p = round(100 * outer_train_ct / n_samples)
    outer_test_p = round(100 * outer_test_ct / n_samples)
    
    # Calculate inner fold sizes (train = outer_train - validation)
    inner_valid_ct = outer_train_ct // k_inner
    inner_train_ct = outer_train_ct - inner_valid_ct
    inner_train_p = round(inner_train_ct / n_samples * 100)
    inner_valid_p = round(inner_valid_ct / n_samples * 100)
    
    total_models = n_repeats * k_outer * k_inner
    
    # Generate visualization bars
    bar_width = 50
    full_dataset_bar = generate_visualization_bars(100, length=bar_width, char='░')
    outer_train_bar = generate_visualization_bars(outer_train_p, length=bar_width)
    outer_test_bar = generate_visualization_bars(outer_test_p, length=bar_width, char='█')
    inner_train_bar = generate_visualization_bars(inner_train_p, length=bar_width)
    inner_valid_bar = generate_visualization_bars(inner_valid_p, length=bar_width, char='▓')
    
    # Print output
    print()
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Target column:  {args.target}")
    print(f"ID column:      {args.id}")
    print(f"- Dataset: {filename}")
    print(f"- Total samples: {n_samples}")
    print()
    print("CROSS-VALIDATION SETUP")
    print("-" * 40)
    print(f"Folds:    {k_outer} outer / {k_inner} inner")
    print(f"Repeats:  {n_repeats} with different random splits")
    print()
    print("MODELS")
    print("-" * 40)
    print(f"Total trained: {total_models} ({k_outer} outer folds × {k_inner} inner folds × {n_repeats} repeats)")
    print(f"Total saved:   {k_outer * n_repeats} ({k_outer} best models per repeat × {n_repeats} repeats)")
    print()   
    print("KEY")
    print("-" * 40)
    print("  ░ Full dataset")
    print("  ▒ Training   → Model learning")
    print("  ▓ Validation → Hyperparameter tuning")
    print("  █ Testing    → Final evaluation")
    
    print()
    print("NESTED PROCESS WITH VISUALIZATION")
    print("=" * 80)
    print()
    
    # Show first repeat with full details
    
    print(f"Full dataset: {n_samples} samples (100%)  | {full_dataset_bar}")
    print()
    print("REPEAT 1/3")
    for outer_fold in range(1, k_outer + 1):
        # Calculate padding for fold numbers
        outer_fold_pad = len(str(k_outer))
        inner_fold_pad = len(str(k_inner))
        
        print(f"  └─ OUTER FOLD {outer_fold:>{outer_fold_pad}}/{k_outer}")
        print(f"       • Training: {outer_train_ct:>4} samples ({outer_train_p:>2}%)  | {outer_train_bar}")
        print(f"       • Testing:  {outer_test_ct:>4} samples ({outer_test_p:>2}%)  | {outer_test_bar}")
        
        # Show inner fold details only for the first outer fold
        if outer_fold == 1:
            for inner_fold in range(1, k_inner + 1):
                print(f"       └─ INNER FOLD {inner_fold:>{inner_fold_pad}}/{k_inner}")
                print(f"            • Training:   {inner_train_ct:>4} samples ({inner_train_p:>2}%)  | {inner_train_bar}")
                print(f"            • Validation: {inner_valid_ct:>4} samples ({inner_valid_p:>2}%)  | {inner_valid_bar}")
    
    # Show remaining repeats with just the message
    for repeat in range(2, n_repeats + 1):
        print()
        print(f"REPEAT {repeat}/{n_repeats}")
        print("  (Identical structure, new random splits)")
    

    
    # Get rough estimate of samples per fold
    avg_validation_size = n_samples * 0.8 / k_inner
    
    # Only show CV guidance when there are potential issues
    warnings_exist = (avg_validation_size < 50 or avg_validation_size > 1000 or k_outer > 10 or n_repeats > 5)
    
    # Only add guidance section if we have warnings
    if warnings_exist:
        print()
        print("CV CONFIGURATION GUIDANCE:")
        print("================================================================================")
    
    if avg_validation_size < 50:
        print(f"  • WARNING: Very small validation folds (<50 samples) may lead to unstable hyperparameter tuning")
        print(f"  • Consider reducing the inner folds (--inner {max(2, k_inner-1)}) or using more data")
    elif avg_validation_size > 1000:
        print(f"  • NOTE: Large validation folds (>1000 samples) provide stable evaluation but slow down training")
        print(f"  • Consider increasing inner folds for more efficient training if compute allows")
    
    if k_outer > 10:
        print(f"  • NOTE: Using {k_outer} outer folds may be computationally expensive")
        print(f"  • Consider reducing to 5-10 outer folds for faster overall training")
    
    if n_repeats > 5:
        print(f"  • NOTE: {n_repeats} repeats provides robust evaluation but multiplies training time")
        print(f"  • 3-5 repeats is typically sufficient for most applications")
    
    print()
    print("Next steps:")
    print("===========================================================")
    print("Run DAFTAR-ML with the optimized CV parameters:")
    
    # Conditionally include output_dir if specified
    cmd = f"  daftar --input {args.input} --target {args.target} --id {args.id} --outer {k_outer} --inner {k_inner} --repeats {n_repeats} --model [xgb|rf]"
    if args.output_dir:
        cmd += f" --output_dir {args.output_dir}"
    print(cmd)
    
    print("===========================================================\n")

if __name__ == "__main__":
    main()
