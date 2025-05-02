"""Command-line interface for DAFTAR-ML.

This module provides the CLI entry point for the DAFTAR-ML pipeline.
Preserves the original behavior and argument structure of the DAFTAR-ML CLI.
"""

import argparse
import os
import sys
import yaml
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from daftar.core.config import Config
from daftar.core.pipeline import Pipeline
from daftar.utils.warnings import suppress_xgboost_warnings

# Suppress XGBoost warnings globally
suppress_xgboost_warnings()


def load_config_from_yaml(filepath: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        Dictionary with configuration values
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def parse_args(args: Optional[List[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    """Parse command line arguments.
    
    Args:
        args: Command line arguments (uses sys.argv if None)
        
    Returns:
        Tuple of (parsed_args, remaining_args)
    """
    parser = argparse.ArgumentParser(description="DAFTAR-ML: Feature Importance via Repeated Annotation of Trees", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Customize help message to capitalize 'Show'
    parser._actions[0].help = "Show this help message and exit"
    
    # Required arguments
    required_args = parser.add_argument_group('Required arguments')
    
    required_args.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    
    required_args.add_argument(
        "--target",
        type=str,
        required=True,
        help="Name of target column to predict"
    )
    
    required_args.add_argument(
        "--id",
        type=str,
        required=True,
        help="Column with sample IDs (required)"
    )
    
    required_args.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["xgb", "rf"],
        help="Model type (xgb=XGBoost, rf=Random Forest)"
    )
    
    # Optional arguments
    optional_args = parser.add_argument_group('Optional arguments')
    
    out_args = parser.add_argument_group('Output arguments')
    
    out_args.add_argument(
        "--output_dir",
        type=str,
        help="Directory where output files will be saved"
    )
    
    # Cross-validation arguments
    cv_args = parser.add_argument_group('Cross-validation arguments')
    
    cv_args.add_argument(
        "--inner",
        type=int,
        default=3,
        help="Number of inner CV folds"
    )
    
    cv_args.add_argument(
        "--outer",
        type=int,
        default=5,
        help="Number of outer CV folds"
    )
    
    optional_args.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    
    optional_args.add_argument(
        "--task",
        type=str,
        choices=["regression", "classification"],
        help="Type of problem (regression or classification). Optional - will be auto-detected if not specified"
    )
    
    optional_args.add_argument(
        "--metric",
        type=str,
        help="Metric to optimize (regression: mse/rmse/mae/r2, classification: accuracy/f1/roc_auc)"
    )
    
    # Transformations now handled in preprocessing stage
    

    
    cv_args.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of CV repetitions"
    )
    
    # Optimization arguments
    opt_args = parser.add_argument_group('Optimization arguments')
    
    opt_args.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Number of trials to wait without improvement before early stopping (defaults to 50 for thorough optimization)"
    )
    
    opt_args.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Minimum improvement threshold for early stopping (defaults vary by metric: 1e-6 for MSE, 1e-4 for RMSE/MAE, 1e-3 for R² and classification metrics)"
    )
    
    # Visualization arguments
    viz_args = parser.add_argument_group('Visualization arguments')
    
    viz_args.add_argument(
        "--top_n",
        type=int,
        default=25,
        help="Number of top features to include in visualizations"
    )
    

    
    out_args.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory without asking"
    )
    
    # Execution arguments
    exec_args = parser.add_argument_group('Execution arguments')
    
    exec_args.add_argument(
        "--cores",
        type=int,
        default=-1,
        help="Number of CPU cores to use (-1 for all available)"
    )
    
    exec_args.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Parse arguments
    parsed_args, remaining = parser.parse_known_args(args)
        
    return parsed_args, remaining











def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for DAFTAR-ML CLI.
    
    Args:
        args: Command line arguments (uses sys.argv if None)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Capture the original command
    if args is None:
        import shlex
        cmd_args = ' '.join(shlex.quote(arg) for arg in sys.argv[1:])
        original_command = f"daftar {cmd_args}"
    else:
        cmd_args = ' '.join(str(arg) for arg in args)
        original_command = f"daftar {cmd_args}"
    
    parsed_args, remaining = parse_args(args)
    
    # Load config from YAML if provided
    if parsed_args.config:
        config_dict = load_config_from_yaml(parsed_args.config)
    else:
        config_dict = {}
    
    # Override with command line arguments
    # Check if any required args are missing
    required_args = ['input', 'target', 'model']
    for arg in required_args:
        arg_value = getattr(parsed_args, arg, None)
        if arg_value:
            config_dict[arg] = arg_value
        elif arg not in config_dict:
            print(f"ERROR: Required argument --{arg} is missing and not found in config file.")
            return 1
            
    # Handle task auto-detection if not specified
    if getattr(parsed_args, 'task', None):
        # User specified task
        config_dict['problem_type'] = parsed_args.task
    elif 'task' not in config_dict and 'problem_type' not in config_dict:
        # Auto-detect task type from data
        # No message needed - this is normal behavior
        try:
            import pandas as pd
            import os
            input_file = config_dict.get('input')
            target_col = config_dict.get('target')
            
            if not input_file or not target_col:
                print("ERROR: Cannot auto-detect task without input file and target column.")
                return 1
            
            # Check if file exists before attempting to read it
            if not os.path.exists(input_file):
                print(f"ERROR: Input file not found: {input_file}")
                print("Please check the file path and try again.")
                return 1
                
            try:
                data = pd.read_csv(input_file)
                
                # Check if target column exists in the data
                if target_col not in data.columns:
                    print(f"ERROR: Target column '{target_col}' not found in the input file.")
                    print(f"Available columns: {', '.join(data.columns)}")
                    return 1
                    
                target_data = data[target_col].dropna()
                unique_values = target_data.unique()
                
                # If only 2 unique values or boolean values, it's classification
                if len(unique_values) <= 2 or target_data.dtype == bool:
                    task_type = "classification"
                    print(f"Auto-detected CLASSIFICATION task")
                else:
                    # If more than 2 unique values and numeric, it's likely regression
                    if pd.api.types.is_numeric_dtype(target_data):
                        task_type = "regression"
                        print(f"Auto-detected REGRESSION task")
                    else:
                        # If more than 2 unique values but not numeric, default to classification with a warning
                        task_type = "classification"
                        print(f"Auto-detection unclear - defaulting to CLASSIFICATION task")
                        print(f"   Use --task regression|classification to specify if needed")
                
                config_dict['problem_type'] = task_type
                
            except pd.errors.EmptyDataError:
                print(f"ERROR: The input file is empty: {input_file}")
                return 1
            except pd.errors.ParserError:
                print(f"ERROR: Unable to parse the input file: {input_file}")
                print("Please ensure it's a valid CSV file.")
                return 1
                
        except Exception as e:
            print(f"ERROR: An unexpected error occurred: {str(e)}")
            print("Please specify --task [regression|classification] explicitly.")
            return 1
    
    # Handle optional arguments
    for arg, value in vars(parsed_args).items():
        if arg not in ['config'] and value is not None and arg not in required_args:
            # Skip None values to allow defaults from Config to apply
            if arg == 'threshold':
                config_dict['relative_threshold'] = value
            elif arg == 'force':
                config_dict['force_overwrite'] = value
            else:
                config_dict[arg] = value
    
    # Map CLI parameter names to Config field names
    parameter_mapping = {
        'input': 'input_file',
        'id': 'id_column',
        'task': 'problem_type',
        'inner': 'inner_folds',
        'outer': 'outer_folds',
        'repeats': 'repeats',
        'cores': 'cores',
        'seed': 'seed',
        'top_n': 'top_n',
    }
    
    # Apply all the mappings
    for cli_param, config_param in parameter_mapping.items():
        if cli_param in config_dict:
            config_dict[config_param] = config_dict.pop(cli_param)
    
    # Create config
    try:
        config = Config(**config_dict)
        # Add command to config for logging
        config.original_command = original_command
    except Exception as e:
        print(f"ERROR: Failed to create configuration: {str(e)}")
        return 1
    
    # Create and run pipeline
    try:
        pipeline = Pipeline(config)
        pipeline.run()
    except Exception as e:
        print(f"ERROR: Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Get the output directory path
    output_dir = config.get_output_dir()
    
    print(f"DAFTAR-ML analysis completed successfully.")
    print(f"Results saved to: {output_dir}")
    
    # Print example command for reproduction with ALL parameters used
    print("\nTo reproduce this run in the future, use:")
    cmd = f"daftar --input {config.input_file} --target {config.target} "
    
    # Only include --task if it was explicitly provided
    if getattr(parsed_args, 'task', None):
        cmd += f"--task {config.problem_type} "
        
    cmd += f"--model {config.model}"
    
    # Add all non-default parameters that were explicitly provided
    if hasattr(config, 'outer_folds') and config.outer_folds is not None:
        cmd += f" --outer {config.outer_folds}"
    if hasattr(config, 'inner_folds') and config.inner_folds is not None:
        cmd += f" --inner {config.inner_folds}"
    if hasattr(config, 'repeats') and config.repeats is not None:
        cmd += f" --repeats {config.repeats}"
    if hasattr(config, 'cores') and config.cores is not None and config.cores != -1:
        cmd += f" --cores {config.cores}"
    if hasattr(config, 'metric') and config.metric is not None:
        cmd += f" --metric {config.metric}"
    # trials parameter is not exposed in CLI, so don't include it in reproduction command
    if hasattr(config, 'patience') and config.patience is not None:
        cmd += f" --patience {config.patience}"
    if hasattr(config, 'relative_threshold') and config.relative_threshold != 1e-6:
        cmd += f" --threshold {config.relative_threshold}"
    if hasattr(config, 'id_column') and config.id_column is not None and config.id_column != "ID":
        cmd += f" --id {config.id_column}"
    if hasattr(config, 'seed') and config.seed is not None:
        cmd += f" --seed {config.seed}"
    
    print(cmd)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
