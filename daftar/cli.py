"""Command-line interface for DAFTAR-ML.

This module provides the CLI entry point for the DAFTAR-ML pipeline.
Preserves the original behavior and argument structure of the DAFTAR-ML CLI.
"""

import argparse
import os
import sys
import yaml
import warnings
import signal
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from daftar.core.config import Config
from daftar.core.pipeline import Pipeline
from daftar.utils.warnings import suppress_xgboost_warnings
from daftar.models.hyperparams import clear_hyperparams_cache
import daftar  # For version information

# Clear hyperparameters cache on startup to ensure fresh config
clear_hyperparams_cache()

# Suppress XGBoost warnings globally
suppress_xgboost_warnings()

# Terminal color codes
GREEN = '\033[32m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
PINK = '\033[95m'
BRIGHT_GREEN = '\033[92;1m'
BOLD = '\033[1m'
RESET = '\033[0m'


def signal_handler(sig, frame):
    """Handle keyboard interrupt (Ctrl+C) gracefully."""
    print(f"\n{YELLOW}Pipeline interrupted by user (Ctrl+C){RESET}")
    print("Cleaning up and exiting...")
    sys.exit(0)


# Set up signal handler for keyboard interrupt
signal.signal(signal.SIGINT, signal_handler)

# Custom formatter that doesn't show defaults for required arguments
class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        # Don't add default text for required arguments
        if action.required:
            return action.help
        return super()._get_help_string(action)


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
        formatter_class=CustomHelpFormatter,
        usage="daftar --input PATH --target COLUMN --sample COLUMN --model {xgb,rf} [--output_dir PATH] [--inner INTEGER] [--outer INTEGER] [--repeats INTEGER] [--stratify {true,false}] [--config PATH] [--task_type {regression,classification}] [--metric {mse,rmse,mae,r2,accuracy,f1,roc_auc}] [--patience INTEGER] [--threshold FLOAT] [--top_n INTEGER] [--force] [--verbose] [--cores INTEGER] [--seed INTEGER]"
    )
    
    # Customize help message to capitalize 'Show'
    parser._actions[0].help = "Show this help message and exit"
    
    # Required arguments
    required_args = parser.add_argument_group('Required arguments')
    
    required_args.add_argument(
        "--input",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to preprocessed .csv file"
    )
    
    required_args.add_argument(
        "--target",
        type=str,
        required=True,
        metavar="COLUMN",
        help="Name of target column to predict"
    )
    
    required_args.add_argument(
        "--sample",
        type=str,
        required=True,
        help="Name of the column containing sample identifiers"
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
        metavar="PATH",
        help="Directory where output files will be saved"
    )
    
    # Cross-validation arguments
    cv_args = parser.add_argument_group('Cross-validation arguments')
    
    cv_args.add_argument(
        "--inner",
        type=int,
        default=3,
        metavar="INTEGER",
        help="Number of inner CV folds"
    )
    
    cv_args.add_argument(
        "--outer",
        type=int,
        default=5,
        metavar="INTEGER",
        help="Number of outer CV folds"
    )
    
    cv_args.add_argument(
        "--repeats",
        type=int,
        default=3,
        metavar="INTEGER",
        help="Number of CV repetitions"
    )
    
    optional_args.add_argument(
        "--stratify",
        type=str,
        choices=["true", "false"],
        help="Whether to use stratified splitting for classification tasks (default: true for classification, false for regression)"
    )
    
    optional_args.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        help="Path to YAML configuration file"
    )
    
    optional_args.add_argument(
        "--task_type",
        type=str,
        choices=["regression", "classification"],
        help="Type of problem (regression or classification). Optional - will be auto-detected if not specified"
    )
    
    optional_args.add_argument(
        "--metric",
        type=str,
        choices=["mse", "rmse", "mae", "r2", "accuracy", "f1", "roc_auc"],
        help="Metric to optimize (regression: {mse|rmse|mae|r2}, classification: {accuracy|f1|roc_auc})"
    )
        
    # Optimization arguments
    opt_args = parser.add_argument_group('Optimization arguments')
    
    opt_args.add_argument(
        "--patience",
        type=int,
        default=50,
        metavar="INTEGER",
        help="Number of trials to wait without improvement before early stopping"
    )
    
    opt_args.add_argument(
        "--threshold",
        type=float,
        default=None,  # No default so we can detect explicit usage
        metavar="FLOAT",
        help="Minimum improvement threshold for early stopping"
    )
    
    # Visualization arguments
    viz_args = parser.add_argument_group('Visualization arguments')
    
    viz_args.add_argument(
        "--top_n",
        type=int,
        default=15,
        metavar="INTEGER",
        help="Number of top features to include in visualizations"
    )
    
    viz_args.add_argument(
        "--skip_interaction",
        action="store_true",
        help="Skip SHAP interaction calculations (faster execution)"
    )
    
    out_args.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory without asking"
    )
    
    out_args.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output showing all details in the console"
    )
    
    # Execution arguments
    exec_args = parser.add_argument_group('Execution arguments')
    
    exec_args.add_argument(
        "--cores",
        type=int,
        default=-1,
        metavar="INTEGER",
        help="Number of CPU cores to use (-1 for all available)"
    )
    
    exec_args.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INTEGER",
        help="Random seed"
    )
    
    # Parse arguments
    parsed_args, remaining = parser.parse_known_args(args)
    
    # Check for unknown arguments and throw an error if any are found
    if remaining:
        unknown_args = " ".join(remaining)
        parser.error(f"Unrecognized arguments: {unknown_args}")
        
    return parsed_args, remaining

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for DAFTAR-ML CLI.
    
    Args:
        args: Command line arguments (uses sys.argv if None)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Display the ASCII art logo
    print("")
    print(f"""{GREEN}
    ██████╗  █████╗ ███████╗████████╗ █████╗ ██████╗       ███╗   ███╗██╗     
    ██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔══██╗      ████╗ ████║██║     
    ██║  ██║███████║█████╗     ██║   ███████║██████╔╝█████╗██╔████╔██║██║     
    ██║  ██║██╔══██║██╔══╝     ██║   ██╔══██║██╔══██╗╚════╝██║╚██╔╝██║██║     
    ███████╗██║  ██║██║        ██║   ██║  ██║██║  ██║      ██║ ╚═╝ ██║███████╗
    ╚══════╝╚═╝  ╚═╝╚═╝        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝      ╚═╝     ╚═╝╚══════╝{RESET}""")
    print("")  
    # Print the version below the banner
    print(f"{CYAN}Version {daftar.__version__}{RESET}")
    print("")
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
                
                # First check the data type
                if pd.api.types.is_numeric_dtype(target_data):
                    # For numeric data, check number of unique values
                    if len(unique_values) <= 2 or target_data.dtype == bool:
                        # Binary numeric values (like 0/1) suggest classification
                        task_type = "classification"
                    else:
                        # Multiple numeric values suggest regression
                        task_type = "regression"
                else:
                    # Non-numeric (string/categorical) data is always classification
                    # (regardless of number of unique values)
                    task_type = "classification"
                
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
            print("Please specify --task_type [regression|classification] explicitly.")
            return 1
    
    # Handle optional arguments
    for arg, value in vars(parsed_args).items():
        if arg not in ['config'] and value is not None and arg not in required_args:
            # Skip None values to allow defaults from Config to apply
            if arg == 'threshold':
                config_dict['relative_threshold'] = value
            elif arg == 'force':
                config_dict['force_overwrite'] = value
            elif arg == 'verbose':
                config_dict['verbose'] = value
            elif arg == 'stratify':
                config_dict['use_stratified'] = value == 'true'
            elif arg == 'skip_interaction':
                config_dict['skip_interaction'] = value
            else:
                config_dict[arg] = value
    
    # Map CLI parameter names to Config field names
    parameter_mapping = {
        'input': 'input_file',
        'sample': 'id_column',
        'task_type': 'problem_type',
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
    except FileExistsError as e:
        print(f"ERROR: {str(e)}")
        return 1
    except Exception as e:
        print(f"ERROR: Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Get the output directory path
    output_dir = config.get_output_dir()
    
    # Message already printed from pipeline
    print(f"Results saved to: {output_dir}")
    
    # Print separator before reproduction command
    print("\n" + CYAN + "=" * 80 + RESET)
    
    # Print example command for reproduction with ALL parameters used
    print("\nTo reproduce this run in the future, use:")
    print("\n")
    cmd = f"daftar --input {config.input_file}"


    # Add ID column if specified
    if hasattr(config, 'id_column') and config.id_column is not None:
        cmd += f" --sample {config.id_column}"
    
    # Add target column
    cmd += f" --target {config.target}"
    
    # Only include --task_type if it was explicitly provided
    if getattr(parsed_args, 'task_type', None):
        cmd += f" --task_type {config.problem_type}"
        
    cmd += f" --model {config.model}"
    
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
    if hasattr(config, 'patience') and config.patience is not None:
        cmd += f" --patience {config.patience}"
    # Only include threshold if it was explicitly provided by the user
    if hasattr(parsed_args, 'threshold') and parsed_args.threshold is not None:
        cmd += f" --threshold {config.relative_threshold}"
    if hasattr(config, 'top_n') and config.top_n != 15:
        cmd += f" --top_n {config.top_n}"
    # Only include stratify if it was explicitly set to true for regression or false for classification
    if hasattr(config, 'use_stratified') and hasattr(config, 'problem_type'):
        is_classification = config.problem_type == 'classification'
        # Default is to stratify for classification only
        default_stratification = is_classification
        if config.use_stratified != default_stratification:
            cmd += f" --stratify {'true' if config.use_stratified else 'false'}"
    
    # Add seed parameter - but only once
    if hasattr(parsed_args, 'seed') and parsed_args.seed is not None:
        cmd += f" --seed {parsed_args.seed}"
    
    # Include the output_dir if it was specified
    if hasattr(parsed_args, 'output_dir') and parsed_args.output_dir:
        cmd += f" --output_dir {parsed_args.output_dir}"
    
    print(cmd)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
