#!/usr/bin/env python3
"""
Fast preprocess script with progress bar for DAFTAR-ML
"""

import sys
import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import Parallel, delayed
from tqdm import tqdm


def main():
    """Preprocess a dataset for DAFTAR-ML with progress bar."""
    parser = argparse.ArgumentParser(description="Preprocess a dataset for DAFTAR-ML", allow_abbrev=False)
    
    # Required parameters
    required_args = parser.add_argument_group('Required parameters')
    required_args.add_argument("--input", required=True, help="Path to input CSV file containing features and target")
    required_args.add_argument("--target", required=True, help="Name of the target column to predict")
    required_args.add_argument("--id", required=True, help="Name of the ID column (e.g., species identifiers, sample names)")
    
    # Output configuration
    output_args = parser.add_argument_group('Output configuration')
    output_args.add_argument("--output_dir", help="Directory where output files will be saved. If not specified, files will be saved in the input file's directory")
    output_args.add_argument("--force", action="store_true", help="Force overwrite if output file already exists")
    output_args.add_argument("--no_report", action="store_true", help="Skip generating the detailed text report and feature importance CSV file")
    output_args.add_argument("--quiet", action="store_true", help="Suppress detailed console output during processing")
    
    # Analysis configuration
    analysis_args = parser.add_argument_group('Analysis configuration')
    analysis_args.add_argument("--task", choices=["regression", "classification"], 
                        help="Problem type: 'regression' for continuous targets, 'classification' for binary targets (optional - will be auto-detected if not specified)")
    analysis_args.add_argument("--k", type=int, default=500, 
                        help="Number of top features to select based on mutual information scores (default: 500)")
    
    # Transformation options
    transform_args = parser.add_argument_group('Data transformations')
    transform_args.add_argument("--trans_feat", type=str, choices=["log1p", "standard", "minmax"],
                        help="Feature transformation: 'log1p' for log(x+1), 'standard' for z-scores, 'minmax' for [0,1] scaling")
    transform_args.add_argument("--trans_target", type=str, choices=["log1p", "standard", "minmax"],
                        help="Target transformation (regression only): 'log1p', 'standard' (z-scores), or 'minmax' ([0,1] scaling)")
    
    # Processing options
    process_args = parser.add_argument_group('Processing options')
    process_args.add_argument("--jobs", type=int, default=-1, 
                        help="Number of parallel jobs for feature selection. -1 uses all CPU cores (default: -1)")
    process_args.add_argument("--keep_na", action="store_true", 
                        help="Keep rows with missing values. By default, rows with NaN are removed")
    process_args.add_argument("--keep_constant", action="store_true", 
                        help="Keep features with zero variance. By default, constant features are removed")
    process_args.add_argument("--no_rename", action="store_true", 
                        help="Do not rename duplicate column names. By default, duplicates are renamed with numeric suffixes")
    process_args.add_argument("--keep_zero_mi", action="store_true",
                        help="Keep features with zero mutual information with target (not recommended)")
    
    # Add an epilog with examples section
    parser.epilog = """
EXAMPLES:

  Default usage (auto-detects task type):
    python preprocess.py --input data.csv --target TARGET --id ID --output_dir OUTPUT_DIRECTORY

  Select top 200 features with log transformation:
    python preprocess.py --input data.csv --target TARGET --id ID --k 200 --trans_feat log1p --output_dir OUTPUT_DIRECTORY
"""
    # Set formatter to preserve formatting
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    
    # Update the usage message
    parser.usage = parser.format_usage().replace("usage: ", "Usage: ")
    parser._optionals.title = "Other optional arguments"
    parser._actions[0].help = "Show this help message and exit"
    
    args = parser.parse_args()
    
    try:
        # First, determine output file path to check if it exists
        input_path = Path(args.input)
        
        # Load the data for task detection
        print(f"Loading data from {args.input}...")
        data = pd.read_csv(args.input)
        print(f"Total samples: {len(data)}")
        print(f"Total columns: {data.shape[1]}")
        
        # Auto-detect task type if not specified
        if not args.task:
            target_data = data[args.target].dropna()
            unique_values = target_data.unique()
            
            # If only 2 unique values or boolean values, it's classification
            if len(unique_values) <= 2 or target_data.dtype == bool:
                args.task = "classification"
                print(f"\nBinary data detected in '{args.target}' column - performing CLASSIFICATION analysis")
            else:
                # If more than 2 unique values and numeric, it's likely regression
                if pd.api.types.is_numeric_dtype(target_data):
                    args.task = "regression"
                    print(f"\nContinuous data detected in '{args.target}' column - performing REGRESSION analysis")
                else:
                    # If more than 2 unique values but not numeric, default to classification with a warning
                    args.task = "classification"
                    print(f"\n[WARNING] Unable to auto-detect task type. Defaulting to CLASSIFICATION.")
                    print(f"   Consider specifying --task explicitly if this is incorrect.")
        else:
            # User specified task type explicitly
            if args.task == "regression":
                print(f"\n[USER-SPECIFIED] Performing REGRESSION analysis")
            else:
                print(f"\n[USER-SPECIFIED] Performing CLASSIFICATION analysis")
        
        # Generate filename based on input and parameters
        transform_suffix = ""
        if args.trans_feat:
            transform_suffix += f"_{args.trans_feat}_x"
        if args.trans_target:
            transform_suffix += f"_{args.trans_target}_y"
            
        # Create filename with the task type
        # Use shorter classification abbreviation
        task_abbr = "classif" if args.task == "classification" else args.task
        filename = f"{input_path.stem}_MI{args.k}_{task_abbr}{transform_suffix}{input_path.suffix}"
        
        # Determine output directory
        if args.output_dir:
            # User provided an output directory
            output_dir = Path(args.output_dir)
            # Create directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Use input file's directory
            output_dir = input_path.parent
            
        output_path = output_dir / filename
        
        # Check if the output file already exists - do this early to avoid unnecessary computation
        if output_path.exists() and not args.force:
            print(f"\nERROR: Output file already exists: {output_path}")
            print(f"To overwrite the existing file, add the --force flag:")
            
            # Build command without including None values and optional parameters
            cmd = f"python preprocess.py --input {args.input}"
            if args.output_dir:
                cmd += f" --output_dir {args.output_dir}"
            cmd += f" --target {args.target} --id {args.id}"
            # Don't include task since it's auto-detected
            cmd += f" --force"
            print(cmd)
            return 1
            
        # Data was already loaded for task detection
        # Continue with preprocessing
        
        # 2. Define target and features
        target_column = args.target
        id_column = args.id
        
        # Drop rows with NaN values if requested
        if not args.keep_na:
            nan_count = data.isna().sum().sum()
            print(f"Total NaNs in the dataset before dropping: {nan_count}")
            data = data.dropna()
            print(f"Total samples after dropping NaNs: {data.shape[0]}")
        
        # Extract target and features
        y = data[target_column].values
        feature_columns = [col for col in data.columns if col not in [id_column, target_column]]
        X = data[feature_columns].values
        print(f"Features shape: {X.shape}")
        
        # 3. Handle constant features if requested
        if not args.keep_constant:
            # Find non-constant columns
            non_constant_cols = []
            constant_cols = []
            for i, col in enumerate(feature_columns):
                if np.std(X[:, i]) > 1e-6:  # Not constant
                    non_constant_cols.append(i)
                else:
                    constant_cols.append(col)
            
            if constant_cols:
                print(f"Dropping {len(constant_cols)} constant features")
                X = X[:, non_constant_cols]
                feature_columns = [feature_columns[i] for i in non_constant_cols]
        
        # 4. Define mutual information computation based on task
        if args.task == 'classification':
            # Check if target is actually binary/discrete
            unique_values = np.unique(y)
            if len(unique_values) > 10 or np.issubdtype(y.dtype, np.floating):
                raise ValueError(
                    f"Error: Your target variable '{args.target}' appears to be continuous (found {len(unique_values)} unique values). "
                    f"For classification tasks, the target should contain discrete classes (e.g., 0/1 or True/False). "
                    f"If your target is continuous, use --task regression instead."
                )
            from sklearn.feature_selection import mutual_info_classif
            mi_func = mutual_info_classif
            print(f"Using mutual_info_classif for classification analysis")
        else:  # args.task == 'regression'
            from sklearn.feature_selection import mutual_info_regression
            mi_func = mutual_info_regression
            print(f"Using mutual_info_regression for regression analysis")
            
        def compute_mi(feature):
            return mi_func(feature.reshape(-1, 1), y, discrete_features=False)[0]
        
        # 5. Compute MI scores with progress bar
        n_jobs = args.jobs
        if n_jobs == -1:
            from multiprocessing import cpu_count
            n_cores = cpu_count()
            print(f"Computing Mutual Information using all available cores ({n_cores} cores)...")
        else:
            print(f"Computing Mutual Information using {n_jobs} cores...")
        try:
            mi_scores = Parallel(n_jobs=n_jobs)(
                delayed(compute_mi)(X[:, i]) for i in tqdm(range(X.shape[1]), desc="Computing Mutual Information")
            )
        except Exception as e:
            # Clear the progress bar before raising the error
            print("\r" + " " * 120 + "\r", end="")
            raise e
        mi_scores = np.array(mi_scores)
        
        # Create a DataFrame with all features and their MI scores for reporting
        all_features_mi = pd.DataFrame({
            'feature': feature_columns,
            'mutual_information': mi_scores
        }).sort_values('mutual_information', ascending=False).reset_index(drop=True)
        
        # 6. Select top k features
        k = min(args.k, len(feature_columns))
        top_k_indices = np.argsort(mi_scores)[-k:][::-1]  # Descending order
        top_k_features = [feature_columns[i] for i in top_k_indices]
        print(f"Top {k} features selected based on mutual information.")
        
        # Check for zero mutual information in selected features
        zero_mi_indices = [i for i in top_k_indices if mi_scores[i] == 0]
        zero_mi_features = [feature_columns[i] for i in zero_mi_indices]
        
        if zero_mi_features:
            print(f"\nWARNING: {len(zero_mi_features)} selected features have zero mutual information with the target!")
            print("Example features with zero MI:")
            for feature in zero_mi_features[:5]:  # Show just 5 examples
                print(f"  {feature}")
            if len(zero_mi_features) > 5:
                print(f"  ... and {len(zero_mi_features) - 5} more")
                
            if not args.keep_zero_mi:
                print(f"\nRemoving {len(zero_mi_features)} features with zero mutual information...")
                print("To keep these features (not recommended), run:")
                print(f"python preprocess.py --input {args.input} --target {args.target} --id {args.id} --task {args.task} --output_dir {args.output_dir} --k {args.k} --keep_zero_mi")
                
                # Remove zero MI features
                non_zero_indices = [i for i in top_k_indices if i not in zero_mi_indices]
                top_k_indices = non_zero_indices
                top_k_features = [feature_columns[i] for i in top_k_indices]
                print(f"\nFeature selection updated: {len(top_k_features)} features retained")
            else:
                print("\nKeeping features with zero mutual information (not recommended)")
        
        # Add 'selected' column to the MI DataFrame for reporting
        all_features_mi['selected'] = all_features_mi['feature'].isin(top_k_features)
        
        # 7. Create reduced dataset
        X_reduced = X[:, top_k_indices]
        
        # Print final stats
        print(f"\nFinal dataset statistics:")
        print(f"* Original features: {len(feature_columns)}")
        print(f"* Features after removing constant: {X.shape[1]}")
        print(f"* Features after MI selection: {len(top_k_features)}")
        if zero_mi_features and not args.keep_zero_mi:
            print(f"* Features removed due to zero MI: {len(zero_mi_features)}")
        print(f"* Final shape: {X_reduced.shape[0]} samples, {X_reduced.shape[1]} features")
        print()
        reduced_data = pd.DataFrame(X_reduced, columns=top_k_features)
        
        # 8. Apply transformations if requested
        if args.trans_feat:
            print(f"Applying {args.trans_feat} transformation to features...")
            # Apply transformations to the feature columns only
            feature_df = reduced_data[top_k_features].copy()
            
            if args.trans_feat == 'log1p':
                reduced_data[top_k_features] = np.log1p(feature_df)
                print("Log1p transformation applied to features")
            elif args.trans_feat == 'standard':
                scaler = StandardScaler()
                reduced_data[top_k_features] = scaler.fit_transform(feature_df)
                print("Standard scaling applied to features")
            elif args.trans_feat == 'minmax':
                scaler = MinMaxScaler()
                reduced_data[top_k_features] = scaler.fit_transform(feature_df)
                print("Min-max scaling applied to features")
        
        # Apply target transformation if requested (regression only)
        if args.trans_target and args.task == 'regression':
            print(f"Applying {args.trans_target} transformation to target...")
            y_series = pd.Series(y.copy())
            
            if args.trans_target == 'log1p':
                y = np.log1p(y_series)
                print("Log1p transformation applied to target")
            elif args.trans_target == 'standard':
                y = StandardScaler().fit_transform(y_series.values.reshape(-1, 1)).ravel()
                print("Standard scaling applied to target")
            elif args.trans_target == 'minmax':
                y = MinMaxScaler().fit_transform(y_series.values.reshape(-1, 1)).ravel()
                print("Min-max scaling applied to target")
        
        # Add ID and target to the dataset
        reduced_data[id_column] = data[id_column].values
        reduced_data[target_column] = y
        
        # Reorder columns to have ID and target first
        reordered_columns = [id_column, target_column] + top_k_features
        reduced_data = reduced_data[reordered_columns]
        
        # 8. Save the reduced dataset
            
        reduced_data.to_csv(output_path, index=False)
        print(f"Reduced dataset saved to {output_path}")
        
        # Write report and feature importance files by default unless --no_report is used
        if not args.no_report:
            # Calculate base output path
            base_output_path = str(output_path).rsplit('.', 1)[0]
            
            # Generate the feature importance CSV file
            # Shorter but still descriptive file name
            # Use shorter classification abbreviation
            task_abbr = "classif" if args.task == "classification" else args.task
            mi_file_path = f"{input_path.stem}_MI{args.k}_{task_abbr}_feature_scores.csv"
            mi_file_full_path = os.path.join(output_dir if output_dir else ".", mi_file_path)
            all_features_mi.to_csv(mi_file_path, index=False)
            print(f"Feature importance rankings saved to {mi_file_full_path}")
            
            # Generate detailed report
            # Shorter but still descriptive report name
            # Use shorter classification abbreviation
            task_abbr = "classif" if args.task == "classification" else args.task
            report_path = f"{input_path.stem}_MI{args.k}_{task_abbr}_report.txt"
            report_full_path = os.path.join(output_dir if output_dir else ".", report_path)
            with open(report_path, 'w') as f:
                # Header
                f.write(f"# DAFTAR-ML Preprocessing Report\n\n")
                f.write(f"Generated: {pd.Timestamp.now()}\n\n")
                
                # Dataset info
                f.write(f"## Dataset Information\n")
                f.write(f"* Input file: {args.input}\n")
                f.write(f"* Output file: {output_path}\n")
                f.write(f"* Original shape: {data.shape[0]} samples, {data.shape[1]} features\n")
                f.write(f"* Final shape: {reduced_data.shape[0]} samples, {reduced_data.shape[1]} columns\n\n")
                
                # Processing steps
                f.write(f"## Processing Steps\n")
                if not args.keep_na:
                    f.write(f"* Dropped rows with missing values\n")
                if not args.keep_constant and constant_cols:
                    f.write(f"* Dropped {len(constant_cols)} constant features\n")
                f.write(f"* Selected top {k} features using mutual information\n")
                if args.trans_feat:
                    f.write(f"* Applied {args.trans_feat} transformation to features\n")
                if args.trans_target and args.task == 'regression':
                    f.write(f"* Applied {args.trans_target} transformation to target\n")
                f.write("\n")
                
                # Feature information
                f.write(f"## Feature Selection\n")
                f.write(f"* Selected {k} out of {len(feature_columns)} features\n\n")
                
                # Top features
                f.write(f"### Top 20 Selected Features\n")
                for i, (_, row) in enumerate(all_features_mi[all_features_mi['selected']].head(20).iterrows()):
                    f.write(f"{i+1}. {row['feature']}: {row['mutual_information']:.6f}\n")
                f.write("\n")
                
                # Alert for features with zero mutual information
                zero_mi_features = all_features_mi[all_features_mi['selected'] & (all_features_mi['mutual_information'] == 0)]
                if not zero_mi_features.empty:
                    f.write(f"### WARNING: Features with Zero Mutual Information\n")
                    f.write(f"The following {len(zero_mi_features)} selected features have zero mutual information with the target:\n\n")
                    for i, (_, row) in enumerate(zero_mi_features.head(20).iterrows()):
                        f.write(f"{i+1}. {row['feature']}\n")
                    if len(zero_mi_features) > 20:
                        f.write(f"...and {len(zero_mi_features) - 20} more features\n")
                    f.write("\nThese features provide no information about the target variable and should be considered for removal.\n\n")
                
                # Next steps
                f.write(f"## Next Steps\n")
                
                f.write(f"### Option 1: Visualize CV splits and optimize them for data structure\n")
                f.write(f"**Default CV splits:**\n")
                # Basic CV calculator command
                cv_cmd1 = f"python cv_calculator.py --input {output_path} --id {args.id} --target {args.target}"
                if args.output_dir:
                    cv_cmd1 += f" --output_dir {args.output_dir}"
                f.write(f"{cv_cmd1}\n\n")
                
                f.write(f"**Custom CV parameters:**\n")
                # Custom CV parameters
                cv_cmd2 = f"python cv_calculator.py --input {output_path} --id {args.id} --target {args.target} --outer INTEGER --inner INTEGER --repeats INTEGER"
                if args.output_dir:
                    cv_cmd2 += f" --output_dir {args.output_dir}"
                f.write(f"{cv_cmd2}\n\n")
                
                f.write(f"### Option 2: Run the full DAFTAR-ML pipeline\n")
                # Run DAFTAR-ML command
                run_cmd = f"python run_daftar.py --input {output_path} --target {args.target} --id {args.id} --model [xgb|rf]"
                if args.output_dir:
                    run_cmd += f" --output_dir {args.output_dir}"
                f.write(f"{run_cmd}\n")
                

            
            print(f"Detailed preprocessing report saved to {report_full_path}")
        
        print(f"\nPreprocessing completed successfully!\n")
        
        # Print next steps
        print("Next steps:")
        print("===========================================================")
        print("Option 1: Visualize CV splits and optimize them for data structure")
        print(f"  Default CV splits : ")
        print(f"  daftar-cv --input {output_path} --id {args.id} --target {args.target} --output_dir {args.output_dir}")
        print()
        print(f"  Custom CV parameters:")
        print(f"  daftar-cv --input {output_path} --id {args.id} --target {args.target} --outer INTEGER --inner INTEGER --repeats INTEGER --output_dir {args.output_dir}")
        print()
        print("Option 2: Run the DAFTAR-ML pipeline with default settings")
        print(f"  daftar --input {output_path} --target {args.target} --id {args.id} --model [xgb|rf] --output_dir {args.output_dir}")
        print("===========================================================\n")
        
        return 0
    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
