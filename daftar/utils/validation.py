"""Data validation utilities for DAFTAR-ML.

This module provides utilities for checking data validity:
- Duplicate column detection
- Missing value detection
- Data type validation
- Feature distribution analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from pathlib import Path


def check_duplicate_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Check for duplicate columns in the dataset.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary mapping canonical column names to lists of duplicate column names
    """
    # Initialize dictionary to store duplicate columns
    duplicates = {}
    
    # Get column names and data
    columns = df.columns
    data = df.values
    
    # Check each pair of columns
    for i in range(len(columns)):
        col_i = columns[i]
        
        # Skip if column is already known to be a duplicate
        if any(col_i in dups for dups in duplicates.values()):
            continue
        
        dups = []
        
        for j in range(i + 1, len(columns)):
            col_j = columns[j]
            
            # Skip if column is already known to be a duplicate
            if any(col_j in dups for dups in duplicates.values()):
                continue
            
            # Check if columns are identical
            if np.array_equal(data[:, i], data[:, j]):
                dups.append(col_j)
        
        # Store duplicates if any
        if dups:
            duplicates[col_i] = dups
    
    return duplicates


def check_missing_values(df: pd.DataFrame) -> Dict[str, float]:
    """Check for missing values in the dataset.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary mapping column names to percentage of missing values
    """
    # Calculate percentage of missing values for each column
    missing = {}
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct > 0:
            missing[col] = pct
    
    return missing


def check_constant_features(df: pd.DataFrame) -> List[str]:
    """Check for constant features in the dataset.
    
    Args:
        df: Input dataframe
        
    Returns:
        List of constant feature names
    """
    # Find columns with only one unique value
    constant = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant.append(col)
    
    return constant


def check_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Check data types of columns in the dataset.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary mapping column names to data types
    """
    # Get data types for each column
    types = {}
    for col in df.columns:
        dtype = df[col].dtype
        
        # Map numpy/pandas types to simpler types
        if np.issubdtype(dtype, np.integer):
            types[col] = 'integer'
        elif np.issubdtype(dtype, np.floating):
            types[col] = 'float'
        elif np.issubdtype(dtype, np.bool_):
            types[col] = 'boolean'
        elif np.issubdtype(dtype, np.datetime64):
            types[col] = 'datetime'
        elif np.issubdtype(dtype, np.object_):
            # Check if string or categorical
            if df[col].nunique() < len(df[col]) * 0.5:
                types[col] = 'categorical'
            else:
                types[col] = 'string'
        else:
            types[col] = str(dtype)
    
    return types


def analyze_feature_distribution(
    df: pd.DataFrame, output_dir: Optional[Path] = None, 
    max_features: int = 20
) -> None:
    """Analyze the distribution of features in the dataset.
    
    Args:
        df: Input dataframe
        output_dir: Directory to save distribution plots
        max_features: Maximum number of features to analyze
    """
    # Create output directory if provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get feature data types
    types = check_data_types(df)
    
    # Select subset of features if there are too many
    if len(df.columns) > max_features:
        selected_cols = list(df.columns[:max_features])
    else:
        selected_cols = list(df.columns)
    
    # Analyze numeric features
    numeric_cols = [col for col in selected_cols if types[col] in ['integer', 'float']]
    if numeric_cols:
        plt.figure(figsize=(12, 8))
        
        for i, col in enumerate(numeric_cols):
            plt.subplot(len(numeric_cols), 1, i+1)
            plt.hist(df[col].dropna(), bins=50)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir / "numeric_distributions.png")
            plt.close()
        else:
            plt.show()
    
    # Analyze categorical features
    cat_cols = [col for col in selected_cols if types[col] in ['categorical', 'boolean']]
    for col in cat_cols:
        plt.figure(figsize=(10, 6))
        
        # Get value counts
        counts = df[col].value_counts().sort_values(ascending=False)
        
        # Limit number of categories to show
        if len(counts) > 15:
            other_count = counts[15:].sum()
            counts = counts[:15]
            counts['Other'] = other_count
        
        counts.plot(kind='bar')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        
        if output_dir:
            plt.savefig(output_dir / f"categorical_{col}.png")
            plt.close()
        else:
            plt.show()


def validate_dataset(
    df: pd.DataFrame, target: Optional[str] = None,
    id_column: Optional[str] = None, output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Validate the dataset and generate a report.
    
    Args:
        df: Input dataframe
        target: Name of target column
        id_column: Column with sample IDs
        output_dir: Directory to save validation results
        
    Returns:
        Dictionary with validation results
    """
    # Initialize results
    results = {
        'dataset_shape': df.shape,
        'n_samples': len(df),
        'n_features': len(df.columns),
        'data_types': check_data_types(df),
        'duplicates': check_duplicate_columns(df),
        'missing_values': check_missing_values(df),
        'constant_features': check_constant_features(df)
    }
    
    # Check target column if provided
    if target:
        if target not in df.columns:
            results['target_found'] = False
            results['target_error'] = f"Target column '{target}' not found in data"
        else:
            results['target_found'] = True
            results['target_type'] = results['data_types'][target]
            
            # Check if regression or classification based on target
            if results['target_type'] in ['categorical', 'boolean']:
                results['problem_type'] = 'classification'
                results['n_classes'] = df[target].nunique()
                results['class_balance'] = df[target].value_counts(normalize=True).to_dict()
            else:
                results['problem_type'] = 'regression'
                results['target_stats'] = {
                    'min': float(df[target].min()),
                    'max': float(df[target].max()),
                    'mean': float(df[target].mean()),
                    'median': float(df[target].median()),
                    'std': float(df[target].std())
                }
    
    # Check ID column if provided
    if id_column:
        if id_column not in df.columns:
            results['id_found'] = False
            results['id_error'] = f"ID column '{id_column}' not found in data"
        else:
            results['id_found'] = True
            results['id_unique'] = df[id_column].nunique() == len(df)
    
    # Generate feature distribution analysis if output directory is provided
    if output_dir:
        try:
            analyze_feature_distribution(df, output_dir)
            results['distribution_plots'] = True
        except Exception as e:
            results['distribution_plots'] = False
            results['distribution_error'] = str(e)
    
    return results


def print_validation_report(results: Dict[str, Any]) -> None:
    """Print a validation report.
    
    Args:
        results: Validation results dictionary
    """
    print("=== Dataset Validation Report ===")
    print(f"Dataset shape: {results['n_samples']} samples, {results['n_features']} features")
    
    # Target information
    if 'target_found' in results:
        if results['target_found']:
            print(f"\nTarget column:")
            print(f"  Type: {results['target_type']}")
            print(f"  Problem type: {results['problem_type']}")
            
            if results['problem_type'] == 'classification':
                print(f"  Number of classes: {results['n_classes']}")
                print("  Class balance:")
                for cls, pct in results['class_balance'].items():
                    print(f"    {cls}: {pct:.2%}")
            else:
                print("  Target statistics:")
                for stat, val in results['target_stats'].items():
                    print(f"    {stat}: {val}")
        else:
            print(f"\nTarget column: {results['target_error']}")
    
    # ID column information
    if 'id_found' in results:
        if results['id_found']:
            print(f"\nID column: {'Unique values' if results['id_unique'] else 'DUPLICATE VALUES FOUND'}")
        else:
            print(f"\nID column: {results['id_error']}")
    
    # Data quality issues
    print("\nData quality issues:")
    
    # Duplicate columns
    if results['duplicates']:
        print("  Duplicate columns found:")
        for canonical, dups in results['duplicates'].items():
            print(f"    {canonical}: {', '.join(dups)}")
    else:
        print("  No duplicate columns found")
    
    # Missing values
    if results['missing_values']:
        print("  Missing values found:")
        for col, pct in results['missing_values'].items():
            print(f"    {col}: {pct:.2f}%")
    else:
        print("  No missing values found")
    
    # Constant features
    if results['constant_features']:
        print(f"  Constant features found: {', '.join(results['constant_features'])}")
    else:
        print("  No constant features found")
    
    # Distribution plots
    if 'distribution_plots' in results:
        if results['distribution_plots']:
            print("\nFeature distribution plots generated")
        else:
            print(f"\nError generating distribution plots: {results['distribution_error']}")
