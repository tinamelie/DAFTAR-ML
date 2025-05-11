#!/usr/bin/env python
"""
Test script to verify that CV splits are the same when using the same seed,
both for regular and stratified cross-validation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
import argparse

def setup_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test CV split consistency with seeds")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--id", required=True, help="ID column name")
    parser.add_argument("--seed", type=int, default=42, help="Primary random seed")
    parser.add_argument("--seed2", type=int, default=None, help="Secondary random seed for comparison")
    parser.add_argument("--outer", type=int, default=5, help="Number of outer folds")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats")
    return parser.parse_args()

def test_cv_consistency(args):
    """Test if CV splits are the same when using the same seed."""
    print(f"Loading data from {args.input}...")
    data = pd.read_csv(args.input)
    print(f"Loaded dataset with {len(data)} samples and {len(data.columns)} columns")
    
    # Extract target and IDs
    y = data[args.target].values
    ids = data[args.id].values
    
    # Create dummy feature matrix
    X = np.zeros((len(data), 1))
    
    # Determine if target is continuous or categorical
    unique_values = np.unique(y)
    is_categorical = len(unique_values) < 10  # Arbitrary threshold
    
    print(f"Target variable type: {'Categorical' if is_categorical else 'Continuous'}")
    
    print("\n1. Testing Regular K-Fold splits with same seed...")
    # Create two identical CV splitters with the same seed
    cv1 = RepeatedKFold(n_splits=args.outer, n_repeats=args.repeats, random_state=args.seed)
    cv2 = RepeatedKFold(n_splits=args.outer, n_repeats=args.repeats, random_state=args.seed)
    
    # Get splits from both splitters
    splits1 = list(cv1.split(X))
    splits2 = list(cv2.split(X))
    
    # Check if they're identical
    kfold_identical = True
    for i, ((train1, test1), (train2, test2)) in enumerate(zip(splits1, splits2)):
        if not np.array_equal(test1, test2) or not np.array_equal(train1, train2):
            kfold_identical = False
            print(f"Split {i} differs!")
            break
    
    if kfold_identical:
        print("✅ SUCCESS: Regular KFold splits are identical with the same seed!")
    else:
        print("❌ FAILURE: Regular KFold splits differ with the same seed!")
    
    # Only do stratified fold validation for categorical targets
    stratified_splits1 = None
    if is_categorical:
        print("\n2. Testing Stratified K-Fold splits with same seed...")
        # Create two identical stratified CV splitters with the same seed
        stratified_cv1 = RepeatedStratifiedKFold(n_splits=args.outer, n_repeats=args.repeats, random_state=args.seed)
        stratified_cv2 = RepeatedStratifiedKFold(n_splits=args.outer, n_repeats=args.repeats, random_state=args.seed)
        
        # Get splits from both splitters
        stratified_splits1 = list(stratified_cv1.split(X, y))
        stratified_splits2 = list(stratified_cv2.split(X, y))
        
        # Check if they're identical
        stratified_identical = True
        for i, ((train1, test1), (train2, test2)) in enumerate(zip(stratified_splits1, stratified_splits2)):
            if not np.array_equal(test1, test2) or not np.array_equal(train1, train2):
                stratified_identical = False
                print(f"Stratified split {i} differs!")
                break
        
        if stratified_identical:
            print("✅ SUCCESS: Stratified KFold splits are identical with the same seed!")
        else:
            print("❌ FAILURE: Stratified KFold splits differ with the same seed!")
    else:
        print("\nSkipping stratified K-Fold tests for continuous target variable")
    
    # Compare splits with different seeds if seed2 is provided
    if args.seed2 is not None:
        print(f"\n3. Testing splits with different seeds ({args.seed} vs {args.seed2})...")
        
        # Create CV splitter with different seed
        cv_different = RepeatedKFold(n_splits=args.outer, n_repeats=args.repeats, random_state=args.seed2)
        splits_different = list(cv_different.split(X))
        
        # Check if they're different
        different_kfold = False
        for i, ((train1, test1), (train_diff, test_diff)) in enumerate(zip(splits1, splits_different)):
            if not np.array_equal(test1, test_diff) or not np.array_equal(train1, train_diff):
                different_kfold = True
                print(f"Regular split {i} differs between seeds!")
                break
        
        if different_kfold:
            print(f"✅ EXPECTED: Regular KFold splits are different with different seeds ({args.seed} vs {args.seed2})!")
        else:
            print(f"⚠️ UNEXPECTED: Regular KFold splits are identical with different seeds ({args.seed} vs {args.seed2})!")
        
        # Test stratified splits with different seeds
        if is_categorical:
            stratified_cv_different = RepeatedStratifiedKFold(n_splits=args.outer, n_repeats=args.repeats, random_state=args.seed2)
            stratified_splits_different = list(stratified_cv_different.split(X, y))
            
            different_stratified = False
            for i, ((train1, test1), (train_diff, test_diff)) in enumerate(zip(stratified_splits1, stratified_splits_different)):
                if not np.array_equal(test1, test_diff) or not np.array_equal(train1, train_diff):
                    different_stratified = True
                    print(f"Stratified split {i} differs between seeds!")
                    break
            
            if different_stratified:
                print(f"✅ EXPECTED: Stratified KFold splits are different with different seeds ({args.seed} vs {args.seed2})!")
            else:
                print(f"⚠️ UNEXPECTED: Stratified KFold splits are identical with different seeds ({args.seed} vs {args.seed2})!")
    
    # Save the first few folds as CSV files to inspect
    print("\n4. Saving sample lists for visual inspection...")
    for fold_idx in range(min(2, len(splits1))):
        _, test_idx = splits1[fold_idx]
        test_ids = ids[test_idx]
        
        fold_samples = pd.DataFrame({
            'ID': ids,
            'Target': y,
            'Set': ['Test' if i in test_idx else 'Train' for i in range(len(ids))]
        })
        
        output_file = f"regular_fold_{fold_idx}_samples.csv"
        fold_samples.to_csv(output_file, index=False)
        print(f"Saved regular samples for fold {fold_idx} to {output_file}")
    
    if is_categorical and stratified_splits1:
        for fold_idx in range(min(2, len(stratified_splits1))):
            _, test_idx = stratified_splits1[fold_idx]
            test_ids = ids[test_idx]
            
            fold_samples = pd.DataFrame({
                'ID': ids,
                'Target': y,
                'Set': ['Test' if i in test_idx else 'Train' for i in range(len(ids))]
            })
            
            output_file = f"stratified_fold_{fold_idx}_samples.csv"
            fold_samples.to_csv(output_file, index=False)
            print(f"Saved stratified samples for fold {fold_idx} to {output_file}")
    
    # Compare train/test distributions for categorical data
    if is_categorical and stratified_splits1:
        print("\n5. Analyzing class distributions in folds...")
        # Get the first fold from both regular and stratified
        _, regular_test_idx = splits1[0]
        _, stratified_test_idx = stratified_splits1[0]
        
        # Calculate class distributions
        all_classes = np.unique(y)
        all_dist = np.array([np.sum(y == c) / len(y) for c in all_classes])
        reg_test_dist = np.array([np.sum(y[regular_test_idx] == c) / len(regular_test_idx) for c in all_classes])
        strat_test_dist = np.array([np.sum(y[stratified_test_idx] == c) / len(stratified_test_idx) for c in all_classes])
        
        print("\nClass distribution comparison:")
        print("Class   | Overall | Regular Test | Stratified Test")
        print("--------|---------|--------------|----------------")
        for i, c in enumerate(all_classes):
            print(f"{c}      | {all_dist[i]:.3f}   | {reg_test_dist[i]:.3f}         | {strat_test_dist[i]:.3f}")
        
        # Calculate distribution similarity
        reg_diff = np.sum(np.abs(all_dist - reg_test_dist))
        strat_diff = np.sum(np.abs(all_dist - strat_test_dist))
        
        print(f"\nSum of absolute differences from overall distribution:")
        print(f"Regular KFold: {reg_diff:.3f}")
        print(f"Stratified KFold: {strat_diff:.3f}")
        
        if strat_diff < reg_diff:
            print("✅ Stratified sampling produces more representative test sets as expected")
        else:
            print("⚠️ Stratified sampling did not improve distribution balance in this case")
    else:
        print("\nSkipping distribution analysis for continuous target variable")
    
    print("\nTest completed successfully!")
    
if __name__ == "__main__":
    args = setup_args()
    test_cv_consistency(args) 