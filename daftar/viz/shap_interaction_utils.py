"""
SHAP Feature Interaction Analysis
=================================

Technical Implementation:
• ALL models use SHAP TreeExplainer directly for interaction computation
• Only supports regression models for simplicity and reliability
• Compatible with tree-based regression models: XGBoost, Random Forest

Interaction Computation Process:
1. TreeExplainer computes pairwise feature interactions on test data (regression only)
2. Sample-level: averages interactions across test samples per fold
3. Cross-fold: aggregates interactions preserving feature union (no absence penalty)

Three visualizations generated:
1. HEATMAP: Top 20 features by total interaction strength
2. NETWORK: Strongest feature interaction relationships
3. TOP/BOTTOM: Interactions between top positive/negative SHAP features

Mathematical Properties:
• Interaction matrices are forced symmetric (feature A×B = feature B×A)
• Diagonal zeroed (self-interactions not meaningful)
• Absolute values used for ranking/visualization only
• Original computation preserves interaction directionality

Note: Classification models are NOT supported due to TreeExplainer shape complexity
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import logging
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import save_plot function
from daftar.viz.common import save_plot

# Import colors
from daftar.viz.colors import (
    SHAP_NETWORK_EDGE_COLOR,
    SHAP_NETWORK_NODE_CMAP
)

# Core parameters
MAX_FEATURES_DISPLAY = 20  # Features to display in heatmap (matching network visualization)
MAX_INTERACTIONS_NETWORK = 20  # Strongest individual interactions for general network
TOP_BOTTOM_PER_SIDE = 10  # Features per side for top/bottom network (10 pos + 10 neg = 20 total)
HEATMAP_CMAP = "viridis"
EDGE_COLOR = SHAP_NETWORK_EDGE_COLOR
NETWORK_COLORMAP = SHAP_NETWORK_NODE_CMAP  # Colormap from YAML config used for all network visualizations

# NetworkX import
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

LOGGER = logging.getLogger(__name__)


def compute_fold_shap_interactions(
    fold_result: Dict,
    fold_idx: int,
    output_dir: Path
) -> Optional[Tuple[pd.DataFrame, np.ndarray, List[str]]]:
    """
    Compute SHAP interactions for a single fold using TreeExplainer directly.
    Only supports regression models for simplicity.
    
    Returns:
        (interactions_df, interaction_matrix, feature_names) or None if failed
    """
    # Get model - try from fold result first, then load from disk
    model = fold_result.get("model")
    if model is None:
        # Try to load model from disk
        model_path = output_dir / f"fold_{fold_idx}" / f"best_model_fold_{fold_idx}.pkl"
        if model_path.exists():
            try:
                import pickle
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                (f"Fold {fold_idx}: Loaded model from {model_path}")
            except Exception as e:
                LOGGER.warning(f"Fold {fold_idx}: Failed to load model from {model_path}: {e}")
                return None
        else:
            LOGGER.warning(f"Fold {fold_idx}: No model available and no saved model file")
            return None
    
    # Check if this is a regression model - interactions only supported for regression
    if hasattr(model, 'model') and hasattr(model.model, 'objective'):
        # XGBoost model - check objective
        objective = model.model.objective
        if 'reg:' not in str(objective):
            LOGGER.warning(f"Fold {fold_idx}: SHAP interactions only supported for regression models, skipping")
            return None
    elif hasattr(model, 'model') and hasattr(model.model, 'n_classes_'):
        # Random Forest classifier
        LOGGER.warning(f"Fold {fold_idx}: SHAP interactions only supported for regression models, skipping")
        return None
    
    # Get SHAP data 
    shap_data = fold_result.get("shap_data")
    if not shap_data or len(shap_data) < 3:
        LOGGER.warning(f"Fold {fold_idx}: No SHAP data available")
        return None
    
    shap_values, X_test_df, y_test = shap_data
    
    if not isinstance(X_test_df, pd.DataFrame):
        LOGGER.warning(f"Fold {fold_idx}: X_test is not a DataFrame")
        return None
    
    # Select top features based on magnitude (mean absolute SHAP values)
    if shap_values is None or len(shap_values) == 0:
        LOGGER.warning(f"Fold {fold_idx}: No SHAP values available")
        return None
    
    # Get feature names and validate data shape
    feature_names = X_test_df.columns.tolist()
    n_samples, n_features = X_test_df.shape
    
    LOGGER.debug(f"Fold {fold_idx}: X_test shape: {X_test_df.shape}, features: {n_features}")
    
    # Validate that we have consistent data
    if len(feature_names) != n_features:
        LOGGER.error(f"Fold {fold_idx}: Feature name count ({len(feature_names)}) doesn't match data columns ({n_features})")
        return None
    
    (f"Fold {fold_idx}: Computing interactions for all {len(feature_names)} features")
    
    # Get underlying model for TreeExplainer
    underlying_model = getattr(model, 'model', model)
    
    # Validate model expects the correct number of features
    if hasattr(underlying_model, 'n_features_in_'):
        expected_features = underlying_model.n_features_in_
        if expected_features != n_features:
            LOGGER.error(f"Fold {fold_idx}: Model expects {expected_features} features but X_test has {n_features}")
            return None
        LOGGER.debug(f"Fold {fold_idx}: Model feature count validation passed: {expected_features}")
    
    try:
        import shap
        
        # Convert DataFrame to numpy array for SHAP computation
        X_test_array = X_test_df.values
        LOGGER.debug(f"Fold {fold_idx}: Converted DataFrame to array shape: {X_test_array.shape}")
        
        # Use TreeExplainer directly for simplicity and reliability
        (f"Fold {fold_idx}: Computing SHAP interactions using TreeExplainer (direct)")
        explainer = shap.TreeExplainer(underlying_model)
        interaction_values = explainer.shap_interaction_values(X_test_array)
        
        # Validate interaction values shape - should be (n_samples, n_features, n_features) for regression
        LOGGER.debug(f"Fold {fold_idx}: Raw interaction values shape: {interaction_values.shape}")
        
        # Regression should have shape (n_samples, n_features, n_features)
        expected_shape = (n_samples, n_features, n_features)
        if interaction_values.shape != expected_shape:
            LOGGER.error(f"Fold {fold_idx}: Unexpected interaction values shape {interaction_values.shape}, expected {expected_shape}")
            return None
        
        # Average absolute values across samples to get feature-level interactions
        interaction_matrix = np.abs(interaction_values).mean(axis=0)
        LOGGER.debug(f"Fold {fold_idx}: Final interaction matrix shape: {interaction_matrix.shape}")
        
        # Ensure interaction matrix is symmetric and zero diagonal
        interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
        np.fill_diagonal(interaction_matrix, 0)
        
        # Create tidy DataFrame with all pairwise interactions
        interactions_list = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                # Matrix is now symmetric, so just use one element
                strength = interaction_matrix[i, j]
                interactions_list.append({
                    'feature1': feature_names[i],
                    'feature2': feature_names[j], 
                    'interaction_strength': strength
                })
        
        interactions_df = pd.DataFrame(interactions_list).sort_values(
            'interaction_strength', ascending=False
        ).reset_index(drop=True)
        
        (f"Fold {fold_idx}: Successfully computed {len(interactions_df)} interactions")
        return interactions_df, interaction_matrix, feature_names
        
    except Exception as e:
        LOGGER.error(f"Fold {fold_idx}: Failed to compute interactions: {e}")
        return None


def aggregate_fold_interactions(
    fold_interactions: List[Tuple[pd.DataFrame, np.ndarray, List[str]]]
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Aggregate interaction results across all folds.
    
    Returns:
        (combined_interactions_df, mean_interaction_matrix, all_features)
    """
    # Get union of all features across folds (sorted for consistency)
    all_features = sorted(set(
        feature for _, _, features in fold_interactions for feature in features
    ))
    
    n_features = len(all_features)
    n_folds = len(fold_interactions)
    
    # Create feature name to index mapping for O(1) lookups
    feature_to_idx = {feat: idx for idx, feat in enumerate(all_features)}
    
    # Create aggregation tensor - track sum and count separately for proper averaging
    fold_sum = np.zeros((n_features, n_features))
    fold_count = np.zeros((n_features, n_features))
    
    # Aggregate interaction matrices with proper indexing
    for fold_idx, (_, matrix, features) in enumerate(fold_interactions):
        # Create mapping for this fold's features
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                # Get indices in the combined feature space
                idx1 = feature_to_idx[feat1]
                idx2 = feature_to_idx[feat2]
                
                # Add to sum and increment count
                fold_sum[idx1, idx2] += matrix[i, j]
                fold_count[idx1, idx2] += 1
    
    # Calculate mean where we have data (avoid division by zero)
    mean_matrix = np.zeros((n_features, n_features))
    mask = fold_count > 0
    mean_matrix[mask] = fold_sum[mask] / fold_count[mask]
    
    # Ensure symmetry in the final matrix (interactions should be symmetric)
    mean_matrix = (mean_matrix + mean_matrix.T) / 2
    
    # Zero out diagonal (self-interactions are not meaningful)
    np.fill_diagonal(mean_matrix, 0)
    
    # Create combined interactions DataFrame from the symmetric matrix
    interactions_list = []
    for i in range(n_features):
        for j in range(i + 1, n_features):  # Only upper triangle to avoid duplicates
            strength = mean_matrix[i, j]
            if strength > 0:  # Only include non-zero interactions
                interactions_list.append({
                    'feature1': all_features[i],
                    'feature2': all_features[j],
                    'interaction_strength': strength
                })
    
    combined_df = pd.DataFrame(interactions_list).sort_values(
        'interaction_strength', ascending=False
    ).reset_index(drop=True)
    
    (f"Aggregated {len(combined_df)} interactions across {n_folds} folds")
    LOGGER.debug(f"Feature count per fold: {[len(features) for _, _, features in fold_interactions]}")
    LOGGER.debug(f"Total unique features: {n_features}")
    
    return combined_df, mean_matrix, all_features


def plot_interaction_heatmap(
    interaction_matrix: np.ndarray,
    feature_names: List[str],
    output_path: Path
) -> None:
    """Create and save interaction heatmap for top 20 features."""
    # Limit to top 20 features for visualization
    display_limit = min(MAX_FEATURES_DISPLAY, len(feature_names))
    
    if display_limit < len(feature_names):
        # Get top features by total interaction strength
        total_interactions = np.abs(interaction_matrix).sum(axis=1)
        top_display_indices = np.argsort(-total_interactions)[:display_limit]
        
        display_matrix = interaction_matrix[np.ix_(top_display_indices, top_display_indices)]
        display_names = [feature_names[i] for i in top_display_indices]
        
        (f"Displaying top {display_limit} features by interaction strength in heatmap")
    else:
        display_matrix = interaction_matrix
        display_names = feature_names
    
    plt.figure(figsize=(10, 8))
    
    # Matrix should already be symmetric from aggregation, but ensure consistency
    # Take absolute values for visualization (interaction strength is always positive)
    symmetric_matrix = np.abs(display_matrix)
    # Ensure symmetry (should already be symmetric but being extra careful)
    symmetric_matrix = (symmetric_matrix + symmetric_matrix.T) / 2
    np.fill_diagonal(symmetric_matrix, 0)  # Zero diagonal for clean visualization
    
    # Use lower triangle mask
    mask = np.triu(np.ones_like(symmetric_matrix, dtype=bool), k=1)
    
    # Check if values are very small
    max_abs_value = np.abs(symmetric_matrix).max()
    if max_abs_value < 0.01:
        # For very small values, use scientific notation
        fmt = '.2e'
    else:
        # For regular values, use standard decimal format
        fmt = '.3f'
    
    # Create the heatmap
    sns.heatmap(
        symmetric_matrix,
        mask=mask,
        annot=False,  # Remove numbers from heatmap squares
        cmap=HEATMAP_CMAP,
        square=True,
        linewidths=0.5,
        xticklabels=display_names,
        yticklabels=display_names,
        cbar_kws={'label': 'SHAP Interaction Strength'}
    )
    
    plt.title(f'Feature Interactions Heatmap\n(Top {len(display_names)} Features by SHAP Magnitude)')
    plt.tight_layout()
    fig = plt.gcf()
    save_plot(fig, output_path, tight_layout=False)
    

def plot_interaction_network(
    interactions_df: pd.DataFrame,
    shap_df: Optional[pd.DataFrame],
    output_path: Path,
    top_n_edges: int = MAX_INTERACTIONS_NETWORK
) -> None:
    """Create and save interaction network plot for top interactions."""
    if not HAS_NETWORKX:
        LOGGER.warning("NetworkX not available - skipping network plot")
        return
    
    # Create graph from top interactions
    G = nx.Graph()
    top_interactions = interactions_df.head(top_n_edges)
        
    # Add nodes and edges
    max_strength = top_interactions['interaction_strength'].max()
    for _, row in top_interactions.iterrows():
        G.add_edge(
            row['feature1'], 
            row['feature2'], 
            weight=row['interaction_strength'] / max_strength
        )
    
    if len(G.edges()) == 0:
        LOGGER.warning("No edges for network plot")
        return
    
    # Layout
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Node properties
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        # Color by SHAP value if available
        if shap_df is not None and node in shap_df.index:
            shap_val = shap_df.loc[node, 'Mean_SHAP']
            node_colors.append(shap_val)
            # Size by magnitude (absolute SHAP value)
            node_sizes.append(500 + abs(shap_val) * 2000)
        else:
            node_colors.append(0)
            node_sizes.append(800)
    
    # Edge properties  
    edge_widths = [3 * G[u][v]['weight'] + 0.5 for u, v in G.edges()]
    
    # Create plot
    plt.figure(figsize=(12, 7), facecolor='#FFFFFF')
    ax = plt.gca()
    ax.set_facecolor('#FFFFFF')  # Set a single background color
    
    # Draw network
    nx.draw(
        G, pos, 
        node_color=node_colors,
        node_size=node_sizes,
        cmap=NETWORK_COLORMAP,
        edgecolors='white',
        linewidths=1,
        alpha=0.8,
        width=edge_widths,
        edge_color=EDGE_COLOR,
        with_labels=True,
        font_size=10,
        font_weight='bold'
    )
    
    # Add colorbar if we have SHAP values
    if shap_df is not None and len(set(node_colors)) > 1:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Create a ScalarMappable for the colorbar
        norm = mcolors.Normalize(vmin=min(node_colors), vmax=max(node_colors))
        sm = cm.ScalarMappable(norm=norm, cmap=NETWORK_COLORMAP)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='SHAP Value', shrink=0.7)
    
    # Get actual number of edges for more accurate title
    actual_edges = len(G.edges())
    plt.title(f'SHAP Interaction Network\n({actual_edges} Strongest Feature Interactions)')
    plt.axis('off')
    plt.tight_layout()
    fig = plt.gcf()
    save_plot(fig, output_path, tight_layout=False)
    
    (f"Saved interaction network to {output_path}")


def plot_topbottom_network(
    interactions_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    output_path: Path,
    top_n_features: int = 10  # 10 positive + 10 negative = 20 total
) -> None:
    """Create network showing interactions between top positive and negative SHAP features."""
    if not HAS_NETWORKX:
        LOGGER.warning("NetworkX not available - skipping top/bottom network plot")
        return
    
    # Use exactly 10 features per side
    top_n_features = 10
        
    # Get top positive and negative features using integer indexing
    positive_mask = shap_df['Mean_SHAP'] > 0
    negative_mask = shap_df['Mean_SHAP'] < 0
    
    positive_features = shap_df[positive_mask].sort_values(
        'Mean_SHAP', ascending=False
    ).head(top_n_features).index.tolist()
    
    negative_features = shap_df[negative_mask].sort_values(
        'Mean_SHAP', ascending=True  
    ).head(top_n_features).index.tolist()
    
    selected_features = positive_features + negative_features
        
    # Filter interactions to selected features
    filtered_interactions = interactions_df[
        (interactions_df['feature1'].isin(selected_features)) &
        (interactions_df['feature2'].isin(selected_features))
    ].head(20)  # Show more interactions between these features
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for feature in selected_features:
        is_positive = feature in positive_features
        shap_value = shap_df.loc[feature, 'Mean_SHAP']
        G.add_node(feature, 
                  direction='positive' if is_positive else 'negative',
                  color=shap_value)  # Store actual SHAP value for colormap usage
    
    # Add edges
    connected_nodes = set()
    if not filtered_interactions.empty:
        max_strength = filtered_interactions['interaction_strength'].max()
        for _, row in filtered_interactions.iterrows():
            feature1, feature2 = row['feature1'], row['feature2']
            G.add_edge(feature1, feature2, 
                      weight=row['interaction_strength'] / max_strength)
            connected_nodes.add(feature1)
            connected_nodes.add(feature2)
    
    # Identify unconnected nodes
    unconnected_nodes = set(G.nodes()) - connected_nodes
    pos_unconnected = [n for n in unconnected_nodes if n in positive_features]
    neg_unconnected = [n for n in unconnected_nodes if n in negative_features]
    (f"Found {len(unconnected_nodes)} unconnected features")
    
    # Create a main network layout
    pos = {}
    
    # Connected nodes in center
    if connected_nodes:
        connected_subgraph = G.subgraph(connected_nodes)
        pos_connected = nx.spring_layout(connected_subgraph, k=0.5, iterations=50, seed=42)
        pos.update(pos_connected)
    
    # Create plot
    plt.figure(figsize=(14, 10), facecolor='#FFFFFF')
    ax = plt.gca()
    ax.set_facecolor('#FFFFFF')  # Set a single background color
    
    # Node colors based on SHAP values (same as interaction_network)
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        # Color based on SHAP value (like in interaction_network)
        if node in shap_df.index:
            shap_value = shap_df.loc[node, 'Mean_SHAP']
            node_colors.append(shap_value)
            size = 500 + abs(shap_value) * 3000
        else:
            node_colors.append(0)
            size = 800
        node_sizes.append(size)
    
    # Draw nodes for connected features
    if connected_nodes:
        # Extract node colors and sizes for connected nodes only
        connected_node_colors = [node_colors[i] for i, n in enumerate(G.nodes()) if n in connected_nodes]
        connected_node_sizes = [node_sizes[i] for i, n in enumerate(G.nodes()) if n in connected_nodes]
        
        # Draw the subgraph with connected nodes
        nx.draw(
            G.subgraph(connected_nodes), pos,
            node_color=connected_node_colors,
            node_size=connected_node_sizes,
            cmap=NETWORK_COLORMAP,
            edgecolors='white',
            linewidths=1,
            alpha=0.8,
            width=[2 + 4 * G[u][v]['weight'] for u, v in G.edges()],
            edge_color=EDGE_COLOR,
            with_labels=True,
            font_size=10,
            font_weight='bold'
        )
    
    # Add a simple note about unconnected features instead of boxes
    if pos_unconnected or neg_unconnected:
        pos_count = len(pos_unconnected)
        neg_count = len(neg_unconnected)
        note = f"Note: {pos_count} positive and {neg_count} negative features had no interactions"
        plt.figtext(0.5, 0.02, note, ha='center', fontsize=10, fontstyle='italic')
    
    plt.title(f'Feature Interactions: Top {len(positive_features)} Positive vs {len(negative_features)} Negative SHAP Features')
    plt.axis('off')
    
    # Add colorbar for SHAP values (same as interaction_network)
    if len(set(node_colors)) > 1:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Create a ScalarMappable for the colorbar
        norm = mcolors.Normalize(vmin=min(node_colors), vmax=max(node_colors))
        sm = cm.ScalarMappable(norm=norm, cmap=NETWORK_COLORMAP)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='SHAP Value', shrink=0.7)
    
    # Adjust figure size and margins to avoid tight layout warnings
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    fig = plt.gcf()
    save_plot(fig, output_path, tight_layout=False)
    
def save_shap_interactions_analysis(
    fold_results: Sequence[Dict],
    output_dir: Union[str, Path], 
    shap_df: Optional[pd.DataFrame] = None,
    max_features_compute: Optional[int] = None,
    top_n_interactions: Optional[int] = None,  # Alternative parameter name for compatibility
    **kwargs  # Ignore other parameters for compatibility
) -> Dict[str, str]:
    """
    Main function to compute and visualize SHAP interactions using TreeExplainer.
    
    Args:
        fold_results: List of fold result dictionaries
        output_dir: Output directory path
        shap_df: DataFrame with SHAP values and feature importance
        max_features_compute: Maximum number of top features to use for computation
        
    Returns:
        Dictionary mapping artifact names to file paths
    """
    output_dir = Path(output_dir)
    interactions_dir = output_dir / "shap_feature_interactions"
    interactions_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle parameter compatibility
    if max_features_compute is None and top_n_interactions is not None:
        max_features_compute = top_n_interactions
    
    artifacts = {}
    
    # First try to compute interactions from fold results if not already computed
    fold_interactions = []
    existing_files = []
    
    # Check for existing interaction files from pipeline processing
    for i in range(1, len(fold_results) + 1):
        fold_csv = output_dir / f"fold_{i}" / f"shap_interactions_fold_{i}.csv"
        LOGGER.debug(f"Checking for existing interactions at: {fold_csv}")
        if fold_csv.exists():
            existing_files.append((i, fold_csv))
            LOGGER.debug(f"Found existing interaction file: {fold_csv}")
    
    if existing_files:
        # Load existing interaction files
        for fold_idx, csv_path in existing_files:
            try:
                df = pd.read_csv(csv_path)
                
                # Check if the CSV has the expected structure
                required_cols = ['feature1', 'feature2', 'interaction_strength']
                if not all(col in df.columns for col in required_cols):
                    LOGGER.warning(f"Fold {fold_idx} CSV missing required columns, skipping")
                    continue
                
                # Reconstruct interaction matrix from the CSV
                features = sorted(set(df['feature1'].tolist() + df['feature2'].tolist()))
                n_features = len(features)
                matrix = np.zeros((n_features, n_features))
                
                feature_to_idx = {feat: idx for idx, feat in enumerate(features)}
                for _, row in df.iterrows():
                    if pd.notna(row['interaction_strength']):  # Skip NaN values
                        i = feature_to_idx[row['feature1']]
                        j = feature_to_idx[row['feature2']]
                        strength = row['interaction_strength']
                        # Fill both positions to ensure symmetry
                        matrix[i, j] = strength
                        matrix[j, i] = strength
                
                # Zero diagonal for consistency (self-interactions not meaningful)
                np.fill_diagonal(matrix, 0)
                
                fold_interactions.append((df, matrix, features))
                
                # Keep track of loaded interactions
                artifacts[f"fold_{fold_idx}_interactions"] = "loaded_from_fold_directory"                
            except Exception as e:
                LOGGER.warning(f"Could not load interactions from fold {fold_idx}: {e}")
                continue
    # Final check - if no interactions were loaded or computed, fail cleanly
    if not fold_interactions:
        LOGGER.error("No SHAP interactions computed successfully")
        LOGGER.error("SHAP interaction analysis requires TreeExplainer-compatible models with computed interactions")
        return {}  # Return empty artifacts dictionary
    
    # Aggregate across folds
    combined_df, mean_matrix, all_features = aggregate_fold_interactions(fold_interactions)
    
    # Save combined results
    combined_csv = interactions_dir / "all_interactions.csv"
    combined_df.to_csv(combined_csv, index=False)
    artifacts["all_interactions"] = str(combined_csv)
    
    # Create summary text first (always create this)
    max_computed = max_features_compute if max_features_compute is not None else len(all_features)
    summary_lines = [
        "SHAP Interaction Analysis Summary",
        "================================",
        f"Features analyzed: {len(all_features)} (top {max_computed} by SHAP magnitude)",
        f"Features displayed: {min(MAX_FEATURES_DISPLAY, len(all_features))}",
        f"Total interactions: {len(combined_df)}",
        f"Folds processed: {len(fold_interactions)}",
        "",
        "Top 10 strongest interactions:",
    ]
    
    for i, row in combined_df.head(10).iterrows():
        summary_lines.append(
            f"{i+1:>2}. {row['feature1']} × {row['feature2']}: "
            f"{row['interaction_strength']:.4f}"
        )
    
    summary_text = "\n".join(summary_lines)
    summary_path = interactions_dir / "interaction_summary.txt"
    summary_path.write_text(summary_text)
    artifacts["interaction_summary"] = str(summary_path)
    
    # Generate visualizations
    try:
        # 1. Heatmap
        heatmap_path = interactions_dir / "interaction_heatmap.png"
        plot_interaction_heatmap(mean_matrix, all_features, heatmap_path)
        artifacts["heatmap"] = str(heatmap_path)
        
        # 2. Network plot
        network_path = interactions_dir / "interaction_network.png"
        plot_interaction_network(combined_df, shap_df, network_path)
        artifacts["interaction_network"] = str(network_path)
        
        # 3. Top/bottom network (if SHAP data available)
        if shap_df is not None and not shap_df.empty:
            topbottom_path = interactions_dir / "top_bottom_network.png"
            plot_topbottom_network(combined_df, shap_df, topbottom_path)
            artifacts["top_bottom_network"] = str(topbottom_path)
        
    except Exception as e:
        LOGGER.error(f"Error during visualization creation: {e}")
        
    ("SHAP interaction analysis complete.")
    
    if "heatmap" in artifacts:
        (f"Generated {len(artifacts)} interaction visualization artifacts")
    else:
        LOGGER.warning("No visualizations created - insufficient data")
    
    return artifacts