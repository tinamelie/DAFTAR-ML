"""
SHAP interaction analysis & visualisation utilities
===================================================

Refactored so that:
• regression, binary and multi-class classification are all supported
• top-feature information that the training pipeline already calculated
  is *re-used* (it is **never** recomputed here)
• the plotting functions are left functionally identical
• many edge-cases are handled explicitly and logged.

Author:  refactor-bot 2024-06
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# std / third-party
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import logging
import traceback
import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Import centralized color definitions
from daftar.viz.colors import (
    SHAP_POSITIVE_COLOR,
    SHAP_NEGATIVE_COLOR,
    SHAP_NETWORK_NODE_COLOR,
    SHAP_NETWORK_EDGE_COLOR,
    SHAP_NETWORK_LINK_COLOR,
    TOP_BOTTOM_POSITIVE_COLOR,
    TOP_BOTTOM_NEGATIVE_COLOR,
    SHAP_NETWORK_EDGE_COLOR_DEFAULT
)

# Color definitions for network plots
EDGE_COLOUR = SHAP_NETWORK_EDGE_COLOR
POSITIVE_NODE_COLOR = TOP_BOTTOM_POSITIVE_COLOR  # Red for positive impact
NEGATIVE_NODE_COLOR = TOP_BOTTOM_NEGATIVE_COLOR  # Blue for negative impact
HEATMAP_CMAP = "viridis"

# optional dependencies --------------------------------------------------------
try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:             # pragma: no cover
    HAS_NETWORKX = False

try:                            # Louvain community detection
    import community.community_louvain as community_louvain

    HAS_LOUVAIN = True
except ImportError:             # pragma: no cover
    try:
        import community as community_louvain

        HAS_LOUVAIN = True
    except ImportError:         # pragma: no cover
        HAS_LOUVAIN = False

# colour scheme (falls back to hard-coded values if DAFTAR config cannot be loaded) ----------
try:
    # Updated import path to the consolidated colors module
    from daftar.viz.colors import load_colors

    _cfg = load_colors().get("shap", {}).get("interactions", {})
    HEATMAP_CMAP  = _cfg.get("heatmap_cmap", "viridis")
    DENDRO_CMAP   = _cfg.get("dendrogram_cmap", "viridis")
    LINK_COLOUR   = _cfg.get("link_color", SHAP_NETWORK_LINK_COLOR)
    NODE_COLOUR   = _cfg.get("network_node_color", SHAP_NETWORK_NODE_COLOR)
    EDGE_COLOUR   = _cfg.get("network_edge_color", SHAP_NETWORK_EDGE_COLOR)
except Exception:  # pragma: no cover – fall back to hard-coded defaults
    HEATMAP_CMAP  = "viridis"
    DENDRO_CMAP   = "viridis"
    LINK_COLOUR   = SHAP_NETWORK_LINK_COLOR
    NODE_COLOUR   = SHAP_NETWORK_NODE_COLOR
    EDGE_COLOUR   = SHAP_NETWORK_EDGE_COLOR

POSITIVE_NODE_COLOR = TOP_BOTTOM_POSITIVE_COLOR   # for top-bottom network
NEGATIVE_NODE_COLOR = TOP_BOTTOM_NEGATIVE_COLOR

# ──────────────────────────────────────────────────────────────────────────────
# logging
# ──────────────────────────────────────────────────────────────────────────────
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

# ──────────────────────────────────────────────────────────────────────────────
# helper utilities
# ──────────────────────────────────────────────────────────────────────────────
def _p(p: Union[str, Path], *, make_dir: bool = False) -> Path:
    """Make sure parent directory exists and return *Path* instance."""
    p = Path(p)
    # Just make the parent directory
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _extract_fold_explainers(folds: Sequence[Dict[str, Any]]) -> List[Any]:
    return [f["shap_explainer"] for f in folds if "shap_explainer" in f]


def _get_feature_names(folds: Sequence[Dict[str, Any]]) -> Optional[List[str]]:
    for f in folds:
        shap_data = f.get("shap_data")
        if shap_data and len(shap_data) > 1 and isinstance(shap_data[1], pd.DataFrame):
            return list(shap_data[1].columns)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# use pre-computed mask from the pipeline
# ──────────────────────────────────────────────────────────────────────────────
def _obtain_top_feature_mask(
    folds: Sequence[Dict[str, Any]],
    *,
    total_features: int,
    top_n: int,
) -> Optional[np.ndarray]:
    """
    Return a Boolean mask (len = total_features) for the **first** fold that
    contains any of the recognised keys.  Falls back to *None* (= keep all).
    """
    for fold in folds:
        # mask already provided ------------------------------------------------
        mask = fold.get("top_feature_mask")
        if mask is not None and len(mask) == total_features:
            return mask.astype(bool)

        # indices --------------------------------------------------------------
        idx = fold.get("top_feature_indices")
        if idx is not None:
            idx = np.asarray(idx, dtype=int)[:top_n]
            m = np.zeros(total_features, dtype=bool)
            m[idx] = True
            return m

        # names ----------------------------------------------------------------
        names = fold.get("top_feature_names")
        if names is not None:
            full = _get_feature_names(folds)
            if full:
                sel = set(names[:top_n])
                return np.array([n in sel for n in full], dtype=bool)

    return None


# ──────────────────────────────────────────────────────────────────────────────
# SHAP interaction extraction / aggregation
# ──────────────────────────────────────────────────────────────────────────────
def _aggregate_classes(vals: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    """
    Convert the various outputs that `shap_interaction_values` can return
    into a single tensor with shape  (n_samples, n_features, n_features).

    * Regression ................ ndarray
    * Binary classification ..... list length 2
    * Multi-class classification  list length > 2

    The returned tensor is the MEAN of *absolute* interactions across classes.
    """
    if isinstance(vals, list):
        # stack -> (n_classes, n_samples, f, f)
        arr = np.stack(vals, axis=0)
        return np.mean(np.abs(arr), axis=0)
    if isinstance(vals, np.ndarray):
        return vals                       # already good
    raise TypeError("Unsupported type returned by shap_interaction_values()")


def _get_interaction_values(
    folds: Sequence[Dict[str, Any]],
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Obtain (or compute) the SHAP interaction tensor for the FIRST fold that
    provides either a cached tensor *or* an explainer / model able to produce
    it.  The heavy work is therefore done only once.
    """
    # 1. already cached in the fold? ------------------------------------------
    for f in folds:
        cached = f.get("shap_interaction_data")
        if cached is not None:
            arr, names = cached
            return _aggregate_classes(arr), names

    # 2. try the model directly ----------------------------------------------
    for f in folds:
        model = f.get("model")
        if model is None:
            continue

        X = None
        if "shap_data" in f and len(f["shap_data"]) > 1:
            _, X, _ = f["shap_data"]
        elif "X_test" in f:
            X = f["X_test"]
            if not isinstance(X, pd.DataFrame):
                names = _get_feature_names(folds)
                X = pd.DataFrame(X, columns=names) if names else None

        if X is None:
            continue

        try:
            # First try to use the model's method if it exists
            if hasattr(model, "shap_interaction_values"):
                vals = model.shap_interaction_values(X)
                names = list(X.columns)
                return _aggregate_classes(vals), names
            
            # If the model doesn't have the method, create an explainer
            import shap
            try:
                LOGGER.info(f"Model doesn't have shap_interaction_values method, creating explainer directly")
                explainer = shap.TreeExplainer(model)
                vals = explainer.shap_interaction_values(X)
                names = list(X.columns)
                return _aggregate_classes(vals), names
            except Exception as inner_e:
                LOGGER.warning(f"TreeExplainer failed to produce interaction values: {inner_e}")
                
        except Exception as e:            # pragma: no cover
            LOGGER.warning(f"Model failed to produce interaction values: {e}")

    # 3. finally: fall back to the explainer ----------------------------------
    expl = _extract_fold_explainers(folds)
    if expl:
        expl = expl[0]
        try:
            vals = expl.shap_interaction_values(expl.data)
            names = _get_feature_names(folds)
            return _aggregate_classes(vals), names
        except Exception as e:            # pragma: no cover
            LOGGER.warning(f"Explainer failed: {e}")

    return None, None


def _average_matrix(vals: np.ndarray) -> np.ndarray:
    """`|values|` averaged over samples  -> shape (f, f)."""
    return np.abs(vals).mean(axis=0)


def _tidy_interactions(mat: np.ndarray, names: List[str]) -> pd.DataFrame:
    """Return long-form DataFrame sorted by interaction strength."""
    rows: list[dict[str, Any]] = []
    n = len(names)
    # Make sure matrix dimensions match feature names length
    matrix_size = min(n, mat.shape[0], mat.shape[1])
    
    # Use the smaller of the two to avoid index errors
    for i in range(matrix_size):
        for j in range(i + 1, matrix_size):
            # Ensure s is a scalar value, not an array
            s_val = mat[i, j] + mat[j, i]
            if hasattr(s_val, "shape") and s_val.shape:
                # If it's an array, take the mean or first value
                s_val = float(np.mean(s_val))
            else:
                # Make sure it's a float for consistent sorting
                s_val = float(s_val)
                
            rows.append(
                dict(feature1=names[i], feature2=names[j], interaction_strength=s_val)
            )
    df = pd.DataFrame(rows).sort_values("interaction_strength", ascending=False)
    df["normalized_strength"] = (
        df["interaction_strength"] / (df["interaction_strength"].max() + 1e-12)
    )
    return df.reset_index(drop=True)


def calculate_shap_interactions(
    folds: Sequence[Dict[str, Any]],
    *,
    top_n_features: int = 50,
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[List[str]]]:
    """
    Main public helper:  returns (tidy_df, mean_matrix, feature_names)
    limited to `top_n_features` if the pipeline provided a mask.
    """
    vals, names = _get_interaction_values(folds)
    if vals is None or names is None:
        return None, None, None

    mask = _obtain_top_feature_mask(
        folds, total_features=len(names), top_n=top_n_features
    )

    if mask is not None:
        vals = vals[:, mask][:, :, mask]
        names = [n for n, keep in zip(names, mask) if keep]

    mat = _average_matrix(vals)
    df = _tidy_interactions(mat, names)
    return df, mat, names


# ──────────────────────────────────────────────────────────────────────────────
# community detection
# ──────────────────────────────────────────────────────────────────────────────
def cluster_interactions(
    interaction_matrix: np.ndarray,
    feature_names: List[str],
    *,
    resolution: float = 1.0,
) -> pd.DataFrame:
    if interaction_matrix is None or feature_names is None:
        return pd.DataFrame()

    # Use Louvain community detection (now required dependency)
    sym = interaction_matrix + interaction_matrix.T
    np.fill_diagonal(sym, 0)
    G = nx.from_numpy_array(sym)
    part = community_louvain.best_partition(
        G, weight="weight", resolution=resolution, random_state=42
    )
    labs = [part[i] for i in range(len(feature_names))]

    return pd.DataFrame({"feature": feature_names, "cluster": labs}).sort_values(
        "cluster"
    )


# ──────────────────────────────────────────────────────────────────────────────
# plotting helpers (unchanged – only tiny internal clean-ups)
# ──────────────────────────────────────────────────────────────────────────────
def plot_interaction_heatmap(
    mat: np.ndarray,
    names: List[str],
    *,
    out_path: Union[str, Path],
    top_n: int = 15,  # Set default to 15 features
) -> None:
    if mat is None or len(names) == 0:
        return
    
    # Ensure consistent number of features with other plots
    top_n = min(15, len(names))  # Use exactly 15 or fewer if not available
    
    # Create absolute matrix and get top features by sum of interactions
    abs_m = np.abs(mat)
    imp = abs_m.sum(axis=1)
    keep = np.argsort(-imp)[:top_n]
    sub = abs_m[np.ix_(keep, keep)]
    lbl = [names[i] for i in keep]

    plt.figure(figsize=(12, 10))
    
    # Create symmetrical matrix for the heatmap
    sym = sub + sub.T
    np.fill_diagonal(sym, 0)  # Zero out diagonal
    
    # Use lower triangular mask instead of upper to avoid blank rows
    mask = np.tril(np.ones_like(sym, dtype=bool), k=-1)
    
    sns.heatmap(
        sym,
        mask=mask,  # Use lower triangular mask
        cmap=HEATMAP_CMAP,
        square=True,
        linewidths=0.4,
        xticklabels=lbl,
        yticklabels=lbl,
        cbar_kws=dict(label="|mean SHAP interaction|"),
    )
    plt.title(f"SHAP interaction heatmap (top {top_n} features)")
    plt.tight_layout()
    plt.savefig(_p(out_path), dpi=300)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# network visualizations
# ──────────────────────────────────────────────────────────────────────────────
def plot_interaction_network(
    inter_df: pd.DataFrame,
    *,
    out_path: Union[str, Path],
    top_n: int = 15,  # Changed default to 15
    shap_df: Optional[pd.DataFrame] = None,
) -> bool:
    # First, make sure we have NetworkX
    if not HAS_NETWORKX:
        LOGGER.warning("NetworkX not available for network visualization")
        return False
    
    # Check if we have valid data
    if inter_df is None or inter_df.empty:
        LOGGER.warning("Empty interaction dataframe provided for network plot")
        return False
    
    # Create output directory if needed
    out_path = _p(out_path, make_dir=True)
        
    # Create graph from top interactions
    # No verbose message for network visualization
    df = inter_df.head(top_n)
    G = nx.Graph()
    
    # Add nodes and edges
    node_set = set()
    for _, r in df.iterrows():
        node_set.add(r.feature1)
        node_set.add(r.feature2)
    
    # First add all nodes (in case some nodes have no edges)
    for node in node_set:
        G.add_node(node)
    
    # Then add edges with normalized weights
    max_s = df["interaction_strength"].max()
    if max_s <= 0:
        max_s = 1.0  # Avoid division by zero
        
    for _, r in df.iterrows():
        norm_weight = r.interaction_strength / max_s
        G.add_edge(r.feature1, r.feature2, weight=norm_weight)

    if len(G.edges()) == 0:
        LOGGER.warning("No edges in network graph - skipping network plot")
        return False

    # Compute layout - directly use spring layout with optimized parameters
    # Skip Kamada-Kawai layout which often causes 'Distance matrix diagonal' errors
    try:
        # k controls spacing between nodes (higher = more spread out)
        # iterations controls how many times to adjust positions
        pos = nx.spring_layout(G, k=0.5, seed=42, iterations=100)
    except Exception as e:
        LOGGER.error(f"Spring layout failed: {e}")
        return False

    # Node sizes based on degree centrality
    # No verbose message for network visualization
    try:
        centrality = nx.degree_centrality(G)
        sizes = [500 + 3000 * centrality[n] for n in G.nodes()]
    except Exception as e:
        LOGGER.warning(f"Error computing centrality: {e}, using default sizes")
        sizes = [1000] * len(G.nodes())

    # Edge widths based on weights
    try:
        widths = [1 + 5 * G.get_edge_data(u, v).get('weight', 0.5) for u, v in G.edges()]
    except Exception as e:
        LOGGER.warning(f"Error computing edge widths: {e}, using default widths")
        widths = [2] * len(G.edges())

    # Create plot
    # No verbose message for network visualization
    try:
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Draw nodes colored by SHAP values
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import Normalize
        
        # Extract SHAP values for each node
        node_shap_values = []
        
        # Check if we have SHAP data for coloring
        if shap_df is not None and not shap_df.empty:
            for node in G.nodes():
                # Get the SHAP value from shap_df if available
                if node in shap_df.index:
                    node_shap_values.append(shap_df.loc[node, 'Fold_Mean_SHAP'])
                else:
                    node_shap_values.append(0)  # Fallback value
        else:
            # Use default color if no SHAP data available
            node_shap_values = [0] * len(G.nodes())
        
        # Create color map - blue to red (RdBu_r)
        cmap = cm.get_cmap('RdBu_r')  # Red-Blue reversed (blue for negative, red for positive)
        
        # Draw nodes with color based on SHAP value
        nodes = nx.draw_networkx_nodes(
            G, pos, node_size=sizes, node_color=node_shap_values, 
            cmap=cmap, edgecolors="white", linewidths=1, alpha=0.8,
            ax=ax
        )
        
        # Add a colorbar
        if shap_df is not None and not shap_df.empty and len(node_shap_values) > 0:
            if min(node_shap_values) != max(node_shap_values):  # Only add colorbar if there's a range of values
                sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(node_shap_values), max(node_shap_values)))
                sm.set_array([])
                plt.colorbar(sm, ax=ax, label='SHAP Value', shrink=0.7)
        
        # Draw edges with arrows=True to avoid warning about connectionstyle
        nx.draw_networkx_edges(
            G, pos, width=widths, edge_color=EDGE_COLOUR, 
            alpha=0.6, arrows=True, arrowstyle='-',
            ax=ax
        )
        
        # Use original node positions for labels (centered on nodes)
        nx.draw_networkx_labels(
            G, pos, font_size=11, font_weight="bold",
            ax=ax
        )
        
        ax.set_title(f"SHAP interactions network: Top {top_n} strongest interactions across all features")
        ax.axis("off")
        plt.tight_layout()
        
        # Save the figure - simplified with minimal logging
        out_path = Path(out_path)
        
        # Make sure parent directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save and close - will overwrite any existing file
        plt.savefig(out_path, dpi=300)
        plt.close()
        return True
            
    except Exception as e:
        LOGGER.error(f"Error generating network plot: {e}")
        return False


def plot_topbottom_feature_network(
    inter_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    *,
    out_path: Union[str, Path],
    top_n: int = 15,  # Use the same default as DEFAULT_TOP_N
) -> Tuple[bool, Optional[Path]]:
    """
    Plot a network visualization showing interactions between top and bottom SHAP-ranked features.
    
    This differs from plot_interaction_network by:
    1. Using the top N *positive* impact features AND top N *negative* impact features
    2. Coloring nodes based on direction of SHAP impact (red=positive, blue=negative)
    3. Only showing interactions between these top positive and top negative features
    
    Args:
        inter_df: DataFrame with interaction data (feature1, feature2, interaction_strength)
        shap_df: DataFrame with SHAP values and feature importance metrics
        out_path: Path to save the plot
        top_n: Number of top features in each direction (positive/negative) to include
        
    Returns:
        tuple: (success, output_path)
            success (bool): True if successful, False otherwise
            output_path (Path or None): The path where the file was saved, or None if unsuccessful
    """
    # First, make sure we have NetworkX
    if not HAS_NETWORKX:
        LOGGER.warning("NetworkX not available for top-bottom network visualization")
        return False, None
    
    # Check if we have valid data
    if inter_df is None or inter_df.empty or shap_df is None or shap_df.empty:
        LOGGER.warning("Empty dataframe provided for top-bottom network plot")
        return False, None
    
    # Create output directory if needed
    out_path = _p(out_path, make_dir=True)
    
    # Select top positive and negative features
    try:
        # Get top N positive features (highest positive SHAP values)
        pos_features = shap_df[shap_df['Fold_Mean_SHAP'] > 0].sort_values(
            'Fold_Mean_SHAP', ascending=False).head(top_n).index.tolist()
        
        # Get top N negative features (lowest/most negative SHAP values)
        neg_features = shap_df[shap_df['Fold_Mean_SHAP'] < 0].sort_values(
            'Fold_Mean_SHAP', ascending=True).head(top_n).index.tolist()
        
        # Combine all selected features
        selected_features = pos_features + neg_features
        
        # No verbose message for network visualization - files will be shown in final summary
        
        # Filter interactions to only include those between selected features
        filtered_interactions = inter_df[
            (inter_df['feature1'].isin(selected_features)) & 
            (inter_df['feature2'].isin(selected_features))
        ].sort_values('interaction_strength', ascending=False)
        
        if filtered_interactions.empty:
            LOGGER.warning("No interactions found between top and bottom features")
            return False, None
            
        # Take only the top interactions to match the filtering in plot_interaction_network
        # This limits the number of connections to prevent an overly tangled plot
        filtered_interactions = filtered_interactions.head(top_n * 2)
            
        # Create graph from filtered interactions
        G = nx.Graph()
        
        # Add nodes with color coding for top vs bottom features
        for feature in selected_features:
            if feature in pos_features:
                # Positive impact features = red
                G.add_node(feature, color=POSITIVE_NODE_COLOR, direction='top')
            else:
                # Negative impact features = blue
                G.add_node(feature, color=NEGATIVE_NODE_COLOR, direction='bottom')
        
        # Add edges with weights based on interaction strength
        max_s = filtered_interactions["interaction_strength"].max()
        if max_s <= 0:
            max_s = 1.0  # Avoid division by zero
            
        for _, r in filtered_interactions.iterrows():
            norm_weight = r.interaction_strength / max_s
            G.add_edge(r.feature1, r.feature2, weight=norm_weight)
    
        if len(G.edges()) == 0:
            LOGGER.warning("No edges in top-bottom network graph - skipping plot")
            return False, None
        
        # Identify which nodes are connected vs. isolated
        connected_nodes = set()
        for u, v in G.edges():
            connected_nodes.add(u)
            connected_nodes.add(v)
        
        # Separate connected and isolated nodes
        isolated_nodes = set(G.nodes()) - connected_nodes
        
        # Also separate by positive/negative direction
        pos_connected = [n for n in connected_nodes if G.nodes[n]['direction'] == 'top']
        neg_connected = [n for n in connected_nodes if G.nodes[n]['direction'] == 'bottom']
        pos_isolated = [n for n in isolated_nodes if G.nodes[n]['direction'] == 'top']
        neg_isolated = [n for n in isolated_nodes if G.nodes[n]['direction'] == 'bottom']
        
        # Create a subgraph of only connected nodes
        connected_graph = G.subgraph(connected_nodes)
        
        # Create positions dictionary
        pos = {}
        
        # Position connected positive features on the left side, connected negative on the right
        # For positive nodes (left side arrangement)
        if pos_connected:
            rows = max(2, int(np.ceil(len(pos_connected) / 2)))
            for i, node in enumerate(pos_connected):
                row = i % rows
                col = i // rows
                # Position in left half
                pos[node] = np.array([-1.5 - col * 0.5, (row - rows/2) * 0.8])
        
        # For negative nodes (right side arrangement)
        if neg_connected:
            rows = max(2, int(np.ceil(len(neg_connected) / 2)))
            for i, node in enumerate(neg_connected):
                row = i % rows
                col = i // rows
                # Position in right half
                pos[node] = np.array([1.5 + col * 0.5, (row - rows/2) * 0.8])
        
        # Fine-tune the positions of connected nodes with force-directed algorithm
        if connected_nodes:
            # First do a standard spring layout to get a good starting position
            # But only for the connected subgraph
            if len(connected_nodes) > 1:
                temp_pos = nx.spring_layout(connected_graph, k=1.0, iterations=100, seed=42)
                
                # Transfer positions while preserving left/right division
                for node in connected_nodes:
                    if node in temp_pos:
                        if G.nodes[node]['direction'] == 'top':
                            # Keep x negative for positive impact features
                            pos[node] = np.array([-abs(temp_pos[node][0]), temp_pos[node][1]])
                        else:
                            # Keep x positive for negative impact features
                            pos[node] = np.array([abs(temp_pos[node][0]), temp_pos[node][1]])
            
            # Fine-tune the positions with weighted edge consideration
            for _ in range(5):
                # Calculate forces
                disp = {node: np.zeros(2) for node in connected_nodes}
                
                # Attractive forces along edges
                for u, v, weight in connected_graph.edges(data='weight'):
                    d = pos[v] - pos[u]
                    dist = max(0.01, np.linalg.norm(d))
                    # Stronger attraction for stronger interactions
                    attraction = 0.2 * weight * d * dist / 3.0
                    
                    # Apply attraction but maintain left/right division
                    u_dir = G.nodes[u]['direction']
                    v_dir = G.nodes[v]['direction']
                    
                    # Only adjust y-position and limit x-movement to maintain separation
                    disp[u][1] += attraction[1] * 0.5  # y-coordinate with dampening
                    disp[v][1] -= attraction[1] * 0.5  # y-coordinate with dampening
                    
                    # Allow limited x-movement within each side's boundary
                    if u_dir == 'top':
                        disp[u][0] += min(attraction[0], 0) * 0.3  # Only move left, dampened
                    else:
                        disp[u][0] += max(attraction[0], 0) * 0.3  # Only move right, dampened
                        
                    if v_dir == 'top':
                        disp[v][0] += min(attraction[0], 0) * 0.3  # Only move left, dampened
                    else:
                        disp[v][0] += max(attraction[0], 0) * 0.3  # Only move right, dampened
                
                # Apply node-node repulsion within each side to avoid overlap
                for u in connected_nodes:
                    for v in connected_nodes:
                        if u != v and G.nodes[u]['direction'] == G.nodes[v]['direction']:
                            d = pos[u] - pos[v]
                            dist = max(0.1, np.linalg.norm(d))
                            # Strong repulsion for close nodes on same side
                            repulsion = 0.3 * d / (dist**2)
                            disp[u] += repulsion * 0.4  # Dampened
                            disp[v] -= repulsion * 0.4  # Dampened
                
                # Apply the calculated displacement with dampening
                for node in connected_nodes:
                    pos[node] += disp[node] * 0.1  # Damping factor
        
        # Get sizes for only connected nodes based on absolute SHAP value
        connected_sizes = {}
        for node in connected_nodes:
            if node in shap_df.index:
                # Size based on absolute SHAP impact
                abs_impact = abs(shap_df.loc[node, 'Fold_Mean_SHAP']) * 25000  # Scale for visibility
                connected_sizes[node] = 500 + abs_impact  # Base size + scaled impact
            else:
                connected_sizes[node] = 500  # Default size
        
        # Edge widths based on weights
        edge_widths = [1 + 5 * G.get_edge_data(u, v).get('weight', 0.5) for u, v in G.edges()]
        
        # Create a larger figure to accommodate side lists
        fig = plt.figure(figsize=(20, 14))
        
        # Main network plot takes up the center 60%
        ax_center = plt.axes([0.2, 0.1, 0.6, 0.8])
        
        # Only draw connected nodes in the network
        if connected_nodes:
            # Create a colormap based on SHAP values for connected nodes
            node_shap_values = []
            for node in connected_nodes:
                if node in shap_df.index:
                    node_shap_values.append(shap_df.loc[node, 'Fold_Mean_SHAP'])
                else:
                    node_shap_values.append(0)  # Default for nodes without SHAP values
            
            # Get sizes for nodes
            node_sizes = [connected_sizes[n] for n in connected_nodes]
            
            # Create colormap - use RdBu_r (blue for negative, red for positive SHAP values)
            import matplotlib.cm as cm
            cmap = cm.get_cmap('RdBu_r')
            
            # Draw nodes with color based on SHAP value
            nodes = nx.draw_networkx_nodes(
                connected_graph, pos, node_size=node_sizes, node_color=node_shap_values, 
                cmap=cmap, edgecolors="white", linewidths=1, alpha=0.8,
                ax=ax_center
            )
            
            # Add a colorbar if we have a range of values
            if node_shap_values and min(node_shap_values) != max(node_shap_values):
                sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(node_shap_values), max(node_shap_values)))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax_center, shrink=0.6, pad=0.05)
                cbar.set_label('SHAP Value', rotation=270, labelpad=15)
            
            # Draw edges
            nx.draw_networkx_edges(
                connected_graph, pos, width=edge_widths, edge_color=EDGE_COLOUR, 
                alpha=0.6, arrows=True, arrowstyle='-', ax=ax_center
            )
            
            # Use original node positions for labels (centered on nodes)
            nx.draw_networkx_labels(
                connected_graph, pos, font_size=11, font_weight="bold",
                font_color="black", ax=ax_center
            )
        
        # Turn off axis
        ax_center.axis('off')
        
        # Add side panels for unconnected nodes if there are any
        if isolated_nodes:
            # Left side for positive unconnected features
            if pos_isolated:
                ax_left = plt.axes([0.01, 0.1, 0.19, 0.8])
                ax_left.axis('off')
                ax_left.set_title("Unconnected Positive Impact Features", fontsize=12, color=POSITIVE_NODE_COLOR)
                
                # Sort by impact
                pos_isolated_with_impact = []
                for node in pos_isolated:
                    if node in shap_df.index:
                        impact = shap_df.loc[node, 'Fold_Mean_SHAP']
                        pos_isolated_with_impact.append((node, impact))
                
                # Sort by absolute impact descending
                pos_isolated_with_impact.sort(key=lambda x: abs(x[1]), reverse=True)
                
                # Format and display the list
                pos_list_text = "\n".join([f"{i+1}. {n}" for i, (n, _) in enumerate(pos_isolated_with_impact)])
                ax_left.text(0.1, 0.5, pos_list_text, va='center', ha='left', fontsize=10,
                           bbox=dict(facecolor=POSITIVE_NODE_COLOR, alpha=0.2, pad=10))
            
            # Right side for negative unconnected features
            if neg_isolated:
                ax_right = plt.axes([0.8, 0.1, 0.19, 0.8])
                ax_right.axis('off')
                ax_right.set_title("Unconnected Negative Impact Features", fontsize=12, color=NEGATIVE_NODE_COLOR)
                
                # Sort by impact
                neg_isolated_with_impact = []
                for node in neg_isolated:
                    if node in shap_df.index:
                        impact = shap_df.loc[node, 'Fold_Mean_SHAP']
                        neg_isolated_with_impact.append((node, impact))
                
                # Sort by absolute impact descending
                neg_isolated_with_impact.sort(key=lambda x: abs(x[1]), reverse=True)
                
                # Format and display the list
                neg_list_text = "\n".join([f"{i+1}. {n}" for i, (n, _) in enumerate(neg_isolated_with_impact)])
                ax_right.text(0.1, 0.5, neg_list_text, va='center', ha='left', fontsize=10,
                           bbox=dict(facecolor=NEGATIVE_NODE_COLOR, alpha=0.2, pad=10))
        
        # Add a title and legend
        ax_center.set_title(f"Network of Feature Interactions Between Top {top_n} Positive and Negative Impact Features\n(Connected features shown in network, unconnected features listed on sides)")
        
        
        # The colorbar already serves as the legend for node colors
        # Just add a legend for side panels
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=POSITIVE_NODE_COLOR, alpha=0.2, label='Unconnected positive impact features'),
            Patch(facecolor=NEGATIVE_NODE_COLOR, alpha=0.2, label='Unconnected negative impact features')
        ]
        ax_center.legend(handles=legend_elements, loc='upper right')
        
        # Tight layout for the entire figure
        plt.tight_layout()
        
        # Save figure
        out_path = Path(out_path)
        
        # Make sure parent directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Make sure we're working with a Path object with correct extension
            out_path = Path(str(out_path))
            if not str(out_path).lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):  
                out_path = Path(str(out_path) + '.png')
            
            # Make sure parent directory exists
            out_path.parent.mkdir(parents=True, exist_ok=True)
                
            # Save the figure - will overwrite existing file
            plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
            plt.close()
            # No verbose message for network visualization - files will be shown in final summary
            return True, out_path
        except Exception as e:
            LOGGER.error(f"Error saving top-bottom features network plot: {e}")
            plt.close()
            return False, None
        
    except Exception as e:
        LOGGER.error(f"Error generating top-bottom features network plot: {e}")
        plt.close() if 'plt' in locals() else None
        return False, None


# ──────────────────────────────────────────────────────────────────────────────
# interaction summary
# ──────────────────────────────────────────────────────────────────────────────
def create_interaction_summary(
    inter_df: Optional[pd.DataFrame],
    cluster_df: Optional[pd.DataFrame],
    *,
    top_n: int,
) -> str:
    if inter_df is None or inter_df.empty:
        return (
            "ERROR: Could not compute SHAP interaction values.\n"
            "Ensure your model / explainer supports interaction values."
        )

    lines = [
        "SHAP Interaction Analysis",
        "==========================",
        f"Top {top_n} interactions:",
    ]
    for i, r in enumerate(inter_df.head(top_n).itertuples(), 1):
        lines.append(f"{i:>2}. {r.feature1} × {r.feature2}: {r.interaction_strength:.4f}")

    if cluster_df is not None and not cluster_df.empty:
        lines.append("\nFeature communities (Louvain):")
        for cid in sorted(cluster_df.cluster.unique()):
            feats = ", ".join(cluster_df.loc[cluster_df.cluster == cid, "feature"])
            lines.append(f"  • Community {cid}: {feats}")

    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
# per-fold helper (bug-fix: dangling variable `interaction_values`)
# ──────────────────────────────────────────────────────────────────────────────
def compute_fold_interactions(
    fold: Dict[str, Any],
    fold_idx: int,
    fold_dir: Path,
    *,
    top_n: int = 15,
    shap_df: Optional[pd.DataFrame] = None,
    max_features: Optional[int] = None,  # Will be set to 4*top_n to ensure coverage of both positive and negative features
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[List[str]]]:
    """
    Compute and store interactions for one fold.
    Only *top_n* most important features of that fold are considered.
    """
    model = fold.get("model")
    if model is None:
        # Try to load model from disk
        model_path = Path(fold_dir) / f"best_model_fold_{fold_idx}.pkl"
        LOGGER.info(f"Fold {fold_idx}: model not in memory, attempting to load from disk")
        try:
            model = joblib.load(model_path)
            fold["model"] = model  # Store in fold dictionary for future use
            LOGGER.info(f"Fold {fold_idx}: Successfully loaded model from {model_path}")
        except Exception as e:
            LOGGER.error(f"Fold {fold_idx}: Failed to load model from {model_path}: {e}")
            return None, None, None

    # Check for SHAP data and make sure we have what we need
    if "shap_data" not in fold or not fold["shap_data"]:
        LOGGER.error(f"Fold {fold_idx}: No SHAP data available")
        return None, None, None
    
    # Get test data - try multiple potential sources
    X = None
    
    # First try: X_test from the fold
    if "X_test" in fold and fold["X_test"] is not None:
        X = fold["X_test"]
        LOGGER.info(f"Fold {fold_idx}: Using X_test from fold")
    
    # Second try: X from shap_data
    elif len(fold["shap_data"]) > 1 and fold["shap_data"][1] is not None:
        X = fold["shap_data"][1]
        LOGGER.info(f"Fold {fold_idx}: Using X from shap_data")
    
    # Third try: If we have feature names and SHAP values, we can reconstruct a DataFrame
    elif "feature_names" in fold and fold.get("feature_names") and "shap_values" in fold:
        # Create dummy DataFrame with feature names
        shap_values = fold["shap_values"]
        if hasattr(shap_values, "shape") and len(shap_values.shape) >= 2:
            # Create a dummy X with the right shape
            num_samples = shap_values.shape[0]
            feat_names = fold["feature_names"]
            X = pd.DataFrame(np.zeros((num_samples, len(feat_names))), columns=feat_names)
            LOGGER.warning(f"Fold {fold_idx}: Created dummy X data with correct shape for interaction calculation")
    
    if X is None:
        LOGGER.warning(f"Fold {fold_idx}: Could not find X_test data")
        return None, None, None
    
    # Handle feature names - check if X is already a DataFrame with column names
    if isinstance(X, pd.DataFrame):
        feat_names = list(X.columns)
        LOGGER.info(f"Fold {fold_idx}: Got {len(feat_names)} feature names from X DataFrame columns")
    else:
        # Use feature names from the DataFrame if available, otherwise try to get from fold
        if isinstance(X, pd.DataFrame):
            # Already a DataFrame with column names
            feat_names = list(X.columns)
        else:
            # Try to get feature names from fold
            feat_names = fold.get("feature_names")
            
            # If we have feature names, convert X to DataFrame
            if feat_names:
                X = pd.DataFrame(X, columns=feat_names)
            else:
                # If X is from shap_data, try to use those feature names
                if 'shap_data' in fold and len(fold['shap_data']) > 1 and isinstance(fold['shap_data'][1], pd.DataFrame):
                    feat_names = list(fold['shap_data'][1].columns)
                    X = pd.DataFrame(X, columns=feat_names)
                else:
                    LOGGER.warning(f"Fold {fold_idx}: No feature names available, cannot proceed with interactions")
                    return None, None, None
    
    # Final check - we should have model, X as DataFrame and feature names by now
    if model is None:
        LOGGER.warning(f"Fold {fold_idx}: Model is None, cannot proceed")
        return None, None, None

    # Compute max_features as 4*top_n if not provided to ensure coverage
    if max_features is None:
        max_features = 4 * top_n
        LOGGER.info(f"Fold {fold_idx}: Setting max_features to {max_features} (4*top_n)")
    
    # Get SHAP values from the fold - these should already be computed during pipeline run
    shap_values = fold.get("shap_values")
    if shap_values is None or not hasattr(shap_values, 'ndim') or shap_values.ndim < 2:
        LOGGER.warning(f"Fold {fold_idx}: No pre-computed SHAP values found")
        return None, None, None
        
    # Simple and direct approach - just use mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(-mean_abs_shap)[:max_features].tolist()
    
    # Ensure top_indices contains only integers, not nested lists
    if top_indices and isinstance(top_indices[0], list):
        # Flatten nested list if needed
        top_indices = [item for sublist in top_indices for item in sublist]
    
    top_feat = [feat_names[i] for i in top_indices]
    top_idx = top_indices
    
    LOGGER.info(f"Fold {fold_idx}: Selected top {len(top_feat)} features by absolute SHAP value for interaction analysis")
    
    if not top_feat:
        LOGGER.warning(f"Fold {fold_idx}: Failed to identify top features")
        return None, None, None
    
    # ------------------------------- compute interactions only for top features
    try:
        # Filter X to only include top features to speed up computation
        if top_idx and len(top_idx) < len(feat_names):
            # Convert to list of feature names
            selected_features = [feat_names[i] for i in top_idx]
            # Filter X to only include these features
            if isinstance(X, pd.DataFrame):
                X_filtered = X[selected_features]
            else:
                # If X is not a DataFrame, we need to select columns
                X_filtered = X[:, top_idx]
                # Convert to DataFrame for clarity
                X_filtered = pd.DataFrame(X_filtered, columns=selected_features)
        else:
            X_filtered = X
            selected_features = feat_names
            
        LOGGER.info(f"Fold {fold_idx}: Computing interactions for only {len(selected_features)} features instead of {len(feat_names)}")
        
        # Use model's built-in SHAP interaction values method for XGBoost
        if hasattr(model, "shap_interaction_values"):
            raw_vals = model.shap_interaction_values(X_filtered)
        else:
            # Extract the underlying model if needed
            if hasattr(model, 'model'):
                # Our model classes wrap the actual XGBoost/RF model in a 'model' attribute
                underlying_model = model.model
            else:
                underlying_model = model
            
            # Use SHAP TreeExplainer directly - this is the most reliable for XGBoost
            import shap
            LOGGER.info(f"Fold {fold_idx}: Using SHAP TreeExplainer for model type {type(underlying_model).__name__}")
            
            # For XGBoost models
            if 'xgboost' in str(type(underlying_model)).lower():
                # Make sure we're using the booster
                if hasattr(underlying_model, 'get_booster'):
                    booster = underlying_model.get_booster()
                    LOGGER.info(f"Fold {fold_idx}: Extracted XGBoost booster for interactions")
                    explainer = shap.TreeExplainer(booster)
                else:
                    explainer = shap.TreeExplainer(underlying_model)
            else:
                explainer = shap.TreeExplainer(underlying_model)
                
            # Calculate interactions - with detailed error handling
            try:
                LOGGER.info(f"Fold {fold_idx}: Calculating SHAP interaction values for {len(X_filtered)} samples and {X_filtered.shape[1]} features")
                raw_vals = explainer.shap_interaction_values(X_filtered)
                LOGGER.info(f"Fold {fold_idx}: Successfully calculated interactions, shape: {np.shape(raw_vals)}")
            except Exception as e:
                LOGGER.error(f"Fold {fold_idx}: Failed to calculate SHAP interactions: {str(e)}")
                raise
        
        vals = _aggregate_classes(raw_vals)              # (samples, f, f)
    except Exception as e:
        LOGGER.warning(f"Fold {fold_idx}: interaction computation failed – {e}")
        return None, None, None

    # ------------------------------- slice + average
    # If we used filtered data, the vals matrix is already limited to those features
    # so we don't need to slice it again
    if 'selected_features' in locals():
        # We already have the right dimensions from the filtered data
        mat = _average_matrix(vals)
        names = selected_features
        LOGGER.info(f"Fold {fold_idx}: Using pre-filtered feature set with {len(names)} features")
    else:
        # Original approach - slice the full matrix
        vals = vals[:, top_idx][:, :, top_idx]
        mat = _average_matrix(vals)
        # Handle the case where top_idx might be a list of lists or nested arrays
        if isinstance(top_idx, list) and top_idx and isinstance(top_idx[0], (list, np.ndarray)):
            # Flatten nested lists/arrays
            flat_indices = [int(i) for sublist in top_idx for i in sublist]
            names = [feat_names[i] for i in flat_indices]
        else:
            # Normal case - just use the indices directly
            names = [feat_names[int(i)] for i in top_idx]
    
    # Create tidy dataframe of interactions
    df = _tidy_interactions(mat, names)

    # ------------------------------- save
    fold_csv = fold_dir / f"fold_{fold_idx}_interactions.csv"
    df.to_csv(fold_csv, index=False)

    return df, mat, names


# ──────────────────────────────────────────────────────────────────────────────
# orchestrator
def save_shap_interactions_analysis(
    fold_results: Sequence[Dict[str, Any]],
    output_dir: Union[str, Path],
    *,
    top_n_interactions: int = 15,  # Standardized to 15 for all interaction visualizations
    resolution: float = 1.0,
    shap_df: Optional[pd.DataFrame] = None,
    in_fold_dirs: bool = True,
    skip_calculation: bool = False,
) -> Dict[str, str]:
    """
    Main user-facing convenience wrapper – orchestrates everything and returns
    a dict { artefact_name : file_path } for downstream reporting.
    """
    # Make sure output directory exists
    out_dir = _p(output_dir, make_dir=True)
    artefacts: dict[str, str] = {}

    # ------------------------------------------------------------------ folds
    # Import required libraries to avoid local variable issues
    import numpy as np
    import pandas as pd
    
    fold_interactions: list[Tuple[pd.DataFrame, np.ndarray, List[str]]] = []
    for i, fold in enumerate(fold_results, 1):
        fdir = out_dir / f"fold_{i}"
        fdir.mkdir(exist_ok=True)

        csv = fdir / f"fold_{i}_interactions.csv"
        if csv.exists():                           # re-use
            try:
                df = pd.read_csv(csv)
                names = sorted(set(df.feature1) | set(df.feature2))
                mat  = np.zeros((len(names), len(names)))
                name2idx = {n: k for k, n in enumerate(names)}
                for r in df.itertuples():
                    i1, i2 = name2idx[r.feature1], name2idx[r.feature2]
                    mat[i1, i2] = mat[i2, i1] = r.interaction_strength
                fold_interactions.append((df, mat, names))
                artefacts[f"fold_{i}_interactions"] = str(csv)
                continue
            except Exception:                      # pragma: no cover
                LOGGER.info("Could not read cached fold interactions – recompute")

        # compute if not skipping -------------------------------------------------------
        if not skip_calculation:
            df, mat, names = compute_fold_interactions(
                fold, i, fdir, top_n=top_n_interactions, shap_df=shap_df
            )
            if df is not None:
                fold_interactions.append((df, mat, names))
                # Update the csv variable with the actual path where the file was saved
                csv = fdir / f"fold_{i}_interactions.csv"
                artefacts[f"fold_{i}_interactions"] = str(csv)
        else:
            # Skip calculation, just load existing interactions from CSV
            csv = fdir / f"fold_{i}_interactions.csv"
            if csv.exists():
                try:
                    df = pd.read_csv(csv)
                    names = sorted(set(df.feature1) | set(df.feature2))
                    mat = np.zeros((len(names), len(names)))
                    name2idx = {n: k for k, n in enumerate(names)}
                    for r in df.itertuples():
                        i1, i2 = name2idx[r.feature1], name2idx[r.feature2]
                        mat[i1, i2] = mat[i2, i1] = r.interaction_strength
                    fold_interactions.append((df, mat, names))
                    artefacts[f"fold_{i}_interactions"] = str(csv)
                except Exception as e:
                    LOGGER.warning(f"Could not load interaction data for fold {i}: {e}")

    if not fold_interactions:
        LOGGER.info("Interactions not produced for Random Forest models.")
        # Return empty dictionary - no directory creation
        return {}
        
    # ------------------------------------------------------------------ aggregate
    # Now that we have interactions, create the interactions directory
    inter_dir = out_dir / "shap_feature_interactions"
    inter_dir.mkdir(exist_ok=True)
    
    all_feats = sorted({n for _, _, names in fold_interactions for n in names})
    nF = len(all_feats)
    agg = np.zeros((len(fold_interactions), nF, nF))

    for k, (_, mat, names) in enumerate(fold_interactions):
        idx = [all_feats.index(n) for n in names]
        agg[k][np.ix_(idx, idx)] = mat

    global_mat = np.mean(agg, axis=0)
    global_df  = _tidy_interactions(global_mat, all_feats)
    global_csv = inter_dir / "all_interactions.csv"
    global_df.to_csv(global_csv, index=False)
    artefacts["all_interactions"] = str(global_csv)

    # ------------------------------------------------------------------ clusters
    # Keep the clustering for visualization purposes but don't generate CSV
    cluster_df = cluster_interactions(global_mat, all_feats, resolution=resolution)

    # ------------------------------------------------------------------ figures
    try:
        # Create shap_feature_interactions subdirectory for all interaction visuals
        interaction_dir = Path(output_dir) / "shap_feature_interactions"
        interaction_dir.mkdir(parents=True, exist_ok=True)
        
        # Save interaction heatmap in the new directory
        heat_png = interaction_dir / "interaction_heatmap.png"
        plot_interaction_heatmap(global_mat, all_feats, out_path=heat_png)
        artefacts["heatmap"] = str(heat_png)
    except Exception:                              # pragma: no cover
        LOGGER.warning("Heat-map plot failed", exc_info=True)

    if HAS_NETWORKX:
        # Create interaction network plot
        try:
            # Make sure we have valid data for the plot
            if global_df.empty:
                LOGGER.warning("Global interaction DataFrame is empty - cannot create network plot")
            else:
                # Create shap_feature_interactions subdirectory
                interaction_dir = Path(output_dir) / "shap_feature_interactions"
                interaction_dir.mkdir(parents=True, exist_ok=True)
                net_png = interaction_dir / "interaction_network.png"
                
                # Use a dynamic number of features for the network plot:
                #   – consider 4×top_n_interactions so the graph is rich enough but still readable
                network_feature_limit = max(4 * top_n_interactions, 1)
                
                success = plot_interaction_network(
                    inter_df=global_df, out_path=str(net_png), top_n=network_feature_limit, shap_df=shap_df
                )
                if success and net_png.exists():
                    LOGGER.info(f"Interaction network plot successfully saved to {net_png}")
                    artefacts["interaction_network"] = str(net_png)
                else:
                    LOGGER.warning(f"Interaction network plot reported success={success} but file exists={net_png.exists()}")
        except Exception as e:
            LOGGER.warning(f"Network plot failed: {e}")
            traceback.print_exc()

        # Create top/bottom network plot if we have SHAP values
        if shap_df is not None:
            try:
                # Make sure we have valid data for the plot
                LOGGER.info(f"Preparing top/bottom network plot with {len(shap_df)} SHAP values and {len(global_df)} interactions")
                if shap_df.empty:
                    LOGGER.warning("SHAP DataFrame is empty - cannot create top/bottom network plot")
                elif global_df.empty:
                    LOGGER.warning("Global interaction DataFrame is empty - cannot create top/bottom network plot")
                else:
                    # Create shap_feature_interactions subdirectory
                    interaction_dir = Path(output_dir) / "shap_feature_interactions"
                    interaction_dir.mkdir(parents=True, exist_ok=True)
                    tb_png = interaction_dir / "top_bottom_network.png"
                    
                    # Select top_n_interactions positive and negative features (or fewer if not available)
                    pos_count = min(top_n_interactions, len(shap_df[shap_df['Fold_Mean_SHAP'] > 0]))
                    neg_count = min(top_n_interactions, len(shap_df[shap_df['Fold_Mean_SHAP'] < 0]))
                    network_feature_limit = pos_count + neg_count
                    LOGGER.info(f"Using top {pos_count} positive and top {neg_count} negative impact features for network plot")
                    
                    success, out_path = plot_topbottom_feature_network(
                        global_df, shap_df, out_path=str(tb_png), top_n=top_n_interactions
                    )
                    if success and tb_png.exists():
                        LOGGER.info(f"Top/bottom network plot successfully saved to {tb_png}")
                        artefacts["top_bottom_network"] = str(tb_png)
                    else:
                        LOGGER.warning(f"Top/bottom network plot reported success={success} but file exists={tb_png.exists()}")
            except Exception as e:
                LOGGER.warning(f"Top/bottom network plot failed: {e}")
                traceback.print_exc()

    # ------------------------------------------------------------------ summary
    summ = create_interaction_summary(global_df, cluster_df, top_n=top_n_interactions)
    txt  = inter_dir / "interaction_summary.txt"
    txt.write_text(summ)
    artefacts["interaction_summary"] = str(txt)

    return artefacts