"""Common visualization utilities for DAFTAR-ML."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Set common styling for all plots
def set_plot_style():
    """Set common styling for all visualizations."""
    # Use seaborn's whitegrid style as base
    sns.set_style("whitegrid")
    # Set larger font sizes for readability
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    # Higher DPI for clearer plots
    plt.rcParams['figure.dpi'] = 150
    # Modern color palette - using default matplotlib colors instead of custom
    # Transparent background for better embedding
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'


def save_plot(fig, output_path, tight_layout=True):
    """Save a figure to the specified path with standard formatting.
    
    Args:
        fig: Matplotlib figure object
        output_path: Path to save the figure
        tight_layout: Whether to apply tight_layout before saving
    """
    if tight_layout:
        fig.tight_layout()
    
    # Create directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with high quality
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def create_subplots(rows=1, cols=1, figsize=None):
    """Create figure and axes with standard sizing.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        figsize: Optional figure size tuple (width, height)
        
    Returns:
        Tuple of (figure, axes)
    """
    if figsize is None:
        # Calculate appropriate figure size based on rows and columns
        width = 7 * cols
        height = 5 * rows
        figsize = (width, height)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    return fig, axes 