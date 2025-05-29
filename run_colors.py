#!/usr/bin/env python3
"""
Direct runner for DAFTAR-ML color visualization tool.

This script allows running the colors visualization tool directly from
the repository without needing to install the package.

Usage:
    python run_colors.py [--output_dir OUTPUT_DIR]
    python run_colors.py [--path]
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(current_dir))

# Import and run the main function from the colors module
from daftar.tools.colors import main, run_main

if __name__ == "__main__":
    run_main()
