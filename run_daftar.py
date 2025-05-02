#!/usr/bin/env python
"""Main entry point for DAFTAR-ML.

This script serves as the main entry point for the DAFTAR-ML pipeline,
supporting both regression and classification tasks with a unified interface.
Maintains the original CLI parameter structure.
"""

import sys
from daftar.cli import main

if __name__ == "__main__":
    sys.exit(main())
