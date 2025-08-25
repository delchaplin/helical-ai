#!/usr/bin/env python3
import os, sys

# Make the repo root importable so 'src' (a package) is found
repo_root = os.path.dirname(__file__)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.helical_poc import main  # <-- import FROM THE PACKAGE, not from this file

if __name__ == "__main__":
    main()
