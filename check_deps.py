#!/usr/bin/env python3
"""
Script to check if all required dependencies for the weight iteration analysis are installed.
"""

import sys
import importlib.util
import subprocess

def check_module(module_name):
    """Check if a module is installed."""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False
    return True

required_modules = [
    'numpy',
    'pandas',
    'matplotlib',
    'seaborn',
    'sklearn',
    'tqdm',
    'networkx',
    'pyvis',
    'tabulate'
]

print("Checking required modules:")
missing_modules = []

for module in required_modules:
    if check_module(module):
        print(f"✅ {module}: Installed")
    else:
        print(f"❌ {module}: Missing")
        missing_modules.append(module)

if missing_modules:
    print("\nMissing modules:")
    install_cmd = "pip install " + " ".join(missing_modules)
    print(f"Run: {install_cmd}")
    
    # Attempt to check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("\nWarning: You don't appear to be in a virtual environment.")
        print("Consider creating one with: python3 -m venv venv && source venv/bin/activate")
else:
    print("\nAll required modules are installed!")
    
    # Check if our package is installed
    try:
        from src import similarity, multi_comparison, preprocessing, features
        print("\nProject modules:")
        print("✅ src.similarity: Found")
        print("✅ src.multi_comparison: Found")
        print("✅ src.preprocessing: Found")
        print("✅ src.features: Found")
    except ImportError as e:
        print(f"\nProject module error: {e}")
        print("Make sure you're running this script from the project root directory.") 