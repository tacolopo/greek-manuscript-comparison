#!/usr/bin/env python3
"""
Script to clear previous analysis results and rerun the similarity weight iterations
and combined MDS visualization.
"""

import os
import shutil
import subprocess
import sys

def clear_directory(directory):
    """
    Clear all contents of a directory without removing the directory itself.
    
    Args:
        directory: Path to the directory to clear
    """
    if os.path.exists(directory):
        print(f"Clearing contents of {directory}...")
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"Error removing {item_path}: {e}")
                continue
    else:
        print(f"Directory {directory} does not exist, will be created.")
        os.makedirs(directory, exist_ok=True)


def main():
    """Main function."""
    # Directories to clear
    similarity_dir = "similarity_iterations"
    sensitivity_dir = "whole_book_sensitivity"
    
    # Clear directories
    print("Clearing previous analysis results...")
    clear_directory(similarity_dir)
    clear_directory(sensitivity_dir)
    
    # Run similarity weight iterations
    print("\nRunning similarity weight iterations...")
    try:
        subprocess.run(
            ["python3", "iterate_similarity_weights.py", 
             "--method", "hierarchical", 
             "--clusters", "8", 
             "--advanced-nlp"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running iterate_similarity_weights.py: {e}")
        return 1
    
    # Generate combined MDS visualization
    print("\nGenerating combined MDS visualization...")
    try:
        subprocess.run(
            ["python3", "generate_combined_mds.py"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running generate_combined_mds.py: {e}")
        return 1
    
    print("\nAll tasks completed successfully!")
    print(f"Check {similarity_dir} for detailed results and")
    print(f"{sensitivity_dir} for the combined visualization.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 