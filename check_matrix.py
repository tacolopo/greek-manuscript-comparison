#!/usr/bin/env python3
"""
Script to check the similarity matrices for all weight configurations.
"""

import os
import pickle
import numpy as np
import pandas as pd

def main():
    """Check all similarity matrices."""
    # List of weight configurations
    configs = ['baseline', 'nlp_only', 'equal', 'vocabulary_focused', 'structure_focused']
    base_dir = "similarity_iterations"
    
    for config in configs:
        matrix_path = os.path.join(base_dir, config, 'similarity_matrix.pkl')
        
        try:
            with open(matrix_path, 'rb') as f:
                matrix = pickle.load(f)
                
                # Get non-diagonal elements
                mask = ~np.eye(matrix.shape[0], dtype=bool)
                non_diag = matrix.values[mask]
                
                print(f"\n=== {config.upper()} MATRIX STATISTICS ===")
                print(f"Shape: {matrix.shape}")
                print(f"All zeros: {(matrix.values == 0).all()}")
                print(f"All non-diagonal zeros: {(non_diag == 0).all()}")
                print(f"Any non-diagonal zeros: {(non_diag == 0).any()}")
                print(f"Number of zeros: {np.sum(non_diag == 0)} out of {len(non_diag)}")
                print(f"Average: {non_diag.mean():.6f}")
                print(f"Min: {non_diag.min():.6f}")
                print(f"Max: {non_diag.max():.6f}")
                print(f"Std Dev: {non_diag.std():.6f}")
                
        except Exception as e:
            print(f"Error loading {config} matrix: {e}")
            continue

if __name__ == "__main__":
    main() 