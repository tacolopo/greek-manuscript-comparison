#!/usr/bin/env python3
"""
Test script to understand why Euclidean distance similarities are so high.
"""

import numpy as np
import pandas as pd

def test_euclidean_formula():
    """Test our Euclidean distance formula with different parameters."""
    
    print("Testing Euclidean distance formula...")
    
    # Load actual similarity matrix
    df = pd.read_csv('exact_cleaned_analysis/nlp_only/similarity_matrix.csv', index_col=0)
    
    # Test with some example vectors (simulating syntactic features)
    # These represent different "syntactic fingerprints"
    vector_a = np.array([0.3, 0.2, 0.1, 0.05, 0.35, 0.0, 0.15, 0.0, 0.0, 0.0, 
                        0.4, 2.5, 0.1, 0.0, 0.05, 0.5, 0.2, 0.0, 0.0, 1.2])
    
    vector_b = np.array([0.35, 0.18, 0.0, 0.0, 0.47, 0.0, 0.12, 0.0, 0.0, 0.0,
                        0.3, 2.2, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 2.0])
    
    vector_c = np.array([0.14, 0.14, 0.07, 0.0, 0.43, 0.21, 0.07, 0.0, 0.0, 0.0,
                        0.57, 2.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0])
    
    vectors = [("Julian-like", vector_a), ("Julian-like 2", vector_b), ("Paul-like", vector_c)]
    
    print("\nVector magnitudes:")
    for name, vec in vectors:
        print(f"{name}: {np.linalg.norm(vec):.4f}")
    
    print("\nPairwise comparisons:")
    for i, (name1, vec1) in enumerate(vectors):
        for j, (name2, vec2) in enumerate(vectors):
            if i < j:
                # Calculate Euclidean distance
                euclidean_dist = np.linalg.norm(vec1 - vec2)
                
                # Our current formula
                max_dist = np.linalg.norm(vec1) + np.linalg.norm(vec2)
                normalized_dist = euclidean_dist / max_dist
                current_sim = np.exp(-3 * normalized_dist)
                
                # Alternative formulas
                alt1_sim = np.exp(-5 * normalized_dist)  # Steeper decay
                alt2_sim = np.exp(-10 * normalized_dist)  # Much steeper
                
                # Raw distance as percentage of max possible
                raw_percentage = (euclidean_dist / max_dist) * 100
                
                print(f"\n{name1} vs {name2}:")
                print(f"  Raw Euclidean distance: {euclidean_dist:.4f}")
                print(f"  Normalized distance: {normalized_dist:.4f} ({raw_percentage:.1f}% of max)")
                print(f"  Current formula (exp(-3*d)): {current_sim:.4f}")
                print(f"  Alternative 1 (exp(-5*d)): {alt1_sim:.4f}")
                print(f"  Alternative 2 (exp(-10*d)): {alt2_sim:.4f}")
    
    print("\nActual results from analysis:")
    julian_names = ['Τῷ αὐτῷ', 'φραγμεντυμ επιστολαε', 'Σαραπίωνι τῷ λαμπροτάτῳ', 
                   'Διονυσίῳ', 'Ἀνεπίγραφος ὑπὲρ Ἀργείων', 'Λιβανίῳ σοφιστῇ καὶ κοιαίστωρι']
    pauline_names = ['2CO', '1CO', 'ROM', '2TI', '1TI', 'PHM', 'EPH', '2TH', 'PHP', 'COL', '1TH', 'TIT', 'GAL']
    
    cross_sims = [df.loc[j, p] for j in julian_names for p in pauline_names]
    
    print(f"Julian vs Pauline similarities:")
    print(f"  Range: {min(cross_sims):.4f} to {max(cross_sims):.4f}")
    print(f"  Mean: {np.mean(cross_sims):.4f}")
    print(f"  Standard deviation: {np.std(cross_sims):.4f}")
    
    # Check if the problem is that syntactic features are too similar
    print(f"\nProblem diagnosis:")
    print(f"  Standard deviation of {np.std(cross_sims):.4f} suggests limited differentiation")
    print(f"  Mean of {np.mean(cross_sims):.4f} suggests generally high similarities")
    print(f"  This could indicate:")
    print(f"    1. Ancient Greek syntactic patterns are genuinely similar across authors")
    print(f"    2. Our exponential decay factor (-3) is too gentle")
    print(f"    3. Our normalization method compresses differences")

if __name__ == "__main__":
    test_euclidean_formula() 