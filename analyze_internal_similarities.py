#!/usr/bin/env python3
"""
Script to analyze internal similarities within corpus groups:
1. Internal similarities within Marcus Aurelius' Meditations chapters
2. Internal similarities within Pauline letters
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List

def analyze_internal_similarities(similarity_matrices: Dict[str, pd.DataFrame]):
    """
    Analyze internal similarities within different corpora.
    
    Args:
        similarity_matrices: Dictionary mapping configuration names to similarity matrices
    """
    # Define groups for analysis
    auth_prefix = 'AUTH_'
    pauline_corpus = ['ROM', '1CO', '2CO', 'GAL', 'EPH', 'PHP', 'COL', '1TH', '2TH', '1TI', '2TI', 'TIT', 'PHM']
    
    # Results for each configuration and corpus
    results = {}
    
    for config_name, matrix in similarity_matrices.items():
        # Get all text names from matrix
        text_names = matrix.index.tolist()
        
        # Identify author's articles and Pauline letters
        author_texts = [name for name in text_names if name.startswith(auth_prefix)]
        pauline_texts = [name for name in text_names if name in pauline_corpus]
        
        # Skip if not enough texts
        if len(author_texts) < 2 or len(pauline_texts) < 2:
            print(f"Warning: Not enough texts available in {config_name} for comparison")
            continue
        
        # Calculate internal similarities for each corpus
        results[config_name] = {
            'author_internal': [],
            'pauline_internal': []
        }
        
        # Author internal pairs
        for book1, book2 in combinations(author_texts, 2):
            similarity = matrix.loc[book1, book2]
            results[config_name]['author_internal'].append((book1, book2, similarity))
        
        # Pauline internal pairs
        for book1, book2 in combinations(pauline_texts, 2):
            similarity = matrix.loc[book1, book2]
            results[config_name]['pauline_internal'].append((book1, book2, similarity))
    
    # Print detailed results
    print("\n===== INTERNAL SIMILARITY ANALYSIS =====\n")
    
    # Create a table for comparing internal similarities across configurations
    comparison_data = []
    
    for config_name, data in results.items():
        print(f"==== {config_name.upper()} ====")
        
        # Author internal similarities
        author_similarities = [s for _, _, s in data['author_internal']]
        author_avg = np.mean(author_similarities) if author_similarities else np.nan
        
        print(f"\nMeditations Internal Similarities:")
        print(f"  Pairs analyzed: {len(data['author_internal'])}")
        print(f"  Average similarity: {author_avg:.4f}")
        print(f"  Min similarity: {np.min(author_similarities):.4f}")
        print(f"  Max similarity: {np.max(author_similarities):.4f}")
        print(f"  Std Dev: {np.std(author_similarities):.4f}")
        
        # Pauline internal similarities
        pauline_similarities = [s for _, _, s in data['pauline_internal']]
        pauline_avg = np.mean(pauline_similarities) if pauline_similarities else np.nan
        
        print(f"\nPauline Internal Similarities:")
        print(f"  Pairs analyzed: {len(data['pauline_internal'])}")
        print(f"  Average similarity: {pauline_avg:.4f}")
        print(f"  Min similarity: {np.min(pauline_similarities):.4f}")
        print(f"  Max similarity: {np.max(pauline_similarities):.4f}")
        print(f"  Std Dev: {np.std(pauline_similarities):.4f}")
        
        # Add to comparison data
        comparison_data.append({
            'Config': config_name,
            'Meditations Internal': author_avg,
            'Pauline Internal': pauline_avg,
            'Difference': author_avg - pauline_avg
        })
        
        print("\nTop 5 most similar Meditations pairs:")
        most_similar = sorted(data['author_internal'], key=lambda x: x[2], reverse=True)[:5]
        for book1, book2, sim in most_similar:
            print(f"  {book1.replace('AUTH_', '')} - {book2.replace('AUTH_', '')}: {sim:.4f}")
            
        print("\nTop 5 most similar Pauline pairs:")
        most_similar = sorted(data['pauline_internal'], key=lambda x: x[2], reverse=True)[:5]
        for book1, book2, sim in most_similar:
            print(f"  {book1} - {book2}: {sim:.4f}")
        
        print("\n" + "-" * 50)
    
    # Print comparison table
    print("\n===== CORPUS INTERNAL SIMILARITY COMPARISON =====\n")
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Create a relative comparison
    print("\n===== RELATIVE SIMILARITY COMPARISON =====\n")
    for row in comparison_data:
        config = row['Config']
        med_val = row['Meditations Internal']
        paul_val = row['Pauline Internal']
        if med_val != 0 and paul_val != 0:
            if med_val > paul_val:
                rel_diff = (med_val - paul_val) / abs(paul_val) * 100
                print(f"{config}: Meditations internal similarity is {rel_diff:.1f}% higher than Pauline")
            else:
                rel_diff = (paul_val - med_val) / abs(med_val) * 100
                print(f"{config}: Pauline internal similarity is {rel_diff:.1f}% higher than Meditations")

def main():
    # Load similarity matrices from author_analysis directory
    matrices = {}
    configs = ['baseline', 'nlp_only', 'equal', 'vocabulary_focused', 'structure_focused']
    
    for config in configs:
        filepath = os.path.join('author_analysis', f'{config}_similarity.pkl')
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                matrices[config] = pickle.load(f)
        else:
            print(f"Warning: Could not find similarity matrix for {config}")
    
    if not matrices:
        print("Error: No similarity matrices found. Run compare_author_to_pauline.py first.")
        return 1
    
    # Analyze internal similarities
    analyze_internal_similarities(matrices)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 