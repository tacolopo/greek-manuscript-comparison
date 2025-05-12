#!/usr/bin/env python3
"""
Script to analyze similarities between the Johannine letters, Petrine letters,
and Pauline corpus across all weight configurations.
"""

import os
import pickle
import numpy as np
import pandas as pd
from itertools import combinations

def main():
    """Analyze corpus similarities."""
    # List of weight configurations
    configs = ['baseline', 'nlp_only', 'equal', 'vocabulary_focused', 'structure_focused']
    base_dir = "similarity_iterations"
    
    # Define the groups we want to analyze
    johannine_letters = ['1JN', '2JN', '3JN']
    petrine_letters = ['1PE', '2PE']
    pauline_corpus = ['ROM', '1CO', '2CO', 'GAL', 'EPH', 'PHP', 'COL', '1TH', '2TH', '1TI', '2TI', 'TIT', 'PHM']
    
    # Store results for each configuration
    results = {}
    
    for config in configs:
        matrix_path = os.path.join(base_dir, config, 'similarity_matrix.pkl')
        
        try:
            with open(matrix_path, 'rb') as f:
                matrix = pickle.load(f)
                
                # Verify that we have all required books
                available_books = set(matrix.index)
                johannine_available = [book for book in johannine_letters if book in available_books]
                petrine_available = [book for book in petrine_letters if book in available_books]
                pauline_available = [book for book in pauline_corpus if book in available_books]
                
                if len(johannine_available) < 2 or len(petrine_available) < 2 or len(pauline_available) < 2:
                    print(f"Warning: Not enough books available in {config} for comparison")
                    continue
                
                # Calculate average similarities within each group
                results[config] = {
                    'johannine_pairs': [],
                    'petrine_pairs': [],
                    'pauline_pairs': []
                }
                
                # Extract similarities between pairs of Johannine letters
                for book1, book2 in combinations(johannine_available, 2):
                    similarity = matrix.loc[book1, book2]
                    results[config]['johannine_pairs'].append((book1, book2, similarity))
                
                # Extract similarities between pairs of Petrine letters
                for book1, book2 in combinations(petrine_available, 2):
                    similarity = matrix.loc[book1, book2]
                    results[config]['petrine_pairs'].append((book1, book2, similarity))
                
                # Extract similarities between pairs of Pauline letters
                for book1, book2 in combinations(pauline_available, 2):
                    similarity = matrix.loc[book1, book2]
                    results[config]['pauline_pairs'].append((book1, book2, similarity))
                
        except Exception as e:
            print(f"Error loading {config} matrix: {e}")
            continue
    
    # Print detailed results
    print("\n===== DETAILED SIMILARITY ANALYSIS =====")
    for config in configs:
        if config not in results:
            continue
        
        print(f"\n== {config.upper()} ==")
        
        # Johannine letters
        print("\nJohannine Letters:")
        johannine_similarities = [s for _, _, s in results[config]['johannine_pairs']]
        print(f"Pairs: {len(results[config]['johannine_pairs'])}")
        for book1, book2, sim in results[config]['johannine_pairs']:
            print(f"  {book1}-{book2}: {sim:.4f}")
        if johannine_similarities:
            print(f"  Average: {np.mean(johannine_similarities):.4f}")
            print(f"  Min: {np.min(johannine_similarities):.4f}")
            print(f"  Max: {np.max(johannine_similarities):.4f}")
            print(f"  Std Dev: {np.std(johannine_similarities):.4f}")
        
        # Petrine letters
        print("\nPetrine Letters:")
        petrine_similarities = [s for _, _, s in results[config]['petrine_pairs']]
        print(f"Pairs: {len(results[config]['petrine_pairs'])}")
        for book1, book2, sim in results[config]['petrine_pairs']:
            print(f"  {book1}-{book2}: {sim:.4f}")
        if petrine_similarities:
            print(f"  Average: {np.mean(petrine_similarities):.4f}")
            print(f"  Min: {np.min(petrine_similarities):.4f}")
            print(f"  Max: {np.max(petrine_similarities):.4f}")
            print(f"  Std Dev: {np.std(petrine_similarities):.4f}")
        
        # Pauline letters
        print("\nPauline Corpus:")
        pauline_similarities = [s for _, _, s in results[config]['pauline_pairs']]
        print(f"Pairs: {len(results[config]['pauline_pairs'])}")
        print(f"  Average: {np.mean(pauline_similarities):.4f}")
        print(f"  Min: {np.min(pauline_similarities):.4f}, Max: {np.max(pauline_similarities):.4f}")
        print(f"  Std Dev: {np.std(pauline_similarities):.4f}")
        
        # Instead of printing all Pauline pairs (too many), find most and least similar
        if pauline_similarities:
            most_similar = sorted(results[config]['pauline_pairs'], key=lambda x: x[2], reverse=True)[:3]
            least_similar = sorted(results[config]['pauline_pairs'], key=lambda x: x[2])[:3]
            
            print("  Most similar Pauline pairs:")
            for book1, book2, sim in most_similar:
                print(f"    {book1}-{book2}: {sim:.4f}")
                
            print("  Least similar Pauline pairs:")
            for book1, book2, sim in least_similar:
                print(f"    {book1}-{book2}: {sim:.4f}")
    
    # Print summary comparing the different corpora across configurations
    print("\n===== SUMMARY COMPARISON =====")
    summary_data = []
    
    for config in configs:
        if config not in results:
            continue
        
        johannine_avg = np.mean([s for _, _, s in results[config]['johannine_pairs']]) if results[config]['johannine_pairs'] else np.nan
        petrine_avg = np.mean([s for _, _, s in results[config]['petrine_pairs']]) if results[config]['petrine_pairs'] else np.nan
        pauline_avg = np.mean([s for _, _, s in results[config]['pauline_pairs']]) if results[config]['pauline_pairs'] else np.nan
        
        summary_data.append({
            'Config': config,
            'Johannine Avg': johannine_avg,
            'Petrine Avg': petrine_avg,
            'Pauline Avg': pauline_avg
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Add ranking of internal similarities
    print("\n===== INTERNAL SIMILARITY RANKING =====")
    for config in configs:
        if config not in results:
            continue
        
        values = {}
        if results[config]['johannine_pairs']:
            values['Johannine'] = np.mean([s for _, _, s in results[config]['johannine_pairs']])
        if results[config]['petrine_pairs']:
            values['Petrine'] = np.mean([s for _, _, s in results[config]['petrine_pairs']])
        if results[config]['pauline_pairs']:
            values['Pauline'] = np.mean([s for _, _, s in results[config]['pauline_pairs']])
        
        ranking = sorted(values.items(), key=lambda x: x[1], reverse=True)
        rank_str = ' > '.join([f"{name} ({value:.4f})" for name, value in ranking])
        print(f"{config.upper()}: {rank_str}")

if __name__ == "__main__":
    main() 