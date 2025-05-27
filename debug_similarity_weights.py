#!/usr/bin/env python3
"""
Debug script to investigate why equal and baseline similarity matrices are coming back similar.
This script will:
1. Load the similarity matrices from both configurations
2. Compare them to see how similar they are
3. Check if the SimilarityCalculator is correctly applying weights
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try imports with appropriate error handling
try:
    from src.similarity import SimilarityCalculator
    from src.feature_extraction import extract_all_features
except ImportError as e:
    print(f"Error: {e}")
    print("Could not import required modules. Make sure you're running from the project root.")
    sys.exit(1)

def load_similarity_matrices(base_dir):
    """
    Load baseline and equal weight matrices from the given directory
    """
    baseline_path = os.path.join(base_dir, 'baseline', 'similarity_matrix.csv')
    equal_path = os.path.join(base_dir, 'equal', 'similarity_matrix.csv')
    
    print(f"Loading baseline matrix from: {baseline_path}")
    print(f"Loading equal weights matrix from: {equal_path}")
    
    if not os.path.exists(baseline_path) or not os.path.exists(equal_path):
        print("Error: One or both similarity matrices not found!")
        print(f"Baseline exists: {os.path.exists(baseline_path)}")
        print(f"Equal exists: {os.path.exists(equal_path)}")
        return None, None
    
    baseline_df = pd.read_csv(baseline_path, index_col=0)
    equal_df = pd.read_csv(equal_path, index_col=0)
    
    return baseline_df, equal_df

def compare_matrices(baseline_df, equal_df):
    """
    Compare the baseline and equal matrices to determine similarity
    """
    if baseline_df is None or equal_df is None:
        return
    
    # Check if indices match
    if not baseline_df.index.equals(equal_df.index) or not baseline_df.columns.equals(equal_df.columns):
        print("Warning: Matrix indices don't match! Reindexing for comparison.")
        # Get union of all indices
        all_indices = sorted(list(set(baseline_df.index) | set(equal_df.index)))
        # Reindex both matrices
        baseline_df = baseline_df.reindex(index=all_indices, columns=all_indices, fill_value=0)
        equal_df = equal_df.reindex(index=all_indices, columns=all_indices, fill_value=0)
    
    # Calculate element-wise difference
    diff_matrix = (baseline_df - equal_df).abs()
    
    # Calculate statistics
    mean_diff = diff_matrix.values.mean()
    max_diff = diff_matrix.values.max()
    min_diff = diff_matrix.values.min()
    
    print("\nDifference Statistics:")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Maximum absolute difference: {max_diff:.6f}")
    print(f"Minimum absolute difference: {min_diff:.6f}")
    
    # Calculate correlation
    flat_baseline = baseline_df.values.flatten()
    flat_equal = equal_df.values.flatten()
    correlation = np.corrcoef(flat_baseline, flat_equal)[0, 1]
    
    print(f"Correlation between matrices: {correlation:.6f}")
    
    # Visualize the differences
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Heatmap of the difference matrix
    plt.subplot(1, 3, 1)
    plt.imshow(diff_matrix.values, cmap='viridis')
    plt.colorbar(label='Absolute Difference')
    plt.title('Difference Between Matrices')
    
    # Plot 2: Histogram of differences
    plt.subplot(1, 3, 2)
    plt.hist(diff_matrix.values.flatten(), bins=50)
    plt.title('Histogram of Differences')
    plt.xlabel('Absolute Difference')
    plt.ylabel('Count')
    
    # Plot 3: Scatter plot of baseline vs equal
    plt.subplot(1, 3, 3)
    plt.scatter(flat_baseline, flat_equal, alpha=0.3)
    plt.plot([0, 1], [0, 1], 'r--')  # Line y=x for perfect correlation
    plt.title(f'Baseline vs Equal (r={correlation:.4f})')
    plt.xlabel('Baseline Similarity')
    plt.ylabel('Equal Weights Similarity')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "similarity_comparison.png")
    print(f"Saved visualization to debug_output/similarity_comparison.png")
    
    # Also save the diff matrix
    diff_matrix.to_csv(output_dir / "difference_matrix.csv")
    print(f"Saved difference matrix to debug_output/difference_matrix.csv")

def test_similarity_calculator():
    """
    Test if the SimilarityCalculator correctly applies different weights
    """
    print("\nTesting SimilarityCalculator weight application...")
    
    # Create several test texts with Greek content for more thorough testing
    test_texts = [
        # Ephesians opening (Pauline)
        """
        Παῦλος ἀπόστολος Χριστοῦ Ἰησοῦ διὰ θελήματος θεοῦ τοῖς ἁγίοις τοῖς οὖσιν ἐν Ἐφέσῳ 
        καὶ πιστοῖς ἐν Χριστῷ Ἰησοῦ, χάρις ὑμῖν καὶ εἰρήνη ἀπὸ θεοῦ πατρὸς ἡμῶν καὶ κυρίου 
        Ἰησοῦ Χριστοῦ.
        """,
        
        # Titus opening (Pauline)
        """
        Παῦλος δοῦλος θεοῦ, ἀπόστολος δὲ Ἰησοῦ Χριστοῦ κατὰ πίστιν ἐκλεκτῶν θεοῦ καὶ 
        ἐπίγνωσιν ἀληθείας τῆς κατ' εὐσέβειαν ἐπ' ἐλπίδι ζωῆς αἰωνίου, ἣν ἐπηγγείλατο ὁ 
        ἀψευδὴς θεὸς πρὸ χρόνων αἰωνίων.
        """,
        
        # Romans passage (Pauline)
        """
        οὐ γὰρ ἐπαισχύνομαι τὸ εὐαγγέλιον, δύναμις γὰρ θεοῦ ἐστιν εἰς σωτηρίαν παντὶ τῷ 
        πιστεύοντι, Ἰουδαίῳ τε πρῶτον καὶ Ἕλληνι. δικαιοσύνη γὰρ θεοῦ ἐν αὐτῷ ἀποκαλύπτεται 
        ἐκ πίστεως εἰς πίστιν, καθὼς γέγραπται· ὁ δὲ δίκαιος ἐκ πίστεως ζήσεται.
        """,
        
        # Hebrews passage (Non-Pauline)
        """
        Πολυμερῶς καὶ πολυτρόπως πάλαι ὁ θεὸς λαλήσας τοῖς πατράσιν ἐν τοῖς προφήταις 
        ἐπ' ἐσχάτου τῶν ἡμερῶν τούτων ἐλάλησεν ἡμῖν ἐν υἱῷ, ὃν ἔθηκεν κληρονόμον πάντων, 
        δι' οὗ καὶ ἐποίησεν τοὺς αἰῶνας·
        """,
        
        # Julian letter excerpt (Non-Biblical)
        """
        Ἐγὼ μὲν οὐδὲν προσδοκῶν τοιοῦτον ἐπανῆλθον ἀπὸ τῆς Γαλατίας, ἀλλ᾽ ὡς ἀνδρὶ 
        φιλοσόφῳ καὶ γενναίῳ πρέπον ἦν, προσῆλθόν σοι καὶ ἔδωκα τὴν χεῖρα, πειθόμενος
        ὅτι οὐδὲν ἄλλο πλὴν καλοῦ καὶ ἀγαθοῦ βουλεύσῃ περὶ ἐμοῦ.
        """
    ]
    
    # Define weight configurations to test
    weight_configs = [
        {'name': 'baseline', 'vocabulary': 0.7, 'sentence': 0.15, 'transitions': 0.15, 'ngrams': 0.0, 'syntactic': 0.0},
        {'name': 'equal', 'vocabulary': 0.2, 'sentence': 0.2, 'transitions': 0.2, 'ngrams': 0.2, 'syntactic': 0.2},
        {'name': 'nlp_only', 'vocabulary': 0.0, 'sentence': 0.0, 'transitions': 0.0, 'ngrams': 0.0, 'syntactic': 1.0},
        {'name': 'structure_focused', 'vocabulary': 0.1, 'sentence': 0.4, 'transitions': 0.4, 'ngrams': 0.05, 'syntactic': 0.05},
        {'name': 'vocabulary_focused', 'vocabulary': 0.5, 'sentence': 0.1, 'transitions': 0.1, 'ngrams': 0.2, 'syntactic': 0.1},
    ]
    
    # Store results for all text pairs and configurations
    results = []
    similarity_by_config = {config['name']: [] for config in weight_configs}
    
    # Run all pairwise comparisons
    for i in range(len(test_texts)):
        for j in range(i+1, len(test_texts)):
            text1 = test_texts[i]
            text2 = test_texts[j]
            pair_name = f"Text{i+1}-Text{j+1}"
            
            print(f"\nComparing {pair_name}:")
            
            # Calculate similarity with each configuration
            for config in weight_configs:
                calc = SimilarityCalculator()
                calc.set_weights(config['weights'])
                
                # Calculate similarity
                sim = calc.calculate_similarity(text1, text2)
                
                # Calculate component contributions
                components = calc.calculate_component_similarities(text1, text2)
                
                # Store results
                result = {
                    'pair': pair_name,
                    'config': config['name'],
                    'similarity': sim,
                    'components': components
                }
                results.append(result)
                similarity_by_config[config['name']].append(sim)
                
                # Print results
                print(f"  {config['name']}: {sim:.4f}")
                if components:
                    for comp, value in components.items():
                        print(f"    - {comp}: {value:.4f}")
    
    # Create comparative visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Compare average similarities by configuration
    plt.subplot(2, 2, 1)
    avgs = [np.mean(similarity_by_config[config['name']]) for config in weight_configs]
    plt.bar([config['name'] for config in weight_configs], avgs)
    plt.title('Average Similarity by Configuration')
    plt.ylabel('Average Similarity')
    plt.xticks(rotation=45)
    
    # Plot 2: Distribution of similarities for each configuration
    plt.subplot(2, 2, 2)
    for config in weight_configs:
        values = similarity_by_config[config['name']]
        plt.hist(values, alpha=0.5, label=config['name'], bins=10)
    plt.title('Distribution of Similarities')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 3: Component contributions for baseline vs equal
    plt.subplot(2, 2, 3)
    
    # Get average component contribution for each configuration
    comp_avgs = {}
    for config in ['baseline', 'equal']:
        comp_avgs[config] = {}
        for comp in ['vocabulary', 'sentence', 'transitions', 'ngrams', 'syntactic']:
            values = []
            for r in results:
                if r['config'] == config and r['components'] and comp in r['components']:
                    values.append(r['components'][comp])
            comp_avgs[config][comp] = np.mean(values) if values else 0
    
    # Plot component contributions
    x = np.arange(len(comp_avgs['baseline']))
    width = 0.35
    plt.bar(x - width/2, list(comp_avgs['baseline'].values()), width, label='baseline')
    plt.bar(x + width/2, list(comp_avgs['equal'].values()), width, label='equal')
    plt.xticks(x, list(comp_avgs['baseline'].keys()), rotation=45)
    plt.title('Component Contributions')
    plt.ylabel('Average Contribution')
    plt.legend()
    
    # Plot 4: Similarity variation by text pair
    plt.subplot(2, 2, 4)
    # Get unique pairs
    pairs = sorted(list(set(r['pair'] for r in results)))
    
    # For each pair, get similarity for baseline and equal
    baseline_sims = []
    equal_sims = []
    for pair in pairs:
        for r in results:
            if r['pair'] == pair:
                if r['config'] == 'baseline':
                    baseline_sims.append(r['similarity'])
                elif r['config'] == 'equal':
                    equal_sims.append(r['similarity'])
    
    # Plot pair similarities
    x = np.arange(len(pairs))
    width = 0.35
    plt.bar(x - width/2, baseline_sims, width, label='baseline')
    plt.bar(x + width/2, equal_sims, width, label='equal')
    plt.xticks(x, pairs, rotation=45)
    plt.title('Similarity by Text Pair')
    plt.ylabel('Similarity')
    plt.legend()
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "test_similarities.png")
    print(f"\nSaved test similarity visualization to debug_output/test_similarities.png")

def analyze_loaded_matrices(baseline_df, equal_df, directory):
    """
    Perform deeper analysis on the loaded matrices
    """
    print("\nPerforming deeper analysis on similarity matrices...")
    
    if baseline_df is None or equal_df is None:
        print("Cannot perform analysis - matrices not loaded")
        return
    
    # 1. Distribution analysis
    baseline_vals = baseline_df.values.flatten()
    equal_vals = equal_df.values.flatten()
    
    # Remove diagonal values (self-similarities)
    n = baseline_df.shape[0]
    baseline_vals = np.array([baseline_vals[i] for i in range(len(baseline_vals)) 
                             if i % (n+1) != 0])
    equal_vals = np.array([equal_vals[i] for i in range(len(equal_vals)) 
                          if i % (n+1) != 0])
    
    # Calculate statistics
    print("\nDistribution Statistics:")
    print(f"Baseline - Mean: {np.mean(baseline_vals):.4f}, Std: {np.std(baseline_vals):.4f}, "
          f"Min: {np.min(baseline_vals):.4f}, Max: {np.max(baseline_vals):.4f}")
    print(f"Equal - Mean: {np.mean(equal_vals):.4f}, Std: {np.std(equal_vals):.4f}, "
          f"Min: {np.min(equal_vals):.4f}, Max: {np.max(equal_vals):.4f}")
    
    # 2. Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Distribution histogram for baseline
    plt.subplot(2, 2, 1)
    plt.hist(baseline_vals, bins=20, alpha=0.7, color='blue')
    plt.title('Baseline Similarity Distribution')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    
    # Plot 2: Distribution histogram for equal
    plt.subplot(2, 2, 2)
    plt.hist(equal_vals, bins=20, alpha=0.7, color='green')
    plt.title('Equal Weights Similarity Distribution')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    
    # Plot 3: Overlay distributions
    plt.subplot(2, 2, 3)
    plt.hist(baseline_vals, bins=20, alpha=0.5, color='blue', label='Baseline')
    plt.hist(equal_vals, bins=20, alpha=0.5, color='green', label='Equal')
    plt.title('Overlaid Distributions')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 4: Boxplot comparison
    plt.subplot(2, 2, 4)
    plt.boxplot([baseline_vals, equal_vals], labels=['Baseline', 'Equal'])
    plt.title('Similarity Distribution Comparison')
    plt.ylabel('Similarity')
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{directory}_distribution_analysis.png")
    print(f"Saved distribution analysis to debug_output/{directory}_distribution_analysis.png")
    
    # 3. Find manuscripts with largest and smallest differences
    diff_matrix = (baseline_df - equal_df).abs()
    
    # Get the largest differences
    max_diff_idx = np.unravel_index(diff_matrix.values.argmax(), diff_matrix.shape)
    max_diff_docs = (diff_matrix.index[max_diff_idx[0]], diff_matrix.columns[max_diff_idx[1]])
    max_diff_val = diff_matrix.iloc[max_diff_idx]
    
    print("\nLargest similarity difference:")
    print(f"Documents: {max_diff_docs[0]} vs {max_diff_docs[1]}")
    print(f"Baseline similarity: {baseline_df.loc[max_diff_docs[0], max_diff_docs[1]]:.4f}")
    print(f"Equal weights similarity: {equal_df.loc[max_diff_docs[0], max_diff_docs[1]]:.4f}")
    print(f"Absolute difference: {max_diff_val:.4f}")
    
    # Find pairs with consistent similarities
    consistent_mask = diff_matrix.values < 0.01
    np.fill_diagonal(consistent_mask, False)  # Exclude diagonal
    consistent_pairs = np.where(consistent_mask)
    
    if len(consistent_pairs[0]) > 0:
        print(f"\nFound {len(consistent_pairs[0])} document pairs with nearly identical similarities")
        print("Sample of consistent pairs:")
        for i in range(min(5, len(consistent_pairs[0]))):
            doc1 = diff_matrix.index[consistent_pairs[0][i]]
            doc2 = diff_matrix.columns[consistent_pairs[1][i]]
            base_sim = baseline_df.loc[doc1, doc2]
            equal_sim = equal_df.loc[doc1, doc2]
            print(f"  {doc1} vs {doc2}: Baseline={base_sim:.4f}, Equal={equal_sim:.4f}, Diff={abs(base_sim-equal_sim):.4f}")
    else:
        print("\nNo document pairs with nearly identical similarities found")

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Debug similarity matrix differences')
    parser.add_argument('--dir', default='exact_cleaned_analysis', 
                        help='Directory containing baseline and equal subdirectories with similarity matrices')
    parser.add_argument('--all', action='store_true',
                        help='Also analyze matrices from other analysis directories')
    args = parser.parse_args()
    
    print("="*80)
    print("DEBUGGING SIMILARITY MATRIX DIFFERENCES")
    print("="*80)
    
    # Load matrices from the specified directory
    baseline_df, equal_df = load_similarity_matrices(args.dir)
    
    # Compare matrices if both were loaded successfully
    if baseline_df is not None and equal_df is not None:
        compare_matrices(baseline_df, equal_df)
        analyze_loaded_matrices(baseline_df, equal_df, os.path.basename(args.dir))
    
    # If --all flag is provided, analyze matrices from other directories
    if args.all:
        other_dirs = ['full_greek_analysis', 'pauline_analysis']
        for dir_name in other_dirs:
            if os.path.exists(dir_name) and dir_name != args.dir:
                print(f"\n{'-'*60}")
                print(f"ANALYZING MATRICES FROM {dir_name}")
                print(f"{'-'*60}")
                
                other_baseline, other_equal = load_similarity_matrices(dir_name)
                if other_baseline is not None and other_equal is not None:
                    compare_matrices(other_baseline, other_equal)
                    analyze_loaded_matrices(other_baseline, other_equal, os.path.basename(dir_name))
    
    # Test SimilarityCalculator
    test_similarity_calculator()
    
    print("\nDebug completed!")

if __name__ == "__main__":
    main() 