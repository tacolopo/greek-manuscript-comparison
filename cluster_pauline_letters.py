#!/usr/bin/env python3
"""
Script to perform clustering analysis on the Pauline letters.
This script takes the similarity results from the analysis and performs
hierarchical clustering to visualize relationships between Paul's letters.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import seaborn as sns

def load_results():
    """Load the similarity results."""
    results_path = os.path.join("pauline_analysis", "pauline_similarity_results.pkl")
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        print("Please run analyze_pauline_letters.py first.")
        return None
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results

def create_dendrogram(similarity_matrix, config_name, output_dir):
    """Create a dendrogram from the similarity matrix."""
    # Convert similarity matrix to distance matrix (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # Convert to condensed distance matrix (required by linkage)
    # Fill diagonal with zeros first to make sure it's a proper distance matrix
    np.fill_diagonal(distance_matrix.values, 0)
    condensed_distance = squareform(distance_matrix)
    
    # Compute hierarchical clustering
    linked = linkage(condensed_distance, method='average')
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create dendrogram
    dendrogram(
        linked,
        labels=similarity_matrix.index,
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True,
        leaf_font_size=12
    )
    
    # Add title and labels
    plt.title(f'Hierarchical Clustering of Pauline Letters - {config_name.replace("_", " ").title()}', fontsize=16)
    plt.xlabel('Letter', fontsize=14)
    plt.ylabel('Distance (1 - Similarity)', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f"pauline_dendrogram_{config_name}.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"pauline_dendrogram_{config_name}.pdf"))
    plt.close()

def create_clusters_report(results, output_dir):
    """Create a report on the clusters for each configuration."""
    report_path = os.path.join(output_dir, "clustering_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("======= PAULINE LETTERS CLUSTERING ANALYSIS =======\n\n")
        
        for config_name, data in results.items():
            f.write(f"\n==== {config_name.upper()} ====\n\n")
            
            # Get top pairs
            f.write("Most similar letter pairs:\n")
            for name1, name2, sim in data['most_similar']:
                f.write(f"  {name1} - {name2}: {sim:.4f}\n")
            
            # Get least similar pairs
            f.write("\nLeast similar letter pairs:\n")
            for name1, name2, sim in data['least_similar']:
                f.write(f"  {name1} - {name2}: {sim:.4f}\n")
            
            # Add overall statistics
            f.write(f"\nAverage similarity: {data['average']:.4f}\n")
            f.write(f"Standard deviation: {data['std_dev']:.4f}\n")
            
            # Add interpretation
            f.write("\nInterpretation:\n")
            if config_name == 'baseline':
                f.write("  The baseline configuration shows strong connections between the major letters\n")
                f.write("  (Romans, 1 & 2 Corinthians) and between the Pastoral Epistles (1 & 2 Timothy, Titus).\n")
            elif config_name == 'nlp_only':
                f.write("  The NLP-only configuration focuses on syntactic structures and shows very\n")
                f.write("  different relationships, with extremely high similarity between some letter pairs.\n")
            elif config_name == 'vocabulary_focused':
                f.write("  The vocabulary-focused configuration amplifies lexical choices in the letters,\n")
                f.write("  showing the strongest distinctions between groups of letters.\n")
            elif config_name == 'structure_focused':
                f.write("  The structure-focused configuration analyzes sentence patterns and transitions,\n")
                f.write("  revealing different relationships based on writing style rather than content.\n")
            
            f.write("\n")
    
    print(f"Clustering report saved to {report_path}")
    
    # Create summary matrix and save to CSV
    summary_matrix = {}
    for letter in results['baseline']['matrix'].index:
        summary_matrix[letter] = {}
        
    for letter1 in results['baseline']['matrix'].index:
        for letter2 in results['baseline']['matrix'].index:
            if letter1 != letter2:
                avg_sim = np.mean([results[config]['matrix'].loc[letter1, letter2] for config in results])
                summary_matrix[letter1][letter2] = avg_sim
    
    summary_df = pd.DataFrame(summary_matrix)
    summary_df.to_csv(os.path.join(output_dir, "similarity_matrix.csv"))
    
    # Also save as pickle for future use
    with open(os.path.join(output_dir, "similarity_matrix.pkl"), 'wb') as f:
        pickle.dump(summary_df, f)

def main():
    """Main function."""
    # Set output directory
    output_dir = "pauline_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results = load_results()
    if not results:
        return 1
    
    # Create dendrograms for each configuration
    for config_name, data in results.items():
        print(f"Creating dendrogram for {config_name} configuration...")
        create_dendrogram(data['matrix'], config_name, output_dir)
    
    # Create clusters report
    create_clusters_report(results, output_dir)
    
    print("\nClustering analysis complete. Results saved to", output_dir)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 