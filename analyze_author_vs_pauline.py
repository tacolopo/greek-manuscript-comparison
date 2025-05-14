#!/usr/bin/env python3
"""
Analyze similarities between author's articles and Pauline letters
across different weight configurations.
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from typing import Dict, List, Tuple

# Configuration
input_dir = 'author_analysis'
output_dir = 'author_analysis'
weight_configs = ['baseline', 'nlp_only', 'equal', 'vocabulary_focused', 'structure_focused']

def load_similarity_matrices():
    """Load all similarity matrices from the input directory."""
    similarity_matrices = {}
    for config in weight_configs:
        matrix_path = os.path.join(input_dir, f'{config}_similarity.pkl')
        try:
            with open(matrix_path, 'rb') as f:
                matrix = pickle.load(f)
                similarity_matrices[config] = matrix
                print(f"Loaded similarity matrix from {config}")
        except Exception as e:
            print(f"Error loading {config} similarity matrix: {e}")
    
    return similarity_matrices

def analyze_corpus_similarities(matrices: Dict[str, pd.DataFrame]):
    """
    Analyze similarities between and within author's articles and Pauline letters.
    
    Args:
        matrices: Dictionary mapping configuration names to similarity matrices
    """
    # Results for each configuration
    results = {}
    summary_data = []
    
    for config_name, matrix in matrices.items():
        # Get all text names from matrix
        text_names = matrix.index.tolist()
        
        # Identify author's articles and Pauline letters
        author_articles = [name for name in text_names if name.startswith('AUTH_')]
        pauline_letters = [name for name in text_names if not name.startswith('AUTH_')]
        
        # Calculate similarity statistics
        # 1. Average similarity within author's articles
        author_internal_sim = []
        for i, art1 in enumerate(author_articles):
            for art2 in author_articles[i+1:]:
                author_internal_sim.append(matrix.loc[art1, art2])
                
        author_internal_avg = np.mean(author_internal_sim) if author_internal_sim else np.nan
        author_internal_std = np.std(author_internal_sim) if author_internal_sim else np.nan
        
        # 2. Average similarity within Pauline letters
        pauline_internal_sim = []
        for i, let1 in enumerate(pauline_letters):
            for let2 in pauline_letters[i+1:]:
                pauline_internal_sim.append(matrix.loc[let1, let2])
                
        pauline_internal_avg = np.mean(pauline_internal_sim) if pauline_internal_sim else np.nan
        pauline_internal_std = np.std(pauline_internal_sim) if pauline_internal_sim else np.nan
        
        # 3. Average similarity between author's articles and Pauline letters
        cross_sim = []
        for art in author_articles:
            for let in pauline_letters:
                cross_sim.append(matrix.loc[art, let])
                
        cross_avg = np.mean(cross_sim) if cross_sim else np.nan
        cross_std = np.std(cross_sim) if cross_sim else np.nan
        
        # Store results
        results[config_name] = {
            'author_internal': {
                'avg': author_internal_avg,
                'std': author_internal_std,
                'values': author_internal_sim
            },
            'pauline_internal': {
                'avg': pauline_internal_avg,
                'std': pauline_internal_std,
                'values': pauline_internal_sim
            },
            'cross': {
                'avg': cross_avg,
                'std': cross_std,
                'values': cross_sim
            }
        }
        
        # Add to summary dataframe
        summary_data.append({
            'Config': config_name,
            'Author Internal Avg': author_internal_avg,
            'Author Internal StdDev': author_internal_std,
            'Pauline Internal Avg': pauline_internal_avg,
            'Pauline Internal StdDev': pauline_internal_std,
            'Author-Pauline Avg': cross_avg,
            'Author-Pauline StdDev': cross_std,
            'Avg Difference (Author - Pauline)': author_internal_avg - pauline_internal_avg,
            'Avg Difference (Author - Cross)': author_internal_avg - cross_avg,
            'Avg Difference (Pauline - Cross)': pauline_internal_avg - cross_avg
        })
        
        # Find most similar and most different pairs
        # Most similar author-pauline pair
        auth_paul_pairs = [(art, let, matrix.loc[art, let]) for art in author_articles for let in pauline_letters]
        most_similar_pair = max(auth_paul_pairs, key=lambda x: x[2])
        least_similar_pair = min(auth_paul_pairs, key=lambda x: x[2])
        
        print(f"\n=== {config_name.upper()} ===")
        print(f"Author internal similarity: {author_internal_avg:.4f} ± {author_internal_std:.4f}")
        print(f"Pauline internal similarity: {pauline_internal_avg:.4f} ± {pauline_internal_std:.4f}")
        print(f"Author-Pauline similarity: {cross_avg:.4f} ± {cross_std:.4f}")
        print(f"Most similar pair: {most_similar_pair[0]}-{most_similar_pair[1]} ({most_similar_pair[2]:.4f})")
        print(f"Least similar pair: {least_similar_pair[0]}-{least_similar_pair[1]} ({least_similar_pair[2]:.4f})")
    
    # Create summary table
    summary_df = pd.DataFrame(summary_data)
    print("\n=== SUMMARY TABLE ===")
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(summary_df.to_string(index=False))
    
    # Save summary to CSV
    summary_df.to_csv(os.path.join(output_dir, 'similarity_summary.csv'), index=False)
    
    return results

def plot_similarity_distributions(results):
    """Plot distributions of similarities for visual comparison."""
    # Create a figure
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 3*len(results)))
    
    # If only one configuration, make axes into a list for consistent indexing
    if len(results) == 1:
        axes = [axes]
    
    # Plot each configuration
    for i, (config, data) in enumerate(results.items()):
        ax = axes[i]
        
        # Get similarity values
        author_values = data['author_internal']['values']
        pauline_values = data['pauline_internal']['values']
        cross_values = data['cross']['values']
        
        # Plot distributions
        if author_values:
            sns.kdeplot(author_values, ax=ax, label="Author Internal", color="green")
        if pauline_values:
            sns.kdeplot(pauline_values, ax=ax, label="Pauline Internal", color="blue")
        if cross_values:
            sns.kdeplot(cross_values, ax=ax, label="Author-Pauline", color="purple")
        
        # Add vertical lines for means
        if author_values:
            ax.axvline(data['author_internal']['avg'], color="green", linestyle="--")
        if pauline_values:
            ax.axvline(data['pauline_internal']['avg'], color="blue", linestyle="--")
        if cross_values:
            ax.axvline(data['cross']['avg'], color="purple", linestyle="--")
        
        # Add labels and title
        ax.set_xlabel("Similarity Score")
        ax.set_ylabel("Density")
        ax.set_title(f"{config.replace('_', ' ').title()} Configuration")
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_distributions.png'), dpi=300)
    
def create_heatmap_visualization(matrices):
    """Create heatmap visualizations of the similarity matrices."""
    # For each configuration
    for config_name, matrix in matrices.items():
        plt.figure(figsize=(12, 10))
        
        # Sort matrix to group author articles and Pauline letters
        sorted_names = sorted(matrix.index, key=lambda x: "A" if x.startswith("AUTH_") else "B" + x)
        sorted_matrix = matrix.loc[sorted_names, sorted_names]
        
        # Create mask for upper triangle to make it more readable
        mask = np.triu(np.ones_like(sorted_matrix, dtype=bool))
        
        # Plot heatmap
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        sns.heatmap(
            sorted_matrix, 
            mask=mask,
            cmap=cmap, 
            vmin=-1, 
            vmax=1, 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5}
        )
        
        # Add title
        plt.title(f"Similarity Matrix - {config_name.replace('_', ' ').title()}")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{config_name}_heatmap.png'), dpi=300)
        plt.close()

def create_bar_chart_comparison(results):
    """Create bar chart comparing average similarities across configurations."""
    # Prepare data for plotting
    configs = list(results.keys())
    author_avgs = [results[c]['author_internal']['avg'] for c in configs]
    pauline_avgs = [results[c]['pauline_internal']['avg'] for c in configs]
    cross_avgs = [results[c]['cross']['avg'] for c in configs]
    
    # Set width of bars
    barWidth = 0.25
    
    # Set positions of bars on X axis
    r1 = np.arange(len(configs))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create bars
    plt.bar(r1, author_avgs, width=barWidth, edgecolor='white', label='Author Internal')
    plt.bar(r2, pauline_avgs, width=barWidth, edgecolor='white', label='Pauline Internal')
    plt.bar(r3, cross_avgs, width=barWidth, edgecolor='white', label='Author-Pauline')
    
    # Add labels and title
    plt.xlabel('Weight Configuration')
    plt.ylabel('Average Similarity')
    plt.title('Average Similarities by Weight Configuration')
    plt.xticks([r + barWidth for r in range(len(configs))], [c.replace('_', ' ').title() for c in configs], rotation=45)
    plt.legend()
    
    # Add grid and adjust layout
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'average_similarities_comparison.png'), dpi=300)
    plt.close()

def main():
    """Main function to analyze author vs Pauline similarities."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load similarity matrices
    print("Loading similarity matrices...")
    matrices = load_similarity_matrices()
    
    if not matrices:
        print("No similarity matrices could be loaded. Exiting.")
        return 1
    
    # Analyze similarities
    print("\nAnalyzing corpus similarities...")
    results = analyze_corpus_similarities(matrices)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_similarity_distributions(results)
    create_heatmap_visualization(matrices)
    create_bar_chart_comparison(results)
    
    print("\nAnalysis complete. Results saved to", output_dir)
    return 0

if __name__ == "__main__":
    exit(main()) 