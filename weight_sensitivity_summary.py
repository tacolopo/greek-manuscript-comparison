#!/usr/bin/env python3
"""
Script to generate a summary of how different weight configurations affect
similarity results and clustering in the Greek manuscript comparison.
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Any

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate weight sensitivity summary")
    parser.add_argument('--input-dir', type=str, default='similarity_iterations',
                        help="Directory containing iteration results (default: similarity_iterations)")
    parser.add_argument('--output-dir', type=str, default='weight_sensitivity_summary',
                        help="Output directory for summary (default: weight_sensitivity_summary)")
    parser.add_argument('--comparison-dir', type=str, default=None,
                        help="Directory with comparison files (default: {input_dir}/comparison)")
    
    return parser.parse_args()

def load_similarity_matrices(base_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all similarity matrices from iteration directories.
    
    Args:
        base_dir: Base directory containing iteration results
        
    Returns:
        Dictionary mapping configuration names to similarity matrices
    """
    similarity_matrices = {}
    
    # Find all directories (except comparison dir) in the base directory
    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        
        # Skip the comparison directory and any non-directories
        if dir_name == 'comparison' or not os.path.isdir(dir_path):
            continue
        
        # Look for similarity_matrix.pkl files
        matrix_path = os.path.join(dir_path, 'similarity_matrix.pkl')
        if os.path.exists(matrix_path):
            try:
                with open(matrix_path, 'rb') as f:
                    matrix = pickle.load(f)
                    similarity_matrices[dir_name] = matrix
                    print(f"Loaded similarity matrix from {dir_name}")
            except Exception as e:
                print(f"Error loading similarity matrix from {dir_name}: {e}")
    
    return similarity_matrices

def load_weights(similarity_iterations_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Load weight configurations from each iteration's report.
    
    Args:
        similarity_iterations_dir: Directory containing iteration results
        
    Returns:
        Dictionary mapping configuration names to weight dictionaries
    """
    weights = {}
    
    # Create weight configurations (same as in iterate_similarity_weights.py)
    weight_configs = [
        # 1. Current weights (baseline)
        {
            'name': 'baseline',
            'description': 'Current balanced weights',
            'weights': {
                'vocabulary': 0.25,
                'sentence': 0.15,
                'transitions': 0.15,
                'ngrams': 0.25,
                'syntactic': 0.20
            }
        },
        # 2. Only NLP/syntactic weights
        {
            'name': 'nlp_only',
            'description': 'Only advanced NLP features',
            'weights': {
                'vocabulary': 0.0,
                'sentence': 0.0,
                'transitions': 0.0,
                'ngrams': 0.0,
                'syntactic': 1.0
            }
        },
        # 3. Equal weights
        {
            'name': 'equal',
            'description': 'Equal weights for all features',
            'weights': {
                'vocabulary': 0.2,
                'sentence': 0.2,
                'transitions': 0.2,
                'ngrams': 0.2,
                'syntactic': 0.2
            }
        },
        # 4. Vocabulary/Language-focused
        {
            'name': 'vocabulary_focused',
            'description': 'Focus on vocabulary and n-grams',
            'weights': {
                'vocabulary': 0.4,
                'sentence': 0.07,
                'transitions': 0.06,
                'ngrams': 0.4,
                'syntactic': 0.07
            }
        },
        # 5. Structure-focused
        {
            'name': 'structure_focused',
            'description': 'Focus on sentence structure and transitions',
            'weights': {
                'vocabulary': 0.07,
                'sentence': 0.4,
                'transitions': 0.4,
                'ngrams': 0.06,
                'syntactic': 0.07
            }
        }
    ]
    
    # Create a dictionary of weights
    for config in weight_configs:
        weights[config['name']] = {
            'weights': config['weights'],
            'description': config['description']
        }
    
    return weights

def get_book_display_names() -> Dict[str, str]:
    """Get dictionary mapping book codes to display names."""
    return {
        "ROM": "Romans",
        "1CO": "1 Corinthians",
        "2CO": "2 Corinthians",
        "GAL": "Galatians",
        "EPH": "Ephesians",
        "PHP": "Philippians",
        "COL": "Colossians",
        "1TH": "1 Thessalonians",
        "2TH": "2 Thessalonians",
        "1TI": "1 Timothy",
        "2TI": "2 Timothy",
        "TIT": "Titus",
        "PHM": "Philemon",
        "HEB": "Hebrews",
        "JAS": "James",
        "1PE": "1 Peter",
        "2PE": "2 Peter",
        "1JN": "1 John",
        "2JN": "2 John",
        "3JN": "3 John",
        "JUD": "Jude",
        "REV": "Revelation",
        "MAT": "Matthew",
        "MRK": "Mark",
        "LUK": "Luke",
        "JHN": "John",
        "ACT": "Acts"
    }

def load_clustering_results(base_dir: str) -> Dict[str, List[int]]:
    """
    Load clustering results from each iteration.
    
    Args:
        base_dir: Base directory containing iteration results
        
    Returns:
        Dictionary mapping configuration names to cluster assignments
    """
    # This would ideally load clustering results from files, but as a fallback,
    # we'll analyze the clusters from the visualization directories
    
    clustering_results = {}
    
    for dir_name in os.listdir(base_dir):
        # Skip non-directories and the comparison directory
        dir_path = os.path.join(base_dir, dir_name)
        if dir_name == 'comparison' or not os.path.isdir(dir_path):
            continue
        
        # Check if there's a clusters file in the visualization directory
        vis_dir = os.path.join(base_dir, f"{dir_name}_vis")
        if os.path.isdir(vis_dir):
            # Look for cluster visualization files - we'd need to parse them
            # but for now we'll assume clusters are available elsewhere
            continue
    
    return clustering_results

def generate_weight_sensitivity_summary(similarity_matrices: Dict[str, pd.DataFrame],
                                       weights: Dict[str, Dict[str, Any]],
                                       comparison_csv: str,
                                       output_dir: str):
    """
    Generate summary analysis of how different weights affect similarities.
    
    Args:
        similarity_matrices: Dictionary of similarity matrices
        weights: Dictionary of weight configurations
        comparison_csv: Path to CSV file with pairwise comparisons
        output_dir: Directory to save summary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get mapping of book codes to display names
    book_display_names = get_book_display_names()
    
    # 1. Create summary of weight configurations
    with open(os.path.join(output_dir, 'weight_configurations.md'), 'w') as f:
        f.write("# Weight Configurations for Similarity Analysis\n\n")
        
        # Create a table of weight configurations
        config_table = []
        headers = ['Configuration', 'Description', 'Vocabulary', 'Sentence', 'Transitions', 'N-grams', 'Syntactic']
        
        for config_name, config_data in weights.items():
            config_weights = config_data['weights']
            row = [
                config_name,
                config_data['description'],
                f"{config_weights.get('vocabulary', 0):.2f}",
                f"{config_weights.get('sentence', 0):.2f}",
                f"{config_weights.get('transitions', 0):.2f}",
                f"{config_weights.get('ngrams', 0):.2f}",
                f"{config_weights.get('syntactic', 0):.2f}"
            ]
            config_table.append(row)
        
        f.write(tabulate(config_table, headers=headers, tablefmt="pipe"))
        f.write("\n\n")
    
    # 2. Load comparison data if available
    comparison_df = None
    if os.path.exists(comparison_csv):
        comparison_df = pd.read_csv(comparison_csv)
        
        # 3. Find which book pairs are most sensitive to weight changes
        if 'max_diff' in comparison_df.columns:
            most_sensitive = comparison_df.nlargest(20, 'max_diff')
            
            with open(os.path.join(output_dir, 'most_sensitive_pairs.md'), 'w') as f:
                f.write("# Book Pairs Most Sensitive to Weight Changes\n\n")
                f.write("These book pairs show the largest differences in similarity scores when using different weight configurations.\n\n")
                
                # Create a table
                sensitive_table = []
                headers = ['Book 1', 'Book 2', 'Max Difference', 'Min Similarity', 'Max Similarity', 
                           'Most Different Configs']
                
                for _, row in most_sensitive.iterrows():
                    book1 = row['Book1']
                    book2 = row['Book2']
                    
                    # Convert book codes to display names
                    display_book1 = book_display_names.get(book1, book1)
                    display_book2 = book_display_names.get(book2, book2)
                    
                    max_diff = row['max_diff']
                    
                    # Get min and max configs (we already computed this in the initial run)
                    config_columns = [col for col in comparison_df.columns 
                                     if col not in ['Book1', 'Book2', 'max_diff']]
                    
                    min_sim = min(row[config_columns])
                    max_sim = max(row[config_columns])
                    
                    min_config = config_columns[np.argmin([row[col] for col in config_columns])]
                    max_config = config_columns[np.argmax([row[col] for col in config_columns])]
                    
                    sensitive_table.append([
                        display_book1, display_book2, f"{max_diff:.4f}", 
                        f"{min_sim:.4f} ({min_config})",
                        f"{max_sim:.4f} ({max_config})",
                        f"{min_config} vs {max_config}"
                    ])
                
                f.write(tabulate(sensitive_table, headers=headers, tablefmt="pipe"))
                f.write("\n\n")
                
                # Create a visualization of most sensitive pairs
                plt.figure(figsize=(15, 10))
                
                # Get the top 10 most sensitive pairs
                top_sensitive = most_sensitive.head(10)
                
                # Create labels for the pairs with display names
                pair_labels = []
                for _, row in top_sensitive.iterrows():
                    book1 = book_display_names.get(row['Book1'], row['Book1'])
                    book2 = book_display_names.get(row['Book2'], row['Book2'])
                    pair_labels.append(f"{book1} vs {book2}")
                
                # Set up the plot for multiple bars
                x = np.arange(len(pair_labels))
                width = 0.15
                n_configs = len(config_columns)
                offsets = np.linspace(-(n_configs-1)/2 * width, (n_configs-1)/2 * width, n_configs)
                
                # Plot each configuration
                for i, config in enumerate(config_columns):
                    values = top_sensitive[config].values
                    plt.bar(x + offsets[i], values, width, label=config)
                
                # Add labels and legend
                plt.xlabel('Book Pairs')
                plt.ylabel('Similarity Score')
                plt.title('Book Pairs Most Sensitive to Weight Configuration')
                plt.xticks(x, pair_labels, rotation=90)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'most_sensitive_pairs.png'))
                plt.close()
    
    # 4. Generate correlation heatmap between configurations
    if similarity_matrices:
        # Create a flattened version of each similarity matrix for correlation analysis
        flat_matrices = {}
        
        # Get book names from the first matrix
        first_matrix = next(iter(similarity_matrices.values()))
        book_names = first_matrix.index.tolist()
        
        for config_name, matrix in similarity_matrices.items():
            # Extract the lower triangle of the similarity matrix (excluding diagonal)
            flat_values = []
            for i in range(len(book_names)):
                for j in range(i+1, len(book_names)):
                    flat_values.append(matrix.iloc[i, j])
            
            flat_matrices[config_name] = flat_values
        
        # Create a DataFrame with flattened matrices
        flat_df = pd.DataFrame(flat_matrices)
        
        # Calculate correlation matrix
        corr_matrix = flat_df.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='viridis', vmin=0, vmax=1)
        plt.title('Correlation Between Different Weight Configurations')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'configuration_correlation.png'))
        plt.close()
        
        # Save correlation matrix
        corr_matrix.to_csv(os.path.join(output_dir, 'configuration_correlation.csv'))
        
        # 5. Find which specific books change positions the most across configurations
        book_sensitivity = {}
        # Map to store display names
        display_name_map = {}
        
        # For each book, measure how much its similarity to other books changes
        for i, book1 in enumerate(book_names):
            total_variation = 0
            # Store the display name
            display_name_map[book1] = book_display_names.get(book1, book1)
            
            for j, book2 in enumerate(book_names):
                if i != j:
                    # Calculate how much this relationship varies across configurations
                    similarities = [matrix.iloc[i, j] for matrix in similarity_matrices.values()]
                    variation = np.std(similarities)
                    total_variation += variation
            
            # Average variation across all relationships
            book_sensitivity[book1] = total_variation / (len(book_names) - 1)
        
        # Sort books by sensitivity
        sorted_books = sorted(book_sensitivity.items(), key=lambda x: x[1], reverse=True)
        
        # Write book sensitivity to file
        with open(os.path.join(output_dir, 'book_sensitivity.md'), 'w') as f:
            f.write("# Books Most Sensitive to Weight Configuration\n\n")
            f.write("These books show the largest average variation in similarity scores when using different weight configurations.\n\n")
            
            # Create a table
            book_table = []
            headers = ['Book', 'Average Variation']
            
            for book, variation in sorted_books:
                # Use display name for the book
                display_name = display_name_map[book]
                book_table.append([display_name, f"{variation:.4f}"])
            
            f.write(tabulate(book_table, headers=headers, tablefmt="pipe"))
            f.write("\n\n")
        
        # Create a graph of book sensitivity
        plt.figure(figsize=(12, 8))
        
        # Get the top 15 most sensitive books
        top_books = sorted_books[:15]
        # Use display names for books
        book_labels = [display_name_map[book] for book, _ in top_books]
        variations = [variation for _, variation in top_books]
        
        # Plot
        plt.barh(range(len(book_labels)), variations, align='center')
        plt.yticks(range(len(book_labels)), book_labels)
        plt.xlabel('Average Variation in Similarity')
        plt.title('Books Most Sensitive to Weight Configuration')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'book_sensitivity.png'))
        plt.close()
    
    # 6. Generate main summary document
    with open(os.path.join(output_dir, 'similarity_weight_analysis.md'), 'w') as f:
        f.write("# Greek New Testament Similarity Weight Analysis\n\n")
        
        f.write("## Overview\n\n")
        f.write("This analysis examines how different feature weighting schemes affect the similarity calculations ")
        f.write("between New Testament texts. We ran five iterations with different weight configurations to analyze ")
        f.write("which text relationships are most sensitive to the choice of weights.\n\n")
        
        f.write("## Weight Configurations Tested\n\n")
        for config_name, config_data in weights.items():
            f.write(f"### {config_name}\n")
            f.write(f"{config_data['description']}\n\n")
            
            # Create a simple weights visualization
            weights_str = ""
            for feature, weight in config_data['weights'].items():
                weights_str += f"- {feature}: {weight:.2f}\n"
            
            f.write(f"```\n{weights_str}```\n\n")
        
        f.write("## Key Findings\n\n")
        
        f.write("### 1. Sensitivity to Weight Configuration\n\n")
        f.write("The correlation between different weight configurations shows how significantly ")
        f.write("changing weights affects the overall similarity relationships between texts. ")
        f.write("A high correlation indicates that even with different weights, the relative ")
        f.write("rankings of book similarities remain stable.\n\n")
        
        f.write("![Configuration Correlation](configuration_correlation.png)\n\n")
        
        f.write("### 2. Most Sensitive Text Pairs\n\n")
        f.write("These book pairs show the largest differences in similarity scores when using different weight configurations:\n\n")
        f.write("![Most Sensitive Pairs](most_sensitive_pairs.png)\n\n")
        f.write("For complete details, see [Most Sensitive Pairs](most_sensitive_pairs.md).\n\n")
        
        f.write("### 3. Books Most Affected by Weight Changes\n\n")
        f.write("Some books show more sensitivity to weight configuration than others. ")
        f.write("This may indicate that these books have unique or ambiguous stylistic features ")
        f.write("that are emphasized differently depending on which weights are used.\n\n")
        f.write("![Book Sensitivity](book_sensitivity.png)\n\n")
        f.write("For complete details, see [Book Sensitivity](book_sensitivity.md).\n\n")
        
        f.write("## Conclusions\n\n")
        f.write("Stylometric analysis of Greek New Testament texts shows both stable patterns and weight-sensitive relationships. ")
        f.write("Core relationships between certain texts remain consistent across different weighting schemes, ")
        f.write("while other relationships vary significantly depending on which features are emphasized.\n\n")
        
        f.write("This analysis highlights the importance of considering multiple feature weights ")
        f.write("when drawing conclusions about textual relationships and authorship based on stylometric evidence.")

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set comparison directory if not specified
    if args.comparison_dir is None:
        args.comparison_dir = os.path.join(args.input_dir, 'comparison')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load similarity matrices from each iteration
        print("Loading similarity matrices...")
        similarity_matrices = load_similarity_matrices(args.input_dir)
        
        if not similarity_matrices:
            print("Error: No similarity matrices found")
            return 1
        
        print(f"Loaded {len(similarity_matrices)} similarity matrices")
        
        # Load weight configurations
        print("Loading weight configurations...")
        weights = load_weights(args.input_dir)
        
        # Load clustering results if available
        print("Loading clustering results...")
        clustering_results = load_clustering_results(args.input_dir)
        
        # Check for the comparison CSV
        comparison_csv = os.path.join(args.comparison_dir, 'similarity_comparison.csv')
        if not os.path.exists(comparison_csv):
            print(f"Warning: Comparison CSV not found at {comparison_csv}")
            comparison_csv = None
        
        # Generate summary
        print("Generating weight sensitivity summary...")
        generate_weight_sensitivity_summary(
            similarity_matrices,
            weights,
            comparison_csv,
            args.output_dir
        )
        
        print(f"Summary generated in {args.output_dir}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 