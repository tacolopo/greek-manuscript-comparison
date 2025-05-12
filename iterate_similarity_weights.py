#!/usr/bin/env python3
"""
Script to run multiple iterations of New Testament text comparison with different
similarity weight configurations to analyze how they affect clustering and text relationships.
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import copy

from src.multi_comparison import MultipleManuscriptComparison
from src.similarity import SimilarityCalculator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare NT texts with different similarity weights")
    parser.add_argument('--method', type=str, choices=['hierarchical', 'kmeans', 'dbscan'],
                        default='hierarchical', help="Clustering method (default: hierarchical)")
    parser.add_argument('--clusters', type=int, default=8, 
                        help="Number of clusters to use (default: 8)")
    parser.add_argument('--advanced-nlp', action='store_true', default=True,
                        help="Use advanced NLP features (default: True)")
    parser.add_argument('--base-output-dir', type=str, default='similarity_iterations',
                        help="Base output directory (default: similarity_iterations)")
    
    return parser.parse_args()


def parse_nt_filename(filename: str) -> Tuple[str, str, str, str]:
    """
    Parse a chapter filename to extract manuscript info.
    Expected format: path/to/grcsbl_XXX_BBB_CC_read.txt
    where XXX is the manuscript number, BBB is the book code, CC is the chapter
    
    Args:
        filename: Name of the chapter file
        
    Returns:
        Tuple of (manuscript_id, book_code, chapter_num, full_path)
    """
    # Define valid NT book codes - expanded to include all NT books
    VALID_BOOKS = r'(?:ROM|1CO|2CO|GAL|EPH|PHP|COL|1TH|2TH|1TI|2TI|TIT|PHM|' + \
                 r'ACT|JHN|1JN|2JN|3JN|1PE|2PE|JUD|REV|JAS|HEB|MAT|MRK|LUK)'
    
    # Extract components using regex - allow for full path
    pattern = rf'.*?grcsbl_(\d+)_({VALID_BOOKS})_(\d+)_read\.txt$'
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
        
    manuscript_id = match.group(1)
    book_code = match.group(2)
    chapter_num = match.group(3)
    
    return manuscript_id, book_code, chapter_num, filename


def combine_chapter_texts(chapter_files: List[str]) -> str:
    """
    Combine multiple chapter files into a single text.
    
    Args:
        chapter_files: List of chapter file paths
        
    Returns:
        Combined text from all chapters
    """
    combined_text = []
    
    for file_path in sorted(chapter_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chapter_text = f.read().strip()
                combined_text.append(chapter_text)
        except Exception as e:
            print(f"Warning: Error reading {file_path}: {e}")
            continue
    
    return "\n\n".join(combined_text)


def group_and_combine_books(chapter_files: List[str]) -> Dict[str, str]:
    """
    Group chapter files by book and combine each book's chapters into a single text.
    
    Args:
        chapter_files: List of all chapter files
        
    Returns:
        Dictionary mapping book names to their combined texts
    """
    # First, group chapters by book
    book_chapters = defaultdict(list)
    
    for file_path in chapter_files:
        try:
            manuscript_id, book_code, _, _ = parse_nt_filename(file_path)
            
            # Use just the book code without the manuscript ID
            # This ensures we combine all chapters of the same book together
            book_name = book_code
            book_chapters[book_name].append(file_path)
        except ValueError as e:
            print(f"Warning: Skipping invalid file {file_path}: {e}")
            continue
    
    # Sort chapters within each book
    for book_name in book_chapters:
        book_chapters[book_name] = sorted(book_chapters[book_name])
    
    # Now combine chapters for each book
    combined_books = {}
    for book_name, chapters in book_chapters.items():
        print(f"  - {book_name}: {len(chapters)} chapters")
        combined_books[book_name] = combine_chapter_texts(chapters)
    
    return combined_books


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


def create_weight_configs() -> List[Dict[str, float]]:
    """Define the weight configurations for different iterations."""
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
        # 2. NLP-only configuration (syntactic features only)
        {
            'name': 'nlp_only',
            'description': 'Only advanced NLP/syntactic features',
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
    
    return weight_configs


def create_consolidated_cluster_comparison(similarity_matrices: List[Dict], output_dir: str, book_display_names: Dict[str, str]):
    """
    Create a consolidated visualization of cluster assignments across different configurations.
    
    Args:
        similarity_matrices: List of dictionaries with similarity matrices and metadata
        output_dir: Directory to save the visualization
        book_display_names: Dictionary mapping book codes to display names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter matrices that have cluster data
    valid_matrices = [m for m in similarity_matrices if 'clusters' in m and isinstance(m['clusters'], list)]
    
    if not valid_matrices:
        print("No valid clustering data found for comparison")
        return
    
    # Get book names from the first matrix
    book_names = valid_matrices[0]['matrix'].index.tolist()
    
    # Number of configurations to compare
    n_configs = len(valid_matrices)
    
    # Create a consolidated dataframe to store cluster assignments for all configurations
    cluster_data = []
    
    for book_idx, book_code in enumerate(book_names):
        book_display = book_display_names.get(book_code, book_code)
        row_data = {'Book': book_display}
        
        # Add cluster from each configuration
        for config_idx, config in enumerate(valid_matrices):
            config_name = config['config_name']
            cluster = config['clusters'][book_idx]
            row_data[config_name] = cluster
            
        cluster_data.append(row_data)
    
    # Create a dataframe
    cluster_df = pd.DataFrame(cluster_data)
    
    # Create a visualization showing cluster assignments across configurations
    plt.figure(figsize=(15, 12))
    
    # Define a colormap to use for clusters
    cmap = plt.cm.get_cmap('tab10', 10)  # Use maximum 10 colors
    
    # Get all configuration names
    config_names = [m['config_name'] for m in valid_matrices]
    
    # Create a grid layout for the visualizations
    n_rows = 1
    n_cols = n_configs
    
    # For each configuration
    for i, config_name in enumerate(config_names):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Extract the data for this configuration and sort by cluster
        config_data = cluster_df[['Book', config_name]].sort_values(config_name)
        books = config_data['Book'].tolist()
        clusters = config_data[config_name].tolist()
        
        # Color based on cluster assignment
        unique_clusters = sorted(set(clusters))
        cluster_colors = [cmap(unique_clusters.index(c) % 10) for c in clusters]
        
        # Create horizontal bars for each book, colored by cluster
        y_pos = np.arange(len(books))
        plt.barh(y_pos, [1] * len(books), color=cluster_colors, height=0.8)
        
        # Add book names
        plt.yticks(y_pos, books, fontsize=8)
        
        # Add cluster numbers at the end of each bar
        for j, (book, cluster) in enumerate(zip(books, clusters)):
            plt.text(0.5, j, f"C{cluster}", ha='center', va='center', fontsize=8)
        
        # Set title and remove axes
        plt.title(config_name, fontsize=12)
        plt.xlim(0, 1)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_comparison_across_configs.png'), dpi=300)
    plt.close()
    
    # Create a cluster stability visualization to show which books change clusters
    plt.figure(figsize=(14, 10))
    
    # Create a stability score for each book (how consistently it stays in the same cluster)
    stability_data = []
    
    for _, row in cluster_df.iterrows():
        book = row['Book']
        
        # Get all cluster assignments excluding the 'Book' column
        clusters = [row[config] for config in config_names]
        
        # Count how many different clusters this book is assigned to
        unique_clusters = len(set(clusters))
        
        # Calculate stability score (1 = always same cluster, 0 = different cluster each time)
        stability = 1 - ((unique_clusters - 1) / max(1, n_configs - 1))
        
        # Add to data
        stability_data.append({
            'Book': book,
            'Stability': stability,
            'Unique_Clusters': unique_clusters,
            'Cluster_Changes': unique_clusters - 1
        })
    
    # Create dataframe and sort by stability
    stability_df = pd.DataFrame(stability_data).sort_values('Stability')
    
    # Plot stability scores
    plt.barh(np.arange(len(stability_df)), stability_df['Stability'], height=0.6)
    plt.yticks(np.arange(len(stability_df)), stability_df['Book'], fontsize=10)
    
    # Add labels for number of unique clusters
    for i, (_, row) in enumerate(stability_df.iterrows()):
        plt.text(row['Stability'] + 0.02, i, f"{row['Unique_Clusters']} clusters", 
                 va='center', fontsize=9)
    
    plt.xlim(0, 1.2)
    plt.title("Book Clustering Stability Across Weight Configurations", fontsize=14)
    plt.xlabel("Stability Score (1 = always same cluster)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'book_cluster_stability.png'), dpi=300)
    plt.close()
    
    # Create a cluster assignment heatmap to show all configurations at once
    plt.figure(figsize=(12, 14))
    
    # Create a matrix of books x configurations with cluster numbers
    cluster_matrix = cluster_df.set_index('Book')
    
    # Plot the heatmap
    sns.heatmap(cluster_matrix, cmap='tab10', annot=True, fmt='d', cbar=False)
    plt.title("Cluster Assignments Across Weight Configurations", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_assignment_heatmap.png'), dpi=300)
    plt.close()
    
    # Create a cluster transition analysis
    with open(os.path.join(output_dir, 'cluster_transition_analysis.md'), 'w') as f:
        f.write("# Cluster Transition Analysis\n\n")
        f.write("This analysis shows how books transition between different clusters based on weight configuration.\n\n")
        
        # Books that always stay in the same cluster
        stable_books = stability_df[stability_df['Stability'] == 1.0]['Book'].tolist()
        f.write("## Books with Consistent Cluster Assignments\n\n")
        if stable_books:
            f.write("These books remain in the same cluster across all weight configurations:\n\n")
            for book in stable_books:
                cluster = cluster_df[cluster_df['Book'] == book][config_names[0]].iloc[0]
                f.write(f"- **{book}**: Always in Cluster {cluster}\n")
        else:
            f.write("No books remain in the same cluster across all configurations.\n")
        
        # Books that change clusters the most
        unstable_books = stability_df[stability_df['Stability'] < 0.5]['Book'].tolist()
        f.write("\n## Books with Most Variable Cluster Assignments\n\n")
        if unstable_books:
            f.write("These books frequently change clusters based on weight configuration:\n\n")
            for book in unstable_books:
                cluster_row = cluster_df[cluster_df['Book'] == book]
                clusters = [f"Cluster {cluster_row[config].iloc[0]}" for config in config_names]
                f.write(f"- **{book}**: {' → '.join(clusters)}\n")
        else:
            f.write("No books show high cluster variability across configurations.\n")


def create_comparison_charts(similarity_matrices: List[Dict], output_dir: str):
    """
    Create comparison charts of similarity matrices across different iterations.
    
    Args:
        similarity_matrices: List of dictionaries with similarity matrices and metadata
        output_dir: Directory to save charts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract book names from the first matrix (they should be same across all iterations)
    book_names = similarity_matrices[0]['matrix'].index.tolist()
    n_books = len(book_names)
    
    # Create a summary dataframe to compare similarity differences between iterations
    comparison_data = []
    
    # For each pair of books
    for i in range(n_books):
        for j in range(i+1, n_books):
            book1 = book_names[i]
            book2 = book_names[j]
            
            row_data = {
                'Book1': book1,
                'Book2': book2
            }
            
            # Add similarity from each iteration
            for sim_data in similarity_matrices:
                config_name = sim_data['config_name']
                sim_value = sim_data['matrix'].iloc[i, j]
                row_data[config_name] = sim_value
                
            comparison_data.append(row_data)
    
    # Create a dataframe with all comparison data
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save the raw comparison data
    comparison_df.to_csv(os.path.join(output_dir, 'similarity_comparison.csv'), index=False)
    
    # Create a correlation heatmap between different iterations
    plt.figure(figsize=(12, 10))
    config_columns = [d['config_name'] for d in similarity_matrices]
    correlation_df = comparison_df[config_columns].corr()
    
    sns.heatmap(correlation_df, annot=True, cmap='viridis', vmin=0, vmax=1)
    plt.title('Correlation Between Different Weight Configurations')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iteration_correlation_heatmap.png'))
    plt.close()
    
    # Find the pairs with the most differences between iterations
    comparison_df['max_diff'] = comparison_df[config_columns].max(axis=1) - comparison_df[config_columns].min(axis=1)
    most_different = comparison_df.nlargest(15, 'max_diff')
    
    # Create a chart for the most different pairs
    plt.figure(figsize=(15, 10))
    
    # Create readable book pair labels
    book_pair_labels = [f"{row['Book1']} vs {row['Book2']}" for _, row in most_different.iterrows()]
    
    # Set up the plot
    x = np.arange(len(book_pair_labels))
    width = 0.15
    offsets = np.linspace(-(len(config_columns)-1)/2 * width, (len(config_columns)-1)/2 * width, len(config_columns))
    
    # Plot each configuration
    for i, config in enumerate(config_columns):
        values = most_different[config].values
        plt.bar(x + offsets[i], values, width, label=config)
    
    # Add labels and legend
    plt.xlabel('Book Pairs')
    plt.ylabel('Similarity Score')
    plt.title('Book Pairs with Most Variation Between Iterations')
    plt.xticks(x, book_pair_labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'most_variable_pairs.png'))
    plt.close()
    
    # Create cluster visualization comparison
    for sim_data in similarity_matrices:
        config_name = sim_data['config_name']
        clusters = sim_data.get('clusters')
        
        # Skip if no clusters data
        if not isinstance(clusters, list) or len(clusters) != len(book_names):
            continue
            
        # Create a dataframe with book names and cluster assignments
        cluster_data = {
            'Book': book_names,
            'Cluster': clusters
        }
        cluster_df = pd.DataFrame(cluster_data)
        
        # Sort by cluster
        cluster_df = cluster_df.sort_values('Cluster')
        
        # Plot
        plt.figure(figsize=(14, 8))
        unique_clusters = sorted(cluster_df['Cluster'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        
        for i, (_, row) in enumerate(cluster_df.iterrows()):
            plt.text(0.5, len(cluster_df) - i, row['Book'], 
                     ha='center', va='center',
                     bbox=dict(facecolor=colors[unique_clusters.index(row['Cluster'])], alpha=0.6))
            
        plt.xlim(0, 1)
        plt.ylim(0, len(cluster_df) + 1)
        plt.title(f'Clustering Results - {config_name}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'clusters_{config_name}.png'))
        plt.close()
    
    # Create a summary report
    with open(os.path.join(output_dir, 'weight_iteration_summary.md'), 'w') as f:
        f.write('# Similarity Weight Iteration Analysis\n\n')
        
        f.write('## Weight Configurations\n\n')
        for config in similarity_matrices:
            f.write(f"### {config['config_name']}\n")
            f.write(f"Description: {config['description']}\n")
            f.write('Weights:\n')
            for feature, weight in config['weight_config'].items():
                f.write(f'- {feature}: {weight}\n')
            f.write('\n')
        
        f.write('## Analysis Results\n\n')
        
        f.write('### Correlation Between Configurations\n\n')
        f.write('The correlation heatmap shows how similar the similarity matrices are across different weight configurations.\n')
        f.write('A high correlation indicates that changing weights does not significantly affect relative relationships between texts.\n\n')
        
        f.write('### Most Variable Book Pairs\n\n')
        f.write('These book pairs show the largest differences in similarity scores across weight configurations:\n\n')
        for _, row in most_different.head(10).iterrows():
            book1 = row['Book1']
            book2 = row['Book2']
            min_sim = min(row[config_columns])
            max_sim = max(row[config_columns])
            f.write(f"- **{book1} vs {book2}**: Similarity ranges from {min_sim:.3f} to {max_sim:.3f} (difference: {row['max_diff']:.3f})\n")
            
            # Add which configuration gives min and max
            min_config = config_columns[np.argmin([row[col] for col in config_columns])]
            max_config = config_columns[np.argmax([row[col] for col in config_columns])]
            f.write(f"  - Lowest with '{min_config}', highest with '{max_config}'\n")
        
        f.write('\n### Clustering Stability\n\n')
        f.write('Analyze how book cluster assignments change across different weight configurations.\n')
        f.write('Books that frequently change clusters are more sensitive to the choice of weights.\n\n')


def calculate_similarity_across_weights(combined_books: Dict[str, str], 
                                       display_names: Dict[str, str],
                                       args,
                                       weight_configs: List[Dict]) -> List[Dict]:
    """
    Run multiple iterations of similarity calculations with different weights.
    
    Args:
        combined_books: Dictionary of book texts
        display_names: Dictionary mapping book codes to display names
        args: Command line arguments
        weight_configs: List of weight configuration dictionaries
        
    Returns:
        List of dictionaries with similarity matrices and metadata for each iteration
    """
    # List to store similarity matrices and metadata for each iteration
    similarity_results = []
    
    # Base directory for outputs
    os.makedirs(args.base_output_dir, exist_ok=True)
    
    # For each weight configuration
    for config in weight_configs:
        config_name = config['name']
        print(f"\n\n=== Running iteration: {config_name} ===")
        print(f"Description: {config['description']}")
        print("Weights:")
        for feature, weight in config['weights'].items():
            print(f"  - {feature}: {weight}")
        
        # Create output directories for this iteration
        output_dir = os.path.join(args.base_output_dir, config_name)
        vis_dir = os.path.join(args.base_output_dir, f"{config_name}_vis")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create a custom SimilarityCalculator with specified weights
        custom_calculator = SimilarityCalculator()
        custom_calculator.weights = config['weights']
        
        # Initialize comparison object
        comparison = MultipleManuscriptComparison(
            use_advanced_nlp=args.advanced_nlp,
            output_dir=output_dir,
            visualizations_dir=vis_dir,
            similarity_calculator=custom_calculator  # Use our custom calculator
        )
        
        # Run comparison
        results = comparison.compare_multiple_manuscripts(
            manuscripts=combined_books,
            display_names=display_names,
            method=args.method,
            n_clusters=args.clusters,
            use_advanced_nlp=args.advanced_nlp
        )
        
        # Save the similarity matrix and cluster results to our results list
        result_data = {
            'config_name': config_name,
            'description': config['description'],
            'weight_config': config['weights'],
            'matrix': results.get('similarity_matrix', pd.DataFrame())
        }
        
        # Add clustering information if available in the right format
        # The cluster labels are in results['clusters']['labels']
        if 'clusters' in results and isinstance(results['clusters'], dict) and 'labels' in results['clusters']:
            result_data['clusters'] = results['clusters']['labels']
            
        similarity_results.append(result_data)
    
    return similarity_results


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Define directories
    pauline_dir = os.path.join("data", "Paul Texts")
    non_pauline_dir = os.path.join("data", "Non-Pauline NT")
    comparison_dir = os.path.join(args.base_output_dir, "comparison")
    
    try:
        # Get all chapter files from both directories
        pauline_files = glob.glob(os.path.join(pauline_dir, "*.txt"))
        non_pauline_files = glob.glob(os.path.join(non_pauline_dir, "*.txt"))
        all_chapter_files = pauline_files + non_pauline_files
        
        if not all_chapter_files:
            print("Error: No chapter files found")
            return 1
            
        print(f"Found {len(all_chapter_files)} total chapter files")
        
        # Group chapters by book and combine them
        print("\nGrouping and combining chapters by book:")
        combined_books = group_and_combine_books(all_chapter_files)
        print(f"\nSuccessfully processed {len(combined_books)} complete books")
        
        # Get book names for better display
        book_display_names = get_book_display_names()
        
        # Create a mapping of book codes to display names
        display_names = {}
        for book_key in combined_books.keys():
            # The book_key is now just the book code (ROM, GAL, etc.)
            # so we can directly map it to its display name
            if book_key in book_display_names:
                display_names[book_key] = book_display_names[book_key]
            else:
                display_names[book_key] = book_key
        
        # Create weight configurations
        weight_configs = create_weight_configs()
        
        # Run similarity calculations with different weights
        similarity_results = calculate_similarity_across_weights(
            combined_books, 
            display_names,
            args,
            weight_configs
        )
        
        # Create comparison charts
        create_comparison_charts(similarity_results, comparison_dir)
        
        # Create consolidated cluster comparison
        create_consolidated_cluster_comparison(similarity_results, comparison_dir, book_display_names)
        
        print("\nAll iterations completed successfully!")
        print(f"Results saved to {args.base_output_dir}")
        print(f"Comparison charts saved to {comparison_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 