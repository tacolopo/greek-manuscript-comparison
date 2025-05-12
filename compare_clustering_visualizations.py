#!/usr/bin/env python3
"""
Script to generate combined visualizations comparing clustering results
across different weight configurations for Greek New Testament texts.
"""

import os
import sys
import argparse
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
from sklearn.manifold import MDS, TSNE
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate combined clustering visualizations")
    parser.add_argument('--input-dir', type=str, default='whole_book_iterations',
                        help="Directory containing iteration results (default: whole_book_iterations)")
    parser.add_argument('--output-dir', type=str, default='clustering_comparisons',
                        help="Output directory for visualizations (default: clustering_comparisons)")
    
    return parser.parse_args()

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

def load_clustering_results(base_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load clustering results from each iteration.
    
    Args:
        base_dir: Base directory containing iteration results
        
    Returns:
        Dictionary mapping configuration names to clustering results
    """
    clustering_results = {}
    
    # Find all directories (except comparison dir) in the base directory
    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        
        # Skip the comparison directory and any non-directories
        if dir_name == 'comparison' or not os.path.isdir(dir_path):
            continue
        
        # Look for clusters.pkl files
        cluster_path = os.path.join(dir_path, 'clusters.pkl')
        if os.path.exists(cluster_path):
            try:
                with open(cluster_path, 'rb') as f:
                    clusters = pickle.load(f)
                    clustering_results[dir_name] = clusters
                    print(f"Loaded clustering results from {dir_name}")
            except Exception as e:
                print(f"Error loading clustering results from {dir_name}: {e}")
    
    return clustering_results

def create_combined_mds_visualization(similarity_matrices: Dict[str, pd.DataFrame], 
                                     output_path: str):
    """
    Create a combined MDS visualization showing books across different weight configurations.
    
    Args:
        similarity_matrices: Dictionary mapping configuration names to similarity matrices
        output_path: Path to save the visualization
    """
    if not similarity_matrices:
        print("No similarity matrices found for MDS visualization")
        return
    
    # Get book display names
    book_display_names = get_book_display_names()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    axes = axes.flatten()
    
    # Get the list of all configurations
    configs = list(similarity_matrices.keys())
    
    # Create a central MDS visualization with data from all configurations
    all_distance_matrices = []
    config_labels = []
    book_names = []
    
    # For each configuration, calculate MDS coordinates
    for i, (config_name, matrix) in enumerate(similarity_matrices.items()):
        # Convert similarity to distance
        distance_matrix = 1 - matrix.values
        distance_matrix = np.maximum(distance_matrix, 0)
        
        # Get book names (if not already set)
        if not book_names:
            book_names = matrix.index.tolist()
        
        # Apply MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coordinates = mds.fit_transform(distance_matrix)
        
        # Sort axis by configuration
        ax_idx = min(i, len(axes) - 1)
        ax = axes[ax_idx]
        
        # Create scatter plot with distinct colors for books
        unique_books = sorted(set(book_names))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_books)))
        color_map = {book: colors[i] for i, book in enumerate(unique_books)}
        
        # Plot each book
        for j, book in enumerate(book_names):
            # Get display name
            display_name = book_display_names.get(book, book)
            
            ax.scatter(
                coordinates[j, 0], 
                coordinates[j, 1], 
                c=[color_map[book]],
                s=100, 
                alpha=0.8,
                edgecolors='black'
            )
            
            ax.annotate(
                display_name, 
                (coordinates[j, 0], coordinates[j, 1]),
                fontsize=9,
                ha='center', 
                va='bottom'
            )
        
        ax.set_title(f"MDS - {config_name}")
        ax.grid(alpha=0.3)
        
        # Store for combined visualization
        all_distance_matrices.append(distance_matrix)
        config_labels.extend([config_name] * len(book_names))
    
    # Create a combined visualization in the last panel
    if all_distance_matrices:
        # Use the last axes panel for combined visualization
        combined_ax = axes[-1]
        
        # Combine all distance matrices
        combined_matrix = np.mean(all_distance_matrices, axis=0)
        
        # Apply MDS to the combined matrix
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        combined_coords = mds.fit_transform(combined_matrix)
        
        # Plot each book
        for j, book in enumerate(book_names):
            # Get display name
            display_name = book_display_names.get(book, book)
            
            combined_ax.scatter(
                combined_coords[j, 0], 
                combined_coords[j, 1], 
                c=[color_map[book]],
                s=100, 
                alpha=0.8,
                edgecolors='black'
            )
            
            combined_ax.annotate(
                display_name, 
                (combined_coords[j, 0], combined_coords[j, 1]),
                fontsize=9,
                ha='center', 
                va='bottom'
            )
        
        combined_ax.set_title("Combined MDS (All Configurations)")
        combined_ax.grid(alpha=0.3)
    
    plt.suptitle("MDS Visualizations Across Weight Configurations", fontsize=16)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_cluster_comparison_chart(similarity_matrices: Dict[str, pd.DataFrame],
                                   clustering_results: Dict[str, Dict[str, Any]],
                                   output_path: str,
                                   method: str = 'hierarchical',
                                   n_clusters: int = 8):
    """
    Create a chart comparing clustering results across different weight configurations.
    
    Args:
        similarity_matrices: Dictionary mapping configuration names to similarity matrices
        clustering_results: Dictionary mapping configuration names to clustering results
        output_path: Path to save the visualization
        method: Clustering method to use if results are not available
        n_clusters: Number of clusters to use if results are not available
    """
    # Get book display names
    book_display_names = get_book_display_names()
    
    # If no clustering results are found, generate them from similarity matrices
    if not clustering_results and similarity_matrices:
        print("No clustering results found, generating from similarity matrices...")
        from sklearn.cluster import AgglomerativeClustering
        
        for config_name, matrix in similarity_matrices.items():
            # Convert similarity to distance
            distance_matrix = 1 - matrix.values
            distance_matrix = np.maximum(distance_matrix, 0)
            
            # Apply clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(distance_matrix)
            
            # Create clustering result dictionary
            clustering_results[config_name] = {
                'labels': labels,
                'manuscript_names': matrix.index.tolist(),
                'clustering_method': method,
                'n_clusters': n_clusters
            }
    
    if not clustering_results:
        print("No clustering results available for comparison")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, len(clustering_results) * 2.5 + 2))
    
    # Set up colors for clusters
    max_clusters = max(result.get('n_clusters', n_clusters) for result in clustering_results.values())
    colors = plt.cm.tab20(np.linspace(0, 1, max_clusters))
    
    # For each configuration, show the clustering
    config_y_positions = {}
    y_offset = 1
    
    for i, (config_name, result) in enumerate(sorted(clustering_results.items())):
        labels = result.get('labels', [])
        manuscript_names = result.get('manuscript_names', [])
        
        # Fix: Check properly if the labels or manuscript_names are empty
        if len(labels) == 0 or len(manuscript_names) == 0:
            continue
        
        # Store vertical positions for this configuration
        config_y_positions[config_name] = y_offset
        
        # Plot configuration label
        ax.text(0, y_offset, config_name, fontsize=12, ha='right', va='center', fontweight='bold')
        
        # Group manuscripts by cluster
        clusters = defaultdict(list)
        for j, book in enumerate(manuscript_names):
            clusters[labels[j]].append((j, book))
        
        # Plot each cluster
        x_offset = 0.1
        for cluster_id, members in sorted(clusters.items()):
            # Color for this cluster
            cluster_color = colors[cluster_id % len(colors)]
            
            # Plot each member
            for j, (book_idx, book) in enumerate(sorted(members, key=lambda x: x[1])):
                # Get display name
                display_name = book_display_names.get(book, book)
                
                # Plot a colored box for this book
                rect = plt.Rectangle(
                    (x_offset, y_offset - 0.4), 
                    0.8, 
                    0.8, 
                    facecolor=cluster_color,
                    alpha=0.7,
                    edgecolor='black'
                )
                ax.add_patch(rect)
                
                # Add book name
                ax.text(
                    x_offset + 0.4, 
                    y_offset, 
                    display_name,
                    ha='center',
                    va='center',
                    fontsize=8,
                    rotation=90
                )
                
                x_offset += 1
            
            # Add space between clusters
            x_offset += 0.5
        
        # Move to next configuration
        y_offset += 2
    
    # Set up the plot
    ax.set_xlim(-1, x_offset + 1)
    ax.set_ylim(0, y_offset + 1)
    ax.set_title('Clustering Comparison Across Weight Configurations', fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_book_movement_chart(similarity_matrices: Dict[str, pd.DataFrame],
                              output_path: str):
    """
    Create a chart showing how books move in the embedding space across configurations.
    
    Args:
        similarity_matrices: Dictionary mapping configuration names to similarity matrices
        output_path: Path to save the visualization
    """
    if not similarity_matrices:
        print("No similarity matrices found for book movement visualization")
        return
    
    # Get book display names
    book_display_names = get_book_display_names()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Get the list of all configurations and books
    configs = list(similarity_matrices.keys())
    first_matrix = next(iter(similarity_matrices.values()))
    book_names = first_matrix.index.tolist()
    
    # Compute MDS coordinates for each configuration
    all_coordinates = {}
    for config_name, matrix in similarity_matrices.items():
        # Convert similarity to distance
        distance_matrix = 1 - matrix.values
        distance_matrix = np.maximum(distance_matrix, 0)
        
        # Apply MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coordinates = mds.fit_transform(distance_matrix)
        
        # Store coordinates
        all_coordinates[config_name] = coordinates
    
    # Create a reference configuration for centering the plots
    reference_coords = np.mean([coords for coords in all_coordinates.values()], axis=0)
    
    # Colors for each configuration
    config_colors = {
        'baseline': 'blue',
        'nlp_only': 'red',
        'equal': 'green',
        'vocabulary_focused': 'orange',
        'structure_focused': 'purple'
    }
    
    # Default colors if a configuration is not in the map
    default_colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    
    # Plot each book
    for i, book in enumerate(book_names):
        # Get display name
        display_name = book_display_names.get(book, book)
        
        # Average position for the book label
        avg_x = np.mean([coords[i, 0] for coords in all_coordinates.values()])
        avg_y = np.mean([coords[i, 1] for coords in all_coordinates.values()])
        
        # Plot book name at average position
        ax.text(avg_x, avg_y, display_name, fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))
        
        # Plot each configuration's position for this book
        for j, (config_name, coords) in enumerate(all_coordinates.items()):
            # Get color for this configuration
            color = config_colors.get(config_name, default_colors[j])
            
            # Plot point
            ax.scatter(coords[i, 0], coords[i, 1], color=color, s=80, alpha=0.6, edgecolors='black')
            
            # Draw lines from average position to each configuration
            ax.plot([avg_x, coords[i, 0]], [avg_y, coords[i, 1]], color=color, linestyle='-', alpha=0.4)
    
    # Add legend
    handles = []
    labels = []
    for config_name in configs:
        color = config_colors.get(config_name, default_colors[configs.index(config_name)])
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
        labels.append(config_name)
    
    ax.legend(handles, labels, loc='best', title='Weight Configuration')
    
    # Set up the plot
    ax.set_title('Book Movement Across Weight Configurations in MDS Space', fontsize=16)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_cluster_stability_chart(similarity_matrices: Dict[str, pd.DataFrame],
                                  clustering_results: Dict[str, Dict[str, Any]],
                                  output_path: str):
    """
    Create a chart showing the stability of clusters across different weight configurations.
    
    Args:
        similarity_matrices: Dictionary mapping configuration names to similarity matrices
        clustering_results: Dictionary mapping configuration names to clustering results
        output_path: Path to save the visualization
    """
    if not clustering_results:
        print("No clustering results found for stability visualization")
        return
    
    # Get book display names
    book_display_names = get_book_display_names()
    
    # Create a mapping of book and configuration to cluster assignment
    book_clusters = defaultdict(dict)
    
    # Get the list of books from the first clustering result
    first_result = next(iter(clustering_results.values()))
    book_names = first_result.get('manuscript_names', [])
    
    # If no book names found, try to get them from similarity matrices
    if not book_names and similarity_matrices:
        first_matrix = next(iter(similarity_matrices.values()))
        book_names = first_matrix.index.tolist()
    
    if not book_names:
        print("No book names found for stability visualization")
        return
    
    # Store cluster assignments for each book and configuration
    for config_name, result in clustering_results.items():
        labels = result.get('labels', [])
        result_book_names = result.get('manuscript_names', book_names)
        
        for i, book in enumerate(result_book_names):
            if i < len(labels):
                book_clusters[book][config_name] = labels[i]
    
    # Calculate stability score for each book
    book_stability = {}
    config_names = list(clustering_results.keys())
    
    for book in book_names:
        # Count how many times the book stays in the same cluster across configurations
        cluster_counts = defaultdict(int)
        
        for config in config_names:
            if config in book_clusters[book]:
                cluster_id = book_clusters[book][config]
                cluster_counts[cluster_id] += 1
        
        # Calculate stability as the percentage of configurations where the book is in its most common cluster
        if cluster_counts:
            most_common_count = max(cluster_counts.values())
            stability_score = most_common_count / len(config_names)
        else:
            stability_score = 0
        
        book_stability[book] = stability_score
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Sort books by stability score
    sorted_books = sorted(book_stability.items(), key=lambda x: x[1])
    
    # Create bar chart
    y_pos = np.arange(len(sorted_books))
    stability_scores = [score for _, score in sorted_books]
    
    # Create gradient color based on stability score
    colors = plt.cm.RdYlGn(stability_scores)
    
    # Plot bars
    bars = ax.barh(y_pos, stability_scores, color=colors)
    
    # Add book names with display names
    book_labels = [book_display_names.get(book, book) for book, _ in sorted_books]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(book_labels)
    
    # Add values to bars
    for i, v in enumerate(stability_scores):
        ax.text(v + 0.02, i, f'{v:.2f}', va='center')
    
    # Set up the plot
    ax.set_title('Book Cluster Stability Across Weight Configurations', fontsize=16)
    ax.set_xlabel('Stability Score (0-1)', fontsize=12)
    ax.set_xlim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load similarity matrices
        print("Loading similarity matrices...")
        similarity_matrices = load_similarity_matrices(args.input_dir)
        
        if not similarity_matrices:
            print("Error: No similarity matrices found")
            return 1
        
        print(f"Loaded {len(similarity_matrices)} similarity matrices")
        
        # Load clustering results
        print("Loading clustering results...")
        clustering_results = load_clustering_results(args.input_dir)
        print(f"Loaded {len(clustering_results)} clustering results")
        
        # Create combined MDS visualization
        print("Creating combined MDS visualization...")
        create_combined_mds_visualization(
            similarity_matrices,
            os.path.join(args.output_dir, 'combined_mds_visualization.png')
        )
        
        # Create cluster comparison chart
        print("Creating cluster comparison chart...")
        create_cluster_comparison_chart(
            similarity_matrices,
            clustering_results,
            os.path.join(args.output_dir, 'cluster_comparison_chart.png')
        )
        
        # Create book movement chart
        print("Creating book movement chart...")
        create_book_movement_chart(
            similarity_matrices,
            os.path.join(args.output_dir, 'book_movement_chart.png')
        )
        
        # Create cluster stability chart
        print("Creating cluster stability chart...")
        create_cluster_stability_chart(
            similarity_matrices,
            clustering_results,
            os.path.join(args.output_dir, 'cluster_stability_chart.png')
        )
        
        print(f"All visualizations saved to {args.output_dir}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 