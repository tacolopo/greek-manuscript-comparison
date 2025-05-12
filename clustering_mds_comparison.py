#!/usr/bin/env python3
"""
Script to generate MDS visualizations of clustering results
for all weight configurations side by side.
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate MDS clustering visualizations")
    parser.add_argument('--input-dir', type=str, default='whole_book_iterations',
                        help="Directory containing iteration results (default: whole_book_iterations)")
    parser.add_argument('--output-dir', type=str, default='mds_comparisons',
                        help="Output directory for visualizations (default: mds_comparisons)")
    
    return parser.parse_args()

def get_book_display_names():
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

def load_similarity_matrices(base_dir):
    """Load similarity matrices from each weight configuration directory."""
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

def generate_mds_coordinates(similarity_matrix):
    """Generate MDS coordinates from a similarity matrix."""
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix.values
    distance_matrix = np.maximum(distance_matrix, 0)
    
    # Apply MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coordinates = mds.fit_transform(distance_matrix)
    
    return coordinates

def generate_clusters(similarity_matrix, n_clusters=8):
    """Generate clusters from a similarity matrix."""
    from sklearn.cluster import AgglomerativeClustering
    
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix.values
    distance_matrix = np.maximum(distance_matrix, 0)
    
    # Apply clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)
    
    return labels

def create_mds_comparison_chart(similarity_matrices, output_path, n_clusters=8):
    """Create MDS comparison chart for all weight configurations."""
    if not similarity_matrices:
        print("No similarity matrices found for MDS visualization")
        return
    
    # Get book display names
    book_display_names = get_book_display_names()
    
    # Number of configurations to display
    n_configs = len(similarity_matrices)
    
    # Determine grid layout
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols
    
    # Create figure and gridspec for layout
    fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    # Iterate through each configuration
    for i, (config_name, matrix) in enumerate(similarity_matrices.items()):
        # Calculate row and column in grid
        row = i // n_cols
        col = i % n_cols
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        # Get book names
        book_names = matrix.index.tolist()
        
        # Generate MDS coordinates
        coordinates = generate_mds_coordinates(matrix)
        
        # Generate clusters
        labels = generate_clusters(matrix, n_clusters=n_clusters)
        
        # Create a colormap for clusters
        unique_clusters = sorted(np.unique(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        
        # Plot each book
        for j, book in enumerate(book_names):
            cluster_id = labels[j]
            color = colors[unique_clusters.index(cluster_id)]
            
            # Get display name
            display_name = book_display_names.get(book, book)
            
            # Plot point
            ax.scatter(
                coordinates[j, 0],
                coordinates[j, 1],
                color=color,
                s=100,
                alpha=0.7,
                edgecolors='black'
            )
            
            # Add text label
            ax.text(
                coordinates[j, 0],
                coordinates[j, 1] + 0.1,
                display_name,
                fontsize=8,
                ha='center',
                va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', edgecolor='none')
            )
        
        # Add legend for clusters
        legend_handles = []
        for cluster_id in unique_clusters:
            color = colors[unique_clusters.index(cluster_id)]
            patch = mpatches.Patch(color=color, label=f'Cluster {cluster_id}')
            legend_handles.append(patch)
        
        ax.legend(handles=legend_handles, loc='upper right', fontsize=8, title='Clusters')
        
        # Add title and grid
        ax.set_title(f"MDS Clustering - {config_name}", fontsize=14)
        ax.grid(alpha=0.3)
        
        # Set axes limits with some padding
        pad = 0.5
        ax.set_xlim(np.min(coordinates[:, 0]) - pad, np.max(coordinates[:, 0]) + pad)
        ax.set_ylim(np.min(coordinates[:, 1]) - pad, np.max(coordinates[:, 1]) + pad)
    
    # Add overall title
    fig.suptitle("MDS Clustering Comparison Across Weight Configurations", fontsize=16, y=0.99)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_mds_chart(similarity_matrices, output_path, n_clusters=8):
    """Create a combined MDS chart with data from all configurations."""
    if not similarity_matrices:
        print("No similarity matrices found for combined MDS visualization")
        return
    
    # Get book display names
    book_display_names = get_book_display_names()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get the first matrix to extract book names
    first_matrix = next(iter(similarity_matrices.values()))
    book_names = first_matrix.index.tolist()
    
    # Calculate average distance matrix
    distance_matrices = []
    for matrix in similarity_matrices.values():
        # Convert similarity to distance
        distance_matrix = 1 - matrix.values
        distance_matrix = np.maximum(distance_matrix, 0)
        distance_matrices.append(distance_matrix)
    
    avg_distance_matrix = np.mean(distance_matrices, axis=0)
    
    # Apply MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coordinates = mds.fit_transform(avg_distance_matrix)
    
    # Generate clusters
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(avg_distance_matrix)
    
    # Create a colormap for clusters
    unique_clusters = sorted(np.unique(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    # Plot each book
    for j, book in enumerate(book_names):
        cluster_id = labels[j]
        color = colors[unique_clusters.index(cluster_id)]
        
        # Get display name
        display_name = book_display_names.get(book, book)
        
        # Plot point
        ax.scatter(
            coordinates[j, 0],
            coordinates[j, 1],
            color=color,
            s=150,
            alpha=0.7,
            edgecolors='black'
        )
        
        # Add text label
        ax.text(
            coordinates[j, 0],
            coordinates[j, 1] + 0.1,
            display_name,
            fontsize=10,
            ha='center',
            va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
        )
    
    # Add legend for clusters
    legend_handles = []
    for cluster_id in unique_clusters:
        color = colors[unique_clusters.index(cluster_id)]
        patch = mpatches.Patch(color=color, label=f'Cluster {cluster_id}')
        legend_handles.append(patch)
    
    ax.legend(handles=legend_handles, loc='upper right', title='Clusters')
    
    # Add title and grid
    ax.set_title("Combined MDS Clustering (Average of All Weight Configurations)", fontsize=14)
    ax.grid(alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_pauline_vs_nonpauline_chart(similarity_matrices, output_path):
    """Create a chart highlighting Pauline vs non-Pauline books in MDS space."""
    if not similarity_matrices:
        print("No similarity matrices found for Pauline vs non-Pauline visualization")
        return
    
    # Get book display names
    book_display_names = get_book_display_names()
    
    # Define Pauline books (traditional and disputed)
    traditional_pauline = {"ROM", "1CO", "2CO", "GAL", "PHP", "1TH", "PHM"}
    disputed_pauline = {"EPH", "COL", "2TH", "1TI", "2TI", "TIT"}
    all_pauline = traditional_pauline.union(disputed_pauline)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get the first matrix to extract book names
    first_matrix = next(iter(similarity_matrices.values()))
    book_names = first_matrix.index.tolist()
    
    # Calculate average distance matrix
    distance_matrices = []
    for matrix in similarity_matrices.values():
        # Convert similarity to distance
        distance_matrix = 1 - matrix.values
        distance_matrix = np.maximum(distance_matrix, 0)
        distance_matrices.append(distance_matrix)
    
    avg_distance_matrix = np.mean(distance_matrices, axis=0)
    
    # Apply MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coordinates = mds.fit_transform(avg_distance_matrix)
    
    # Plot each book
    for j, book in enumerate(book_names):
        # Determine category and color
        if book in traditional_pauline:
            color = 'blue'
            marker = 'o'
            category = 'Traditional Pauline'
        elif book in disputed_pauline:
            color = 'cyan'
            marker = 's'
            category = 'Disputed Pauline'
        else:
            color = 'red'
            marker = '^'
            category = 'Non-Pauline'
        
        # Get display name
        display_name = book_display_names.get(book, book)
        
        # Plot point
        ax.scatter(
            coordinates[j, 0],
            coordinates[j, 1],
            color=color,
            marker=marker,
            s=150,
            alpha=0.7,
            edgecolors='black',
            label=category
        )
        
        # Add text label
        ax.text(
            coordinates[j, 0],
            coordinates[j, 1] + 0.1,
            display_name,
            fontsize=10,
            ha='center',
            va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
        )
    
    # Create custom legend (to avoid duplicates)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Traditional Pauline'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='cyan', markersize=10, label='Disputed Pauline'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Non-Pauline')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add title and grid
    ax.set_title("Pauline vs Non-Pauline Books in MDS Space", fontsize=14)
    ax.grid(alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
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
            print("Error: No valid similarity matrices found")
            return 1
        
        print(f"Loaded {len(similarity_matrices)} similarity matrices")
        
        # Create MDS comparison chart
        print("Creating MDS comparison chart...")
        create_mds_comparison_chart(
            similarity_matrices,
            os.path.join(args.output_dir, 'mds_comparison.png')
        )
        
        # Create combined MDS chart
        print("Creating combined MDS chart...")
        create_combined_mds_chart(
            similarity_matrices,
            os.path.join(args.output_dir, 'combined_mds.png')
        )
        
        # Create Pauline vs non-Pauline chart
        print("Creating Pauline vs non-Pauline chart...")
        create_pauline_vs_nonpauline_chart(
            similarity_matrices,
            os.path.join(args.output_dir, 'pauline_vs_nonpauline.png')
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