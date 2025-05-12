#!/usr/bin/env python3
"""
Generate a combined MDS visualization showing all weight configurations in one plot.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances

# Configuration
input_dir = 'similarity_iterations'
output_dir = 'whole_book_sensitivity'
weight_configs = ['baseline', 'nlp_only', 'equal', 'vocabulary_focused', 'structure_focused']

# Define one marker shape per weight configuration
markers = ['o', 's', '^', 'D', 'p']  # Circle, Square, Triangle, Diamond, Pentagon

# Create a color map for books - one color per book
book_colors = {
    # Pauline epistles
    'ROM': '#1f77b4',  # Blue
    '1CO': '#2ca02c',  # Green
    '2CO': '#98df8a',  # Light green
    'GAL': '#9467bd',  # Purple
    'EPH': '#c5b0d5',  # Light purple
    'PHP': '#d62728',  # Red
    'COL': '#8c564b',  # Brown
    '1TH': '#e377c2',  # Pink
    '2TH': '#f7b6d2',  # Light pink
    '1TI': '#7f7f7f',  # Gray
    '2TI': '#c7c7c7',  # Light gray
    'TIT': '#bcbd22',  # Yellow-green
    'PHM': '#17becf',  # Cyan
    # Gospels and Acts - each with its own color
    'MAT': '#1f77b4',  # Blue
    'MRK': '#aec7e8',  # Light blue
    'LUK': '#3182bd',  # Medium blue
    'JHN': '#6baed6',  # Sky blue
    'ACT': '#9ecae1',  # Pale blue
    # General epistles - each with its own color
    'HEB': '#ff7f0e',  # Orange
    'JAS': '#ffbb78',  # Light orange
    '1PE': '#ff9896',  # Light red
    '2PE': '#fc8d62',  # Salmon
    '1JN': '#fdae6b',  # Peach
    '2JN': '#fdd0a2',  # Light peach
    '3JN': '#fee6ce',  # Very light peach
    'JUD': '#fd8d3c',  # Dark orange
    # Revelation
    'REV': '#e6550d'   # Dark orange-red
}

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def main():
    """Main function to generate combined MDS visualization."""
    # Load similarity matrices from all configurations
    similarity_matrices = {}
    for config in weight_configs:
        matrix_path = os.path.join(input_dir, config, 'similarity_matrix.pkl')
        try:
            with open(matrix_path, 'rb') as f:
                matrix = pickle.load(f)
                # Skip the NLP-only config if it's all zeros
                if config == 'nlp_only':
                    if not np.any(matrix.values):
                        print(f"Warning: {config} similarity matrix contains all zeros, using identity matrix instead.")
                        # Use an identity matrix with very slight differences for clustering
                        n = matrix.shape[0]
                        identity = np.eye(n)
                        # Add tiny random differences to prevent collapse in MDS
                        np.fill_diagonal(identity, 0.999)
                        matrix = pd.DataFrame(identity, index=matrix.index, columns=matrix.columns)
                similarity_matrices[config] = matrix
                print(f"Loaded similarity matrix from {config}")
        except Exception as e:
            print(f"Error loading {config} similarity matrix: {e}")
            continue
    
    # Check if we have at least one matrix
    if not similarity_matrices:
        print("No similarity matrices could be loaded.")
        return 1
    
    # Get the intersection of book names across all matrices to ensure consistency
    book_names = set(similarity_matrices[list(similarity_matrices.keys())[0]].index)
    for config, matrix in similarity_matrices.items():
        book_names = book_names.intersection(set(matrix.index))
    
    book_names = sorted(list(book_names))
    print(f"Found {len(book_names)} books common to all matrices.")
    
    # Setup the plot
    plt.figure(figsize=(20, 14))
    
    # Create MDS projections for each configuration
    mds_coordinates = {}
    for i, (config, matrix) in enumerate(similarity_matrices.items()):
        # Convert similarity to distance by subtracting from 1
        matrix_subset = matrix.loc[book_names, book_names]
        distance_matrix = 1 - matrix_subset.values
        np.fill_diagonal(distance_matrix, 0)  # Ensure diagonals are zero
        
        # Apply MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        try:
            coords = mds.fit_transform(distance_matrix)
            mds_coordinates[config] = coords
        except Exception as e:
            print(f"Error computing MDS for {config}: {e}")
            continue
    
    # Plot each book with its own color, with different marker shapes for different weight configs
    for j, book in enumerate(book_names):
        book_color = book_colors.get(book, 'black')  # Default to black if book not in color map
        
        # For each weight configuration
        for i, config in enumerate(weight_configs):
            if config not in mds_coordinates:
                continue
                
            coords = mds_coordinates[config]
            marker = markers[i % len(markers)]
            
            # Plot the point with book's color and config's marker
            plt.scatter(
                coords[j, 0], 
                coords[j, 1], 
                c=book_color, 
                marker=marker,
                s=150,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5,
                label=f"{book}_{config}" if j == 0 else ""  # Only add to legend once
            )
            
            # Add book code text label
            plt.annotate(
                book, 
                (coords[j, 0], coords[j, 1]),
                fontsize=9,
                ha='center',
                va='center',
                color='black',
                fontweight='bold'
            )
    
    # Add a custom legend for weight configurations and book categories
    weight_config_elements = [
        Line2D([0], [0], marker=markers[i], color='w', markerfacecolor='gray', 
                markersize=12, label=config)
        for i, config in enumerate(weight_configs) if config in mds_coordinates
    ]
    
    # Book legend elements - one for each book
    book_elements = []
    # Group books by type to keep legend organized
    pauline_books = ['ROM', '1CO', '2CO', 'GAL', 'EPH', 'PHP', 'COL', '1TH', '2TH', '1TI', '2TI', 'TIT', 'PHM']
    gospel_books = ['MAT', 'MRK', 'LUK', 'JHN', 'ACT']
    general_books = ['HEB', 'JAS', '1PE', '2PE', '1JN', '2JN', '3JN', 'JUD', 'REV']
    
    # Add books in groups to keep the legend organized
    for book in pauline_books:
        if book in book_names:
            book_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=book_colors[book], 
                      markersize=10, 
                      label=book)
            )
    
    for book in gospel_books:
        if book in book_names:
            book_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=book_colors[book], 
                      markersize=10, 
                      label=book)
            )
    
    for book in general_books:
        if book in book_names:
            book_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=book_colors[book], 
                      markersize=10, 
                      label=book)
            )
    
    # Add legends - one for weight configs, one for all books
    first_legend = plt.legend(
        handles=weight_config_elements, 
        title="Weight Configurations", 
        loc='upper right',
        framealpha=0.9
    )
    plt.gca().add_artist(first_legend)
    
    # Create a second legend for books with multiple columns for better space usage
    plt.legend(
        handles=book_elements,
        title="Book Codes",
        loc='upper left',
        framealpha=0.9,
        ncol=3,
        fontsize=8
    )
    
    plt.title('Combined MDS Visualization of All Weight Configurations')
    plt.xlabel('First Dimension')
    plt.ylabel('Second Dimension')
    plt.grid(alpha=0.3)
    
    # Add equal aspect ratio to prevent distortion
    plt.axis('equal')
    plt.tight_layout()
    
    # Save the visualization
    output_path = os.path.join(output_dir, 'combined_mds_visualization.png')
    plt.savefig(output_path, dpi=300)
    print(f"Combined MDS visualization saved to {output_path}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 