#!/usr/bin/env python3
"""
Generate MDS visualization for author analysis results comparing author's articles
with Pauline corpus across different weight configurations.
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
input_dir = 'author_analysis'
output_dir = 'author_analysis'
weight_configs = ['baseline', 'nlp_only', 'equal', 'vocabulary_focused', 'structure_focused']

# Define one marker shape per weight configuration
markers = ['o', 's', '^', 'D', 'p']  # Circle, Square, Triangle, Diamond, Pentagon

# Create a color map for books - group by author vs Pauline
book_colors = {
    # Marcus Aurelius Meditations - shades of green
    'AUTH_Meditations 1': '#00a651',
    'AUTH_Meditations 2': '#07a857',
    'AUTH_Meditations 3': '#0eab5d',
    'AUTH_Meditations 4': '#15ae63',
    'AUTH_Meditations 5': '#1cb169',
    'AUTH_Meditations 6': '#23b46f',
    'AUTH_Meditations 7': '#2ab775',
    'AUTH_Meditations 8': '#31ba7b',
    'AUTH_Meditations 9': '#38bd81',
    'AUTH_Meditations 10': '#3fc087',
    'AUTH_Meditations 11': '#46c38d',
    'AUTH_Meditations 12': '#4dc693',
    
    # Pauline epistles - shades of blue
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
}

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def main():
    """Main function to generate MDS visualization for author analysis."""
    # Load similarity matrices from all configurations
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
    print(f"Found {len(book_names)} texts common to all matrices.")
    
    # Setup the plot
    plt.figure(figsize=(20, 14))
    
    # Create MDS projections for each configuration
    mds_coordinates = {}
    for i, (config, matrix) in enumerate(similarity_matrices.items()):
        # Convert similarity to distance
        # Since similarity values range from -1 to 1, transform them to 0-1 range first
        matrix_subset = matrix.loc[book_names, book_names]
        similarity_values = matrix_subset.values
        
        # Normalize from [-1, 1] to [0, 1]
        normalized_similarity = (similarity_values + 1) / 2
        
        # Convert to distance (1 - similarity)
        distance_matrix = 1 - normalized_similarity
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
    # Group author's articles and Pauline texts differently
    author_texts = [text for text in book_names if text.startswith('AUTH_')]
    pauline_texts = [text for text in book_names if not text.startswith('AUTH_')]
    
    # Create subplots for each configuration
    fig, axes = plt.subplots(1, len(mds_coordinates), figsize=(20, 12), sharex=False, sharey=False)
    
    # If there's only one configuration, wrap the axis in a list for consistent indexing
    if len(mds_coordinates) == 1:
        axes = [axes]
    
    # Plot each configuration in its own subplot
    for ax_idx, (config, coords) in enumerate(mds_coordinates.items()):
        ax = axes[ax_idx]
        
        # Plot points for each book
        for j, book in enumerate(book_names):
            book_color = book_colors.get(book, 'black')  # Default to black if book not in color map
            marker = 'o' if book.startswith('AUTH_') else 's'  # Use different markers for author vs Pauline
            
            # Determine point size - larger for author's texts
            point_size = 200 if book.startswith('AUTH_') else 150
            
            # Plot the point
            ax.scatter(
                coords[j, 0], 
                coords[j, 1], 
                c=book_color, 
                marker=marker,
                s=point_size,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5
            )
            
            # Add text label
            ax.annotate(
                book, 
                (coords[j, 0], coords[j, 1]),
                fontsize=9,
                ha='center',
                va='center',
                color='black',
                fontweight='bold'
            )
        
        # Set subplot title
        ax.set_title(f'{config.replace("_", " ").title()} Configuration')
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')
    
    # Add a combined plot with all configurations in one
    fig2, ax_combined = plt.subplots(figsize=(20, 14))
    
    # Plot each text with its own color, with different marker shapes for different weight configs
    for j, book in enumerate(book_names):
        book_color = book_colors.get(book, 'black')  # Default to black if book not in color map
        
        # For each weight configuration
        for i, config in enumerate(mds_coordinates.keys()):
            coords = mds_coordinates[config]
            marker = markers[i % len(markers)]
            
            # Plot the point with book's color and config's marker
            ax_combined.scatter(
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
            ax_combined.annotate(
                book, 
                (coords[j, 0], coords[j, 1]),
                fontsize=9,
                ha='center',
                va='center',
                color='black',
                fontweight='bold'
            )
    
    # Add a custom legend for weight configurations
    weight_config_elements = [
        Line2D([0], [0], marker=markers[i], color='w', markerfacecolor='gray', 
                markersize=12, label=config.replace('_', ' ').title())
        for i, config in enumerate(mds_coordinates.keys())
    ]
    
    # Group texts for legend
    author_elements = []
    for book in author_texts:
        author_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=book_colors[book], 
                  markersize=10, 
                  label=book)
        )
    
    pauline_elements = []
    for book in pauline_texts:
        pauline_elements.append(
            Line2D([0], [0], marker='s', color='w', 
                  markerfacecolor=book_colors[book], 
                  markersize=10, 
                  label=book)
        )
    
    # Add legends
    first_legend = ax_combined.legend(
        handles=weight_config_elements, 
        title="Weight Configurations", 
        loc='upper right',
        framealpha=0.9
    )
    ax_combined.add_artist(first_legend)
    
    # Create a legend for author texts
    auth_legend = ax_combined.legend(
        handles=author_elements,
        title="Author's Articles",
        loc='upper left',
        framealpha=0.9,
        fontsize=8
    )
    ax_combined.add_artist(auth_legend)
    
    # Create a legend for Pauline texts
    pauline_legend = ax_combined.legend(
        handles=pauline_elements,
        title="Pauline Letters",
        loc='lower left',
        framealpha=0.9,
        ncol=3,
        fontsize=8
    )
    
    ax_combined.set_title('Combined MDS Visualization of Author vs. Pauline Texts')
    ax_combined.set_xlabel('First Dimension')
    ax_combined.set_ylabel('Second Dimension')
    ax_combined.grid(alpha=0.3)
    ax_combined.set_aspect('equal')
    
    # Save the visualizations
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'author_mds_by_config.png'), dpi=300)
    
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'author_combined_mds.png'), dpi=300)
    
    print(f"MDS visualizations saved to {output_dir}")
    return 0

if __name__ == "__main__":
    exit(main()) 