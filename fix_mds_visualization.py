#!/usr/bin/env python3
"""
Script to fix MDS visualizations to better show differences between weight configurations.
This script adjusts how MDS projections are created to ensure that the differences in
similarity matrices are actually visible in the visualizations.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# Configuration
input_dir = 'full_greek_analysis'
output_dir = 'full_greek_visualizations/enhanced'
weight_configs = ['baseline', 'equal', 'vocabulary_focused']

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def load_similarity_matrices():
    """Load similarity matrices for all weight configurations."""
    similarity_matrices = {}
    for config in weight_configs:
        matrix_path = os.path.join(input_dir, config, 'similarity_matrix.pkl')
        try:
            with open(matrix_path, 'rb') as f:
                matrix = pickle.load(f)
                similarity_matrices[config] = matrix
                print(f"Loaded similarity matrix from {config}")
        except Exception as e:
            print(f"Error loading {config} similarity matrix: {e}")
    
    return similarity_matrices

def create_enhanced_mds_visualization(similarity_matrices):
    """
    Create enhanced MDS visualizations that better show differences between configurations.
    
    This function applies several techniques to enhance the visualization of differences:
    1. Uses the same scaling for all configurations for direct comparison
    2. Amplifies differences between matrices
    3. Uses fixed axes for easier comparison
    4. Shows a direct difference heatmap between configurations
    """
    if not similarity_matrices:
        print("No similarity matrices found for MDS visualization")
        return
    
    # Get the common books across all matrices
    book_names = set(similarity_matrices[list(similarity_matrices.keys())[0]].index)
    for config, matrix in similarity_matrices.items():
        book_names = book_names.intersection(set(matrix.index))
    
    book_names = sorted(list(book_names))
    print(f"Found {len(book_names)} books common to all matrices.")
    
    # Create figure for multiple MDS plots
    fig, axes = plt.subplots(1, len(weight_configs), figsize=(6*len(weight_configs), 5))
    
    # Create figure for difference heatmap
    fig_diff, axes_diff = plt.subplots(1, len(weight_configs)-1, figsize=(6*(len(weight_configs)-1), 5))
    if len(weight_configs) == 2:
        axes_diff = [axes_diff]
    
    # Store all coordinates for comparison
    all_coords = {}
    
    # Calculate MDS coordinates for each configuration
    for i, config in enumerate(weight_configs):
        matrix = similarity_matrices[config]
        # Get the subset of the matrix for common books
        matrix_subset = matrix.loc[book_names, book_names]
        
        # Convert similarity to distance
        distance_matrix = 1 - matrix_subset.values
        np.fill_diagonal(distance_matrix, 0)  # Ensure diagonals are zero
        
        # Apply MDS with a larger n_init to find a better solution
        mds = MDS(n_components=2, 
                  dissimilarity='precomputed', 
                  random_state=42, 
                  n_init=10, 
                  max_iter=1000)
        
        coords = mds.fit_transform(distance_matrix)
        all_coords[config] = coords
        
        # Plot MDS visualization
        ax = axes[i]
        
        # Create a scatter plot with labels
        scatter = ax.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.7)
        
        # Add book labels
        for j, book in enumerate(book_names):
            ax.annotate(
                book, 
                (coords[j, 0], coords[j, 1]),
                fontsize=8,
                ha='center', 
                va='center',
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.1')
            )
            
        ax.set_title(f"MDS - {config}", fontsize=12)
        ax.grid(alpha=0.3)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
    
    # Calculate and visualize differences between first configuration and others
    baseline_config = weight_configs[0]
    for i, config in enumerate(weight_configs[1:], 0):
        # Calculate difference matrix
        baseline_matrix = similarity_matrices[baseline_config].loc[book_names, book_names]
        comparison_matrix = similarity_matrices[config].loc[book_names, book_names]
        difference_matrix = comparison_matrix.values - baseline_matrix.values
        
        # Visualization of the difference
        ax = axes_diff[i]
        im = ax.imshow(difference_matrix, cmap='coolwarm', vmin=-0.01, vmax=0.01)
        ax.set_title(f"Difference: {config} - {baseline_config}")
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set tick labels
        ax.set_xticks(np.arange(len(book_names)))
        ax.set_yticks(np.arange(len(book_names)))
        ax.set_xticklabels(book_names, rotation=90, fontsize=8)
        ax.set_yticklabels(book_names, fontsize=8)
        
        # Compute and display statistics
        max_diff = np.max(np.abs(difference_matrix))
        avg_diff = np.mean(np.abs(difference_matrix))
        print(f"Difference between {config} and {baseline_config}: Max={max_diff:.6f}, Avg={avg_diff:.6f}")
    
    # Create a figure comparing book positions across configurations
    fig_comp, ax_comp = plt.subplots(figsize=(10, 8))
    
    # Define markers and colors
    markers = ['o', 's', '^']  # Different marker per configuration
    
    # Plot each book with a different color, and each configuration with a different marker
    for j, book in enumerate(book_names):
        for i, config in enumerate(weight_configs):
            coords = all_coords[config]
            
            # Different color for each book, different marker for each config
            color = plt.cm.tab20(j % 20)
            marker = markers[i % len(markers)]
            
            # Plot the point
            ax_comp.scatter(
                coords[j, 0], 
                coords[j, 1], 
                c=[color], 
                marker=marker,
                s=100,
                alpha=0.7,
                edgecolors='black',
                label=f"{book}_{config}" if j == 0 else ""  # Only add to legend once per config
            )
            
            # Add book code text label
            if i == 0:  # Only label for the first configuration
                ax_comp.annotate(
                    book, 
                    (coords[j, 0], coords[j, 1]),
                    fontsize=8,
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.1')
                )
    
    # Create custom legend for configurations
    legend_elements = []
    for i, config in enumerate(weight_configs):
        marker = markers[i % len(markers)]
        legend_elements.append(plt.Line2D([0], [0], marker=marker, color='gray', 
                                         markerfacecolor='gray', markersize=10, 
                                         label=config))
    
    ax_comp.legend(handles=legend_elements, loc='upper right')
    ax_comp.set_title("Comparison of Book Positions Across Configurations")
    ax_comp.grid(alpha=0.3)
    
    # Save figures
    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mds_individual.png'), dpi=300)
    
    plt.figure(fig_diff.number)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mds_differences.png'), dpi=300)
    
    plt.figure(fig_comp.number)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mds_comparison.png'), dpi=300)
    
    plt.close('all')
    print(f"Saved MDS visualizations to {output_dir}")

def create_amplified_mds(similarity_matrices):
    """
    Create MDS visualizations with amplified differences to better show configuration impacts.
    """
    if not similarity_matrices:
        print("No similarity matrices found for MDS visualization")
        return
    
    # Get the common books across all matrices
    book_names = set(similarity_matrices[list(similarity_matrices.keys())[0]].index)
    for config, matrix in similarity_matrices.items():
        book_names = book_names.intersection(set(matrix.index))
    
    book_names = sorted(list(book_names))
    
    # Create figure for multiple MDS plots
    fig, axes = plt.subplots(1, len(weight_configs), figsize=(6*len(weight_configs), 5))
    
    # Process each configuration's matrix
    for i, config in enumerate(weight_configs):
        matrix = similarity_matrices[config]
        matrix_subset = matrix.loc[book_names, book_names].values
        
        # Amplify differences from identity matrix
        identity = np.eye(len(book_names))
        diff_from_identity = matrix_subset - identity
        # Amplify differences by a factor
        amplified_diff = diff_from_identity * 5
        # Reconstruct similarity matrix with amplified differences
        amplified_matrix = identity + amplified_diff
        # Ensure values are within reasonable bounds
        amplified_matrix = np.clip(amplified_matrix, 0, 1)
        
        # Convert to distance matrix
        distance_matrix = 1 - amplified_matrix
        np.fill_diagonal(distance_matrix, 0)
        
        # Apply MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=10)
        coords = mds.fit_transform(distance_matrix)
        
        # Plot MDS visualization
        ax = axes[i]
        
        # Plot each book
        for j, book in enumerate(book_names):
            ax.scatter(coords[j, 0], coords[j, 1], s=100, alpha=0.7)
            ax.annotate(
                book, 
                (coords[j, 0], coords[j, 1]),
                fontsize=8,
                ha='center', 
                va='center',
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.1')
            )
        
        ax.set_title(f"Amplified MDS - {config}", fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')
    
    # Save amplified MDS figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mds_amplified.png'), dpi=300)
    plt.close()
    print(f"Saved amplified MDS visualization to {output_dir}")

def main():
    """Main function to fix MDS visualizations."""
    # Load similarity matrices
    similarity_matrices = load_similarity_matrices()
    
    # Create enhanced MDS visualizations
    create_enhanced_mds_visualization(similarity_matrices)
    
    # Create amplified MDS visualizations
    create_amplified_mds(similarity_matrices)

if __name__ == "__main__":
    main() 