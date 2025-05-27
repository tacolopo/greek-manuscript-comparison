#!/usr/bin/env python3
"""
Script to fix MDS visualizations to better show differences between weight configurations
for the exact_cleaned_analysis dataset (without Pauline quotes).
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from matplotlib.gridspec import GridSpec

# Configuration
input_dir = 'exact_cleaned_analysis'
output_dir = 'exact_cleaned_visualizations/enhanced'
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

def create_explanation_visualization(similarity_matrices):
    """Create visualization to explain why MDS looks identical across configurations."""
    if not similarity_matrices or len(similarity_matrices) < 2:
        print("Need at least two similarity matrices for comparison")
        return
    
    # Get the common books across all matrices
    book_names = set(similarity_matrices[list(similarity_matrices.keys())[0]].index)
    for config, matrix in similarity_matrices.items():
        book_names = book_names.intersection(set(matrix.index))
    
    book_names = sorted(list(book_names))
    
    # Get the baseline and equal configurations
    baseline_matrix = similarity_matrices['baseline'].loc[book_names, book_names]
    equal_matrix = similarity_matrices['equal'].loc[book_names, book_names]
    
    # Calculate difference between matrices
    diff_matrix = equal_matrix - baseline_matrix
    max_diff = np.max(np.abs(diff_matrix.values))
    avg_diff = np.mean(np.abs(diff_matrix.values))
    print(f"Difference statistics: Max = {max_diff:.6f}, Average = {avg_diff:.6f}")
    
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Display difference matrix
    ax_diff = fig.add_subplot(gs[0, 0])
    im_diff = ax_diff.imshow(diff_matrix.values, cmap='coolwarm', vmin=-0.01, vmax=0.01)
    ax_diff.set_title(f"Difference: Equal - Baseline (Max={max_diff:.6f})")
    plt.colorbar(im_diff, ax=ax_diff)
    
    # Calculate distance change for each pair of manuscripts
    distance_changes = {}
    for i, book1 in enumerate(book_names):
        for j, book2 in enumerate(book_names):
            if i < j:
                baseline_sim = baseline_matrix.loc[book1, book2]
                equal_sim = equal_matrix.loc[book1, book2]
                distance_changes[(book1, book2)] = equal_sim - baseline_sim
    
    # Sort changes by magnitude
    sorted_changes = sorted(distance_changes.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Get top 10 pairs with biggest differences
    top_pairs = sorted_changes[:10]
    
    # Plot as horizontal bar chart
    ax_top = fig.add_subplot(gs[0, 1])
    pair_labels = [f"{p[0][0]}-{p[0][1]}" for p in top_pairs]
    pair_values = [p[1] for p in top_pairs]
    
    bars = ax_top.barh(range(len(pair_labels)), pair_values, color=['red' if v < 0 else 'green' for v in pair_values])
    ax_top.set_yticks(range(len(pair_labels)))
    ax_top.set_yticklabels(pair_labels)
    ax_top.set_xlabel("Change in Similarity (Equal - Baseline)")
    ax_top.set_title("Top 10 Manuscript Pairs with Biggest Differences")
    ax_top.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax_top.grid(axis='x', alpha=0.3)
    
    # Add values to bars
    for i, bar in enumerate(bars):
        value = pair_values[i]
        ax_top.text(
            value + (0.0005 if value >= 0 else -0.0005),
            bar.get_y() + bar.get_height()/2,
            f"{value:.6f}",
            va='center',
            ha='left' if value >= 0 else 'right',
            fontsize=8
        )
    
    # Apply MDS to both standard and amplified matrices for comparison
    # Convert to distance matrices
    baseline_distance = 1 - baseline_matrix.values
    equal_distance = 1 - equal_matrix.values
    
    # Standard MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    baseline_coords = mds.fit_transform(baseline_distance)
    equal_coords = mds.fit_transform(equal_distance)
    
    # Amplified MDS
    identity = np.eye(len(book_names))
    amplification = 5
    
    # Amplify baseline
    baseline_diff = baseline_matrix.values - identity
    baseline_amplified = identity + baseline_diff * amplification
    baseline_amplified = np.clip(baseline_amplified, 0, 1)
    baseline_amp_distance = 1 - baseline_amplified
    
    # Amplify equal
    equal_diff = equal_matrix.values - identity
    equal_amplified = identity + equal_diff * amplification
    equal_amplified = np.clip(equal_amplified, 0, 1)
    equal_amp_distance = 1 - equal_amplified
    
    # Apply MDS to amplified matrices
    mds_amp = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    baseline_amp_coords = mds_amp.fit_transform(baseline_amp_distance)
    equal_amp_coords = mds_amp.fit_transform(equal_amp_distance)
    
    # Plot standard MDS comparison
    ax_standard = fig.add_subplot(gs[1, 0])
    colors = ['blue', 'red']
    markers = ['o', 's']
    labels = ['Baseline', 'Equal Weights']
    
    for i, coords in enumerate([baseline_coords, equal_coords]):
        ax_standard.scatter(
            coords[:, 0], 
            coords[:, 1], 
            color=colors[i],
            marker=markers[i],
            s=80, 
            alpha=0.5,
            label=labels[i]
        )
    
    ax_standard.legend()
    ax_standard.set_title("Standard MDS: Baseline vs Equal")
    ax_standard.grid(alpha=0.3)
    ax_standard.set_aspect('equal')
    
    # Plot amplified MDS comparison
    ax_amplified = fig.add_subplot(gs[1, 1])
    
    for i, coords in enumerate([baseline_amp_coords, equal_amp_coords]):
        ax_amplified.scatter(
            coords[:, 0], 
            coords[:, 1], 
            color=colors[i],
            marker=markers[i],
            s=80, 
            alpha=0.5,
            label=labels[i]
        )
    
    ax_amplified.legend()
    ax_amplified.set_title("Amplified MDS: Baseline vs Equal")
    ax_amplified.grid(alpha=0.3)
    ax_amplified.set_aspect('equal')
    
    # Add main title
    plt.suptitle(
        "Exact Cleaned Dataset: Comparison of Similarity Matrices and MDS Visualizations",
        fontsize=16, 
        y=0.98
    )
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'mds_explanation.png'), dpi=300)
    plt.close()

def main():
    """Main function to fix MDS visualizations."""
    # Load similarity matrices
    similarity_matrices = load_similarity_matrices()
    
    # Create enhanced MDS visualizations
    create_enhanced_mds_visualization(similarity_matrices)
    
    # Create amplified MDS visualizations
    create_amplified_mds(similarity_matrices)
    
    # Create explanation visualization
    create_explanation_visualization(similarity_matrices)

if __name__ == "__main__":
    main() 