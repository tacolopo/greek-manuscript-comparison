#!/usr/bin/env python3
"""
Script to explain why the MDS visualizations appear identical across weight configurations
and demonstrate how to fix the issue.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Configuration
input_dir = 'full_greek_analysis'
output_dir = 'full_greek_visualizations/explanation'
weight_configs = ['baseline', 'equal']

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def load_similarity_matrices():
    """Load similarity matrices for selected weight configurations."""
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
    print(f"Found {len(book_names)} books common to all matrices.")
    
    # Get the baseline and equal configurations
    baseline_matrix = similarity_matrices['baseline'].loc[book_names, book_names]
    equal_matrix = similarity_matrices['equal'].loc[book_names, book_names]
    
    # Calculate difference between matrices
    diff_matrix = equal_matrix - baseline_matrix
    max_diff = np.max(np.abs(diff_matrix.values))
    avg_diff = np.mean(np.abs(diff_matrix.values))
    print(f"Difference statistics: Max = {max_diff:.6f}, Average = {avg_diff:.6f}")
    
    # Create figure with 2x3 grid
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. Original matrices comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_diff = fig.add_subplot(gs[0, 2])
    
    # Display baseline matrix
    im1 = ax1.imshow(baseline_matrix.values, cmap='viridis', vmin=0, vmax=1)
    ax1.set_title("Baseline Similarity Matrix")
    plt.colorbar(im1, ax=ax1)
    
    # Display equal weights matrix
    im2 = ax2.imshow(equal_matrix.values, cmap='viridis', vmin=0, vmax=1)
    ax2.set_title("Equal Weights Similarity Matrix")
    plt.colorbar(im2, ax=ax2)
    
    # Display difference matrix
    im_diff = ax_diff.imshow(diff_matrix.values, cmap='coolwarm', vmin=-0.01, vmax=0.01)
    ax_diff.set_title(f"Difference (Max={max_diff:.6f})")
    plt.colorbar(im_diff, ax=ax_diff)
    
    # 2. Standard MDS results - looks identical
    # Convert to distance matrices
    baseline_distance = 1 - baseline_matrix.values
    equal_distance = 1 - equal_matrix.values
    
    # Apply MDS to both matrices
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    baseline_coords = mds.fit_transform(baseline_distance)
    equal_coords = mds.fit_transform(equal_distance)
    
    # Plot standard MDS results
    ax_mds1 = fig.add_subplot(gs[1, 0])
    ax_mds2 = fig.add_subplot(gs[1, 1])
    ax_mds_diff = fig.add_subplot(gs[1, 2])
    
    # Plot baseline MDS
    ax_mds1.scatter(baseline_coords[:, 0], baseline_coords[:, 1], s=100, alpha=0.7)
    for i, book in enumerate(book_names):
        ax_mds1.annotate(
            book, 
            (baseline_coords[i, 0], baseline_coords[i, 1]),
            fontsize=8,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.1')
        )
    ax_mds1.set_title("Baseline MDS Visualization")
    ax_mds1.grid(alpha=0.3)
    ax_mds1.set_aspect('equal')
    
    # Plot equal weights MDS
    ax_mds2.scatter(equal_coords[:, 0], equal_coords[:, 1], s=100, alpha=0.7)
    for i, book in enumerate(book_names):
        ax_mds2.annotate(
            book, 
            (equal_coords[i, 0], equal_coords[i, 1]),
            fontsize=8,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.1')
        )
    ax_mds2.set_title("Equal Weights MDS Visualization")
    ax_mds2.grid(alpha=0.3)
    ax_mds2.set_aspect('equal')
    
    # Plot both configurations in one chart to show overlap
    colors = ['blue', 'red']
    markers = ['o', 's']
    labels = ['Baseline', 'Equal Weights']
    
    for i, coords in enumerate([baseline_coords, equal_coords]):
        ax_mds_diff.scatter(
            coords[:, 0], 
            coords[:, 1], 
            color=colors[i],
            marker=markers[i],
            s=100, 
            alpha=0.5,
            label=labels[i]
        )
    
    # Add legend
    ax_mds_diff.legend()
    ax_mds_diff.set_title("Overlaid MDS Visualizations")
    ax_mds_diff.grid(alpha=0.3)
    ax_mds_diff.set_aspect('equal')
    
    # Add main title explaining the issue
    plt.suptitle(
        "Why MDS Visualizations Look Identical Across Weight Configurations",
        fontsize=16, 
        y=0.98
    )
    
    # Add explanatory text
    fig.text(
        0.5, 
        0.01, 
        "The similarity matrices differ very slightly (avg diff: {:.6f}), but MDS only preserves relative distances.\n"
        "Even with different weights, the relative distances between manuscripts remain nearly identical.\n"
        "To make differences visible, we need to amplify the differences or use alternative visualization methods.".format(avg_diff),
        ha='center',
        fontsize=12,
        bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=0.5')
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'mds_explanation.png'), dpi=300)
    plt.close()
    
    # Create a visualization demonstrating the solution
    demonstrate_solution(similarity_matrices, book_names)
    
def demonstrate_solution(similarity_matrices, book_names):
    """Demonstrate how to make MDS visualizations show differences between configurations."""
    baseline_matrix = similarity_matrices['baseline'].loc[book_names, book_names]
    equal_matrix = similarity_matrices['equal'].loc[book_names, book_names]
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. Amplify differences between matrices
    baseline_values = baseline_matrix.values
    equal_values = equal_matrix.values
    
    # Method 1: Amplify differences from identity
    identity = np.eye(len(book_names))
    baseline_diff = baseline_values - identity
    equal_diff = equal_values - identity
    
    # Amplify by a factor
    amplification = 5
    baseline_amplified = identity + baseline_diff * amplification
    equal_amplified = identity + equal_diff * amplification
    
    # Ensure values are within reasonable bounds
    baseline_amplified = np.clip(baseline_amplified, 0, 1)
    equal_amplified = np.clip(equal_amplified, 0, 1)
    
    # Convert to distance matrices
    baseline_distance = 1 - baseline_amplified
    equal_distance = 1 - equal_amplified
    
    # Apply MDS to amplified matrices
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    baseline_coords = mds.fit_transform(baseline_distance)
    equal_coords = mds.fit_transform(equal_distance)
    
    # Plot amplified MDS results
    ax_mds1 = fig.add_subplot(gs[0, 0])
    ax_mds2 = fig.add_subplot(gs[0, 1])
    ax_mds_diff = fig.add_subplot(gs[0, 2])
    
    # Plot baseline MDS
    ax_mds1.scatter(baseline_coords[:, 0], baseline_coords[:, 1], s=100, alpha=0.7)
    for i, book in enumerate(book_names):
        ax_mds1.annotate(
            book, 
            (baseline_coords[i, 0], baseline_coords[i, 1]),
            fontsize=8,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.1')
        )
    ax_mds1.set_title("Amplified Baseline MDS")
    ax_mds1.grid(alpha=0.3)
    ax_mds1.set_aspect('equal')
    
    # Plot equal weights MDS
    ax_mds2.scatter(equal_coords[:, 0], equal_coords[:, 1], s=100, alpha=0.7)
    for i, book in enumerate(book_names):
        ax_mds2.annotate(
            book, 
            (equal_coords[i, 0], equal_coords[i, 1]),
            fontsize=8,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.1')
        )
    ax_mds2.set_title("Amplified Equal Weights MDS")
    ax_mds2.grid(alpha=0.3)
    ax_mds2.set_aspect('equal')
    
    # Plot both configurations in one chart
    colors = ['blue', 'red']
    markers = ['o', 's']
    labels = ['Baseline', 'Equal Weights']
    
    for i, coords in enumerate([baseline_coords, equal_coords]):
        ax_mds_diff.scatter(
            coords[:, 0], 
            coords[:, 1], 
            color=colors[i],
            marker=markers[i],
            s=100, 
            alpha=0.5,
            label=labels[i]
        )
    
    # Add legend
    ax_mds_diff.legend()
    ax_mds_diff.set_title("Overlaid Amplified MDS")
    ax_mds_diff.grid(alpha=0.3)
    ax_mds_diff.set_aspect('equal')
    
    # Method 2: Direct visualization of differences
    ax_diff = fig.add_subplot(gs[1, 0:2])
    
    # Compute the difference matrix
    diff_matrix = equal_matrix.values - baseline_matrix.values
    
    # Visualize the difference
    im = ax_diff.imshow(diff_matrix, cmap='coolwarm', vmin=-0.01, vmax=0.01)
    ax_diff.set_title("Difference Matrix: Equal - Baseline")
    plt.colorbar(im, ax=ax_diff)
    
    # Set tick labels
    ax_diff.set_xticks(np.arange(len(book_names)))
    ax_diff.set_yticks(np.arange(len(book_names)))
    ax_diff.set_xticklabels(book_names, rotation=90, fontsize=8)
    ax_diff.set_yticklabels(book_names, fontsize=8)
    
    # Method 3: Comparative visualization
    ax_comp = fig.add_subplot(gs[1, 2])
    
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
    pair_labels = [f"{p[0][0]}-{p[0][1]}" for p in top_pairs]
    pair_values = [p[1] for p in top_pairs]
    
    bars = ax_comp.barh(range(len(pair_labels)), pair_values, color=['red' if v < 0 else 'green' for v in pair_values])
    ax_comp.set_yticks(range(len(pair_labels)))
    ax_comp.set_yticklabels(pair_labels)
    ax_comp.set_xlabel("Change in Similarity (Equal - Baseline)")
    ax_comp.set_title("Top 10 Manuscript Pairs with Biggest Differences")
    ax_comp.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax_comp.grid(axis='x', alpha=0.3)
    
    # Add values to bars
    for i, bar in enumerate(bars):
        value = pair_values[i]
        ax_comp.text(
            value + (0.0005 if value >= 0 else -0.0005),
            bar.get_y() + bar.get_height()/2,
            f"{value:.6f}",
            va='center',
            ha='left' if value >= 0 else 'right',
            fontsize=8
        )
    
    # Add main title
    plt.suptitle(
        "Solutions to Make MDS Visualization Differences Visible",
        fontsize=16, 
        y=0.98
    )
    
    # Add explanatory text
    fig.text(
        0.5, 
        0.01, 
        "Solution 1: Amplify differences between matrices to make MDS show configuration differences.\n"
        "Solution 2: Directly visualize the difference matrix to identify which manuscript pairs change most.\n"
        "Solution 3: Create focused visualizations comparing specific differences between configurations.",
        ha='center',
        fontsize=12,
        bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=0.5')
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'mds_solutions.png'), dpi=300)
    plt.close()

def main():
    """Main function to create explanation visualizations."""
    # Load similarity matrices
    similarity_matrices = load_similarity_matrices()
    
    # Create explanation visualization
    create_explanation_visualization(similarity_matrices)
    
    print(f"Saved explanation visualizations to {output_dir}")

if __name__ == "__main__":
    main() 