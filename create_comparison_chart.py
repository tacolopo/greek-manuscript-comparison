#!/usr/bin/env python3
"""
Script to create visualization charts for the internal similarity comparison
between Marcus Aurelius' Meditations and Pauline letters.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_bar_chart(results):
    """Create a bar chart comparing internal similarities."""
    # Extract data for the chart
    configs = []
    med_values = []
    paul_values = []
    
    for config_name, data in results.items():
        configs.append(config_name)
        med_values.append(data['meditations']['average'])
        paul_values.append(data['pauline']['average'])
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.35
    
    # Set position of bars on x axis
    r1 = np.arange(len(configs))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    plt.bar(r1, med_values, width=bar_width, color='#3498db', label="Meditations")
    plt.bar(r2, paul_values, width=bar_width, color='#e74c3c', label="Pauline Letters")
    
    # Add labels and title
    plt.xlabel('Weight Configuration', fontsize=14)
    plt.ylabel('Average Internal Similarity', fontsize=14)
    plt.title('Internal Similarity Comparison: Meditations vs. Pauline Letters', fontsize=16)
    
    # Format x-axis labels
    plt.xticks([r + bar_width/2 for r in range(len(configs))], 
               [c.replace('_', ' ').title() for c in configs],
               fontsize=12)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Add value labels on top of bars
    for i, v in enumerate(med_values):
        plt.text(i - 0.1, v + 0.02 if v >= 0 else v - 0.06, 
                 f"{v:.3f}", color='black', fontsize=10)
    
    for i, v in enumerate(paul_values):
        plt.text(i + bar_width - 0.1, v + 0.02 if v >= 0 else v - 0.06, 
                 f"{v:.3f}", color='black', fontsize=10)
    
    # Show percentage difference between pairs
    for i in range(len(configs)):
        med_val = med_values[i]
        paul_val = paul_values[i]
        diff = med_val - paul_val
        
        # Calculate percentage difference
        if paul_val != 0 and med_val > paul_val:
            pct_diff = (med_val - paul_val) / abs(paul_val) * 100
            pct_text = f"+{pct_diff:.1f}%"
            color = 'green'
        elif med_val != 0 and paul_val > med_val:
            pct_diff = (paul_val - med_val) / abs(med_val) * 100
            pct_text = f"-{pct_diff:.1f}%"
            color = 'red'
        else:
            pct_text = "0%"
            color = 'black'
        
        # Position the text above the higher bar
        y_pos = max(med_val, paul_val) + 0.05
        plt.text(i + bar_width/2 - 0.1, y_pos, pct_text, color=color, fontsize=11)
    
    # Adjust y-axis to ensure all labels are visible
    y_min, y_max = plt.ylim()
    plt.ylim(y_min - 0.05, y_max + 0.15)
    
    # Add explanatory note
    plt.figtext(0.5, 0.01, 
                "Positive values indicate greater similarity within corpus; negative values indicate greater diversity.\n"
                "Percentages show how much higher/lower Meditations' internal similarity is relative to Pauline letters.", 
                ha="center", fontsize=11, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # Save the chart
    output_dir = "author_analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "corpus_similarity_comparison.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "corpus_similarity_comparison.pdf"))
    
    print(f"Chart saved to {output_dir}")

def create_radar_chart(results):
    """Create a radar chart to visualize multiple dimensions of the comparison."""
    # Extract data for the chart
    configs = list(results.keys())
    
    # Prepare data for radar chart
    categories = ['Average Similarity', 'Maximum Similarity', 'Similarity Variation']
    
    # Set up the figure
    fig = plt.figure(figsize=(12, 10))
    
    # Create subplots for each configuration
    num_configs = len(configs)
    rows = (num_configs + 1) // 2  # Ceiling division
    cols = 2 if num_configs > 1 else 1
    
    for i, config in enumerate(configs):
        ax = fig.add_subplot(rows, cols, i+1, polar=True)
        
        # Extract data
        med_data = results[config]['meditations']
        paul_data = results[config]['pauline']
        
        # Calculate normalized metrics
        med_metrics = [
            med_data['average'],  # Average similarity
            med_data['max'],      # Maximum similarity
            1.0 - med_data['std_dev']  # Consistency (inverse of std dev)
        ]
        
        paul_metrics = [
            paul_data['average'],  # Average similarity
            paul_data['max'],      # Maximum similarity
            1.0 - paul_data['std_dev']  # Consistency (inverse of std dev)
        ]
        
        # Number of categories
        N = len(categories)
        
        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add the first point at the end to close the polygon
        med_metrics += med_metrics[:1]
        paul_metrics += paul_metrics[:1]
        
        # Draw the chart
        ax.plot(angles, med_metrics, 'b-', linewidth=2, label='Meditations')
        ax.fill(angles, med_metrics, 'b', alpha=0.1)
        
        ax.plot(angles, paul_metrics, 'r-', linewidth=2, label='Pauline Letters')
        ax.fill(angles, paul_metrics, 'r', alpha=0.1)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Add title
        ax.set_title(config.replace('_', ' ').title(), size=14)
        
        # Add legend for the first subplot
        if i == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join("author_analysis", "radar_comparison.png"), dpi=300)
    print(f"Radar chart saved to author_analysis")

def create_heatmap(results):
    """Create a heatmap of the most similar pairs within each corpus."""
    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Get data for the baseline configuration
    baseline_data = results['baseline']
    
    # Extract matrices
    med_matrix = baseline_data['meditations']['matrix']
    paul_matrix = baseline_data['pauline']['matrix']
    
    # Clean up names for better display
    med_matrix.index = [name.replace('AUTH_', '') for name in med_matrix.index]
    med_matrix.columns = [name.replace('AUTH_', '') for name in med_matrix.columns]
    
    # Create heatmaps
    cmap = plt.cm.RdBu_r
    
    # Meditations heatmap
    im1 = axes[0].imshow(med_matrix.values, cmap=cmap, vmin=-1, vmax=1)
    axes[0].set_title("Meditations Internal Similarity", fontsize=14)
    axes[0].set_xticks(np.arange(len(med_matrix.columns)))
    axes[0].set_yticks(np.arange(len(med_matrix.index)))
    axes[0].set_xticklabels(med_matrix.columns, rotation=45, ha="right", rotation_mode="anchor")
    axes[0].set_yticklabels(med_matrix.index)
    
    # Pauline heatmap
    im2 = axes[1].imshow(paul_matrix.values, cmap=cmap, vmin=-1, vmax=1)
    axes[1].set_title("Pauline Letters Internal Similarity", fontsize=14)
    axes[1].set_xticks(np.arange(len(paul_matrix.columns)))
    axes[1].set_yticks(np.arange(len(paul_matrix.index)))
    axes[1].set_xticklabels(paul_matrix.columns, rotation=45, ha="right", rotation_mode="anchor")
    axes[1].set_yticklabels(paul_matrix.index)
    
    # Add colorbar
    cbar = fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Similarity Score', fontsize=12)
    
    # Add text annotations showing similarity values
    for i in range(len(med_matrix.index)):
        for j in range(len(med_matrix.columns)):
            # Only show values for upper triangle, excluding diagonal
            if i < j:
                axes[0].text(j, i, f"{med_matrix.iloc[i, j]:.2f}", 
                           ha="center", va="center", 
                           color="black" if abs(med_matrix.iloc[i, j]) < 0.5 else "white",
                           fontsize=8)
    
    for i in range(len(paul_matrix.index)):
        for j in range(len(paul_matrix.columns)):
            # Only show values for upper triangle, excluding diagonal
            if i < j:
                axes[1].text(j, i, f"{paul_matrix.iloc[i, j]:.2f}", 
                           ha="center", va="center", 
                           color="black" if abs(paul_matrix.iloc[i, j]) < 0.5 else "white",
                           fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join("author_analysis", "similarity_heatmap.png"), dpi=300)
    print(f"Heatmap saved to author_analysis")

def main():
    # Load results
    results_path = os.path.join("author_analysis", "corpus_comparison_results.pkl")
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        print("Please run compare_internal_similarities.py first.")
        return 1
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    # Create bar chart
    create_bar_chart(results)
    
    # Create radar chart
    create_radar_chart(results)
    
    # Create heatmap
    create_heatmap(results)
    
    return 0

if __name__ == "__main__":
    main() 