#!/usr/bin/env python3
"""
Script to create a 3-way comparison chart showing:
1. Author (Meditations) internal similarity
2. Pauline internal similarity 
3. Author-Pauline cross-corpus similarity
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_3way_comparison_chart(results):
    """Create a bar chart comparing all three types of similarities."""
    # Extract data for the chart
    configs = []
    author_internal = []
    pauline_internal = []
    cross_similarities = []
    
    # Load cross-corpus similarities
    for config_name, data in results.items():
        # For each configuration, we need to extract cross-corpus similarity
        # Use the first author and first pauline text to get the matrix
        med_name = list(data['meditations']['matrix'].index)[0]
        paul_name = list(data['pauline']['matrix'].index)[0]
        
        # Sample cross-similarity value
        sample_cross = data['meditations']['matrix'].loc[med_name, paul_name]
        
        # If we need to calculate mean cross-similarity, we'd need to extract all
        # cross-pairs and calculate their mean. Using a sample for now.
        
        # Append values
        configs.append(config_name)
        author_internal.append(data['meditations']['average'])
        pauline_internal.append(data['pauline']['average'])
        cross_similarities.append(sample_cross)
    
    # Calculate cross-corpus averages properly
    # We need to load all author-pauline pairs for each configuration
    for i, config_name in enumerate(configs):
        data = results[config_name]
        
        # Get all author-pauline pairs
        med_names = list(data['meditations']['matrix'].index)
        paul_names = list(data['pauline']['matrix'].index)
        
        cross_values = []
        for med_name in med_names:
            for paul_name in paul_names:
                cross_values.append(data['meditations']['matrix'].loc[med_name, paul_name])
        
        # Calculate average
        if cross_values:
            cross_similarities[i] = np.mean(cross_values)
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Configuration': configs,
        'Author Internal': author_internal,
        'Pauline Internal': pauline_internal,
        'Author-Pauline': cross_similarities
    })
    
    # Set up the figure
    plt.figure(figsize=(14, 8))
    
    # Set position of bars on x axis
    x = np.arange(len(configs))
    width = 0.25  # Width of bars
    
    # Create bars
    plt.bar(x - width, df['Author Internal'], width, color='#1f77b4', label='Author Internal')
    plt.bar(x, df['Pauline Internal'], width, color='#ff7f0e', label='Pauline Internal')
    plt.bar(x + width, df['Author-Pauline'], width, color='#2ca02c', label='Author-Pauline')
    
    # Add labels and title
    plt.xlabel('Weight Configuration', fontsize=14)
    plt.ylabel('Average Similarity', fontsize=14)
    plt.title('Average Similarities by Weight Configuration', fontsize=16)
    
    # Format x-axis labels
    plt.xticks(x, [c.replace('_', ' ').title() for c in configs], rotation=45, ha='right')
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the chart
    output_dir = "author_analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "3way_comparison.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "3way_comparison.pdf"))
    
    print(f"3-way comparison chart saved to {output_dir}")

def main():
    # Load results
    results_path = os.path.join("author_analysis", "corpus_comparison_results.pkl")
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        print("Please run compare_internal_similarities.py first.")
        return 1
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    # Create 3-way comparison chart
    create_3way_comparison_chart(results)
    
    return 0

if __name__ == "__main__":
    main() 