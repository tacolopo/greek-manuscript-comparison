#!/usr/bin/env python3
"""
Script to improve MDS plots with more informative labels.
This script enhances the MDS plots by adding labels indicating traditional classifications,
but without using these classifications for the actual clustering analysis.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define traditional groupings (for labeling only, not for analysis)
TRADITIONALLY_UNDISPUTED = ['ROM', '1CO', '2CO', 'GAL', 'PHP', '1TH', 'PHM']
TRADITIONALLY_DISPUTED = ['EPH', 'COL', '2TH', '1TI', '2TI', 'TIT']

# Full names for better readability in plots
LETTER_FULL_NAMES = {
    'ROM': 'Romans',
    '1CO': '1 Corinthians',
    '2CO': '2 Corinthians',
    'GAL': 'Galatians',
    'EPH': 'Ephesians',
    'PHP': 'Philippians',
    'COL': 'Colossians',
    '1TH': '1 Thessalonians',
    '2TH': '2 Thessalonians',
    '1TI': '1 Timothy',
    '2TI': '2 Timothy',
    'TIT': 'Titus',
    'PHM': 'Philemon'
}

# Define categories for plotting
LETTER_CATEGORIES = {
    'ROM': 'Major Letters',
    '1CO': 'Major Letters',
    '2CO': 'Major Letters',
    'GAL': 'Major Letters',
    'EPH': 'Prison Letters',
    'PHP': 'Prison Letters',
    'COL': 'Prison Letters',
    '1TH': 'Thessalonian Letters',
    '2TH': 'Thessalonian Letters',
    '1TI': 'Pastoral Letters',
    '2TI': 'Pastoral Letters',
    'TIT': 'Pastoral Letters',
    'PHM': 'Personal Letter'
}

def enhance_mds_plot(config_name, output_dir):
    """Create enhanced MDS plot from saved cluster data."""
    # Load the cluster data
    input_dir = "pauline_analysis_unbiased"
    data_path = os.path.join(input_dir, f"cluster_data_{config_name}.csv")
    
    if not os.path.exists(data_path):
        print(f"Cluster data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    # Add traditional classification for color-coding
    df['traditional_group'] = df['letter'].apply(
        lambda x: 'Traditionally Undisputed' if x in TRADITIONALLY_UNDISPUTED 
        else 'Traditionally Disputed'
    )
    
    # Add letter categories for marker styles
    df['category'] = df['letter'].apply(lambda x: LETTER_CATEGORIES.get(x, 'Other'))
    
    # Create the enhanced plot
    plt.figure(figsize=(14, 10))
    
    # Define color palette for traditional grouping
    trad_colors = {'Traditionally Undisputed': '#1f77b4', 'Traditionally Disputed': '#ff7f0e'}
    
    # Define marker styles for letter categories
    markers = {
        'Major Letters': 'o',       # circle
        'Prison Letters': 's',      # square
        'Thessalonian Letters': '^', # triangle up
        'Pastoral Letters': 'D',    # diamond
        'Personal Letter': '*'      # star
    }
    
    # Create legend handles
    legend1_handles = []
    for trad_group, color in trad_colors.items():
        legend1_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=color, markersize=10, label=trad_group))
    
    legend2_handles = []
    for cat, marker in markers.items():
        legend2_handles.append(plt.Line2D([0], [0], marker=marker, color='black',
                              markersize=10, label=cat))
    
    # Add cluster legend handles
    legend3_handles = []
    cluster_colors = sns.color_palette("viridis", len(df['cluster'].unique()))
    for i, cluster_id in enumerate(sorted(df['cluster'].unique())):
        legend3_handles.append(plt.Line2D([0], [0], linestyle='none', marker='o', 
                              markerfacecolor='none', markeredgecolor=cluster_colors[i],
                              markeredgewidth=2, markersize=10, 
                              label=f'Cluster {cluster_id}'))
    
    # Plot points with traditional group colors and category markers
    for i, row in df.iterrows():
        trad_group = row['traditional_group']
        category = row['category']
        cluster_id = row['cluster']
        
        # Plot point
        plt.scatter(row['x'], row['y'], s=100, color=trad_colors[trad_group],
                   marker=markers[category], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Draw cluster boundary
        plt.scatter(row['x'], row['y'], s=130, facecolors='none', 
                   edgecolors=cluster_colors[int(cluster_id)], linewidth=2)
        
        # Add letter code
        plt.text(row['x'] + 0.02, row['y'] + 0.02, row['letter'], 
                fontsize=12, fontweight='bold')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add title and labels
    plt.title(f'New Testament Stylometric Analysis (MDS) - {config_name.upper()}', fontsize=16)
    
    # Add multiple legends
    plt.legend(handles=legend1_handles, title="Traditional Classification", 
               loc='upper right', framealpha=0.9)
    
    # Add second legend (letter categories)
    plt.legend(handles=legend2_handles, title="Letter Category",
               loc='upper left', framealpha=0.9)
    
    # Add third legend (clusters)
    plt.legend(handles=legend3_handles, title="Cluster Assignment",
               loc='lower right', framealpha=0.9)
    
    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])
    
    # Add annotation explaining the plot
    plt.figtext(0.5, 0.01, 
                "Note: Traditional classifications are shown for reference only and were NOT used in the clustering analysis.",
                ha='center', fontsize=10, style='italic')
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"enhanced_mds_{config_name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"enhanced_mds_{config_name}.pdf"), bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced MDS plot for {config_name} saved to {output_dir}")

def create_side_by_side_view(output_dir):
    """Create a side-by-side comparison of all configurations."""
    configurations = ['baseline', 'vocabulary_focused', 'structure_focused', 'nlp_only', 'equal']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(configurations), figsize=(24, 8))
    
    for i, config_name in enumerate(configurations):
        # Load the cluster data
        input_dir = "pauline_analysis_unbiased"
        data_path = os.path.join(input_dir, f"cluster_data_{config_name}.csv")
        
        if not os.path.exists(data_path):
            print(f"Cluster data file not found: {data_path}")
            continue
        
        df = pd.read_csv(data_path)
        
        # Add traditional classification for color-coding
        df['traditional_group'] = df['letter'].apply(
            lambda x: 'Traditionally Undisputed' if x in TRADITIONALLY_UNDISPUTED 
            else 'Traditionally Disputed'
        )
        
        # Plot in the corresponding subplot
        ax = axes[i]
        
        # Define color palette for traditional grouping
        trad_colors = {'Traditionally Undisputed': '#1f77b4', 'Traditionally Disputed': '#ff7f0e'}
        
        # Define cluster colors
        cluster_colors = sns.color_palette("viridis", len(df['cluster'].unique()))
        
        # Plot points
        for _, row in df.iterrows():
            trad_group = row['traditional_group']
            cluster_id = row['cluster']
            
            # Plot point with traditional group color
            ax.scatter(row['x'], row['y'], s=80, color=trad_colors[trad_group], alpha=0.8, 
                      edgecolor='black', linewidth=1)
            
            # Draw cluster boundary
            ax.scatter(row['x'], row['y'], s=100, facecolors='none', 
                      edgecolors=cluster_colors[int(cluster_id)], linewidth=2)
            
            # Add letter code
            ax.text(row['x'] + 0.02, row['y'] + 0.02, row['letter'], fontsize=9)
        
        # Set title
        ax.set_title(f'{config_name.replace("_", " ").title()}', fontsize=12)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linestyle='--', alpha=0.4)
    
    # Add common legend
    handles = []
    for trad_group, color in trad_colors.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markersize=10, label=trad_group))
    
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
              ncol=2, framealpha=0.9)
    
    # Add overall title
    fig.suptitle('MDS Clusters Across Different Weight Configurations', fontsize=16, y=0.98)
    
    # Add annotation
    fig.text(0.5, 0.01, 
            "Note: Traditional classifications are shown for reference only and were NOT used in the clustering analysis.",
            ha='center', fontsize=10, style='italic')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "mds_comparison.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "mds_comparison.pdf"))
    plt.close()
    
    print(f"Side-by-side comparison saved to {output_dir}")

def main():
    """Main function."""
    # Set output directory
    output_dir = "pauline_enhanced_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurations
    configurations = ['baseline', 'nlp_only', 'equal', 'vocabulary_focused', 'structure_focused']
    
    # Create enhanced plots
    for config_name in configurations:
        enhance_mds_plot(config_name, output_dir)
    
    # Create side-by-side comparison
    create_side_by_side_view(output_dir)
    
    print("\nEnhanced plots created successfully.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 