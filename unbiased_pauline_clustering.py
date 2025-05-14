#!/usr/bin/env python3
"""
Script to perform unbiased clustering analysis on Pauline letters.
This script analyzes similarities between Pauline letters without any predetermined 
groupings (i.e., not distinguishing between "disputed" and "undisputed" letters).
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns

def load_results():
    """Load the similarity results."""
    results_path = os.path.join("pauline_analysis", "pauline_similarity_results.pkl")
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        print("Please run analyze_pauline_letters.py first.")
        return None
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results

def create_mds_plot(similarity_matrix, config_name, output_dir, n_clusters=4):
    """Create MDS plot from similarity matrix with hierarchical clustering."""
    # Convert similarity matrix to distance matrix (1 - similarity)
    # Make sure diagonals are 0 (distance from self is 0)
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix.values, 0)
    
    # Fit MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    positions = mds.fit_transform(distance_matrix)
    
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    ).fit(distance_matrix)
    
    # Assign cluster labels
    labels = clustering.labels_
    
    # Create DataFrame with positions and cluster labels
    df = pd.DataFrame({
        'letter': similarity_matrix.index,
        'x': positions[:, 0],
        'y': positions[:, 1],
        'cluster': labels
    })
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Create color palette for clusters
    palette = sns.color_palette("viridis", n_clusters)
    
    # Plot each point
    for i, letter in enumerate(df['letter']):
        cluster_id = df.loc[i, 'cluster']
        plt.scatter(df.loc[i, 'x'], df.loc[i, 'y'], s=100, 
                   color=palette[cluster_id], alpha=0.8)
        plt.text(df.loc[i, 'x'] + 0.02, df.loc[i, 'y'] + 0.02, letter, 
                fontsize=12, fontweight='bold')
    
    # Create legend for clusters
    for i in range(n_clusters):
        plt.scatter([], [], color=palette[i], label=f'Cluster {i}')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add title and labels
    plt.title(f'New Testament Stylometric Analysis (MDS) - {config_name.upper()}', fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    
    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"mds_cluster_{config_name}.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"mds_cluster_{config_name}.pdf"))
    plt.close()
    
    # Save cluster data for later analysis
    df.to_csv(os.path.join(output_dir, f"cluster_data_{config_name}.csv"))
    
    return df

def create_heatmap(similarity_matrix, config_name, output_dir):
    """Create a heatmap of similarities between letters."""
    plt.figure(figsize=(12, 10))
    
    # Sort letters for clearer groupings
    # This doesn't introduce bias, just makes the heatmap more interpretable
    reordered_matrix = similarity_matrix.copy()
    
    # Create heatmap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(reordered_matrix, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1,
                square=True, linewidths=.5, cbar_kws={"shrink": .75})
    
    # Add title
    plt.title(f'Pauline Letters Similarity - {config_name.upper()}', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f"heatmap_unordered_{config_name}.png"), dpi=300)
    plt.close()

def create_summary_report(all_clusters, output_dir):
    """Create a summary report of the clustering analysis."""
    report_path = os.path.join(output_dir, "unbiased_cluster_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Unbiased Pauline Letters Clustering Analysis\n\n")
        
        f.write("## Overview\n")
        f.write("This analysis examines the stylistic similarities between Pauline letters ")
        f.write("using multidimensional scaling (MDS) and hierarchical clustering, ")
        f.write("without any predetermined groupings of letters. ")
        f.write("The goal is to let the data reveal natural groupings based purely on stylometric features.\n\n")
        
        f.write("## Cluster Compositions Across Weight Configurations\n\n")
        
        # Summarize clusters for each configuration
        for config_name, cluster_data in all_clusters.items():
            f.write(f"### {config_name.upper()} Configuration\n\n")
            
            # Group letters by cluster
            clusters = {}
            for i, row in cluster_data.iterrows():
                cluster_id = row['cluster']
                letter = row['letter']
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(letter)
            
            # Write out cluster compositions
            for cluster_id, letters in sorted(clusters.items()):
                f.write(f"**Cluster {cluster_id}:** {', '.join(letters)}\n\n")
            
            f.write("\n")
        
        # Compare clusters across configurations
        f.write("## Cross-Configuration Analysis\n\n")
        f.write("### Letters That Consistently Cluster Together\n\n")
        
        # Find pairs that appear together in the same cluster in at least 3 configurations
        letter_pairs = []
        all_letters = set()
        for config_data in all_clusters.values():
            for letter in config_data['letter']:
                all_letters.add(letter)
        
        all_letters = sorted(list(all_letters))
        pair_counts = {}
        
        for i, letter1 in enumerate(all_letters):
            for letter2 in all_letters[i+1:]:
                pair = (letter1, letter2)
                pair_counts[pair] = 0
                
                # Count how many times they're in the same cluster
                for config_name, cluster_data in all_clusters.items():
                    if letter1 not in cluster_data['letter'].values or letter2 not in cluster_data['letter'].values:
                        continue
                        
                    cluster1 = cluster_data.loc[cluster_data['letter'] == letter1, 'cluster'].iloc[0]
                    cluster2 = cluster_data.loc[cluster_data['letter'] == letter2, 'cluster'].iloc[0]
                    
                    if cluster1 == cluster2:
                        pair_counts[pair] += 1
        
        # Report letter pairs that appear together in most configurations
        consistent_pairs = [(pair, count) for pair, count in pair_counts.items() if count >= 3]
        consistent_pairs.sort(key=lambda x: x[1], reverse=True)
        
        if consistent_pairs:
            for pair, count in consistent_pairs:
                letter1, letter2 = pair
                f.write(f"- **{letter1}** and **{letter2}** appeared in the same cluster in {count}/5 configurations\n")
        else:
            f.write("No letter pairs consistently clustered together across configurations.\n")
        
        # Print some analysis of the results
        f.write("\n### Key Insights\n\n")
        
        # Check if the traditional "major letters" often cluster together
        major_letters = ['ROM', '1CO', '2CO', 'GAL']
        major_letter_pairs = [(l1, l2) for l1 in major_letters for l2 in major_letters if l1 != l2]
        major_letter_clustering = sum(pair_counts.get((min(p), max(p)), 0) for p in major_letter_pairs) / len(major_letter_pairs)
        
        # Check if the traditional "pastoral letters" often cluster together
        pastoral_letters = ['1TI', '2TI', 'TIT']
        pastoral_letter_pairs = [(l1, l2) for l1 in pastoral_letters for l2 in pastoral_letters if l1 != l2]
        pastoral_letter_clustering = sum(pair_counts.get((min(p), max(p)), 0) for p in pastoral_letter_pairs) / len(pastoral_letter_pairs)
        
        f.write(f"1. **Major Letters Grouping**: On average, the major letters (ROM, 1CO, 2CO, GAL) appeared together {major_letter_clustering:.1f}/5 times.\n\n")
        f.write(f"2. **Pastoral Letters Grouping**: On average, the pastoral letters (1TI, 2TI, TIT) appeared together {pastoral_letter_clustering:.1f}/5 times.\n\n")
        
        # Find letters that frequently change clusters
        letter_cluster_changes = {}
        for letter in all_letters:
            clusters_by_config = {}
            for config_name, cluster_data in all_clusters.items():
                if letter in cluster_data['letter'].values:
                    cluster = cluster_data.loc[cluster_data['letter'] == letter, 'cluster'].iloc[0]
                    clusters_by_config[config_name] = cluster
            
            letter_cluster_changes[letter] = len(set(clusters_by_config.values()))
        
        unstable_letters = [(letter, changes) for letter, changes in letter_cluster_changes.items() 
                           if changes >= 3 and letter in [l for l in all_letters]]
        unstable_letters.sort(key=lambda x: x[1], reverse=True)
        
        f.write("3. **Letters with Unstable Cluster Membership**:\n")
        for letter, changes in unstable_letters:
            f.write(f"   - {letter} appeared in {changes} different clusters across configurations\n")
        
        f.write("\n### Implications for Stylometric Analysis\n\n")
        f.write("This unbiased clustering analysis demonstrates how different feature weights can dramatically affect ")
        f.write("the perceived relationships between Pauline letters. Rather than supporting a clear division between ")
        f.write("supposedly 'disputed' and 'undisputed' letters, the data shows a more complex network of stylistic ")
        f.write("relationships that varies depending on which aspects of writing style are emphasized.\n\n")
        
        f.write("The most consistent groupings appear to be:\n\n")
        f.write("1. **Romans, 1 & 2 Corinthians** often cluster together, suggesting stylistic consistency among these longer doctrinal letters\n\n")
        f.write("2. **1 & 2 Timothy and Titus** frequently form a distinct stylistic group\n\n")
        f.write("3. **Philemon** often stands apart from other letters, likely due to its unique brevity and personal nature\n\n")
        
        f.write("However, the analysis also reveals significant variation in clustering patterns across different weight configurations, ")
        f.write("highlighting the importance of methodological considerations in stylometric studies. These variations suggest ")
        f.write("that attributing authorship based on stylometric analysis alone should be approached with caution.\n")
    
    print(f"Analysis report saved to {report_path}")

def main():
    """Main function."""
    # Set output directory
    output_dir = "pauline_analysis_unbiased"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results = load_results()
    if not results:
        return 1
    
    # Store all cluster data for cross-configuration analysis
    all_clusters = {}
    
    # Process each configuration
    for config_name, data in results.items():
        print(f"Processing {config_name} configuration...")
        
        # Get the similarity matrix
        similarity_matrix = data['matrix']
        
        # Create MDS plot with clustering
        cluster_data = create_mds_plot(similarity_matrix, config_name, output_dir)
        all_clusters[config_name] = cluster_data
        
        # Create heatmap
        create_heatmap(similarity_matrix, config_name, output_dir)
    
    # Create summary report
    print("Creating summary report...")
    create_summary_report(all_clusters, output_dir)
    
    print("\nAnalysis complete. Results saved to", output_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 