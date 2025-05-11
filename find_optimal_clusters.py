#!/usr/bin/env python3
"""
Script to determine the optimal number of clusters for the Pauline letters analysis
using multiple evaluation methods.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

def load_similarity_matrix(output_dir="output"):
    """Load the similarity matrix from files."""
    try:
        # Try pickle first
        matrix_file = os.path.join(output_dir, "similarity_matrix.pkl")
        if os.path.exists(matrix_file):
            import pickle
            with open(matrix_file, 'rb') as f:
                return pickle.load(f)
        
        # Otherwise try CSV
        matrix_file = os.path.join(output_dir, "similarity_matrix.csv")
        if os.path.exists(matrix_file):
            return pd.read_csv(matrix_file, index_col=0)
        
        print(f"Error: Could not find similarity matrix in {output_dir}")
        return None
    except Exception as e:
        print(f"Error loading similarity matrix: {e}")
        return None

def plot_dendrogram(similarity_df, output_dir="output"):
    """
    Plot a dendrogram to visually inspect the hierarchical clustering structure.
    
    Args:
        similarity_df: Similarity matrix DataFrame
        output_dir: Directory to save output
    """
    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1 - similarity_df.values
    
    # Ensure distance matrix is valid (no negative values)
    distance_matrix = np.maximum(distance_matrix, 0)
    
    # Compute linkage
    Z = linkage(squareform(distance_matrix), method='average')
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot dendrogram
    dendrogram(
        Z,
        labels=similarity_df.index,
        orientation='top',
        leaf_font_size=12,
        color_threshold=0.7 * max(Z[:, 2])  # Adjust color threshold for better visualization
    )
    
    plt.title('Hierarchical Clustering Dendrogram for Pauline Letters', fontsize=16)
    plt.xlabel('Letters', fontsize=14)
    plt.ylabel('Distance (1 - Similarity)', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'dendrogram.png'), dpi=300)
    plt.close()
    
    print(f"Dendrogram saved to {os.path.join(output_dir, 'dendrogram.png')}")
    
    return Z

def calculate_silhouette_scores(similarity_df, max_clusters=10, output_dir="output"):
    """
    Calculate silhouette scores for different numbers of clusters.
    
    Args:
        similarity_df: Similarity matrix DataFrame
        max_clusters: Maximum number of clusters to evaluate
        output_dir: Directory to save output
    """
    # Convert similarity to distance
    distance_matrix = 1 - similarity_df.values
    
    # Ensure distance matrix is valid
    distance_matrix = np.maximum(distance_matrix, 0)
    
    # Calculate silhouette scores for different cluster counts
    silhouette_scores = []
    cluster_range = range(2, min(max_clusters + 1, len(similarity_df)))
    
    for n_clusters in cluster_range:
        # Skip if we have too few samples
        if n_clusters >= len(similarity_df):
            continue
            
        # Perform clustering
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        cluster_labels = clusterer.fit_predict(distance_matrix)
        
        # Calculate silhouette score
        if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette
            score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.plot(list(cluster_range), silhouette_scores, 'bo-', linewidth=2.5)
    plt.xlabel('Number of Clusters', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)
    plt.title('Silhouette Score for Different Numbers of Clusters', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(list(cluster_range))
    
    # Add labels at data points
    for i, score in enumerate(silhouette_scores):
        plt.annotate(f"{score:.3f}", 
                    (list(cluster_range)[i], score),
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center')
        
    # Highlight the optimal number of clusters
    optimal_n_clusters = list(cluster_range)[np.argmax(silhouette_scores)]
    plt.axvline(x=optimal_n_clusters, color='r', linestyle='--', alpha=0.5)
    plt.annotate(f"Optimal: {optimal_n_clusters} clusters",
                xy=(optimal_n_clusters, max(silhouette_scores)),
                xytext=(optimal_n_clusters + 1, max(silhouette_scores) - 0.05),
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'silhouette_scores.png'), dpi=300)
    plt.close()
    
    print(f"Silhouette analysis saved to {os.path.join(output_dir, 'silhouette_scores.png')}")
    print(f"Optimal number of clusters based on silhouette score: {optimal_n_clusters}")
    
    return optimal_n_clusters, silhouette_scores

def run_clustering_with_optimal(similarity_df, n_clusters, output_dir="output"):
    """
    Run the clustering with the optimal number of clusters and visualize results.
    
    Args:
        similarity_df: Similarity matrix DataFrame
        n_clusters: Number of clusters to use
        output_dir: Directory to save output
    """
    # Convert similarity to distance
    distance_matrix = 1 - similarity_df.values
    
    # Ensure distance matrix is valid
    distance_matrix = np.maximum(distance_matrix, 0)
    
    # Perform clustering
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)
    
    # Create cluster mapping
    letter_clusters = {}
    for i, letter in enumerate(similarity_df.index):
        letter_clusters[letter] = cluster_labels[i]
    
    # Create cluster summary
    clusters = {}
    for letter, cluster in letter_clusters.items():
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(letter)
    
    # Create a heat map with ordered clustering
    plt.figure(figsize=(14, 12))
    
    # Create a mask for the upper triangle
    mask = np.zeros_like(similarity_df.values, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    
    # Get the order of letters based on clustering
    order = []
    for cluster_id in sorted(clusters.keys()):
        order.extend(sorted(clusters[cluster_id]))
    
    # Reorder the similarity matrix
    ordered_df = similarity_df.loc[order, order]
    
    # Create heatmap
    sns.heatmap(
        ordered_df,
        annot=True,
        cmap='RdYlGn',
        fmt='.2f',
        mask=mask,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8},
        annot_kws={"size": 10}
    )
    
    plt.title(f'Pauline Letters Similarity Heatmap with {n_clusters} Clusters', fontsize=16)
    plt.tight_layout()
    
    # Save heatmap
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'similarity_heatmap_{n_clusters}_clusters.png'), dpi=300)
    plt.close()
    
    # Create text report
    report_path = os.path.join(output_dir, f'optimal_clustering_{n_clusters}.txt')
    with open(report_path, 'w') as f:
        f.write(f"# Optimal Clustering Analysis with {n_clusters} Clusters\n\n")
        
        for cluster_id, letters in sorted(clusters.items()):
            f.write(f"## Cluster {cluster_id}:\n\n")
            
            # Map codes to full names
            letter_names = {
                "ROM-075": "Romans",
                "1CO-076": "1 Corinthians",
                "2CO-077": "2 Corinthians",
                "GAL-078": "Galatians",
                "EPH-079": "Ephesians",
                "PHP-080": "Philippians",
                "COL-081": "Colossians",
                "1TH-082": "1 Thessalonians",
                "2TH-083": "2 Thessalonians",
                "1TI-084": "1 Timothy",
                "2TI-085": "2 Timothy",
                "TIT-086": "Titus",
                "PHM-087": "Philemon"
            }
            
            for letter in letters:
                full_name = letter_names.get(letter, letter)
                f.write(f"- {full_name} ({letter})\n")
            
            f.write("\n")
            
            # Calculate within-cluster similarities
            within_similarities = []
            for i, letter1 in enumerate(letters):
                idx1 = similarity_df.index.get_loc(letter1)
                for j, letter2 in enumerate(letters):
                    if i < j:
                        idx2 = similarity_df.index.get_loc(letter2)
                        within_similarities.append(similarity_df.iloc[idx1, idx2])
            
            if within_similarities:
                f.write("Within-cluster similarity statistics:\n")
                f.write(f"- Average: {np.mean(within_similarities):.4f}\n")
                f.write(f"- Min: {np.min(within_similarities):.4f}\n")
                f.write(f"- Max: {np.max(within_similarities):.4f}\n")
                f.write("\n")
            
        f.write("# Between-Cluster Similarities\n\n")
        
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                between_similarities = []
                
                for letter1 in clusters[i]:
                    idx1 = similarity_df.index.get_loc(letter1)
                    for letter2 in clusters[j]:
                        idx2 = similarity_df.index.get_loc(letter2)
                        between_similarities.append(similarity_df.iloc[idx1, idx2])
                
                if between_similarities:
                    f.write(f"Between Cluster {i} and Cluster {j}:\n")
                    f.write(f"- Average: {np.mean(between_similarities):.4f}\n")
                    f.write(f"- Min: {np.min(between_similarities):.4f}\n")
                    f.write(f"- Max: {np.max(between_similarities):.4f}\n")
                    f.write("\n")
    
    print(f"Clustering report with {n_clusters} clusters saved to {report_path}")
    print(f"Heatmap with {n_clusters} clusters saved to {os.path.join(output_dir, f'similarity_heatmap_{n_clusters}_clusters.png')}")
    
    return letter_clusters

def main():
    """Main function."""
    # Load similarity matrix
    similarity_df = load_similarity_matrix()
    if similarity_df is None:
        return
    
    print("Loaded similarity matrix with shape:", similarity_df.shape)
    
    # Create output directory
    output_dir = "cluster_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot dendrogram
    print("\n1. Generating dendrogram for visual inspection...")
    Z = plot_dendrogram(similarity_df, output_dir)
    
    # Calculate silhouette scores
    print("\n2. Calculating silhouette scores...")
    optimal_n_clusters, silhouette_scores = calculate_silhouette_scores(similarity_df, max_clusters=8, output_dir=output_dir)
    
    # Run clustering with the optimal number of clusters
    print(f"\n3. Running clustering with optimal {optimal_n_clusters} clusters...")
    letter_clusters = run_clustering_with_optimal(similarity_df, optimal_n_clusters, output_dir)
    
    # Let's also try 4, 5, and 6 clusters as suggested
    for n_clusters in [4, 5, 6]:
        if n_clusters != optimal_n_clusters:
            print(f"\n4. Additional analysis with {n_clusters} clusters...")
            run_clustering_with_optimal(similarity_df, n_clusters, output_dir)
    
    print("\nAnalysis complete! Check the 'cluster_analysis' directory for results.")

if __name__ == "__main__":
    main() 