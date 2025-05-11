#!/usr/bin/env python3
"""
Script to determine the optimal number of clusters for the New Testament analysis
using multiple evaluation methods.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from kneed import KneeLocator

def load_similarity_matrix(output_dir="all_nt_output"):
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

def plot_dendrogram(similarity_df, output_dir="cluster_analysis_nt"):
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
    plt.figure(figsize=(14, 10))
    
    # Create a dictionary mapping indices to full book names
    book_names = {}
    for i, name in enumerate(similarity_df.index):
        if "-" in name:
            code = name.split("-")[0]
            if code == "ROM":
                book_names[i] = "Romans"
            elif code == "1CO":
                book_names[i] = "1 Corinthians"
            elif code == "2CO":
                book_names[i] = "2 Corinthians"
            elif code == "GAL":
                book_names[i] = "Galatians"
            elif code == "EPH":
                book_names[i] = "Ephesians"
            elif code == "PHP":
                book_names[i] = "Philippians"
            elif code == "COL":
                book_names[i] = "Colossians"
            elif code == "1TH":
                book_names[i] = "1 Thessalonians"
            elif code == "2TH":
                book_names[i] = "2 Thessalonians"
            elif code == "1TI":
                book_names[i] = "1 Timothy"
            elif code == "2TI":
                book_names[i] = "2 Timothy"
            elif code == "TIT":
                book_names[i] = "Titus"
            elif code == "PHM":
                book_names[i] = "Philemon"
            elif code == "HEB":
                book_names[i] = "Hebrews"
            elif code == "JAS":
                book_names[i] = "James"
            elif code == "1PE":
                book_names[i] = "1 Peter"
            elif code == "2PE":
                book_names[i] = "2 Peter"
            elif code == "1JN":
                book_names[i] = "1 John"
            elif code == "2JN":
                book_names[i] = "2 John"
            elif code == "3JN":
                book_names[i] = "3 John"
            elif code == "JUD":
                book_names[i] = "Jude"
            elif code == "REV":
                book_names[i] = "Revelation"
            elif code == "MAT":
                book_names[i] = "Matthew"
            elif code == "MRK":
                book_names[i] = "Mark"
            elif code == "LUK":
                book_names[i] = "Luke"
            elif code == "JHN":
                book_names[i] = "John"
            elif code == "ACT":
                book_names[i] = "Acts"
            else:
                book_names[i] = name
        else:
            book_names[i] = name
    
    # Use the full book names for the dendrogram labels
    labels = [book_names.get(i, name) for i, name in enumerate(similarity_df.index)]
    
    # Plot dendrogram
    dendrogram(
        Z,
        labels=labels,
        orientation='right',
        leaf_font_size=12,
        color_threshold=0.7 * max(Z[:, 2])  # Adjust color threshold for better visualization
    )
    
    plt.title('New Testament Hierarchical Clustering Dendrogram', fontsize=16)
    plt.xlabel('Distance (1 - Similarity)', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'dendrogram.png'), dpi=300)
    plt.close()
    
    print(f"Dendrogram saved to {os.path.join(output_dir, 'dendrogram.png')}")
    
    return Z

def calculate_cluster_metrics(similarity_df, max_clusters=10, output_dir="cluster_analysis_nt"):
    """
    Calculate silhouette scores, Calinski-Harabasz Index, and Davies-Bouldin Index
    for different numbers of clusters.
    
    Args:
        similarity_df: Similarity matrix DataFrame
        max_clusters: Maximum number of clusters to evaluate
        output_dir: Directory to save output
    """
    # Convert similarity to distance
    distance_matrix = 1 - similarity_df.values
    
    # Ensure distance matrix is valid
    distance_matrix = np.maximum(distance_matrix, 0)
    
    # Calculate metrics for different cluster counts
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    
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
        
        # Calculate silhouette score (higher is better)
        sil_score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
        silhouette_scores.append(sil_score)
        
        # Calculate Calinski-Harabasz Index (higher is better)
        # Need to use the coordinates for this
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coordinates = mds.fit_transform(distance_matrix)
        ch_score = calinski_harabasz_score(coordinates, cluster_labels)
        ch_scores.append(ch_score)
        
        # Calculate Davies-Bouldin Index (lower is better)
        db_score = davies_bouldin_score(coordinates, cluster_labels)
        db_scores.append(db_score)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure for silhouette scores
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
    
    # Try to find the elbow point (optimal number of clusters)
    try:
        kl = KneeLocator(
            list(cluster_range), 
            silhouette_scores, 
            curve='concave', 
            direction='increasing'
        )
        if kl.elbow is not None:
            optimal_n_clusters_sil = kl.elbow
            plt.axvline(x=optimal_n_clusters_sil, color='r', linestyle='--', alpha=0.5)
            plt.annotate(f"Optimal: {optimal_n_clusters_sil} clusters",
                        xy=(optimal_n_clusters_sil, silhouette_scores[list(cluster_range).index(optimal_n_clusters_sil)]),
                        xytext=(optimal_n_clusters_sil + 1, silhouette_scores[list(cluster_range).index(optimal_n_clusters_sil)] - 0.05),
                        arrowprops=dict(facecolor='red', shrink=0.05),
                        fontsize=12)
        else:
            optimal_n_clusters_sil = np.argmax(silhouette_scores) + 2
    except Exception as e:
        print(f"Error finding elbow point: {e}")
        optimal_n_clusters_sil = list(cluster_range)[np.argmax(silhouette_scores)]
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'silhouette_scores.png'), dpi=300)
    plt.close()
    
    # Create figure for Calinski-Harabasz Index
    plt.figure(figsize=(12, 8))
    plt.plot(list(cluster_range), ch_scores, 'go-', linewidth=2.5)
    plt.xlabel('Number of Clusters', fontsize=14)
    plt.ylabel('Calinski-Harabasz Index', fontsize=14)
    plt.title('Calinski-Harabasz Index for Different Numbers of Clusters', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(list(cluster_range))
    
    # Add labels at data points
    for i, score in enumerate(ch_scores):
        plt.annotate(f"{score:.1f}", 
                    (list(cluster_range)[i], score),
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center')
    
    # Find optimal number of clusters (highest CH index)
    optimal_n_clusters_ch = list(cluster_range)[np.argmax(ch_scores)]
    plt.axvline(x=optimal_n_clusters_ch, color='r', linestyle='--', alpha=0.5)
    plt.annotate(f"Optimal: {optimal_n_clusters_ch} clusters",
                xy=(optimal_n_clusters_ch, ch_scores[list(cluster_range).index(optimal_n_clusters_ch) - 2]),
                xytext=(optimal_n_clusters_ch + 1, ch_scores[list(cluster_range).index(optimal_n_clusters_ch) - 2] - 100),
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calinski_harabasz_scores.png'), dpi=300)
    plt.close()
    
    # Create figure for Davies-Bouldin Index
    plt.figure(figsize=(12, 8))
    plt.plot(list(cluster_range), db_scores, 'ro-', linewidth=2.5)
    plt.xlabel('Number of Clusters', fontsize=14)
    plt.ylabel('Davies-Bouldin Index', fontsize=14)
    plt.title('Davies-Bouldin Index for Different Numbers of Clusters', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(list(cluster_range))
    
    # Add labels at data points
    for i, score in enumerate(db_scores):
        plt.annotate(f"{score:.3f}", 
                    (list(cluster_range)[i], score),
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center')
    
    # Find optimal number of clusters (lowest DB index)
    optimal_n_clusters_db = list(cluster_range)[np.argmin(db_scores)]
    plt.axvline(x=optimal_n_clusters_db, color='r', linestyle='--', alpha=0.5)
    plt.annotate(f"Optimal: {optimal_n_clusters_db} clusters",
                xy=(optimal_n_clusters_db, db_scores[list(cluster_range).index(optimal_n_clusters_db) - 2]),
                xytext=(optimal_n_clusters_db + 1, db_scores[list(cluster_range).index(optimal_n_clusters_db) - 2] + 0.1),
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'davies_bouldin_scores.png'), dpi=300)
    plt.close()
    
    # Create a summary plot with all metrics normalized
    plt.figure(figsize=(14, 10))
    
    # Normalize scores for comparison
    norm_sil = np.array(silhouette_scores) / np.max(silhouette_scores)
    norm_ch = np.array(ch_scores) / np.max(ch_scores)
    norm_db = 1 - (np.array(db_scores) / np.max(db_scores))  # Invert DB so higher is better
    
    plt.plot(list(cluster_range), norm_sil, 'bo-', linewidth=2.5, label='Silhouette Score (higher is better)')
    plt.plot(list(cluster_range), norm_ch, 'go-', linewidth=2.5, label='Calinski-Harabasz Index (higher is better)')
    plt.plot(list(cluster_range), norm_db, 'ro-', linewidth=2.5, label='Davies-Bouldin Index (lower is better, inverted)')
    
    plt.xlabel('Number of Clusters', fontsize=14)
    plt.ylabel('Normalized Score', fontsize=14)
    plt.title('Comparison of Clustering Metrics', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(list(cluster_range))
    plt.legend(fontsize=12)
    
    # Determine consensus optimal number of clusters
    cluster_votes = [optimal_n_clusters_sil, optimal_n_clusters_ch, optimal_n_clusters_db]
    consensus = max(set(cluster_votes), key=cluster_votes.count)
    
    plt.figtext(0.5, 0.01, 
                f"Optimal clusters by method: Silhouette={optimal_n_clusters_sil}, CH Index={optimal_n_clusters_ch}, DB Index={optimal_n_clusters_db}\nConsensus: {consensus} clusters", 
                ha="center", 
                fontsize=12, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'cluster_metrics_comparison.png'), dpi=300)
    plt.close()
    
    print(f"Silhouette analysis suggests {optimal_n_clusters_sil} clusters")
    print(f"Calinski-Harabasz Index suggests {optimal_n_clusters_ch} clusters")
    print(f"Davies-Bouldin Index suggests {optimal_n_clusters_db} clusters")
    print(f"Consensus optimal number of clusters: {consensus}")
    
    return consensus, silhouette_scores, ch_scores, db_scores

def main():
    """Main function."""
    from sklearn.manifold import MDS
    
    # Load similarity matrix
    similarity_df = load_similarity_matrix()
    if similarity_df is None:
        print("Please run compare_all_nt_texts.py first to generate the similarity matrix.")
        return
    
    print("Loaded similarity matrix with shape:", similarity_df.shape)
    
    # Create output directory
    output_dir = "cluster_analysis_nt"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot dendrogram
    print("\n1. Generating dendrogram for visual inspection...")
    Z = plot_dendrogram(similarity_df, output_dir)
    
    # Calculate cluster metrics
    print("\n2. Calculating cluster evaluation metrics...")
    consensus, silhouette_scores, ch_scores, db_scores = calculate_cluster_metrics(
        similarity_df, max_clusters=10, output_dir=output_dir
    )
    
    # Save the consensus to a file
    with open(os.path.join(output_dir, "optimal_clusters.txt"), "w") as f:
        f.write(f"Consensus optimal number of clusters: {consensus}\n")
        f.write("\nSilhouette Scores:\n")
        for i, score in enumerate(silhouette_scores):
            f.write(f"{i+2} clusters: {score:.4f}\n")
        
        f.write("\nCalinski-Harabasz Scores:\n")
        for i, score in enumerate(ch_scores):
            f.write(f"{i+2} clusters: {score:.4f}\n")
        
        f.write("\nDavies-Bouldin Scores:\n")
        for i, score in enumerate(db_scores):
            f.write(f"{i+2} clusters: {score:.4f}\n")
    
    print(f"\nAnalysis complete! Check the '{output_dir}' directory for results.")
    print(f"The consensus optimal number of clusters is: {consensus}")
    
    # Now, suggest running compare_all_nt_texts.py with this optimal number
    print(f"\nRecommendation: Run compare_all_nt_texts.py with n_clusters={consensus}")
    print(f"Command: python compare_all_nt_texts.py --clusters {consensus}")

if __name__ == "__main__":
    main() 