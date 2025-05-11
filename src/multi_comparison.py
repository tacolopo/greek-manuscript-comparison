"""
Module for comparing multiple Greek manuscripts simultaneously.
"""

import os
from typing import List, Dict, Tuple, Optional, Union, Any
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pyvis.network import Network
from tabulate import tabulate

from .preprocessing import GreekTextPreprocessor
from .features import FeatureExtractor
from .similarity import SimilarityCalculator
from .advanced_nlp import AdvancedGreekProcessor


class MultipleManuscriptComparison:
    """Compare multiple Greek manuscripts simultaneously."""
    
    def __init__(self, 
                 use_advanced_nlp: bool = True,
                 output_dir: str = 'output',
                 visualizations_dir: str = 'visualizations'):
        """
        Initialize the multiple manuscript comparison.
        
        Args:
            use_advanced_nlp: Whether to use advanced NLP features
            output_dir: Directory to save output files
            visualizations_dir: Directory to save visualizations
        """
        self.use_advanced_nlp = use_advanced_nlp
        self.output_dir = output_dir
        self.visualizations_dir = visualizations_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(visualizations_dir, exist_ok=True)
        
        # Initialize components
        self.preprocessor = GreekTextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.similarity_calculator = SimilarityCalculator()
        
        # Initialize advanced NLP processor if requested
        self.advanced_processor = None
        if use_advanced_nlp:
            try:
                self.advanced_processor = AdvancedGreekProcessor()
            except Exception as e:
                warnings.warn(f"Error initializing advanced NLP processor: {e}. Advanced NLP features will not be available.")
    
    def preprocess_manuscripts(self, 
                              manuscript_paths: List[str], 
                              manuscript_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Preprocess multiple manuscripts.
        
        Args:
            manuscript_paths: List of paths to manuscript files
            manuscript_names: Optional list of names for the manuscripts
            
        Returns:
            Dictionary mapping manuscript names to preprocessed data
        """
        preprocessed_data = {}
        
        # Generate names if not provided
        if manuscript_names is None:
            manuscript_names = [os.path.basename(path).split('.')[0] for path in manuscript_paths]
        
        # Preprocess each manuscript
        for name, path in tqdm(zip(manuscript_names, manuscript_paths), 
                               desc="Preprocessing manuscripts", 
                               total=len(manuscript_paths)):
            try:
                # Preprocess the manuscript
                preprocessed = self.preprocessor.preprocess_file(path)
                preprocessed_data[name] = preprocessed
                
                # Add advanced NLP features if available
                if self.advanced_processor:
                    try:
                        nlp_features = self.advanced_processor.process_document(preprocessed['normalized_text'])
                        preprocessed['nlp_features'] = nlp_features
                    except Exception as e:
                        warnings.warn(f"Error processing advanced NLP features for {name}: {e}")
                
            except Exception as e:
                warnings.warn(f"Error preprocessing manuscript {name}: {e}")
        
        return preprocessed_data
    
    def extract_features(self, preprocessed_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Extract features from preprocessed manuscripts.
        
        Args:
            preprocessed_data: Dictionary of preprocessed manuscripts
            
        Returns:
            Dictionary mapping manuscript names to extracted features
        """
        features_data = {}
        
        # Extract features for each manuscript
        for name, preprocessed in tqdm(preprocessed_data.items(), 
                                      desc="Extracting features", 
                                      total=len(preprocessed_data)):
            try:
                # Extract basic features
                features = self.feature_extractor.extract_all_features(preprocessed)
                features_data[name] = features
                
                # Add advanced syntactic features if available
                if self.advanced_processor and 'nlp_features' in preprocessed and 'pos_tags' in preprocessed['nlp_features']:
                    try:
                        syntactic_features = self.advanced_processor.extract_syntactic_features(
                            preprocessed['nlp_features']['pos_tags']
                        )
                        features['syntactic_features'] = syntactic_features
                    except Exception as e:
                        warnings.warn(f"Error extracting syntactic features for {name}: {e}")
                
            except Exception as e:
                warnings.warn(f"Error extracting features for manuscript {name}: {e}")
        
        return features_data
    
    def calculate_similarity_matrix(self, features: Dict[str, Dict]) -> pd.DataFrame:
        """
        Calculate similarity matrix between manuscripts based on their features.
        
        Args:
            features: Dictionary mapping manuscript IDs to their extracted features
            
        Returns:
            DataFrame containing pairwise similarities
        """
        # Use the similarity calculator's matrix calculation method directly
        return self.similarity_calculator.calculate_similarity_matrix(features)
    
    def cluster_manuscripts(self, 
                           similarity_df: pd.DataFrame, 
                           n_clusters: int = 3,
                           method: str = 'kmeans',
                           min_samples: int = 2,
                           eps: float = 0.5) -> Dict[str, Any]:
        """
        Cluster manuscripts based on similarity.
        
        Args:
            similarity_df: DataFrame with similarity matrix
            n_clusters: Number of clusters to create
            method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
            min_samples: Minimum samples for DBSCAN
            eps: Maximum distance between samples for DBSCAN
            
        Returns:
            Dictionary with clustering results
        """
        # Convert similarity matrix to distance matrix (1 - similarity)
        distance_matrix = 1 - similarity_df.values
        
        # Ensure the distance matrix is valid (no negative values)
        distance_matrix = np.maximum(distance_matrix, 0)
        
        # Apply dimensionality reduction for visualization
        try:
            # Multi-dimensional scaling to 2D
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            coordinates = mds.fit_transform(distance_matrix)
            
            # t-SNE with adjusted perplexity for small sample size
            n_samples = len(similarity_df)
            perplexity = min(30, n_samples - 1)  # Adjust perplexity based on sample size
            tsne = TSNE(
                n_components=2,
                metric='precomputed',
                random_state=42,
                perplexity=perplexity,
                init='random'  # Use random initialization for precomputed metric
            )
            coordinates_tsne = tsne.fit_transform(distance_matrix)
        except Exception as e:
            warnings.warn(f"Error in dimensionality reduction: {e}")
            coordinates = np.zeros((len(similarity_df), 2))
            coordinates_tsne = np.zeros((len(similarity_df), 2))
        
        # Perform clustering
        labels = None
        if method == 'kmeans':
            # Need to use the coordinates for K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(coordinates)
        elif method == 'hierarchical':
            # Can use the distance matrix directly
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters, 
                metric='precomputed',  # Use precomputed distances
                linkage='average'
            )
            labels = hierarchical.fit_predict(distance_matrix)
        elif method == 'dbscan':
            # Can use the distance matrix
            dbscan = DBSCAN(
                eps=eps,  # Maximum distance between samples
                min_samples=min_samples,  # Minimum number of samples in a neighborhood
                metric='precomputed'
            )
            labels = dbscan.fit_predict(distance_matrix)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Create dictionary with results
        result = {
            'coordinates': coordinates,
            'coordinates_tsne': coordinates_tsne,
            'labels': labels,
            'manuscript_names': similarity_df.index.tolist(),
            'distance_matrix': distance_matrix,
            'similarity_matrix': similarity_df.values,
            'clustering_method': method,
            'n_clusters': n_clusters
        }
        
        return result
    
    def generate_visualizations(self, clustering_result: Dict[str, Any], 
                              similarity_df: pd.DataFrame,
                              threshold: float = 0.5) -> Dict[str, str]:
        """
        Generate visualizations for clustering results.
        
        Args:
            clustering_result: Dictionary with clustering results
            similarity_df: DataFrame with similarity matrix
            threshold: Similarity threshold for network visualization
            
        Returns:
            Dictionary mapping visualization types to their file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # Get data from clustering result
        coordinates = clustering_result['coordinates']
        coordinates_tsne = clustering_result['coordinates_tsne']
        labels = clustering_result['labels']
        manuscript_names = clustering_result['manuscript_names']
        method = clustering_result['clustering_method']
        
        # Add descriptive labels for Pauline letters
        letter_labels = {}
        for name in manuscript_names:
            if "ROM" in name:
                letter_labels[name] = "Romans"
            elif "1CO" in name:
                letter_labels[name] = "1 Corinthians"
            elif "2CO" in name:
                letter_labels[name] = "2 Corinthians"
            elif "GAL" in name:
                letter_labels[name] = "Galatians"
            elif "EPH" in name:
                letter_labels[name] = "Ephesians"
            elif "PHP" in name:
                letter_labels[name] = "Philippians"
            elif "COL" in name:
                letter_labels[name] = "Colossians"
            elif "1TH" in name:
                letter_labels[name] = "1 Thessalonians"
            elif "2TH" in name:
                letter_labels[name] = "2 Thessalonians"
            elif "1TI" in name:
                letter_labels[name] = "1 Timothy"
            elif "2TI" in name:
                letter_labels[name] = "2 Timothy"
            elif "TIT" in name:
                letter_labels[name] = "Titus"
            elif "PHM" in name:
                letter_labels[name] = "Philemon"
            else:
                letter_labels[name] = name
        
        # Plot MDS visualization
        plt.figure(figsize=(12, 10))
        
        # Create a color map for the clusters
        unique_labels = sorted(set(labels))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        # Plot each cluster with a different color
        for label in unique_labels:
            mask = labels == label
            plt.scatter(
                coordinates[mask, 0], 
                coordinates[mask, 1], 
                c=[color_map[label]],
                s=150, 
                label=f"Cluster {label}"
            )
        
        # Add manuscript names as labels
        for i, name in enumerate(manuscript_names):
            plt.annotate(
                letter_labels[name], 
                (coordinates[i, 0], coordinates[i, 1]),
                fontsize=12,
                font='serif',
                weight='bold'
            )
            
        plt.title(f'Pauline Letters Stylometric Analysis (MDS) - {method.upper()}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        mds_path = os.path.join(self.visualizations_dir, 'clustering_mds.png')
        plt.savefig(mds_path, dpi=300)
        plt.close()
        
        # Plot t-SNE visualization
        plt.figure(figsize=(12, 10))
        
        # Plot each cluster with a different color
        for label in unique_labels:
            mask = labels == label
            plt.scatter(
                coordinates_tsne[mask, 0], 
                coordinates_tsne[mask, 1], 
                c=[color_map[label]],
                s=150, 
                label=f"Cluster {label}"
            )
        
        # Add manuscript names as labels
        for i, name in enumerate(manuscript_names):
            plt.annotate(
                letter_labels[name], 
                (coordinates_tsne[i, 0], coordinates_tsne[i, 1]),
                fontsize=12,
                font='serif',
                weight='bold'
            )
            
        plt.title(f'Pauline Letters Stylometric Analysis (t-SNE) - {method.upper()}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        tsne_path = os.path.join(self.visualizations_dir, 'clustering_tsne.png')
        plt.savefig(tsne_path, dpi=300)
        plt.close()
        
        # Create similarity heatmap
        plt.figure(figsize=(14, 12))
        
        # Rename the index and columns with full letter names
        similarity_df_renamed = similarity_df.copy()
        similarity_df_renamed.index = [letter_labels[name] for name in similarity_df.index]
        similarity_df_renamed.columns = [letter_labels[name] for name in similarity_df.columns]
        
        # Create a mask for the upper triangle to avoid redundancy
        mask = np.zeros_like(similarity_df_renamed.values, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
        
        # Create the heatmap
        sns.heatmap(
            similarity_df_renamed, 
            annot=True, 
            cmap='RdYlGn', 
            fmt='.2f',
            mask=mask,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8},
            annot_kws={"size": 10}
        )
        
        plt.title('Pauline Letters Similarity Heatmap', fontsize=16)
        plt.tight_layout()
        
        heatmap_path = os.path.join(self.visualizations_dir, 'similarity_heatmap.png')
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        
        # Create network visualization
        net = Network(height="900px", width="100%", notebook=False, directed=False)
        net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=150, spring_strength=0.05)
        
        # Add nodes with cluster colors
        unique_labels = np.unique(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        for i, name in enumerate(manuscript_names):
            cluster_id = labels[i]
            color = '#{:02x}{:02x}{:02x}'.format(
                int(colors[cluster_id][0] * 255),
                int(colors[cluster_id][1] * 255),
                int(colors[cluster_id][2] * 255)
            )
            
            # Calculate node size based on manuscript length (normalized)
            
            # Add the node with label, title, and color
            net.add_node(
                name, 
                label=letter_labels[name], 
                title=f"{letter_labels[name]} (Cluster {cluster_id})", 
                color=color,
                size=40,
                font={'size': 18, 'face': 'serif', 'color': 'black'}
            )
        
        # Add edges for similarities above threshold
        for i, name1 in enumerate(manuscript_names):
            for j, name2 in enumerate(manuscript_names):
                if i < j:
                    similarity = similarity_df.iloc[i, j]
                    if similarity >= threshold:
                        width = similarity * 8  # Scale width by similarity
                        edge_color = '#{:02x}{:02x}{:02x}'.format(
                            int(255 * min(1, max(0, (similarity - threshold) / (1 - threshold)))),
                            int(255 * min(1, max(0, (similarity - threshold) / (1 - threshold)))),
                            0
                        )
                        net.add_edge(
                            name1, 
                            name2, 
                            value=width, 
                            title=f"Similarity: {similarity:.4f}",
                            color=edge_color
                        )
        
        # Configure the physics
        net.show_buttons(filter_=['physics'])
        
        network_path = os.path.join(self.visualizations_dir, 'manuscript_network.html')
        net.save_graph(network_path)
        
        return {
            'mds': mds_path,
            'tsne': tsne_path,
            'heatmap': heatmap_path,
            'network': network_path
        }
    
    def visualize_similarity_heatmap(self, 
                                    similarity_df: pd.DataFrame, 
                                    clustering_result: Dict) -> None:
        """Create enhanced similarity heatmap."""
        plt.figure(figsize=(12, 10))
        
        # Create clustered heatmap
        linkage = hierarchy.linkage(similarity_df, method='average')
        sns.clustermap(similarity_df,
                      cmap='viridis',
                      annot=True,
                      fmt='.2f',
                      row_linkage=linkage,
                      col_linkage=linkage,
                      figsize=(15, 15))
        
        plt.title("Manuscript Similarity Heatmap (Clustered)", pad=20)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.visualizations_dir, 'similarity_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_clustering(self, 
                            clustering_result: Dict[str, Any],
                            title: str = 'Manuscript Clustering',
                            filename: str = 'manuscript_clustering.png') -> None:
        """
        Visualize the clustering of manuscripts.
        
        Args:
            clustering_result: Dictionary with clustering results
            title: Plot title
            filename: Output filename
        """
        # Extract data
        coordinates = clustering_result['coordinates']
        labels = clustering_result['labels']
        manuscript_names = clustering_result['manuscript_names']
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot with clusters
        scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], 
                             c=labels, 
                             cmap='viridis', 
                             s=100, 
                             alpha=0.8)
        
        # Add labels for each point
        for i, name in enumerate(manuscript_names):
            plt.annotate(name, 
                        (coordinates[i, 0], coordinates[i, 1]),
                        fontsize=9,
                        ha='right', 
                        va='bottom')
        
        # Add legend
        plt.colorbar(scatter, label='Cluster')
        
        plt.title(title, fontsize=16)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.visualizations_dir, filename)
        plt.savefig(output_path)
        plt.close()
        
        print(f"Saved clustering visualization to {output_path}")
    
    def visualize_similarity_network(self, 
                                    similarity_df: pd.DataFrame,
                                    threshold: float = 0.5,
                                    title: str = 'Manuscript Similarity Network',
                                    filename: str = 'manuscript_network.html') -> None:
        """
        Create an interactive network visualization of manuscript similarities.
        
        Args:
            similarity_df: DataFrame with similarity matrix
            threshold: Minimum similarity to show a connection
            title: Plot title
            filename: Output filename
        """
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (manuscripts)
        for name in similarity_df.index:
            G.add_node(name)
        
        # Add edges (similarities above threshold)
        for i, name1 in enumerate(similarity_df.index):
            for j, name2 in enumerate(similarity_df.index):
                if i < j:  # Avoid duplicate edges and self-loops
                    similarity = similarity_df.iloc[i, j]
                    if similarity >= threshold:
                        G.add_edge(name1, name2, weight=similarity)
        
        # Convert to interactive visualization
        net = Network(height="750px", width="100%", notebook=False, directed=False)
        
        # Add nodes and edges
        for node in G.nodes():
            net.add_node(node, label=node, title=node, size=25)
        
        for edge in G.edges(data=True):
            source, target, attr = edge
            weight = attr['weight']
            width = weight * 5  # Scale width by similarity
            net.add_edge(source, target, value=width, title=f"Similarity: {weight:.4f}")
        
        # Set options
        net.set_options("""
        var options = {
            "nodes": {
                "font": {
                    "size": 16
                }
            },
            "edges": {
                "color": {
                    "inherit": true
                },
                "smooth": {
                    "enabled": false,
                    "type": "continuous"
                }
            },
            "physics": {
                "repulsion": {
                    "centralGravity": 0.2,
                    "springLength": 200,
                    "springConstant": 0.05,
                    "nodeDistance": 100,
                    "damping": 0.09
                },
                "maxVelocity": 50,
                "minVelocity": 0.75,
                "solver": "repulsion"
            }
        }
        """)
        
        # Save network
        output_path = os.path.join(self.visualizations_dir, filename)
        net.save_graph(output_path)
        
        print(f"Saved interactive network visualization to {output_path}")
    
    def generate_cluster_report(self, 
                               clustering_result: Dict[str, Any],
                               preprocessed_data: Dict[str, Dict],
                               features_data: Dict[str, Dict],
                               output_file: str = 'clustering_report.txt') -> None:
        """
        Generate a detailed report about the clustering results.
        
        Args:
            clustering_result: Dictionary with clustering results
            preprocessed_data: Dictionary of preprocessed manuscripts
            features_data: Dictionary of manuscript features
            output_file: Output filename
        """
        # Extract data
        labels = clustering_result['labels']
        manuscript_names = clustering_result['manuscript_names']
        similarity_matrix = clustering_result['similarity_matrix']
        method = clustering_result['clustering_method']
        
        # Group manuscripts by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[int(label)].append(manuscript_names[i])
        
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id, members in clusters.items():
            # Calculate within-cluster similarity
            within_similarities = []
            for i, name1 in enumerate(members):
                idx1 = manuscript_names.index(name1)
                for j, name2 in enumerate(members):
                    if i < j:
                        idx2 = manuscript_names.index(name2)
                        within_similarities.append(similarity_matrix[idx1, idx2])
            
            # Calculate average word count for the cluster
            word_counts = [len(preprocessed_data[name]['words']) for name in members]
            
            # Calculate average sentence length
            sentence_lengths = []
            for name in members:
                lengths = features_data[name]['sentence_stats']['mean_sentence_length']
                sentence_lengths.append(lengths)
            
            # Store statistics
            cluster_stats[cluster_id] = {
                'members': members,
                'size': len(members),
                'avg_within_similarity': np.mean(within_similarities) if within_similarities else 0,
                'min_within_similarity': np.min(within_similarities) if within_similarities else 0,
                'max_within_similarity': np.max(within_similarities) if within_similarities else 0,
                'avg_word_count': np.mean(word_counts),
                'avg_sentence_length': np.mean(sentence_lengths)
            }
        
        # Write report
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w') as f:
            f.write(f"Manuscript Clustering Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Clustering Method: {method}\n")
            f.write(f"Number of Clusters: {len(clusters)}\n")
            f.write(f"Total Manuscripts: {len(manuscript_names)}\n\n")
            
            f.write("Cluster Summary\n")
            f.write("-" * 60 + "\n\n")
            
            for cluster_id, stats in cluster_stats.items():
                f.write(f"Cluster {cluster_id}:\n")
                f.write(f"  Members ({stats['size']}):\n")
                for member in stats['members']:
                    f.write(f"    - {member}\n")
                
                f.write(f"  Average Within-Cluster Similarity: {stats['avg_within_similarity']:.4f}\n")
                f.write(f"  Min/Max Within-Cluster Similarity: {stats['min_within_similarity']:.4f} / {stats['max_within_similarity']:.4f}\n")
                f.write(f"  Average Word Count: {stats['avg_word_count']:.1f}\n")
                f.write(f"  Average Sentence Length: {stats['avg_sentence_length']:.2f}\n")
                f.write("\n")
            
            # Calculate between-cluster similarities
            f.write("Between-Cluster Similarities\n")
            f.write("-" * 60 + "\n\n")
            
            cluster_ids = sorted(clusters.keys())
            for i, cluster1 in enumerate(cluster_ids):
                for j, cluster2 in enumerate(cluster_ids):
                    if i < j:
                        # Calculate similarity between clusters
                        between_similarities = []
                        for name1 in clusters[cluster1]:
                            idx1 = manuscript_names.index(name1)
                            for name2 in clusters[cluster2]:
                                idx2 = manuscript_names.index(name2)
                                between_similarities.append(similarity_matrix[idx1, idx2])
                        
                        avg_similarity = np.mean(between_similarities)
                        min_similarity = np.min(between_similarities)
                        max_similarity = np.max(between_similarities)
                        
                        f.write(f"Cluster {cluster1} <--> Cluster {cluster2}:\n")
                        f.write(f"  Average Similarity: {avg_similarity:.4f}\n")
                        f.write(f"  Min/Max Similarity: {min_similarity:.4f} / {max_similarity:.4f}\n\n")
            
            # Calculate discriminative features
            f.write("Cluster Characteristics\n")
            f.write("-" * 60 + "\n\n")
            
            # For simplicity, just report key statistics for each cluster
            for cluster_id, stats in cluster_stats.items():
                f.write(f"Cluster {cluster_id} Characteristics:\n")
                
                # Collect vocabulary richness metrics
                richness_metrics = {
                    'unique_tokens_ratio': [],
                    'hapax_legomena_ratio': [],
                    'yule_k': []
                }
                
                for name in stats['members']:
                    for metric in richness_metrics:
                        value = features_data[name]['vocabulary_richness'][metric]
                        richness_metrics[metric].append(value)
                
                # Report average values
                f.write("  Vocabulary Richness:\n")
                for metric, values in richness_metrics.items():
                    avg_value = np.mean(values)
                    f.write(f"    - {metric.replace('_', ' ').title()}: {avg_value:.4f}\n")
                
                f.write("\n")
        
        print(f"Saved clustering report to {output_path}")
    
    def generate_summary_table(self, 
                             features_data: Dict[str, Dict],
                             similarity_df: pd.DataFrame,
                             clustering_result: Dict) -> str:
        """
        Generate a summary table of the analysis results.
        
        Args:
            features_data: Dictionary of extracted features
            similarity_df: Similarity matrix
            clustering_result: Clustering results
            
        Returns:
            Formatted table string
        """
        # Prepare summary data
        summary_data = []
        manuscript_names = list(features_data.keys())
        labels = clustering_result['labels']
        
        for i, name in enumerate(manuscript_names):
            features = features_data[name]
            
            # Calculate average similarity with other manuscripts
            similarities = similarity_df.loc[name].drop(name)
            avg_similarity = similarities.mean()
            
            # Get key metrics
            vocab_richness = features['vocabulary_richness']
            sentence_stats = features['sentence_stats']
            transition_patterns = features['transition_patterns']
            
            summary_data.append([
                name,  # Manuscript name
                f"Cluster {labels[i]}",  # Cluster assignment
                f"{avg_similarity:.3f}",  # Average similarity
                f"{vocab_richness['unique_tokens_ratio']:.3f}",  # Vocabulary richness
                f"{sentence_stats['mean_sentence_length']:.1f}",  # Avg sentence length
                f"{transition_patterns['length_transition_smoothness']:.3f}",  # Style consistency
                f"{vocab_richness['hapax_ratio']:.3f}",  # Unique word usage
                f"{transition_patterns['sentence_rhythm_consistency']:.3f}"  # Writing rhythm
            ])
        
        # Create table with headers
        headers = [
            "Manuscript",
            "Cluster",
            "Avg Similarity",
            "Vocab Richness",
            "Avg Sent Length",
            "Style Consistency",
            "Unique Words",
            "Writing Rhythm"
        ]
        
        return tabulate(summary_data, headers=headers, tablefmt="grid")
    
    def generate_cluster_summary(self, clustering_result: Dict) -> str:
        """
        Generate a summary of the clustering results.
        
        Args:
            clustering_result: Clustering results
            
        Returns:
            Formatted table string
        """
        # Group manuscripts by cluster
        clusters = {}
        for i, label in enumerate(clustering_result['labels']):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(clustering_result['manuscript_names'][i])
        
        # Prepare cluster summary
        summary_data = []
        for cluster_id in sorted(clusters.keys()):
            members = clusters[cluster_id]
            summary_data.append([
                f"Cluster {cluster_id}",
                len(members),
                ", ".join(members)
            ])
        
        headers = ["Cluster", "Size", "Members"]
        return tabulate(summary_data, headers=headers, tablefmt="grid")
    
    def visualize_feature_distributions(self, features_data: Dict[str, Dict]) -> None:
        """Create feature distribution plots."""
        # Extract key metrics for visualization
        metrics = {
            'Vocabulary Richness': [d['vocabulary_richness']['unique_tokens_ratio'] for d in features_data.values()],
            'Sentence Length': [d['sentence_stats']['mean_sentence_length'] for d in features_data.values()],
            'Style Consistency': [d['transition_patterns']['length_transition_smoothness'] for d in features_data.values()],
            'Writing Rhythm': [d['transition_patterns']['sentence_rhythm_consistency'] for d in features_data.values()]
        }
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Distribution of Key Stylometric Features', fontsize=16)
        
        for (title, values), ax in zip(metrics.items(), axes.flat):
            sns.histplot(values, ax=ax, kde=True)
            ax.set_title(title)
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
        
        plt.tight_layout()
        output_path = os.path.join(self.visualizations_dir, 'feature_distributions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_multiple_manuscripts(self, manuscripts: Dict[str, str], 
                                   method: str = 'hierarchical',
                                   n_clusters: int = 3,
                                   min_samples: int = 2,
                                   eps: float = 0.5,
                                   use_advanced_nlp: bool = False) -> Dict[str, Any]:
        """
        Compare multiple manuscripts and analyze their relationships.
        
        Args:
            manuscripts: Dictionary mapping manuscript IDs to their texts
            method: Clustering method ('hierarchical', 'kmeans', or 'dbscan')
            n_clusters: Number of clusters for hierarchical/kmeans
            min_samples: Minimum samples for DBSCAN
            eps: Maximum distance between samples for DBSCAN
            use_advanced_nlp: Whether to use advanced NLP features
            
        Returns:
            Dictionary with analysis results
        """
        print("Processing manuscripts and extracting features...")
        
        # Extract the base letter name without chapter info (e.g., "ROM-075" from "ROM-075-1")
        letter_texts = {}
        for manuscript_id, text in manuscripts.items():
            # Extract just the letter identifier without chapter numbers
            letter_id = manuscript_id
            if letter_id not in letter_texts:
                letter_texts[letter_id] = text
            else:
                letter_texts[letter_id] += " " + text
        
        print(f"Analyzing {len(letter_texts)} complete letters (combined from chapters)")
        
        # Preprocess all letters (not individual chapters)
        preprocessed = {}
        for letter_id, text in tqdm(letter_texts.items()):
            preprocessed[letter_id] = self.preprocessor.preprocess(text)
        
        # Fit TF-IDF vectorizer on all texts
        normalized_texts = [
            preprocessed[letter_id].get('normalized_text', ' '.join(preprocessed[letter_id]['words']))
            for letter_id in letter_texts
        ]
        self.feature_extractor.fit(normalized_texts)
        
        # Extract features for each letter
        features = {}
        for letter_id in tqdm(letter_texts):
            features[letter_id] = self.feature_extractor.extract_all_features(preprocessed[letter_id])
            
        # Calculate similarity matrix
        similarity_matrix = self.calculate_similarity_matrix(features)
        
        # Save similarity matrix to disk for future reference
        os.makedirs(self.output_dir, exist_ok=True)
        similarity_matrix.to_csv(os.path.join(self.output_dir, "similarity_matrix.csv"))
        try:
            import pickle
            with open(os.path.join(self.output_dir, "similarity_matrix.pkl"), "wb") as f:
                pickle.dump(similarity_matrix, f)
        except Exception as e:
            print(f"Warning: Could not save pickle version of similarity matrix: {e}")
        
        # Perform clustering
        clusters = self.cluster_manuscripts(
            similarity_matrix,
            n_clusters=n_clusters,
            method=method,
            min_samples=min_samples,
            eps=eps
        )
        
        # Generate visualizations
        visualizations = self.generate_visualizations(
            clusters,
            similarity_matrix,
            threshold=0.5
        )
        
        # Generate report
        report = self.generate_report(
            clusters,
            preprocessed,
            features,
            similarity_matrix
        )
        
        return {
            'preprocessed': preprocessed,
            'features': features,
            'similarity_matrix': similarity_matrix,
            'clusters': clusters,
            'visualizations': visualizations,
            'report': report
        }
    
    def generate_report(self, clustering_result: Dict[str, Any],
                        preprocessed_data: Dict[str, Dict],
                        features_data: Dict[str, Dict],
                        similarity_df: pd.DataFrame) -> str:
        """
        Generate a detailed report of the analysis results.
        
        Args:
            clustering_result: Dictionary with clustering results
            preprocessed_data: Dictionary of preprocessed texts
            features_data: Dictionary of extracted features
            similarity_df: DataFrame with similarity matrix
            
        Returns:
            Path to the generated report file
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get clustering information
        labels = clustering_result['labels']
        manuscript_names = clustering_result['manuscript_names']
        method = clustering_result['clustering_method']
        
        # Group manuscripts by cluster
        clusters = defaultdict(list)
        for i, name in enumerate(manuscript_names):
            clusters[labels[i]].append(name)
            
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id, members in clusters.items():
            # Calculate within-cluster similarity
            within_similarities = []
            for i, name1 in enumerate(members):
                for j, name2 in enumerate(members):
                    if i < j:
                        similarity = similarity_df.loc[name1, name2]
                        within_similarities.append(similarity)
            
            # Calculate average word count for the cluster
            word_counts = []
            for name in members:
                if 'words' in preprocessed_data[name]:
                    word_counts.append(len(preprocessed_data[name]['words']))
            
            # Calculate average sentence length
            sentence_lengths = []
            for name in members:
                if 'sentence_stats' in features_data[name]:
                    lengths = features_data[name]['sentence_stats']['mean_sentence_length']
                    sentence_lengths.append(lengths)
            
            # Store statistics
            cluster_stats[cluster_id] = {
                'members': members,
                'size': len(members),
                'avg_within_similarity': np.mean(within_similarities) if within_similarities else 0,
                'min_within_similarity': np.min(within_similarities) if within_similarities else 0,
                'max_within_similarity': np.max(within_similarities) if within_similarities else 0,
                'avg_word_count': np.mean(word_counts) if word_counts else 0,
                'avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0
            }
        
        # Generate report text
        report_path = os.path.join(self.output_dir, 'clustering_report.txt')
        with open(report_path, 'w') as f:
            f.write("Pauline Letters Stylometric Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis Method: {method.upper()}\n")
            f.write(f"Number of Clusters: {len(clusters)}\n")
            f.write(f"Total Letters Analyzed: {len(manuscript_names)}\n\n")
            
            # Overall similarity statistics
            similarities = []
            for i in range(len(manuscript_names)):
                for j in range(i + 1, len(manuscript_names)):
                    similarities.append(similarity_df.iloc[i, j])
            
            f.write("Overall Similarity Statistics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Similarity: {np.mean(similarities):.4f}\n")
            f.write(f"Minimum Similarity: {np.min(similarities):.4f}\n")
            f.write(f"Maximum Similarity: {np.max(similarities):.4f}\n")
            f.write(f"Similarity Std Dev: {np.std(similarities):.4f}\n\n")
            
            # Cluster details
            f.write("Cluster Analysis\n")
            f.write("-" * 40 + "\n\n")
            
            for cluster_id, stats in cluster_stats.items():
                f.write(f"Cluster {cluster_id}:\n")
                f.write(f"  Members ({stats['size']}):\n")
                for member in stats['members']:
                    f.write(f"    - {member}\n")
                
                f.write("\n  Statistics:\n")
                f.write(f"    Average Within-Cluster Similarity: {stats['avg_within_similarity']:.4f}\n")
                f.write(f"    Min/Max Within-Cluster Similarity: {stats['min_within_similarity']:.4f} / {stats['max_within_similarity']:.4f}\n")
                f.write(f"    Average Word Count: {stats['avg_word_count']:.1f}\n")
                f.write(f"    Average Sentence Length: {stats['avg_sentence_length']:.2f}\n\n")
            
            # Between-cluster analysis
            f.write("Between-Cluster Analysis\n")
            f.write("-" * 40 + "\n\n")
            
            cluster_ids = sorted(clusters.keys())
            for i, cluster1 in enumerate(cluster_ids):
                for j, cluster2 in enumerate(cluster_ids):
                    if i < j:
                        # Calculate similarity between clusters
                        between_similarities = []
                        for name1 in clusters[cluster1]:
                            for name2 in clusters[cluster2]:
                                similarity = similarity_df.loc[name1, name2]
                                between_similarities.append(similarity)
                        
                        avg_similarity = np.mean(between_similarities)
                        min_similarity = np.min(between_similarities)
                        max_similarity = np.max(between_similarities)
                        
                        f.write(f"Cluster {cluster1} <--> Cluster {cluster2}:\n")
                        f.write(f"  Average Similarity: {avg_similarity:.4f}\n")
                        f.write(f"  Min/Max Similarity: {min_similarity:.4f} / {max_similarity:.4f}\n\n")
            
            # Interpretation guidelines
            f.write("\nInterpretation Guidelines\n")
            f.write("-" * 40 + "\n")
            f.write("Similarity Scores:\n")
            f.write("  0.8 - 1.0: Very High Similarity (likely same author/very close relationship)\n")
            f.write("  0.6 - 0.8: High Similarity (strong stylistic connection)\n")
            f.write("  0.4 - 0.6: Moderate Similarity (some stylistic overlap)\n")
            f.write("  0.2 - 0.4: Low Similarity (limited stylistic connection)\n")
            f.write("  0.0 - 0.2: Very Low Similarity (distinct styles)\n")
        
        return report_path 