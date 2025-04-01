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

from preprocessing import GreekTextPreprocessor
from features import FeatureExtractor
from similarity import SimilarityCalculator
from advanced_nlp import AdvancedGreekProcessor


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
    
    def calculate_similarity_matrix(self, 
                                   features_data: Dict[str, Dict], 
                                   preprocessed_data: Dict[str, Dict]) -> pd.DataFrame:
        """
        Calculate similarity matrix between all manuscripts.
        
        Args:
            features_data: Dictionary of manuscript features
            preprocessed_data: Dictionary of preprocessed manuscript data
            
        Returns:
            DataFrame containing pairwise similarity scores
        """
        # Get list of manuscript names
        manuscript_names = list(features_data.keys())
        num_manuscripts = len(manuscript_names)
        
        # Initialize similarity matrix
        similarity_matrix = np.zeros((num_manuscripts, num_manuscripts))
        
        # Calculate pairwise similarities
        for i, name1 in enumerate(tqdm(manuscript_names, desc="Calculating similarity matrix")):
            features1 = features_data[name1]
            
            for j, name2 in enumerate(manuscript_names):
                if i == j:
                    # Similarity with self is 1.0
                    similarity_matrix[i, j] = 1.0
                elif j > i:
                    # Calculate similarity
                    features2 = features_data[name2]
                    similarity_scores = self.similarity_calculator.calculate_overall_similarity(features1, features2)
                    
                    # Get overall similarity
                    overall_similarity = similarity_scores['overall_similarity']
                    
                    # Add semantic similarity if available
                    if self.advanced_processor and 'normalized_text' in preprocessed_data[name1] and 'normalized_text' in preprocessed_data[name2]:
                        try:
                            semantic_similarity = self.advanced_processor.get_semantic_similarity(
                                preprocessed_data[name1]['normalized_text'],
                                preprocessed_data[name2]['normalized_text']
                            )
                            
                            # Combine with overall similarity (giving semantic similarity some weight)
                            overall_similarity = 0.7 * overall_similarity + 0.3 * semantic_similarity
                        except Exception as e:
                            warnings.warn(f"Error calculating semantic similarity between {name1} and {name2}: {e}")
                    
                    # Store similarity
                    similarity_matrix[i, j] = overall_similarity
                    similarity_matrix[j, i] = overall_similarity  # Symmetric matrix
        
        # Create DataFrame for better visualization
        similarity_df = pd.DataFrame(similarity_matrix, 
                                    index=manuscript_names, 
                                    columns=manuscript_names)
        
        return similarity_df
    
    def cluster_manuscripts(self, 
                           similarity_df: pd.DataFrame, 
                           n_clusters: int = 3,
                           method: str = 'kmeans') -> Dict[str, Any]:
        """
        Cluster manuscripts based on similarity.
        
        Args:
            similarity_df: DataFrame with similarity matrix
            n_clusters: Number of clusters to create
            method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
            
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
            
            # t-SNE for alternative visualization
            tsne = TSNE(n_components=2, metric='precomputed', random_state=42)
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
                affinity='precomputed', 
                linkage='average'
            )
            labels = hierarchical.fit_predict(distance_matrix)
        elif method == 'dbscan':
            # Can use the distance matrix
            dbscan = DBSCAN(
                eps=0.3,  # Maximum distance between samples
                min_samples=2,  # Minimum number of samples in a neighborhood
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
    
    def visualize_similarity_heatmap(self, 
                                    similarity_df: pd.DataFrame, 
                                    title: str = 'Manuscript Similarity Heatmap',
                                    filename: str = 'manuscript_similarity_heatmap.png') -> None:
        """
        Create a heatmap of the similarity matrix.
        
        Args:
            similarity_df: DataFrame with similarity matrix
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(similarity_df, 
                   annot=True, 
                   cmap='viridis', 
                   linewidths=0.5,
                   vmin=0, 
                   vmax=1)
        
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.visualizations_dir, filename)
        plt.savefig(output_path)
        plt.close()
        
        print(f"Saved similarity heatmap to {output_path}")
    
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
    
    def compare_multiple_manuscripts(self, 
                                    manuscript_paths: List[str], 
                                    manuscript_names: Optional[List[str]] = None,
                                    n_clusters: int = 3,
                                    clustering_method: str = 'hierarchical',
                                    similarity_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Run the complete pipeline for comparing multiple manuscripts.
        
        Args:
            manuscript_paths: List of paths to manuscript files
            manuscript_names: Optional list of names for the manuscripts
            n_clusters: Number of clusters to create
            clustering_method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
            similarity_threshold: Threshold for similarity network visualization
            
        Returns:
            Dictionary with all results
        """
        # 1. Preprocess manuscripts
        print(f"Processing {len(manuscript_paths)} manuscripts...")
        preprocessed_data = self.preprocess_manuscripts(manuscript_paths, manuscript_names)
        
        # 2. Extract features
        print("Extracting features from manuscripts...")
        features_data = self.extract_features(preprocessed_data)
        
        # 3. Calculate similarity matrix
        print("Calculating similarity matrix...")
        similarity_df = self.calculate_similarity_matrix(features_data, preprocessed_data)
        
        # 4. Cluster manuscripts
        print(f"Clustering manuscripts using {clustering_method}...")
        clustering_result = self.cluster_manuscripts(
            similarity_df, 
            n_clusters=n_clusters,
            method=clustering_method
        )
        
        # 5. Generate visualizations
        print("Generating visualizations...")
        self.visualize_similarity_heatmap(similarity_df)
        self.visualize_clustering(clustering_result)
        self.visualize_similarity_network(similarity_df, threshold=similarity_threshold)
        
        # 6. Generate reports
        print("Generating reports...")
        self.generate_cluster_report(clustering_result, preprocessed_data, features_data)
        
        # Return all results
        return {
            'preprocessed_data': preprocessed_data,
            'features_data': features_data,
            'similarity_matrix': similarity_df,
            'clustering_result': clustering_result
        } 