"""
Module for NLP-based comparison of multiple Greek manuscripts.
Simplified version focusing on essential machine learning analysis.
"""

import os
from typing import List, Dict, Tuple, Optional, Any
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

from .preprocessing import GreekTextPreprocessor
from .features import FeatureExtractor
from .similarity import SimilarityCalculator
from .advanced_nlp import AdvancedGreekProcessor


class MultipleManuscriptComparison:
    """Compare multiple Greek manuscripts using NLP analysis."""
    
    def __init__(self, 
                 output_dir: str = "output", 
                 visualizations_dir: str = "visualizations"):
        """
        Initialize the comparison object.
        
        Args:
            output_dir: Directory to save output files
            visualizations_dir: Directory to save visualization files
        """
        self.output_dir = output_dir
        self.visualizations_dir = visualizations_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(visualizations_dir, exist_ok=True)
        
        # Initialize components
        self.preprocessor = GreekTextPreprocessor(
            remove_stopwords=False, 
            normalize_accents=True, 
            lowercase=True
        )
        self.feature_extractor = FeatureExtractor()
        self.similarity_calculator = SimilarityCalculator()
        
        # Advanced NLP processor
        try:
            self.advanced_processor = AdvancedGreekProcessor()
            self.preprocessor.advanced_processor = self.advanced_processor
            print("Successfully initialized advanced NLP processor")
        except ImportError as e:
            warnings.warn(f"Could not initialize advanced NLP processor: {e}")
            self.advanced_processor = None
        
        # For displaying book names
        self.display_names = {}
    
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
        Extract features from preprocessed documents.
        
        Args:
            preprocessed_data: Dictionary mapping names to preprocessed documents
            
        Returns:
            Dictionary mapping names to extracted features
        """
        features = {}
        
        # First pass - collect all texts for TF-IDF fitting
        all_texts = []
        for name, preprocessed in preprocessed_data.items():
            if 'words' in preprocessed:
                all_texts.append(' '.join(preprocessed['words']))
        
        # Fit the TF-IDF vectorizer on all texts
        if all_texts:
            self.feature_extractor.fit(all_texts)
        
        # Second pass - extract features for each document
        for name, preprocessed in preprocessed_data.items():
            try:
                doc_features = self.feature_extractor.extract_all_features(preprocessed)
                
                # Add advanced NLP features if available
                if self.advanced_processor and 'nlp_features' in preprocessed:
                    # Add syntactic features from POS tags
                    if 'pos_tags' in preprocessed['nlp_features']:
                        try:
                            syntactic_features = self.advanced_processor.extract_syntactic_features(
                                preprocessed['nlp_features']['pos_tags']
                            )
                            doc_features['syntactic_features'] = syntactic_features
                        except Exception as e:
                            print(f"Warning: Error extracting syntactic features for {name}: {e}")
                
                features[name] = doc_features
            except Exception as e:
                warnings.warn(f"Error extracting features for manuscript {name}: {e}")
                    
        return features
    
    def calculate_similarity_matrix(self, features: Dict[str, Dict]) -> pd.DataFrame:
        """
        Calculate similarity matrix between manuscripts based on their features.
        
        Args:
            features: Dictionary mapping manuscript IDs to their extracted features
            
        Returns:
            DataFrame containing pairwise similarities
        """
        return self.similarity_calculator.calculate_similarity_matrix(features)
    
    def cluster_manuscripts(self, 
                           similarity_df: pd.DataFrame, 
                           n_clusters: int = 3,
                           method: str = 'hierarchical') -> Dict[str, Any]:
        """
        Cluster manuscripts based on similarity.
        
        Args:
            similarity_df: DataFrame with similarity matrix
            n_clusters: Number of clusters to create
            method: Clustering method ('kmeans' or 'hierarchical')
            
        Returns:
            Dictionary with clustering results
        """
        # Convert similarity matrix to distance matrix (1 - similarity)
        distance_matrix = 1 - similarity_df.values
        distance_matrix = np.maximum(distance_matrix, 0)  # Ensure non-negative
        
        # Apply clustering
        if method == 'kmeans':
            # Use MDS to get coordinates for k-means
            mds = MDS(n_components=min(10, len(similarity_df)-1), 
                     dissimilarity='precomputed', random_state=42)
            coordinates = mds.fit_transform(distance_matrix)
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(coordinates)
        else:  # hierarchical
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters, 
                metric='precomputed', 
                linkage='average'
            )
            cluster_labels = clusterer.fit_predict(distance_matrix)
        
        # Dimensionality reduction for visualization
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        mds_coordinates = mds.fit_transform(distance_matrix)
        
        # t-SNE for alternative visualization
        n_samples = len(similarity_df)
        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=2, metric='precomputed', 
                   perplexity=perplexity, random_state=42)
        tsne_coordinates = tsne.fit_transform(distance_matrix)
        
        # Calculate silhouette score
        try:
            silhouette = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
        except:
            silhouette = 0.0
        
        return {
            'cluster_labels': cluster_labels,
            'manuscript_names': list(similarity_df.index),
            'mds_coordinates': mds_coordinates,
            'tsne_coordinates': tsne_coordinates,
            'silhouette_score': silhouette,
            'similarity_matrix': similarity_df,
            'n_clusters': n_clusters,
            'method': method
        }
    
    def generate_visualizations(self, clustering_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate visualization plots.
        
        Args:
            clustering_result: Results from clustering
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        visualization_files = {}
        
        # MDS plot
        mds_file = os.path.join(self.visualizations_dir, 'mds_clustering.png')
        self._plot_clustering(
            clustering_result['mds_coordinates'], 
            clustering_result['cluster_labels'],
            clustering_result['manuscript_names'],
            'MDS Clustering of Greek Manuscripts',
            mds_file
        )
        visualization_files['mds'] = mds_file
        
        # t-SNE plot
        tsne_file = os.path.join(self.visualizations_dir, 'tsne_clustering.png')
        self._plot_clustering(
            clustering_result['tsne_coordinates'], 
            clustering_result['cluster_labels'],
            clustering_result['manuscript_names'],
            't-SNE Clustering of Greek Manuscripts',
            tsne_file
        )
        visualization_files['tsne'] = tsne_file
        
        # Similarity heatmap
        heatmap_file = os.path.join(self.visualizations_dir, 'similarity_heatmap.png')
        self._plot_similarity_heatmap(
            clustering_result['similarity_matrix'],
            clustering_result['cluster_labels'],
            heatmap_file
        )
        visualization_files['heatmap'] = heatmap_file
        
        return visualization_files
    
    def _plot_clustering(self, coordinates, labels, names, title, filename):
        """Plot clustering results."""
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], 
                            c=labels, cmap='tab10', s=100, alpha=0.7)
        
        # Add labels for each point
        for i, name in enumerate(names):
            display_name = self.display_names.get(name, name)
            plt.annotate(display_name, (coordinates[i, 0], coordinates[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title(title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_similarity_heatmap(self, similarity_matrix, cluster_labels, filename):
        """Plot similarity matrix as heatmap."""
        plt.figure(figsize=(10, 8))
        
        # Sort by cluster labels for better visualization
        sorted_indices = np.argsort(cluster_labels)
        sorted_matrix = similarity_matrix.iloc[sorted_indices, sorted_indices]
        
        # Create heatmap
        sns.heatmap(sorted_matrix, annot=False, cmap='viridis', 
                   xticklabels=True, yticklabels=True)
        
        plt.title('Manuscript Similarity Matrix')
        plt.xlabel('Manuscripts')
        plt.ylabel('Manuscripts')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, 
                       clustering_result: Dict[str, Any],
                       features_data: Dict[str, Dict]) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            clustering_result: Results from clustering
            features_data: Features extracted from manuscripts
            
        Returns:
            Report text
        """
        report = []
        report.append("# Greek Manuscript NLP Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Basic statistics
        n_manuscripts = len(clustering_result['manuscript_names'])
        n_clusters = clustering_result['n_clusters']
        silhouette = clustering_result['silhouette_score']
        
        report.append(f"## Analysis Summary")
        report.append(f"- Number of manuscripts: {n_manuscripts}")
        report.append(f"- Number of clusters: {n_clusters}")
        report.append(f"- Clustering method: {clustering_result['method']}")
        report.append(f"- Silhouette score: {silhouette:.3f}")
        report.append("")
        
        # Cluster assignments
        report.append("## Cluster Assignments")
        cluster_dict = {}
        for i, (name, label) in enumerate(zip(clustering_result['manuscript_names'], 
                                            clustering_result['cluster_labels'])):
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(name)
        
        for cluster_id in sorted(cluster_dict.keys()):
            manuscripts = cluster_dict[cluster_id]
            report.append(f"**Cluster {cluster_id}** ({len(manuscripts)} manuscripts):")
            for manuscript in manuscripts:
                display_name = self.display_names.get(manuscript, manuscript)
                report.append(f"  - {display_name}")
            report.append("")
        
        # Feature importance
        if features_data:
            try:
                importance = self.similarity_calculator.get_feature_importance(features_data)
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                
                report.append("## Most Important Features")
                for feature, score in top_features:
                    report.append(f"- {feature}: {score:.4f}")
                report.append("")
            except Exception as e:
                report.append(f"## Feature Importance")
                report.append(f"Error calculating feature importance: {e}")
                report.append("")
        
        return "\n".join(report)
    
    def compare_multiple_manuscripts(self, 
                                   manuscripts: Dict[str, str], 
                                   display_names: Dict[str, str] = None,
                                   method: str = 'hierarchical',
                                   n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Complete workflow for comparing multiple manuscripts.
        
        Args:
            manuscripts: Dictionary mapping manuscript IDs to file paths
            display_names: Dictionary mapping manuscript IDs to display names
            method: Clustering method to use
            n_clusters: Number of clusters (auto-determined if None)
            
        Returns:
            Dictionary containing all analysis results
        """
        # Store display names
        if display_names:
            self.display_names = display_names
        
        # Extract manuscript paths and names
        manuscript_names = list(manuscripts.keys())
        manuscript_paths = list(manuscripts.values())
        
        print(f"Processing {len(manuscript_names)} manuscripts...")
        
        # Preprocess manuscripts
        preprocessed_data = self.preprocess_manuscripts(manuscript_paths, manuscript_names)
        print(f"Successfully preprocessed {len(preprocessed_data)} manuscripts")
        
        # Extract features
        print("Extracting features...")
        features_data = self.extract_features(preprocessed_data)
        print(f"Successfully extracted features from {len(features_data)} manuscripts")
        
        # Calculate similarity matrix
        print("Calculating similarity matrix...")
        similarity_df = self.calculate_similarity_matrix(features_data)
        print(f"Similarity matrix shape: {similarity_df.shape}")
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = min(6, max(2, len(manuscript_names) // 3))
        
        # Perform clustering
        print(f"Performing clustering with {n_clusters} clusters...")
        clustering_result = self.cluster_manuscripts(similarity_df, n_clusters, method)
        print(f"Clustering completed. Silhouette score: {clustering_result['silhouette_score']:.3f}")
        
        # Generate visualizations
        print("Generating visualizations...")
        visualization_files = self.generate_visualizations(clustering_result)
        print(f"Generated {len(visualization_files)} visualizations")
        
        # Generate report
        print("Generating report...")
        report = self.generate_report(clustering_result, features_data)
        
        # Save report
        report_file = os.path.join(self.output_dir, 'analysis_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save similarity matrix
        similarity_file = os.path.join(self.output_dir, 'similarity_matrix.csv')
        similarity_df.to_csv(similarity_file)
        
        print(f"Analysis complete. Results saved to {self.output_dir}")
        
        return {
            'preprocessing_data': preprocessed_data,
            'features_data': features_data,
            'similarity_matrix': similarity_df,
            'clustering_result': clustering_result,
            'visualization_files': visualization_files,
            'report': report,
            'report_file': report_file,
            'similarity_file': similarity_file
        } 