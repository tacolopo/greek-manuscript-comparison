"""
Module for enhanced NLP-based comparison of multiple Greek manuscripts.
Includes multiple clustering algorithms, validation metrics, and ensemble methods.
"""

import os
from typing import List, Dict, Tuple, Optional, Any
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

from .preprocessing import GreekTextPreprocessor
from .features import FeatureExtractor
from .similarity import SimilarityCalculator
from .advanced_nlp import AdvancedGreekProcessor


class MultipleManuscriptComparison:
    """Enhanced comparison of multiple Greek manuscripts using ML clustering."""
    
    def __init__(self, use_advanced_nlp: bool = True):
        """
        Initialize the manuscript comparison system.
        
        Args:
            use_advanced_nlp: Whether to use advanced NLP features
        """
        self.preprocessor = GreekTextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.similarity_calculator = SimilarityCalculator()
        self.use_advanced_nlp = use_advanced_nlp
        
        if use_advanced_nlp:
            try:
                self.advanced_processor = AdvancedGreekProcessor()
            except Exception as e:
                print(f"Warning: Could not initialize advanced NLP processor: {e}")
                self.advanced_processor = None
                self.use_advanced_nlp = False
        else:
            self.advanced_processor = None
        
        # Store results
        self.manuscript_features = {}
        self.feature_matrices = []
        self.manuscript_names = []
        self.similarity_matrices = {}
        self.clustering_results = {}
    
    def preprocess_manuscripts(self, 
                              manuscript_paths: List[str], 
                              manuscript_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Preprocess multiple manuscripts and extract features."""
        if manuscript_names is None:
            manuscript_names = [os.path.basename(path) for path in manuscript_paths]
        
        print(f"Processing {len(manuscript_paths)} manuscripts...")
        processed_manuscripts = {}
        
        for path, name in tqdm(zip(manuscript_paths, manuscript_names), 
                              desc="Preprocessing manuscripts"):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                preprocessed = self.preprocessor.preprocess(text)
                
                if self.use_advanced_nlp and self.advanced_processor:
                    try:
                        nlp_features = self.advanced_processor.process_document(
                            preprocessed['normalized_text']
                        )
                        preprocessed['nlp_features'] = nlp_features
                    except Exception as e:
                        print(f"Warning: Advanced NLP processing failed for {name}: {e}")
                        preprocessed['nlp_features'] = {}
                else:
                    preprocessed['nlp_features'] = {}
                
                processed_manuscripts[name] = preprocessed
                
            except Exception as e:
                print(f"Error processing {name}: {e}")
                continue
        
        return processed_manuscripts
    
    def extract_features(self, processed_manuscripts: Dict[str, Dict]) -> Dict[str, Dict]:
        """Extract comprehensive features from preprocessed manuscripts."""
        print("Extracting features...")
        
        all_texts = [ms['normalized_text'] for ms in processed_manuscripts.values()]
        self.feature_extractor.fit(all_texts)
        
        features_data = {}
        
        # First pass: extract all features
        for name, preprocessed in tqdm(processed_manuscripts.items(), desc="Extracting features"):
            try:
                features = self.feature_extractor.extract_all_features(
                    preprocessed, self.advanced_processor
                )
                features_data[name] = features
            except Exception as e:
                print(f"Error extracting features for {name}: {e}")
                continue
        
        # Build global vocabulary for consistent feature vectors
        self.similarity_calculator.build_global_vocabulary(list(features_data.values()))
        
        # Second pass: create consistent feature vectors
        feature_matrices = []
        manuscript_names = []
        
        for name, features in tqdm(features_data.items(), desc="Creating feature vectors"):
            try:
                feature_vector = self.similarity_calculator.extract_nlp_features(features)
                
                if len(feature_vector) > 0:
                    feature_matrices.append(feature_vector)
                    manuscript_names.append(name)
                    print(f"Feature vector for {name}: {len(feature_vector)} dimensions")
                else:
                    print(f"Warning: No features extracted for {name}")
                    
            except Exception as e:
                print(f"Error creating feature vector for {name}: {e}")
                continue
        
        self.manuscript_features = features_data
        self.feature_matrices = feature_matrices
        self.manuscript_names = manuscript_names
        
        return features_data
    
    def perform_clustering(self, n_clusters_range: Tuple[int, int] = (2, 8)) -> Dict:
        """Perform clustering with deterministic random states for reproducibility."""
        print("Performing clustering analysis...")
        
        X = self.similarity_calculator.fit_transform_features(
            self.feature_matrices, self.manuscript_names
        )
        
        clustering_results = {}
        min_clusters, max_clusters = n_clusters_range
        max_clusters = min(max_clusters, len(self.manuscript_names) - 1)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            print(f"Testing {n_clusters} clusters...")
            cluster_results = {}
            
            # K-Means with fixed random state
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(X)
                cluster_results['kmeans'] = {
                    'labels': kmeans_labels,
                    'algorithm': 'KMeans',
                    'silhouette': silhouette_score(X, kmeans_labels),
                    'calinski_harabasz': calinski_harabasz_score(X, kmeans_labels),
                    'davies_bouldin': davies_bouldin_score(X, kmeans_labels)
                }
            except Exception as e:
                print(f"KMeans failed: {e}")
            
            # Hierarchical clustering (deterministic)
            try:
                hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                hierarchical_labels = hierarchical.fit_predict(X)
                cluster_results['hierarchical'] = {
                    'labels': hierarchical_labels,
                    'algorithm': 'Hierarchical',
                    'silhouette': silhouette_score(X, hierarchical_labels),
                    'calinski_harabasz': calinski_harabasz_score(X, hierarchical_labels),
                    'davies_bouldin': davies_bouldin_score(X, hierarchical_labels)
                }
            except Exception as e:
                print(f"Hierarchical clustering failed: {e}")
            
            # Gaussian Mixture with fixed random state
            try:
                gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                gmm_labels = gmm.fit_predict(X)
                cluster_results['gaussian_mixture'] = {
                    'labels': gmm_labels,
                    'algorithm': 'GaussianMixture',
                    'silhouette': silhouette_score(X, gmm_labels),
                    'calinski_harabasz': calinski_harabasz_score(X, gmm_labels),
                    'davies_bouldin': davies_bouldin_score(X, gmm_labels)
                }
            except Exception as e:
                print(f"Gaussian Mixture Model failed: {e}")
            
            clustering_results[n_clusters] = cluster_results
        
        self.clustering_results = clustering_results
        return clustering_results
    
    def find_optimal_clustering(self) -> Dict:
        """Find optimal clustering based on validation metrics."""
        print("Finding optimal clustering...")
        
        all_results = []
        for n_clusters, algorithms in self.clustering_results.items():
            for alg_name, result in algorithms.items():
                if 'silhouette' in result:
                    all_results.append({
                        'n_clusters': n_clusters,
                        'algorithm': alg_name,
                        'silhouette': result['silhouette'],
                        'calinski_harabasz': result['calinski_harabasz'],
                        'davies_bouldin': result['davies_bouldin'],
                        'labels': result['labels']
                    })
        
        results_df = pd.DataFrame(all_results)
        best_idx = results_df['silhouette'].idxmax()
        best_result = results_df.iloc[best_idx]
        
        optimal_clustering = {
            'n_clusters': int(best_result['n_clusters']),
            'algorithm': best_result['algorithm'],
            'labels': best_result['labels'],
            'silhouette_score': best_result['silhouette'],
            'manuscript_clusters': dict(zip(self.manuscript_names, best_result['labels']))
        }
        
        print(f"Optimal: {optimal_clustering['algorithm']} with {optimal_clustering['n_clusters']} clusters")
        print(f"Silhouette score: {optimal_clustering['silhouette_score']:.3f}")
        
        return optimal_clustering
    
    def create_visualizations(self, optimal_clustering: Dict, output_dir: str) -> Dict[str, str]:
        """Create visualizations with deterministic random states."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"Creating visualizations in {output_dir}...")
        
        X = self.similarity_calculator.fit_transform_features(
            self.feature_matrices, self.manuscript_names
        )
        
        labels = optimal_clustering['labels']
        visualization_files = {}
        
        # MDS Plot with fixed random state
        try:
            mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
            X_mds = mds.fit_transform(X)
            
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(X_mds[:, 0], X_mds[:, 1], c=labels, cmap='tab10', s=100, alpha=0.7)
            
            for i, name in enumerate(self.manuscript_names):
                plt.annotate(name, (X_mds[i, 0], X_mds[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.title(f'MDS Clustering - {optimal_clustering["algorithm"]}\n'
                     f'{optimal_clustering["n_clusters"]} Clusters (Silhouette: {optimal_clustering["silhouette_score"]:.3f})')
            plt.xlabel('MDS Dimension 1')
            plt.ylabel('MDS Dimension 2')
            plt.colorbar(scatter, label='Cluster')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            mds_file = os.path.join(output_dir, 'mds_clustering.png')
            plt.savefig(mds_file, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_files['mds'] = mds_file
            
        except Exception as e:
            print(f"Warning: Could not create MDS plot: {e}")
        
        return visualization_files
    
    def generate_report(self, optimal_clustering: Dict, output_file: str) -> str:
        """Generate analysis report."""
        print(f"Generating report: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("GREEK MANUSCRIPT CLUSTERING ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("ANALYSIS SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total manuscripts analyzed: {len(self.manuscript_names)}\n")
            f.write(f"Optimal clustering algorithm: {optimal_clustering['algorithm']}\n")
            f.write(f"Optimal number of clusters: {optimal_clustering['n_clusters']}\n")
            f.write(f"Silhouette score: {optimal_clustering['silhouette_score']:.4f}\n\n")
            
            f.write("CLUSTER ASSIGNMENTS\n")
            f.write("-" * 20 + "\n")
            
            clusters = {}
            for manuscript, cluster_id in optimal_clustering['manuscript_clusters'].items():
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(manuscript)
            
            for cluster_id in sorted(clusters.keys()):
                f.write(f"Cluster {cluster_id}:\n")
                for manuscript in sorted(clusters[cluster_id]):
                    f.write(f"  - {manuscript}\n")
                f.write("\n")
            
            f.write("Analysis completed successfully.\n")
        
        return output_file

    def run_complete_analysis_from_texts(self,
                                       manuscript_texts: List[str],
                                       manuscript_names: Optional[List[str]] = None,
                                       output_dir: str = "clustering_analysis") -> Dict:
        """
        Run complete analysis pipeline from text content.
        
        Args:
            manuscript_texts: List of text content for each manuscript
            manuscript_names: Optional list of manuscript names
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing analysis results
        """
        print("Starting DETERMINISTIC NLP clustering analysis...")
        print("All random states are fixed for reproducible results!")
        
        if manuscript_names is None:
            manuscript_names = [f"Manuscript_{i+1}" for i in range(len(manuscript_texts))]
        
        print(f"Processing {len(manuscript_texts)} manuscripts...")
        
        # Preprocess texts directly
        processed_manuscripts = {}
        for text, name in tqdm(zip(manuscript_texts, manuscript_names), 
                              desc="Preprocessing manuscripts", total=len(manuscript_texts)):
            try:
                preprocessed = self.preprocessor.preprocess(text)
                
                if self.use_advanced_nlp and self.advanced_processor:
                    try:
                        nlp_features = self.advanced_processor.process_document(
                            preprocessed['normalized_text']
                        )
                        preprocessed['nlp_features'] = nlp_features
                    except Exception as e:
                        print(f"Warning: Advanced NLP processing failed for {name}: {e}")
                        preprocessed['nlp_features'] = {}
                else:
                    preprocessed['nlp_features'] = {}
                
                processed_manuscripts[name] = preprocessed
                
            except Exception as e:
                print(f"Error processing {name}: {e}")
                continue
        
        # Continue with the rest of the analysis pipeline
        features_data = self.extract_features(processed_manuscripts)
        clustering_results = self.perform_clustering()
        optimal_clustering = self.find_optimal_clustering()
        visualizations = self.create_visualizations(optimal_clustering, output_dir)
        report_file = self.generate_report(optimal_clustering, 
                                         os.path.join(output_dir, "clustering_analysis_report.txt"))
        
        print(f"\nDETERMINISTIC analysis complete! Results saved to: {output_dir}")
        print(f"Report: {report_file}")
        
        return {
            'processed_manuscripts': processed_manuscripts,
            'features': features_data,
            'clustering_results': clustering_results,
            'optimal_clustering': optimal_clustering,
            'visualizations': visualizations,
            'report_file': report_file
        }

    def run_complete_analysis(self, 
                            manuscript_paths: List[str],
                            manuscript_names: Optional[List[str]] = None,
                            output_dir: str = "clustering_analysis") -> Dict:
        """
        Run the complete deterministic clustering analysis pipeline.
        
        Args:
            manuscript_paths: List of paths to manuscript files
            manuscript_names: Optional list of manuscript names
            output_dir: Directory to save results
            
        Returns:
            Dictionary with all analysis results
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"Starting DETERMINISTIC NLP clustering analysis...")
        print("All random states are fixed for reproducible results!")
        
        # Step 1: Preprocess manuscripts
        processed_manuscripts = self.preprocess_manuscripts(manuscript_paths, manuscript_names)
        
        # Step 2: Extract features
        features_data = self.extract_features(processed_manuscripts)
        
        # Step 3: Calculate similarities (deterministic)
        similarities = self.similarity_calculator.calculate_multiple_similarities(
            self.feature_matrices, self.manuscript_names
        )
        
        # Step 4: Perform clustering (with fixed random states)
        clustering_results = self.perform_clustering()
        
        # Step 5: Find optimal clustering
        optimal_clustering = self.find_optimal_clustering()
        
        # Step 6: Create visualizations (with fixed random states)
        visualization_files = self.create_visualizations(optimal_clustering, output_dir)
        
        # Step 7: Generate report
        report_file = os.path.join(output_dir, "clustering_analysis_report.txt")
        self.generate_report(optimal_clustering, report_file)
        
        results = {
            'processed_manuscripts': processed_manuscripts,
            'features_data': features_data,
            'similarity_matrices': similarities,
            'clustering_results': clustering_results,
            'optimal_clustering': optimal_clustering,
            'visualization_files': visualization_files,
            'report_file': report_file,
            'output_directory': output_dir
        }
        
        print(f"\nDETERMINISTIC analysis complete! Results saved to: {output_dir}")
        print(f"Report: {report_file}")
        
        return results
