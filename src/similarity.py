"""
Module for calculating enhanced NLP-based similarities between manuscripts.
Includes feature selection, dimensionality reduction, and multiple similarity metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import warnings

class SimilarityCalculator:
    """Calculate enhanced similarities between manuscripts for clustering."""
    
    def __init__(self, feature_selection_k: int = 100, pca_components: float = 0.95):
        """
        Initialize similarity calculator with feature selection and PCA.
        
        Args:
            feature_selection_k: Number of top features to select
            pca_components: Number of PCA components (if float, explained variance ratio)
        """
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_selector = SelectKBest(score_func=f_classif, k=feature_selection_k)
        self.variance_filter = VarianceThreshold(threshold=0.01)  # Remove low-variance features
        self.pca = PCA(n_components=pca_components, svd_solver='auto')
        
        self.is_fitted = False
        self.feature_names = []
        self.selected_features = []
        
    def extract_nlp_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Extract and flatten NLP features from manuscript features.
        
        Args:
            features: Dictionary of manuscript features
            
        Returns:
            Flattened feature vector
        """
        feature_vector = []
        feature_names = []
        
        # Extract vocabulary richness features
        if 'vocabulary_richness' in features:
            vocab_features = features['vocabulary_richness']
            for key, value in vocab_features.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    feature_vector.append(float(value))
                    feature_names.append(f'vocab_{key}')
        
        # Extract sentence complexity features
        if 'sentence_complexity' in features:
            sent_features = features['sentence_complexity']
            for key, value in sent_features.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    feature_vector.append(float(value))
                    feature_names.append(f'sent_{key}')
        
        # Extract function word features
        if 'function_words' in features:
            func_features = features['function_words']
            for key, value in func_features.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    feature_vector.append(float(value))
                    feature_names.append(f'func_{key}')
        
        # Extract morphological features
        if 'morphological' in features:
            morph_features = features['morphological']
            for key, value in morph_features.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    feature_vector.append(float(value))
                    feature_names.append(f'morph_{key}')
        
        # Extract semantic features
        if 'semantic' in features:
            sem_features = features['semantic']
            for key, value in sem_features.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    feature_vector.append(float(value))
                    feature_names.append(f'sem_{key}')
        
        # Extract punctuation features
        if 'punctuation' in features:
            punct_features = features['punctuation']
            for key, value in punct_features.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    feature_vector.append(float(value))
                    feature_names.append(f'punct_{key}')
        
        # Extract TF-IDF features (sample top features to avoid dimensionality explosion)
        if 'tfidf' in features:
            tfidf_features = features['tfidf']
            # Sort by value and take top features
            sorted_tfidf = sorted(tfidf_features.items(), key=lambda x: x[1], reverse=True)[:50]
            for key, value in sorted_tfidf:
                if isinstance(value, (int, float)) and not np.isnan(value):
                    feature_vector.append(float(value))
                    feature_names.append(f'tfidf_{key}')
        
        # Extract n-gram features (sample top features)
        for ngram_type in ['word_bigrams', 'word_trigrams']:
            if ngram_type in features:
                ngram_features = features[ngram_type]
                # Sort by frequency and take top features
                sorted_ngrams = sorted(ngram_features.items(), key=lambda x: x[1], reverse=True)[:20]
                for ngram, freq in sorted_ngrams:
                    if isinstance(freq, (int, float)) and not np.isnan(freq):
                        feature_vector.append(float(freq))
                        feature_names.append(f'{ngram_type}_{str(ngram)[:20]}')  # Truncate long n-gram names
        
        if not self.is_fitted:
            self.feature_names = feature_names
        
        return np.array(feature_vector)
    
    def fit_transform_features(self, feature_matrices: List[np.ndarray], manuscript_names: List[str]) -> np.ndarray:
        """
        Fit preprocessing pipeline and transform features.
        
        Args:
            feature_matrices: List of feature vectors for each manuscript
            manuscript_names: List of manuscript names
            
        Returns:
            Transformed feature matrix
        """
        # Stack feature matrices
        X = np.vstack(feature_matrices)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Original feature matrix shape: {X.shape}")
        
        # Remove low-variance features
        X_filtered = self.variance_filter.fit_transform(X)
        print(f"After variance filtering: {X_filtered.shape}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_filtered)
        
        # Apply PCA for dimensionality reduction
        X_pca = self.pca.fit_transform(X_scaled)
        print(f"After PCA: {X_pca.shape}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        self.is_fitted = True
        return X_pca
    
    def transform_features(self, feature_matrices: List[np.ndarray]) -> np.ndarray:
        """
        Transform new feature matrices using fitted pipeline.
        
        Args:
            feature_matrices: List of feature vectors
            
        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Must fit the pipeline first")
        
        X = np.vstack(feature_matrices)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_filtered = self.variance_filter.transform(X)
        X_scaled = self.scaler.transform(X_filtered)
        X_pca = self.pca.transform(X_scaled)
        
        return X_pca
    
    def calculate_similarity_matrix(self, feature_matrix: np.ndarray, 
                                   metric: str = 'cosine') -> np.ndarray:
        """
        Calculate similarity matrix using specified metric.
        
        Args:
            feature_matrix: Transformed feature matrix
            metric: Similarity metric ('cosine', 'euclidean', 'correlation')
            
        Returns:
            Similarity matrix
        """
        if metric == 'cosine':
            # Cosine similarity (higher = more similar)
            return cosine_similarity(feature_matrix)
        
        elif metric == 'euclidean':
            # Convert Euclidean distance to similarity (lower distance = higher similarity)
            distances = euclidean_distances(feature_matrix)
            # Normalize to [0, 1] similarity scale
            max_dist = np.max(distances)
            similarities = 1 - (distances / max_dist) if max_dist > 0 else np.ones_like(distances)
            return similarities
        
        elif metric == 'correlation':
            # Pearson correlation as similarity
            n = feature_matrix.shape[0]
            corr_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        try:
                            corr, _ = pearsonr(feature_matrix[i], feature_matrix[j])
                            corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                        except:
                            corr_matrix[i, j] = 0.0
            
            # Convert to [0, 1] scale (correlation is [-1, 1])
            return (corr_matrix + 1) / 2
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def calculate_multiple_similarities(self, feature_matrices: List[np.ndarray], 
                                      manuscript_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Calculate multiple similarity matrices for ensemble clustering.
        
        Args:
            feature_matrices: List of feature vectors
            manuscript_names: List of manuscript names
            
        Returns:
            Dictionary of similarity matrices for different metrics
        """
        # Transform features
        X_transformed = self.fit_transform_features(feature_matrices, manuscript_names)
        
        # Calculate similarities using different metrics
        similarities = {}
        
        for metric in ['cosine', 'euclidean', 'correlation']:
            try:
                sim_matrix = self.calculate_similarity_matrix(X_transformed, metric)
                similarities[metric] = sim_matrix
                print(f"{metric.capitalize()} similarity calculated")
            except Exception as e:
                print(f"Warning: Could not calculate {metric} similarity: {e}")
        
        return similarities
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from PCA components.
        
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_fitted:
            return {}
        
        # Calculate feature importance from PCA components
        # Use the sum of absolute loadings across all components
        feature_importance = {}
        
        if hasattr(self.pca, 'components_'):
            n_features = self.pca.components_.shape[1]
            importance_scores = np.sum(np.abs(self.pca.components_), axis=0)
            
            # Get feature names that survived variance filtering
            surviving_features = np.array(self.feature_names)[self.variance_filter.get_support()]
            
            if len(surviving_features) == len(importance_scores):
                for feature, score in zip(surviving_features, importance_scores):
                    feature_importance[feature] = float(score)
        
        return feature_importance
    
    def ensemble_similarity(self, similarities: Dict[str, np.ndarray], 
                           weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Create ensemble similarity matrix by combining multiple metrics.
        
        Args:
            similarities: Dictionary of similarity matrices
            weights: Optional weights for each similarity metric
            
        Returns:
            Ensemble similarity matrix
        """
        if not similarities:
            raise ValueError("No similarity matrices provided")
        
        if weights is None:
            # Equal weights
            weights = {metric: 1.0 for metric in similarities.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Combine similarities
        ensemble_sim = None
        
        for metric, sim_matrix in similarities.items():
            weight = weights.get(metric, 0.0)
            if weight > 0:
                if ensemble_sim is None:
                    ensemble_sim = weight * sim_matrix
                else:
                    ensemble_sim += weight * sim_matrix
        
        return ensemble_sim if ensemble_sim is not None else np.eye(len(similarities[list(similarities.keys())[0]]))
    
    def get_distance_matrix(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Convert similarity matrix to distance matrix for clustering.
        
        Args:
            similarity_matrix: Similarity matrix
            
        Returns:
            Distance matrix
        """
        # Convert similarity to distance: distance = 1 - similarity
        # Ensure similarity is in [0, 1] range
        sim_normalized = np.clip(similarity_matrix, 0, 1)
        distance_matrix = 1 - sim_normalized
        
        # Ensure diagonal is zero (distance from item to itself)
        np.fill_diagonal(distance_matrix, 0)
        
        return distance_matrix 