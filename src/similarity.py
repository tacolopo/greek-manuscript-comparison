"""
Module for calculating NLP-based similarities between manuscripts.
Simplified version focusing on essential NLP features.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityCalculator:
    """Calculate similarities between manuscripts based on NLP features."""
    
    def __init__(self):
        """Initialize similarity calculator."""
        self.scaler = StandardScaler()
    
    def extract_nlp_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Extract NLP features from manuscript features.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Numpy array with NLP feature vector
        """
        feature_vector = []
        
        # Vocabulary richness features
        if 'vocabulary_richness' in features:
            vocab = features['vocabulary_richness']
            feature_vector.extend([
                vocab.get('unique_tokens_ratio', 0),
                vocab.get('hapax_legomena_ratio', 0),
                vocab.get('vocab_size', 0) / 1000.0,  # Normalize vocab size
                vocab.get('total_tokens', 0) / 10000.0  # Normalize token count
            ])
        else:
            feature_vector.extend([0, 0, 0, 0])
        
        # Sentence structure features
        if 'sentence_stats' in features:
            sent = features['sentence_stats']
            feature_vector.extend([
                sent.get('mean_sentence_length', 0) / 50.0,  # Normalize sentence length
                sent.get('std_sentence_length', 0) / 20.0,   # Normalize std dev
                sent.get('num_sentences', 0) / 1000.0        # Normalize sentence count
            ])
        else:
            feature_vector.extend([0, 0, 0])
        
        # Advanced NLP syntactic features (if available)
        if 'nlp_features' in features and 'syntactic_features' in features:
            syntactic = features['syntactic_features']
            feature_vector.extend([
                syntactic.get('noun_ratio', 0),
                syntactic.get('verb_ratio', 0),
                syntactic.get('adj_ratio', 0),
                syntactic.get('adv_ratio', 0),
                syntactic.get('function_word_ratio', 0),
                syntactic.get('pronoun_ratio', 0),
                syntactic.get('conjunction_ratio', 0),
                syntactic.get('tag_diversity', 0),
                syntactic.get('tag_entropy', 0),
                syntactic.get('noun_verb_ratio', 0)
            ])
        else:
            # If no advanced NLP features, use zeros
            feature_vector.extend([0] * 10)
        
        # Character n-gram features (simplified)
        if 'char_ngrams' in features:
            char_features = features['char_ngrams']
            # Use only the top character n-gram features
            top_features = sorted(char_features.items(), key=lambda x: x[1], reverse=True)[:20]
            for _, score in top_features:
                feature_vector.append(score)
            # Pad with zeros if we have fewer than 20 features
            while len(feature_vector) < 37:  # 4 + 3 + 10 + 20 = 37
                feature_vector.append(0)
        else:
            feature_vector.extend([0] * 20)
        
        return np.array(feature_vector, dtype=float)
    
    def calculate_similarity_matrix(self, features_data: Dict[str, Dict]) -> pd.DataFrame:
        """
        Calculate similarity matrix between all manuscripts.
        
        Args:
            features_data: Dictionary mapping manuscript names to their features
            
        Returns:
            DataFrame containing pairwise similarities
        """
        manuscript_names = list(features_data.keys())
        
        # Extract feature vectors for all manuscripts
        feature_vectors = []
        for name in manuscript_names:
            vector = self.extract_nlp_features(features_data[name])
            feature_vectors.append(vector)
        
        # Convert to numpy array and standardize
        feature_matrix = np.array(feature_vectors)
        
        # Handle case where all features are zero
        if np.all(feature_matrix == 0):
            print("Warning: All feature vectors are zero. Using identity matrix.")
            similarity_matrix = np.eye(len(manuscript_names))
        else:
            # Standardize features
            try:
                feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
            except ValueError as e:
                print(f"Warning: Scaling failed ({e}). Using unscaled features.")
                feature_matrix_scaled = feature_matrix
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(feature_matrix_scaled)
        
        # Convert to DataFrame
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=manuscript_names,
            columns=manuscript_names
        )
        
        return similarity_df
    
    def calculate_pairwise_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two manuscripts.
        
        Args:
            features1: Features of first manuscript
            features2: Features of second manuscript
            
        Returns:
            Similarity score between 0 and 1
        """
        # Extract feature vectors
        vector1 = self.extract_nlp_features(features1)
        vector2 = self.extract_nlp_features(features2)
        
        # Calculate cosine similarity
        if np.all(vector1 == 0) and np.all(vector2 == 0):
            return 1.0  # Both empty, consider similar
        elif np.all(vector1 == 0) or np.all(vector2 == 0):
            return 0.0  # One empty, not similar
        
        # Normalize vectors
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(vector1, vector2) / (norm1 * norm2)
        
        # Ensure similarity is between 0 and 1
        return max(0, min(1, similarity))
    
    def get_feature_importance(self, features_data: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate feature importance based on variance across manuscripts.
        
        Args:
            features_data: Dictionary mapping manuscript names to their features
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Feature names corresponding to the feature vector
        feature_names = [
            'unique_tokens_ratio', 'hapax_legomena_ratio', 'vocab_size_norm', 'total_tokens_norm',
            'mean_sentence_length_norm', 'std_sentence_length_norm', 'num_sentences_norm',
            'noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio', 'function_word_ratio',
            'pronoun_ratio', 'conjunction_ratio', 'tag_diversity', 'tag_entropy', 'noun_verb_ratio'
        ] + [f'char_ngram_{i}' for i in range(20)]
        
        # Extract all feature vectors
        feature_vectors = []
        for features in features_data.values():
            vector = self.extract_nlp_features(features)
            feature_vectors.append(vector)
        
        feature_matrix = np.array(feature_vectors)
        
        # Calculate variance for each feature
        feature_variances = np.var(feature_matrix, axis=0)
        
        # Create importance dictionary
        importance = {}
        for i, name in enumerate(feature_names):
            if i < len(feature_variances):
                importance[name] = float(feature_variances[i])
            else:
                importance[name] = 0.0
        
        return importance 