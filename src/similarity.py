"""
Module for calculating similarities between manuscripts.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler

class SimilarityCalculator:
    """Calculate similarities between manuscripts based on their features."""
    
    def __init__(self):
        """Initialize similarity calculator."""
        self.scaler = StandardScaler()
        
        # Feature weights for different aspects
        self.weights = {
            'vocabulary': 0.3,  # Vocabulary richness and distribution
            'sentence': 0.2,    # Sentence structure and length
            'transitions': 0.2,  # Writing flow and transitions
            'ngrams': 0.3       # Character and word patterns
        }
    
    def calculate_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert features dictionary to numeric vector.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Numpy array of numeric features
        """
        feature_vector = []
        
        # Vocabulary features - these are already relative/normalized
        vocab = features['vocabulary_richness']
        feature_vector.extend([
            vocab['unique_tokens_ratio'],
            vocab['hapax_legomena_ratio'],
            vocab['dis_legomena_ratio'],
            vocab['yule_k'],
            vocab['simpson_d'],
            vocab['herdan_c'],
            vocab['guiraud_r'],
            vocab['sichel_s']
        ])
        
        # Sentence features - normalize sentence length metrics
        sent_stats = features['sentence_stats']
        # We keep mean and median sentence length as they're stylistic choices
        # But we normalize the std_dev by the mean to get coefficient of variation
        # This makes the variation measure size-independent
        sentence_cv = sent_stats['std_sentence_length'] / (sent_stats['mean_sentence_length'] + 1e-10)
        feature_vector.extend([
            sent_stats['mean_sentence_length'],
            sent_stats['median_sentence_length'],
            sentence_cv,  # Coefficient of variation instead of std
            sent_stats['sentence_length_variance'] / (sent_stats['mean_sentence_length']**2 + 1e-10)  # Normalized variance
        ])
        
        # Transition features - already normalized
        transitions = features['transition_patterns']
        feature_vector.extend([
            transitions['length_transition_smoothness'],
            transitions['length_pattern_repetition'],
            transitions['clause_boundary_regularity'],
            transitions['sentence_rhythm_consistency']
        ])
        
        # N-gram features - using relative frequencies instead of counts
        ngram_features = []
        for ngram_dict in [features['word_bigrams'], features['word_trigrams']]:
            if ngram_dict and len(ngram_dict) > 0:
                # Get normalized distribution shape metrics
                values = list(ngram_dict.values())
                total = sum(values) + 1e-10
                normalized_values = [v/total for v in values]
                ngram_features.extend([
                    np.mean(normalized_values),
                    np.std(normalized_values) / (np.mean(normalized_values) + 1e-10)  # Coefficient of variation
                ])
            else:
                ngram_features.extend([0, 0])
        feature_vector.extend(ngram_features)
        
        return np.array(feature_vector)
    
    def calculate_similarity_matrix(self, features_data: Dict[str, Dict]) -> pd.DataFrame:
        """
        Calculate similarity matrix between all manuscripts.
        
        Args:
            features_data: Dictionary mapping manuscript names to their features
            
        Returns:
            DataFrame containing pairwise similarities
        """
        # Convert features to vectors
        manuscript_names = list(features_data.keys())
        feature_vectors = []
        
        for name in manuscript_names:
            vector = self.calculate_feature_vector(features_data[name])
            feature_vectors.append(vector)
        
        # Convert to array and normalize
        X = np.array(feature_vectors)
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate cosine similarity
        similarity_matrix = np.zeros((len(manuscript_names), len(manuscript_names)))
        
        for i in range(len(manuscript_names)):
            for j in range(len(manuscript_names)):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Calculate weighted similarity for each feature group
                    start_idx = 0
                    similarity = 0
                    
                    # Vocabulary features (first 8 features)
                    vocab_sim = self._cosine_similarity(
                        X_scaled[i, start_idx:start_idx+8],
                        X_scaled[j, start_idx:start_idx+8]
                    )
                    similarity += self.weights['vocabulary'] * vocab_sim
                    start_idx += 8
                    
                    # Sentence features (next 4 features)
                    sent_sim = self._cosine_similarity(
                        X_scaled[i, start_idx:start_idx+4],
                        X_scaled[j, start_idx:start_idx+4]
                    )
                    similarity += self.weights['sentence'] * sent_sim
                    start_idx += 4
                    
                    # Transition features (next 4 features)
                    trans_sim = self._cosine_similarity(
                        X_scaled[i, start_idx:start_idx+4],
                        X_scaled[j, start_idx:start_idx+4]
                    )
                    similarity += self.weights['transitions'] * trans_sim
                    start_idx += 4
                    
                    # N-gram features (remaining features)
                    ngram_sim = self._cosine_similarity(
                        X_scaled[i, start_idx:],
                        X_scaled[j, start_idx:]
                    )
                    similarity += self.weights['ngrams'] * ngram_sim
                    
                    similarity_matrix[i, j] = similarity
        
        return pd.DataFrame(
            similarity_matrix,
            index=manuscript_names,
            columns=manuscript_names
        )
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return np.dot(a, b) / (norm_a * norm_b) 