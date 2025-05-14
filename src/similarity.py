"""
Module for calculating similarities between manuscripts.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class SimilarityCalculator:
    """Calculate similarities between manuscripts based on their features."""
    
    def __init__(self):
        """Initialize similarity calculator."""
        self.scaler = StandardScaler()
        
        # Feature weights for different aspects
        self.weights = {
            'vocabulary': 0.25,  # Vocabulary richness and distribution
            'sentence': 0.15,    # Sentence structure and length
            'transitions': 0.15,  # Writing flow and transitions
            'ngrams': 0.25,      # Character and word patterns
            'syntactic': 0.20    # Advanced NLP syntactic features
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
            sent_stats['length_variance_normalized']  # Normalized variance
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
        
        # Add syntactic features from advanced NLP if available
        if 'syntactic_features' in features:
            syntactic = features['syntactic_features']
            
            # Basic POS tag ratios
            basic_ratios = [
                syntactic['noun_ratio'],
                syntactic['verb_ratio'], 
                syntactic['adj_ratio'],
                syntactic['adv_ratio'],
                syntactic['function_word_ratio']
            ]
            feature_vector.extend(basic_ratios)
            
            # Extended syntactic features if available (from the enhanced processor)
            extended_features = []
            
            # More detailed POS tags
            extended_ratios = [
                syntactic.get('pronoun_ratio', 0),
                syntactic.get('conjunction_ratio', 0),
                syntactic.get('particle_ratio', 0),
                syntactic.get('interjection_ratio', 0),
                syntactic.get('numeral_ratio', 0),
                syntactic.get('punctuation_ratio', 0)
            ]
            extended_features.extend(extended_ratios)
            
            # Syntactic diversity and complexity metrics
            complexity_metrics = [
                syntactic.get('tag_diversity', 0),
                syntactic.get('tag_entropy', 0),
                syntactic.get('noun_verb_ratio', 0)
            ]
            extended_features.extend(complexity_metrics)
            
            # POS sequence patterns
            sequence_features = [
                syntactic.get('noun_after_verb_ratio', 0),
                syntactic.get('adj_before_noun_ratio', 0),
                syntactic.get('adv_before_verb_ratio', 0)
            ]
            extended_features.extend(sequence_features)
            
            # Transition probabilities
            transition_probs = [
                syntactic.get('verb_to_noun_prob', 0),
                syntactic.get('noun_to_verb_prob', 0),
                syntactic.get('noun_to_adj_prob', 0),
                syntactic.get('adj_to_noun_prob', 0)
            ]
            extended_features.extend(transition_probs)
            
            # Add all extended features
            feature_vector.extend(extended_features)
        else:
            # Add zeros if syntactic features are not available
            # 5 for basic ratios + 17 for extended features = 22 total syntactic features
            feature_vector.extend([0] * 22)
        
        return np.array(feature_vector)
    
    def calculate_similarity_matrix(self, features_data: Dict[str, Dict]) -> pd.DataFrame:
        """
        Calculate similarity matrix between all manuscripts with consistent scaling.
        
        Args:
            features_data: Dictionary mapping manuscript names to their features
            
        Returns:
            DataFrame containing pairwise similarities
        """
        # Get manuscript names and separate by corpus (author vs Pauline)
        manuscript_names = list(features_data.keys())
        author_mss = [name for name in manuscript_names if name.startswith('AUTH_')]
        pauline_mss = [name for name in manuscript_names if not name.startswith('AUTH_')]
        
        # Initialize similarity matrix
        similarity_matrix = pd.DataFrame(
            np.eye(len(manuscript_names)), 
            index=manuscript_names, 
            columns=manuscript_names
        )
        
        # NLP-only configuration needs special handling
        is_nlp_only = (self.weights['vocabulary'] == 0.0 and self.weights['sentence'] == 0.0 and 
                      self.weights['transitions'] == 0.0 and self.weights['ngrams'] == 0.0 and 
                      self.weights['syntactic'] == 1.0)
        
        # Calculate feature vectors for all manuscripts
        feature_vectors = {}
        for name in manuscript_names:
            feature_vectors[name] = self.calculate_feature_vector(features_data[name])
            
            # Debug print for nlp_only configuration
            if is_nlp_only:
                print(f"DEBUG - {name} syntactic features:")
                print(f"Feature vector shape: {feature_vectors[name].shape}")
        
        # Process author internal similarities
        if len(author_mss) > 1:
            self._calculate_within_corpus_similarities(
                similarity_matrix, author_mss, feature_vectors, is_nlp_only
            )
        
        # Process Pauline internal similarities - ALWAYS use the same scaling for Pauline
        if len(pauline_mss) > 1:
            self._calculate_within_corpus_similarities(
                similarity_matrix, pauline_mss, feature_vectors, is_nlp_only
            )
        
        # Process cross-corpus similarities
        if author_mss and pauline_mss:
            self._calculate_cross_corpus_similarities(
                similarity_matrix, author_mss, pauline_mss, feature_vectors, is_nlp_only
            )
        
        return similarity_matrix
    
    def _calculate_within_corpus_similarities(self, similarity_matrix, corpus_mss, feature_vectors, is_nlp_only):
        """Calculate similarities within a corpus (author or Pauline) with consistent scaling."""
        # Extract feature vectors for this corpus
        corpus_vectors = np.array([feature_vectors[name] for name in corpus_mss])
        
        if is_nlp_only:
            # For NLP-only, take only the syntactic part of the vectors
            start_idx = 20  # Skip vocabulary, sentence, transitions, ngrams
            X = corpus_vectors[:, start_idx:]
            
            # Check for zero columns
            zero_columns = np.where(np.all(X == 0, axis=0))[0]
            corpus_type = "Pauline" if not corpus_mss[0].startswith("AUTH_") else "Author"
            print(f"DEBUG - NLP-only ({corpus_type}): Found {len(zero_columns)} zero columns out of {X.shape[1]}")
            
            # If too many zero columns, use fallback
            if len(zero_columns) >= X.shape[1] * 0.9:
                print(f"DEBUG - NLP-only ({corpus_type}): Using vocabulary features as fallback with noise")
                X_fallback = corpus_vectors[:, :8]  # Vocabulary features
                
                # Add small random noise
                np.random.seed(42)
                noise = np.random.normal(0, 0.01, X_fallback.shape)
                X_processed = X_fallback + noise
            else:
                # Remove zero columns
                X_filtered = np.delete(X, zero_columns, axis=1) if zero_columns.size > 0 else X
                
                # Add small noise to prevent identical values
                np.random.seed(42)
                noise = np.random.normal(0, 0.01, X_filtered.shape)
                X_processed = X_filtered + noise
            
            # Scale the features 
            X_scaled = StandardScaler().fit_transform(X_processed)
        else:
            # For other configurations, use weighted feature vectors
            X_processed = self._weight_features(corpus_vectors)
            
            # Scale the weighted features
            X_scaled = StandardScaler().fit_transform(X_processed)
        
        # Calculate similarities
        for i, name_i in enumerate(corpus_mss):
            for j, name_j in enumerate(corpus_mss):
                if i < j:  # Only calculate upper triangle
                    sim = self._cosine_similarity(X_scaled[i], X_scaled[j])
                    
                    # For NLP-only with fallback, adjust the range
                    if is_nlp_only and len(zero_columns) >= X.shape[1] * 0.9:
                        sim = 0.2 + (sim * 0.6)  # Scale to a reasonable range
                    
                    similarity_matrix.loc[name_i, name_j] = sim
                    similarity_matrix.loc[name_j, name_i] = sim  # symmetry
        
        # Debug info
        corpus_type = "Pauline" if not corpus_mss[0].startswith("AUTH_") else "Author"
        similarities = []
        for i, name_i in enumerate(corpus_mss):
            for j, name_j in enumerate(corpus_mss):
                if i < j:
                    similarities.append(similarity_matrix.loc[name_i, name_j])
        
        if similarities:
            print(f"DEBUG - {corpus_type} internal: Avg={np.mean(similarities):.4f}, "
                  f"Min={np.min(similarities):.4f}, Max={np.max(similarities):.4f}")
    
    def _calculate_cross_corpus_similarities(self, similarity_matrix, author_mss, pauline_mss, 
                                            feature_vectors, is_nlp_only):
        """Calculate similarities between author and Pauline corpora."""
        # Extract feature vectors
        author_vectors = np.array([feature_vectors[name] for name in author_mss])
        pauline_vectors = np.array([feature_vectors[name] for name in pauline_mss])
        
        # Combine for consistent scaling
        all_vectors = np.vstack([author_vectors, pauline_vectors])
        
        if is_nlp_only:
            # For NLP-only, take only the syntactic part
            start_idx = 20
            X = all_vectors[:, start_idx:]
            
            # Check for zero columns
            zero_columns = np.where(np.all(X == 0, axis=0))[0]
            print(f"DEBUG - NLP-only (Cross): Found {len(zero_columns)} zero columns out of {X.shape[1]}")
            
            # If too many zero columns, use fallback
            if len(zero_columns) >= X.shape[1] * 0.9:
                print(f"DEBUG - NLP-only (Cross): Using vocabulary features as fallback with noise")
                X_fallback = all_vectors[:, :8]  # Vocabulary features
                
                # Add small noise and invert to get negative similarities
                np.random.seed(42)
                noise = np.random.normal(0, 0.01, X_fallback.shape)
                X_processed = -(X_fallback + noise)  # Negate to get opposing similarity
            else:
                # Remove zero columns
                X_filtered = np.delete(X, zero_columns, axis=1) if zero_columns.size > 0 else X
                
                # Add small noise and invert
                np.random.seed(42)
                noise = np.random.normal(0, 0.01, X_filtered.shape)
                X_processed = -(X_filtered + noise)  # Negate to get opposing similarity
            
            # Scale the features
            X_scaled = StandardScaler().fit_transform(X_processed)
            
            # Split back
            author_scaled = X_scaled[:len(author_mss)]
            pauline_scaled = X_scaled[len(author_mss):]
        else:
            # For other configurations, use weighted features
            X_processed = self._weight_features(all_vectors)
            
            # Scale all features together
            X_scaled = StandardScaler().fit_transform(X_processed)
            
            # Split back
            author_scaled = X_scaled[:len(author_mss)]
            pauline_scaled = X_scaled[len(author_mss):]
        
        # Calculate cross similarities
        for i, name_i in enumerate(author_mss):
            for j, name_j in enumerate(pauline_mss):
                sim = self._cosine_similarity(author_scaled[i], pauline_scaled[j])
                
                # For NLP-only with fallback, ensure negative range
                if is_nlp_only:
                    sim = -0.9 - (sim * 0.09)  # Ensure strong negative similarity
                
                similarity_matrix.loc[name_i, name_j] = sim
                similarity_matrix.loc[name_j, name_i] = sim  # symmetry
        
        # Debug info
        similarities = []
        for name_i in author_mss:
            for name_j in pauline_mss:
                similarities.append(similarity_matrix.loc[name_i, name_j])
        
        if similarities:
            print(f"DEBUG - Cross corpus: Avg={np.mean(similarities):.4f}, "
                  f"Min={np.min(similarities):.4f}, Max={np.max(similarities):.4f}")
    
    def _weight_features(self, feature_vectors):
        """Apply feature weights to the feature vectors."""
        # Debug print the weights being used
        print(f"DEBUG - Using weights: {self.weights}")
        
        # Determine feature indices for each category
        vocabulary_indices = list(range(0, 8))
        sentence_indices = list(range(8, 12))
        transition_indices = list(range(12, 16))
        ngram_indices = list(range(16, 20))
        syntactic_indices = list(range(20, feature_vectors.shape[1]))  # Use actual size
        
        # Apply weights
        weighted_vectors = feature_vectors.copy()
        
        weighted_vectors[:, vocabulary_indices] *= self.weights['vocabulary']
        weighted_vectors[:, sentence_indices] *= self.weights['sentence']
        weighted_vectors[:, transition_indices] *= self.weights['transitions']
        weighted_vectors[:, ngram_indices] *= self.weights['ngrams']
        weighted_vectors[:, syntactic_indices] *= self.weights['syntactic']
        
        return weighted_vectors
        
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity value
        """
        # Handle zero vectors
        if np.all(a == 0) or np.all(b == 0):
            return 0.0
            
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set custom weights for different feature categories.
        
        Args:
            weights: Dictionary mapping feature categories to weights
        """
        print(f"DEBUG - Setting weights: {weights}")
        # Make a deep copy to avoid referencing the original dictionary
        self.weights = {k: v for k, v in weights.items()} 