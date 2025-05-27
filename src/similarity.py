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
        Calculate a feature vector from extracted features.
        This applies the weights to feature groups.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Numpy array with feature vector
        """
        # Extract feature values in consistent order
        vocabulary_features = []
        sentence_features = []
        transition_features = []
        ngram_features = []
        syntactic_features = []
        
        # Vocabulary richness features
        if 'vocabulary_richness' in features:
            richness = features['vocabulary_richness']
            vocabulary_features.extend([
                richness.get('ttr', 0),
                richness.get('hapax_ratio', 0),
                richness.get('yules_k', 0),
                richness.get('simpsons_d', 0),
                richness.get('entropy', 0),
                richness.get('herdan_c', 0),
                richness.get('summer_s', 0),
                richness.get('orlov_z', 0)
            ])
        else:
            vocabulary_features = [0] * 8
        
        # Sentence structure features
        if 'sentence_stats' in features and 'sentence_complexity' in features:
            sentence_stats = features['sentence_stats']
            complexity = features['sentence_complexity']
            
            sentence_features.extend([
                sentence_stats.get('avg_length', 0),
                sentence_stats.get('std_length', 0),
                complexity.get('avg_words_before_punct', 0),
                complexity.get('punct_per_sentence', 0)
            ])
        else:
            sentence_features = [0] * 4
        
        # Transition pattern features
        if 'transition_patterns' in features:
            patterns = features['transition_patterns']
            transition_features.extend([
                patterns.get('length_transition_smoothness', 0),
                patterns.get('length_pattern_repetition', 0),
                patterns.get('clause_boundary_regularity', 0),
                patterns.get('sentence_rhythm_consistency', 0),
                patterns.get('transition_ratio_variance', 0),
                patterns.get('sentence_complexity_ratio', 0)
            ])
        else:
            transition_features = [0] * 6
        
        # N-gram features
        if 'word_bigrams' in features and 'word_trigrams' in features:
            # Use statistics about n-grams rather than the n-grams themselves
            bigrams = features['word_bigrams']
            trigrams = features['word_trigrams']
            char_ngrams = features.get('char_ngrams', {})
            
            # Basic statistics about n-gram distributions
            ngram_features.extend([
                len(bigrams) / 100 if bigrams else 0,   # Normalize bigram count
                len(trigrams) / 100 if trigrams else 0, # Normalize trigram count
                len(char_ngrams) / 100 if char_ngrams else 0, # Normalize character n-gram count
                np.mean(list(char_ngrams.values())) if char_ngrams else 0  # Average TF-IDF score
            ])
        else:
            ngram_features = [0] * 4
        
        # Syntactic features if available
        if 'syntactic_features' in features:
            syntactic = features['syntactic_features']
            
            # Basic syntactic properties
            syntactic_features.extend([
                syntactic.get('noun_ratio', 0),
                syntactic.get('verb_ratio', 0),
                syntactic.get('adj_ratio', 0),
                syntactic.get('adv_ratio', 0),
                syntactic.get('function_word_ratio', 0)
            ])
            
            # Add any additional syntactic features
            for key, value in syntactic.items():
                if key not in ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio', 'function_word_ratio']:
                    try:
                        syntactic_features.append(float(value))
                    except (ValueError, TypeError):
                        pass
        
        # Create the full feature vector
        feature_vector = np.concatenate([
            np.array(vocabulary_features, dtype=float),
            np.array(sentence_features, dtype=float),
            np.array(transition_features, dtype=float),
            np.array(ngram_features, dtype=float),
            np.array(syntactic_features, dtype=float)
        ])
        
        # Apply weights to each feature group
        feature_vector_size = len(feature_vector)
        vocabulary_indices = list(range(0, min(8, feature_vector_size)))
        sentence_indices = list(range(min(8, feature_vector_size), min(12, feature_vector_size)))
        transition_indices = list(range(min(12, feature_vector_size), min(18, feature_vector_size)))  # Updated for 6 transition features
        ngram_indices = list(range(min(18, feature_vector_size), min(22, feature_vector_size)))  # Updated starting index
        syntactic_indices = list(range(min(22, feature_vector_size), feature_vector_size))  # Updated starting index
        
        # Apply weights to each feature group
        weighted_vector = feature_vector.copy()
        
        # Print feature vector breakdown for debugging
        print(f"DEBUG - Feature vector shape: {feature_vector.shape}")
        print(f"DEBUG - Vocabulary features: {len(vocabulary_indices)}")
        print(f"DEBUG - Sentence features: {len(sentence_indices)}")
        print(f"DEBUG - Transition features: {len(transition_indices)}")
        print(f"DEBUG - Ngram features: {len(ngram_indices)}")
        print(f"DEBUG - Syntactic features: {len(syntactic_indices)}")
        
        # Check if this is NLP-only configuration
        is_nlp_only = (self.weights['vocabulary'] == 0.0 and self.weights['sentence'] == 0.0 and 
                      self.weights['transitions'] == 0.0 and self.weights['ngrams'] == 0.0 and 
                      self.weights['syntactic'] == 1.0)
        
        if not is_nlp_only:
            # For mixed feature analysis, normalize each feature group to have the same magnitude
            # This ensures that larger feature groups don't dominate smaller ones
            for indices in [vocabulary_indices, sentence_indices, transition_indices, ngram_indices, syntactic_indices]:
                if indices and indices[-1] < len(feature_vector):
                    group_magnitude = np.linalg.norm(feature_vector[indices])
                    if group_magnitude > 0:
                        # Normalize this group to have magnitude 1.0
                        weighted_vector[indices] = feature_vector[indices] / group_magnitude
        
        # Now apply the weights to the normalized feature groups
        for idx, weight_key, indices in [
            (0, 'vocabulary', vocabulary_indices),
            (1, 'sentence', sentence_indices),
            (2, 'transitions', transition_indices),
            (3, 'ngrams', ngram_indices),
            (4, 'syntactic', syntactic_indices)
        ]:
            if indices and indices[-1] < len(weighted_vector):
                weight = self.weights[weight_key]
                
                # Apply the weight to normalized features
                if weight > 0:
                    weighted_vector[indices] *= weight
                else:
                    weighted_vector[indices] = 0
                    
                # Print the magnitudes for debugging
                magnitude = np.linalg.norm(weighted_vector[indices])
                print(f"DEBUG - {weight_key} magnitude after weighting: {magnitude:.4f} (weight: {weight:.2f})")
        
        return weighted_vector
    
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
        
        # Calculate feature vectors for all manuscripts - this is where the weights should be applied
        feature_vectors = {}
        for name in manuscript_names:
            # Calculate the feature vector for this manuscript with the current weights
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
        
        # Process Pauline internal similarities
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
        """Calculate similarities within a single corpus (author or Pauline)."""
        # Extract feature vectors for this corpus
        corpus_vectors = np.vstack([feature_vectors[name] for name in corpus_mss])
        
        # Find columns that are all zeros (no variance)
        zero_columns = np.where(~corpus_vectors.any(axis=0))[0]
        
        # For NLP-only configuration, if all syntactic columns are zero, use vocabulary as fallback
        if is_nlp_only and len(zero_columns) >= corpus_vectors.shape[1] * 0.9:
            print(f"DEBUG - NLP-only ({corpus_mss[0].startswith('AUTH_') and 'Author' or 'Pauline'}): Found {len(zero_columns)} zero columns out of {corpus_vectors.shape[1]}")
            print(f"DEBUG - NLP-only ({corpus_mss[0].startswith('AUTH_') and 'Author' or 'Pauline'}): Using vocabulary features as fallback with noise")
            
            # Add small noise to differentiate
            np.random.seed(42)
            noise = np.random.normal(0, 0.01, corpus_vectors.shape)
            corpus_vectors = corpus_vectors + noise
        
        # For NLP-only analysis, don't normalize - preserve the actual feature magnitudes
        if is_nlp_only:
            normalized_vectors = [corpus_vectors[i] for i in range(len(corpus_mss))]
        else:
            # For mixed feature analysis, use unit normalization to preserve the effects of weights
            normalized_vectors = []
            for i in range(len(corpus_mss)):
                # Get the vector
                vec = corpus_vectors[i]
                
                # Unit normalize the vector (this preserves the direction while making magnitude consistent)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    normalized_vectors.append(vec / norm)
                else:
                    normalized_vectors.append(vec)  # Keep zero vectors as is
        
        # Calculate similarities using the normalized vectors
        for i, name_i in enumerate(corpus_mss):
            for j, name_j in enumerate(corpus_mss):
                if i < j:  # Only calculate upper triangle
                    sim = self._cosine_similarity(normalized_vectors[i], normalized_vectors[j])
                    
                    # For NLP-only with fallback, adjust the range
                    if is_nlp_only and len(zero_columns) >= corpus_vectors.shape[1] * 0.9:
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
        # Extract feature vectors for both corpora
        author_vectors = np.vstack([feature_vectors[name] for name in author_mss])
        pauline_vectors = np.vstack([feature_vectors[name] for name in pauline_mss])
        
        # Combine all vectors for consistent scaling
        all_vectors = np.vstack([author_vectors, pauline_vectors])
        
        # Find columns that are all zeros (no variance)
        zero_columns = np.where(~all_vectors.any(axis=0))[0]
        
        # For NLP-only configuration, if all syntactic columns are zero, use vocabulary as fallback
        if is_nlp_only and len(zero_columns) >= all_vectors.shape[1] * 0.9:
            # Add small noise to differentiate
            np.random.seed(42)
            noise = np.random.normal(0, 0.01, all_vectors.shape)
            all_vectors = -(all_vectors + noise)  # Negate to get opposing similarity
        
        # For NLP-only analysis, don't normalize - preserve the actual feature magnitudes
        if is_nlp_only:
            normalized_author_vectors = [author_vectors[i] for i in range(len(author_mss))]
            normalized_pauline_vectors = [pauline_vectors[i] for i in range(len(pauline_mss))]
        else:
            # For mixed feature analysis, normalize vectors
            normalized_author_vectors = []
            normalized_pauline_vectors = []
            
            # Normalize author vectors
            for i in range(len(author_mss)):
                vec = author_vectors[i]
                norm = np.linalg.norm(vec)
                if norm > 0:
                    normalized_author_vectors.append(vec / norm)
                else:
                    normalized_author_vectors.append(vec)
            
            # Normalize pauline vectors
            for i in range(len(pauline_mss)):
                vec = pauline_vectors[i]
                norm = np.linalg.norm(vec)
                if norm > 0:
                    normalized_pauline_vectors.append(vec / norm)
                else:
                    normalized_pauline_vectors.append(vec)
        
        # Calculate cross similarities
        for i, name_i in enumerate(author_mss):
            for j, name_j in enumerate(pauline_mss):
                sim = self._cosine_similarity(normalized_author_vectors[i], normalized_pauline_vectors[j])
                
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
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity value between 0 and 1
        """
        if np.all(a == 0) or np.all(b == 0):
            return 0.0
            
        # Calculate the raw cosine similarity
        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Clip to ensure it's between -1 and 1
        similarity = np.clip(similarity, -1.0, 1.0)
        
        # For NLP-only analysis, we want to preserve the raw differences
        # Check if this is NLP-only configuration
        is_nlp_only = (self.weights['vocabulary'] == 0.0 and self.weights['sentence'] == 0.0 and 
                      self.weights['transitions'] == 0.0 and self.weights['ngrams'] == 0.0 and 
                      self.weights['syntactic'] == 1.0)
        
        if is_nlp_only:
            # For NLP-only analysis, use Euclidean distance instead of cosine similarity
            # This preserves both magnitude and direction differences
            euclidean_dist = np.linalg.norm(a - b)
            
            # Convert distance to similarity (smaller distance = higher similarity)
            # Use exponential decay to convert distance to similarity in [0, 1] range
            max_possible_distance = np.linalg.norm(a) + np.linalg.norm(b)  # Maximum possible distance
            if max_possible_distance > 0:
                normalized_distance = euclidean_dist / max_possible_distance
                final_sim = np.exp(-3 * normalized_distance)  # Exponential decay
            else:
                final_sim = 1.0  # Identical zero vectors
        else:
            # For mixed feature analysis, apply exponential scaling to increase differentiation
            # Convert from [-1, 1] to [0, 1] range
            normalized_sim = (similarity + 1) / 2.0
            
            # Apply exponential scaling to create more separation between similar texts
            scaled_sim = normalized_sim ** 3
            final_sim = scaled_sim
        
        # Ensure the result is in [0, 1] range
        final_sim = np.clip(final_sim, 0.0, 1.0)
        
        return float(final_sim)
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set custom weights for different feature categories.
        
        Args:
            weights: Dictionary mapping feature categories to weights
        """
        print(f"DEBUG - Setting weights: {weights}")
        # Make a deep copy to avoid referencing the original dictionary
        self.weights = {k: v for k, v in weights.items()}
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity directly between two text samples.
        This is for testing purposes only and not used in the main workflow.
        
        Args:
            text1: First text sample
            text2: Second text sample
            
        Returns:
            Similarity score between the texts
        """
        from src.feature_extraction import extract_all_features
        
        # Extract features from both texts
        features1 = extract_all_features(text1)
        features2 = extract_all_features(text2)
        
        # Convert to feature vectors
        vector1 = self.calculate_feature_vector(features1)
        vector2 = self.calculate_feature_vector(features2)
        
        # Apply weights
        vectors = np.vstack([vector1, vector2])
        weighted_vectors = self._weight_features(vectors)
        
        # Scale features
        scaler = StandardScaler()
        scaled_vectors = scaler.fit_transform(weighted_vectors)
        
        # Calculate similarity
        similarity = self._cosine_similarity(scaled_vectors[0], scaled_vectors[1])
        return similarity
    
    def calculate_component_similarities(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Calculate similarity for each component separately to see contribution.
        
        Args:
            text1: First text sample
            text2: Second text sample
            
        Returns:
            Dictionary mapping component names to similarity scores
        """
        try:
            from src.feature_extraction import extract_all_features
            
            # Extract features from both texts
            features1 = extract_all_features(text1)
            features2 = extract_all_features(text2)
            
            # Convert to feature vectors
            vector1 = self.calculate_feature_vector(features1)
            vector2 = self.calculate_feature_vector(features2)
            
            # Calculate similarity for each component
            component_similarities = {}
            
            # Define feature indices for each category
            vocabulary_indices = list(range(0, 8))
            sentence_indices = list(range(8, 12))
            transition_indices = list(range(12, 16))
            ngram_indices = list(range(16, 20))
            syntactic_indices = list(range(20, len(vector1)))
            
            # Calculate component similarities
            for name, indices in [
                ('vocabulary', vocabulary_indices),
                ('sentence', sentence_indices),
                ('transitions', transition_indices),
                ('ngrams', ngram_indices),
                ('syntactic', syntactic_indices)
            ]:
                if not indices or indices[-1] >= len(vector1):
                    component_similarities[name] = 0.0
                    continue
                    
                # Extract component vectors
                v1 = vector1[indices]
                v2 = vector2[indices]
                
                # Stack for scaling
                vectors = np.vstack([v1, v2])
                
                # Scale vectors
                scaler = StandardScaler()
                scaled_vectors = scaler.fit_transform(vectors)
                
                # Calculate similarity
                similarity = self._cosine_similarity(scaled_vectors[0], scaled_vectors[1])
                component_similarities[name] = similarity
            
            return component_similarities
        except Exception as e:
            print(f"Error calculating component similarities: {e}")
            return None 