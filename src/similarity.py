"""
Module for calculating similarity between Greek manuscripts.
"""

import math
from typing import Dict, List, Tuple, Set, Any, Optional

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityCalculator:
    """Calculate similarity between Greek manuscripts."""
    
    def __init__(self):
        """Initialize the similarity calculator."""
        pass
    
    def jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """
        Calculate Jaccard similarity between two sets.
        
        Args:
            set1: First set
            set2: Second set
            
        Returns:
            Jaccard similarity score
        """
        if not set1 and not set2:
            return 1.0
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def cosine_similarity_dictionaries(self, dict1: Dict[Any, float], dict2: Dict[Any, float]) -> float:
        """
        Calculate cosine similarity between two dictionaries.
        
        Args:
            dict1: First dictionary mapping items to frequencies
            dict2: Second dictionary mapping items to frequencies
            
        Returns:
            Cosine similarity score
        """
        if not dict1 or not dict2:
            return 0.0
            
        # Get all keys
        all_keys = set(dict1.keys()).union(dict2.keys())
        
        # Create vectors
        vec1 = np.array([dict1.get(key, 0.0) for key in all_keys])
        vec2 = np.array([dict2.get(key, 0.0) for key in all_keys])
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def calculate_vocabulary_similarity(self, features1: Dict, features2: Dict) -> Dict[str, float]:
        """
        Calculate vocabulary similarity between two texts.
        
        Args:
            features1: Features of first text
            features2: Features of second text
            
        Returns:
            Dictionary with vocabulary similarity scores
        """
        # Extract word frequencies
        word_freq1 = features1['word_frequencies']
        word_freq2 = features2['word_frequencies']
        
        # Calculate Jaccard similarity between vocabularies
        vocab1 = set(word_freq1.keys())
        vocab2 = set(word_freq2.keys())
        jaccard_score = self.jaccard_similarity(vocab1, vocab2)
        
        # Calculate cosine similarity between frequency distributions
        cosine_score = self.cosine_similarity_dictionaries(word_freq1, word_freq2)
        
        return {
            'vocabulary_jaccard': jaccard_score,
            'vocabulary_cosine': cosine_score
        }
    
    def calculate_ngram_similarity(self, features1: Dict, features2: Dict) -> Dict[str, float]:
        """
        Calculate n-gram similarity between two texts.
        
        Args:
            features1: Features of first text
            features2: Features of second text
            
        Returns:
            Dictionary with n-gram similarity scores
        """
        # Extract n-gram frequencies
        bigram_freq1 = features1['bigram_frequencies']
        bigram_freq2 = features2['bigram_frequencies']
        trigram_freq1 = features1['trigram_frequencies']
        trigram_freq2 = features2['trigram_frequencies']
        
        # Calculate Jaccard similarity between sets of n-grams
        bigram_jaccard = self.jaccard_similarity(set(bigram_freq1.keys()), set(bigram_freq2.keys()))
        trigram_jaccard = self.jaccard_similarity(set(trigram_freq1.keys()), set(trigram_freq2.keys()))
        
        # Calculate cosine similarity between n-gram frequency distributions
        bigram_cosine = self.cosine_similarity_dictionaries(bigram_freq1, bigram_freq2)
        trigram_cosine = self.cosine_similarity_dictionaries(trigram_freq1, trigram_freq2)
        
        return {
            'bigram_jaccard': bigram_jaccard,
            'bigram_cosine': bigram_cosine,
            'trigram_jaccard': trigram_jaccard,
            'trigram_cosine': trigram_cosine
        }
    
    def calculate_sentence_stats_similarity(self, features1: Dict, features2: Dict) -> Dict[str, float]:
        """
        Calculate similarity between sentence statistics.
        
        Args:
            features1: Features of first text
            features2: Features of second text
            
        Returns:
            Dictionary with sentence statistics similarity scores
        """
        # Extract sentence stats
        stats1 = features1['sentence_stats']
        stats2 = features2['sentence_stats']
        
        # Calculate stats
        mean_diff = abs(stats1['mean_sentence_length'] - stats2['mean_sentence_length'])
        median_diff = abs(stats1['median_sentence_length'] - stats2['median_sentence_length'])
        std_diff = abs(stats1['std_sentence_length'] - stats2['std_sentence_length'])
        
        # Normalize differences to similarity scores (0 to 1)
        # Using a simple exponential decay function: sim = exp(-diff)
        # This makes smaller differences result in higher similarity
        mean_sim = math.exp(-mean_diff / 5)  # Divide by 5 to make the decay less steep
        median_sim = math.exp(-median_diff / 5)
        std_sim = math.exp(-std_diff / 3)
        
        return {
            'sentence_length_mean_sim': mean_sim,
            'sentence_length_median_sim': median_sim,
            'sentence_length_std_sim': std_sim
        }
    
    def calculate_vocabulary_richness_similarity(self, features1: Dict, features2: Dict) -> Dict[str, float]:
        """
        Calculate similarity between vocabulary richness metrics.
        
        Args:
            features1: Features of first text
            features2: Features of second text
            
        Returns:
            Dictionary with vocabulary richness similarity scores
        """
        # Extract vocabulary richness metrics
        vr1 = features1['vocabulary_richness']
        vr2 = features2['vocabulary_richness']
        
        # Calculate differences
        unique_ratio_diff = abs(vr1['unique_tokens_ratio'] - vr2['unique_tokens_ratio'])
        hapax_ratio_diff = abs(vr1['hapax_legomena_ratio'] - vr2['hapax_legomena_ratio'])
        yule_k_diff = abs(vr1['yule_k'] - vr2['yule_k'])
        
        # Normalize differences to similarity scores (0 to 1)
        unique_ratio_sim = math.exp(-unique_ratio_diff * 5)  # Higher weight as this is in [0,1]
        hapax_ratio_sim = math.exp(-hapax_ratio_diff * 5)    # Higher weight as this is in [0,1]
        yule_k_sim = math.exp(-yule_k_diff / 100)            # Lower weight as Yule's K can be larger
        
        return {
            'unique_ratio_sim': unique_ratio_sim,
            'hapax_ratio_sim': hapax_ratio_sim,
            'yule_k_sim': yule_k_sim
        }
    
    def calculate_word_position_similarity(self, features1: Dict, features2: Dict) -> Dict[str, float]:
        """
        Calculate similarity between word positions.
        
        Args:
            features1: Features of first text
            features2: Features of second text
            
        Returns:
            Dictionary with word position similarity scores
        """
        # Extract word positions
        positions1 = features1['word_positions']['sentence_positions']
        positions2 = features2['word_positions']['sentence_positions']
        
        # Find common words
        common_words = set(positions1.keys()).intersection(positions2.keys())
        
        if not common_words:
            return {'word_position_correlation': 0.0}
            
        # Create position vectors for common words
        pos_vec1 = [positions1[word] for word in common_words]
        pos_vec2 = [positions2[word] for word in common_words]
        
        # Calculate correlation
        if len(common_words) < 2:
            correlation = 0.0
        else:
            try:
                correlation, _ = pearsonr(pos_vec1, pos_vec2)
                if math.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0
                
        # Convert correlation to similarity (0 to 1)
        # Correlation is in [-1, 1], so we transform it to [0, 1]
        position_sim = (correlation + 1) / 2
        
        return {'word_position_correlation': position_sim}
    
    def calculate_overall_similarity(self, features1: Dict, features2: Dict) -> Dict[str, float]:
        """
        Calculate overall similarity between two texts based on all features.
        
        Args:
            features1: Features of first text
            features2: Features of second text
            
        Returns:
            Dictionary with similarity scores
        """
        # Calculate individual similarity scores
        vocabulary_sim = self.calculate_vocabulary_similarity(features1, features2)
        ngram_sim = self.calculate_ngram_similarity(features1, features2)
        sentence_stats_sim = self.calculate_sentence_stats_similarity(features1, features2)
        vocab_richness_sim = self.calculate_vocabulary_richness_similarity(features1, features2)
        word_position_sim = self.calculate_word_position_similarity(features1, features2)
        
        # Combine all similarity scores
        all_scores = {}
        all_scores.update(vocabulary_sim)
        all_scores.update(ngram_sim)
        all_scores.update(sentence_stats_sim)
        all_scores.update(vocab_richness_sim)
        all_scores.update(word_position_sim)
        
        # Calculate overall similarity as weighted average of all scores
        # These weights can be adjusted based on what's most important
        weights = {
            'vocabulary_jaccard': 1.0,
            'vocabulary_cosine': 1.5,
            'bigram_jaccard': 1.0,
            'bigram_cosine': 2.0,
            'trigram_jaccard': 1.0,
            'trigram_cosine': 2.5,
            'sentence_length_mean_sim': 0.5,
            'sentence_length_median_sim': 0.5,
            'sentence_length_std_sim': 0.5,
            'unique_ratio_sim': 1.0,
            'hapax_ratio_sim': 1.0,
            'yule_k_sim': 1.0,
            'word_position_correlation': 1.5
        }
        
        total_weight = sum(weights.values())
        weighted_sum = sum(weights.get(key, 1.0) * value for key, value in all_scores.items())
        
        overall_similarity = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Add overall similarity to results
        all_scores['overall_similarity'] = overall_similarity
        
        return all_scores 