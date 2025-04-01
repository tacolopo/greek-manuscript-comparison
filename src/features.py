"""
Module for extracting linguistic features from Greek texts.
"""

import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional

import numpy as np
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    """Extract linguistic features from Greek texts."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        pass
    
    def extract_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """
        Extract n-grams from a list of tokens.
        
        Args:
            tokens: List of tokens
            n: Size of n-grams
            
        Returns:
            List of n-grams
        """
        return list(ngrams(tokens, n))
    
    def calculate_frequency_distribution(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate normalized frequency distribution of tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary mapping tokens to their normalized frequencies
        """
        if not tokens:
            return {}
            
        counter = Counter(tokens)
        total = len(tokens)
        
        # Normalize frequencies
        return {token: count / total for token, count in counter.items()}
    
    def calculate_ngram_frequency(self, tokens: List[str], n: int = 2) -> Dict[Tuple[str, ...], float]:
        """
        Calculate normalized frequency distribution of n-grams.
        
        Args:
            tokens: List of tokens
            n: Size of n-grams
            
        Returns:
            Dictionary mapping n-grams to their normalized frequencies
        """
        token_ngrams = self.extract_ngrams(tokens, n)
        
        if not token_ngrams:
            return {}
            
        counter = Counter(token_ngrams)
        total = len(token_ngrams)
        
        # Normalize frequencies
        return {ngram: count / total for ngram, count in counter.items()}
    
    def calculate_sentence_length_stats(self, tokenized_sentences: List[List[str]]) -> Dict[str, float]:
        """
        Calculate statistics about sentence lengths.
        
        Args:
            tokenized_sentences: List of tokenized sentences
            
        Returns:
            Dictionary with sentence length statistics
        """
        if not tokenized_sentences:
            return {
                'mean_sentence_length': 0,
                'median_sentence_length': 0,
                'std_sentence_length': 0,
                'min_sentence_length': 0,
                'max_sentence_length': 0
            }
            
        sentence_lengths = [len(sentence) for sentence in tokenized_sentences]
        
        return {
            'mean_sentence_length': float(np.mean(sentence_lengths)),
            'median_sentence_length': float(np.median(sentence_lengths)),
            'std_sentence_length': float(np.std(sentence_lengths)),
            'min_sentence_length': float(min(sentence_lengths)),
            'max_sentence_length': float(max(sentence_lengths))
        }
    
    def calculate_vocabulary_richness(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate vocabulary richness metrics.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary with vocabulary richness metrics
        """
        if not tokens:
            return {
                'unique_tokens_ratio': 0,
                'hapax_legomena_ratio': 0,
                'yule_k': 0
            }
            
        # Number of tokens
        N = len(tokens)
        
        # Number of unique tokens (vocabulary size)
        V = len(set(tokens))
        
        # Frequency distribution
        freq_dist = Counter(tokens)
        
        # Number of hapax legomena (words occurring exactly once)
        V1 = sum(1 for freq in freq_dist.values() if freq == 1)
        
        # Calculate Yule's K (a measure of vocabulary richness)
        M1 = N
        M2 = sum(freq ** 2 for freq in freq_dist.values())
        
        # Avoid division by zero
        if N > 0 and M1 > 0:
            yule_k = 10000 * (M2 - M1) / (M1 ** 2)
        else:
            yule_k = 0
            
        return {
            'unique_tokens_ratio': V / N if N > 0 else 0,
            'hapax_legomena_ratio': V1 / N if N > 0 else 0,
            'yule_k': yule_k
        }
    
    def calculate_word_position_features(self, tokenized_sentences: List[List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate features related to word positions.
        
        Args:
            tokenized_sentences: List of tokenized sentences
            
        Returns:
            Dictionary with word position features
        """
        if not tokenized_sentences:
            return {'sentence_positions': {}}
            
        # Calculate word positions in sentences
        sentence_positions = defaultdict(list)
        
        for sentence in tokenized_sentences:
            sentence_length = len(sentence)
            if sentence_length == 0:
                continue
                
            for i, token in enumerate(sentence):
                # Calculate normalized position (0 to 1)
                norm_position = i / (sentence_length - 1) if sentence_length > 1 else 0.5
                sentence_positions[token].append(norm_position)
        
        # Calculate average position for each word
        avg_positions = {}
        for token, positions in sentence_positions.items():
            if positions:
                avg_positions[token] = sum(positions) / len(positions)
                
        return {
            'sentence_positions': avg_positions
        }
        
    def extract_all_features(self, preprocessed_text: Dict) -> Dict:
        """
        Extract all linguistic features from preprocessed text.
        
        Args:
            preprocessed_text: Dictionary with preprocessed text
            
        Returns:
            Dictionary with extracted features
        """
        words = preprocessed_text['words']
        tokenized_sentences = preprocessed_text['tokenized_sentences']
        
        # Extract various features
        features = {}
        
        # Word frequency distribution
        features['word_frequencies'] = self.calculate_frequency_distribution(words)
        
        # N-gram frequencies
        features['bigram_frequencies'] = self.calculate_ngram_frequency(words, n=2)
        features['trigram_frequencies'] = self.calculate_ngram_frequency(words, n=3)
        
        # Sentence length statistics
        features['sentence_stats'] = self.calculate_sentence_length_stats(tokenized_sentences)
        
        # Vocabulary richness
        features['vocabulary_richness'] = self.calculate_vocabulary_richness(words)
        
        # Word position features
        features['word_positions'] = self.calculate_word_position_features(tokenized_sentences)
        
        return features 