"""
Module for extracting NLP-focused linguistic features from Greek texts.
Simplified version focusing on essential features for machine learning analysis.
"""

import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    """Extract essential linguistic features from Greek texts for NLP analysis."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.tfidf = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),  # Character n-grams from size 3 to 5
            max_features=1000
        )
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """
        Fit the TF-IDF vectorizer on a corpus of texts.
        
        Args:
            texts: List of texts to fit on
        """
        self.tfidf.fit(texts)
        self.is_fitted = True
    
    def calculate_vocabulary_richness(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate essential vocabulary richness metrics.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary with vocabulary richness metrics
        """
        if not tokens:
            return {
                'unique_tokens_ratio': 0,
                'hapax_legomena_ratio': 0,
                'vocab_size': 0,
                'total_tokens': 0
            }
            
        # Number of tokens and vocabulary size
        N = len(tokens)
        V = len(set(tokens))
        
        # Frequency distribution
        freq_dist = Counter(tokens)
        
        # Hapax legomena (words occurring once)
        V1 = sum(1 for freq in freq_dist.values() if freq == 1)
            
        return {
            'unique_tokens_ratio': V / N if N > 0 else 0,
            'hapax_legomena_ratio': V1 / N if N > 0 else 0,
            'vocab_size': V,
            'total_tokens': N
        }
    
    def calculate_sentence_stats(self, tokenized_sentences: List[List[str]]) -> Dict[str, float]:
        """
        Calculate basic sentence length statistics.
        
        Args:
            tokenized_sentences: List of tokenized sentences
            
        Returns:
            Dictionary with sentence statistics
        """
        if not tokenized_sentences:
            return {
                'mean_sentence_length': 0,
                'std_sentence_length': 0,
                'num_sentences': 0
            }
            
        sentence_lengths = [len(sentence) for sentence in tokenized_sentences]
        
        return {
            'mean_sentence_length': float(np.mean(sentence_lengths)),
            'std_sentence_length': float(np.std(sentence_lengths)),
            'num_sentences': len(sentence_lengths)
        }
    
    def extract_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """
        Extract n-grams from a list of tokens.
        
        Args:
            tokens: List of tokens
            n: Size of n-grams
            
        Returns:
            List of n-grams
        """
        from nltk.util import ngrams
        return list(ngrams(tokens, n))
    
    def calculate_ngram_frequency(self, tokens: List[str], n: int = 2) -> Dict[Tuple[str, ...], float]:
        """
        Calculate normalized frequency distribution of n-grams.
        
        Args:
            tokens: List of tokens
            n: Size of n-grams
            
        Returns:
            Dictionary mapping n-grams to their normalized frequencies
        """
        if len(tokens) < n:
            return {}
            
        token_ngrams = self.extract_ngrams(tokens, n)
        
        if not token_ngrams:
            return {}
            
        counter = Counter(token_ngrams)
        total = len(token_ngrams)
        
        # Return only the most frequent n-grams to keep feature space manageable
        most_common = counter.most_common(50)  # Top 50 n-grams
        return {ngram: count / total for ngram, count in most_common}
    
    def get_tfidf_features(self, text: str) -> Dict[str, float]:
        """
        Get TF-IDF character n-gram features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of TF-IDF features
        """
        if not self.is_fitted:
            return {}
            
        tfidf_matrix = self.tfidf.transform([text])
        feature_names = self.tfidf.get_feature_names_out()
        
        # Convert to dictionary (only non-zero features)
        features = {}
        for i, feature_name in enumerate(feature_names):
            score = tfidf_matrix[0, i]
            if score > 0:
                features[f'tfidf_{feature_name}'] = score
        
        return features
    
    def extract_all_features(self, preprocessed_text: Dict) -> Dict:
        """
        Extract all essential features from preprocessed text.
        
        Args:
            preprocessed_text: Dictionary containing preprocessed text data
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # Extract vocabulary features
        if 'words' in preprocessed_text:
            words = preprocessed_text['words']
            features['vocabulary_richness'] = self.calculate_vocabulary_richness(words)
            
            # Extract n-gram features
            features['word_bigrams'] = self.calculate_ngram_frequency(words, n=2)
            features['word_trigrams'] = self.calculate_ngram_frequency(words, n=3)
        
        # Extract sentence features
        if 'sentences' in preprocessed_text:
            sentences = preprocessed_text['sentences']
            tokenized_sentences = [sentence.split() for sentence in sentences]
            features['sentence_stats'] = self.calculate_sentence_stats(tokenized_sentences)
        
        # Extract TF-IDF character n-gram features
        if 'normalized_text' in preprocessed_text and self.is_fitted:
            tfidf_features = self.get_tfidf_features(preprocessed_text['normalized_text'])
            features['char_ngrams'] = tfidf_features
        
        # Add NLP features if available
        if 'nlp_features' in preprocessed_text:
            features['nlp_features'] = preprocessed_text['nlp_features']
        
        return features 