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
                'max_sentence_length': 0,
                'sentence_length_variance': 0,
                'length_variance_normalized': 0
            }
            
        sentence_lengths = [len(sentence) for sentence in tokenized_sentences]
        
        mean_length = float(np.mean(sentence_lengths))
        variance = float(np.var(sentence_lengths))
        
        return {
            'mean_sentence_length': mean_length,
            'median_sentence_length': float(np.median(sentence_lengths)),
            'std_sentence_length': float(np.std(sentence_lengths)),
            'min_sentence_length': float(min(sentence_lengths)),
            'max_sentence_length': float(max(sentence_lengths)),
            'sentence_length_variance': variance,
            'length_variance_normalized': variance / (mean_length**2 + 1e-10)  # Normalized variance
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
                'dis_legomena_ratio': 0,
                'yule_k': 0,
                'simpson_d': 0,
                'herdan_c': 0,
                'guiraud_r': 0,
                'sichel_s': 0
            }
            
        # Number of tokens
        N = len(tokens)
        
        # Number of unique tokens (vocabulary size)
        V = len(set(tokens))
        
        # Frequency distribution
        freq_dist = Counter(tokens)
        
        # Hapax and dis legomena (words occurring once and twice)
        V1 = sum(1 for freq in freq_dist.values() if freq == 1)
        V2 = sum(1 for freq in freq_dist.values() if freq == 2)
        
        # Calculate Yule's K
        M1 = N
        M2 = sum(freq ** 2 for freq in freq_dist.values())
        yule_k = 10000 * (M2 - M1) / (M1 ** 2) if N > 0 and M1 > 0 else 0
        
        # Simpson's D (probability that two randomly chosen words are the same)
        simpson_d = sum((freq * (freq - 1)) / (N * (N - 1)) for freq in freq_dist.values()) if N > 1 else 0
        
        # Herdan's C (log vocabulary size / log tokens)
        herdan_c = math.log(V) / math.log(N) if N > 0 and V > 0 else 0
        
        # Guiraud's R (vocabulary size / sqrt(tokens))
        guiraud_r = V / math.sqrt(N) if N > 0 else 0
        
        # Sichel's S (proportion of dis legomena)
        sichel_s = V2 / V if V > 0 else 0
            
        return {
            'unique_tokens_ratio': V / N if N > 0 else 0,
            'hapax_legomena_ratio': V1 / N if N > 0 else 0,
            'dis_legomena_ratio': V2 / N if N > 0 else 0,
            'yule_k': yule_k,
            'simpson_d': simpson_d,
            'herdan_c': herdan_c,
            'guiraud_r': guiraud_r,
            'sichel_s': sichel_s
        }
    
    def calculate_word_position_features(self, tokenized_sentences: List[List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate features related to word positions and sentence structure.
        
        Args:
            tokenized_sentences: List of tokenized sentences
            
        Returns:
            Dictionary with word position and structural features
        """
        if not tokenized_sentences:
            return {
                'sentence_positions': {},
                'sentence_complexity': {
                    'avg_words_before_punct': 0,
                    'punct_per_sentence': 0,
                    'words_per_punct': 0
                }
            }
            
        # Calculate word positions in sentences
        sentence_positions = defaultdict(list)
        words_before_punct = []
        total_punct = 0
        total_words = 0
        
        for sentence in tokenized_sentences:
            sentence_length = len(sentence)
            if sentence_length == 0:
                continue
                
            # Track words and punctuation
            word_count = 0
            punct_count = 0
            words_since_punct = 0
            
            for i, token in enumerate(sentence):
                # Calculate normalized position
                norm_position = i / (sentence_length - 1) if sentence_length > 1 else 0.5
                sentence_positions[token].append(norm_position)
                
                # Count words and punctuation
                if any(c.isalpha() for c in token):
                    word_count += 1
                    words_since_punct += 1
                elif any(c in '.,;:!?' for c in token):
                    punct_count += 1
                    if words_since_punct > 0:
                        words_before_punct.append(words_since_punct)
                    words_since_punct = 0
            
            total_punct += punct_count
            total_words += word_count
        
        # Calculate average position for each word
        avg_positions = {}
        for token, positions in sentence_positions.items():
            if positions:
                avg_positions[token] = sum(positions) / len(positions)
        
        # Calculate sentence complexity metrics
        avg_words_before_punct = (
            np.mean(words_before_punct) if words_before_punct else 0
        )
        punct_per_sentence = total_punct / len(tokenized_sentences) if tokenized_sentences else 0
        words_per_punct = total_words / total_punct if total_punct > 0 else 0
                
        return {
            'sentence_positions': avg_positions,
            'sentence_complexity': {
                'avg_words_before_punct': float(avg_words_before_punct),
                'punct_per_sentence': float(punct_per_sentence),
                'words_per_punct': float(words_per_punct)
            }
        }
        
    def analyze_transition_patterns(self, tokenized_sentences: List[List[str]]) -> Dict[str, float]:
        """
        Analyze patterns in how sentences and clauses transition between each other.
        This captures the flow and rhythm of writing without considering specific content.
        
        Args:
            tokenized_sentences: List of tokenized sentences
            
        Returns:
            Dictionary with transition pattern features
        """
        if not tokenized_sentences:
            return {
                'length_transition_smoothness': 0,
                'length_pattern_repetition': 0,
                'clause_boundary_regularity': 0,
                'sentence_rhythm_consistency': 0,
                'transition_ratio_variance': 0,  # New feature
                'sentence_complexity_ratio': 0   # New feature
            }

        # Calculate sentence length transitions
        length_diffs = []
        length_patterns = []
        clause_positions = []
        rhythm_patterns = []
        transition_ratios = []
        complexity_scores = []

        for i, sentence in enumerate(tokenized_sentences):
            # Get sentence length
            current_length = len(sentence)
            length_patterns.append(current_length)

            # Calculate length transition if not first sentence
            if i > 0:
                prev_length = len(tokenized_sentences[i-1])
                length_diffs.append(abs(current_length - prev_length))
                
                # Calculate transition ratio (relative change in length)
                if prev_length > 0:
                    transition_ratios.append(current_length / prev_length)

            # Identify potential clause boundaries using punctuation and conjunctions
            clause_boundary_markers = [',', ';', ':', '.', 'και', 'δε', 'γαρ', 'ουν']
            
            # Get positions of clause boundaries normalized by sentence length
            if current_length > 0:
                positions = []
                for j, token in enumerate(sentence):
                    if any(marker in token.lower() for marker in clause_boundary_markers):
                        positions.append(j/current_length)
                
                # Only add if we found some boundaries
                if positions:
                    clause_positions.extend(positions)
                    
                    # Calculate complexity based on number of clauses per sentence length
                    complexity_scores.append(len(positions) / current_length)

            # Create rhythm pattern based on token lengths and variance
            if current_length > 0:
                # Get lengths of words in the sentence
                word_lengths = [len(token) for token in sentence if any(c.isalpha() for c in token)]
                if word_lengths:
                    # Calculate standard deviation of word lengths in this sentence
                    length_std = np.std(word_lengths)
                    # Calculate alternation pattern (differences between consecutive words)
                    alternation = [abs(word_lengths[j] - word_lengths[j-1]) for j in range(1, len(word_lengths))]
                    
                    # Store both metrics in rhythm patterns
                    if alternation:
                        rhythm_patterns.append((length_std, np.mean(alternation)))
                    else:
                        rhythm_patterns.append((length_std, 0))

        features = {}

        # Measure how smoothly sentence lengths transition (lower value = smoother)
        features['length_transition_smoothness'] = (
            float(np.std(length_diffs)) if length_diffs else 0
        )

        # Measure repetition in sentence length patterns
        if len(length_patterns) > 1:
            # Use autocorrelation to detect patterns
            autocorr = np.correlate(length_patterns, length_patterns, mode='full')
            center_idx = len(autocorr)//2
            if center_idx < len(autocorr) - 1:
                max_corr = np.max(autocorr[center_idx+1:])
                features['length_pattern_repetition'] = float(
                    max_corr / autocorr[center_idx] if autocorr[center_idx] > 0 else 0
                )
            else:
                features['length_pattern_repetition'] = 0
        else:
            features['length_pattern_repetition'] = 0

        # Measure regularity of clause boundaries
        features['clause_boundary_regularity'] = float(
            np.std(clause_positions) if clause_positions else 0
        )

        # Measure consistency in sentence rhythm - use both metrics from rhythm_patterns
        if rhythm_patterns:
            std_values = [rp[0] for rp in rhythm_patterns]
            alt_values = [rp[1] for rp in rhythm_patterns]
            # Combine both metrics in a weighted manner
            features['sentence_rhythm_consistency'] = float(
                0.7 * np.std(std_values) + 0.3 * np.std(alt_values) if std_values and alt_values else 0
            )
        else:
            features['sentence_rhythm_consistency'] = 0
            
        # New feature: variance in transition ratios (how consistently sentences change in length)
        features['transition_ratio_variance'] = float(
            np.var(transition_ratios) if transition_ratios else 0
        )
        
        # New feature: average sentence complexity ratio
        features['sentence_complexity_ratio'] = float(
            np.mean(complexity_scores) if complexity_scores else 0
        )

        return features

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
        normalized_text = preprocessed_text.get('normalized_text', ' '.join(words))
        
        # Extract various features
        features = {}
        
        # Word frequency distribution
        features['word_frequencies'] = self.calculate_frequency_distribution(words)
        
        # N-gram frequencies (both word and character level)
        features['word_bigrams'] = self.calculate_ngram_frequency(words, n=2)
        features['word_trigrams'] = self.calculate_ngram_frequency(words, n=3)
        
        # TF-IDF features
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer must be fitted before extracting features. Call fit() first.")
        
        tfidf_features = self.tfidf.transform([normalized_text]).toarray()[0]
        features['char_ngrams'] = dict(zip(
            self.tfidf.get_feature_names_out(),
            tfidf_features
        ))
        
        # Sentence length and structure statistics
        features['sentence_stats'] = self.calculate_sentence_length_stats(tokenized_sentences)
        
        # Vocabulary richness metrics
        features['vocabulary_richness'] = self.calculate_vocabulary_richness(words)
        
        # Word position and sentence complexity features
        position_features = self.calculate_word_position_features(tokenized_sentences)
        features['word_positions'] = position_features['sentence_positions']
        features['sentence_complexity'] = position_features['sentence_complexity']

        # Transition pattern analysis
        features['transition_patterns'] = self.analyze_transition_patterns(tokenized_sentences)
        
        return features 