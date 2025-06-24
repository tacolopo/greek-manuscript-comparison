"""
Module for extracting comprehensive NLP features from Greek texts.
Enhanced version with semantic embeddings and advanced stylistic features for clustering.
"""

import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats

class FeatureExtractor:
    """Extract comprehensive linguistic features from Greek texts for ML clustering."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.tfidf = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),  # Character n-grams from size 3 to 5
            max_features=1000
        )
        self.word_tfidf = TfidfVectorizer(
            analyzer='word',
            max_features=500,
            min_df=2,  # Ignore words that appear in less than 2 documents
            max_df=0.8  # Ignore words that appear in more than 80% of documents
        )
        self.is_fitted = False
        
        # Greek function words and particles for stylistic analysis
        self.function_words = {
            'articles': ['ὁ', 'ἡ', 'τό', 'οἱ', 'αἱ', 'τά', 'τοῦ', 'τῆς', 'τῶν', 'τῷ', 'τῇ', 'τοῖς', 'ταῖς', 'τόν', 'τήν'],
            'particles': ['δέ', 'γάρ', 'οὖν', 'μέν', 'δή', 'τε', 'γε', 'μήν', 'τοι', 'ἄρα'],
            'conjunctions': ['καί', 'ἀλλά', 'ἤ', 'οὐδέ', 'μηδέ', 'εἰ', 'ἐάν', 'ὅτι', 'ἵνα', 'ὡς'],
            'prepositions': ['ἐν', 'εἰς', 'ἐκ', 'ἀπό', 'πρός', 'διά', 'ἐπί', 'κατά', 'μετά', 'περί', 'ὑπό', 'παρά'],
            'pronouns': ['αὐτός', 'οὗτος', 'ἐκεῖνος', 'τις', 'τί', 'ὅς', 'ἥ', 'ὅ', 'ἐγώ', 'σύ']
        }
    
    def fit(self, texts: List[str]):
        """
        Fit the TF-IDF vectorizers on a corpus of texts.
        
        Args:
            texts: List of texts to fit on
        """
        # Fit character n-gram TF-IDF
        self.tfidf.fit(texts)
        
        # Fit word-level TF-IDF
        self.word_tfidf.fit(texts)
        
        self.is_fitted = True
    
    def calculate_vocabulary_richness(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate comprehensive vocabulary richness metrics.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary with vocabulary richness metrics
        """
        if not tokens:
            return {
                'type_token_ratio': 0, 'hapax_legomena_ratio': 0, 'dis_legomena_ratio': 0,
                'yules_k': 0, 'simpsons_d': 0, 'herdan_c': 0, 'guiraud_r': 0,
                'vocab_size': 0, 'total_tokens': 0, 'entropy': 0
            }
            
        N = len(tokens)  # Total tokens
        V = len(set(tokens))  # Vocabulary size
        freq_dist = Counter(tokens)
        
        # Hapax and dis legomena
        V1 = sum(1 for freq in freq_dist.values() if freq == 1)
        V2 = sum(1 for freq in freq_dist.values() if freq == 2)
        
        # Advanced vocabulary measures
        # Yule's K (vocabulary distribution)
        M1 = N
        M2 = sum(freq ** 2 for freq in freq_dist.values())
        yules_k = 10000 * (M2 - M1) / (M1 ** 2) if N > 0 and M1 > 0 else 0
        
        # Simpson's D (concentration)
        simpsons_d = sum((freq * (freq - 1)) / (N * (N - 1)) for freq in freq_dist.values()) if N > 1 else 0
        
        # Herdan's C (logarithmic type-token ratio)
        herdan_c = math.log(V) / math.log(N) if N > 0 and V > 0 else 0
        
        # Guiraud's R (corrected type-token ratio)
        guiraud_r = V / math.sqrt(N) if N > 0 else 0
        
        # Entropy (lexical diversity)
        probs = [freq / N for freq in freq_dist.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            
        return {
            'type_token_ratio': V / N if N > 0 else 0,
            'hapax_legomena_ratio': V1 / N if N > 0 else 0,
            'dis_legomena_ratio': V2 / N if N > 0 else 0,
            'yules_k': yules_k,
            'simpsons_d': simpsons_d,
            'herdan_c': herdan_c,
            'guiraud_r': guiraud_r,
            'vocab_size': V,
            'total_tokens': N,
            'entropy': entropy
        }
    
    def calculate_sentence_complexity(self, tokenized_sentences: List[List[str]]) -> Dict[str, float]:
        """
        Calculate comprehensive sentence complexity metrics.
        
        Args:
            tokenized_sentences: List of tokenized sentences
            
        Returns:
            Dictionary with sentence complexity metrics
        """
        if not tokenized_sentences:
            return {
                'mean_length': 0, 'std_length': 0, 'median_length': 0,
                'length_variance': 0, 'num_sentences': 0, 'complexity_score': 0,
                'short_sentence_ratio': 0, 'long_sentence_ratio': 0
            }
            
        lengths = [len(sent) for sent in tokenized_sentences if sent]
        
        if not lengths:
            return {
                'mean_length': 0, 'std_length': 0, 'median_length': 0,
                'length_variance': 0, 'num_sentences': 0, 'complexity_score': 0,
                'short_sentence_ratio': 0, 'long_sentence_ratio': 0
            }
        
        mean_length = np.mean(lengths)
        
        # Calculate complexity score based on length variation
        complexity_score = np.std(lengths) / mean_length if mean_length > 0 else 0
        
        # Ratio of short (< 10 words) and long (> 20 words) sentences
        short_ratio = sum(1 for l in lengths if l < 10) / len(lengths)
        long_ratio = sum(1 for l in lengths if l > 20) / len(lengths)
        
        return {
            'mean_length': float(mean_length),
            'std_length': float(np.std(lengths)),
            'median_length': float(np.median(lengths)),
            'length_variance': float(np.var(lengths)),
            'num_sentences': len(lengths),
            'complexity_score': float(complexity_score),
            'short_sentence_ratio': short_ratio,
            'long_sentence_ratio': long_ratio
        }
    
    def calculate_function_word_features(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate function word usage patterns for stylistic analysis.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary with function word features
        """
        if not tokens:
            return {f'{category}_ratio': 0 for category in self.function_words.keys()}
        
        total_tokens = len(tokens)
        token_counts = Counter(tokens)
        features = {}
        
        for category, words in self.function_words.items():
            count = sum(token_counts.get(word, 0) for word in words)
            features[f'{category}_ratio'] = count / total_tokens
        
        # Calculate overall function word density
        total_function_words = sum(features.values())
        features['total_function_word_ratio'] = total_function_words
        
        return features
    
    def calculate_morphological_features(self, nlp_features: Dict) -> Dict[str, float]:
        """
        Calculate morphological complexity features from NLP analysis.
        
        Args:
            nlp_features: Dictionary containing NLP features
            
        Returns:
            Dictionary with morphological features
        """
        features = {
            'morphological_diversity': 0,
            'case_variation': 0,
            'tense_variation': 0,
            'lemma_token_ratio': 0
        }
        
        if 'morphological_features' in nlp_features and nlp_features['morphological_features']:
            morph_features = nlp_features['morphological_features']
            
            # Count unique morphological patterns
            unique_patterns = len(set(str(f) for f in morph_features if f))
            total_words = len(morph_features)
            features['morphological_diversity'] = unique_patterns / total_words if total_words > 0 else 0
        
        if 'lemmas' in nlp_features and nlp_features['lemmas']:
            lemmas = nlp_features['lemmas']
            words = len(lemmas)
            unique_lemmas = len(set(lemmas))
            features['lemma_token_ratio'] = unique_lemmas / words if words > 0 else 0
        
        return features
    
    def get_semantic_embeddings(self, text: str, advanced_processor) -> Dict[str, float]:
        """
        Extract semantic embeddings using Greek BERT and sentence transformers.
        
        Args:
            text: Input text
            advanced_processor: AdvancedGreekProcessor instance
            
        Returns:
            Dictionary with semantic features
        """
        if not advanced_processor:
            return {}
        
        try:
            # Get sentence embeddings
            sentences = text.split('.')[:5]  # Use first 5 sentences for efficiency
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return {}
            
            sentence_embeddings = advanced_processor.get_sentence_embeddings(sentences)
            
            # Calculate semantic statistics
            if len(sentence_embeddings) > 1:
                # Mean pairwise similarity between sentences (coherence)
                similarities = []
                for i in range(len(sentence_embeddings)):
                    for j in range(i + 1, len(sentence_embeddings)):
                        sim = np.dot(sentence_embeddings[i], sentence_embeddings[j]) / (
                            np.linalg.norm(sentence_embeddings[i]) * np.linalg.norm(sentence_embeddings[j])
                        )
                        similarities.append(sim)
                
                semantic_coherence = np.mean(similarities) if similarities else 0
                semantic_variance = np.var(similarities) if similarities else 0
            else:
                semantic_coherence = 1.0
                semantic_variance = 0.0
            
            # Overall text embedding statistics
            mean_embedding = np.mean(sentence_embeddings, axis=0)
            embedding_norm = np.linalg.norm(mean_embedding)
            
            return {
                'semantic_coherence': float(semantic_coherence),
                'semantic_variance': float(semantic_variance),
                'embedding_norm': float(embedding_norm),
                'embedding_dimensionality': len(mean_embedding)
            }
            
        except Exception as e:
            print(f"Warning: Error extracting semantic embeddings: {e}")
            return {}
    
    def calculate_punctuation_features(self, text: str) -> Dict[str, float]:
        """
        Calculate punctuation and formatting features.
        
        Args:
            text: Original text
            
        Returns:
            Dictionary with punctuation features
        """
        if not text:
            return {
                'period_ratio': 0, 'comma_ratio': 0, 'question_ratio': 0,
                'exclamation_ratio': 0, 'colon_ratio': 0, 'semicolon_ratio': 0,
                'punctuation_density': 0
            }
        
        punct_counts = {
            'period': text.count('.'),
            'comma': text.count(','),
            'question': text.count('?') + text.count(';'),  # Greek question mark
            'exclamation': text.count('!'),
            'colon': text.count(':'),
            'semicolon': text.count(';')
        }
        
        total_chars = len(text)
        total_punct = sum(punct_counts.values())
        
        features = {}
        for punct_type, count in punct_counts.items():
            features[f'{punct_type}_ratio'] = count / total_chars if total_chars > 0 else 0
        
        features['punctuation_density'] = total_punct / total_chars if total_chars > 0 else 0
        
        return features
    
    def extract_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from a list of tokens."""
        from nltk.util import ngrams
        return list(ngrams(tokens, n))
    
    def calculate_ngram_frequency(self, tokens: List[str], n: int = 2) -> Dict[Tuple[str, ...], float]:
        """Calculate normalized frequency distribution of n-grams."""
        if len(tokens) < n:
            return {}
            
        token_ngrams = self.extract_ngrams(tokens, n)
        if not token_ngrams:
            return {}
            
        counter = Counter(token_ngrams)
        total = len(token_ngrams)
        
        # Return only the most frequent n-grams
        most_common = counter.most_common(30)  # Reduced for efficiency
        return {ngram: count / total for ngram, count in most_common}
    
    def get_tfidf_features(self, text: str) -> Dict[str, float]:
        """Get TF-IDF character and word features."""
        if not self.is_fitted:
            return {}
        
        features = {}
        
        # Character n-gram TF-IDF
        char_tfidf = self.tfidf.transform([text])
        char_features = self.tfidf.get_feature_names_out()
        
        for i, feature_name in enumerate(char_features):
            score = char_tfidf[0, i]
            if score > 0.1:  # Only keep significant features
                features[f'char_tfidf_{feature_name}'] = score
        
        # Word-level TF-IDF
        word_tfidf = self.word_tfidf.transform([text])
        word_features = self.word_tfidf.get_feature_names_out()
        
        for i, feature_name in enumerate(word_features):
            score = word_tfidf[0, i]
            if score > 0.1:  # Only keep significant features
                features[f'word_tfidf_{feature_name}'] = score
        
        return features
    
    def extract_all_features(self, preprocessed_text: Dict, advanced_processor=None) -> Dict:
        """
        Extract comprehensive features from preprocessed text.
        
        Args:
            preprocessed_text: Dictionary containing preprocessed text data
            advanced_processor: Optional AdvancedGreekProcessor for semantic features
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # Extract vocabulary features
        if 'words' in preprocessed_text:
            words = preprocessed_text['words']
            features['vocabulary_richness'] = self.calculate_vocabulary_richness(words)
            features['function_words'] = self.calculate_function_word_features(words)
            
            # Extract n-gram features
            features['word_bigrams'] = self.calculate_ngram_frequency(words, n=2)
            features['word_trigrams'] = self.calculate_ngram_frequency(words, n=3)
        
        # Extract sentence features
        if 'sentences' in preprocessed_text:
            sentences = preprocessed_text['sentences']
            tokenized_sentences = [sentence.split() for sentence in sentences]
            features['sentence_complexity'] = self.calculate_sentence_complexity(tokenized_sentences)
        
        # Extract punctuation features
        if 'cleaned_text' in preprocessed_text:
            features['punctuation'] = self.calculate_punctuation_features(preprocessed_text['cleaned_text'])
        
        # Extract TF-IDF features
        if 'normalized_text' in preprocessed_text and self.is_fitted:
            features['tfidf'] = self.get_tfidf_features(preprocessed_text['normalized_text'])
        
        # Extract advanced NLP features
        if 'nlp_features' in preprocessed_text:
            nlp_features = preprocessed_text['nlp_features']
            features['morphological'] = self.calculate_morphological_features(nlp_features)
            
            # Extract semantic embeddings if advanced processor is available
            if advanced_processor and 'normalized_text' in preprocessed_text:
                features['semantic'] = self.get_semantic_embeddings(
                    preprocessed_text['normalized_text'], advanced_processor
                )
        
        return features 