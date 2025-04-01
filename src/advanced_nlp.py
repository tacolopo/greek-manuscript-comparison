"""
Module for advanced NLP processing of Greek texts.
"""

import os
from typing import List, Dict, Tuple, Optional, Union, Any
import pickle
import warnings

import numpy as np
import torch
from tqdm import tqdm
from cltk.tag.pos import POSTag
from cltk.lemmatize.greek.backoff import BackoffGreekLemmatizer
from cltk.dependency.tree import DependencyTree
from cltk.dependency.greek import GreekDependencyParser
from cltk.ner.ner import tag_ner
from cltk.core.data_types import Doc
from cltk.nlp import NLP
from cltk.embeddings.embeddings import FastTextEmbeddings, Word2VecEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
import torch.nn.functional as F

# Initialize CLTK pipeline for Greek
try:
    greek_nlp = NLP(language="grc")
except Exception as e:
    warnings.warn(f"Error initializing Greek NLP pipeline: {e}. Some features may not be available.")
    greek_nlp = None

# Initialize Greek-specific NLP tools
try:
    greek_pos_tagger = POSTag('greek')
except Exception as e:
    warnings.warn(f"Error initializing Greek POS tagger: {e}. POS tagging may not be available.")
    greek_pos_tagger = None

try:
    greek_lemmatizer = BackoffGreekLemmatizer()
except Exception as e:
    warnings.warn(f"Error initializing Greek lemmatizer: {e}. Lemmatization may not be available.")
    greek_lemmatizer = None

try:
    greek_dependency_parser = GreekDependencyParser()
except Exception as e:
    warnings.warn(f"Error initializing Greek dependency parser: {e}. Dependency parsing may not be available.")
    greek_dependency_parser = None

# Initialize transformer model for semantic analysis
# Using multilingual BERT by default, but can be changed to a more specific model
CACHE_DIR = os.path.expanduser("~/.cache/greek_manuscript_comparison")
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    # Try to use a model fine-tuned for ancient languages or multilingual models
    transformer_model = SentenceTransformer('distiluse-base-multilingual-cased', 
                                            cache_folder=CACHE_DIR)
except Exception as e:
    warnings.warn(f"Error loading transformer model: {e}. Using a fallback model.")
    try:
        transformer_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', 
                                               cache_folder=CACHE_DIR)
    except Exception as e2:
        warnings.warn(f"Error loading fallback model: {e2}. Semantic analysis may not be available.")
        transformer_model = None


class AdvancedGreekProcessor:
    """Process Greek texts with advanced NLP techniques."""
    
    def __init__(self, use_embeddings: bool = True, 
                 use_transformers: bool = True,
                 embedding_dim: int = 300):
        """
        Initialize the advanced Greek processor.
        
        Args:
            use_embeddings: Whether to use word embeddings
            use_transformers: Whether to use transformer models
            embedding_dim: Dimension of word embeddings
        """
        self.use_embeddings = use_embeddings
        self.use_transformers = use_transformers
        self.embedding_dim = embedding_dim
        
        # Load or create word embeddings
        self.word_embedding_model = None
        if use_embeddings:
            try:
                # First try to load pretrained embeddings
                embedding_path = os.path.join(CACHE_DIR, 'greek_embeddings.pkl')
                if os.path.exists(embedding_path):
                    with open(embedding_path, 'rb') as f:
                        self.word_embedding_model = pickle.load(f)
                    print(f"Loaded word embeddings from {embedding_path}")
                else:
                    # If not available, use CLTK's embeddings or create a new one
                    try:
                        self.word_embedding_model = Word2VecEmbeddings(iso_code="grc")
                        print("Loaded CLTK Word2Vec embeddings")
                    except Exception as e:
                        warnings.warn(f"Could not load CLTK embeddings: {e}. Word embeddings not available.")
            except Exception as e:
                warnings.warn(f"Error loading word embeddings: {e}. Word embeddings not available.")
        
        # Initialize transformer model for contextualized embeddings
        self.transformer_model = transformer_model if use_transformers else None
    
    def process_document(self, text: str) -> Dict[str, Any]:
        """
        Process a Greek text with advanced NLP techniques.
        
        Args:
            text: Input Greek text
            
        Returns:
            Dictionary with processed NLP features
        """
        results = {}
        
        if greek_nlp:
            try:
                # Use the CLTK pipeline for full NLP processing
                doc = greek_nlp.analyze(text)
                
                # Extract tokens, lemmas, and POS tags from the document
                tokens = [token.string for token in doc.tokens]
                lemmas = [token.lemma for token in doc.tokens if token.lemma]
                pos_tags = [(token.string, token.pos) for token in doc.tokens if token.pos]
                
                results['tokens'] = tokens
                results['lemmas'] = lemmas
                results['pos_tags'] = pos_tags
                
                # Extract dependency parse information if available
                if hasattr(doc, 'dependency_parse'):
                    results['dependency_parse'] = doc.dependency_parse
                
                # Extract entities if available
                if hasattr(doc, 'entities'):
                    results['entities'] = doc.entities
                
            except Exception as e:
                warnings.warn(f"Error in CLTK pipeline: {e}. Falling back to individual components.")
        
        # If CLTK pipeline failed or is not available, use individual components
        if 'tokens' not in results and greek_pos_tagger:
            try:
                # Perform POS tagging
                pos_tags = greek_pos_tagger.tag_tnt(text.split())
                results['pos_tags'] = pos_tags
                results['tokens'] = [word for word, _ in pos_tags]
            except Exception as e:
                warnings.warn(f"Error in POS tagging: {e}")
        
        if 'lemmas' not in results and greek_lemmatizer and 'tokens' in results:
            try:
                # Perform lemmatization
                lemmas = [greek_lemmatizer.lemmatize(token) for token in results['tokens']]
                results['lemmas'] = [lemma[0][1] if lemma else token 
                                    for lemma, token in zip(lemmas, results['tokens'])]
            except Exception as e:
                warnings.warn(f"Error in lemmatization: {e}")
        
        if greek_dependency_parser and 'pos_tags' in results:
            try:
                # Perform dependency parsing
                dependency_tree = greek_dependency_parser.parse(results['pos_tags'])
                results['dependency_tree'] = dependency_tree
            except Exception as e:
                warnings.warn(f"Error in dependency parsing: {e}")
        
        # Generate word embeddings if model is available
        if self.word_embedding_model and 'tokens' in results:
            try:
                # Get embeddings for each token
                token_embeddings = {}
                for token in set(results['tokens']):
                    try:
                        token_embeddings[token] = self.word_embedding_model.get_embedding(token)
                    except:
                        # If token not in embedding vocabulary, use zeros
                        token_embeddings[token] = np.zeros(self.embedding_dim)
                
                results['token_embeddings'] = token_embeddings
                
                # Calculate average document embedding
                if token_embeddings:
                    embeddings = [token_embeddings.get(token, np.zeros(self.embedding_dim)) 
                                 for token in results['tokens']]
                    doc_embedding = np.mean(embeddings, axis=0)
                    results['doc_embedding'] = doc_embedding
            except Exception as e:
                warnings.warn(f"Error generating word embeddings: {e}")
        
        # Generate transformer-based embeddings
        if self.transformer_model:
            try:
                # Get document-level embedding using transformer model
                doc_embedding = self.transformer_model.encode(text)
                results['transformer_embedding'] = doc_embedding
            except Exception as e:
                warnings.warn(f"Error generating transformer embeddings: {e}")
        
        # Extract syntactic features
        if 'pos_tags' in results:
            pos_counts = {}
            for _, pos in results['pos_tags']:
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
            results['pos_counts'] = pos_counts
            
            # Calculate POS n-grams for syntactic patterns
            pos_sequence = [pos for _, pos in results['pos_tags']]
            pos_bigrams = list(zip(pos_sequence[:-1], pos_sequence[1:]))
            pos_trigrams = list(zip(pos_sequence[:-2], pos_sequence[1:-1], pos_sequence[2:]))
            
            # Count frequencies
            pos_bigram_counts = {}
            for bigram in pos_bigrams:
                pos_bigram_counts[bigram] = pos_bigram_counts.get(bigram, 0) + 1
                
            pos_trigram_counts = {}
            for trigram in pos_trigrams:
                pos_trigram_counts[trigram] = pos_trigram_counts.get(trigram, 0) + 1
                
            results['pos_bigrams'] = pos_bigram_counts
            results['pos_trigrams'] = pos_trigram_counts
        
        return results
    
    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using transformers.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Semantic similarity score (0-1)
        """
        if self.transformer_model is None:
            warnings.warn("Transformer model not available. Semantic similarity cannot be calculated.")
            return 0.0
            
        try:
            # Encode both texts
            embedding1 = self.transformer_model.encode(text1)
            embedding2 = self.transformer_model.encode(text2)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            # Ensure result is between 0 and 1
            similarity = float(max(0.0, min(1.0, similarity)))
            
            return similarity
        except Exception as e:
            warnings.warn(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def extract_syntactic_features(self, pos_tags: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Extract detailed syntactic features from POS tags.
        
        Args:
            pos_tags: List of (token, POS tag) pairs
            
        Returns:
            Dictionary with syntactic features
        """
        features = {}
        
        # Extract POS tag sequence
        pos_sequence = [pos for _, pos in pos_tags]
        
        # Calculate POS tag frequencies
        pos_counts = {}
        for pos in pos_sequence:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        # Calculate ratio of different parts of speech
        total_tags = len(pos_sequence)
        if total_tags > 0:
            pos_ratios = {pos: count / total_tags for pos, count in pos_counts.items()}
        else:
            pos_ratios = {}
        
        # Calculate verb-to-noun ratio (approximation for Ancient Greek)
        # This depends on the tag set used by the POS tagger
        noun_tags = {'N', 'n', 'NP', 'np', 'noun'}
        verb_tags = {'V', 'v', 'verb'}
        
        noun_count = sum(pos_counts.get(tag, 0) for tag in noun_tags if tag in pos_counts)
        verb_count = sum(pos_counts.get(tag, 0) for tag in verb_tags if tag in pos_counts)
        
        verb_noun_ratio = verb_count / noun_count if noun_count > 0 else 0
        
        # Extract POS n-grams
        pos_bigrams = list(zip(pos_sequence[:-1], pos_sequence[1:]))
        pos_trigrams = list(zip(pos_sequence[:-2], pos_sequence[1:-1], pos_sequence[2:]))
        
        # Count frequencies
        pos_bigram_counts = {}
        for bigram in pos_bigrams:
            pos_bigram_counts[bigram] = pos_bigram_counts.get(bigram, 0) + 1
            
        pos_trigram_counts = {}
        for trigram in pos_trigrams:
            pos_trigram_counts[trigram] = pos_trigram_counts.get(trigram, 0) + 1
        
        # Gather all features
        features['pos_counts'] = pos_counts
        features['pos_ratios'] = pos_ratios
        features['verb_noun_ratio'] = verb_noun_ratio
        features['pos_bigrams'] = pos_bigram_counts
        features['pos_trigrams'] = pos_trigram_counts
        
        return features
    
    def analyze_semantic_context(self, text: str, window_size: int = 5) -> Dict[str, Any]:
        """
        Analyze semantic context of words in the text using word embeddings.
        
        Args:
            text: Input text
            window_size: Context window size for word co-occurrence
            
        Returns:
            Dictionary with semantic context analysis
        """
        if not self.word_embedding_model:
            warnings.warn("Word embedding model not available. Semantic context analysis cannot be performed.")
            return {}
            
        try:
            # First get tokens
            tokens = text.split()
            
            # Calculate co-occurrence matrix
            vocab = list(set(tokens))
            vocab_size = len(vocab)
            word_to_id = {word: i for i, word in enumerate(vocab)}
            
            # Initialize co-occurrence matrix
            cooc_matrix = np.zeros((vocab_size, vocab_size))
            
            # Fill co-occurrence matrix
            for i, token in enumerate(tokens):
                token_id = word_to_id[token]
                
                # Define context window
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                
                # Update co-occurrence counts
                for j in range(start, end):
                    if j != i:  # Skip the word itself
                        context_token = tokens[j]
                        context_id = word_to_id[context_token]
                        cooc_matrix[token_id, context_id] += 1
            
            # Get embeddings for each token
            token_embeddings = {}
            for token in vocab:
                try:
                    token_embeddings[token] = self.word_embedding_model.get_embedding(token)
                except:
                    # If token not in embedding vocabulary, use zeros
                    token_embeddings[token] = np.zeros(self.embedding_dim)
            
            # Calculate semantic context for each word
            semantic_context = {}
            for token in vocab:
                token_id = word_to_id[token]
                
                # Get most common co-occurring words
                cooc_scores = cooc_matrix[token_id]
                top_cooc_indices = np.argsort(cooc_scores)[-10:]  # Top 10 co-occurring words
                top_cooc_words = [(vocab[idx], cooc_scores[idx]) for idx in top_cooc_indices if cooc_scores[idx] > 0]
                
                # Calculate semantic similarity with co-occurring words
                semantic_similarities = []
                for cooc_word, score in top_cooc_words:
                    if token in token_embeddings and cooc_word in token_embeddings:
                        sim = cosine_similarity([token_embeddings[token]], [token_embeddings[cooc_word]])[0][0]
                        semantic_similarities.append((cooc_word, float(sim)))
                
                semantic_context[token] = {
                    'co_occurrences': top_cooc_words,
                    'semantic_similarities': semantic_similarities
                }
            
            return {
                'semantic_context': semantic_context,
                'token_embeddings': token_embeddings
            }
        except Exception as e:
            warnings.warn(f"Error in semantic context analysis: {e}")
            return {} 