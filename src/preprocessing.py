"""
Module for preprocessing Greek texts.
"""

import re
import unicodedata
from typing import Dict, List, Any, Optional

class GreekTextPreprocessor:
    """Preprocess Greek texts for analysis."""
    
    def __init__(self, remove_stopwords=False, normalize_accents=True, lowercase=True):
        """Initialize preprocessor."""
        # Common Greek punctuation marks
        self.punctuation = '.,;:!?·'
        
        # Additional processing options
        self.remove_stopwords = remove_stopwords
        self.normalize_accents = normalize_accents
        self.lowercase = lowercase
        
        # Advanced processor (will be set externally if needed)
        self.advanced_processor = None
        
        # Greek stopwords (if needed)
        self.stopwords = set([
            'ὁ', 'ἡ', 'τό', 'οἱ', 'αἱ', 'τά', 'τοῦ', 'τῆς', 'τῶν',
            'τῷ', 'τῇ', 'τοῖς', 'ταῖς', 'τόν', 'τήν', 'καί', 'δέ', 
            'γάρ', 'οὖν', 'μέν', 'δή', 'τε', 'ἀλλά', 'ἐν', 'εἰς', 
            'ἐκ', 'ἀπό', 'πρός', 'διά', 'ἐπί', 'κατά', 'μετά', 'περί'
        ])
    
    def clean_text(self, text: str) -> str:
        """
        Clean raw text by normalizing whitespace and removing unwanted characters.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove line numbers and other non-text markers
        text = re.sub(r'[\d\[\]⟦⟧\{\}]+', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Greek text by handling accents and case.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Handle accents if specified
        if self.normalize_accents:
            # NFD normalization decomposes combined characters
            text = unicodedata.normalize('NFD', text)
            # Remove diacritical marks (category Mn)
            text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
            
        return text
    
    def filter_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from a list of tokens.
        
        Args:
            tokens: List of tokens to filter
            
        Returns:
            List of tokens with stopwords removed
        """
        return [token for token in tokens if token.lower() not in self.stopwords]
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using Greek punctuation.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.;!?]', text)
        
        # Clean and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Split on whitespace and punctuation
        tokens = []
        for word in text.split():
            # Handle punctuation attached to words
            if word[-1] in self.punctuation:
                tokens.extend([word[:-1], word[-1]])
            else:
                tokens.append(word)
        
        return [t for t in tokens if t]
    
    def preprocess(self, text: str) -> Dict[str, Any]:
        """
        Preprocess text for analysis.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Dictionary with preprocessed data
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Normalize text (lowercase, remove accents)
        normalized_text = self.normalize_text(cleaned_text)
        
        # Split into sentences
        sentences = self.split_sentences(normalized_text)
        
        # Tokenize
        tokenized_sentences = [self.tokenize(sent) for sent in sentences]
        tokens = [token for sent in tokenized_sentences for token in sent]
        
        # Apply additional processing
        if self.remove_stopwords:
            tokens = self.filter_stopwords(tokens)
        
        # Process with advanced NLP if available
        nlp_features = {}
        if self.advanced_processor:
            nlp_features = self.advanced_processor.process_document(normalized_text)
            # Debug print for POS tags
            if 'pos_tags' in nlp_features:
                print(f"DEBUG: Found {len(nlp_features['pos_tags'])} POS tags")
                print(f"POS tag sample: {nlp_features['pos_tags'][:10]}")
                tag_counts = {}
                for tag in nlp_features['pos_tags']:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                print(f"Unique POS tags: {sorted(tag_counts.keys())}")
        
        result = {
            'cleaned_text': cleaned_text,
            'normalized_text': normalized_text,
            'sentences': sentences,
            'tokenized_sentences': tokenized_sentences,
            'words': tokens,
            'nlp_features': nlp_features
        }
        
        return result
    
    def preprocess_file(self, file_path: str) -> Dict[str, Any]:
        """
        Preprocess a text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Dictionary containing preprocessed text data
        """
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Use the text preprocessing method
        return self.preprocess(text) 