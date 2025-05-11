"""
Module for preprocessing Greek texts.
"""

import re
from typing import Dict, List, Any

class GreekTextPreprocessor:
    """Preprocess Greek texts for analysis."""
    
    def __init__(self):
        """Initialize preprocessor."""
        # Common Greek punctuation marks
        self.punctuation = '.,;:!?·'
        
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
        Preprocess a text string.
        
        Args:
            text: Text string to preprocess
            
        Returns:
            Dictionary containing preprocessed text data
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Split into sentences
        sentences = self.split_sentences(cleaned_text)
        
        # Tokenize
        words = self.tokenize(cleaned_text)
        
        # Create tokenized sentences
        tokenized_sentences = [self.tokenize(s) for s in sentences]
        
        return {
            'raw_text': text,
            'normalized_text': cleaned_text,
            'sentences': sentences,
            'words': words,
            'tokenized_sentences': tokenized_sentences
        }
    
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