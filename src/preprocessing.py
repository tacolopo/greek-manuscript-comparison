"""
Module for preprocessing Greek manuscript texts.
"""

import re
import unicodedata
from typing import List, Dict, Optional

import nltk
from cltk.corpus.greek.beta_to_unicode import Replacer
from cltk.tokenize.greek.sentence import GreekRegexSentenceTokenizer
from cltk.tokenize.greek.word import GreekWordTokenizer
from cltk.stops.greek.stops import STOPS as GREEK_STOPS

# Initialize CLTK tools
greek_sentence_tokenizer = GreekRegexSentenceTokenizer()
greek_word_tokenizer = GreekWordTokenizer()
beta_code_replacer = Replacer()

class GreekTextPreprocessor:
    """Class for preprocessing Greek manuscript texts."""
    
    def __init__(self, remove_stopwords: bool = True, 
                 normalize_accents: bool = True,
                 lowercase: bool = True):
        """
        Initialize the Greek text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove Greek stopwords
            normalize_accents: Whether to normalize Greek accents
            lowercase: Whether to convert text to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.normalize_accents = normalize_accents
        self.lowercase = lowercase
        self.stopwords = set(GREEK_STOPS)
        
    def clean_text(self, text: str) -> str:
        """
        Clean the input text by removing unwanted characters.
        
        Args:
            text: Input Greek text
            
        Returns:
            Cleaned text
        """
        # Replace beta code if present
        text = beta_code_replacer.beta_code_to_unicode(text)
        
        # Remove non-Greek characters (keeping punctuation)
        # Keep Greek characters (including accented ones), spaces, and basic punctuation
        text = re.sub(r'[^\u0370-\u03FF\u1F00-\u1FFF\s.,;!?·]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_accents_and_case(self, text: str) -> str:
        """
        Normalize accents and case of Greek text.
        
        Args:
            text: Input Greek text
            
        Returns:
            Normalized text
        """
        if self.normalize_accents:
            # Normalize Unicode characters to composed form
            text = unicodedata.normalize('NFC', text)
            
        if self.lowercase:
            text = text.lower()
            
        return text
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize Greek text into sentences.
        
        Args:
            text: Input Greek text
            
        Returns:
            List of sentences
        """
        return greek_sentence_tokenizer.tokenize(text)
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize Greek text into words.
        
        Args:
            text: Input Greek text or sentence
            
        Returns:
            List of words
        """
        tokens = greek_word_tokenizer.tokenize(text)
        
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stopwords]
            
        return tokens
    
    def process_text(self, text: str) -> Dict:
        """
        Process a full Greek text, performing all preprocessing steps.
        
        Args:
            text: Raw Greek manuscript text
            
        Returns:
            Dictionary containing the processed text in various forms
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Normalize accents and case
        normalized_text = self.normalize_accents_and_case(cleaned_text)
        
        # Tokenize into sentences
        sentences = self.tokenize_sentences(normalized_text)
        
        # Tokenize each sentence into words
        tokenized_sentences = [self.tokenize_words(sentence) for sentence in sentences]
        
        # Create a flat list of all words
        all_words = [word for sentence in tokenized_sentences for word in sentence]
        
        return {
            'raw_text': text,
            'cleaned_text': cleaned_text,
            'normalized_text': normalized_text,
            'sentences': sentences,
            'tokenized_sentences': tokenized_sentences,
            'words': all_words
        }
    
    def preprocess_file(self, file_path: str) -> Dict:
        """
        Process a Greek text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dictionary containing the processed text in various forms
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.process_text(text) 