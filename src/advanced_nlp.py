"""
Advanced NLP processing for Greek texts.
"""

import torch
from typing import List, Dict, Any, Optional
import numpy as np
from cltk import NLP
from cltk.core.data_types import Doc
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import spacy

class AdvancedGreekProcessor:
    """Process Greek texts using advanced NLP techniques."""
    
    def __init__(self):
        """Initialize the processor."""
        # Initialize CLTK pipeline
        self.cltk_nlp = NLP(language="grc")
        
        # Initialize spaCy
        try:
            self.spacy_nlp = spacy.load("el_core_news_lg")
        except:
            print("Warning: Could not load Greek spaCy model. Some features will be limited.")
            self.spacy_nlp = None
        
        # Initialize transformers
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
        self.model = AutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
        
        # Initialize sentence transformer
        self.sentence_transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def process_document(self, text: str) -> Dict[str, Any]:
        """
        Process a Greek text document.
        
        Args:
            text: Text to process
            
        Returns:
            Dictionary of extracted features
        """
        # Process with CLTK
        doc: Doc = self.cltk_nlp.analyze(text=text)
        
        # Extract features
        features = {
            'lemmas': [word.lemma for word in doc.words],
            'pos_tags': [word.pos for word in doc.words],
            'morphological_features': [word.features for word in doc.words],
            'sentences': [sent.raw for sent in doc.sentences],
            'dependency_relations': [(word.string, word.governor.string if word.governor else None) 
                                   for word in doc.words]
        }
        
        # Add spaCy features if available
        if self.spacy_nlp:
            spacy_doc = self.spacy_nlp(text)
            features.update({
                'named_entities': [(ent.text, ent.label_) for ent in spacy_doc.ents],
                'noun_chunks': [chunk.text for chunk in spacy_doc.noun_chunks]
            })
        
        return features
    
    def get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Get embeddings for sentences using multilingual sentence transformer.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Array of sentence embeddings
        """
        return self.sentence_transformer.encode(sentences)
    
    def get_word_embeddings(self, text: str) -> torch.Tensor:
        """
        Get contextual word embeddings using Greek BERT.
        
        Args:
            text: Input text
            
        Returns:
            Tensor of word embeddings
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state
    
    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Get sentence embeddings
        emb1 = self.sentence_transformer.encode([text1])[0]
        emb2 = self.sentence_transformer.encode([text2])[0]
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def extract_syntactic_features(self, pos_tags: List[str]) -> Dict[str, float]:
        """
        Extract syntactic features from POS tags.
        
        Args:
            pos_tags: List of POS tags
            
        Returns:
            Dictionary of syntactic features
        """
        # Count tag frequencies
        tag_counts = {}
        for tag in pos_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        total_tags = len(pos_tags)
        
        # Calculate ratios
        features = {
            'noun_ratio': tag_counts.get('NOUN', 0) / total_tags,
            'verb_ratio': tag_counts.get('VERB', 0) / total_tags,
            'adj_ratio': tag_counts.get('ADJ', 0) / total_tags,
            'adv_ratio': tag_counts.get('ADV', 0) / total_tags,
            'function_word_ratio': sum(tag_counts.get(tag, 0) for tag in ['ADP', 'DET', 'CONJ']) / total_tags
        }
        
        return features 