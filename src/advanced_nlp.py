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
from collections import Counter

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
            'pos_tags': [str(word.pos) for word in doc.words],  # Convert POS to string
            'morphological_features': [word.features for word in doc.words],
            'sentences': [str(sent) for sent in doc.sentences],
            'dependency_relations': [(word.string, word.governor) for word in doc.words]
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
            pos_tags: List of POS tags as strings
            
        Returns:
            Dictionary of syntactic features
        """
        # Count tag frequencies
        tag_counts = {}
        for tag in pos_tags:
            tag_str = str(tag).lower()  # Ensure we're working with strings and lowercase
            tag_counts[tag_str] = tag_counts.get(tag_str, 0) + 1
        
        total_tags = len(pos_tags)
        if total_tags == 0:
            return {
                'noun_ratio': 0,
                'verb_ratio': 0,
                'adj_ratio': 0,
                'adv_ratio': 0,
                'function_word_ratio': 0,
                'pronoun_ratio': 0,
                'conjunction_ratio': 0,
                'particle_ratio': 0,
                'interjection_ratio': 0,
                'numeral_ratio': 0,
                'punctuation_ratio': 0,
                'tag_diversity': 0,
                'tag_entropy': 0,
                'noun_after_verb_ratio': 0,
                'adj_before_noun_ratio': 0,
                'adv_before_verb_ratio': 0,
                'verb_to_noun_prob': 0,
                'noun_to_verb_prob': 0,
                'noun_to_adj_prob': 0,
                'adj_to_noun_prob': 0,
                'noun_verb_ratio': 0
            }
        
        # Get strings for each tag in lowercase
        string_pos_tags = [str(tag).lower() for tag in pos_tags]
        
        # Calculate tag bigrams for transition probabilities
        tag_bigrams = []
        for i in range(len(string_pos_tags) - 1):
            tag_bigrams.append((string_pos_tags[i], string_pos_tags[i+1]))
        
        # Count bigram frequencies
        bigram_counts = Counter(tag_bigrams)
        total_bigrams = len(tag_bigrams) if tag_bigrams else 1
        
        # Basic tag frequencies - use both uppercase versions and full names to catch different tag formats
        # Updated based on OdyCy model tag set
        noun_tags = ['noun', 'substantive', 'noun_substantive', 'proper_noun']
        noun_count = sum(tag_counts.get(tag, 0) for tag in noun_tags)
        
        verb_tags = ['verb', 'finite_verb', 'auxiliary']
        verb_count = sum(tag_counts.get(tag, 0) for tag in verb_tags)
        
        adj_tags = ['adj', 'adjective']
        adj_count = sum(tag_counts.get(tag, 0) for tag in adj_tags)
        
        adv_tags = ['adv', 'adverb']
        adv_count = sum(tag_counts.get(tag, 0) for tag in adv_tags)
        
        function_tags = ['adposition', 'determiner', 'coordinating_conjunction', 'subordinating_conjunction']
        function_count = sum(tag_counts.get(tag, 0) for tag in function_tags)
        
        features = {
            'noun_ratio': noun_count / total_tags,
            'verb_ratio': verb_count / total_tags,
            'adj_ratio': adj_count / total_tags,
            'adv_ratio': adv_count / total_tags,
            'function_word_ratio': function_count / total_tags
        }
        
        # More granular tag ratios for better differentiation
        pron_tags = ['pron', 'pronoun']
        conj_tags = ['coordinating_conjunction', 'subordinating_conjunction']
        part_tags = ['part', 'particle']
        intj_tags = ['intj', 'interjection']
        num_tags = ['num', 'numeral']
        punct_tags = ['punct', 'punctuation']
        
        features.update({
            'pronoun_ratio': sum(tag_counts.get(tag, 0) for tag in pron_tags) / total_tags,
            'conjunction_ratio': sum(tag_counts.get(tag, 0) for tag in conj_tags) / total_tags,
            'particle_ratio': sum(tag_counts.get(tag, 0) for tag in part_tags) / total_tags,
            'interjection_ratio': sum(tag_counts.get(tag, 0) for tag in intj_tags) / total_tags,
            'numeral_ratio': sum(tag_counts.get(tag, 0) for tag in num_tags) / total_tags,
            'punctuation_ratio': sum(tag_counts.get(tag, 0) for tag in punct_tags) / total_tags
        })
        
        # Calculate syntactic tag diversity
        unique_tags = len(tag_counts)
        features['tag_diversity'] = unique_tags / total_tags if total_tags > 0 else 0
        
        # Calculate tag entropy (syntactic complexity)
        tag_probs = [count / total_tags for count in tag_counts.values()]
        features['tag_entropy'] = -sum(p * np.log2(p) if p > 0 else 0 for p in tag_probs)
        
        # Define pattern matching for bigrams
        # We need to check multiple possible tag formats using lowercase
        verb_noun_patterns = [('verb', 'noun'), ('verb', 'proper_noun'), ('auxiliary', 'noun')]
        noun_verb_patterns = [('noun', 'verb'), ('proper_noun', 'verb'), ('noun', 'auxiliary')]
        adj_noun_patterns = [('adjective', 'noun'), ('adjective', 'proper_noun')]
        adv_verb_patterns = [('adverb', 'verb'), ('adverb', 'auxiliary')]
        
        # POS tag sequence patterns
        # Ratio of nouns following verbs
        noun_after_verb = sum(bigram_counts.get(pattern, 0) for pattern in verb_noun_patterns) / total_bigrams
        features['noun_after_verb_ratio'] = noun_after_verb
        
        # Ratio of adjectives preceding nouns
        adj_before_noun = sum(bigram_counts.get(pattern, 0) for pattern in adj_noun_patterns) / total_bigrams
        features['adj_before_noun_ratio'] = adj_before_noun
        
        # Ratio of adverbs preceding verbs
        adv_before_verb = sum(bigram_counts.get(pattern, 0) for pattern in adv_verb_patterns) / total_bigrams
        features['adv_before_verb_ratio'] = adv_before_verb
        
        # Characteristic transition probabilities
        features['verb_to_noun_prob'] = (
            sum(bigram_counts.get(pattern, 0) for pattern in verb_noun_patterns) / 
            verb_count if verb_count > 0 else 0
        )
        
        features['noun_to_verb_prob'] = (
            sum(bigram_counts.get(pattern, 0) for pattern in noun_verb_patterns) / 
            noun_count if noun_count > 0 else 0
        )
        
        features['noun_to_adj_prob'] = (
            (bigram_counts.get(('noun', 'adjective'), 0) + bigram_counts.get(('proper_noun', 'adjective'), 0)) / 
            noun_count if noun_count > 0 else 0
        )
        
        features['adj_to_noun_prob'] = (
            sum(bigram_counts.get(pattern, 0) for pattern in adj_noun_patterns) / 
            adj_count if adj_count > 0 else 0
        )
        
        # Calculate syntactic complexity metrics
        if verb_count > 0:
            features['noun_verb_ratio'] = noun_count / verb_count
        else:
            features['noun_verb_ratio'] = 0
            
        return features 