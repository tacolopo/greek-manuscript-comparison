#!/usr/bin/env python3
"""
Simple test to isolate NLP feature extraction issues.
"""

import sys
import os
sys.path.append('src')

from src.multi_comparison import MultipleManuscriptComparison
from src.similarity import SimilarityCalculator

def test_simple_nlp():
    """Test NLP feature extraction with a simple example."""
    
    print("Creating MultipleManuscriptComparison with advanced NLP...")
    
    # Initialize comparison object with advanced NLP
    comparison = MultipleManuscriptComparison(
        use_advanced_nlp=True,
        output_dir="test_output",
        visualizations_dir="test_viz"
    )
    
    print(f"use_advanced_nlp: {comparison.use_advanced_nlp}")
    print(f"advanced_processor: {comparison.advanced_processor}")
    
    # Set up NLP-only configuration
    nlp_config = {
        'weights': {
            'vocabulary': 0.0,
            'sentence': 0.0,
            'transitions': 0.0,
            'ngrams': 0.0,
            'syntactic': 1.0
        }
    }
    
    # Set custom weights in the similarity calculator
    custom_calculator = SimilarityCalculator()
    custom_calculator.weights = nlp_config['weights']
    comparison.similarity_calculator = custom_calculator
    
    print(f"Using weights: {custom_calculator.weights}")
    
    # Test with simple Greek texts
    test_texts = {
        'test1': 'καὶ εἶπεν ὁ θεὸς γενηθήτω φῶς καὶ ἐγένετο φῶς',
        'test2': 'ἐν ἀρχῇ ἦν ὁ λόγος καὶ ὁ λόγος ἦν πρὸς τὸν θεόν'
    }
    
    print("\nTesting with simple texts...")
    
    try:
        # Test preprocessing first
        print("Testing preprocessing...")
        preprocessed = {}
        for name, text in test_texts.items():
            print(f"Preprocessing {name}...")
            preprocessed[name] = comparison.preprocessor.preprocess(text)
            
            # Add advanced NLP features if available
            if comparison.advanced_processor:
                try:
                    print(f"Processing NLP features for {name}...")
                    nlp_features = comparison.advanced_processor.process_document(text)
                    preprocessed[name]['nlp_features'] = nlp_features
                    print(f"NLP features keys: {list(nlp_features.keys())}")
                    if 'pos_tags' in nlp_features:
                        print(f"POS tags: {nlp_features['pos_tags']}")
                except Exception as e:
                    print(f"Error processing NLP features for {name}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Test feature extraction
        print("\nTesting feature extraction...")
        features = comparison.extract_features(preprocessed)
        
        for name, feature_dict in features.items():
            print(f"\n{name} features:")
            print(f"  Keys: {list(feature_dict.keys())}")
            if 'syntactic_features' in feature_dict:
                syntactic = feature_dict['syntactic_features']
                print(f"  Syntactic features: {len(syntactic)}")
                non_zero = sum(1 for v in syntactic.values() if v != 0)
                print(f"  Non-zero syntactic features: {non_zero}")
                print(f"  Sample features: {list(syntactic.items())[:3]}")
            else:
                print(f"  No syntactic features found!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_nlp() 