#!/usr/bin/env python3
"""
Debug syntactic feature vectors to understand why similarities are so high.
"""

import sys
import os
import numpy as np
sys.path.append('src')

from src.multi_comparison import MultipleManuscriptComparison
from src.similarity import SimilarityCalculator

def debug_syntactic_features():
    """Debug the actual syntactic feature values."""
    
    print("Creating MultipleManuscriptComparison with advanced NLP...")
    
    # Initialize comparison object with advanced NLP
    comparison = MultipleManuscriptComparison(
        use_advanced_nlp=True,
        output_dir="debug_output",
        visualizations_dir="debug_viz"
    )
    
    # Test with a few different Greek texts
    test_texts = {
        'julian1': 'καὶ εἶπεν ὁ θεὸς γενηθήτω φῶς καὶ ἐγένετο φῶς καὶ εἶδεν ὁ θεὸς τὸ φῶς ὅτι καλόν',
        'julian2': 'ἐν ἀρχῇ ἦν ὁ λόγος καὶ ὁ λόγος ἦν πρὸς τὸν θεόν καὶ θεὸς ἦν ὁ λόγος',
        'pauline1': 'οὗτος ἦν ἐν ἀρχῇ πρὸς τὸν θεόν πάντα δι αὐτοῦ ἐγένετο καὶ χωρὶς αὐτοῦ'
    }
    
    print("\nProcessing texts and extracting syntactic features...")
    
    # Preprocess texts
    preprocessed = {}
    for name, text in test_texts.items():
        print(f"\nProcessing {name}...")
        preprocessed[name] = comparison.preprocessor.preprocess(text)
        
        # Add advanced NLP features
        if comparison.advanced_processor:
            nlp_features = comparison.advanced_processor.process_document(text)
            preprocessed[name]['nlp_features'] = nlp_features
            
            # Print POS tags
            pos_tags = nlp_features['pos_tags']
            print(f"  POS tags: {pos_tags}")
            
            # Extract syntactic features
            syntactic_features = comparison.advanced_processor.extract_syntactic_features(pos_tags)
            print(f"  Syntactic features:")
            for key, value in syntactic_features.items():
                print(f"    {key}: {value:.4f}")
    
    # Extract full features using the comparison object
    print("\nExtracting full feature vectors...")
    features = comparison.extract_features(preprocessed)
    
    # Examine the feature vectors
    print("\nFeature vector analysis:")
    for name, feature_dict in features.items():
        print(f"\n{name}:")
        if 'syntactic_features' in feature_dict:
            syntactic = feature_dict['syntactic_features']
            print(f"  Syntactic features: {len(syntactic)}")
            
            # Convert to numpy array for analysis
            syntactic_values = np.array(list(syntactic.values()))
            print(f"  Min value: {np.min(syntactic_values):.4f}")
            print(f"  Max value: {np.max(syntactic_values):.4f}")
            print(f"  Mean value: {np.mean(syntactic_values):.4f}")
            print(f"  Std dev: {np.std(syntactic_values):.4f}")
            print(f"  Non-zero features: {np.sum(syntactic_values != 0)}")
            
            # Print the actual values
            print(f"  Values: {syntactic_values}")
    
    # Test similarity calculation
    print("\nTesting similarity calculation...")
    calc = SimilarityCalculator()
    calc.weights = {
        'vocabulary': 0.0,
        'sentence': 0.0,
        'transitions': 0.0,
        'ngrams': 0.0,
        'syntactic': 1.0
    }
    
    # Calculate feature vectors
    feature_vectors = {}
    for name in test_texts.keys():
        feature_vectors[name] = calc.calculate_feature_vector(features[name])
        print(f"\n{name} full feature vector:")
        print(f"  Shape: {feature_vectors[name].shape}")
        print(f"  Values: {feature_vectors[name]}")
        
        # Extract just the syntactic part (last 20 elements)
        syntactic_part = feature_vectors[name][-20:]
        print(f"  Syntactic part: {syntactic_part}")
        print(f"  Syntactic magnitude: {np.linalg.norm(syntactic_part):.4f}")
    
    # Calculate pairwise similarities
    print("\nPairwise similarities:")
    names = list(test_texts.keys())
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i < j:
                # Raw cosine similarity
                vec1 = feature_vectors[name1]
                vec2 = feature_vectors[name2]
                raw_cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                
                # Scaled similarity
                scaled_sim = calc._cosine_similarity(vec1, vec2)
                
                print(f"  {name1} vs {name2}:")
                print(f"    Raw cosine: {raw_cosine:.4f}")
                print(f"    Scaled sim: {scaled_sim:.4f}")

if __name__ == "__main__":
    debug_syntactic_features() 