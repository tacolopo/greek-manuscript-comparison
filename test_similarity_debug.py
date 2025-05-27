#!/usr/bin/env python3
"""
Debug similarity calculation issues.
"""

import numpy as np
import sys
sys.path.append('src')

from src.similarity import SimilarityCalculator

def test_cosine_similarity():
    """Test the cosine similarity calculation."""
    
    calc = SimilarityCalculator()
    
    # Set NLP-only weights
    calc.weights = {
        'vocabulary': 0.0,
        'sentence': 0.0,
        'transitions': 0.0,
        'ngrams': 0.0,
        'syntactic': 1.0
    }
    
    # Test with very different vectors
    vec1 = np.array([1.0, 0.0, 0.0, 0.0])  # All nouns
    vec2 = np.array([0.0, 1.0, 0.0, 0.0])  # All verbs
    
    # Raw cosine similarity should be 0 (orthogonal)
    raw_cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    print(f"Raw cosine similarity (orthogonal vectors): {raw_cosine}")
    
    # Scaled similarity using the current method
    scaled_sim = calc._cosine_similarity(vec1, vec2)
    print(f"Scaled similarity (should be low): {scaled_sim}")
    
    # Test with identical vectors
    vec3 = np.array([0.5, 0.3, 0.2, 0.0])
    vec4 = np.array([0.5, 0.3, 0.2, 0.0])
    
    raw_cosine_identical = np.dot(vec3, vec4) / (np.linalg.norm(vec3) * np.linalg.norm(vec4))
    print(f"Raw cosine similarity (identical vectors): {raw_cosine_identical}")
    
    scaled_sim_identical = calc._cosine_similarity(vec3, vec4)
    print(f"Scaled similarity (should be 1.0): {scaled_sim_identical}")
    
    # Test with similar but not identical vectors
    vec5 = np.array([0.5, 0.3, 0.2, 0.0])
    vec6 = np.array([0.4, 0.35, 0.25, 0.0])
    
    raw_cosine_similar = np.dot(vec5, vec6) / (np.linalg.norm(vec5) * np.linalg.norm(vec6))
    print(f"Raw cosine similarity (similar vectors): {raw_cosine_similar}")
    
    scaled_sim_similar = calc._cosine_similarity(vec5, vec6)
    print(f"Scaled similarity (should be moderate): {scaled_sim_similar}")

if __name__ == "__main__":
    test_cosine_similarity() 