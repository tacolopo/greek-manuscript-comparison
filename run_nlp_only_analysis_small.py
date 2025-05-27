#!/usr/bin/env python3
"""
Small version of NLP-only analysis for debugging.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from src.multi_comparison import MultipleManuscriptComparison
from src.similarity import SimilarityCalculator

def main():
    """Main function to run a small NLP-only analysis."""
    print("Starting small NLP-only analysis for debugging...")
    
    # Set up directories
    base_output_dir = "exact_cleaned_analysis"
    nlp_output_dir = os.path.join(base_output_dir, "nlp_only")
    nlp_viz_dir = os.path.join(nlp_output_dir, "visualizations")
    
    # Create directories
    os.makedirs(nlp_output_dir, exist_ok=True)
    os.makedirs(nlp_viz_dir, exist_ok=True)
    
    try:
        # Use just a few simple texts for testing
        all_texts = {
            'ROM': 'καὶ εἶπεν ὁ θεὸς γενηθήτω φῶς καὶ ἐγένετο φῶς καὶ εἶδεν ὁ θεὸς τὸ φῶς ὅτι καλόν',
            'GAL': 'ἐν ἀρχῇ ἦν ὁ λόγος καὶ ὁ λόγος ἦν πρὸς τὸν θεόν καὶ θεὸς ἦν ὁ λόγος',
            'EPH': 'οὗτος ἦν ἐν ἀρχῇ πρὸς τὸν θεόν πάντα δι αὐτοῦ ἐγένετο καὶ χωρὶς αὐτοῦ'
        }
        
        print(f"Total texts for analysis: {len(all_texts)}")
        
        # Set up NLP-only configuration
        nlp_config = {
            'name': 'nlp_only',
            'description': 'Only advanced NLP/syntactic features (without punctuation_ratio)',
            'weights': {
                'vocabulary': 0.0,
                'sentence': 0.0,
                'transitions': 0.0,
                'ngrams': 0.0,
                'syntactic': 1.0
            }
        }
        
        print(f"Running analysis with {nlp_config['name']} configuration:")
        print(f"  - {nlp_config['description']}")
        print(f"  - Weights: {nlp_config['weights']}")
        
        # Initialize comparison object
        comparison = MultipleManuscriptComparison(
            use_advanced_nlp=True,
            output_dir=nlp_output_dir,
            visualizations_dir=nlp_viz_dir
        )
        
        # Set custom weights in the similarity calculator
        custom_calculator = SimilarityCalculator()
        custom_calculator.weights = nlp_config['weights']
        comparison.similarity_calculator = custom_calculator
        
        print(f"Using NLP-only weights: {custom_calculator.weights}")
        
        # Run the comparison
        print(f"Processing manuscripts and extracting features...")
        result = comparison.compare_multiple_manuscripts(
            manuscripts=all_texts,
            display_names=all_texts,  # Use same names
            method='hierarchical',
            n_clusters=2,
            use_advanced_nlp=True
        )
        
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 