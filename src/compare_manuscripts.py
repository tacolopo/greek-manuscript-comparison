#!/usr/bin/env python3
"""
Main script for comparing Greek manuscripts.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

from preprocessing import GreekTextPreprocessor
from features import FeatureExtractor
from similarity import SimilarityCalculator
from visualization import SimilarityVisualizer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare Greek manuscripts for similarity")
    
    # Input files
    parser.add_argument('--file1', type=str, required=True, help="Path to first manuscript file")
    parser.add_argument('--file2', type=str, required=True, help="Path to second manuscript file")
    
    # Names for the manuscripts (defaults to filenames)
    parser.add_argument('--name1', type=str, help="Name of first manuscript (default: filename)")
    parser.add_argument('--name2', type=str, help="Name of second manuscript (default: filename)")
    
    # Preprocessing options
    parser.add_argument('--remove-stopwords', action='store_true', help="Remove Greek stopwords")
    parser.add_argument('--normalize-accents', action='store_true', help="Normalize Greek accents")
    parser.add_argument('--lowercase', action='store_true', help="Convert text to lowercase")
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='output', help="Output directory")
    parser.add_argument('--visualize', action='store_true', help="Generate visualizations")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Check if input files exist
    if not os.path.exists(args.file1):
        print(f"Error: File not found: {args.file1}")
        return 1
        
    if not os.path.exists(args.file2):
        print(f"Error: File not found: {args.file2}")
        return 1
    
    # Extract manuscript names
    name1 = args.name1 if args.name1 else os.path.basename(args.file1).split('.')[0]
    name2 = args.name2 if args.name2 else os.path.basename(args.file2).split('.')[0]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Initialize the preprocessor
    print(f"Initializing preprocessor...")
    preprocessor = GreekTextPreprocessor(
        remove_stopwords=args.remove_stopwords,
        normalize_accents=args.normalize_accents,
        lowercase=args.lowercase
    )
    
    # Step 2: Preprocess manuscripts
    print(f"Preprocessing first manuscript: {name1}...")
    preprocessed1 = preprocessor.preprocess_file(args.file1)
    
    print(f"Preprocessing second manuscript: {name2}...")
    preprocessed2 = preprocessor.preprocess_file(args.file2)
    
    # Step 3: Extract features
    print("Extracting linguistic features...")
    feature_extractor = FeatureExtractor()
    features1 = feature_extractor.extract_all_features(preprocessed1)
    features2 = feature_extractor.extract_all_features(preprocessed2)
    
    # Step 4: Calculate similarity
    print("Calculating similarity between manuscripts...")
    similarity_calculator = SimilarityCalculator()
    similarity_scores = similarity_calculator.calculate_overall_similarity(features1, features2)
    
    # Step 5: Display results
    overall_similarity = similarity_scores['overall_similarity']
    print(f"\nComparison Results: {name1} vs {name2}")
    print(f"Overall Similarity Score: {overall_similarity:.4f}")
    
    # Interpret overall similarity
    if overall_similarity >= 0.8:
        interpretation = "The manuscripts are extremely similar, suggesting very close relationships."
    elif overall_similarity >= 0.6:
        interpretation = "The manuscripts show high similarity, indicating strong influences."
    elif overall_similarity >= 0.4:
        interpretation = "The manuscripts have moderate similarity, with some shared characteristics."
    elif overall_similarity >= 0.2:
        interpretation = "The manuscripts show low similarity, with limited shared characteristics."
    else:
        interpretation = "The manuscripts are very different, suggesting distinct origins."
        
    print(f"Interpretation: {interpretation}")
    
    # Step 6: Visualize results if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        visualizer = SimilarityVisualizer(output_dir=args.output_dir)
        visualizer.visualize_all(similarity_scores, name1, name2)
        
        print(f"Visualizations saved to {args.output_dir}/")
    
    # Save the similarity report
    visualizer = SimilarityVisualizer(output_dir=args.output_dir)
    visualizer.create_similarity_report(
        similarity_scores, name1, name2,
        output_file=f'similarity_report_{name1}_vs_{name2}.txt'
    )
    print(f"Similarity report saved to {args.output_dir}/similarity_report_{name1}_vs_{name2}.txt")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 