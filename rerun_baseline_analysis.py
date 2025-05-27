#!/usr/bin/env python3
"""
Script to rerun only the baseline analysis with equal weights (0.2 for each feature)
and regenerate the visualizations from the new data.
"""

import os
import sys
import shutil
import traceback
import glob
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict

# Import from the main script
from src.multi_comparison import MultipleManuscriptComparison
from src.similarity import SimilarityCalculator
from run_full_greek_analysis import (
    parse_nt_filename, combine_chapter_texts,
    group_and_combine_books, get_display_names, load_julian_letters
)

def main():
    """Main function to rerun the baseline analysis."""
    # Create custom arguments
    parser = argparse.ArgumentParser(description="Run baseline Greek manuscript analysis")
    
    parser.add_argument('--method', type=str, choices=['hierarchical', 'kmeans', 'dbscan'],
                      default='hierarchical', help="Clustering method (default: hierarchical)")
    parser.add_argument('--clusters', type=int, default=5, 
                      help="Number of clusters to use (default: 5)")
    parser.add_argument('--advanced-nlp', action='store_true', default=True,
                      help="Use advanced NLP features (default: True)")
    parser.add_argument('--output-dir', type=str, default='full_greek_analysis/baseline',
                      help="Output directory for baseline analysis (default: full_greek_analysis/baseline)")
    parser.add_argument('--viz-dir', type=str, default='full_greek_visualizations',
                      help="Directory for visualizations (default: full_greek_visualizations)")
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.viz_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)
    
    # Initialize the comparison object with correct directories
    comparison = MultipleManuscriptComparison(
        use_advanced_nlp=args.advanced_nlp,
        output_dir=args.output_dir,
        visualizations_dir=os.path.join(args.output_dir, "visualizations")
    )
    
    # Define the weights explicitly to ensure we're using 0.2 for each
    baseline_weights = {
        'vocabulary': 0.2,
        'sentence': 0.2,
        'transitions': 0.2,
        'ngrams': 0.2,
        'syntactic': 0.2
    }
    
    # Set weights on the similarity calculator
    comparison.similarity_calculator.set_weights(baseline_weights)
    print(f"Using baseline weights: {baseline_weights}")
    
    try:
        print("Loading and preparing manuscripts...")
        
        # Load and combine Non-Pauline texts (by chapter -> book)
        print("Loading and combining Non-Pauline texts...")
        non_pauline_dir = os.path.join("data", "Non-Pauline Texts")
        non_pauline_chapters = glob.glob(os.path.join(non_pauline_dir, "*.txt"))
        non_pauline_books = group_and_combine_books(non_pauline_chapters)
        
        # Load and combine Pauline texts (by chapter -> book)
        print("Loading and combining Pauline texts...")
        pauline_dir = os.path.join("data", "Paul Texts")
        pauline_chapters = glob.glob(os.path.join(pauline_dir, "*.txt"))
        pauline_books = group_and_combine_books(pauline_chapters)
        
        # Load Julian letters
        print("Loading Julian letters...")
        julian_dir = os.path.join("data", "Julian")
        julian_letters = load_julian_letters(julian_dir)
        
        # Combine all texts into a single dictionary
        all_texts = {}
        all_texts.update(julian_letters)
        all_texts.update(non_pauline_books)
        all_texts.update(pauline_books)
        
        print(f"\nTotal texts for analysis: {len(all_texts)}")
        print(f"  - Julian letters: {len(julian_letters)}")
        print(f"  - Non-Pauline books: {len(non_pauline_books)}")
        print(f"  - Pauline books: {len(pauline_books)}")
        
        # Get display names for manuscripts
        display_names = get_display_names()
        
        # Run the comparison
        print("\nRunning baseline analysis with equal weights (0.2 each)...")
        
        # Process the texts and calculate similarity
        manuscript_texts = {}
        for name, text in all_texts.items():
            manuscript_texts[name] = text
            
        # Print feature vector sizes for debugging
        print(f"Total manuscripts: {len(manuscript_texts)}")
        
        # Explicitly get features without calculating similarity to see feature values
        if manuscript_texts:
            first_ms_name = list(manuscript_texts.keys())[0]
            print(f"Extracting features for sample manuscript: {first_ms_name}")
            
            # Use the standalone SimilarityCalculator to see detailed debug output
            print("Setting up standalone similarity calculator for debugging...")
            standalone_calc = SimilarityCalculator()
            standalone_calc.set_weights(baseline_weights)
        
        # Calculate full similarity matrix
        results = comparison.compare_multiple_manuscripts(
            manuscripts=manuscript_texts,
            display_names=display_names,
            method=args.method,
            n_clusters=args.clusters
        )
        
        # Extract the similarity matrix from results
        similarity_df = results['similarity_matrix']
        
        # Save the similarity matrix
        similarity_matrix_path = os.path.join(args.output_dir, "similarity_matrix.csv")
        similarity_df.to_csv(similarity_matrix_path)
        print(f"Saved similarity matrix to {similarity_matrix_path}")
        
        # Save the clustering results
        clustering_report_path = os.path.join(args.output_dir, "clustering_report.txt")
        with open(clustering_report_path, 'w') as f:
            f.write(results['report'])
        print(f"Saved clustering report to {clustering_report_path}")
        
        # Visualize the results
        print("\nGenerating visualizations...")
        
        # First, cluster the manuscripts
        clustering_result = comparison.cluster_manuscripts(
            similarity_df=similarity_df,
            n_clusters=args.clusters,
            method=args.method
        )
        
        # Generate visualizations
        visualizations = comparison.generate_visualizations(
            clustering_result=clustering_result,
            similarity_df=similarity_df,
            threshold=0.5
        )
        
        # Copy the visualizations to the main visualization directory
        print("\nCopying visualizations to main directory...")
        src_viz_dir = os.path.join(args.output_dir, "visualizations")
        for viz_file in glob.glob(os.path.join(src_viz_dir, "*.png")):
            file_name = os.path.basename(viz_file)
            prefix = "baseline_"
            dest_path = os.path.join(args.viz_dir, prefix + file_name)
            shutil.copy(viz_file, dest_path)
            print(f"Copied {file_name} to {dest_path}")
            
        print("\nBaseline analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during baseline analysis: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 