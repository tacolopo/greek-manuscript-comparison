#!/usr/bin/env python3
"""
Enhanced NLP Analysis Script for Greek Manuscripts

This script performs data-driven clustering analysis of Greek manuscripts
using advanced NLP features and multiple clustering algorithms.
No assumptions are made about authorship - the analysis is purely data-driven.
"""

import os
import glob
from src import MultipleManuscriptComparison

def collect_manuscripts(data_dir: str) -> dict:
    """
    Collect manuscript files from the data directory.
    
    Args:
        data_dir: Path to data directory containing text files
        
    Returns:
        Dictionary mapping manuscript names to file paths
    """
    manuscripts = {}
    
    # Look for text files in subdirectories
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.txt') and '_read' in file:
                full_path = os.path.join(root, file)
                # Create a clean name from the file
                name = file.replace('_read.txt', '').replace('grcsbl_', '')
                manuscripts[name] = full_path
    
    return manuscripts

def main():
    """Main analysis function."""
    print("=== Enhanced Greek Manuscript NLP Clustering Analysis ===")
    print("This analysis makes NO assumptions about authorship.")
    print("Clustering is purely data-driven based on linguistic features.\n")
    
    # Collect manuscripts
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        return
    
    print("Collecting manuscripts...")
    manuscripts = collect_manuscripts(data_dir)
    
    if not manuscripts:
        print("No manuscripts found! Please check the data directory.")
        return
    
    print(f"Found {len(manuscripts)} manuscripts")
    
    # Show first few manuscripts
    print("\nFirst 10 manuscripts:")
    for i, name in enumerate(list(manuscripts.keys())[:10]):
        print(f"  {i+1}. {name}")
    
    if len(manuscripts) > 10:
        print(f"  ... and {len(manuscripts) - 10} more")
    
    # Initialize the enhanced comparison system
    print("\nInitializing enhanced NLP analysis system...")
    
    try:
        comparator = MultipleManuscriptComparison(use_advanced_nlp=True)
        
        # Prepare manuscript paths and names
        manuscript_paths = list(manuscripts.values())
        manuscript_names = list(manuscripts.keys())
        
        # Run the complete analysis
        print("\nRunning complete enhanced clustering analysis...")
        print("This may take several minutes depending on the number of manuscripts...")
        
        results = comparator.run_complete_analysis(
            manuscript_paths=manuscript_paths,
            manuscript_names=manuscript_names,
            output_dir="enhanced_clustering_results"
        )
        
        print("\n" + "="*60)
        print("ENHANCED ANALYSIS COMPLETE!")
        print("="*60)
        print(f"Results saved to: enhanced_clustering_results/")
        print("\nThe analysis used:")
        print("✓ Advanced vocabulary richness metrics")
        print("✓ Sentence complexity analysis")
        print("✓ Function word usage patterns")
        print("✓ Morphological diversity measures")
        print("✓ Semantic embeddings (when available)")
        print("✓ Multiple clustering algorithms (K-Means, Hierarchical, GMM, Spectral, DBSCAN)")
        print("✓ Comprehensive validation metrics")
        print("✓ Feature selection and dimensionality reduction")
        print("\nCheck the report and visualizations for detailed results!")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        print("This might be due to missing dependencies or data issues.")
        print("Try running with fewer manuscripts or check the requirements.")
        
        # Fallback: basic analysis
        print("\nAttempting basic analysis without advanced NLP...")
        try:
            comparator = MultipleManuscriptComparison(use_advanced_nlp=False)
            results = comparator.run_complete_analysis(
                manuscript_paths=manuscript_paths[:20],  # Limit to first 20
                manuscript_names=manuscript_names[:20],
                output_dir="basic_clustering_results"
            )
            print("Basic analysis completed successfully!")
        except Exception as e2:
            print(f"Basic analysis also failed: {e2}")

if __name__ == "__main__":
    main() 