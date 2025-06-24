#!/usr/bin/env python3
"""
Simple NLP Analysis Script for Greek Manuscripts

This script demonstrates how to use the streamlined Greek manuscript 
NLP analysis system for preprocessing, feature extraction, similarity 
calculation, clustering, and visualization.
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
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                # Use filename without extension as manuscript name
                name = os.path.splitext(file)[0]
                manuscripts[name] = file_path
    
    return manuscripts

def main():
    """Run the NLP analysis workflow."""
    print("Greek Manuscript NLP Analysis")
    print("=" * 40)
    
    # Configuration
    data_dir = "data"  # Directory containing manuscript text files
    output_dir = "nlp_analysis_output"
    visualizations_dir = "nlp_visualizations"
    
    # Collect manuscripts
    print(f"Collecting manuscripts from {data_dir}...")
    manuscripts = collect_manuscripts(data_dir)
    
    if not manuscripts:
        print(f"No text files found in {data_dir}")
        print("Please add some .txt files containing Greek text to analyze.")
        return
    
    print(f"Found {len(manuscripts)} manuscript files:")
    for name in sorted(manuscripts.keys())[:10]:  # Show first 10
        print(f"  - {name}")
    if len(manuscripts) > 10:
        print(f"  ... and {len(manuscripts) - 10} more")
    
    # Initialize the comparison system
    print("\nInitializing NLP analysis system...")
    comparison = MultipleManuscriptComparison(
        output_dir=output_dir,
        visualizations_dir=visualizations_dir
    )
    
    # Optional: Create display names for better visualization labels
    display_names = {}
    for name in manuscripts.keys():
        # Extract book name if it follows the pattern "grcsbl_075_ROM_01_read"
        parts = name.split('_')
        if len(parts) >= 3:
            book_code = parts[2]
            display_names[name] = book_code
        else:
            display_names[name] = name
    
    # Run the complete analysis workflow
    print("\nRunning NLP analysis...")
    try:
        results = comparison.compare_multiple_manuscripts(
            manuscripts=manuscripts,
            display_names=display_names,
            method='hierarchical',  # Use hierarchical clustering
            n_clusters=None  # Auto-determine optimal number of clusters
        )
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"Visualizations saved to: {visualizations_dir}")
        
        # Print summary
        clustering_result = results['clustering_result']
        print(f"\nSummary:")
        print(f"- Analyzed {len(manuscripts)} manuscripts")
        print(f"- Found {clustering_result['n_clusters']} clusters")
        print(f"- Silhouette score: {clustering_result['silhouette_score']:.3f}")
        
        # Print cluster assignments
        print(f"\nCluster assignments:")
        cluster_dict = {}
        for i, (name, label) in enumerate(zip(clustering_result['manuscript_names'], 
                                            clustering_result['cluster_labels'])):
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(display_names.get(name, name))
        
        for cluster_id in sorted(cluster_dict.keys()):
            members = cluster_dict[cluster_id]
            print(f"  Cluster {cluster_id}: {', '.join(members)}")
        
        print(f"\nGenerated files:")
        print(f"  - Analysis report: {results['report_file']}")
        print(f"  - Similarity matrix: {results['similarity_file']}")
        for viz_type, viz_file in results['visualization_files'].items():
            print(f"  - {viz_type.upper()} plot: {viz_file}")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 