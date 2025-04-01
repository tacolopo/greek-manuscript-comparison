#!/usr/bin/env python3
"""
Script to compare multiple Greek manuscripts.
"""

import os
import sys
import argparse
import glob
from typing import List, Optional

from src.multi_comparison import MultipleManuscriptComparison


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare multiple Greek manuscripts")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--files', type=str, nargs='+', help="List of manuscript files to compare")
    input_group.add_argument('--dir', type=str, help="Directory containing manuscript files to compare")
    input_group.add_argument('--pattern', type=str, help="Glob pattern to match manuscript files")
    input_group.add_argument('--sample', action='store_true', help="Use sample manuscripts")
    
    # File pattern/extension
    parser.add_argument('--ext', type=str, default='.txt', help="File extension for manuscripts (default: .txt)")
    
    # Names for the manuscripts
    parser.add_argument('--names', type=str, nargs='+', help="Names for the manuscripts (must match number of files)")
    
    # Clustering options
    parser.add_argument('--clusters', type=int, default=3, help="Number of clusters to create (default: 3)")
    parser.add_argument('--method', type=str, choices=['kmeans', 'hierarchical', 'dbscan'], 
                       default='hierarchical', help="Clustering method (default: hierarchical)")
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help="Similarity threshold for network visualization (default: 0.5)")
    
    # Advanced NLP options
    parser.add_argument('--advanced-nlp', action='store_true', help="Use advanced NLP features")
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='output', help="Output directory (default: output)")
    parser.add_argument('--vis-dir', type=str, default='visualizations', 
                       help="Visualizations directory (default: visualizations)")
    
    return parser.parse_args()


def get_manuscript_files(args) -> List[str]:
    """
    Get list of manuscript files based on command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of file paths
    """
    if args.files:
        # Direct list of files
        return args.files
    
    elif args.dir:
        # All files with specified extension in directory
        if not os.path.isdir(args.dir):
            raise ValueError(f"Directory not found: {args.dir}")
            
        return sorted(glob.glob(os.path.join(args.dir, f"*{args.ext}")))
    
    elif args.pattern:
        # Files matching glob pattern
        return sorted(glob.glob(args.pattern))
    
    elif args.sample:
        # Use sample manuscripts
        sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        sample_files = sorted(glob.glob(os.path.join(sample_dir, f"sample_text*{args.ext}")))
        
        if not sample_files:
            raise ValueError(f"No sample files found in {sample_dir}")
            
        return sample_files
    
    else:
        raise ValueError("No input files specified")


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Get manuscript files
        manuscript_files = get_manuscript_files(args)
        
        if not manuscript_files:
            print("Error: No manuscript files found")
            return 1
            
        print(f"Found {len(manuscript_files)} manuscript files:")
        for file in manuscript_files:
            print(f"  - {file}")
            
        # Validate names if provided
        manuscript_names = args.names
        if manuscript_names and len(manuscript_names) != len(manuscript_files):
            print(f"Error: Number of names ({len(manuscript_names)}) does not match number of files ({len(manuscript_files)})")
            return 1
            
        # Initialize comparison
        comparison = MultipleManuscriptComparison(
            use_advanced_nlp=args.advanced_nlp,
            output_dir=args.output_dir,
            visualizations_dir=args.vis_dir
        )
        
        # Run comparison
        results = comparison.compare_multiple_manuscripts(
            manuscript_paths=manuscript_files,
            manuscript_names=manuscript_names,
            n_clusters=args.clusters,
            clustering_method=args.method,
            similarity_threshold=args.threshold
        )
        
        print("\nComparison completed successfully!")
        print(f"Results saved to {args.output_dir} and {args.vis_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 