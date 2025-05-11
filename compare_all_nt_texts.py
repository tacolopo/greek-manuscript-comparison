#!/usr/bin/env python3
"""
Script to compare all New Testament texts (both Pauline and non-Pauline)
without presupposing any groupings.
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple
import glob
import re
import tempfile
from collections import defaultdict

from src.multi_comparison import MultipleManuscriptComparison

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare all New Testament texts")
    parser.add_argument('--clusters', type=int, default=None, 
                        help="Number of clusters to use (default: determine automatically)")
    parser.add_argument('--method', type=str, choices=['hierarchical', 'kmeans', 'dbscan'],
                        default='hierarchical', help="Clustering method (default: hierarchical)")
    parser.add_argument('--advanced-nlp', action='store_true', default=True,
                        help="Use advanced NLP features (default: True)")
    parser.add_argument('--output-dir', type=str, default='all_nt_output',
                        help="Output directory (default: all_nt_output)")
    parser.add_argument('--vis-dir', type=str, default='all_nt_visualizations',
                        help="Visualizations directory (default: all_nt_visualizations)")
    
    return parser.parse_args()

def parse_nt_filename(filename: str) -> Tuple[str, str, str, str]:
    """
    Parse a chapter filename to extract manuscript info.
    Expected format: path/to/grcsbl_XXX_BBB_CC_read.txt
    where XXX is the manuscript number, BBB is the book code, CC is the chapter
    
    Args:
        filename: Name of the chapter file
        
    Returns:
        Tuple of (manuscript_id, book_code, chapter_num, full_path)
    """
    # Define valid NT book codes - expanded to include all NT books
    VALID_BOOKS = r'(?:ROM|1CO|2CO|GAL|EPH|PHP|COL|1TH|2TH|1TI|2TI|TIT|PHM|' + \
                 r'ACT|JHN|1JN|2JN|3JN|1PE|2PE|JUD|REV|JAS|HEB|MAT|MRK|LUK)'
    
    # Extract components using regex - allow for full path
    pattern = rf'.*?grcsbl_(\d+)_({VALID_BOOKS})_(\d+)_read\.txt$'
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
        
    manuscript_id = match.group(1)
    book_code = match.group(2)
    chapter_num = match.group(3)
    
    return manuscript_id, book_code, chapter_num, filename

def combine_chapter_texts(chapter_files: List[str]) -> str:
    """
    Combine multiple chapter files into a single text.
    
    Args:
        chapter_files: List of chapter file paths
        
    Returns:
        Combined text from all chapters
    """
    combined_text = []
    
    for file_path in sorted(chapter_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chapter_text = f.read().strip()
                combined_text.append(chapter_text)
        except Exception as e:
            print(f"Warning: Error reading {file_path}: {e}")
            continue
    
    return "\n\n".join(combined_text)

def group_and_combine_books(chapter_files: List[str]) -> Dict[str, str]:
    """
    Group chapter files by book and combine each book's chapters into a single text.
    
    Args:
        chapter_files: List of all chapter files
        
    Returns:
        Dictionary mapping book names to their combined texts
    """
    # First, group chapters by book
    book_chapters = defaultdict(list)
    
    for file_path in chapter_files:
        try:
            manuscript_id, book_code, _, _ = parse_nt_filename(file_path)
            book_name = f"{book_code}-{manuscript_id}"
            book_chapters[book_name].append(file_path)
        except ValueError as e:
            print(f"Warning: Skipping invalid file {file_path}: {e}")
            continue
    
    # Sort chapters within each book
    for book_name in book_chapters:
        book_chapters[book_name] = sorted(book_chapters[book_name])
    
    # Now combine chapters for each book
    combined_books = {}
    for book_name, chapters in book_chapters.items():
        print(f"  - {book_name}: {len(chapters)} chapters")
        combined_books[book_name] = combine_chapter_texts(chapters)
    
    return combined_books

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Define directories
    pauline_dir = os.path.join("data", "Paul Texts")
    non_pauline_dir = os.path.join("data", "Non-Pauline NT")
    
    try:
        # Get all chapter files from both directories
        pauline_files = glob.glob(os.path.join(pauline_dir, "*.txt"))
        non_pauline_files = glob.glob(os.path.join(non_pauline_dir, "*.txt"))
        all_chapter_files = pauline_files + non_pauline_files
        
        if not all_chapter_files:
            print("Error: No chapter files found")
            return 1
            
        print(f"Found {len(all_chapter_files)} total chapter files")
        
        # Group chapters by book and combine them
        print("\nGrouping and combining chapters by book:")
        combined_books = group_and_combine_books(all_chapter_files)
        print(f"\nSuccessfully processed {len(combined_books)} complete books")
        
        # Get book names for better display
        book_display_names = {
            "ROM": "Romans",
            "1CO": "1 Corinthians",
            "2CO": "2 Corinthians",
            "GAL": "Galatians",
            "EPH": "Ephesians",
            "PHP": "Philippians",
            "COL": "Colossians",
            "1TH": "1 Thessalonians",
            "2TH": "2 Thessalonians",
            "1TI": "1 Timothy",
            "2TI": "2 Timothy",
            "TIT": "Titus",
            "PHM": "Philemon",
            "HEB": "Hebrews",
            "JAS": "James",
            "1PE": "1 Peter",
            "2PE": "2 Peter",
            "1JN": "1 John",
            "2JN": "2 John",
            "3JN": "3 John",
            "JUD": "Jude",
            "REV": "Revelation",
            "MAT": "Matthew",
            "MRK": "Mark",
            "LUK": "Luke",
            "JHN": "John",
            "ACT": "Acts"
        }
        
        # Create a mapping of book codes to display names
        display_names = {}
        for book_key in combined_books.keys():
            code = book_key.split('-')[0]
            if code in book_display_names:
                display_names[book_key] = book_display_names[code]
            else:
                display_names[book_key] = book_key
                
        # Initialize comparison with advanced NLP features
        comparison = MultipleManuscriptComparison(
            use_advanced_nlp=args.advanced_nlp,
            output_dir=args.output_dir,
            visualizations_dir=args.vis_dir
        )
        
        # Run comparison with specified or automatically determined number of clusters
        print("\nRunning analysis on whole books (not individual chapters)...")
        if args.clusters:
            print(f"Using specified number of clusters: {args.clusters}")
        else:
            print("Number of clusters will be determined automatically")
            
        results = comparison.compare_multiple_manuscripts(
            manuscripts=combined_books,
            display_names=display_names,
            method=args.method,
            n_clusters=args.clusters,
            use_advanced_nlp=args.advanced_nlp
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