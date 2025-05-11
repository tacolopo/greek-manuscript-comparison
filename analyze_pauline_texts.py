#!/usr/bin/env python3
"""
Script to compare only Pauline texts with the updated NLP features.
"""

import os
import sys
import glob
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from src.multi_comparison import MultipleManuscriptComparison

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
    # Define valid Pauline book codes
    VALID_BOOKS = r'(?:ROM|1CO|2CO|GAL|EPH|PHP|COL|1TH|2TH|1TI|2TI|TIT|PHM)'
    
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
    # Define directories
    pauline_dir = os.path.join("data", "Paul Texts")
    output_dir = "pauline_analysis"
    vis_dir = "pauline_visualizations"
    
    try:
        # Get all chapter files from Pauline directory
        pauline_files = glob.glob(os.path.join(pauline_dir, "*.txt"))
        
        # Filter out non-Greek text files (like signature files)
        pauline_files = [f for f in pauline_files if "grcsbl" in f]
        
        if not pauline_files:
            print("Error: No Pauline chapter files found")
            return 1
            
        print(f"Found {len(pauline_files)} Pauline chapter files")
        
        # Group chapters by book and combine them
        print("\nGrouping and combining chapters by book:")
        combined_books = group_and_combine_books(pauline_files)
        print(f"\nSuccessfully processed {len(combined_books)} complete Pauline books")
        
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
            "PHM": "Philemon"
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
            use_advanced_nlp=True,
            output_dir=output_dir,
            visualizations_dir=vis_dir
        )
        
        # Run comparison with automatic determination of clusters
        print("\nRunning analysis on Pauline books with advanced NLP features...")
        
        results = comparison.compare_multiple_manuscripts(
            manuscripts=combined_books,
            display_names=display_names,
            method='hierarchical',
            use_advanced_nlp=True
        )
        
        print("\nComparison completed successfully!")
        print(f"Results saved to {output_dir} and {vis_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 