#!/usr/bin/env python3
"""
Script to debug and fix the NLP-only similarity calculation.
This is a simplified version of iterate_similarity_weights.py focused
just on the NLP-only configuration.
"""

import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from src.multi_comparison import MultipleManuscriptComparison
from src.similarity import SimilarityCalculator
from src.advanced_nlp import AdvancedGreekProcessor

def parse_nt_filename(filename: str):
    """
    Parse a chapter filename to extract manuscript info.
    Expected format: path/to/grcsbl_XXX_BBB_CC_read.txt
    where XXX is the manuscript number, BBB is the book code, CC is the chapter
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

def combine_chapter_texts(chapter_files):
    """
    Combine multiple chapter files into a single text.
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

def group_and_combine_books(chapter_files):
    """
    Group chapter files by book and combine each book's chapters into a single text.
    """
    # First, group chapters by book
    book_chapters = defaultdict(list)
    
    for file_path in chapter_files:
        try:
            manuscript_id, book_code, _, _ = parse_nt_filename(file_path)
            
            # Use just the book code without the manuscript ID
            # This ensures we combine all chapters of the same book together
            book_name = book_code
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

def get_book_display_names():
    """Get dictionary mapping book codes to display names."""
    return {
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

def test_advanced_nlp_processor():
    """
    Directly test the advanced NLP processor to see if it's generating
    non-zero syntactic features.
    """
    print("\n=== Testing Advanced NLP Processor directly ===")
    # Create the processor
    processor = AdvancedGreekProcessor()
    
    # Get a sample text from a file - use a file we know exists
    sample_file = os.path.join("data", "Paul Texts", "grcsbl_076_1CO_02_read.txt")
    with open(sample_file, 'r', encoding='utf-8') as f:
        sample_text = f.read()
    
    print(f"Sample text length: {len(sample_text)} characters")
    
    # Process the document - this returns a dictionary containing processed features
    result = processor.process_document(sample_text)
    
    # Access POS tags directly from the result dictionary
    pos_tags = result['pos_tags']
    print(f"Extracted {len(pos_tags)} POS tags")
    print(f"First 10 POS tags: {pos_tags[:10]}")
    
    # Print the tag distribution to understand what tags are in the text
    tag_counts = {}
    for tag in pos_tags:
        tag_str = str(tag).lower()
        tag_counts[tag_str] = tag_counts.get(tag_str, 0) + 1
    print("\nPOS tag distribution:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {tag}: {count}")
    
    # Extract syntactic features
    syntactic_features = processor.extract_syntactic_features(pos_tags)
    
    # Print the features
    print("\nSyntactic features from direct extraction:")
    for key, value in syntactic_features.items():
        print(f"  {key}: {value}")
    
    # Check for non-zero values
    non_zero_features = {k: v for k, v in syntactic_features.items() if v != 0}
    print(f"\nNon-zero features: {len(non_zero_features)}/{len(syntactic_features)}")
    
    return syntactic_features

def run_nlp_only_config():
    """Run and debug the NLP-only configuration."""
    # Test the advanced NLP processor directly
    syntactic_features = test_advanced_nlp_processor()
    
    # Set up the output directories
    base_output_dir = "debug_nlp_only"
    output_dir = os.path.join(base_output_dir, "nlp_only")
    vis_dir = os.path.join(base_output_dir, "nlp_only_vis")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Define the NLP-only weight configuration
    nlp_only_weights = {
        'vocabulary': 0.0,
        'sentence': 0.0,
        'transitions': 0.0,
        'ngrams': 0.0,
        'syntactic': 1.0
    }
    
    # Create a custom SimilarityCalculator with NLP-only weights
    custom_calculator = SimilarityCalculator()
    custom_calculator.weights = nlp_only_weights
    
    # Initialize comparison object
    comparison = MultipleManuscriptComparison(
        use_advanced_nlp=True,
        output_dir=output_dir,
        visualizations_dir=vis_dir,
        similarity_calculator=custom_calculator  # Use our custom calculator
    )
    
    # Sample a small set of books for faster debugging
    print("\n=== Testing with a small set of books ===")
    
    # Get all chapter files for just a few books
    pauline_dir = os.path.join("data", "Paul Texts")
    
    # Just use chapters from ROM, 1CO, GAL
    sample_books = ['ROM', '1CO', 'GAL']
    sample_files = []
    
    for book in sample_books:
        pattern = os.path.join(pauline_dir, f"*_{book}_*_read.txt")
        book_files = glob.glob(pattern)
        sample_files.extend(book_files)
    
    print(f"Found {len(sample_files)} chapter files for {', '.join(sample_books)}")
    
    # Group and combine chapters
    print("\nGrouping and combining chapters by book:")
    combined_books = group_and_combine_books(sample_files)
    print(f"\nSuccessfully processed {len(combined_books)} sample books")
    
    # Get display names
    book_display_names = get_book_display_names()
    display_names = {}
    for book_key in combined_books.keys():
        if book_key in book_display_names:
            display_names[book_key] = book_display_names[book_key]
        else:
            display_names[book_key] = book_key
    
    # Run the full comparison process with the NLP-only weights
    print("\nRunning the full comparison process with NLP-only weights")
    results = comparison.compare_multiple_manuscripts(
        manuscripts=combined_books,
        display_names=display_names,
        method='hierarchical',
        n_clusters=3,
        use_advanced_nlp=True
    )
    
    # After running the comparison process, check the similarity matrix
    sim_matrix = results.get('similarity_matrix', pd.DataFrame())
    if not sim_matrix.empty:
        # Save to CSV
        csv_path = os.path.join(output_dir, "nlp_only_similarity_matrix.csv")
        sim_matrix.to_csv(csv_path)
        print(f"Saved similarity matrix to {csv_path}")
        
        # Analyze the similarity values
        matrix_values = sim_matrix.values
        non_diag = matrix_values[~np.eye(matrix_values.shape[0], dtype=bool)]
        print("\nSimilarity matrix statistics:")
        print(f"Min: {np.min(non_diag)}")
        print(f"Max: {np.max(non_diag)}")
        print(f"Mean: {np.mean(non_diag)}")
        print(f"Number of non-zero values: {np.count_nonzero(non_diag)}/{len(non_diag)}")
    else:
        print("Error: Similarity matrix is empty")
    
    return 0

if __name__ == "__main__":
    sys.exit(run_nlp_only_config()) 