#!/usr/bin/env python3
"""
Script to run only the NLP-only analysis for the exact cleaned data.
This will recreate the nlp_only folder within exact_cleaned_analysis.
"""

import os
import sys
import glob
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from src.multi_comparison import MultipleManuscriptComparison
from src.similarity import SimilarityCalculator

def parse_nt_filename(filename):
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
    """Combine chapter files into complete book texts."""
    book_texts = {}
    
    for chapter_file in chapter_files:
        try:
            manuscript_id, book_code, chapter_num, filename = parse_nt_filename(chapter_file)
            
            # Read the chapter text
            with open(chapter_file, 'r', encoding='utf-8') as f:
                chapter_text = f.read().strip()
            
            # Initialize book if not exists
            if book_code not in book_texts:
                book_texts[book_code] = ""
            
            # Append chapter text
            book_texts[book_code] += " " + chapter_text
            
        except Exception as e:
            print(f"Warning: Error processing {chapter_file}: {e}")
    
    return book_texts

def group_and_combine_books(chapter_files):
    """Group chapter files by book and combine them."""
    print(f"Processing {len(chapter_files)} chapter files...")
    
    # Combine chapters into books
    book_texts = combine_chapter_texts(chapter_files)
    
    print(f"Combined into {len(book_texts)} complete books:")
    for book_code in sorted(book_texts.keys()):
        word_count = len(book_texts[book_code].split())
        print(f"  - {book_code}: {word_count} words")
    
    return book_texts

def get_display_names():
    """Get display names for books."""
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

def load_julian_letters(julian_dir):
    """Load Julian letters as whole documents."""
    letters = {}
    
    for letter_path in glob.glob(os.path.join(julian_dir, "*.txt")):
        letter_name = os.path.basename(letter_path)
        letter_name = os.path.splitext(letter_name)[0]  # Remove extension
        
        try:
            with open(letter_path, 'r', encoding='utf-8') as f:
                letter_text = f.read().strip()
                letters[letter_name] = letter_text
                print(f"  - Loaded Julian letter: {letter_name}")
        except Exception as e:
            print(f"Warning: Error reading {letter_path}: {e}")
    
    return letters

def main():
    """Main function to run the NLP-only analysis."""
    print("Starting NLP-only analysis for exact cleaned data...")
    
    # Set up directories
    base_output_dir = "exact_cleaned_analysis"
    nlp_output_dir = os.path.join(base_output_dir, "nlp_only")
    nlp_viz_dir = os.path.join(nlp_output_dir, "visualizations")
    
    # Create directories
    os.makedirs(nlp_output_dir, exist_ok=True)
    os.makedirs(nlp_viz_dir, exist_ok=True)
    
    print(f"Output directory: {nlp_output_dir}")
    print(f"Visualizations: {nlp_viz_dir}")
    
    try:
        # Load Julian letters (whole letters)
        print("\nLoading Julian letters...")
        julian_dir = os.path.join("data", "Julian")
        julian_letters = load_julian_letters(julian_dir)
        
        # Load and combine Non-Pauline texts (by chapter -> book)
        print("\nLoading and combining Non-Pauline texts...")
        non_pauline_dir = os.path.join("data", "Non-Pauline Texts")
        non_pauline_chapters = glob.glob(os.path.join(non_pauline_dir, "*.txt"))
        non_pauline_books = group_and_combine_books(non_pauline_chapters)
        
        # Load and combine Pauline texts (by chapter -> book)
        print("\nLoading and combining Pauline texts...")
        pauline_dir = os.path.join("data", "Cleaned_Paul_Texts")
        pauline_chapters = glob.glob(os.path.join(pauline_dir, "*.txt"))
        pauline_books = group_and_combine_books(pauline_chapters)
        
        # Combine all texts into a single dictionary
        all_texts = {}
        all_texts.update(julian_letters)
        all_texts.update(non_pauline_books)
        all_texts.update(pauline_books)
        
        print(f"\nTotal texts for analysis: {len(all_texts)}")
        print(f"  - Julian letters: {len(julian_letters)}")
        print(f"  - Non-Pauline books: {len(non_pauline_books)}")
        print(f"  - Pauline books: {len(pauline_books)}")
        
        # Get display names for better visualization
        display_names = get_display_names()
        
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
        
        print(f"\nRunning analysis with {nlp_config['name']} configuration:")
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
            display_names=display_names,
            method='hierarchical',
            n_clusters=8,
            use_advanced_nlp=True
        )
        
        # Extract similarity matrix from result
        if 'similarity_df' in result:
            sim_matrix = result['similarity_df']
        elif 'similarity_matrix' in result:
            sim_matrix = result['similarity_matrix']
        else:
            sim_matrix = result.get('clustering_result', {}).get('similarity_matrix', None)
            if sim_matrix is None:
                print(f"Error: Could not find similarity matrix in result.")
                return
        
        # Get clustering results
        clustering_result = result.get('clustering_result', {})
        
        # Save the similarity matrix CSV
        matrix_csv_path = os.path.join(nlp_output_dir, 'similarity_matrix.csv')
        if hasattr(sim_matrix, 'to_csv'):
            sim_matrix.to_csv(matrix_csv_path)
            print(f"Similarity matrix saved to: {matrix_csv_path}")
        
        # Save similarity matrix pickle
        matrix_pkl_path = os.path.join(nlp_output_dir, 'similarity_matrix.pkl')
        with open(matrix_pkl_path, 'wb') as f:
            pickle.dump(sim_matrix, f)
        
        # Save clustering result pickle
        result_pkl_path = os.path.join(nlp_output_dir, 'clustering_result.pkl')
        with open(result_pkl_path, 'wb') as f:
            pickle.dump(clustering_result, f)
        
        # Create clustering report
        report_lines = [
            f"NLP-Only Analysis Report (Without Punctuation Ratio)",
            f"=" * 50,
            f"Configuration: {nlp_config['description']}",
            f"Method: hierarchical",
            f"Number of Clusters: 8",
            f"Weight Configuration: {nlp_config['weights']}",
            f"",
            f"Syntactic features used (20 features):",
            f"- Basic POS ratios: noun, verb, adj, adv, function_word, pronoun, conjunction, particle, interjection, numeral",
            f"- Complexity measures: tag_diversity, tag_entropy, noun_verb_ratio", 
            f"- Pattern analysis: noun_after_verb_ratio, adj_before_noun_ratio, adv_before_verb_ratio",
            f"- Transition probabilities: verb_to_noun_prob, noun_to_verb_prob, noun_to_adj_prob, adj_to_noun_prob",
            f"",
            f"Note: punctuation_ratio has been removed from the analysis.",
            f"",
            f"Results are available in the visualization files and similarity matrix."
        ]
        
        # Save clustering report
        report_path = os.path.join(nlp_output_dir, 'clustering_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Generate visualizations
        try:
            comparison.generate_visualizations(
                clustering_result=clustering_result,
                similarity_df=sim_matrix,
                threshold=0.5
            )
            print(f"Visualizations saved to: {nlp_viz_dir}")
        except Exception as e:
            print(f"Warning: Error generating visualizations: {e}")
        
        print(f"\nNLP-only analysis complete!")
        print(f"Results saved to: {nlp_output_dir}")
        print(f"Files created:")
        print(f"  - similarity_matrix.csv")
        print(f"  - similarity_matrix.pkl") 
        print(f"  - clustering_result.pkl")
        print(f"  - clustering_report.txt")
        print(f"  - visualizations/ (directory)")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 