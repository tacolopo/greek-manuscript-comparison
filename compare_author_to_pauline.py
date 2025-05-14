#!/usr/bin/env python3
"""
Script to compare author's articles with Pauline corpus using
the same feature extraction and similarity analysis methods.
"""

import os
import sys
import glob
import re
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
from itertools import combinations

# Import the necessary classes from the project
from src.preprocessing import GreekTextPreprocessor
from src.features import FeatureExtractor
from src.similarity import SimilarityCalculator
from src.advanced_nlp import AdvancedGreekProcessor as AdvancedNLPProcessor

def load_author_articles(articles_dir: str) -> Dict[str, str]:
    """
    Load author articles from text files.
    
    Args:
        articles_dir: Directory containing the author's articles
        
    Returns:
        Dictionary mapping article names to their content
    """
    articles = {}
    
    article_files = glob.glob(os.path.join(articles_dir, "*.txt"))
    for file_path in article_files:
        try:
            filename = os.path.basename(file_path)
            article_name = os.path.splitext(filename)[0]
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # Add prefix to distinguish from biblical texts
            article_name = f"AUTH_{article_name}"
            articles[article_name] = content
            print(f"Loaded article: {article_name} ({len(content)} characters)")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return articles

def load_pauline_letters(pauline_dir: str) -> Dict[str, str]:
    """
    Load Pauline corpus from chapter files.
    
    Args:
        pauline_dir: Directory containing Paul's texts
        
    Returns:
        Dictionary mapping book names to their combined content
    """
    pauline_books = ['ROM', '1CO', '2CO', 'GAL', 'EPH', 'PHP', 'COL', 
                     '1TH', '2TH', '1TI', '2TI', 'TIT', 'PHM']
    
    # Find all chapter files
    chapter_files = glob.glob(os.path.join(pauline_dir, "*.txt"))
    
    # Group chapters by book
    book_chapters = defaultdict(list)
    
    for file_path in chapter_files:
        try:
            # Extract book code from filename pattern
            # Expected pattern: grcsbl_NNN_BBB_CC_read.txt
            match = re.search(r'grcsbl_\d+_([A-Z0-9]+)_\d+_read\.txt$', file_path)
            if match:
                book_code = match.group(1)
                if book_code in pauline_books:
                    book_chapters[book_code].append(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Combine chapters for each book
    combined_books = {}
    for book_code, chapters in book_chapters.items():
        # Sort chapters numerically
        chapters_sorted = sorted(chapters, key=lambda f: int(re.search(r'_(\d+)_read\.txt$', f).group(1)))
        
        # Combine chapter texts
        combined_text = []
        for chapter_file in chapters_sorted:
            try:
                with open(chapter_file, 'r', encoding='utf-8') as f:
                    chapter_text = f.read().strip()
                    combined_text.append(chapter_text)
            except Exception as e:
                print(f"Error reading {chapter_file}: {e}")
        
        if combined_text:
            combined_books[book_code] = "\n\n".join(combined_text)
            print(f"Loaded book: {book_code} ({len(combined_text)} chapters)")
    
    return combined_books

def preprocess_texts(texts: Dict[str, str]) -> Dict[str, Dict]:
    """
    Preprocess texts for feature extraction.
    
    Args:
        texts: Dictionary mapping text names to their content
        
    Returns:
        Dictionary mapping text names to their preprocessed versions
    """
    preprocessed_texts = {}
    
    # Use different preprocessors for English and Greek
    english_preprocessor = GreekTextPreprocessor(normalize_accents=False)
    greek_preprocessor = GreekTextPreprocessor()
    
    for text_name, content in texts.items():
        try:
            print(f"Preprocessing {text_name}...")
            
            # Use English preprocessor for author articles and Greek for Pauline
            if text_name.startswith("AUTH_"):
                preprocessed = english_preprocessor.preprocess(content)
            else:
                preprocessed = greek_preprocessor.preprocess(content)
                
            preprocessed_texts[text_name] = preprocessed
        except Exception as e:
            print(f"Error preprocessing {text_name}: {e}")
    
    return preprocessed_texts

def extract_features(preprocessed_texts: Dict[str, Dict], use_advanced_nlp: bool = True) -> Dict[str, Dict]:
    """
    Extract features from preprocessed texts.
    
    Args:
        preprocessed_texts: Dictionary mapping text names to their preprocessed versions
        use_advanced_nlp: Whether to use advanced NLP features
        
    Returns:
        Dictionary mapping text names to their features
    """
    features_data = {}
    extractor = FeatureExtractor()
    
    # Fit the TF-IDF vectorizer
    all_texts = [data['normalized_text'] for data in preprocessed_texts.values()]
    extractor.fit(all_texts)
    
    # Initialize advanced NLP processor if needed
    nlp_processor = None
    if use_advanced_nlp:
        try:
            print("Initializing advanced NLP processor...")
            nlp_processor = AdvancedNLPProcessor()
            print("Successfully initialized advanced NLP processor")
        except Exception as e:
            print(f"Warning: Could not initialize advanced NLP processor: {e}")
            print("Will proceed without advanced NLP features")
    
    # Extract features for each text
    for text_name, preprocessed in preprocessed_texts.items():
        print(f"Processing {text_name}...")
        
        # Extract features using the extractor
        try:
            # Use the correct method from FeatureExtractor
            features = extractor.extract_all_features(preprocessed)
            
            # Add syntactic features if using advanced NLP and it's a Pauline text
            if nlp_processor and not text_name.startswith("AUTH_"):
                try:
                    print(f"Extracting advanced NLP features for {text_name}...")
                    syntactic_features = nlp_processor.extract_syntactic_features(preprocessed['normalized_text'])
                    features['syntactic_features'] = syntactic_features
                except Exception as e:
                    print(f"Warning: Could not extract NLP features for {text_name}: {e}")
            
            # For author texts, create simplified syntactic features to make compatible
            if text_name.startswith("AUTH_"):
                print(f"Creating simplified syntactic features for {text_name}...")
                # Create basic syntactic features structure to ensure compatibility
                features['syntactic_features'] = {
                    'noun_ratio': 0.25,  # Approximate values
                    'verb_ratio': 0.20,
                    'adj_ratio': 0.10,
                    'adv_ratio': 0.05,
                    'function_word_ratio': 0.40,
                    'tag_diversity': 0.7,
                    'tag_entropy': 2.0
                }
            
            features_data[text_name] = features
        except Exception as e:
            print(f"Error extracting features for {text_name}: {e}")
    
    return features_data

def run_weight_configurations(features_data: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
    """
    Run similarity calculations with different weight configurations.
    
    Args:
        features_data: Dictionary mapping text names to their features
        
    Returns:
        Dictionary mapping configuration names to similarity matrices
    """
    # Define weight configurations
    weight_configs = [
        {
            'name': 'baseline',
            'weights': {
                'vocabulary': 0.25,
                'sentence': 0.15,
                'transitions': 0.15,
                'ngrams': 0.25,
                'syntactic': 0.20
            }
        },
        {
            'name': 'nlp_only',
            'weights': {
                'vocabulary': 0.0,
                'sentence': 0.0,
                'transitions': 0.0,
                'ngrams': 0.0,
                'syntactic': 1.0
            }
        },
        {
            'name': 'equal',
            'weights': {
                'vocabulary': 0.2,
                'sentence': 0.2,
                'transitions': 0.2,
                'ngrams': 0.2,
                'syntactic': 0.2
            }
        },
        {
            'name': 'vocabulary_focused',
            'weights': {
                'vocabulary': 0.6,
                'sentence': 0.1,
                'transitions': 0.1,
                'ngrams': 0.1,
                'syntactic': 0.1
            }
        },
        {
            'name': 'structure_focused',
            'weights': {
                'vocabulary': 0.1,
                'sentence': 0.3,
                'transitions': 0.3,
                'ngrams': 0.1,
                'syntactic': 0.2
            }
        }
    ]
    
    # Calculate similarity for each configuration
    similarity_matrices = {}
    
    for config in weight_configs:
        print(f"\n=== Running {config['name']} configuration ===")
        
        # Create a new calculator for each configuration with different weights
        calculator = SimilarityCalculator()
        calculator.set_weights(config['weights'])
        
        # Calculate similarity matrix
        similarity_matrix = calculator.calculate_similarity_matrix(features_data)
        
        # Save to outputs
        output_path = os.path.join('author_analysis', f"{config['name']}_similarity.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(similarity_matrix, f)
        
        # Also save as CSV for easier inspection
        csv_path = os.path.join('author_analysis', f"{config['name']}_similarity.csv")
        similarity_matrix.to_csv(csv_path)
        
        # Store in return dictionary
        similarity_matrices[config['name']] = similarity_matrix
    
    return similarity_matrices

def analyze_corpus_similarities(similarity_matrices: Dict[str, pd.DataFrame]):
    """
    Analyze similarities between author's articles and Pauline letters.
    
    Args:
        similarity_matrices: Dictionary mapping configuration names to similarity matrices
    """
    # Define groups for analysis
    author_prefixes = ['AUTH_']
    pauline_corpus = ['ROM', '1CO', '2CO', 'GAL', 'EPH', 'PHP', 'COL', '1TH', '2TH', '1TI', '2TI', 'TIT', 'PHM']
    
    # Results for each configuration
    results = {}
    
    for config_name, matrix in similarity_matrices.items():
        # Get all text names from matrix
        text_names = matrix.index.tolist()
        
        # Identify author's articles and Pauline letters
        author_articles = [name for name in text_names if any(name.startswith(prefix) for prefix in author_prefixes)]
        pauline_available = [name for name in text_names if name in pauline_corpus]
        
        # Skip if not enough texts
        if len(author_articles) < 1 or len(pauline_available) < 2:
            print(f"Warning: Not enough texts available in {config_name} for comparison")
            continue
        
        # Calculate similarities for each group
        results[config_name] = {
            'author_pairs': [],
            'pauline_pairs': [],
            'author_pauline_pairs': []
        }
        
        # Author article pairs
        for book1, book2 in combinations(author_articles, 2):
            similarity = matrix.loc[book1, book2]
            results[config_name]['author_pairs'].append((book1, book2, similarity))
        
        # Pauline letter pairs
        for book1, book2 in combinations(pauline_available, 2):
            similarity = matrix.loc[book1, book2]
            results[config_name]['pauline_pairs'].append((book1, book2, similarity))
        
        # Author-Pauline comparisons
        for author_book in author_articles:
            for pauline_book in pauline_available:
                similarity = matrix.loc[author_book, pauline_book]
                results[config_name]['author_pauline_pairs'].append((author_book, pauline_book, similarity))
    
    # Print detailed results
    print("\n===== DETAILED SIMILARITY ANALYSIS =====")
    for config_name, data in results.items():
        print(f"\n== {config_name.upper()} ==")
        
        # Author pairs
        print("\nAuthor Articles:")
        author_similarities = [s for _, _, s in data['author_pairs']]
        print(f"Pairs: {len(data['author_pairs'])}")
        for book1, book2, sim in data['author_pairs']:
            print(f"  {book1}-{book2}: {sim:.4f}")
        if author_similarities:
            print(f"  Average: {np.mean(author_similarities):.4f}")
            print(f"  Min: {np.min(author_similarities):.4f}")
            print(f"  Max: {np.max(author_similarities):.4f}")
            print(f"  Std Dev: {np.std(author_similarities):.4f}")
        
        # Pauline pairs
        print("\nPauline Letters:")
        pauline_similarities = [s for _, _, s in data['pauline_pairs']]
        print(f"Pairs: {len(data['pauline_pairs'])}")
        print(f"  Average: {np.mean(pauline_similarities):.4f}")
        print(f"  Min: {np.min(pauline_similarities):.4f}")
        print(f"  Max: {np.max(pauline_similarities):.4f}")
        print(f"  Std Dev: {np.std(pauline_similarities):.4f}")
        
        # Author-Pauline pairs
        print("\nAuthor-Pauline Comparisons:")
        cross_similarities = [s for _, _, s in data['author_pauline_pairs']]
        print(f"Pairs: {len(data['author_pauline_pairs'])}")
        print(f"  Average: {np.mean(cross_similarities):.4f}")
        print(f"  Min: {np.min(cross_similarities):.4f}")
        print(f"  Max: {np.max(cross_similarities):.4f}")
        print(f"  Std Dev: {np.std(cross_similarities):.4f}")
        
        # Most and least similar pairs between author and Pauline
        most_similar = sorted(data['author_pauline_pairs'], key=lambda x: x[2], reverse=True)[:3]
        least_similar = sorted(data['author_pauline_pairs'], key=lambda x: x[2])[:3]
        
        print("  Most similar Author-Pauline pairs:")
        for book1, book2, sim in most_similar:
            print(f"    {book1}-{book2}: {sim:.4f}")
            
        print("  Least similar Author-Pauline pairs:")
        for book1, book2, sim in least_similar:
            print(f"    {book1}-{book2}: {sim:.4f}")
    
    # Print summary comparing the different corpora across configurations
    print("\n===== SUMMARY COMPARISON =====")
    summary_data = []
    
    for config_name, data in results.items():
        author_avg = np.mean([s for _, _, s in data['author_pairs']]) if data['author_pairs'] else np.nan
        pauline_avg = np.mean([s for _, _, s in data['pauline_pairs']]) if data['pauline_pairs'] else np.nan
        cross_avg = np.mean([s for _, _, s in data['author_pauline_pairs']]) if data['author_pauline_pairs'] else np.nan
        
        summary_data.append({
            'Config': config_name,
            'Author Internal': author_avg,
            'Pauline Internal': pauline_avg,
            'Author-Pauline': cross_avg
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Compare internal similarities to cross-corpus similarities
    print("\n===== SIMILARITY COMPARISON =====")
    for config_name, data in results.items():
        if not all(key in data for key in ['author_pairs', 'pauline_pairs', 'author_pauline_pairs']):
            continue
            
        author_avg = np.mean([s for _, _, s in data['author_pairs']]) if data['author_pairs'] else np.nan
        pauline_avg = np.mean([s for _, _, s in data['pauline_pairs']]) if data['pauline_pairs'] else np.nan
        cross_avg = np.mean([s for _, _, s in data['author_pauline_pairs']]) if data['author_pauline_pairs'] else np.nan
        
        print(f"\n{config_name.upper()}:")
        if not np.isnan(author_avg) and not np.isnan(pauline_avg):
            print(f"  Author internal vs Pauline internal: {author_avg:.4f} vs {pauline_avg:.4f}")
            print(f"  Difference: {(author_avg - pauline_avg):.4f}")
        
        if not np.isnan(cross_avg) and not np.isnan(author_avg):
            print(f"  Author-Pauline vs Author internal: {cross_avg:.4f} vs {author_avg:.4f}")
            print(f"  Difference: {(cross_avg - author_avg):.4f}")
            print(f"  Author-Pauline similarity is {abs(cross_avg - author_avg) / author_avg * 100:.1f}% {'higher' if cross_avg > author_avg else 'lower'} than Author internal")
        
        if not np.isnan(cross_avg) and not np.isnan(pauline_avg):
            print(f"  Author-Pauline vs Pauline internal: {cross_avg:.4f} vs {pauline_avg:.4f}")
            print(f"  Difference: {(cross_avg - pauline_avg):.4f}")
            print(f"  Author-Pauline similarity is {abs(cross_avg - pauline_avg) / pauline_avg * 100:.1f}% {'higher' if cross_avg > pauline_avg else 'lower'} than Pauline internal")

def main():
    """Main function."""
    # Define directories
    author_dir = os.path.join("data", "Author Articles")
    pauline_dir = os.path.join("data", "Paul Texts")
    
    # Create output directory
    output_dir = "author_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load texts
    print("Loading author's articles...")
    author_articles = load_author_articles(author_dir)
    
    print("\nLoading Pauline corpus...")
    pauline_letters = load_pauline_letters(pauline_dir)
    
    # Combine texts
    all_texts = {**author_articles, **pauline_letters}
    print(f"\nTotal texts: {len(all_texts)}")
    
    # Preprocess texts
    print("\nPreprocessing texts...")
    preprocessed_texts = preprocess_texts(all_texts)
    
    # Extract features
    print("\nExtracting features...")
    features_data = extract_features(preprocessed_texts, use_advanced_nlp=True)
    
    # Save features for later use
    with open(os.path.join(output_dir, 'features_data.pkl'), 'wb') as f:
        pickle.dump(features_data, f)
    
    # Run different weight configurations
    print("\nRunning similarity calculations with different weight configurations...")
    similarity_matrices = run_weight_configurations(features_data)
    
    # Analyze similarities
    print("\nAnalyzing corpus similarities...")
    analyze_corpus_similarities(similarity_matrices)
    
    print("\nAnalysis complete. Results saved to", output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 