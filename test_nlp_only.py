#!/usr/bin/env python3
"""
Script to test only the NLP-only configuration
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from src.multi_comparison import MultipleManuscriptComparison
from src.similarity import SimilarityCalculator

def main():
    # Create output directory
    output_dir = "nlp_only_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize similarity calculator with NLP-only weights
    similarity_calculator = SimilarityCalculator()
    # Directly assign to the weights attribute
    similarity_calculator.weights = {
        'vocabulary': 0.0,
        'sentence': 0.0,
        'transitions': 0.0,
        'ngrams': 0.0,
        'syntactic': 1.0
    }
    
    print("Using NLP-only weights:", similarity_calculator.weights)
    
    # Initialize comparison with advanced NLP
    comparison = MultipleManuscriptComparison(
        use_advanced_nlp=True,
        output_dir=output_dir,
        visualizations_dir=output_dir,
        similarity_calculator=similarity_calculator
    )
    
    # Load manuscript data
    data_dirs = [
        Path("data/Paul Texts"),
        Path("data/Non-Pauline NT")
    ]
    
    manuscripts = {}
    
    # Only process these books (matched with the 3-letter code in the filename)
    target_books = {
        "ROM": "Romans",
        "1CO": "1 Corinthians",
        "2CO": "2 Corinthians",
        "GAL": "Galatians",
        "EPH": "Ephesians",
        "PHP": "Philippians",
        "COL": "Colossians",
        "1TH": "1 Thessalonians",
        "2TH": "2 Thessalonians",
        "HEB": "Hebrews",
        "JHN": "John"
    }
    
    print(f"Loading manuscript data...")
    for data_dir in data_dirs:
        print(f"Searching in {data_dir}...")
        if not data_dir.exists():
            print(f"Directory {data_dir} does not exist")
            continue
            
        for file_path in data_dir.glob("*.txt"):
            file_name = file_path.name
            # Parse the filename to extract book code (ROM, 1CO, etc.)
            parts = file_name.split('_')
            if len(parts) >= 3:
                book_code = parts[2]  # E.g., "ROM", "1CO", etc.
                if book_code in target_books:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            manuscripts[file_path.stem] = text
                            print(f"Loaded {file_path.stem}, book code: {book_code}")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
    
    if not manuscripts:
        print("Error: No manuscript files found")
        sys.exit(1)
    
    print(f"Loaded {len(manuscripts)} manuscript files")
    
    # Combine chapters into books
    print("Combining chapters into books...")
    book_texts = {}
    for name, text in manuscripts.items():
        # Extract the book code from parts
        parts = name.split('_')
        if len(parts) >= 3:
            book_code = parts[2]  # E.g., "ROM", "1CO", etc.
            if book_code in target_books:
                if book_code not in book_texts:
                    book_texts[book_code] = ""
                book_texts[book_code] += " " + text
    
    print(f"Created {len(book_texts)} complete books:")
    for book_code in book_texts:
        print(f"  - {book_code} ({target_books.get(book_code, 'Unknown')})")
    
    # Preprocess the manuscripts
    preprocessed = {}
    for book_code, text in book_texts.items():
        print(f"Preprocessing {book_code}...")
        preprocessed[book_code] = comparison.preprocessor.preprocess(text)
    
    # Extract features
    print("Extracting features...")
    features = comparison.extract_features(preprocessed)
    print(f"Features extracted for {len(features)} books")
    
    # Calculate similarity matrix
    print("Calculating similarity matrix...")
    similarity_matrix = comparison.calculate_similarity_matrix(features)
    print("Similarity matrix calculated")
    
    # Save similarity matrix
    with open(os.path.join(output_dir, "similarity_matrix.pkl"), 'wb') as f:
        pickle.dump(similarity_matrix, f)
    
    # Print matrix diagnostics
    print(f"Matrix shape: {similarity_matrix.shape}")
    print(f"Matrix has zeros only: {(similarity_matrix.values == 0).all()}")
    non_diag = similarity_matrix.values[~np.eye(similarity_matrix.shape[0], dtype=bool)]
    print(f"Average non-diagonal value: {non_diag.mean()}")
    print(f"Min value: {non_diag.min()}")
    print(f"Max value: {non_diag.max()}")
    print(f"Number of zeros: {(non_diag == 0).sum()}")
    print(f"Number of non-zeros: {(non_diag != 0).sum()}")
    print(f"First 5x5 region of matrix:")
    print(similarity_matrix.iloc[:5, :5])
    
    # Do clustering with the NLP-only matrix
    if not (similarity_matrix.values == 0).all():
        clusters = comparison.cluster_manuscripts(
            similarity_df=similarity_matrix,
            n_clusters=3,
            method='hierarchical'
        )
        
        # Generate visualizations
        comparison.generate_visualizations(
            clustering_result=clusters,
            similarity_df=similarity_matrix,
            threshold=0.3  # Lower threshold for visualization
        )
        print("Generated visualizations")
    else:
        print("ERROR: Matrix contains all zeros, cannot perform clustering")

if __name__ == "__main__":
    main() 