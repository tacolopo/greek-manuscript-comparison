#!/usr/bin/env python3
"""
Minimal script to test NLP-only configuration with just 2-3 books
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from src.multi_comparison import MultipleManuscriptComparison
from src.similarity import SimilarityCalculator
from src.advanced_nlp import AdvancedGreekProcessor

def main():
    # Create output directory
    output_dir = "nlp_mini_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize similarity calculator with NLP-only weights
    similarity_calculator = SimilarityCalculator()
    similarity_calculator.weights = {
        'vocabulary': 0.0,
        'sentence': 0.0,
        'transitions': 0.0,
        'ngrams': 0.0,
        'syntactic': 1.0
    }
    
    print("Using NLP-only weights:", similarity_calculator.weights)
    
    # Test the AdvancedGreekProcessor directly
    print("\nTesting the AdvancedGreekProcessor directly...")
    processor = AdvancedGreekProcessor()
    
    # Test with a simple Greek text
    test_text = "Παῦλος δοῦλος Χριστοῦ Ἰησοῦ, κλητὸς ἀπόστολος ἀφωρισμένος εἰς εὐαγγέλιον θεοῦ."
    print(f"Test text: {test_text}")
    
    # Process the test text
    nlp_features = processor.process_document(test_text)
    print(f"NLP features keys: {nlp_features.keys()}")
    if 'pos_tags' in nlp_features:
        print(f"POS tags sample: {nlp_features['pos_tags'][:10]}")
        print(f"Number of POS tags: {len(nlp_features['pos_tags'])}")
        
        # Extract syntactic features
        syntactic_features = processor.extract_syntactic_features(nlp_features['pos_tags'])
        print("\nSyntactic features:")
        for k, v in syntactic_features.items():
            print(f"  {k}: {v}")
    else:
        print("No POS tags found in NLP features!")
    
    # Load just 2 books for quick testing
    print("\nLoading just 2 books for quick testing...")
    
    # Only look for Romans and John
    target_books = ["ROM", "JHN"]
    book_texts = {}
    
    # Look in data directories
    data_dirs = [Path("data/Paul Texts"), Path("data/Non-Pauline NT")]
    
    # Dictionary to store combined texts
    for book_code in target_books:
        book_texts[book_code] = ""
    
    # Find and combine chapters
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
            
        for file_path in data_dir.glob("*.txt"):
            file_name = file_path.name
            parts = file_name.split('_')
            if len(parts) >= 3:
                book_code = parts[2]
                if book_code in target_books:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            book_texts[book_code] += " " + text
                            print(f"Added {file_path.name} to {book_code}")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
    
    # Initialize comparison with advanced NLP
    comparison = MultipleManuscriptComparison(
        use_advanced_nlp=True,
        output_dir=output_dir,
        visualizations_dir=output_dir,
        similarity_calculator=similarity_calculator
    )
    
    # Preprocess the books
    preprocessed = {}
    for book_code, text in book_texts.items():
        if text.strip():
            print(f"\nPreprocessing {book_code}...")
            preprocessed[book_code] = comparison.preprocessor.preprocess(text)
            
            # Check if NLP features were properly added
            if 'nlp_features' in preprocessed[book_code]:
                nlp = preprocessed[book_code]['nlp_features']
                print(f"NLP features keys: {nlp.keys()}")
                if 'pos_tags' in nlp:
                    print(f"POS tags found: {len(nlp['pos_tags'])}")
                    pos_counts = {}
                    for tag in nlp['pos_tags'][:100]:  # Count first 100 tags
                        pos_counts[tag] = pos_counts.get(tag, 0) + 1
                    print(f"POS tag distribution (first 100): {pos_counts}")
                else:
                    print("NO POS TAGS FOUND!")
            else:
                print("NO NLP FEATURES FOUND!")
    
    if len(preprocessed) < 1:
        print("Error: Not enough books preprocessed")
        return
    
    # Extract features
    print("\nExtracting features...")
    features = comparison.extract_features(preprocessed)
    
    # Check syntactic features
    for book_code, feature_dict in features.items():
        print(f"\nFeatures for {book_code}:")
        if 'syntactic_features' in feature_dict:
            print("Syntactic features found!")
            for k, v in feature_dict['syntactic_features'].items():
                print(f"  {k}: {v}")
        else:
            print("NO SYNTACTIC FEATURES FOUND!")
    
    # Calculate similarity matrix
    if len(features) >= 2:
        print("\nCalculating similarity matrix...")
        similarity_matrix = comparison.calculate_similarity_matrix(features)
        print("Similarity matrix:")
        print(similarity_matrix)
        
        # Save the matrix
        with open(os.path.join(output_dir, "similarity_matrix.pkl"), 'wb') as f:
            pickle.dump(similarity_matrix, f)
        
        # Check for non-zero values
        non_diag = similarity_matrix.values[~np.eye(similarity_matrix.shape[0], dtype=bool)]
        print(f"Average non-diagonal value: {non_diag.mean()}")
        print(f"Number of zeros: {(non_diag == 0).sum()}")
        print(f"Number of non-zeros: {(non_diag != 0).sum()}")
    else:
        print("Not enough books to calculate similarity")

if __name__ == "__main__":
    main() 