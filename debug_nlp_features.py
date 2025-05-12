#!/usr/bin/env python3
"""
Script to debug the NLP feature extraction and similarity calculation.
"""

import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import glob
import re
from collections import defaultdict

from src.multi_comparison import MultipleManuscriptComparison
from src.similarity import SimilarityCalculator
from src.features import FeatureExtractor

# Copied from iterate_similarity_weights.py
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

def debug_syntactic_features():
    """Debug the syntactic feature extraction and similarity calculation."""
    
    # Create comparison object with advanced NLP
    comparison = MultipleManuscriptComparison(
        use_advanced_nlp=True,
        output_dir="debug_output",
        visualizations_dir="debug_output/vis"
    )
    
    # Load book texts
    pauline_dir = os.path.join("data", "Paul Texts")
    non_pauline_dir = os.path.join("data", "Non-Pauline NT")
    
    # Get all chapter files
    pauline_files = glob.glob(os.path.join(pauline_dir, "*.txt"))
    non_pauline_files = glob.glob(os.path.join(non_pauline_dir, "*.txt"))
    all_chapter_files = pauline_files + non_pauline_files
    
    # Load and group the files by book
    print("Loading chapter files...")
    print(f"Found {len(all_chapter_files)} chapter files")
    combined_books = group_and_combine_books(all_chapter_files)
    print(f"Found {len(combined_books)} complete books")
    
    # Sample a few books for debugging (3 is enough)
    sample_books = {}
    sample_list = list(combined_books.keys())[:3]
    for book in sample_list:
        sample_books[book] = combined_books[book]
    
    # Use our small sample for debugging
    print(f"Using sample of {len(sample_books)} books: {', '.join(sample_books.keys())}")
    
    # Process the texts and extract features
    print("\nPreprocessing texts...")
    preprocessed = {}
    for name, text in tqdm(sample_books.items()):
        preprocessed[name] = comparison.preprocessor.preprocess(text)
    
    # Check NLP features
    print("\nChecking NLP features...")
    for book_code, data in preprocessed.items():
        print(f"\nNLP features for {book_code}:")
        if 'nlp_features' in data:
            nlp = data['nlp_features']
            print(f"NLP features keys: {list(nlp.keys())}")
            
            # Check POS tags
            if 'pos_tags' in nlp:
                print(f"POS tags found: {len(nlp['pos_tags'])}")
                print(f"POS tag sample: {nlp['pos_tags'][:10]}")
                try:
                    unique_tags = sorted(list(set(str(tag) for tag in nlp['pos_tags'])))
                    print(f"Unique POS tags: {unique_tags}")
                except:
                    print("Could not extract unique tags")
            else:
                print("NO POS TAGS FOUND!")
                
            # Check dependency relations
            if 'dependency_relations' in nlp:
                print(f"Dependency relations found: {len(nlp['dependency_relations'])}")
                print(f"Sample: {nlp['dependency_relations'][:2]}")
            else:
                print("NO DEPENDENCY RELATIONS FOUND!")
        else:
            print("NO NLP FEATURES FOUND!")
    
    # Create feature extractor and fit it on all texts
    print("\nFitting feature extractor...")
    all_texts = [preprocessed[name].get('normalized_text', ' '.join(preprocessed[name]['words'])) 
                for name in preprocessed]
    
    # Create and fit the feature extractor
    feature_extractor = FeatureExtractor()
    feature_extractor.fit(all_texts)
    
    # Replace the feature extractor in the comparison object
    comparison.feature_extractor = feature_extractor
    
    # Extract features
    print("\nExtracting features...")
    features = comparison.extract_features(preprocessed)
    
    # Check syntactic features
    print("\nChecking syntactic features...")
    for book_code, feature_dict in features.items():
        print(f"\nFeatures for {book_code}:")
        if 'syntactic_features' in feature_dict:
            print("Syntactic features found!")
            print(f"Number of features: {len(feature_dict['syntactic_features'])}")
            for k, v in feature_dict['syntactic_features'].items():
                print(f"  {k}: {v}")
        else:
            print("NO SYNTACTIC FEATURES FOUND!")
    
    # Calculate feature vectors (like the similarity calculation does)
    print("\nCalculating feature vectors...")
    calculator = SimilarityCalculator()
    
    # Override weights to use only syntactic features (NLP-only configuration)
    calculator.weights = {
        'vocabulary': 0.0,
        'sentence': 0.0,
        'transitions': 0.0,
        'ngrams': 0.0,
        'syntactic': 1.0
    }
    
    # Calculate feature vectors for each book
    feature_vectors = {}
    for book, feature_dict in features.items():
        if 'syntactic_features' not in feature_dict:
            print(f"Warning: No syntactic features for {book}, skipping...")
            continue
            
        vector = calculator.calculate_feature_vector(feature_dict)
        feature_vectors[book] = vector
        print(f"{book} feature vector shape: {vector.shape}")
        
        # Print the syntactic part of the vector (last 22 elements)
        print(f"Syntactic part: {vector[-22:]}")
        
        # Count non-zero elements in the syntactic part
        nonzero = np.count_nonzero(vector[-22:])
        print(f"Non-zero syntactic features: {nonzero}/{22}")
    
    # Check if we have any feature vectors
    if not feature_vectors:
        print("No feature vectors could be calculated. Exiting.")
        return
    
    # Calculate pairwise similarities
    print("\nCalculating pairwise similarities...")
    book_list = list(feature_vectors.keys())
    n_books = len(book_list)
    
    # Create similarity matrix using scaled vectors
    X = np.array([feature_vectors[book] for book in book_list])
    
    # Scale features for fair comparison
    print("\nScaling feature vectors...")
    print(f"X shape: {X.shape}")
    print(f"First book full vector: {X[0]}")
    
    # Check if any of the columns are all zeros (would cause division by zero in scaling)
    zero_columns = np.where(np.all(X == 0, axis=0))[0]
    print(f"Columns with all zeros: {zero_columns}")
    
    # Filter out zero columns
    if len(zero_columns) > 0:
        print("Removing all-zero columns before scaling")
        X_filtered = np.delete(X, zero_columns, axis=1)
        print(f"Filtered X shape: {X_filtered.shape}")
    else:
        X_filtered = X
    
    # Only scale if we have non-zero columns
    if X_filtered.shape[1] > 0:
        # Scale if we have multiple books
        if n_books > 1:
            X_filtered = MinMaxScaler().fit_transform(X_filtered)
        
        # Put back the zero columns
        if len(zero_columns) > 0:
            # Insert zero columns back at their original positions
            X_scaled = np.zeros(X.shape)
            col_idx = 0
            for i in range(X.shape[1]):
                if i in zero_columns:
                    X_scaled[:, i] = 0
                else:
                    X_scaled[:, i] = X_filtered[:, col_idx]
                    col_idx += 1
        else:
            X_scaled = X_filtered
    else:
        # All columns are zero!
        X_scaled = X
    
    print(f"X_scaled shape: {X_scaled.shape}")
    print(f"First book syntactic part (scaled): {X_scaled[0, -22:]}")
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((n_books, n_books))
    for i in range(n_books):
        for j in range(n_books):
            # Use cosine similarity
            if i == j:
                similarity_matrix[i, j] = 1.0  # Self-similarity is always 1
            else:
                # Check if both vectors are non-zero
                if np.any(X_scaled[i]) and np.any(X_scaled[j]):
                    # Cosine similarity
                    similarity = np.dot(X_scaled[i], X_scaled[j]) / (
                        np.linalg.norm(X_scaled[i]) * np.linalg.norm(X_scaled[j])
                    )
                    similarity_matrix[i, j] = similarity
                else:
                    similarity_matrix[i, j] = 0.0  # Zero similarity if either vector is all zeros
    
    print("\nSimilarity matrix:")
    print(similarity_matrix)
    
    # Calculate average non-diagonal similarity
    non_diag = similarity_matrix[~np.eye(n_books, dtype=bool)]
    avg_similarity = np.mean(non_diag)
    print(f"Average non-diagonal similarity: {avg_similarity}")
    print(f"Min similarity: {np.min(non_diag)}")
    print(f"Max similarity: {np.max(non_diag)}")
    
    # Create report of each syntactic feature's values across books
    print("\nSyntactic feature values across books:")
    feature_names = list(features[book_list[0]]['syntactic_features'].keys())
    
    # Create a DataFrame with syntactic feature values for each book
    syntactic_df = pd.DataFrame(index=book_list, columns=feature_names)
    for book in book_list:
        for feature in feature_names:
            syntactic_df.loc[book, feature] = features[book]['syntactic_features'][feature]
    
    print(syntactic_df)
    
    # Check variance of each feature
    variances = syntactic_df.var()
    print("\nFeature variances:")
    for feature, variance in variances.items():
        print(f"{feature}: {variance}")
    
    # Suggest fixes
    print("\nPossible issues and fixes:")
    zero_features = variances[variances == 0].index.tolist()
    if zero_features:
        print(f"Features with zero variance: {zero_features}")
        print("These features are not contributing to the similarity calculation.")
    
    # Check if all tags are being recognized properly
    print("\nChecking if POS tags are being recognized properly...")
    for book_code, data in preprocessed.items():
        if 'nlp_features' in data and 'pos_tags' in data['nlp_features']:
            pos_tags = data['nlp_features']['pos_tags']
            
            # Count how many tags are handled in the extract_syntactic_features method
            noun_tags = ['NOUN', 'SUBSTANTIVE', 'NOUN_SUBSTANTIVE', 'noun']
            verb_tags = ['VERB', 'FINITE_VERB', 'verb']
            adj_tags = ['ADJ', 'ADJECTIVE', 'adjective']
            adv_tags = ['ADV', 'ADVERB', 'adverb']
            
            recognized_tags = noun_tags + verb_tags + adj_tags + adv_tags
            tag_counts = {tag: 0 for tag in recognized_tags}
            unrecognized_tags = set()
            
            # Count occurrences
            for tag in pos_tags:
                tag_str = str(tag).lower()
                matched = False
                for recog_tag in recognized_tags:
                    if recog_tag.lower() == tag_str:
                        tag_counts[recog_tag] += 1
                        matched = True
                        break
                if not matched:
                    unrecognized_tags.add(tag_str)
            
            print(f"\nTag counts for {book_code}:")
            print(f"Recognized tags: {tag_counts}")
            print(f"Unrecognized tags: {unrecognized_tags}")
            print(f"Total recognized: {sum(tag_counts.values())}/{len(pos_tags)} ({sum(tag_counts.values())/len(pos_tags)*100:.1f}%)")

if __name__ == "__main__":
    debug_syntactic_features() 