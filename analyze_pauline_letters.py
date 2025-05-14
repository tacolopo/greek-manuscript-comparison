#!/usr/bin/env python3
"""
Script to analyze internal similarity within Pauline letters.
This script focuses only on analyzing how different weight configurations 
affect the internal similarity patterns within the Pauline corpus.
"""

import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import defaultdict

# Import the necessary classes from the project
from src.preprocessing import GreekTextPreprocessor
from src.features import FeatureExtractor
from src.similarity import SimilarityCalculator
from src.advanced_nlp import AdvancedGreekProcessor as AdvancedNLPProcessor

def load_pauline_texts():
    """Load Pauline letters corpus."""
    # Load Pauline letters
    pauline_dir = os.path.join("data", "Paul Texts")
    pauline_books = ['ROM', '1CO', '2CO', 'GAL', 'EPH', 'PHP', 'COL', 
                     '1TH', '2TH', '1TI', '2TI', 'TIT', 'PHM']
    
    chapter_files = glob.glob(os.path.join(pauline_dir, "*.txt"))
    book_chapters = defaultdict(list)
    
    for file_path in chapter_files:
        for book_code in pauline_books:
            if f"_{book_code}_" in file_path:
                book_chapters[book_code].append(file_path)
                break
    
    pauline_texts = {}
    for book_code, chapters in book_chapters.items():
        # Sort chapters numerically
        chapters_sorted = sorted(chapters, key=lambda f: int(f.split('_')[-2]))
        
        # Combine chapter texts
        combined_text = []
        for chapter_file in chapters_sorted:
            with open(chapter_file, 'r', encoding='utf-8') as f:
                chapter_text = f.read().strip()
                combined_text.append(chapter_text)
        
        if combined_text:
            pauline_texts[book_code] = "\n\n".join(combined_text)
            print(f"Loaded Pauline: {book_code} ({len(chapters_sorted)} chapters)")
    
    return pauline_texts

def preprocess_texts(texts):
    """Preprocess all texts."""
    preprocessed_texts = {}
    
    # Use Greek preprocessor for Pauline texts
    greek_preprocessor = GreekTextPreprocessor()
    
    for text_name, content in texts.items():
        try:
            print(f"Preprocessing {text_name}...")
            preprocessed = greek_preprocessor.preprocess(content)
            preprocessed_texts[text_name] = preprocessed
        except Exception as e:
            print(f"Error preprocessing {text_name}: {e}")
    
    return preprocessed_texts

def extract_features(preprocessed_texts):
    """Extract features from preprocessed texts."""
    features_data = {}
    extractor = FeatureExtractor()
    
    # Fit the extractor on all texts
    all_texts = [data['normalized_text'] for data in preprocessed_texts.values()]
    extractor.fit(all_texts)
    
    # Initialize NLP processor
    try:
        print("Initializing advanced NLP processor...")
        nlp_processor = AdvancedNLPProcessor()
        print("Successfully initialized advanced NLP processor")
    except Exception as e:
        print(f"Warning: Could not initialize advanced NLP processor: {e}")
        print("Will proceed without advanced NLP features")
        nlp_processor = None
    
    # Extract features for each text
    for text_name, preprocessed in preprocessed_texts.items():
        print(f"Processing {text_name}...")
        
        # Extract base features
        features = extractor.extract_all_features(preprocessed)
        
        # Add syntactic features
        if nlp_processor:
            # Use advanced NLP if available
            try:
                print(f"Extracting advanced NLP features for {text_name}...")
                syntactic_features = nlp_processor.extract_syntactic_features(preprocessed['normalized_text'])
                features['syntactic_features'] = syntactic_features
            except Exception as e:
                print(f"Warning: Could not extract NLP features for {text_name}: {e}")
                features['syntactic_features'] = {
                    'noun_ratio': 0.25,
                    'verb_ratio': 0.20,
                    'adj_ratio': 0.10,
                    'adv_ratio': 0.05,
                    'function_word_ratio': 0.40,
                }
        
        features_data[text_name] = features
    
    return features_data

def calculate_similarity(features_data, corpus_names, weight_config):
    """Calculate the internal similarity within a single corpus using specific weights."""
    # Get corpus features
    corpus_features = {name: features_data[name] for name in corpus_names if name in features_data}
    corpus_names = list(corpus_features.keys())
    
    # Extract raw feature vectors for all texts
    raw_feature_vectors = {}
    for name in corpus_names:
        features = corpus_features[name]
        
        # Vocabulary features
        vocab_features = []
        vocab = features['vocabulary_richness']
        vocab_features.extend([
            vocab['unique_tokens_ratio'],
            vocab['hapax_legomena_ratio'],
            vocab['dis_legomena_ratio'],
            vocab['yule_k'],
            vocab['simpson_d'],
            vocab['herdan_c'],
            vocab['guiraud_r'],
            vocab['sichel_s']
        ])
        
        # Sentence features
        sent_features = []
        sent_stats = features['sentence_stats']
        sentence_cv = sent_stats['std_sentence_length'] / (sent_stats['mean_sentence_length'] + 1e-10)
        sent_features.extend([
            sent_stats['mean_sentence_length'],
            sent_stats['median_sentence_length'],
            sentence_cv,
            sent_stats['length_variance_normalized']
        ])
        
        # Transition features
        trans_features = []
        transitions = features['transition_patterns']
        trans_features.extend([
            transitions['length_transition_smoothness'],
            transitions['length_pattern_repetition'],
            transitions['clause_boundary_regularity'],
            transitions['sentence_rhythm_consistency']
        ])
        
        # N-gram features
        ngram_features = []
        for ngram_dict in [features['word_bigrams'], features['word_trigrams']]:
            if ngram_dict and len(ngram_dict) > 0:
                values = list(ngram_dict.values())
                total = sum(values) + 1e-10
                normalized_values = [v/total for v in values]
                ngram_features.extend([
                    np.mean(normalized_values),
                    np.std(normalized_values) / (np.mean(normalized_values) + 1e-10)
                ])
            else:
                ngram_features.extend([0, 0])
        
        # Syntactic features
        syn_features = []
        if 'syntactic_features' in features:
            syntactic = features['syntactic_features']
            
            # Basic ratios
            syn_features.extend([
                syntactic.get('noun_ratio', 0.25),
                syntactic.get('verb_ratio', 0.20),
                syntactic.get('adj_ratio', 0.10),
                syntactic.get('adv_ratio', 0.05),
                syntactic.get('function_word_ratio', 0.40)
            ])
            
            # Add more syntactic features if available
            extended_features = [
                syntactic.get('tag_diversity', 0.7),
                syntactic.get('tag_entropy', 2.0),
                syntactic.get('noun_verb_ratio', 1.25)
            ]
            syn_features.extend(extended_features)
        else:
            # Add default values if features are missing
            syn_features.extend([0.25, 0.20, 0.10, 0.05, 0.40, 0.7, 2.0, 1.25])
        
        # Combine into feature groups
        raw_feature_vectors[name] = {
            'vocabulary': np.array(vocab_features),
            'sentence': np.array(sent_features),
            'transitions': np.array(trans_features),
            'ngrams': np.array(ngram_features),
            'syntactic': np.array(syn_features)
        }
    
    # Extract features into matrices for standardization
    feature_matrices = {
        'vocabulary': np.array([fv['vocabulary'] for fv in raw_feature_vectors.values()]),
        'sentence': np.array([fv['sentence'] for fv in raw_feature_vectors.values()]),
        'transitions': np.array([fv['transitions'] for fv in raw_feature_vectors.values()]),
        'ngrams': np.array([fv['ngrams'] for fv in raw_feature_vectors.values()]),
        'syntactic': np.array([fv['syntactic'] for fv in raw_feature_vectors.values()])
    }
    
    # Standardize each feature group (zero mean, unit variance)
    from sklearn.preprocessing import StandardScaler
    standardized_matrices = {}
    for group, matrix in feature_matrices.items():
        # For NLP-only, if all syntactic features are identical, add noise
        if group == 'syntactic' and np.all(matrix == matrix[0, :]):
            print(f"WARNING: All syntactic features are identical, adding noise")
            matrix = matrix + np.random.normal(0, 0.1, matrix.shape)
        
        # Skip standardization if all values are identical (would result in NaN)
        if np.all(matrix == matrix[0, :]):
            standardized_matrices[group] = matrix
        else:
            try:
                standardized_matrices[group] = StandardScaler().fit_transform(matrix)
            except:
                # If standardization fails, use raw values
                print(f"Warning: Standardization failed for {group}, using raw values")
                standardized_matrices[group] = matrix
    
    # Apply weights and combine
    weighted_features = {}
    for i, name in enumerate(corpus_names):
        # Apply weights to each standardized feature group
        weighted_vector = np.concatenate([
            standardized_matrices['vocabulary'][i] * weight_config['vocabulary'],
            standardized_matrices['sentence'][i] * weight_config['sentence'],
            standardized_matrices['transitions'][i] * weight_config['transitions'],
            standardized_matrices['ngrams'][i] * weight_config['ngrams'],
            standardized_matrices['syntactic'][i] * weight_config['syntactic']
        ])
        weighted_features[name] = weighted_vector
    
    # Calculate pairwise cosine similarities
    similarity_matrix = {}
    for name1 in corpus_names:
        similarity_matrix[name1] = {}
        for name2 in corpus_names:
            if name1 == name2:
                similarity_matrix[name1][name2] = 1.0
            else:
                # Calculate cosine similarity
                vec1 = weighted_features[name1]
                vec2 = weighted_features[name2]
                
                # Handle zero vectors
                if np.all(vec1 == 0) or np.all(vec2 == 0):
                    similarity = 0.0
                else:
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                
                similarity_matrix[name1][name2] = similarity
    
    # Convert to DataFrame
    matrix_df = pd.DataFrame(similarity_matrix)
    
    # Get all pairwise similarities
    similarities = []
    pairs = []
    for i, name1 in enumerate(corpus_names):
        for j, name2 in enumerate(corpus_names):
            if i < j:
                sim = matrix_df.loc[name1, name2]
                similarities.append(sim)
                pairs.append((name1, name2, sim))
    
    # Calculate statistics
    if similarities:
        avg_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)
        max_similarity = np.max(similarities)
        std_similarity = np.std(similarities)
        
        # Get top 5 most similar pairs
        most_similar = sorted(pairs, key=lambda x: x[2], reverse=True)[:5]
        
        # Get 5 least similar pairs
        least_similar = sorted(pairs, key=lambda x: x[2])[:5]
    else:
        avg_similarity = min_similarity = max_similarity = std_similarity = 0
        most_similar = []
        least_similar = []
    
    print(f"Applied weights: {weight_config}")
    print(f"Feature vector shape for {corpus_names[0]}: {weighted_features[corpus_names[0]].shape}")
    print(f"Average similarity: {avg_similarity:.4f}, Min: {min_similarity:.4f}, Max: {max_similarity:.4f}")
    
    return {
        'average': avg_similarity,
        'min': min_similarity,
        'max': max_similarity,
        'std_dev': std_similarity,
        'most_similar': most_similar,
        'least_similar': least_similar,
        'pair_count': len(similarities),
        'matrix': matrix_df,
        'weighted_features': weighted_features
    }

def run_weight_configurations(features_data, pauline_names):
    """
    Run similarity calculations with different weight configurations.
    
    Returns:
        Dictionary mapping configuration names to results
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
    
    # Calculate similarities for each configuration
    results = {}
    
    for config in weight_configs:
        print(f"\n=== Running {config['name']} configuration ===")
        
        # Calculate internal similarities for Pauline letters
        pauline_results = calculate_similarity(
            features_data, pauline_names, config['weights']
        )
        
        # Store results
        results[config['name']] = pauline_results
        
        # Print immediate results
        print(f"Pauline internal similarity: {pauline_results['average']:.4f}")
    
    return results

def analyze_and_print_results(results):
    """Analyze and print the results in a readable format."""
    print("\n\n===== PAULINE LETTERS SIMILARITY ANALYSIS =====\n")
    
    # Create comparison table data
    comparison_data = []
    
    for config_name, data in results.items():
        comparison_data.append({
            'Config': config_name,
            'Average Similarity': data['average'],
            'Min Similarity': data['min'],
            'Max Similarity': data['max'],
            'Std Dev': data['std_dev'],
            'Pairs Analyzed': data['pair_count']
        })
    
    # Convert to DataFrame for nice printing
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Detailed results for each configuration
    for config_name, data in results.items():
        print(f"\n\n==== {config_name.upper()} ====")
        
        # Pauline details
        print("\nPauline Internal Similarities:")
        print(f"  Pairs analyzed: {data['pair_count']}")
        print(f"  Average similarity: {data['average']:.4f}")
        print(f"  Min similarity: {data['min']:.4f}")
        print(f"  Max similarity: {data['max']:.4f}")
        print(f"  Std Dev: {data['std_dev']:.4f}")
        
        print("\nTop 5 most similar Pauline pairs:")
        for name1, name2, sim in data['most_similar']:
            print(f"  {name1} - {name2}: {sim:.4f}")
            
        print("\n5 least similar Pauline pairs:")
        for name1, name2, sim in data['least_similar']:
            print(f"  {name1} - {name2}: {sim:.4f}")

def create_heatmap(results, output_dir):
    """Create heatmaps for each weight configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    for config_name, data in results.items():
        # Get the similarity matrix
        matrix = data['matrix']
        
        # Set up the figure
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1,
                    square=True, linewidths=.5, cbar_kws={"shrink": .75})
        
        # Add title
        plt.title(f'Pauline Letters Similarity - {config_name.replace("_", " ").title()}', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f"pauline_heatmap_{config_name}.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, f"pauline_heatmap_{config_name}.pdf"))
        plt.close()
        
    print(f"Heatmaps saved to {output_dir}")

def create_bar_charts(results, output_dir):
    """Create bar charts comparing similarity across weight configurations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    configs = []
    avg_similarities = []
    min_similarities = []
    max_similarities = []
    
    for config_name, data in results.items():
        configs.append(config_name)
        avg_similarities.append(data['average'])
        min_similarities.append(data['min'])
        max_similarities.append(data['max'])
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Set position of bars on x axis
    x = np.arange(len(configs))
    width = 0.25  # Width of bars
    
    # Create bars
    plt.bar(x - width, avg_similarities, width, color='#1f77b4', label='Average Similarity')
    plt.bar(x, min_similarities, width, color='#ff7f0e', label='Min Similarity')
    plt.bar(x + width, max_similarities, width, color='#2ca02c', label='Max Similarity')
    
    # Add labels and title
    plt.xlabel('Weight Configuration', fontsize=14)
    plt.ylabel('Similarity', fontsize=14)
    plt.title('Pauline Letters Similarity by Weight Configuration', fontsize=16)
    
    # Format x-axis labels
    plt.xticks(x, [c.replace('_', ' ').title() for c in configs])
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add legend
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "pauline_similarity_comparison.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "pauline_similarity_comparison.pdf"))
    plt.close()
    
    print(f"Bar chart saved to {output_dir}")

def main():
    """Main function."""
    # Create output directory
    output_dir = "pauline_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load texts
    print("Loading Pauline texts...")
    pauline_texts = load_pauline_texts()
    
    # Get the names of texts
    pauline_names = list(pauline_texts.keys())
    
    # Step 2: Preprocess texts
    print("\nPreprocessing texts...")
    preprocessed_texts = preprocess_texts(pauline_texts)
    
    # Step 3: Extract features
    print("\nExtracting features...")
    features_data = extract_features(preprocessed_texts)
    
    # Save features for reference
    with open(os.path.join(output_dir, 'pauline_features_data.pkl'), 'wb') as f:
        pickle.dump(features_data, f)
    
    # Step 4: Run different weight configurations
    print("\nRunning similarity calculations with different weight configurations...")
    results = run_weight_configurations(features_data, pauline_names)
    
    # Save results
    with open(os.path.join(output_dir, 'pauline_similarity_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Step 5: Analyze and print results
    analyze_and_print_results(results)
    
    # Step 6: Create visualizations
    create_heatmap(results, output_dir)
    create_bar_charts(results, output_dir)
    
    print("\nAnalysis complete. Results saved to", output_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 