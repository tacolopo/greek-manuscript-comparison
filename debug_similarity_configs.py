#!/usr/bin/env python3
"""
Debug script to investigate why equal and baseline similarity matrices are coming back similar.
This script will explicitly run the analysis with different weight configurations and check
if the similarity matrices are actually different.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try imports with appropriate error handling
try:
    from src.similarity import SimilarityCalculator
    from src.features import FeatureExtractor
    from src.multi_comparison import MultipleManuscriptComparison
except ImportError as e:
    print(f"Error: {e}")
    print("Could not import required modules. Make sure you're running from the project root.")
    sys.exit(1)

def create_weight_configs():
    """Define the weight configurations for different iterations."""
    weight_configs = [
        # 1. Current weights (baseline)
        {
            'name': 'baseline',
            'description': 'Current balanced weights',
            'weights': {
                'vocabulary': 0.25,
                'sentence': 0.15,
                'transitions': 0.15,
                'ngrams': 0.25,
                'syntactic': 0.20
            }
        },
        # 2. Equal weights
        {
            'name': 'equal',
            'description': 'Equal weights for all features',
            'weights': {
                'vocabulary': 0.2,
                'sentence': 0.2,
                'transitions': 0.2,
                'ngrams': 0.2,
                'syntactic': 0.2
            }
        },
        # 3. Vocabulary-focused
        {
            'name': 'vocabulary_focused',
            'description': 'Focus on vocabulary and n-grams',
            'weights': {
                'vocabulary': 0.4,
                'sentence': 0.07,
                'transitions': 0.06,
                'ngrams': 0.4,
                'syntactic': 0.07
            }
        }
    ]
    return weight_configs

def generate_test_data():
    """Generate sample test data with diverse features."""
    manuscripts = {}
    
    # Create some sample Greek texts with variations
    texts = [
        # Sample 1 - Simple text
        "Παῦλος δοῦλος θεοῦ, ἀπόστολος δὲ Ἰησοῦ Χριστοῦ κατὰ πίστιν ἐκλεκτῶν θεοῦ καὶ ἐπίγνωσιν ἀληθείας τῆς κατ' εὐσέβειαν.",
        
        # Sample 2 - Medium text
        "Παῦλος ἀπόστολος Χριστοῦ Ἰησοῦ διὰ θελήματος θεοῦ τοῖς ἁγίοις τοῖς οὖσιν ἐν Ἐφέσῳ καὶ πιστοῖς ἐν Χριστῷ Ἰησοῦ, χάρις ὑμῖν καὶ εἰρήνη ἀπὸ θεοῦ πατρὸς ἡμῶν καὶ κυρίου Ἰησοῦ Χριστοῦ.",
        
        # Sample 3 - Longer text
        "Παῦλος ἀπόστολος οὐκ ἀπ' ἀνθρώπων οὐδὲ δι' ἀνθρώπου ἀλλὰ διὰ Ἰησοῦ Χριστοῦ καὶ θεοῦ πατρὸς τοῦ ἐγείραντος αὐτὸν ἐκ νεκρῶν, καὶ οἱ σὺν ἐμοὶ πάντες ἀδελφοί, ταῖς ἐκκλησίαις τῆς Γαλατίας· χάρις ὑμῖν καὶ εἰρήνη ἀπὸ θεοῦ πατρὸς καὶ κυρίου ἡμῶν Ἰησοῦ Χριστοῦ τοῦ δόντος ἑαυτὸν ὑπὲρ τῶν ἁμαρτιῶν ἡμῶν, ὅπως ἐξέληται ἡμᾶς ἐκ τοῦ αἰῶνος τοῦ ἐνεστῶτος πονηροῦ κατὰ τὸ θέλημα τοῦ θεοῦ καὶ πατρὸς ἡμῶν,",
        
        # Sample 4 - Different style
        "Ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος. οὗτος ἦν ἐν ἀρχῇ πρὸς τὸν θεόν. πάντα δι' αὐτοῦ ἐγένετο, καὶ χωρὶς αὐτοῦ ἐγένετο οὐδὲ ἕν ὃ γέγονεν.",
        
        # Sample 5 - Different vocabulary
        "Ἀποκάλυψις Ἰησοῦ Χριστοῦ ἣν ἔδωκεν αὐτῷ ὁ θεός, δεῖξαι τοῖς δούλοις αὐτοῦ ἃ δεῖ γενέσθαι ἐν τάχει, καὶ ἐσήμανεν ἀποστείλας διὰ τοῦ ἀγγέλου αὐτοῦ τῷ δούλῳ αὐτοῦ Ἰωάννῃ, ὃς ἐμαρτύρησεν τὸν λόγον τοῦ θεοῦ καὶ τὴν μαρτυρίαν Ἰησοῦ Χριστοῦ ὅσα εἶδεν."
    ]
    
    # Create manuscripts
    for i, text in enumerate(texts):
        manuscripts[f"manuscript_{i+1}"] = text
    
    return manuscripts

def run_debug_analysis():
    """Run similarity analysis with different weight configurations and check the matrices."""
    print("Running debug analysis to check similarity matrices with different weights...")
    
    # Get weight configurations
    weight_configs = create_weight_configs()
    
    # Generate test data
    test_manuscripts = generate_test_data()
    
    # Create output directory
    debug_dir = "debug_matrices"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Initialize comparison object
    comparison = MultipleManuscriptComparison(
        use_advanced_nlp=False,
        output_dir=debug_dir,
        visualizations_dir=debug_dir
    )
    
    # Process all manuscripts first to ensure we have consistent feature data
    print("Processing manuscripts...")
    preprocessed = {}
    for name, text in test_manuscripts.items():
        preprocessed[name] = comparison.preprocessor.preprocess(text)
    
    # Extract features for all manuscripts
    print("Extracting features...")
    
    # First collect all texts for TF-IDF fitting
    all_texts = []
    for name, preprocessed_data in preprocessed.items():
        if 'words' in preprocessed_data:
            all_texts.append(' '.join(preprocessed_data['words']))
    
    # Create feature extractor and fit
    feature_extractor = FeatureExtractor()
    feature_extractor.fit(all_texts)
    comparison.feature_extractor = feature_extractor
    
    # Extract features
    features = {}
    for name, preprocessed_data in preprocessed.items():
        features[name] = comparison.feature_extractor.extract_all_features(preprocessed_data)
    
    # Run the analysis with different weight configurations
    similarity_matrices = []
    weighted_feature_vectors = {}
    
    for config in weight_configs:
        print(f"\nRunning analysis with {config['name']} configuration:")
        print(f"  - {config['description']}")
        print(f"  - Weights: {config['weights']}")
        
        # Set custom weights in the similarity calculator
        custom_calculator = SimilarityCalculator()
        custom_calculator.set_weights(config['weights'])
        
        # Replace the default calculator with our custom one
        comparison.similarity_calculator = custom_calculator
        
        # Calculate feature vectors
        print("  - Calculating feature vectors...")
        feature_vectors = {}
        for name, feature_data in features.items():
            vector = custom_calculator.calculate_feature_vector(feature_data)
            feature_vectors[name] = vector
        
        # Store for later comparison
        weighted_feature_vectors[config['name']] = feature_vectors
        
        # Calculate similarity matrix
        similarity_df = comparison.calculate_similarity_matrix(features)
        
        # Save similarity matrix
        matrix_file = os.path.join(debug_dir, f"{config['name']}_similarity_matrix.csv")
        similarity_df.to_csv(matrix_file)
        print(f"Saved {config['name']} similarity matrix to {matrix_file}")
        
        similarity_matrices.append({
            'name': config['name'],
            'description': config['description'],
            'weights': config['weights'],
            'matrix': similarity_df,
            'feature_vectors': feature_vectors
        })
    
    # Compare matrices
    print("\nComparing similarity matrices:")
    for i in range(len(similarity_matrices)):
        for j in range(i+1, len(similarity_matrices)):
            matrix1 = similarity_matrices[i]['matrix'].values
            matrix2 = similarity_matrices[j]['matrix'].values
            
            # Calculate difference statistics
            diff = matrix1 - matrix2
            max_diff = np.max(np.abs(diff))
            avg_diff = np.mean(np.abs(diff))
            nonzero_diff = np.count_nonzero(diff)
            
            print(f"Comparison: {similarity_matrices[i]['name']} vs {similarity_matrices[j]['name']}")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Average difference: {avg_diff:.6f}")
            print(f"  Nonzero differences: {nonzero_diff}/{diff.size}")
            
            if max_diff < 0.001:
                print("  WARNING: Matrices are nearly identical!")
            else:
                print("  Matrices are different.")
    
    # Compare feature vectors to see if weights are actually applied differently
    print("\nComparing feature vectors from different weight configurations:")
    for i in range(len(similarity_matrices)):
        for j in range(i+1, len(similarity_matrices)):
            config1 = similarity_matrices[i]['name']
            config2 = similarity_matrices[j]['name']
            
            print(f"Comparing feature vectors: {config1} vs {config2}")
            
            # Check a few manuscripts
            for name in list(features.keys())[:2]:  # Just check first two manuscripts
                vec1 = similarity_matrices[i]['feature_vectors'][name]
                vec2 = similarity_matrices[j]['feature_vectors'][name]
                
                # Calculate vector differences
                vec_diff = vec1 - vec2
                max_vec_diff = np.max(np.abs(vec_diff))
                avg_vec_diff = np.mean(np.abs(vec_diff))
                nonzero_vec_diff = np.count_nonzero(vec_diff)
                
                print(f"  Manuscript '{name}':")
                print(f"    Max difference: {max_vec_diff:.6f}")
                print(f"    Average difference: {avg_vec_diff:.6f}")
                print(f"    Nonzero differences: {nonzero_vec_diff}/{vec_diff.size}")
                
                # Print first few components of each vector
                print(f"    {config1} first 5 components: {vec1[:5]}")
                print(f"    {config2} first 5 components: {vec2[:5]}")
                
                if max_vec_diff < 0.001:
                    print("    WARNING: Feature vectors are nearly identical!")
                else:
                    print("    Feature vectors are different.")
    
    # Create plots to visualize differences
    print("\nGenerating difference visualizations...")
    
    # Get the baseline and equal matrices
    baseline_matrix = None
    equal_matrix = None
    
    for matrix_data in similarity_matrices:
        if matrix_data['name'] == 'baseline':
            baseline_matrix = matrix_data['matrix']
        elif matrix_data['name'] == 'equal':
            equal_matrix = matrix_data['matrix']
    
    if baseline_matrix is not None and equal_matrix is not None:
        # Calculate difference
        diff_matrix = baseline_matrix - equal_matrix
        
        # Create a heatmap of the difference
        plt.figure(figsize=(12, 10))
        plt.imshow(diff_matrix.values, cmap='coolwarm', vmin=-0.5, vmax=0.5)
        plt.colorbar(label='Difference (Baseline - Equal)')
        plt.title('Difference Between Baseline and Equal Weight Configurations')
        plt.xlabel('Manuscript Index')
        plt.ylabel('Manuscript Index')
        
        # Save the plot
        diff_plot_file = os.path.join(debug_dir, "baseline_equal_diff.png")
        plt.savefig(diff_plot_file)
        print(f"Saved difference visualization to {diff_plot_file}")

if __name__ == "__main__":
    run_debug_analysis() 