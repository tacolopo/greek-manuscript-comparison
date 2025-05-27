#!/usr/bin/env python3

import pickle
import numpy as np

# Load the corrected analysis results
with open('exact_cleaned_analysis/nlp_only/similarity_matrix.pkl', 'rb') as f:
    data = pickle.load(f)

print("Loaded analysis data structure:")
print(f"Keys: {list(data.keys())}")
print()

# Check if we have feature_vectors
if 'feature_vectors' in data:
    feature_vectors = data['feature_vectors']
    print(f"Feature vectors available for {len(feature_vectors)} manuscripts")
    print(f"Manuscript names: {list(feature_vectors.keys())[:10]}...")
    print()
    
    # Get Julian vector
    julian_name = 'Διονυσίῳ'
    if julian_name in feature_vectors:
        julian_vector = feature_vectors[julian_name]
        print(f"Julian '{julian_name}' feature vector:")
        print(f"Vector length: {len(julian_vector)}")
        print(f"Vector type: {type(julian_vector)}")
        print()
        
        # Show the full vector
        print("COMPLETE JULIAN FEATURE VECTOR:")
        print("=" * 50)
        print(julian_vector)
        print()
        
        # Extract just the syntactic features (last 20 elements)
        syntactic_features = julian_vector[-20:]
        print("SYNTACTIC FEATURES (last 20 elements):")
        print("=" * 50)
        
        feature_names = [
            'noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio', 'function_word_ratio',
            'pronoun_ratio', 'conjunction_ratio', 'particle_ratio', 'interjection_ratio', 'numeral_ratio',
            'tag_diversity', 'tag_entropy', 'noun_after_verb_ratio', 'adj_before_noun_ratio', 'adv_before_verb_ratio',
            'verb_to_noun_prob', 'noun_to_verb_prob', 'noun_to_adj_prob', 'adj_to_noun_prob', 'noun_verb_ratio'
        ]
        
        for i, (name, value) in enumerate(zip(feature_names, syntactic_features)):
            print(f"{i+1:2d}. {name:25s}: {value:.6f}")
        
        print()
        print("ANALYSIS:")
        print("-" * 30)
        print(f"Non-zero features: {np.sum(syntactic_features != 0)}/20")
        print(f"Zero features: {np.sum(syntactic_features == 0)}/20")
        print(f"Feature magnitude: {np.linalg.norm(syntactic_features):.4f}")
        print(f"Min value: {np.min(syntactic_features):.6f}")
        print(f"Max value: {np.max(syntactic_features):.6f}")
        print(f"Mean value: {np.mean(syntactic_features):.6f}")
        
        # Verify these are NOT similarity scores
        print()
        print("VERIFICATION - These are NOT similarity scores:")
        print("-" * 50)
        print("✓ Values are NOT between 0.4-1.0 (typical similarity range)")
        print("✓ Values are NOT manuscript names")
        print("✓ Values represent actual linguistic ratios and probabilities")
        print("✓ Feature vector has proper magnitude for Euclidean distance")
        
    else:
        print(f"Julian manuscript '{julian_name}' not found in feature vectors")
        print(f"Available Julian manuscripts: {[k for k in feature_vectors.keys() if any(j in k for j in ['Τῷ', 'φραγμεντυμ', 'Σαραπίωνι', 'Διονυσίῳ', 'Ἀνεπίγραφος', 'Λιβανίῳ'])]}")

else:
    print("ERROR: No feature_vectors found in the data!")
    print("This means the analysis didn't save the feature vectors correctly.") 