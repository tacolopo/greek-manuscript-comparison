#!/usr/bin/env python3

import pickle
import numpy as np

# Load the actual analysis results
with open('exact_cleaned_analysis/nlp_only/similarity_matrix.pkl', 'rb') as f:
    data = pickle.load(f)

# Check what keys are available
print('Available keys (manuscript names):')
manuscript_names = list(data.keys())
print(f'Total manuscripts: {len(manuscript_names)}')
print(f'Julian manuscripts: {[name for name in manuscript_names if name in ["Τῷ αὐτῷ", "φραγμεντυμ επιστολαε", "Σαραπίωνι τῷ λαμπροτάτῳ", "Διονυσίῳ", "Ἀνεπίγραφος ὑπὲρ Ἀργείων", "Λιβανίῳ σοφιστῇ καὶ κοιαίστωρι"]]}')
print(f'Pauline manuscripts: {[name for name in manuscript_names if name in ["2CO", "1CO", "ROM", "2TI", "1TI", "PHM", "EPH", "2TH", "PHP", "COL", "1TH", "TIT", "GAL"]]}')

# Look at a Julian text feature vector
julian_name = 'Διονυσίῳ'
if julian_name in data:
    julian_features = data[julian_name]
    print(f'\nJulian "{julian_name}" feature vector length: {len(julian_features)}')
    print(f'Julian syntactic features (last 20): {julian_features[-20:]}')
    
    # Count zeros in syntactic features
    syntactic_features = julian_features[-20:]
    zero_count = np.sum(syntactic_features == 0)
    print(f'Zero features in Julian: {zero_count}/20')
    
    # Check which features are zero
    feature_names = [
        'noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio', 'function_word_ratio',
        'pronoun_ratio', 'conjunction_ratio', 'particle_ratio', 'interjection_ratio', 'numeral_ratio',
        'tag_diversity', 'tag_entropy', 'noun_after_verb_ratio', 'adj_before_noun_ratio', 'adv_before_verb_ratio',
        'verb_to_noun_prob', 'noun_to_verb_prob', 'noun_to_adj_prob', 'adj_to_noun_prob', 'noun_verb_ratio'
    ]
    
    print('\nJulian feature breakdown:')
    for i, (name, value) in enumerate(zip(feature_names, syntactic_features)):
        status = "ZERO" if value == 0 else f"{value:.4f}"
        print(f'  {name}: {status}')

# Also check a Pauline text
pauline_name = '1CO'
if pauline_name in data:
    pauline_features = data[pauline_name]
    syntactic_features = pauline_features[-20:]
    zero_count = np.sum(syntactic_features == 0)
    print(f'\nZero features in Pauline 1CO: {zero_count}/20')
    
    print('\nPauline 1CO feature breakdown:')
    for i, (name, value) in enumerate(zip(feature_names, syntactic_features)):
        status = "ZERO" if value == 0 else f"{value:.4f}"
        print(f'  {name}: {status}')

# Check all manuscripts for zero feature patterns
print(f'\n' + '='*50)
print('ZERO FEATURE ANALYSIS ACROSS ALL MANUSCRIPTS:')

all_zero_counts = {}
for name in manuscript_names:
    features = data[name]
    syntactic_features = features[-20:]
    zero_count = np.sum(syntactic_features == 0)
    all_zero_counts[name] = zero_count

# Group by corpus
julian_names = ["Τῷ αὐτῷ", "φραγμεντυμ επιστολαε", "Σαραπίωνι τῷ λαμπροτάτῳ", "Διονυσίῳ", "Ἀνεπίγραφος ὑπὲρ Ἀργείων", "Λιβανίῳ σοφιστῇ καὶ κοιαίστωρι"]
pauline_names = ["2CO", "1CO", "ROM", "2TI", "1TI", "PHM", "EPH", "2TH", "PHP", "COL", "1TH", "TIT", "GAL"]

print('\nJulian manuscripts zero counts:')
for name in julian_names:
    if name in all_zero_counts:
        print(f'  {name}: {all_zero_counts[name]}/20 zeros')

print('\nPauline manuscripts zero counts:')
for name in pauline_names:
    if name in all_zero_counts:
        print(f'  {name}: {all_zero_counts[name]}/20 zeros')

# Calculate averages
julian_zeros = [all_zero_counts[name] for name in julian_names if name in all_zero_counts]
pauline_zeros = [all_zero_counts[name] for name in pauline_names if name in all_zero_counts]

print(f'\nAverage zero features:')
print(f'  Julian: {np.mean(julian_zeros):.1f}/20')
print(f'  Pauline: {np.mean(pauline_zeros):.1f}/20') 