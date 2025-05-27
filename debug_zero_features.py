#!/usr/bin/env python3
"""
Debug script to investigate why we're getting many zero features.
"""

import sys
sys.path.append('src')
from src.advanced_nlp import AdvancedGreekProcessor
import numpy as np
from collections import Counter

def debug_zero_features():
    """Debug why we're getting zero features."""
    
    # Initialize processor
    processor = AdvancedGreekProcessor()
    
    # Test with a longer, more complex Greek text that should have diverse syntax
    test_text = '''
    καὶ εἶπεν ὁ θεὸς γενηθήτω φῶς καὶ ἐγένετο φῶς καὶ εἶδεν ὁ θεὸς τὸ φῶς ὅτι καλόν 
    καὶ διεχώρισεν ὁ θεὸς ἀνὰ μέσον τοῦ φωτὸς καὶ ἀνὰ μέσον τοῦ σκότους καὶ ἐκάλεσεν 
    ὁ θεὸς τὸ φῶς ἡμέραν καὶ τὸ σκότος ἐκάλεσεν νύκτα καὶ ἐγένετο ἑσπέρα καὶ ἐγένετο 
    πρωί ἡμέρα μία τρεῖς τέσσαρες πέντε ἓξ ἑπτὰ ὀκτὼ ἐννέα δέκα ἀλλὰ οὖν μὲν δὲ γάρ
    '''
    
    print('Testing feature extraction with longer text...')
    features = processor.process_document(test_text)
    pos_tags = features['pos_tags']
    
    print(f'Total POS tags: {len(pos_tags)}')
    print(f'Unique POS tags: {set(pos_tags)}')
    print(f'\nPOS tag counts:')
    tag_counts = Counter(pos_tags)
    for tag, count in sorted(tag_counts.items()):
        print(f'  {tag}: {count}')
    
    print(f'\nExtracting syntactic features...')
    syntactic_features = processor.extract_syntactic_features(pos_tags)
    print(f'\nSyntactic features:')
    for key, value in syntactic_features.items():
        print(f'  {key}: {value:.4f}')
    
    # Count zeros
    zero_features = [k for k, v in syntactic_features.items() if v == 0.0]
    non_zero_features = [k for k, v in syntactic_features.items() if v != 0.0]
    
    print(f'\nZero features ({len(zero_features)}/20): {zero_features}')
    print(f'Non-zero features ({len(non_zero_features)}/20): {non_zero_features}')
    
    # Let's also test with a real Julian text sample
    print(f'\n' + '='*50)
    print('Testing with actual Julian text sample...')
    
    # Load a Julian text
    try:
        with open('data/julian_letters/julian_letter_1.txt', 'r', encoding='utf-8') as f:
            julian_text = f.read()[:1000]  # First 1000 characters
            
        print(f'Julian text sample: {julian_text[:200]}...')
        
        julian_features = processor.process_document(julian_text)
        julian_pos = julian_features['pos_tags']
        
        print(f'Julian POS tags: {len(julian_pos)}')
        print(f'Julian unique tags: {set(julian_pos)}')
        
        julian_syntactic = processor.extract_syntactic_features(julian_pos)
        julian_zeros = [k for k, v in julian_syntactic.items() if v == 0.0]
        
        print(f'Julian zero features ({len(julian_zeros)}/20): {julian_zeros}')
        
    except FileNotFoundError:
        print('Julian text file not found, skipping real text test')
    
    # Analyze what might be causing zeros
    print(f'\n' + '='*50)
    print('ANALYSIS:')
    print('Common reasons for zero features:')
    print('1. particle_ratio = 0: No particles detected (μέν, δέ, γάρ, etc.)')
    print('2. interjection_ratio = 0: No interjections detected (ὦ, φεῦ, etc.)')
    print('3. numeral_ratio = 0: No numerals detected (εἷς, δύο, τρεῖς, etc.)')
    print('4. pronoun_ratio = 0: No pronouns detected (αὐτός, οὗτος, etc.)')
    print('5. adj_before_noun patterns: Rare in Greek (usually noun-adjective)')
    print('6. Transition probabilities: May be zero in short texts')
    
    # Check if CLTK is properly identifying these categories
    print(f'\nCLTK tag mapping check:')
    expected_tags = ['particle', 'interjection', 'numeral', 'pronoun']
    found_tags = set(pos_tags)
    
    for expected in expected_tags:
        if expected in found_tags:
            print(f'  ✓ {expected}: Found')
        else:
            print(f'  ✗ {expected}: Missing')

if __name__ == "__main__":
    debug_zero_features() 