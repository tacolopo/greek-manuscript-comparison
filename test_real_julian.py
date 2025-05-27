#!/usr/bin/env python3

import sys
sys.path.append('src')
from src.advanced_nlp import AdvancedGreekProcessor
from collections import Counter

# Test with actual Julian text
with open('data/Julian/Διονυσίῳ.txt', 'r', encoding='utf-8') as f:
    julian_text = f.read()[:2000]  # First 2000 characters

print('Julian text sample:')
print(julian_text[:300] + '...')
print()

processor = AdvancedGreekProcessor()
features = processor.process_document(julian_text)
pos_tags = features['pos_tags']

print(f'Total POS tags: {len(pos_tags)}')
print(f'Unique POS tags: {set(pos_tags)}')
print()

tag_counts = Counter(pos_tags)
for tag, count in sorted(tag_counts.items()):
    print(f'{tag}: {count}')

print()
syntactic_features = processor.extract_syntactic_features(pos_tags)
zero_features = [k for k, v in syntactic_features.items() if v == 0.0]
non_zero_features = [k for k, v in syntactic_features.items() if v != 0.0]

print(f'Zero features ({len(zero_features)}/20): {zero_features}')
print(f'Non-zero features ({len(non_zero_features)}/20): {non_zero_features}')

# Check specific words that should be particles/pronouns
print('\nChecking specific Greek words in text:')
words_to_check = ['μέν', 'δέ', 'γάρ', 'οὖν', 'ἀλλὰ', 'αὐτός', 'οὗτος', 'ἐκεῖνος', 'τις', 'τί']
for word in words_to_check:
    if word in julian_text:
        print(f'  Found: {word}')
    else:
        print(f'  Missing: {word}') 