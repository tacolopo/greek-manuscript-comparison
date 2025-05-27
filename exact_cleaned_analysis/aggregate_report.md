# Aggregate Analysis Report
## Overview
This report represents the aggregate analysis across all five weight configurations:
- Baseline (balanced weights)
- NLP-only (syntactic features only)
- Equal (equal weights for all features)
- Vocabulary-focused (emphasis on vocabulary and n-grams)
- Structure-focused (emphasis on sentence structure and transitions)

The similarity matrix used for this analysis is an average of the matrices from all five configurations.

Clustering method: hierarchical
Number of clusters: 6

## Weight Configurations Used
The following weight configurations were averaged:
### baseline
Current balanced weights
Weights: {'vocabulary': 0.25, 'sentence': 0.15, 'transitions': 0.15, 'ngrams': 0.25, 'syntactic': 0.2}

### nlp_only
Only advanced NLP/syntactic features
Weights: {'vocabulary': 0.0, 'sentence': 0.0, 'transitions': 0.0, 'ngrams': 0.0, 'syntactic': 1.0}

### equal
Equal weights for all features
Weights: {'vocabulary': 0.2, 'sentence': 0.2, 'transitions': 0.2, 'ngrams': 0.2, 'syntactic': 0.2}

### vocabulary_focused
Focus on vocabulary and n-grams
Weights: {'vocabulary': 0.4, 'sentence': 0.07, 'transitions': 0.06, 'ngrams': 0.4, 'syntactic': 0.07}

### structure_focused
Focus on sentence structure and transitions
Weights: {'vocabulary': 0.07, 'sentence': 0.4, 'transitions': 0.4, 'ngrams': 0.06, 'syntactic': 0.07}

## Results
See the visualization files for detailed clustering results.