# Similarity Weight Iteration Analysis

## Weight Configurations

### baseline
Description: Current balanced weights
Weights:
- vocabulary: 0.25
- sentence: 0.15
- transitions: 0.15
- ngrams: 0.25
- syntactic: 0.2

### nlp_only
Description: Only advanced NLP/syntactic features
Weights:
- vocabulary: 0.0
- sentence: 0.0
- transitions: 0.0
- ngrams: 0.0
- syntactic: 1.0

### equal
Description: Equal weights for all features
Weights:
- vocabulary: 0.2
- sentence: 0.2
- transitions: 0.2
- ngrams: 0.2
- syntactic: 0.2

### vocabulary_focused
Description: Focus on vocabulary and n-grams
Weights:
- vocabulary: 0.4
- sentence: 0.07
- transitions: 0.06
- ngrams: 0.4
- syntactic: 0.07

### structure_focused
Description: Focus on sentence structure and transitions
Weights:
- vocabulary: 0.07
- sentence: 0.4
- transitions: 0.4
- ngrams: 0.06
- syntactic: 0.07

## Analysis Results

### Correlation Between Configurations

The correlation heatmap shows how similar the similarity matrices are across different weight configurations.
A high correlation indicates that changing weights does not significantly affect relative relationships between texts.

### Most Variable Book Pairs

These book pairs show the largest differences in similarity scores across weight configurations:

- **TIT vs ACT**: Similarity ranges from -0.581 to 0.631 (difference: 1.212)
  - Lowest with 'vocabulary_focused', highest with 'structure_focused'
- **TIT vs MAT**: Similarity ranges from -0.628 to 0.560 (difference: 1.189)
  - Lowest with 'vocabulary_focused', highest with 'structure_focused'
- **1PE vs REV**: Similarity ranges from -0.593 to 0.589 (difference: 1.182)
  - Lowest with 'vocabulary_focused', highest with 'structure_focused'
- **TIT vs JHN**: Similarity ranges from -0.619 to 0.558 (difference: 1.177)
  - Lowest with 'vocabulary_focused', highest with 'structure_focused'
- **JAS vs 2PE**: Similarity ranges from -0.638 to 0.537 (difference: 1.174)
  - Lowest with 'structure_focused', highest with 'vocabulary_focused'
- **ROM vs HEB**: Similarity ranges from -0.660 to 0.509 (difference: 1.169)
  - Lowest with 'structure_focused', highest with 'vocabulary_focused'
- **1CO vs HEB**: Similarity ranges from -0.566 to 0.512 (difference: 1.079)
  - Lowest with 'structure_focused', highest with 'vocabulary_focused'
- **COL vs 1TI**: Similarity ranges from -0.438 to 0.640 (difference: 1.078)
  - Lowest with 'structure_focused', highest with 'vocabulary_focused'
- **GAL vs HEB**: Similarity ranges from -0.600 to 0.452 (difference: 1.052)
  - Lowest with 'structure_focused', highest with 'vocabulary_focused'
- **TIT vs 2TH**: Similarity ranges from -0.620 to 0.424 (difference: 1.043)
  - Lowest with 'structure_focused', highest with 'vocabulary_focused'

### Clustering Stability

Analyze how book cluster assignments change across different weight configurations.
Books that frequently change clusters are more sensitive to the choice of weights.

