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

- **TIT vs ACT**: Similarity ranges from -0.561 to 0.634 (difference: 1.195)
  - Lowest with 'vocabulary_focused', highest with 'structure_focused'
- **TIT vs MAT**: Similarity ranges from -0.636 to 0.559 (difference: 1.195)
  - Lowest with 'vocabulary_focused', highest with 'structure_focused'
- **ROM vs HEB**: Similarity ranges from -0.656 to 0.536 (difference: 1.192)
  - Lowest with 'structure_focused', highest with 'vocabulary_focused'
- **TIT vs JHN**: Similarity ranges from -0.635 to 0.555 (difference: 1.190)
  - Lowest with 'vocabulary_focused', highest with 'structure_focused'
- **JAS vs 2PE**: Similarity ranges from -0.644 to 0.544 (difference: 1.187)
  - Lowest with 'structure_focused', highest with 'nlp_only'
- **1PE vs REV**: Similarity ranges from -0.572 to 0.593 (difference: 1.165)
  - Lowest with 'vocabulary_focused', highest with 'structure_focused'
- **2CO vs 1JN**: Similarity ranges from -0.703 to 0.434 (difference: 1.137)
  - Lowest with 'structure_focused', highest with 'nlp_only'
- **COL vs 1TI**: Similarity ranges from -0.435 to 0.659 (difference: 1.093)
  - Lowest with 'structure_focused', highest with 'vocabulary_focused'
- **1CO vs HEB**: Similarity ranges from -0.567 to 0.504 (difference: 1.072)
  - Lowest with 'structure_focused', highest with 'vocabulary_focused'
- **LUK vs HEB**: Similarity ranges from -0.550 to 0.509 (difference: 1.059)
  - Lowest with 'structure_focused', highest with 'nlp_only'

### Clustering Stability

Analyze how book cluster assignments change across different weight configurations.
Books that frequently change clusters are more sensitive to the choice of weights.

