# Greek New Testament Similarity Weight Analysis Instructions

This document provides instructions for running the weight iteration analysis on the Greek New Testament texts.

## Overview

The analysis is performed in two steps:
1. Run multiple iterations of similarity calculations with different weight configurations
2. Generate a comprehensive analysis comparing the results across iterations

## Prerequisites

The scripts require Python 3.6+ and several dependencies. To install the required packages:

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn tqdm networkx pyvis tabulate
```

## Step 1: Run the Weight Iterations

The first script runs multiple iterations of the analysis with different weight configurations:

```bash
./iterate_similarity_weights.py --clusters 8 --method hierarchical --base-output-dir similarity_iterations
```

### Parameters:

- `--clusters`: Number of clusters to use (default: 8)
- `--method`: Clustering method - hierarchical, kmeans, or dbscan (default: hierarchical)
- `--base-output-dir`: Directory to store results (default: similarity_iterations)
- `--advanced-nlp`: Whether to use advanced NLP features (default: True)

This script will run 5 iterations with different weight configurations:

1. **baseline** - Current balanced weights
   - vocabulary: 0.25, sentence: 0.15, transitions: 0.15, ngrams: 0.25, syntactic: 0.20

2. **nlp_only** - Only advanced NLP features
   - vocabulary: 0.0, sentence: 0.0, transitions: 0.0, ngrams: 0.0, syntactic: 1.0

3. **equal** - Equal weights for all features
   - vocabulary: 0.2, sentence: 0.2, transitions: 0.2, ngrams: 0.2, syntactic: 0.2

4. **vocabulary_focused** - Focus on vocabulary and n-grams
   - vocabulary: 0.4, sentence: 0.07, transitions: 0.06, ngrams: 0.4, syntactic: 0.07

5. **structure_focused** - Focus on sentence structure and transitions
   - vocabulary: 0.07, sentence: 0.4, transitions: 0.4, ngrams: 0.06, syntactic: 0.07

The script creates a separate output directory for each iteration, saves the similarity matrices, and generates a comparison directory with charts.

## Step 2: Generate the Summary Analysis

After all iterations are complete, run the summary script:

```bash
./weight_sensitivity_summary.py --input-dir similarity_iterations --output-dir weight_sensitivity_summary
```

### Parameters:

- `--input-dir`: Directory containing the iteration results (default: similarity_iterations)
- `--output-dir`: Directory to save the summary analysis (default: weight_sensitivity_summary)
- `--comparison-dir`: Directory with comparison files (default: {input_dir}/comparison)

## Output Files

The analysis generates several key output files:

1. **similarity_weight_analysis.md** - Main summary with key findings
2. **configuration_correlation.png** - Heatmap showing correlations between different weight configurations
3. **book_sensitivity.md** - Analysis of which books are most affected by weight changes
4. **weight_configurations.md** - Summary of the weight configurations used

## Interpreting the Results

### Configuration Correlation

The correlation heatmap shows how similar the book relationships are across different weight configurations:
- High correlation (closer to 1.0) indicates that changing weights doesn't significantly affect the relative similarities between books
- Low correlation suggests that the choice of weights substantially changes which books appear similar

### Book Sensitivity

The book sensitivity analysis shows which texts are most affected by weight changes:
- Books with high average variation have relationships that change significantly depending on which features are emphasized
- This may indicate texts with mixed authorship or unique stylistic characteristics

### Most Sensitive Pairs

The most sensitive pairs analysis shows which specific book relationships change the most across weight configurations:
- Large variations may indicate pairs with similarities in some features but differences in others
- This can help identify which specific feature types are most important for distinguishing between these texts

## Customizing the Analysis

To modify the weight configurations, edit the `create_weight_configs()` function in `iterate_similarity_weights.py`.

To add additional visualizations or analyses, modify the `create_comparison_charts()` function in `iterate_similarity_weights.py` or the `generate_weight_sensitivity_summary()` function in `weight_sensitivity_summary.py`.

## Extending the Analysis

The framework can be extended in several ways:
1. Add more weight configurations to test additional hypotheses
2. Add comparison with known author attributions to evaluate which weights best match historical consensus
3. Incorporate additional stylometric features or analysis methods 