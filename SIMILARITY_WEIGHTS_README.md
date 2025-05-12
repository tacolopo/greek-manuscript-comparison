# Greek New Testament Similarity Weight Analysis

This directory contains scripts and tools for analyzing how different similarity weight configurations affect the stylometric analysis of Greek New Testament texts.

## Scripts

1. `iterate_similarity_weights.py` - Main script that runs multiple iterations of the analysis with different weight configurations
2. `weight_sensitivity_summary.py` - Script to generate summary reports and visualizations comparing the results across iterations

## Weight Configurations

Five different weight configurations are tested:

1. **baseline** - Current balanced weights
   - vocabulary: 0.25
   - sentence: 0.15
   - transitions: 0.15
   - ngrams: 0.25
   - syntactic: 0.20

2. **nlp_only** - Only advanced NLP features
   - vocabulary: 0.0
   - sentence: 0.0
   - transitions: 0.0
   - ngrams: 0.0
   - syntactic: 1.0

3. **equal** - Equal weights for all features
   - vocabulary: 0.2
   - sentence: 0.2
   - transitions: 0.2
   - ngrams: 0.2
   - syntactic: 0.2

4. **vocabulary_focused** - Focus on vocabulary and n-grams
   - vocabulary: 0.4
   - sentence: 0.07
   - transitions: 0.06
   - ngrams: 0.4
   - syntactic: 0.07

5. **structure_focused** - Focus on sentence structure and transitions
   - vocabulary: 0.07
   - sentence: 0.4
   - transitions: 0.4
   - ngrams: 0.06
   - syntactic: 0.07

## How to Run

### Step 1: Run the iterations

```bash
./iterate_similarity_weights.py --clusters 8 --method hierarchical --base-output-dir similarity_iterations
```

This will:
- Run 5 iterations with different weight configurations
- Use 8 clusters for each iteration
- Use hierarchical clustering
- Save results to `similarity_iterations` directory

### Step 2: Generate comparison summary

After all iterations have completed, run:

```bash
./weight_sensitivity_summary.py --input-dir similarity_iterations --output-dir weight_sensitivity_summary
```

This will:
- Load results from all iterations
- Generate comparison charts and tables
- Create a comprehensive report analyzing the differences between iterations
- Save all output to `weight_sensitivity_summary` directory

## Output Files

The analysis produces several key outputs:

1. **similarity_weight_analysis.md** - Main summary document with key findings
2. **most_sensitive_pairs.md** - List of book pairs with the largest variation across weight configurations
3. **book_sensitivity.md** - Analysis of which books are most affected by weight changes
4. **configuration_correlation.png** - Heatmap showing correlations between different weight configurations
5. **most_sensitive_pairs.png** - Visualization of book pairs with largest similarity variations
6. **book_sensitivity.png** - Chart of books most affected by weight changes

## Interpreting the Results

The analysis helps answer several important questions:

1. **How stable are the book similarity relationships?** 
   - High correlation between configurations suggests stable relationships
   - Low correlation indicates that weights significantly affect text relationships

2. **Which texts are most affected by weight choices?**
   - Some texts may show large variations in similarity depending on which features are emphasized
   - This can help identify texts with ambiguous or mixed stylistic characteristics

3. **Which feature types most influence clustering?**
   - Comparing clustering results across different weight configurations shows how feature emphasis affects grouping
   - Some clusters may remain stable across all configurations, while others may change significantly

## Understanding the Weight Categories

- **vocabulary** - Features related to vocabulary richness and distribution
- **sentence** - Sentence structure and length statistics
- **transitions** - Writing flow and transition patterns between sentences
- **ngrams** - Character and word pattern frequencies
- **syntactic** - Advanced NLP syntactic features (part-of-speech ratios, etc.) 