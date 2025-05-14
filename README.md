# Greek Manuscript Comparison: Unbiased Stylometric Analysis of Pauline Letters

This project conducts an unbiased stylometric analysis of Pauline letters in the New Testament, allowing natural stylistic patterns to emerge from the data without imposing predetermined categories such as "disputed" vs "undisputed" authorship.

## Project Overview

Traditional approaches to Pauline authorship studies often begin with predetermined classifications, potentially introducing bias. This project takes a data-driven approach, using multidimensional scaling (MDS) and hierarchical clustering to identify natural stylistic relationships across five different feature weighting configurations:

1. **Baseline**: Balanced weighting of all stylometric features
2. **Vocabulary-Focused**: Emphasizes word choice and lexical features
3. **Structure-Focused**: Prioritizes sentence patterns and structural elements
4. **NLP-Only**: Focuses solely on linguistic and syntactic features
5. **Equal Weights**: Assigns equal importance to all feature categories

## Key Findings

The analysis revealed several important insights:

- **Consistent Groupings**: Some letter groups consistently cluster together across feature weightings:
  - The major doctrinal letters (Romans, 1 & 2 Corinthians, Galatians) formed a remarkably stable group (4.5/5 configurations)
  - The Pastoral Epistles (1 & 2 Timothy, Titus) showed strong stylistic affinity (4.3/5 configurations)

- **Feature Sensitivity**: Feature weighting dramatically affected perceived relationships between letters. The same corpus analyzed with different feature weights produced significantly different stylistic groupings.

- **Unstable Affiliations**: Some letters showed inconsistent cluster membership across different feature weightings:
  - Philemon appeared in 4 different clusters, consistent with its unique nature as a brief, personal letter
  - Letters like 1 Thessalonians, Philippians, and the Pastoral Epistles showed moderate instability

- **Natural Stylistic Patterns**: Rather than confirming predetermined categories, the data revealed organic groupings that sometimes, but not always, aligned with traditional classifications.

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download required NLTK and CLTK data:
   ```
   python install_data.py
   ```

## Key Scripts

### Primary Analysis Scripts

- **unbiased_pauline_clustering.py**: Performs clustering analysis on Pauline letters without predetermined groupings, using 5 different feature weighting configurations.
  ```
  python unbiased_pauline_clustering.py
  ```

- **improve_pauline_plots.py**: Enhances the visualizations with better labels and creates side-by-side comparisons.
  ```
  python improve_pauline_plots.py
  ```

- **reorganize_analysis_files.py**: Organizes analysis files, moving biased analysis results to a legacy folder.
  ```
  python reorganize_analysis_files.py
  ```

### Output Directories

- **pauline_analysis_unbiased/**: Contains the unbiased clustering analysis results
- **pauline_enhanced_plots/**: Contains enhanced visualizations with clearer labels and side-by-side comparisons
- **pauline_analysis/legacy_biased_analysis/**: Contains the old analysis that used predetermined disputed/undisputed groupings

## Methodology

The analysis follows these steps:

1. **Feature Extraction**: Calculate stylometric features from each Pauline letter, including:
   - Lexical features (vocabulary richness, unique words, word frequencies)
   - Syntactic features (sentence length, complexity, part-of-speech patterns)
   - Semantic features (word embeddings, contextual relationships)
   - Structural features (discourse markers, connectives, rhetorical elements)

2. **Similarity Calculation**: Compute pairwise similarities between letters using various metrics

3. **Multidimensional Scaling**: Visualize these similarities in 2D space

4. **Hierarchical Clustering**: Identify natural groupings of letters based on stylistic similarities

5. **Cross-Configuration Analysis**: Compare clustering results across different feature weightings

## Summary Reports

- **unbiased_pauline_analysis_summary.md**: Comprehensive summary of the unbiased analysis
- **pauline_analysis_unbiased/unbiased_cluster_report.md**: Detailed report of the clustering results

## Implications

This unbiased analysis challenges simplistic notions of clear stylistic boundaries between Pauline letters:

1. **Methodological Impacts**: Feature selection dramatically affects results, highlighting how methodological choices influence stylometric conclusions.

2. **Complex Stylistic Networks**: The data suggests complex stylistic relationships within the Pauline corpus that defy simple categorization.

3. **Scientific Approach**: By avoiding predetermined categories, this analysis demonstrates a more scientifically sound approach to stylometric studies.

## License

MIT 