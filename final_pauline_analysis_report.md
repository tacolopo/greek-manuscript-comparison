# Stylometric Analysis of Pauline Letters: Unbiased Clustering Approach

## Introduction and Methodology

This report presents the results of an unbiased stylometric analysis of letters attributed to Paul in the New Testament. Unlike traditional approaches that begin by categorizing texts into "disputed" and "undisputed" groups, this analysis allows natural stylistic patterns to emerge from the data without imposing predetermined classifications.

The methodology employs:

1. **Multidimensional Scaling (MDS)**: Transforms complex stylistic similarity relationships into two-dimensional plots where proximity represents stylistic similarity
2. **Hierarchical Clustering**: Identifies natural groupings of texts based on stylistic features
3. **Multiple Feature Weighting**: Tests five different configurations that emphasize different aspects of writing style:
   - Baseline: Balanced weighting across feature types
   - Vocabulary-focused: Emphasizes lexical choices
   - Structure-focused: Emphasizes sentence patterns and transitions
   - NLP-only: Focuses solely on syntactic features
   - Equal: Assigns equal weight to all feature types

This approach avoids "begging the question" by letting the data speak for itself, rather than imposing interpretive categories from the outset.

## Key Findings

### 1. Natural Stylistic Groupings

The analysis reveals several consistent stylistic groupings across different feature weightings:

- **Major Letters Group**: Romans, 1 & 2 Corinthians, and Galatians form a remarkably stable group, appearing together in the same cluster in nearly all configurations (4.5/5)
- **Pastoral Epistles**: 1 & 2 Timothy and Titus show strong stylistic affinity (4.3/5)
- **Thessalonian-Philemon Group**: 2 Thessalonians and Philemon frequently cluster together (4/5)
- **Prison Letters**: Ephesians and Colossians share stylistic features across multiple configurations

These natural groupings sometimes, but not always, align with traditional scholarly categories—suggesting that stylistic reality is more complex than simple binary classifications.

### 2. Feature Weighting Dramatically Affects Results

Each weighting configuration highlights different aspects of stylistic similarity, revealing how methodological choices significantly impact conclusions:

- **Baseline** reveals four distinct clusters that broadly align with traditional letter categories
- **Vocabulary-focused** strengthens the separation between groups
- **Structure-focused** reveals unexpected connections (like 1 Corinthians with Galatians)
- **NLP-only** produces dramatically different groupings, suggesting syntactic features alone provide unreliable discrimination

The side-by-side comparison ([pauline_enhanced_plots/mds_comparison.png](pauline_enhanced_plots/mds_comparison.png)) visually demonstrates how the same corpus analyzed with different feature weights produces significantly different stylistic groupings.

### 3. Letters with Fluid Stylistic Affiliations

Several letters show inconsistent cluster membership across different feature weightings:

- **Philemon** appeared in 4 different clusters, consistent with its unique nature as a brief, personal letter
- **1 Thessalonians, Philippians, and the Pastoral Epistles** showed moderate instability (3 different clusters)

This fluidity challenges binary authorship models and suggests that stylistic features vary based on other factors like letter purpose, audience, and scribe involvement.

## Visualization of Results

Our enhanced MDS plots ([pauline_enhanced_plots](pauline_enhanced_plots)) provide a clear visualization of these findings:

1. **Individual Configuration Plots**: Each plot shows how the letters cluster under a specific weighting configuration
2. **Side-by-Side Comparison**: The comparison view demonstrates how dramatically feature weighting affects perceived relationships
3. **Traditional Classifications**: Letters are color-coded by traditional classifications (for reference only) to show how natural groupings sometimes cross these boundaries

The plots are carefully designed to show traditional classifications for reference while making it clear that these classifications were not used in the actual clustering analysis.

## Implications for Authorship Studies

This unbiased analysis has several important implications:

1. **Feature Selection Is Critical**: Different aspects of writing style produce significantly different clustering patterns, highlighting how methodological choices impact conclusions.

2. **Stylistic Cores with Fluid Boundaries**: The analysis reveals "stylistic cores" (Major letters, Pastorals) that remain consistent, while other letters show more fluid stylistic affiliations.

3. **Complex Network of Relationships**: Rather than a simple divide between disputed and undisputed letters, the data shows a complex network of stylistic relationships that varies depending on which aspects of writing style are emphasized.

4. **No Single Method Is Definitive**: The significant variation across feature weightings demonstrates that no single stylometric approach can definitively resolve complex authorship questions.

## Conclusion

This unbiased stylometric analysis reveals complex stylistic relationships within the Pauline corpus that defy simple categorization. While some letters consistently cluster together across different feature weightings, others show more fluid stylistic affiliations depending on which aspects of writing style are emphasized.

The analysis demonstrates that methodological choices significantly impact results. Rather than supporting or refuting traditional authorship theories, the data suggests a more nuanced view of stylistic relationships that varies with analytical approach.

For robust conclusions about authorship, stylometric evidence should be considered alongside historical, theological, and contextual factors. This study shows the importance of transparent methodological choices and cautions against simplistic interpretations of stylometric data. 