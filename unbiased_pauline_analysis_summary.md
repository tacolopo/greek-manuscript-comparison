# Unbiased Stylometric Analysis of Pauline Letters: Final Report

## Overview

This analysis examines the stylistic relationships between New Testament letters attributed to Paul using an unbiased clustering approach. Unlike previous analyses that imposed predetermined categories of "disputed" and "undisputed" letters, this approach allows natural stylistic patterns to emerge from the data itself.

The analysis uses multidimensional scaling (MDS) to visualize stylistic similarities, with hierarchical clustering to identify natural groupings. Crucially, five different feature weighting configurations were tested to understand how prioritizing different aspects of writing style affects perceived relationships.

## Key Findings

### 1. Consistent Stylistic Groupings

Some letters consistently cluster together across feature weightings:

- **Romans, 1 & 2 Corinthians, and Galatians** form a remarkably stable group, appearing together in the same cluster in nearly all configurations (4.5/5)
- **1 & 2 Timothy and Titus** (the Pastoral Epistles) show strong stylistic affinity (4.3/5)
- **Philemon and 2 Thessalonians** frequently appear in the same cluster (4/5)
- **Ephesians and Colossians** share stylistic features across most configurations

### 2. Weight Configurations Reveal Different Patterns

Each weighting configuration highlights different aspects of stylistic similarity:

- **Baseline Configuration**: Balanced weighting reveals four distinct clusters:
  - Major doctrinal letters (1CO, 2CO, ROM, GAL)
  - Pastoral group (TIT, 1TI, 2TI) + Philippians
  - Prison letters group (COL, EPH) + 1 Thessalonians
  - Philemon and 2 Thessalonians stand apart

- **Vocabulary-Focused**: Emphasizing lexical features produces similar clusters to baseline, but strengthens the separation between groups

- **Structure-Focused**: Sentence patterns reveal connections between:
  - Major letters (1CO, GAL, ROM, 2CO)
  - Pastoral letters (TIT, 1TI, 2TI) + Philippians
  - A mixed group (PHM, COL, 2TH, EPH)
  - 1 Thessalonians stands apart

- **NLP-Only**: Focusing solely on syntactic features produces the most divergent results, with unusual groupings and Romans standing alone

### 3. Letters with Unstable Affiliations

Several letters show inconsistent cluster membership across different feature weightings:

- **Philemon** appeared in 4 different clusters, consistent with its unique nature as a brief, personal letter
- **1 Thessalonians, Philippians, and the Pastoral Epistles** showed moderate instability (3 different clusters)

## Implications for Authorship Studies

This unbiased analysis challenges simplistic notions of clear stylistic boundaries between traditionally "disputed" and "undisputed" Pauline letters:

1. **Feature Selection Dramatically Affects Results**: The same corpus analyzed with different feature weights produces significantly different stylistic groupings, highlighting how methodological choices impact conclusions.

2. **Natural Stylistic Groupings Emerge**: Rather than confirming predetermined categories, the data reveals organic groupings that sometimes, but not always, align with traditional classifications.

3. **Stable Cores with Fluid Boundaries**: The analysis reveals "stylistic cores" (Major letters, Pastorals) that remain consistent, while other letters show more fluid stylistic affiliations.

4. **Genre and Purpose Matter**: Stylistic similarities often correlate with the purpose and nature of the letters. The major doctrinal letters (ROM, 1&2CO, GAL) form a consistent group, as do the Pastoral Epistles.

## Limitations and Methodological Considerations

This analysis demonstrates several important methodological considerations for stylometric studies:

1. **Feature Selection Is Critical**: Different aspects of writing style (vocabulary, sentence structure, syntax) produce different clustering patterns.

2. **Letter Length Affects Results**: Shorter letters (like Philemon) may not provide sufficient data for robust stylometric analysis.

3. **Clustering Is Sensitive to Algorithm Choice**: The number of clusters and clustering method impact results; hierarchical clustering with k=4 was used throughout for consistency.

## Conclusion

This unbiased stylometric analysis reveals complex stylistic relationships within the Pauline corpus that defy simple categorization. While some letters consistently cluster together across different feature weightings, others show more fluid stylistic affiliations depending on which aspects of writing style are emphasized.

The analysis demonstrates that predetermined groupings may introduce bias into stylometric studies, and that methodological choices significantly impact results. Rather than supporting or refuting traditional authorship theories, the data suggests a more nuanced view of stylistic relationships that varies with analytical approach.

Most importantly, the findings highlight how different feature weightings lead to different perceived relationships, emphasizing that no single stylometric approach can definitively resolve complex authorship questions. For robust conclusions, stylometric evidence should be considered alongside historical, theological, and contextual factors. 