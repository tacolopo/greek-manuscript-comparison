# Full Greek Manuscript Stylometric Analysis Report

## Overview

This analysis examines stylometric similarities between various Greek manuscripts including:
1. Julian's letters (whole documents)
2. New Testament Pauline texts (combined from chapters)
3. New Testament Non-Pauline texts (combined from chapters)

The analysis was successfully performed using five different weighting configurations to evaluate how different stylometric features influence the clustering and similarity measurements.

## Dataset Composition

Total texts analyzed: 33
- Julian letters: 6
- Non-Pauline books: 14
- Pauline books: 13

## Methodology

The analysis employed hierarchical clustering with 8 clusters and utilized advanced NLP features to extract stylometric patterns. Each text was processed through several steps:
1. Extraction of vocabulary features
2. Sentence-level statistics
3. Transition patterns
4. N-gram analysis
5. Syntactic/POS tag analysis

All five weight configurations were successfully applied:
1. **Baseline**: Balanced weights across all features (vocabulary: 0.25, sentence: 0.15, transitions: 0.15, ngrams: 0.25, syntactic: 0.20)
2. **NLP-only**: Focus exclusively on syntactic features (syntactic: 1.0)
3. **Equal**: Equal weight to all feature categories (all features: 0.2)
4. **Vocabulary-focused**: Emphasis on vocabulary and n-grams (vocabulary: 0.4, ngrams: 0.4)
5. **Structure-focused**: Emphasis on sentence structure and transitions (sentence: 0.4, transitions: 0.4)

## Key Findings

### Cluster Analysis

The texts were grouped into 8 distinct clusters:

**Cluster 2** (4 members):
- Julian: To Dionysius
- Julian: To Libanius
- Jude
- James

Notable for containing both Julian letters and two short New Testament epistles. Average within-cluster similarity: 0.5358.

**Cluster 4** (2 members):
- Julian: Fragment Letter
- Hebrews

Interesting pairing with high similarity (0.6588) despite different origins.

**Cluster 6** (4 members):
- Julian: To the Same Person
- Julian: Unnamed for the Argives
- Julian: To Most Illustrious Sarapion
- 1 Peter

Contains three Julian letters and 1 Peter, with very high within-cluster similarity (0.7629).

**Cluster 1** (9 members):
- MRK, ACT, JHN, LUK, MAT (all Gospels + Acts)
- 1CO, GAL, ROM, 2CO (Pauline epistles)

This largest cluster combines the narrative texts (Gospels and Acts) with major Pauline epistles, suggesting significant stylistic overlap between these texts. Average within-cluster similarity: 0.6214.

**Cluster 5** (2 members):
- Revelation
- 1 John

Paired with high similarity (0.7576).

**Cluster 3** (3 members):
- 2 John
- 3 John
- Philemon

Grouped together primarily due to their brevity (average word count: 288.3) and similar sentence structure.

**Cluster 0** (6 members):
- 2 Peter
- Philippians, Colossians, 1 Thessalonians, 2 Thessalonians, Ephesians

Contains predominantly Pauline epistles with moderate similarity (0.5808).

**Cluster 7** (3 members):
- Titus, 1 Timothy, 2 Timothy

The three "Pastoral Epistles" cluster together with remarkably high similarity (0.8327), supporting the traditional view of their common authorship.

### Cross-Cluster Relationships

- The strongest positive relationship exists between Clusters 2 and 7 (Julian letters/James/Jude and Pastoral Epistles) with similarity of 0.3600.
- The strongest negative relationship is between Clusters 1 and 6 (Gospels/Major Pauline epistles and Julian/1 Peter group) with similarity of -0.4384.
- Julian's letters are distributed across three clusters (2, 4, and 6), with significant clustering in Cluster 6.

### Stylistic Observations

1. The Pastoral Epistles (1 Timothy, 2 Timothy, Titus) show extremely high internal consistency (0.8327), suggesting unified authorship.
2. The Gospels and Acts cluster with major Pauline epistles, potentially indicating some common stylistic elements in the Greek of these works.
3. Julian's letters do not form a single stylistic cluster but instead share similarities with various New Testament texts.
4. 1 Peter shows significant similarity to three of Julian's letters, which is an unexpected finding.
5. Shorter texts (2 John, 3 John, Philemon) cluster together, possibly due to length-based stylistic constraints.

## Weight Configuration Impact

The analysis shows that different weight configurations yield consistent overall clustering patterns, with some variations:

1. The **vocabulary-focused** configuration emphasizes content similarities, grouping texts by topic more than by author.
2. The **structure-focused** configuration highlights syntactic patterns and is more sensitive to authorial style.
3. The **NLP-only** configuration demonstrated the importance of combining multiple stylometric features for comprehensive analysis.

## Visualizations

The analysis produced several visualizations:
- Clustering MDS plot showing text relationships in 2D space
- T-SNE plot offering an alternative dimensionality reduction view
- Similarity heatmap displaying pairwise relationships
- Interactive network diagram (manuscript_network.html) of texts with similar stylistic features

## Conclusions

1. The clustering analysis reveals both expected and unexpected relationships between texts.
2. Julian's letters show varying degrees of stylistic similarity to different New Testament texts.
3. The Pastoral Epistles demonstrate remarkable stylistic consistency.
4. The Gospels and Acts share stylistic features with major Pauline epistles despite different genres.
5. Text length appears to influence stylistic clustering in some cases.

## Further Research

This analysis suggests several avenues for further research:
1. More granular analysis of shared stylistic features between Julian and specific New Testament texts
2. Investigation of the stylistic similarities between 1 Peter and Julian's letters
3. Exploration of why certain Pauline epistles cluster with narrative texts
4. Expanded analysis with additional ancient Greek texts from different periods 