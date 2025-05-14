# Pauline Letters Similarity Analysis Summary

## Overview
This report summarizes the stylometric analysis of Paul's letters in the New Testament. The analysis uses various weight configurations to measure similarity between the letters, examining how different aspects of writing style (vocabulary, sentence structure, transitions, n-grams, and syntactic features) affect the perceived relationships between these texts.

## Key Findings

### General Patterns
1. The average similarity across all Pauline letters is slightly negative, indicating significant stylistic diversity within the corpus.
2. Certain letter groups consistently show high similarity:
   - The major letters (Romans, 1 & 2 Corinthians) 
   - The Pastoral Epistles (1 & 2 Timothy, Titus)
3. Philemon (PHM) tends to be the most dissimilar from other letters across most configurations.

### By Weight Configuration

#### Baseline Configuration
- Strong connections between Romans, 1 & 2 Corinthians (similarity > 0.85)
- Clear grouping of the Pastoral Epistles (1 & 2 Timothy, Titus)
- Extreme dissimilarity between Romans and Titus (-0.7776)
- Most letters show negative similarity with Philemon

#### NLP-Only Configuration
- Extremely high similarities (1.0000) between certain pairs: 
  - Philippians and 1 Thessalonians
  - 1 Thessalonians and Ephesians
- Very strong connection between 1 & 2 Timothy (0.9999)
- Extreme negative similarities (near -1.0) between certain pairs
- This configuration shows the highest variation (std. dev: 0.7313)
- The extreme values suggest the syntactic features may be less discriminating on their own

#### Vocabulary-Focused Configuration
- Strongest connection between Romans and 2 Corinthians (0.9812)
- Very strong connections within the Corinthians-Romans group
- Extremely negative similarities between Romans and Titus (-0.9466)
- This configuration amplifies the differences between letter groups

#### Structure-Focused Configuration
- Strong connection between 1 Corinthians and Romans (0.9031)
- Strong pastoral group (1 & 2 Timothy, Titus)
- Interesting connection between 1 Corinthians and Galatians (0.8724)
- Most negative similarity between Galatians and Ephesians (-0.7765)

## Letter Groups Identified

Based on the dendrograms and similarity matrices, several consistent letter groupings emerge:

1. **Major Pauline Letters Group**
   - Romans, 1 Corinthians, 2 Corinthians
   - These show consistently high similarity across all configurations
   - Occasionally Galatians joins this group (especially in structure-focused analysis)

2. **Pastoral Epistles Group**
   - 1 Timothy, 2 Timothy, Titus
   - These letters form a distinct cluster in all configurations
   - They often show negative similarity with the major letters

3. **Prison Letters Group**
   - Ephesians, Philippians, Colossians
   - These show variable similarity depending on weight configuration
   - In the NLP-only configuration, they have extremely high similarity

4. **Outliers**
   - Philemon: Consistently dissimilar from most other letters
   - 2 Thessalonians: Often grouped separately from 1 Thessalonians

## Implications for Authorship Studies

The analysis demonstrates how different stylometric features yield different similarity patterns:

1. When emphasizing vocabulary (lexical choices), the major Pauline letters show strong cohesion, while showing strong dissimilarity with the Pastoral Epistles.

2. When emphasizing structural patterns (sentence structure and transitions), some unexpected connections emerge (like 1 Corinthians and Galatians).

3. The NLP-only analysis (focusing on syntactic features) shows extreme values in both directions, suggesting these features alone may not be as reliable for discrimination.

4. The baseline configuration, which balances all feature types, shows moderate clustering that aligns with traditional Pauline letter groupings.

These findings highlight the importance of considering multiple stylometric dimensions when analyzing authorship, as different aspects of writing style may suggest different relationships between texts.

## Conclusion

The Pauline letters show distinct stylistic patterns that vary based on which aspects of writing are emphasized. This analysis demonstrates the complex stylistic relationships within the Pauline corpus and how different analytical approaches can highlight different aspects of these relationships.

The most consistent finding is the clear stylistic distinction between the major letters (Romans, Corinthians) and the Pastoral Epistles (Timothy, Titus), which aligns with scholarly debates about potential differences in authorship within the Pauline corpus. 