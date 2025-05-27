# Aggregate Analysis of Greek Manuscript Comparison

## Overview

This report presents an aggregate analysis of multiple text comparisons using different weight configurations.

The following weight configurations were used:
- **Baseline**: Vocabulary (70%), Structure (30%), NLP (0%)
- **Equal**: Vocabulary (33%), Structure (33%), NLP (34%)
- **NLP Only**: Vocabulary (0%), Structure (0%), NLP (100%)
- **Structure Focused**: Vocabulary (20%), Structure (70%), NLP (10%)
- **Vocabulary Focused**: Vocabulary (70%), Structure (10%), NLP (20%)

## Clustering Results

```
Cluster analysis:

Cluster 5:
  - 1CO
  - 2CO
  - MRK

Cluster 3:
  - GAL
  - ROM
  - HEB
  - REV
  - Julian_φραγμεντυμ επιστολαε

Cluster 0:
  - PHM
  - PHP
  - 1TI
  - EPH
  - JAS
  - 1JN
  - 2PE
  - Julian_Διονυσίῳ
  - Julian_Σαραπίωνι τῷ λαμπροτάτῳ

Cluster 2:
  - TIT
  - 2TH
  - 2TI
  - 3JN

Cluster 4:
  - COL
  - 1TH
  - 1PE
  - JUD
  - 2JN
  - Julian_Τῷ αὐτῷ
  - Julian_Ἀνεπίγραφος ὑπὲρ Ἀργείων
  - Julian_Λιβανίῳ σοφιστῇ καὶ κοιαίστωρι

Cluster 1:
  - ACT
  - JHN
  - LUK
  - MAT

```



## Interpretation

The clusters above represent the average grouping of texts across all five weight configurations.
This provides a more robust analysis than any single configuration alone.
Texts appearing in the same cluster consistently are more likely to share stylometric similarities.

## Note on Cleaned Pauline Text Analysis

This analysis was performed on a version of the texts with Old Testament quotations removed.
This provides a more accurate assessment of writing style without the influence of quoted material.
Each book was processed as a complete text by combining all its chapters rather than analyzing individual chapters separately.
