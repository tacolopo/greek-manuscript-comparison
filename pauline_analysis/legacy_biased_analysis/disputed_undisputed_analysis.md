# Disputed vs. Undisputed Pauline Letters Analysis

## Overview
This analysis compares the stylistic similarity within and between two groups of Pauline letters:

**Undisputed Letters**: ROM, 1CO, 2CO, GAL, PHP, 1TH, PHM

**Disputed Letters**: EPH, COL, 2TH, 1TI, 2TI, TIT

## Similarity Comparison

| Configuration | Undisputed Internal | Disputed Internal | Cross-Group |
|---------------|--------------------:|------------------:|-----------:|
| Baseline | 0.0636 | 0.0358 | -0.1663 |
| Nlp Only | -0.0288 | -0.1974 | -0.0155 |
| Equal | 0.0405 | -0.0160 | -0.1419 |
| Vocabulary Focused | 0.1277 | 0.1159 | -0.2146 |
| Structure Focused | -0.0252 | -0.0998 | -0.0827 |

## Detailed Analysis

### Baseline Configuration

#### Undisputed Pauline Letters

Average Similarity: 0.0636 (from 21 pairs)

Most Similar Pairs:
- 1CO - ROM: 0.9321
- ROM - 2CO: 0.8750
- 1CO - 2CO: 0.8583

Least Similar Pairs:
- ROM - PHM: -0.6594
- 1CO - PHM: -0.6505
- 2CO - PHM: -0.6201

#### Disputed Pauline Letters

Average Similarity: 0.0358 (from 15 pairs)

Most Similar Pairs:
- TIT - 1TI: 0.7836
- TIT - 2TI: 0.7636
- 1TI - 2TI: 0.7513

Least Similar Pairs:
- TIT - EPH: -0.5539
- 2TH - 1TI: -0.5188
- 1TI - EPH: -0.5183

#### Cross-Group Connections

Average Similarity: -0.1663 (from 42 pairs)

Most Similar Pairs:
- PHP - 2TI: 0.6304
- 1TH - EPH: 0.6070
- PHM - 2TH: 0.4254

Least Similar Pairs:
- ROM - TIT: -0.7776
- 2CO - TIT: -0.7702
- 1CO - TIT: -0.6166

### Nlp Only Configuration

#### Undisputed Pauline Letters

Average Similarity: -0.0288 (from 21 pairs)

Most Similar Pairs:
- PHP - 1TH: 1.0000
- 1CO - 2CO: 0.9893
- GAL - 2CO: 0.9454

Least Similar Pairs:
- ROM - PHM: -0.9647
- GAL - 1TH: -0.7505
- GAL - PHP: -0.7460

#### Disputed Pauline Letters

Average Similarity: -0.1974 (from 15 pairs)

Most Similar Pairs:
- 1TI - 2TI: 0.9999
- COL - EPH: 0.9994
- TIT - 1TI: 0.7700

Least Similar Pairs:
- TIT - EPH: -0.9612
- TIT - COL: -0.9510
- COL - 1TI: -0.9296

#### Cross-Group Connections

Average Similarity: -0.0155 (from 42 pairs)

Most Similar Pairs:
- 1TH - EPH: 1.0000
- PHP - EPH: 0.9999
- COL - 1TH: 0.9991

Least Similar Pairs:
- GAL - 2TH: -0.9906
- 2CO - 2TH: -0.9811
- PHP - TIT: -0.9649

### Equal Configuration

#### Undisputed Pauline Letters

Average Similarity: 0.0405 (from 21 pairs)

Most Similar Pairs:
- 1CO - ROM: 0.9227
- ROM - 2CO: 0.8130
- 1CO - 2CO: 0.7736

Least Similar Pairs:
- ROM - PHM: -0.6767
- 1CO - PHM: -0.6617
- 2CO - PHM: -0.5592

#### Disputed Pauline Letters

Average Similarity: -0.0160 (from 15 pairs)

Most Similar Pairs:
- TIT - 1TI: 0.7997
- 1TI - 2TI: 0.7873
- TIT - 2TI: 0.7839

Least Similar Pairs:
- 2TH - 1TI: -0.6096
- TIT - EPH: -0.5649
- 1TI - EPH: -0.5424

#### Cross-Group Connections

Average Similarity: -0.1419 (from 42 pairs)

Most Similar Pairs:
- PHP - 2TI: 0.5826
- 1TH - EPH: 0.5397
- PHP - EPH: 0.3915

Least Similar Pairs:
- 2CO - TIT: -0.6718
- ROM - TIT: -0.5976
- GAL - EPH: -0.5452

### Vocabulary Focused Configuration

#### Undisputed Pauline Letters

Average Similarity: 0.1277 (from 21 pairs)

Most Similar Pairs:
- ROM - 2CO: 0.9812
- 1CO - 2CO: 0.9358
- 1CO - ROM: 0.9355

Least Similar Pairs:
- PHM - PHP: -0.6404
- 2CO - PHM: -0.5967
- GAL - PHP: -0.5376

#### Disputed Pauline Letters

Average Similarity: 0.1159 (from 15 pairs)

Most Similar Pairs:
- TIT - 1TI: 0.8809
- TIT - 2TI: 0.7876
- 1TI - 2TI: 0.7283

Least Similar Pairs:
- TIT - EPH: -0.5532
- 1TI - EPH: -0.4176
- 2TH - 1TI: -0.3553

#### Cross-Group Connections

Average Similarity: -0.2146 (from 42 pairs)

Most Similar Pairs:
- PHP - 2TI: 0.7050
- 1TH - EPH: 0.6004
- PHM - 2TH: 0.5878

Least Similar Pairs:
- ROM - TIT: -0.9466
- 2CO - TIT: -0.9291
- 1CO - TIT: -0.8580

### Structure Focused Configuration

#### Undisputed Pauline Letters

Average Similarity: -0.0252 (from 21 pairs)

Most Similar Pairs:
- 1CO - ROM: 0.9031
- 1CO - GAL: 0.8724
- GAL - ROM: 0.8079

Least Similar Pairs:
- ROM - PHM: -0.7421
- 1CO - PHM: -0.7038
- 2CO - PHP: -0.6323

#### Disputed Pauline Letters

Average Similarity: -0.0998 (from 15 pairs)

Most Similar Pairs:
- 1TI - 2TI: 0.8811
- TIT - 1TI: 0.8500
- TIT - 2TI: 0.8403

Least Similar Pairs:
- 2TH - 1TI: -0.7257
- TIT - COL: -0.6857
- COL - 2TI: -0.6611

#### Cross-Group Connections

Average Similarity: -0.0827 (from 42 pairs)

Most Similar Pairs:
- PHP - 2TI: 0.4877
- 1TH - EPH: 0.4370
- PHM - COL: 0.4176

Least Similar Pairs:
- GAL - EPH: -0.7765
- 1CO - EPH: -0.6628
- 1CO - COL: -0.5998

## Interpretation

### Key Observations

1. The undisputed Pauline letters generally show higher internal similarity than the disputed letters, suggesting stronger stylistic consistency within the undisputed corpus.

2. The cross-group similarity is lower than within-group similarity, suggesting stylistic differences between the undisputed and disputed letters.

3. Different weight configurations yield different similarity patterns, highlighting the importance of considering multiple stylometric dimensions in authorship analysis.

### Implications for Pauline Authorship

The stylometric evidence presents a complex picture that neither clearly confirms nor refutes traditional authorship views. Some key implications:

- The NLP-only configuration shows extreme variations, suggesting syntactic features alone may not be reliable discriminators for authorship in this corpus.

- The vocabulary-focused analysis reveals stronger distinctions between groups, which may reflect either different authorship or different subject matter across the letters.

- The baseline configuration, balancing multiple features, shows moderate differentiation between groups while still preserving some cross-group connections.

These findings show how stylometric analysis can contribute to the authorship debate while also demonstrating the limitations of purely computational approaches to such complex questions.