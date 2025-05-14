# Unbiased Pauline Letters Clustering Analysis

## Overview
This analysis examines the stylistic similarities between Pauline letters using multidimensional scaling (MDS) and hierarchical clustering, without any predetermined groupings of letters. The goal is to let the data reveal natural groupings based purely on stylometric features.

## Cluster Compositions Across Weight Configurations

### BASELINE Configuration

**Cluster 0:** COL, 1TH, EPH

**Cluster 1:** PHM, 2TH

**Cluster 2:** PHP, TIT, 1TI, 2TI

**Cluster 3:** 1CO, GAL, ROM, 2CO


### NLP_ONLY Configuration

**Cluster 0:** PHM, TIT

**Cluster 1:** PHP, COL, 1TH, 2TH, EPH

**Cluster 2:** 1CO, GAL, 2CO, 1TI, 2TI

**Cluster 3:** ROM


### EQUAL Configuration

**Cluster 0:** COL, 1TH, EPH

**Cluster 1:** PHP, TIT, 1TI, 2TI

**Cluster 2:** 1CO, GAL, ROM, 2CO

**Cluster 3:** PHM, 2TH


### VOCABULARY_FOCUSED Configuration

**Cluster 0:** TIT, COL, 1TI, 2TI

**Cluster 1:** PHP, 1TH, EPH

**Cluster 2:** PHM, 2TH

**Cluster 3:** 1CO, GAL, ROM, 2CO


### STRUCTURE_FOCUSED Configuration

**Cluster 0:** PHP, TIT, 1TI, 2TI

**Cluster 1:** PHM, COL, 2TH, EPH

**Cluster 2:** 1CO, GAL, ROM, 2CO

**Cluster 3:** 1TH


## Cross-Configuration Analysis

### Letters That Consistently Cluster Together

- **1CO** and **2CO** appeared in the same cluster in 5/5 configurations
- **1CO** and **GAL** appeared in the same cluster in 5/5 configurations
- **1TI** and **2TI** appeared in the same cluster in 5/5 configurations
- **2CO** and **GAL** appeared in the same cluster in 5/5 configurations
- **1CO** and **ROM** appeared in the same cluster in 4/5 configurations
- **1TH** and **EPH** appeared in the same cluster in 4/5 configurations
- **1TI** and **TIT** appeared in the same cluster in 4/5 configurations
- **2CO** and **ROM** appeared in the same cluster in 4/5 configurations
- **2TH** and **PHM** appeared in the same cluster in 4/5 configurations
- **2TI** and **TIT** appeared in the same cluster in 4/5 configurations
- **COL** and **EPH** appeared in the same cluster in 4/5 configurations
- **GAL** and **ROM** appeared in the same cluster in 4/5 configurations
- **1TH** and **COL** appeared in the same cluster in 3/5 configurations
- **1TI** and **PHP** appeared in the same cluster in 3/5 configurations
- **2TI** and **PHP** appeared in the same cluster in 3/5 configurations
- **PHP** and **TIT** appeared in the same cluster in 3/5 configurations

### Key Insights

1. **Major Letters Grouping**: On average, the major letters (ROM, 1CO, 2CO, GAL) appeared together 4.5/5 times.

2. **Pastoral Letters Grouping**: On average, the pastoral letters (1TI, 2TI, TIT) appeared together 4.3/5 times.

3. **Letters with Unstable Cluster Membership**:
   - PHM appeared in 4 different clusters across configurations
   - 1TH appeared in 3 different clusters across configurations
   - 1TI appeared in 3 different clusters across configurations
   - 2TH appeared in 3 different clusters across configurations
   - 2TI appeared in 3 different clusters across configurations
   - PHP appeared in 3 different clusters across configurations
   - TIT appeared in 3 different clusters across configurations

### Implications for Stylometric Analysis

This unbiased clustering analysis demonstrates how different feature weights can dramatically affect the perceived relationships between Pauline letters. Rather than supporting a clear division between supposedly 'disputed' and 'undisputed' letters, the data shows a more complex network of stylistic relationships that varies depending on which aspects of writing style are emphasized.

The most consistent groupings appear to be:

1. **Romans, 1 & 2 Corinthians** often cluster together, suggesting stylistic consistency among these longer doctrinal letters

2. **1 & 2 Timothy and Titus** frequently form a distinct stylistic group

3. **Philemon** often stands apart from other letters, likely due to its unique brevity and personal nature

However, the analysis also reveals significant variation in clustering patterns across different weight configurations, highlighting the importance of methodological considerations in stylometric studies. These variations suggest that attributing authorship based on stylometric analysis alone should be approached with caution.
