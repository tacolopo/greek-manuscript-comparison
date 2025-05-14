# Pauline Letters Stylometric Analysis

## Directory Structure

### Main Analysis Directories

- **pauline_analysis_unbiased**: Contains the unbiased clustering analysis that lets stylistic patterns emerge naturally without predetermined groupings
- **pauline_enhanced_plots**: Contains enhanced visualizations with clearer labels and side-by-side comparisons
- **pauline_analysis/legacy_biased_analysis**: Contains the old analysis that used predetermined disputed/undisputed groupings

### Key Files

- **unbiased_pauline_analysis_summary.md**: Final summary report of the unbiased analysis
- **pauline_enhanced_plots/mds_comparison.png**: Side-by-side comparison of different feature weightings
- **pauline_analysis_unbiased/unbiased_cluster_report.md**: Detailed report of the unbiased clustering results

### Scripts

- **unbiased_pauline_clustering.py**: Creates the unbiased MDS plots and clustering analysis
- **improve_pauline_plots.py**: Enhances the plots with better labels and creates comparison views
- **reorganize_analysis_files.py**: This script that organizes the file structure

## Methodology

The analysis uses multidimensional scaling (MDS) to visualize stylistic similarities between Pauline letters, with hierarchical clustering to identify natural groupings. Five different feature weighting configurations were tested to understand how prioritizing different aspects of writing style affects perceived relationships. Importantly, no predetermined groupings (like disputed/undisputed) were used in the analysis.
