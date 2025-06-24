# Greek Manuscript NLP Analysis

A streamlined machine learning system for analyzing ancient Greek texts using natural language processing techniques. This system focuses on preprocessing, feature extraction, similarity calculation, clustering, and visualization.

## Features

- **Text Preprocessing**: Cleans and normalizes ancient Greek texts
- **Advanced NLP**: Uses CLTK, Greek BERT, and spaCy for linguistic analysis
- **Feature Extraction**: Extracts vocabulary, syntactic, and stylistic features
- **Similarity Calculation**: Computes NLP-based text similarities
- **Clustering**: Groups texts using machine learning algorithms
- **Visualization**: Generates plots and reports for analysis results

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Your Data**:
   - Place Greek text files (.txt) in the `data/` directory
   - The system will automatically discover and process all text files

3. **Run Analysis**:
   ```bash
   python run_nlp_analysis.py
   ```

4. **View Results**:
   - Analysis report: `nlp_analysis_output/analysis_report.txt`
   - Similarity matrix: `nlp_analysis_output/similarity_matrix.csv`
   - Visualizations: `nlp_visualizations/` directory

## System Components

### Core Modules

- **`src/preprocessing.py`**: Text cleaning and normalization
- **`src/advanced_nlp.py`**: Advanced NLP feature extraction using CLTK and BERT
- **`src/features.py`**: Essential linguistic feature extraction
- **`src/similarity.py`**: NLP-based similarity calculations
- **`src/multi_comparison.py`**: Main analysis pipeline
- **`src/visualization.py`**: Plot generation (if present)

### Main Analysis Script

- **`run_nlp_analysis.py`**: Complete workflow demonstration

## Example Usage

```python
from src import MultipleManuscriptComparison

# Initialize the analysis system
comparison = MultipleManuscriptComparison(
    output_dir="results",
    visualizations_dir="plots"
)

# Define your manuscripts
manuscripts = {
    "text1": "path/to/text1.txt",
    "text2": "path/to/text2.txt",
    # ... more texts
}

# Run the complete analysis
results = comparison.compare_multiple_manuscripts(
    manuscripts=manuscripts,
    method='hierarchical',
    n_clusters=None  # Auto-determine
)
```

## Output Files

The system generates several output files:

1. **Analysis Report** (`analysis_report.txt`): Comprehensive text report with clustering results and feature importance
2. **Similarity Matrix** (`similarity_matrix.csv`): Pairwise similarity scores between all texts
3. **MDS Plot** (`mds_clustering.png`): 2D visualization using Multi-Dimensional Scaling
4. **t-SNE Plot** (`tsne_clustering.png`): Alternative 2D visualization using t-SNE
5. **Similarity Heatmap** (`similarity_heatmap.png`): Matrix visualization of similarities

## NLP Features

The system extracts multiple types of linguistic features:

### Vocabulary Features
- Type-token ratio (vocabulary richness)
- Hapax legomena ratio (unique words)
- Vocabulary size metrics

### Sentence Features  
- Mean and standard deviation of sentence lengths
- Sentence count

### Syntactic Features (Advanced NLP)
- Part-of-speech tag ratios (nouns, verbs, adjectives, etc.)
- Syntactic diversity metrics
- Grammatical pattern analysis

### Character N-gram Features
- TF-IDF weighted character sequences
- Pattern frequency distributions

## Dependencies

Key Python packages required:

- `scikit-learn`: Machine learning algorithms
- `numpy`, `pandas`: Data processing
- `matplotlib`, `seaborn`: Visualization
- `cltk`: Classical Language Toolkit for ancient Greek
- `transformers`: Greek BERT model
- `sentence-transformers`: Semantic similarity
- `spacy`: Named entity recognition (optional)
- `nltk`: N-gram processing

## Data Format

Text files should contain clean Greek text in UTF-8 encoding. The system handles:

- Ancient Greek with or without diacritics
- Mixed case text (automatically normalized)
- Various punctuation marks
- Chapter/verse markers (automatically removed)

## Clustering Methods

Two clustering algorithms are available:

1. **Hierarchical Clustering** (default): Good for nested authorship relationships
2. **K-Means Clustering**: Good for spherical clusters

The system automatically determines the optimal number of clusters using silhouette analysis.

## License

This project is designed for academic research in computational linguistics and digital humanities. 