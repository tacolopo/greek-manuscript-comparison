# Greek Manuscript Comparison Tool

This tool analyzes and compares Greek manuscripts to calculate similarity scores based on linguistic features. It uses various NLP techniques to identify patterns in language use, word choice, sentence structure, and other textual characteristics.

## Features

- Text preprocessing for Greek manuscripts
- Multiple similarity metrics (lexical, syntactic, semantic)
- Visualization of similarity results
- Support for various text formats
- **Advanced NLP Features**:
  - Part-of-speech tagging for Greek texts
  - Syntactic parsing and analysis
  - Semantic similarity using transformer models
  - Word embedding analysis
- **Multiple Manuscript Comparison**:
  - Compare more than two manuscripts simultaneously
  - Cluster manuscripts by similarity
  - Generate interactive network visualizations
  - Comprehensive cluster analysis reports

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

## Usage

### Comparing Two Manuscripts

1. Place your Greek manuscript text files in the `data` directory
2. Run the comparison script:
   ```
   python src/compare_manuscripts.py --file1 data/manuscript1.txt --file2 data/manuscript2.txt
   ```
3. View the similarity report in the console output and generated visualizations

### Comparing Multiple Manuscripts

1. Place multiple Greek manuscript text files in the `data` directory
2. Run the multiple comparison script:
   ```
   python compare_multiple.py --dir data --clusters 3 --method hierarchical
   ```
3. Or use the sample texts:
   ```
   python compare_multiple.py --sample --clusters 2
   ```
4. View the cluster visualizations and reports in the output directory

### Command-line Options for Multiple Comparison

```
usage: compare_multiple.py [-h] (--files FILES [FILES ...] | --dir DIR | --pattern PATTERN | --sample) [--ext EXT]
                          [--names NAMES [NAMES ...]] [--clusters CLUSTERS]
                          [--method {kmeans,hierarchical,dbscan}] [--threshold THRESHOLD]
                          [--advanced-nlp] [--output-dir OUTPUT_DIR] [--vis-dir VIS_DIR]

Compare multiple Greek manuscripts

optional arguments:
  -h, --help            show this help message and exit
  --files FILES [FILES ...]
                        List of manuscript files to compare
  --dir DIR             Directory containing manuscript files to compare
  --pattern PATTERN     Glob pattern to match manuscript files
  --sample              Use sample manuscripts
  --ext EXT             File extension for manuscripts (default: .txt)
  --names NAMES [NAMES ...]
                        Names for the manuscripts (must match number of files)
  --clusters CLUSTERS   Number of clusters to create (default: 3)
  --method {kmeans,hierarchical,dbscan}
                        Clustering method (default: hierarchical)
  --threshold THRESHOLD
                        Similarity threshold for network visualization (default: 0.5)
  --advanced-nlp        Use advanced NLP features
  --output-dir OUTPUT_DIR
                        Output directory (default: output)
  --vis-dir VIS_DIR     Visualizations directory (default: visualizations)
```

## Similarity Metrics

The tool calculates similarity using several approaches:

1. **Lexical Similarity**: Based on shared vocabulary and n-grams
2. **Syntactic Similarity**: Based on sentence structure patterns
3. **Semantic Similarity**: Based on contextual word usage
4. **Stylometric Features**: Analysis of author-specific writing patterns

## Advanced NLP Features

The advanced NLP module provides:

1. **Part-of-Speech Tagging**: Identify grammatical components in Greek texts
2. **Syntactic Parsing**: Analyze sentence structure and dependencies
3. **Semantic Analysis**: Measure semantic similarity using transformer models
4. **Word Embeddings**: Analyze contextual relationships between words

## Multiple Manuscript Comparison

The multiple comparison module allows:

1. **Batch Processing**: Compare any number of manuscripts simultaneously
2. **Clustering**: Group manuscripts by similarity using various algorithms
3. **Interactive Visualization**: Network and cluster visualizations
4. **Detailed Reports**: Comprehensive analysis of manuscript relationships

## License

MIT 