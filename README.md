# Greek Manuscript Comparison Tool

This tool analyzes and compares Greek manuscripts to calculate similarity scores based on linguistic features. It uses various NLP techniques to identify patterns in language use, word choice, sentence structure, and other textual characteristics.

## Features

- Text preprocessing for Greek manuscripts
- Multiple similarity metrics (lexical, syntactic, semantic)
- Visualization of similarity results
- Support for various text formats

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download required NLTK and CLTK data:
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   python -c "from cltk.corpus.utils.importer import CorpusImporter; corpus_importer = CorpusImporter('greek'); corpus_importer.import_corpus('greek_models_cltk'); corpus_importer.import_corpus('greek_text_perseus')"
   ```

## Usage

1. Place your Greek manuscript text files in the `data` directory
2. Run the comparison script:
   ```
   python src/compare_manuscripts.py --file1 data/manuscript1.txt --file2 data/manuscript2.txt
   ```
3. View the similarity report in the console output and generated visualizations

## Similarity Metrics

The tool calculates similarity using several approaches:

1. **Lexical Similarity**: Based on shared vocabulary and n-grams
2. **Syntactic Similarity**: Based on sentence structure patterns
3. **Semantic Similarity**: Based on contextual word usage
4. **Stylometric Features**: Analysis of author-specific writing patterns

## License

MIT 