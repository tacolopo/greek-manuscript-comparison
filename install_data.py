#!/usr/bin/env python3
"""
Script to download required NLTK and CLTK data.
"""

import os
import sys
import nltk
from cltk.corpus.utils.importer import CorpusImporter

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    print("NLTK data downloaded successfully.")

def download_cltk_data():
    """Download required CLTK data for Greek."""
    print("Downloading CLTK data for Greek...")
    corpus_importer = CorpusImporter('greek')
    
    try:
        # Download Greek models
        corpus_importer.import_corpus('greek_models_cltk')
        print("Greek models downloaded successfully.")
        
        # Download Greek texts
        corpus_importer.import_corpus('greek_text_perseus')
        print("Greek Perseus texts downloaded successfully.")
        
        # Download additional resources if available
        corpus_importer.import_corpus('greek_treebank_perseus')
        print("Greek treebank downloaded successfully.")
        
    except Exception as e:
        print(f"Warning: Error downloading some CLTK data: {e}")
        print("You may need to manually download some resources.")

def main():
    """Main function."""
    print("Installing required data for Greek Manuscript Comparison Tool...")
    
    try:
        # Download NLTK data
        download_nltk_data()
        
        # Download CLTK data
        download_cltk_data()
        
        print("\nAll required data downloaded successfully!")
        print("You can now use the Greek Manuscript Comparison Tool.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 