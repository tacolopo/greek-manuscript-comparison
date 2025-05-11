#!/usr/bin/env python3
"""
Script to download and set up required NLP models.
"""

import os
from cltk import NLP
from cltk.data.fetch import FetchCorpus

def setup_nlp_models():
    """Download and set up required NLP models."""
    print("Setting up CLTK models for Ancient Greek...")
    
    # Create CLTK data directory if it doesn't exist
    cltk_home = os.path.expanduser('~/cltk_data')
    os.makedirs(cltk_home, exist_ok=True)
    
    # Download Greek models and resources
    corpus_downloader = FetchCorpus(language='grc')
    
    # Core models and data
    models = [
        'grc_models_cltk',
        'greek_treebank_perseus',
        'greek_lexica_perseus',
        'greek_proper_names_cltk',
        'greek_word2vec_cltk',
        'greek_training_set_sentence_cltk'
    ]
    
    for model in models:
        print(f"\nDownloading {model}...")
        try:
            corpus_downloader.import_corpus(model)
            print(f"Successfully downloaded {model}")
        except Exception as e:
            print(f"Could not download {model}: {str(e)}")
    
    # Initialize Greek pipeline to trigger additional downloads
    print("\nInitializing Greek NLP pipeline...")
    try:
        nlp = NLP(language="grc")
        print("Successfully initialized Greek pipeline")
    except Exception as e:
        print(f"Error initializing Greek pipeline: {str(e)}")
    
    print("\nDownloading spaCy models...")
    import spacy
    try:
        spacy.cli.download("el_core_news_lg")
        print("Successfully downloaded Greek spaCy model")
    except:
        print("Could not download Greek spaCy model. This is not critical.")
    
    print("\nSetup complete!")

if __name__ == "__main__":
    setup_nlp_models() 