#!/usr/bin/env python3
"""
Debug script to test NLP feature extraction.
"""

import sys
sys.path.append('src')

from advanced_nlp import AdvancedGreekProcessor

def test_nlp_extraction():
    """Test the NLP feature extraction on a simple Greek text."""
    
    # Initialize the processor
    print("Initializing AdvancedGreekProcessor...")
    processor = AdvancedGreekProcessor()
    
    # Test with a simple Greek text
    test_text = "καὶ εἶπεν ὁ θεὸς γενηθήτω φῶς καὶ ἐγένετο φῶς"
    print(f"Test text: {test_text}")
    
    # Process the document
    print("\nProcessing document...")
    try:
        nlp_features = processor.process_document(test_text)
        print(f"NLP features keys: {list(nlp_features.keys())}")
        
        if 'pos_tags' in nlp_features:
            pos_tags = nlp_features['pos_tags']
            print(f"POS tags: {pos_tags}")
            print(f"Number of POS tags: {len(pos_tags)}")
            
            # Test syntactic feature extraction
            print("\nExtracting syntactic features...")
            syntactic_features = processor.extract_syntactic_features(pos_tags)
            print(f"Syntactic features: {syntactic_features}")
            print(f"Number of syntactic features: {len(syntactic_features)}")
            
            # Check if any features are non-zero
            non_zero_features = {k: v for k, v in syntactic_features.items() if v != 0}
            print(f"Non-zero features: {non_zero_features}")
            
        else:
            print("No POS tags found in NLP features!")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nlp_extraction() 