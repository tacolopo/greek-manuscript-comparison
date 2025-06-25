#!/usr/bin/env python3
"""
Enhanced NLP Analysis Script for Greek Manuscripts

This script performs data-driven clustering analysis of Greek manuscripts
using advanced NLP features and multiple clustering algorithms.
No assumptions are made about authorship - the analysis is purely data-driven.
"""

import os
import glob
from src import MultipleManuscriptComparison

def collect_manuscripts_by_book(data_dir: str) -> dict:
    """
    Collect and combine manuscript files by complete books/letters, including Julian's letters.
    
    Args:
        data_dir: Path to data directory containing text files
        
    Returns:
        Dictionary mapping book names to combined text content
    """
    # Mapping from codes to readable names for biblical texts
    book_name_mapping = {
        '070_MAT': 'Matthew',
        '071_MRK': 'Mark', 
        '072_LUK': 'Luke',
        '073_JHN': 'John',
        '074_ACT': 'Acts',
        '075_ROM': 'Romans',
        '076_1CO': '1 Corinthians',
        '077_2CO': '2 Corinthians',
        '078_GAL': 'Galatians',
        '079_EPH': 'Ephesians',
        '080_PHP': 'Philippians',
        '081_COL': 'Colossians',
        '082_1TH': '1 Thessalonians',
        '083_2TH': '2 Thessalonians',
        '084_1TI': '1 Timothy',
        '085_2TI': '2 Timothy',
        '086_TIT': 'Titus',
        '087_PHM': 'Philemon',
        '088_HEB': 'Hebrews',
        '089_JAS': 'James',
        '090_1PE': '1 Peter',
        '091_2PE': '2 Peter',
        '092_1JN': '1 John',
        '093_2JN': '2 John',
        '094_3JN': '3 John',
        '095_JUD': 'Jude',
        '096_REV': 'Revelation'
    }
    
    # Mapping for Julian's letters (Greek titles to English translations)
    julian_name_mapping = {
        'Διονυσίῳ.txt': 'Julian: To Dionysius',
        'Λιβανίῳ σοφιστῇ καὶ κοιαίστωρι.txt': 'Julian: To Libanius the Sophist',
        'Σαραπίωνι τῷ λαμπροτάτῳ.txt': 'Julian: To Sarapion the Most Illustrious',
        'Τῷ αὐτῷ.txt': 'Julian: To the Same Person',
        'φραγμεντυμ επιστολαε.txt': 'Julian: Letter Fragment',
        'Ἀνεπίγραφος ὑπὲρ Ἀργείων.txt': 'Julian: Untitled Letter about the Argives'
    }
    
    manuscripts = {}
    
    # Process biblical texts (multi-chapter books)
    chapter_files = {}
    for root, dirs, files in os.walk(data_dir):
        # Skip Julian directory for now
        if 'Julian' in root:
            continue
            
        for file in files:
            if file.endswith('.txt') and '_read' in file:
                full_path = os.path.join(root, file)
                # Extract book code (e.g., "075_ROM" from "grcsbl_075_ROM_01_read.txt")
                name = file.replace('_read.txt', '').replace('grcsbl_', '')
                parts = name.split('_')
                if len(parts) >= 2:
                    book_key = f"{parts[0]}_{parts[1]}"  # e.g., "075_ROM"
                    if book_key not in chapter_files:
                        chapter_files[book_key] = []
                    chapter_files[book_key].append(full_path)
    
    # Combine chapters into complete books
    for book_key, file_paths in chapter_files.items():
        # Sort files by chapter number
        file_paths.sort(key=lambda x: int(x.split('_')[-2]))  # Sort by chapter number
        
        # Read and combine all chapters
        combined_text = ""
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chapter_text = f.read().strip()
                    if chapter_text:
                        combined_text += chapter_text + "\n\n"
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
        
        if combined_text.strip():
            # Use readable book name
            readable_name = book_name_mapping.get(book_key, book_key)
            manuscripts[readable_name] = combined_text.strip()
    
    # Process Julian's letters (single files)
    julian_dir = os.path.join(data_dir, 'Julian')
    if os.path.exists(julian_dir):
        for file in os.listdir(julian_dir):
            if file.endswith('.txt'):
                file_path = os.path.join(julian_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            # Use English translation for the name
                            readable_name = julian_name_mapping.get(file, f"Julian: {file.replace('.txt', '')}")
                            manuscripts[readable_name] = text
                except Exception as e:
                    print(f"Warning: Could not read Julian letter {file}: {e}")
    
    return manuscripts

def main():
    """Main analysis function."""
    print("=== Enhanced Greek Manuscript NLP Clustering Analysis ===")
    print("This analysis makes NO assumptions about authorship.")
    print("Clustering is purely data-driven based on linguistic features.\n")
    
    # Collect manuscripts
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        return
    
    print("Collecting manuscripts by complete books/letters...")
    manuscripts = collect_manuscripts_by_book(data_dir)
    
    if not manuscripts:
        print("No manuscripts found! Please check the data directory.")
        return
    
    print(f"Found {len(manuscripts)} complete books/letters")
    
    # Show all manuscripts since there should be a reasonable number
    print(f"\nComplete books/letters to analyze:")
    for i, (name, text) in enumerate(manuscripts.items()):
        word_count = len(text.split())
        print(f"  {i+1}. {name} ({word_count:,} words)")
    
    # Initialize the enhanced comparison system
    print("\nInitializing enhanced NLP analysis system...")
    
    try:
        comparator = MultipleManuscriptComparison(use_advanced_nlp=True)
        
        # Prepare manuscript texts and names
        manuscript_texts = list(manuscripts.values())
        manuscript_names = list(manuscripts.keys())
        
        # Run the complete analysis
        print("\nRunning complete enhanced clustering analysis...")
        print("This analyzes complete books/letters, not individual chapters...")
        
        results = comparator.run_complete_analysis_from_texts(
            manuscript_texts=manuscript_texts,
            manuscript_names=manuscript_names,
            output_dir="enhanced_clustering_results"
        )
        
        print("\n" + "="*60)
        print("ENHANCED ANALYSIS COMPLETE!")
        print("="*60)
        print(f"Results saved to: enhanced_clustering_results/")
        print("\nThe analysis used:")
        print("✓ Advanced vocabulary richness metrics")
        print("✓ Sentence complexity analysis")
        print("✓ Function word usage patterns")
        print("✓ Morphological diversity measures")
        print("✓ Semantic embeddings (when available)")
        print("✓ Multiple clustering algorithms (K-Means, Hierarchical, GMM, Spectral, DBSCAN)")
        print("✓ Comprehensive validation metrics")
        print("✓ Feature selection and dimensionality reduction")
        print("\nCheck the report and visualizations for detailed results!")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        print("This might be due to missing dependencies or data issues.")
        print("Check the error details above and ensure all requirements are installed.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 