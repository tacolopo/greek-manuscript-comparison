#!/usr/bin/env python3
"""
Script to run a full analysis on Greek manuscript texts including:
- Julian letters (whole letters)
- Non-Pauline Texts (combining chapters into books)
- Paul Texts (combining chapters into books)

Uses the five iterations with various weights method and produces visualizations.
"""

import os
import sys
import argparse
import glob
import re
from collections import defaultdict
import shutil
import traceback
import pickle
import numpy as np
import pandas as pd

# Try imports with appropriate error handling
try:
    from src.multi_comparison import MultipleManuscriptComparison
    from src.similarity import SimilarityCalculator
    # Check if iterate_similarity_weights exists and import functions
    if os.path.exists('iterate_similarity_weights.py'):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from iterate_similarity_weights import create_weight_configs
        
        # We will NOT use the original calculate_similarity_across_weights
        # because it has different argument expectations
        use_original_calc_func = False
    else:
        # Define these functions locally if the module doesn't exist
        def create_weight_configs():
            """Define the weight configurations for different iterations."""
            weight_configs = [
                # 1. Current weights (baseline)
                {
                    'name': 'baseline',
                    'description': 'Current balanced weights',
                    'weights': {
                        'vocabulary': 0.25,
                        'sentence': 0.15,
                        'transitions': 0.15,
                        'ngrams': 0.25,
                        'syntactic': 0.20
                    }
                },
                # 2. NLP-only configuration (syntactic features only)
                {
                    'name': 'nlp_only',
                    'description': 'Only advanced NLP/syntactic features',
                    'weights': {
                        'vocabulary': 0.0,
                        'sentence': 0.0,
                        'transitions': 0.0,
                        'ngrams': 0.0,
                        'syntactic': 1.0
                    }
                },
                # 3. Equal weights
                {
                    'name': 'equal',
                    'description': 'Equal weights for all features',
                    'weights': {
                        'vocabulary': 0.2,
                        'sentence': 0.2,
                        'transitions': 0.2,
                        'ngrams': 0.2,
                        'syntactic': 0.2
                    }
                },
                # 4. Vocabulary/Language-focused
                {
                    'name': 'vocabulary_focused',
                    'description': 'Focus on vocabulary and n-grams',
                    'weights': {
                        'vocabulary': 0.4,
                        'sentence': 0.07,
                        'transitions': 0.06,
                        'ngrams': 0.4,
                        'syntactic': 0.07
                    }
                },
                # 5. Structure-focused
                {
                    'name': 'structure_focused',
                    'description': 'Focus on sentence structure and transitions',
                    'weights': {
                        'vocabulary': 0.07,
                        'sentence': 0.4,
                        'transitions': 0.4,
                        'ngrams': 0.06,
                        'syntactic': 0.07
                    }
                }
            ]
            return weight_configs
        use_original_calc_func = False

    def create_aggregate_visualizations(similarity_matrices, display_names, args):
        """
        Create aggregated visualizations from the different weight configurations.
        
        Args:
            similarity_matrices: List of dictionaries with similarity matrices from different configs
            display_names: Dictionary mapping book codes to display names
            args: Command line arguments with output directories
        """
        try:
            if not similarity_matrices:
                print("Warning: No similarity matrices to aggregate.")
                return
                
            print("\nCreating aggregate visualizations from all weight configurations...")
            
            # Extract similarity matrices
            matrices = []
            for data in similarity_matrices:
                if 'similarity_matrix' in data and data['similarity_matrix'] is not None:
                    matrices.append(data['similarity_matrix'])
            
            if not matrices:
                print("Warning: No valid similarity matrices found for aggregation.")
                return
                
            # Create an average similarity matrix
            try:
                # Convert all matrices to numpy arrays with the same shape
                matrix_shape = None
                np_matrices = []
                
                # Ensure all matrices have the same index/columns
                all_indices = sorted(list(set().union(*[m.index for m in matrices])))
                
                # Reindex all matrices to have the same index/columns
                for matrix in matrices:
                    # First make sure each matrix has the same indices
                    full_matrix = matrix.reindex(index=all_indices, columns=all_indices, fill_value=0)
                    np_matrices.append(full_matrix.values)
                
                # Stack matrices and calculate mean along the first axis
                stacked = np.stack(np_matrices)
                avg_matrix = np.mean(stacked, axis=0)
                
                # Create a DataFrame from the average matrix
                avg_df = pd.DataFrame(avg_matrix, index=all_indices, columns=all_indices)
                
                # Save the average matrix
                output_path = os.path.join(args.output_dir, 'average_similarity_matrix.csv')
                avg_df.to_csv(output_path)
                
                # Create visualizations using the average matrix
                comparison = MultipleManuscriptComparison(
                    use_advanced_nlp=args.advanced_nlp,
                    output_dir=args.output_dir,
                    visualizations_dir=args.viz_dir
                )
                
                # Create clustering result for the average matrix
                average_clustering_result = comparison.cluster_manuscripts(
                    similarity_df=avg_df,
                    n_clusters=args.clusters,
                    method=args.method
                )
                
                # Generate visualizations based on the average similarity matrix
                comparison.generate_visualizations(
                    clustering_result=average_clustering_result,
                    similarity_df=avg_df,
                    threshold=0.5
                )
                
                # Create a combined clustering report
                report_lines = [
                    "# Aggregate Analysis Report",
                    "## Overview",
                    f"This report represents the aggregate analysis across all five weight configurations:",
                    "- Baseline (balanced weights)",
                    "- NLP-only (syntactic features only)",
                    "- Equal (equal weights for all features)",
                    "- Vocabulary-focused (emphasis on vocabulary and n-grams)",
                    "- Structure-focused (emphasis on sentence structure and transitions)",
                    "",
                    "The similarity matrix used for this analysis is an average of the matrices from all five configurations.",
                    "",
                    f"Clustering method: {args.method}",
                    f"Number of clusters: {args.clusters}",
                    "",
                    "## Weight Configurations Used",
                    "The following weight configurations were averaged:"
                ]
                
                for data in similarity_matrices:
                    config_name = data.get('name', 'Unknown')
                    config_desc = data.get('description', 'No description')
                    config_weights = data.get('weights', {})
                    report_lines.append(f"### {config_name}")
                    report_lines.append(f"{config_desc}")
                    report_lines.append(f"Weights: {config_weights}")
                    report_lines.append("")
                
                report_lines.append("## Results")
                report_lines.append("See the visualization files for detailed clustering results.")
                
                # Save the aggregate report
                aggregate_report_path = os.path.join(args.output_dir, 'aggregate_report.md')
                with open(aggregate_report_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(report_lines))
                
                print(f"Aggregate visualizations and report created successfully.")
                
            except Exception as e:
                print(f"Error creating aggregate visualizations: {e}")
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error in aggregate visualization creation: {e}")
            traceback.print_exc()


    def calculate_similarity_across_weights(combined_books, display_names, args, weight_configs):
        """Run similarity analysis with different weight configurations."""
        print("Running custom similarity analysis with weight iterations...")
        
        # Initialize comparison object
        comparison = MultipleManuscriptComparison(
            use_advanced_nlp=args.advanced_nlp,
            output_dir=args.output_dir,
            visualizations_dir=args.viz_dir
        )
        
        similarity_matrices = []
        
        for config in weight_configs:
            print(f"\nRunning analysis with {config['name']} configuration:")
            print(f"  - {config['description']}")
            
            # Set custom weights in the similarity calculator
            custom_calculator = SimilarityCalculator()
            if hasattr(custom_calculator, 'set_weights'):
                custom_calculator.set_weights(config['weights'])
            else:
                # Fall back to direct assignment if set_weights method doesn't exist
                custom_calculator.weights = config['weights']
                print(f"DEBUG - Setting weights: {custom_calculator.weights}")
            
            # Replace the default calculator with our custom one
            comparison.similarity_calculator = custom_calculator
            
            # Run the analysis
            iteration_output_dir = os.path.join(args.output_dir, config['name'])
            os.makedirs(iteration_output_dir, exist_ok=True)
            
            # Create iteration-specific visualization directory
            iteration_viz_dir = os.path.join(iteration_output_dir, "visualizations")
            os.makedirs(iteration_viz_dir, exist_ok=True)
            
            # Set the output directory for this iteration
            comparison.output_dir = iteration_output_dir
            comparison.visualizations_dir = iteration_viz_dir
            
            # Run the comparison
            try:
                print(f"Processing manuscripts and extracting features...")
                result = comparison.compare_multiple_manuscripts(
                    manuscripts=combined_books,
                    display_names=display_names,
                    method=args.method,
                    n_clusters=args.clusters,
                    use_advanced_nlp=args.advanced_nlp
                )
                
                # Handle the result properly with both key possibilities
                # Note: different versions of the code may return different key names
                if 'similarity_df' in result:
                    sim_matrix = result['similarity_df']
                elif 'similarity_matrix' in result:
                    sim_matrix = result['similarity_matrix']
                else:
                    # If neither key exists, extract similarity matrix from clustering result
                    sim_matrix = result.get('clustering_result', {}).get('similarity_matrix', None)
                    if sim_matrix is None:
                        print(f"Error: Could not find similarity matrix in result for {config['name']} configuration.")
                        continue
                
                # Get clustering results
                clustering_result = result.get('clustering_result', {})
                
                # Generate the clustering report if not already in the result
                if 'report' not in result:
                    try:
                        from src.reporting import generate_clustering_report
                        report = generate_clustering_report(
                            clustering_result=clustering_result,
                            similarity_matrix=sim_matrix,
                            display_names=display_names,
                            method=args.method,
                            n_clusters=args.clusters
                        )
                        result['report'] = report
                    except (ImportError, Exception) as e:
                        print(f"Warning: Could not generate clustering report: {e}")
                        # Create a basic report manually
                        report_lines = [
                            f"Clustering Report for {config['name']} Configuration",
                            f"==========================================",
                            f"Method: {args.method}",
                            f"Number of Clusters: {args.clusters}",
                            f"Weight Configuration: {config['weights']}",
                            f"",
                            f"Results are available in the visualization files and similarity matrix."
                        ]
                        result['report'] = "\n".join(report_lines)
                
                # Save the similarity matrix and configuration to the iteration-specific directory
                # Save similarity matrix CSV
                matrix_csv_path = os.path.join(iteration_output_dir, 'similarity_matrix.csv')
                if hasattr(sim_matrix, 'to_csv'):
                    sim_matrix.to_csv(matrix_csv_path)
                
                # Save clustering report
                report_path = os.path.join(iteration_output_dir, 'clustering_report.txt')
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(result.get('report', f"Clustering report for {config['name']} configuration"))
                
                # Generate iteration-specific visualizations directly in the iteration directory
                try:
                    # Force regeneration of visualizations in the iteration-specific directory
                    comparison.generate_visualizations(
                        clustering_result=clustering_result,
                        similarity_df=sim_matrix,
                        threshold=0.5
                    )
                    print(f"Visualizations for {config['name']} configuration saved to {iteration_viz_dir}")
                except Exception as e:
                    print(f"Warning: Error generating visualizations for {config['name']}: {e}")
                
                # Save additional result data for reference
                try:
                    result_data_path = os.path.join(iteration_output_dir, 'clustering_result.pkl')
                    with open(result_data_path, 'wb') as f:
                        pickle.dump(clustering_result, f)
                except Exception as e:
                    print(f"Warning: Error saving clustering result data: {e}")
                
                # Save the similarity matrix and configuration
                matrix_data = {
                    'similarity_matrix': sim_matrix,
                    'name': config['name'],
                    'description': config['description'],
                    'weights': config['weights'],
                    'clusters': clustering_result.get('clusters', []),
                    'cluster_centers': clustering_result.get('cluster_centers', []),
                    'labels': clustering_result.get('labels', [])
                }
                
                similarity_matrices.append(matrix_data)
                
            except Exception as e:
                print(f"Error during analysis with {config['name']} configuration: {e}")
                traceback.print_exc()
                continue
        
        # Create aggregate visualizations after all iterations are complete
        create_aggregate_visualizations(similarity_matrices, display_names, args)
        
        return similarity_matrices

except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you have all the required dependencies installed.")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run full Greek manuscript analysis")
    
    parser.add_argument('--method', type=str, choices=['hierarchical', 'kmeans', 'dbscan'],
                        default='hierarchical', help="Clustering method (default: hierarchical)")
    parser.add_argument('--clusters', type=int, default=8, 
                        help="Number of clusters to use (default: 8)")
    parser.add_argument('--advanced-nlp', action='store_true', default=True,
                        help="Use advanced NLP features (default: True)")
    parser.add_argument('--output-dir', type=str, default='exact_cleaned_analysis',
                        help="Base output directory (default: exact_cleaned_analysis)")
    parser.add_argument('--viz-dir', type=str, default='exact_cleaned_visualizations',
                        help="Visualizations directory (default: exact_cleaned_visualizations)")
    
    args = parser.parse_args()
    
    # Add base_output_dir for compatibility with iterate_similarity_weights
    args.base_output_dir = args.output_dir
    
    return args


def parse_nt_filename(filename):
    """
    Parse a chapter filename to extract manuscript info.
    Expected format: path/to/grcsbl_XXX_BBB_CC_read.txt
    where XXX is the manuscript number, BBB is the book code, CC is the chapter
    
    Args:
        filename: Name of the chapter file
        
    Returns:
        Tuple of (manuscript_id, book_code, chapter_num, full_path)
    """
    # Define valid NT book codes - expanded to include all NT books
    VALID_BOOKS = r'(?:ROM|1CO|2CO|GAL|EPH|PHP|COL|1TH|2TH|1TI|2TI|TIT|PHM|' + \
                 r'ACT|JHN|1JN|2JN|3JN|1PE|2PE|JUD|REV|JAS|HEB|MAT|MRK|LUK)'
    
    # Extract components using regex - allow for full path
    pattern = rf'.*?grcsbl_(\d+)_({VALID_BOOKS})_(\d+)_read\.txt$'
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
        
    manuscript_id = match.group(1)
    book_code = match.group(2)
    chapter_num = match.group(3)
    
    return manuscript_id, book_code, chapter_num, filename


def combine_chapter_texts(chapter_files):
    """
    Combine multiple chapter files into a single text.
    
    Args:
        chapter_files: List of chapter file paths
        
    Returns:
        Combined text from all chapters
    """
    combined_text = []
    
    for file_path in sorted(chapter_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chapter_text = f.read().strip()
                combined_text.append(chapter_text)
        except Exception as e:
            print(f"Warning: Error reading {file_path}: {e}")
            continue
    
    return "\n\n".join(combined_text)


def group_and_combine_books(chapter_files):
    """
    Group chapter files by book and combine each book's chapters into a single text.
    
    Args:
        chapter_files: List of all chapter files
        
    Returns:
        Dictionary mapping book names to their combined texts
    """
    # First, group chapters by book
    book_chapters = defaultdict(list)
    
    for file_path in chapter_files:
        try:
            manuscript_id, book_code, _, _ = parse_nt_filename(file_path)
            
            # Use just the book code without the manuscript ID
            book_name = book_code
            book_chapters[book_name].append(file_path)
        except ValueError as e:
            print(f"Warning: Skipping invalid file {file_path}: {e}")
            continue
    
    # Sort chapters within each book
    for book_name in book_chapters:
        book_chapters[book_name] = sorted(book_chapters[book_name])
    
    # Now combine chapters for each book
    combined_books = {}
    for book_name, chapters in book_chapters.items():
        print(f"  - {book_name}: {len(chapters)} chapters")
        combined_books[book_name] = combine_chapter_texts(chapters)
    
    return combined_books


def get_display_names():
    """Get dictionary mapping book codes and filenames to display names."""
    base_map = {
        "ROM": "Romans",
        "1CO": "1 Corinthians",
        "2CO": "2 Corinthians",
        "GAL": "Galatians",
        "EPH": "Ephesians",
        "PHP": "Philippians",
        "COL": "Colossians",
        "1TH": "1 Thessalonians",
        "2TH": "2 Thessalonians", 
        "1TI": "1 Timothy",
        "2TI": "2 Timothy",
        "TIT": "Titus",
        "PHM": "Philemon",
        "HEB": "Hebrews",
        "JAS": "James",
        "1PE": "1 Peter",
        "2PE": "2 Peter",
        "1JN": "1 John",
        "2JN": "2 John",
        "3JN": "3 John",
        "JUD": "Jude",
        "REV": "Revelation",
        "ACT": "Acts",
        "JHN": "John"
    }
    
    # Add Julian letter display names
    julian_map = {
        "φραγμεντυμ επιστολαε": "Julian: Fragment Letter",
        "Λιβανίῳ σοφιστῇ καὶ κοιαίστωρι": "Julian: To Libanius",
        "Τῷ αὐτῷ": "Julian: To the Same Person",
        "Ἀνεπίγραφος ὑπὲρ Ἀργείων": "Julian: Unnamed for the Argives",
        "Διονυσίῳ": "Julian: To Dionysius",
        "Σαραπίωνι τῷ λαμπροτάτῳ": "Julian: To Most Illustrious Sarapion"
    }
    
    # Combine both maps
    return {**base_map, **julian_map}


def load_julian_letters(julian_dir):
    """
    Load Julian letters as whole documents.
    
    Args:
        julian_dir: Directory with Julian letter files
        
    Returns:
        Dictionary mapping letter names to their text content
    """
    letters = {}
    
    for letter_path in glob.glob(os.path.join(julian_dir, "*.txt")):
        letter_name = os.path.basename(letter_path)
        letter_name = os.path.splitext(letter_name)[0]  # Remove extension
        
        try:
            with open(letter_path, 'r', encoding='utf-8') as f:
                letter_text = f.read().strip()
                letters[letter_name] = letter_text
                print(f"  - Loaded Julian letter: {letter_name}")
        except Exception as e:
            print(f"Warning: Error reading {letter_path}: {e}")
    
    return letters


def main():
    """Main function to run the full analysis."""
    args = parse_arguments()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.viz_dir, exist_ok=True)
    
    print("Starting full Greek manuscript analysis...")
    print(f"Output directory: {args.output_dir}")
    print(f"Visualizations: {args.viz_dir}")
    
    try:
        # Load Julian letters (whole letters)
        print("\nLoading Julian letters...")
        julian_dir = os.path.join("data", "Julian")
        julian_letters = load_julian_letters(julian_dir)
        
        # Load and combine Non-Pauline texts (by chapter -> book)
        print("\nLoading and combining Non-Pauline texts...")
        non_pauline_dir = os.path.join("data", "Non-Pauline Texts")
        non_pauline_chapters = glob.glob(os.path.join(non_pauline_dir, "*.txt"))
        non_pauline_books = group_and_combine_books(non_pauline_chapters)
        
        # Load and combine Pauline texts (by chapter -> book)
        print("\nLoading and combining Pauline texts...")
        pauline_dir = os.path.join("data", "Cleaned_Paul_Texts")
        pauline_chapters = glob.glob(os.path.join(pauline_dir, "*.txt"))
        pauline_books = group_and_combine_books(pauline_chapters)
        
        # Combine all texts into a single dictionary
        all_texts = {}
        all_texts.update(julian_letters)
        all_texts.update(non_pauline_books)
        all_texts.update(pauline_books)
        
        print(f"\nTotal texts for analysis: {len(all_texts)}")
        print(f"  - Julian letters: {len(julian_letters)}")
        print(f"  - Non-Pauline books: {len(non_pauline_books)}")
        print(f"  - Pauline books: {len(pauline_books)}")
        
        # Get display names for better visualization
        display_names = get_display_names()
        
        # Save all texts to a temporary folder for processing
        temp_dir = os.path.join(args.output_dir, "temp_texts")
        os.makedirs(temp_dir, exist_ok=True)
        
        for name, text in all_texts.items():
            filename = os.path.join(temp_dir, f"{name}.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text)
        
        # Create weight configurations
        weight_configs = create_weight_configs()
        
        # Run the five iterations with different weights
        print("\nRunning analysis with five different weight configurations...")
        similarity_matrices = calculate_similarity_across_weights(
            combined_books=all_texts,
            display_names=display_names,
            args=args,
            weight_configs=weight_configs
        )
        
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        print(f"\nAnalysis complete! Results saved to {args.output_dir} and {args.viz_dir}")
        print(f"Individual visualizations are in each weight configuration's 'visualizations' subfolder")
        print(f"Aggregate visualizations combining all weight configurations are in {args.viz_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 