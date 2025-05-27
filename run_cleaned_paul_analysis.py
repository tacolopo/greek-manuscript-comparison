#!/usr/bin/env python3

import os
import sys
import argparse
import glob
import re
import numpy as np
import pandas as pd
from src.multi_comparison import MultipleManuscriptComparison
from src.similarity import SimilarityCalculator

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run analysis on Cleaned Paul texts')
    parser.add_argument('--output_dir', default='cleaned_paul_analysis', help='Output directory for results')
    parser.add_argument('--viz_dir', default='cleaned_paul_visualizations', help='Output directory for visualizations')
    parser.add_argument('--advanced_nlp', action='store_true', help='Use advanced NLP features')
    parser.add_argument('--n_clusters', type=int, default=None, help='Number of clusters to use (default: determined automatically)')
    parser.add_argument('--aggregate_clusters', type=int, default=8, help='Number of clusters to use for aggregate analysis')
    return parser.parse_args()

def combine_chapters(files):
    """
    Combine chapters of the same book into a single document.
    Returns a dictionary mapping book names to their combined content.
    """
    combined_books = {}
    
    # Regular expression to extract book code and chapter number
    # Format is typically grcsbl_075_ROM_01_read.txt where ROM is the book code and 01 is the chapter
    pattern = re.compile(r'grcsbl_(\d+)_([A-Z1-3]+)_(\d+)_read\.txt')
    
    # Group files by book code
    book_files = {}
    for file_path in files:
        filename = os.path.basename(file_path)
        match = pattern.match(filename)
        
        if match:
            book_num = match.group(1)  # e.g., 075
            book_code = match.group(2)  # e.g., ROM
            chapter = int(match.group(3))  # e.g., 01 as integer 1
            
            book_key = f"{book_code}"  # Use book code as key (e.g., "ROM")
            
            if book_key not in book_files:
                book_files[book_key] = []
            
            # Store tuple of (chapter_number, file_path) for sorting
            book_files[book_key].append((chapter, file_path))
    
    # Calculate total size of data
    total_size = 0
    # Now process each book
    for book_key, chapter_files in book_files.items():
        # Sort chapters numerically
        chapter_files.sort(key=lambda x: x[0])
        
        # Combine content
        combined_content = ""
        for _, file_path in chapter_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                combined_content += content + "\n\n"
                total_size += len(content)
        
        # Store combined content
        combined_books[book_key] = combined_content
    
    print(f"Total text size: {total_size/1024:.2f} KB")
    return combined_books

def calculate_similarity_across_weights(args):
    """
    Calculate similarity matrices for different weight configurations
    """
    # Define different weight configurations
    weight_configs = [
        {'name': 'baseline', 'vocabulary': 0.7, 'structure': 0.3, 'nlp': 0.0},
        {'name': 'equal', 'vocabulary': 0.33, 'structure': 0.33, 'nlp': 0.34},
        {'name': 'nlp_only', 'vocabulary': 0.0, 'structure': 0.0, 'nlp': 1.0},
        {'name': 'structure_focused', 'vocabulary': 0.2, 'structure': 0.7, 'nlp': 0.1},
        {'name': 'vocabulary_focused', 'vocabulary': 0.7, 'structure': 0.1, 'nlp': 0.2},
    ]
    
    # Create the main output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create visualization directory if it doesn't exist
    os.makedirs(args.viz_dir, exist_ok=True)
    
    # Create directories for each weight configuration
    for config in weight_configs:
        config_dir = os.path.join(args.output_dir, config['name'])
        os.makedirs(config_dir, exist_ok=True)
        
        # Create visualization subdirectory for this weight configuration
        iteration_viz_dir = os.path.join(config_dir, 'visualizations')
        os.makedirs(iteration_viz_dir, exist_ok=True)
    
    # Store all similarity matrices for later averaging
    all_sim_matrices = []
    all_indices = None
    
    # Define paths to text directories
    pauline_texts = os.path.join('data', 'Cleaned_Paul_Texts')
    non_pauline_texts = os.path.join('data', 'Non-Pauline Texts')
    julian_texts = os.path.join('data', 'Julian')
    
    # Process each weight configuration
    for config in weight_configs:
        print(f"\nProcessing {config['name']} weight configuration...")
        
        # Create a custom similarity calculator
        similarity_calculator = SimilarityCalculator()
        
        # Convert our weight configuration to the format expected by SimilarityCalculator
        # The SimilarityCalculator uses more granular weights for different feature types
        calculator_weights = {
            'vocabulary': config['vocabulary'],       # Vocabulary richness
            'sentence': config['structure'] * 0.5,    # Sentence structure (part of structure)
            'transitions': config['structure'] * 0.5, # Writing flow (part of structure)
            'ngrams': config['vocabulary'],           # Character and word patterns
            'syntactic': config['nlp']                # Advanced NLP features
        }
        
        # Set the weights
        similarity_calculator.set_weights(calculator_weights)
        
        # Set up the comparison for this iteration with the custom similarity calculator
        comparison = MultipleManuscriptComparison(
            use_advanced_nlp=args.advanced_nlp,
            output_dir=os.path.join(args.output_dir, config['name']),
            visualizations_dir=os.path.join(args.output_dir, config['name'], 'visualizations'),
            similarity_calculator=similarity_calculator
        )
        
        # Gather manuscripts from all directories
        manuscripts = {}
        
        # Get all Paul texts and combine chapters
        paul_files = glob.glob(os.path.join(pauline_texts, "*.txt"))
        paul_books = combine_chapters(paul_files)
        for book_code, content in paul_books.items():
            manuscripts[book_code] = content
        
        # Get all Non-Pauline texts and combine chapters
        nonpaul_files = glob.glob(os.path.join(non_pauline_texts, "*.txt"))
        nonpaul_books = combine_chapters(nonpaul_files)
        for book_code, content in nonpaul_books.items():
            manuscripts[book_code] = content
        
        # Add Julian texts (already whole letters)
        for path in glob.glob(os.path.join(julian_texts, "*.txt")):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            name = os.path.basename(path).split('.')[0]
            manuscripts[f"Julian_{name}"] = text
        
        print(f"Total manuscripts after combining chapters: {len(manuscripts)}")
        print(f"Pauline books: {len(paul_books)}")
        print(f"Non-Pauline books: {len(nonpaul_books)}")
        
        # Use the compare_multiple_manuscripts method
        try:
            result = comparison.compare_multiple_manuscripts(
                manuscripts=manuscripts, 
                method='hierarchical',
                n_clusters=args.n_clusters,
                use_advanced_nlp=args.advanced_nlp
            )
            
            # The results include the similarity matrix and clustering results
            sim_matrix = result['similarity_matrix']
            
            # Store for averaging later
            all_sim_matrices.append(sim_matrix)
            all_indices = sim_matrix.index
            
            # Save the similarity matrix
            output_path = os.path.join(args.output_dir, config['name'], 'similarity_matrix.csv')
            sim_matrix.to_csv(output_path)
            print(f"Saved similarity matrix for {config['name']} to {output_path}")
            
            # Save clustering report
            if 'report' in result:
                report_path = os.path.join(args.output_dir, config['name'], 'clustering_report.txt')
                with open(report_path, 'w') as f:
                    f.write(result['report'])
                print(f"Saved clustering report for {config['name']} to {report_path}")
            
            # Note: Visualizations are already generated by compare_multiple_manuscripts
            print(f"Visualizations for {config['name']} configuration saved to {os.path.join(args.output_dir, config['name'], 'visualizations')}")
            
        except Exception as e:
            print(f"Error processing {config['name']} configuration: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Now create the aggregate report and visualizations
    if all_sim_matrices and all_indices is not None:
        create_aggregate_visualizations(all_sim_matrices, all_indices, args)
    else:
        print("No similarity matrices were successfully generated. Cannot create aggregate visualizations.")

def create_aggregate_visualizations(all_matrices, all_indices, args):
    """
    Create aggregate visualizations by averaging all similarity matrices
    """
    print("\nCreating aggregate visualizations...")
    
    try:
        # Convert all matrices to numpy arrays for easier manipulation
        matrix_arrays = [matrix.values for matrix in all_matrices]
        
        # Average all matrices
        avg_matrix = np.mean(matrix_arrays, axis=0)
        
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
        # Use fixed number of clusters for consistency with non-cleaned run
        print(f"Using {args.aggregate_clusters} clusters for aggregate analysis (to match non-cleaned run)")
        clustering_result = comparison.cluster_manuscripts(
            similarity_df=avg_df, 
            n_clusters=args.aggregate_clusters
        )
        
        # Generate visualizations for the aggregate results
        comparison.generate_visualizations(
            clustering_result=clustering_result,
            similarity_df=avg_df,
            threshold=0.5
        )
        
        # Generate a report for the aggregate results
        # First create a dictionary of empty preprocessed data and features data
        # since these are required by the generate_report method but we only need
        # the clustering results for our report
        dummy_preprocessed = {name: {'words': [], 'sentence_stats': {}} for name in avg_df.index}
        dummy_features = {name: {'vocabulary_richness': {}, 'sentence_stats': {}} for name in avg_df.index}
        
        # Generate the report using the available method
        try:
            report = comparison.generate_report(
                clustering_result=clustering_result,
                preprocessed_data=dummy_preprocessed,
                features_data=dummy_features,
                similarity_df=avg_df
            )
            
            # Read the generated report
            with open(report, 'r') as f:
                report_content = f.read()
        except Exception as e:
            # If report generation fails, create a simple report
            report_content = "Cluster analysis:\n\n"
            
            # Extract cluster labels
            labels = clustering_result['labels']
            names = clustering_result['manuscript_names']
            
            # Group manuscripts by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(names[i])
            
            # Add cluster information to report
            for cluster_id, members in clusters.items():
                report_content += f"Cluster {cluster_id}:\n"
                for member in members:
                    report_content += f"  - {member}\n"
                report_content += "\n"
        
        # Save aggregate report
        aggregate_report_path = os.path.join(args.output_dir, 'aggregate_report.md')
        
        # Create a comprehensive report
        with open(aggregate_report_path, 'w') as f:
            f.write("# Aggregate Analysis of Greek Manuscript Comparison\n\n")
            f.write("## Overview\n\n")
            f.write("This report presents an aggregate analysis of multiple text comparisons using different weight configurations.\n\n")
            f.write("The following weight configurations were used:\n")
            f.write("- **Baseline**: Vocabulary (70%), Structure (30%), NLP (0%)\n")
            f.write("- **Equal**: Vocabulary (33%), Structure (33%), NLP (34%)\n")
            f.write("- **NLP Only**: Vocabulary (0%), Structure (0%), NLP (100%)\n")
            f.write("- **Structure Focused**: Vocabulary (20%), Structure (70%), NLP (10%)\n")
            f.write("- **Vocabulary Focused**: Vocabulary (70%), Structure (10%), NLP (20%)\n\n")
            f.write("## Clustering Results\n\n")
            f.write("```\n")
            f.write(report_content)
            f.write("```\n\n")
            f.write("\n\n## Interpretation\n\n")
            f.write("The clusters above represent the average grouping of texts across all five weight configurations.\n")
            f.write("This provides a more robust analysis than any single configuration alone.\n")
            f.write("Texts appearing in the same cluster consistently are more likely to share stylometric similarities.\n")
            f.write("\n## Note on Cleaned Pauline Text Analysis\n\n")
            f.write("This analysis was performed on a version of the texts with Old Testament quotations removed.\n")
            f.write("This provides a more accurate assessment of writing style without the influence of quoted material.\n")
            f.write("Each book was processed as a complete text by combining all its chapters rather than analyzing individual chapters separately.\n")
        
        print(f"Aggregate analysis complete. Results saved to {args.output_dir}")
        print(f"Visualizations saved to {args.viz_dir}")
        
    except Exception as e:
        print(f"Error creating aggregate visualizations: {e}")
        import traceback
        traceback.print_exc()

def main():
    args = parse_arguments()
    calculate_similarity_across_weights(args)

if __name__ == "__main__":
    main() 