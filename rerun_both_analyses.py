#!/usr/bin/env python3
"""
Script to run analyses both with and without Pauline quotes.
This script will:
1. Run analysis with Pauline quotes (regular data)
2. Run analysis without Pauline quotes (cleaned data)
3. Generate all visualizations for both analyses
"""

import os
import sys
import subprocess
import shutil
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run analysis with and without Pauline quotes')
    parser.add_argument('--advanced_nlp', action='store_true', help='Use advanced NLP features')
    parser.add_argument('--clusters', type=int, default=6, help='Number of clusters to use for aggregate analysis')
    parser.add_argument('--method', default='hierarchical', help='Clustering method to use')
    parser.add_argument('--clean_dirs', action='store_true', help='Clean output directories before running')
    return parser.parse_args()

def clean_directory(directory_path):
    """Remove directory if it exists and recreate it"""
    if os.path.exists(directory_path):
        print(f"Cleaning directory: {directory_path}")
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)
    print(f"Created directory: {directory_path}")

def run_analysis_with_quotes(args):
    """Run the analysis with Pauline quotes (original texts)"""
    print("\n" + "="*80)
    print("RUNNING ANALYSIS WITH PAULINE QUOTES")
    print("="*80)
    
    output_dir = "full_greek_analysis"
    viz_dir = "full_greek_visualizations"
    
    # Clean directories if requested
    if args.clean_dirs:
        clean_directory(output_dir)
        clean_directory(viz_dir)
    
    # Construct command
    cmd = [
        "python3", "run_full_greek_analysis.py",
        "--output-dir", output_dir,
        "--viz-dir", viz_dir,
        "--clusters", str(args.clusters),
        "--method", args.method
    ]
    
    if args.advanced_nlp:
        cmd.append("--advanced-nlp")
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Analysis with Pauline quotes completed!")

def run_analysis_without_quotes(args):
    """Run the analysis without Pauline quotes (cleaned texts)"""
    print("\n" + "="*80)
    print("RUNNING ANALYSIS WITHOUT PAULINE QUOTES")
    print("="*80)
    
    output_dir = "exact_cleaned_analysis"
    viz_dir = "exact_cleaned_visualizations"
    
    # Clean directories if requested
    if args.clean_dirs:
        clean_directory(output_dir)
        clean_directory(viz_dir)
    
    # Construct command
    cmd = [
        "python3", "run_exact_cleaned_analysis.py",
        "--output-dir", output_dir,
        "--viz-dir", viz_dir,
        "--clusters", str(args.clusters),
        "--method", args.method
    ]
    
    if args.advanced_nlp:
        cmd.append("--advanced-nlp")
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Analysis without Pauline quotes completed!")

def run_pauline_analysis(args):
    """Run the Pauline-specific analysis"""
    print("\n" + "="*80)
    print("RUNNING PAULINE-SPECIFIC ANALYSIS")
    print("="*80)
    
    output_dir = "pauline_analysis"
    viz_dir = "pauline_visualizations"
    
    # Clean directories if requested
    if args.clean_dirs:
        clean_directory(output_dir)
        clean_directory(viz_dir)
    
    # Construct command
    cmd = [
        "python3", "run_cleaned_paul_analysis.py",
        "--output_dir", output_dir,
        "--viz_dir", viz_dir,
        "--aggregate_clusters", str(args.clusters)
    ]
    
    if args.advanced_nlp:
        cmd.append("--advanced_nlp")
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Pauline-specific analysis completed!")

def main():
    args = parse_arguments()
    
    print("Starting comprehensive analysis...")
    
    # 1. Run analysis with Pauline quotes
    run_analysis_with_quotes(args)
    
    # 2. Run analysis without Pauline quotes
    run_analysis_without_quotes(args)
    
    # 3. Run Pauline-specific analysis
    run_pauline_analysis(args)
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nOutput directories:")
    print("- With Pauline quotes: full_greek_analysis, full_greek_visualizations")
    print("- Without Pauline quotes: exact_cleaned_analysis, exact_cleaned_visualizations")
    print("- Pauline-specific: pauline_analysis, pauline_visualizations")

if __name__ == "__main__":
    main() 