#!/usr/bin/env python3
"""
Script to run a comparison between two Greek manuscripts.
"""

import os
import sys
import argparse
from src.compare_manuscripts import main as compare_main

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a comparison between Greek manuscripts")
    
    # Sample manuscripts
    parser.add_argument('--sample', action='store_true', help="Run with sample manuscripts")
    parser.add_argument('--sample-num', type=int, choices=[1, 2, 3], default=1, 
                       help="Which sample comparison to run (1: similar texts, 2: different texts)")
    
    # Custom manuscripts
    parser.add_argument('--file1', type=str, help="Path to first manuscript file")
    parser.add_argument('--file2', type=str, help="Path to second manuscript file")
    parser.add_argument('--name1', type=str, help="Name of first manuscript")
    parser.add_argument('--name2', type=str, help="Name of second manuscript")
    
    # Processing options
    parser.add_argument('--remove-stopwords', action='store_true', help="Remove Greek stopwords")
    parser.add_argument('--normalize-accents', action='store_true', help="Normalize Greek accents")
    parser.add_argument('--lowercase', action='store_true', help="Convert text to lowercase")
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='output', help="Output directory")
    parser.add_argument('--visualize', action='store_true', help="Generate visualizations")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Set up arguments for compare_manuscripts.py
    sys_argv = sys.argv.copy()
    sys_argv = [sys.argv[0]]  # Reset arguments
    
    # Determine which manuscripts to compare
    if args.sample:
        # Use sample texts
        if args.sample_num == 1:
            # Compare sample1 (John 1:1-5) with sample2 (similar text with minor variations)
            file1 = os.path.join('data', 'sample_text1.txt')
            file2 = os.path.join('data', 'sample_text2.txt')
            name1 = "John-1-Original"
            name2 = "John-1-Variant"
        elif args.sample_num == 2:
            # Compare sample1 with sample3 (completely different text)
            file1 = os.path.join('data', 'sample_text1.txt')
            file2 = os.path.join('data', 'sample_text3.txt')
            name1 = "John-1"
            name2 = "Philosophical-Text"
        else:
            # Compare sample2 with sample3
            file1 = os.path.join('data', 'sample_text2.txt')
            file2 = os.path.join('data', 'sample_text3.txt')
            name1 = "John-1-Variant"
            name2 = "Philosophical-Text"
    else:
        # Use custom manuscripts
        if not args.file1 or not args.file2:
            print("Error: --file1 and --file2 are required when not using --sample")
            return 1
            
        file1 = args.file1
        file2 = args.file2
        name1 = args.name1 if args.name1 else os.path.basename(file1).split('.')[0]
        name2 = args.name2 if args.name2 else os.path.basename(file2).split('.')[0]
    
    # Add arguments
    sys_argv.extend(['--file1', file1])
    sys_argv.extend(['--file2', file2])
    
    if name1:
        sys_argv.extend(['--name1', name1])
    if name2:
        sys_argv.extend(['--name2', name2])
    
    # Add processing options
    if args.remove_stopwords:
        sys_argv.append('--remove-stopwords')
    if args.normalize_accents:
        sys_argv.append('--normalize-accents')
    if args.lowercase:
        sys_argv.append('--lowercase')
    
    # Add output options
    sys_argv.extend(['--output-dir', args.output_dir])
    if args.visualize:
        sys_argv.append('--visualize')
    
    # Override sys.argv
    sys.argv = sys_argv
    
    # Call the main function
    print(f"Comparing {name1} with {name2}...")
    return compare_main()

if __name__ == "__main__":
    sys.exit(main()) 