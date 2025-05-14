#!/usr/bin/env python3
"""
Script to reorganize the analysis folders, separating the biased and unbiased analyses.
This script moves the old biased analysis files to a legacy folder and organizes
the unbiased analysis files for clarity.
"""

import os
import shutil
import sys

def create_directory(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    return path

def move_files(source_files, destination_dir):
    """Move files to destination directory."""
    create_directory(destination_dir)
    
    for file_path in source_files:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            dest_path = os.path.join(destination_dir, filename)
            
            # Handle existing files
            if os.path.exists(dest_path):
                print(f"File exists, replacing: {dest_path}")
                os.remove(dest_path)
                
            # Move the file
            shutil.move(file_path, dest_path)
            print(f"Moved: {file_path} -> {dest_path}")
        else:
            print(f"File not found: {file_path}")

def organize_pauline_analysis():
    """Organize the pauline_analysis directory."""
    
    # Create main directories
    pauline_root = "pauline_analysis"
    create_directory(pauline_root)
    
    # Create subdirectories
    legacy_dir = create_directory(os.path.join(pauline_root, "legacy_biased_analysis"))
    
    # List files in pauline_analysis
    if not os.path.exists(pauline_root):
        print(f"Error: {pauline_root} directory does not exist")
        return
    
    files = [os.path.join(pauline_root, f) for f in os.listdir(pauline_root) 
             if os.path.isfile(os.path.join(pauline_root, f))]
    
    # Identify biased analysis files to move to legacy
    biased_files = []
    for file_path in files:
        filename = os.path.basename(file_path)
        # Files related to disputed/undisputed analysis
        if "disputed" in filename.lower() or "undisputed" in filename.lower():
            biased_files.append(file_path)
    
    # Move biased files to legacy directory
    if biased_files:
        move_files(biased_files, legacy_dir)
    else:
        print("No biased analysis files found in pauline_analysis directory")

def verify_unbiased_structure():
    """Verify that unbiased analysis directories are properly set up."""
    
    # Check unbiased analysis directory
    unbiased_dir = "pauline_analysis_unbiased"
    if not os.path.exists(unbiased_dir):
        print(f"Warning: Unbiased analysis directory {unbiased_dir} does not exist")
        print("Run unbiased_pauline_clustering.py to create it")
    else:
        print(f"Unbiased analysis directory exists: {unbiased_dir}")
        file_count = len([f for f in os.listdir(unbiased_dir) if os.path.isfile(os.path.join(unbiased_dir, f))])
        print(f"Contains {file_count} files")
    
    # Check enhanced plots directory
    enhanced_dir = "pauline_enhanced_plots"
    if not os.path.exists(enhanced_dir):
        print(f"Warning: Enhanced plots directory {enhanced_dir} does not exist")
        print("Run improve_pauline_plots.py to create it")
    else:
        print(f"Enhanced plots directory exists: {enhanced_dir}")
        file_count = len([f for f in os.listdir(enhanced_dir) if os.path.isfile(os.path.join(enhanced_dir, f))])
        print(f"Contains {file_count} files")

def create_readme():
    """Create a README.md file explaining the directory structure."""
    readme_path = "pauline_analysis_README.md"
    
    with open(readme_path, 'w') as f:
        f.write("# Pauline Letters Stylometric Analysis\n\n")
        
        f.write("## Directory Structure\n\n")
        
        f.write("### Main Analysis Directories\n\n")
        f.write("- **pauline_analysis_unbiased**: Contains the unbiased clustering analysis that lets stylistic patterns emerge naturally without predetermined groupings\n")
        f.write("- **pauline_enhanced_plots**: Contains enhanced visualizations with clearer labels and side-by-side comparisons\n")
        f.write("- **pauline_analysis/legacy_biased_analysis**: Contains the old analysis that used predetermined disputed/undisputed groupings\n\n")
        
        f.write("### Key Files\n\n")
        f.write("- **unbiased_pauline_analysis_summary.md**: Final summary report of the unbiased analysis\n")
        f.write("- **pauline_enhanced_plots/mds_comparison.png**: Side-by-side comparison of different feature weightings\n")
        f.write("- **pauline_analysis_unbiased/unbiased_cluster_report.md**: Detailed report of the unbiased clustering results\n\n")
        
        f.write("### Scripts\n\n")
        f.write("- **unbiased_pauline_clustering.py**: Creates the unbiased MDS plots and clustering analysis\n")
        f.write("- **improve_pauline_plots.py**: Enhances the plots with better labels and creates comparison views\n")
        f.write("- **reorganize_analysis_files.py**: This script that organizes the file structure\n\n")
        
        f.write("## Methodology\n\n")
        f.write("The analysis uses multidimensional scaling (MDS) to visualize stylistic similarities between Pauline letters, ")
        f.write("with hierarchical clustering to identify natural groupings. Five different feature weighting configurations ")
        f.write("were tested to understand how prioritizing different aspects of writing style affects perceived relationships. ")
        f.write("Importantly, no predetermined groupings (like disputed/undisputed) were used in the analysis.\n")
    
    print(f"Created README file: {readme_path}")

def main():
    """Main function to reorganize the analysis files."""
    print("Reorganizing the Pauline analysis files...")
    
    # Organize pauline_analysis directory
    organize_pauline_analysis()
    
    # Verify unbiased analysis structure
    verify_unbiased_structure()
    
    # Create README
    create_readme()
    
    print("\nReorganization complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 