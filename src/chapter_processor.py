"""
Module for processing chapter-based manuscript files.
"""

import os
import re
from typing import Dict, List, Tuple
from collections import defaultdict

def parse_filename(filename: str) -> Tuple[str, str, str, str]:
    """
    Parse a chapter filename to extract manuscript info.
    Expected format: grcsbl_XXX_BBB_CC_read.txt
    where XXX is the manuscript number, BBB is the book code, CC is the chapter
    
    Args:
        filename: Name of the chapter file
        
    Returns:
        Tuple of (manuscript_id, book_code, chapter_num, full_path)
    """
    # Extract components using regex
    pattern = r'grcsbl_(\d+)_([A-Z]+)_(\d+)_read\.txt'
    match = re.match(pattern, os.path.basename(filename))
    
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
        
    manuscript_id = match.group(1)
    book_code = match.group(2)
    chapter_num = match.group(3)
    
    return manuscript_id, book_code, chapter_num, filename

def group_chapters_by_letter(chapter_files: List[str]) -> Dict[str, List[str]]:
    """
    Group chapter files by letter/book.
    
    Args:
        chapter_files: List of chapter file paths
        
    Returns:
        Dictionary mapping letter names to lists of chapter files
    """
    letter_groups = defaultdict(list)
    
    for file_path in chapter_files:
        try:
            manuscript_id, book_code, _, _ = parse_filename(file_path)
            letter_name = f"{book_code}-{manuscript_id}"
            letter_groups[letter_name].append(file_path)
        except ValueError as e:
            print(f"Warning: Skipping invalid file {file_path}: {e}")
            continue
    
    # Sort chapters within each letter
    for letter_name in letter_groups:
        letter_groups[letter_name].sort()
    
    return dict(letter_groups)

def combine_chapter_texts(chapter_files: List[str]) -> str:
    """
    Combine multiple chapter files into a single text.
    
    Args:
        chapter_files: List of chapter file paths
        
    Returns:
        Combined text from all chapters
    """
    combined_text = []
    
    for file_path in chapter_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chapter_text = f.read().strip()
                combined_text.append(chapter_text)
        except Exception as e:
            print(f"Warning: Error reading {file_path}: {e}")
            continue
    
    return "\n\n".join(combined_text) 