#!/usr/bin/env python3

import os
import re
import glob

# Define quotation markers
QUOTE_MARKERS = [
    r'γέγραπται(?:\s+γὰρ)?',  # "it is written (for)"
    r'καθὼς\s+γέγραπται',  # "as it is written"
    r'λέγει\s+γὰρ\s+ἡ\s+γραφή',  # "for the Scripture says"
    r'ἡ\s+γραφὴ\s+λέγει',  # "the Scripture says"
    r'προϊδοῦσα\s+δὲ\s+ἡ\s+γραφὴ',  # "the Scripture, foreseeing"
    r'[Ἠη]σαΐας\s+(?:δὲ\s+)?(?:λέγει|κράζει)',  # Isaiah says/cries
    r'ἐν\s+τῷ\s+[Ὡωὁ]\w+\s+λέγει',  # "as he says in [prophet/book]"
    r'[Μμ]ωϋσῆς\s+(?:δὲ\s+)?(?:λέγει|γράφει)',  # "Moses says/writes"
    r'ὁ\s+λόγος\s+ὁ\s+γεγραμμένος',  # "the word that is written"
    r'οὕτως\s+(?:δὲ\s+)?(?:καὶ\s+)?γέγραπται',  # "thus it is also written"
]

# Combined regex pattern for quote markers
MARKER_PATTERN = '|'.join(f'({marker})' for marker in QUOTE_MARKERS)

def process_file(input_file, output_dir):
    """
    Process a single file to remove OT quotations
    """
    # Read the file content
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Get just the filename without path
    filename = os.path.basename(input_file)
    
    # Create output path
    output_file = os.path.join(output_dir, filename)
    
    # Find all potential quotation markers
    modified_content = content
    removed_count = 0
    
    # Split text into lines for better processing
    lines = content.split('\n')
    processed_lines = []
    
    for line in lines:
        # Check for quotation markers in this line
        matches = list(re.finditer(MARKER_PATTERN, line, re.IGNORECASE))
        
        if not matches:
            # No marker in this line, keep as is
            processed_lines.append(line)
            continue
        
        # Process each marker in reverse to avoid shifting indices
        for match in reversed(matches):
            marker_start = match.start()
            marker_end = match.end()
            marker_text = match.group(0)
            
            # Get the text after the marker
            after_marker = line[marker_end:]
            
            # Check if there's a colon after the marker
            colon_match = re.search(r'[:·]\s*', after_marker)
            if colon_match:
                quote_start = marker_end + colon_match.end()
            else:
                # If no colon, try to determine where quote starts
                # Look for a word boundary or space
                quote_start = marker_end
                space_match = re.search(r'\s+', after_marker)
                if space_match and space_match.start() < 3:  # Close to the marker
                    quote_start = marker_end + space_match.end()
            
            # The quotation typically continues to the end of the verse or a punctuation mark
            # This is where Greek text patterns become important
            remainder = line[quote_start:]
            
            # Look for end of quotation - often ends with a period, semicolon, or full line
            punctuation_match = re.search(r'[.;·]', remainder)
            
            if punctuation_match:
                quote_end = quote_start + punctuation_match.end()
                # Keep the marker, remove the quotation
                line = line[:quote_start] + line[quote_end:]
                removed_count += 1
            elif not remainder.strip():
                # If there's nothing after the marker on this line, just keep the marker
                line = line[:quote_start]
                removed_count += 1
            else:
                # If we can't clearly determine the end, use a conservative approach
                # and just remove the rest of the line after the quotation start
                line = line[:quote_start]
                removed_count += 1
        
        processed_lines.append(line)
    
    # Join the processed lines back into content
    modified_content = '\n'.join(processed_lines)
    
    # Write modified content to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    return filename, removed_count

def main():
    input_dir = 'data/Paul Texts'
    output_dir = 'cleaned_texts'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all text files in the input directory
    input_files = glob.glob(os.path.join(input_dir, '*.txt'))
    
    # Process each file
    total_quotes_removed = 0
    for input_file in input_files:
        filename, quotes_removed = process_file(input_file, output_dir)
        total_quotes_removed += quotes_removed
        print(f"Processed {filename}: Removed {quotes_removed} Old Testament quotations")
    
    print(f"\nTotal Old Testament quotations removed: {total_quotes_removed}")

if __name__ == "__main__":
    main() 