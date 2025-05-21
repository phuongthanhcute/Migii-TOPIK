#!/usr/bin/env python3
"""
Korean Listening Test Converter

This script converts Korean listening test data from Excel/TSV format to a formatted text file.
"""

import sys
import os
import pandas as pd

def clean_text(text):
    """Clean up text by removing surrounding quotes and ensuring it's a string"""
    if not isinstance(text, str):
        text = str(text) if pd.notna(text) else ""
    
    # Remove surrounding quotes if present
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    
    return text

def convert_file_to_formatted_output(file_path, output_file=None):
    """
    Convert Excel or TSV file to formatted output
    
    Args:
        file_path: Path to the Excel/TSV file
        output_file: Optional path to save output (if None, prints to stdout)
    """
    results = []
    
    try:
        # Determine file type based on extension
        _, ext = os.path.splitext(file_path)
        
        # Read the file
        if ext.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:  # Default to TSV
            df = pd.read_csv(file_path, sep='\t')
        
        # Find the required columns (handle case-insensitivity)
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'g_text_audio' == col_lower and 'translate' not in col_lower:
                column_mapping['audio'] = col
            elif 'g_text_audio_translate_vi' in col_lower:
                column_mapping['audio_vi'] = col
            elif 'g_text_audio_translate_en' in col_lower:
                column_mapping['audio_en'] = col
            elif 'text_question_1' == col_lower:
                column_mapping['question'] = col
            elif 'correct_answer_1' == col_lower:
                column_mapping['answer'] = col
            elif 'explain_vi_1' == col_lower:
                column_mapping['explain_vi'] = col
            elif 'explain_en_1' == col_lower:
                column_mapping['explain_en'] = col
        
        # Process each row
        for i, row in df.iterrows():
            # Skip rows with empty Korean audio
            if pd.isna(row.get(column_mapping.get('audio', ''), "")):
                continue
                
            # Extract data
            korean_audio = clean_text(row.get(column_mapping.get('audio', ''), ""))
            vietnamese = clean_text(row.get(column_mapping.get('audio_vi', ''), ""))
            english = clean_text(row.get(column_mapping.get('audio_en', ''), ""))
            answer_choices = clean_text(row.get(column_mapping.get('question', ''), ""))
            correct_answer = clean_text(row.get(column_mapping.get('answer', ''), ""))
            vietnamese_explain = clean_text(row.get(column_mapping.get('explain_vi', ''), ""))
            english_explain = clean_text(row.get(column_mapping.get('explain_en', ''), ""))
            
            # Skip completely empty rows
            if not korean_audio.strip():
                continue
            
            # Pre-process string replacements
            korean_audio_formatted = korean_audio.replace('\n', '\n        ')
            vietnamese_formatted = vietnamese.replace('\n', '\n        ')
            english_formatted = english.replace('\n', '\n        ')
            answer_choices_formatted = answer_choices.replace('\n', '\n        ')
            vietnamese_explain_formatted = vietnamese_explain.replace('\n', '\n        ')
            english_explain_formatted = english_explain.replace('\n', '\n        ')
            
            # Format with indentation
            formatted_output = f"""### Text Audio:
        {korean_audio_formatted}
### Vietnamese:
        {vietnamese_formatted}
### English:
        {english_formatted}
### Answer:
        {answer_choices_formatted}
### Correct Answer: {correct_answer}
### Vietnamese Explain:
        {vietnamese_explain_formatted}
### English Explain:
        {english_explain_formatted}"""
            
            results.append(formatted_output)
    
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Format the output
    formatted_output = "\n\n".join([f"{i+1}.  {result}" for i, result in enumerate(results)])
    
    # Output the result
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        print(f"Output written to {output_file}")
    else:
        print(formatted_output)
    
    return formatted_output

if __name__ == "__main__":
    # Check if file path is provided
    if len(sys.argv) < 2:
        print("Usage: python convert.py input_file.xlsx [output.txt]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_file_to_formatted_output(input_file, output_file)