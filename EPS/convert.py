import pandas as pd
import re

def parse_text_to_excel(input_file, output_file):
    # Read the text file
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split content by question numbers (starting with a number followed by period)
    question_blocks = re.split(r'\n\d+\.\n', content)
    
    # Remove any empty block at the beginning
    if not question_blocks[0].strip():
        question_blocks = question_blocks[1:]
    
    # Initialize lists to store data
    questions = []
    answer_choices = []
    correct_answers = []
    vietnamese_explanations = []
    english_explanations = []
    
    # Process each question block
    for block in question_blocks:
        # Extract data using regex patterns
        question_match = re.search(r'### Question:(.*?)(?=### Answer choices:|$)', block, re.DOTALL)
        choices_match = re.search(r'### Answer choices:(.*?)(?=### Correct answer:|$)', block, re.DOTALL)
        answer_match = re.search(r'### Correct answer:(.*?)(?=### Vietnamese explanation:|$)', block, re.DOTALL)
        viet_expl_match = re.search(r'### Vietnamese explanation:(.*?)(?=### English explanation:|$)', block, re.DOTALL)
        eng_expl_match = re.search(r'### English explanation:(.*?)(?=\n\d+\.|$)', block, re.DOTALL)
        
        # Add extracted data to lists
        questions.append(question_match.group(1).strip() if question_match else "")
        answer_choices.append(choices_match.group(1).strip() if choices_match else "")
        correct_answers.append(answer_match.group(1).strip() if answer_match else "")
        vietnamese_explanations.append(viet_expl_match.group(1).strip() if viet_expl_match else "")
        english_explanations.append(eng_expl_match.group(1).strip() if eng_expl_match else "")
    
    # Create DataFrame
    df = pd.DataFrame({
        'Question': questions,
        'Answer Choices': answer_choices,
        'Correct Answer': correct_answers,
        'Vietnamese Explanation': vietnamese_explanations,
        'English Explanation': english_explanations
    })
    
    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"Successfully converted to Excel: {output_file}")

# Usage
input_file = 'output_420012_bqt.txt'  # Update with your actual input file path
output_file = 'korean_questions.xlsx'     # Update with your desired output file path

parse_text_to_excel(input_file, output_file)