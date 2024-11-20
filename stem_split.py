import json
import re
import pandas as pd

def load_json(filepath):
    # Load JSON data from a file
    with open(filepath, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def get_sentences(text):
    # Protect abbreviation endings from splitting
    protected_abbreviations = ['E.D', 'Mr', 'Mrs', 'Ms', 'Dr', 'St', 'Ave', 'Inc', 'Ltd', 'Co', 'vs']
    pattern = r'\b(?:' + '|'.join(protected_abbreviations) + r')\.\s+(?=[A-Z])'
    pattern_no_split = r'(?<=[.!?])\s+(?=[A-Z])'

    # Use a protected markup pattern
    protected_text = re.sub(pattern, lambda x: x.group().replace('. ', '.|'), text.strip())
    sentences = re.split(pattern_no_split, protected_text)
    sentences = [s.replace('.|', '. ') for s in sentences]  # Revert protection
    return sentences

def get_specific_sentences(sentences):
    total_sentences = len(sentences)
    results = {}

    if total_sentences > 0:
        results['first_sentence'] = sentences[0]

    if total_sentences > 1:
        results['second_sentence'] = sentences[0] + ' ' + sentences[1]
    
    half_index = total_sentences // 2
    if half_index > 1:
        half_content = ' '.join(sentences[:half_index])
    else:
        half_content = None
    
    three_quarters_index = 3 * total_sentences // 4
    if three_quarters_index > 1:
        three_quarters_content = ' '.join(sentences[:three_quarters_index])
    else:
        three_quarters_content = None

    if half_content and half_content != results.get('second_sentence'):
        results['half_sentences'] = half_content

    if three_quarters_content and three_quarters_content not in results.values():
        results['three_quarters_sentences'] = three_quarters_content

    results['full_paragraph'] = ' '.join(sentences)
    
    return results

def process_questions_to_dataframe(file_path):
    # Load questions from JSON file
    questions = load_json(file_path)

    data_list = []

    for entry in questions:
        question = entry.get('question', '')
        sentences = get_sentences(question)
        results = get_specific_sentences(sentences)

        data_list.append({
            'question': question,
            'total_sentences': len(sentences),
            'first_sentence': results.get('first_sentence', ''),
            'second_sentence': results.get('second_sentence', ''),
            'half_sentences': results.get('half_sentences', ''),
            'three_quarters_sentences': results.get('three_quarters_sentences', ''),
            'full_paragraph': results.get('full_paragraph', '')
        })

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)
    return df

def save_dataframe_to_csv(df, output_csv):
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False, encoding='utf-8')

# Example usage
file_path = 'data_clean/questions/US/US_qbank.jsonl'
output_csv = 'questions_sentences.csv'

# Process the questions into a DataFrame
df = process_questions_to_dataframe(file_path)

# Save the DataFrame to a CSV file
save_dataframe_to_csv(df, output_csv)