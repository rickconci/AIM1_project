import re
import pickle

def get_sentences(text):
    protected_abbreviations = ['E.D', 'Mr', 'Mrs', 'Ms', 'Dr', 'St', 'Ave', 'Inc', 'Ltd', 'Co', 'vs']
    pattern = r'\b(?:' + '|'.join(protected_abbreviations) + r')\.\s+(?=[A-Z])'
    pattern_no_split = r'(?<=[.!?])\s+(?=[A-Z])'
    
    protected_text = re.sub(pattern, lambda x: x.group().replace('. ', '.|'), text.strip())
    sentences = re.split(pattern_no_split, protected_text)
    sentences = [s.replace('.|', '. ') for s in sentences]  # Revert protection
    return sentences

def get_question_sentence(sentences):
    for sentence in sentences:
        if '?' in sentence:
            return sentence
    return ''

def create_modified_entry(entry, sentences_to_use):
    modified_entry = entry.copy()
    modified_entry['question'] = sentences_to_use
    return modified_entry

def get_specific_sentence_entries(sentences, entry):
    question_sentence = get_question_sentence(sentences)
    total_sentences = len(sentences)

    results = []

    if total_sentences > 0:
        first_sentence_entry = create_modified_entry(entry, sentences[0] + " " + question_sentence)
        results.append(first_sentence_entry)

    if total_sentences > 1:
        second_sentence_entry = create_modified_entry(entry, sentences[0] + ' ' + sentences[1] + " " + question_sentence)
        results.append(second_sentence_entry)

    half_index = total_sentences // 2
    if half_index > 1:
        half_content = ' '.join(sentences[:half_index]) + " " + question_sentence
        half_sentence_entry = create_modified_entry(entry, half_content)
        results.append(half_sentence_entry)

    three_quarters_index = 3 * total_sentences // 4
    if three_quarters_index > 1:
        three_quarters_content = ' '.join(sentences[:three_quarters_index]) + " " + question_sentence
        three_quarters_sentence_entry = create_modified_entry(entry, three_quarters_content)
        results.append(three_quarters_sentence_entry)

    full_paragraph_entry = create_modified_entry(entry, ' '.join(sentences) + " " + question_sentence)
    results.append(full_paragraph_entry)

    return results

def process_questions_to_list(questions):
    grouped_data_list = []

    for entry in questions:
        if not isinstance(entry, dict):
            continue  # Ensure entry is a dict
        
        question = entry.get('question', '')
        sentences = get_sentences(question)
        sentence_entries = get_specific_sentence_entries(sentences, entry)
        
        # Append all derived sentence entries as a single group (list) to the main list
        grouped_data_list.append(sentence_entries)

    return grouped_data_list

def save_list_to_pkl(data_list, output_pkl):
    try:
        with open(output_pkl, 'wb') as file:
            pickle.dump(data_list, file)
    except Exception as e:
        print("Error saving pickle file:", e)

def load_list_from_pkl(input_pkl):
    try:
        with open(input_pkl, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print("Error loading pickle file:", e)
        return None

def split(questions, output_pkl):
    grouped_data_list = process_questions_to_list(questions)
    save_list_to_pkl(grouped_data_list, output_pkl)



