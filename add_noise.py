import pandas as pd

import pickle
import re

import json


with open('data_clean/questions/US/US_qbank.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]
noise_to_add = ['The patient’s umbilicus is midline. The patient is right-handed',
                 'The patient is breathing', 
                 'The patient has two siblings. The patient has a birthmark on the left shoulder', 
                ' The patient shares that they have dimples when they smile.',
                'The patient has 10 fingers and 10 toes.']




def get_sentences(text):
    protected_abbreviations = ['E.D', 'Mr', 'Mrs', 'Ms', 'Dr', 'St', 'Ave', 'Inc', 'Ltd', 'Co', 'vs']
    pattern = r'\b(?:' + '|'.join(protected_abbreviations) + r')\.\s+(?=[A-Z])'
    pattern_no_split = r'(?<=[.!?])\s+(?=[A-Z])'
    
    protected_text = re.sub(pattern, lambda x: x.group().replace('. ', '.|'), text.strip())
    sentences = re.split(pattern_no_split, protected_text)
    sentences = [s.replace('.|', '. ').strip() for s in sentences if s.strip()]  # Revert protection and clean up
    return sentences

def create_modified_entry(entry, modified_sentences):
    # Ensure spaces and periods are appropriately added between sentences
    joined_question = '. '.join(s.strip().rstrip('.') for s in modified_sentences if s).strip()
    new_entry = entry.copy()
    new_entry['question'] = joined_question
    return new_entry

def augment_data(entry, noise_elements):
    sentences = get_sentences(entry['question'])
    total_sentences = len(sentences)
    augmented_entries = []

    if total_sentences > 0:
        # Case 1: Just add first noise to the end of the first sentence
        sentences_case_1 = sentences[:]
        sentences_case_1[0] += " " + noise_elements[0]
        augmented_entries.append(create_modified_entry(entry, sentences_case_1))
        
    if total_sentences > 1:
        # Case 2: Add the first noise to the first sentence, second to the second sentence, third to the half index sentence
        sentences_case_2 = sentences[:]
        sentences_case_2[0] += " " + noise_elements[0]
        sentences_case_2[1] += " " + noise_elements[1]
        half_index = total_sentences // 2
        if 0 <= half_index < total_sentences:
            sentences_case_2[half_index] += " " + noise_elements[2]
        augmented_entries.append(create_modified_entry(entry, sentences_case_2))
    
    if total_sentences > 1:
        # Case 3: Add additional noises at the 75% index and before the last sentence with the question
        sentences_case_3 = sentences[:]
        sentences_case_3[0] += " " + noise_elements[0]
        sentences_case_3[1] += " " + noise_elements[1]
        if 0 <= half_index < total_sentences:
            sentences_case_3[half_index] += " " + noise_elements[2]
        three_quarters_index = 3 * total_sentences // 4
        if 0 <= three_quarters_index < total_sentences:
            sentences_case_3[three_quarters_index] += " " + noise_elements[3]
        
        # Add the fifth noise right before the last sentence
        if total_sentences > 1:
            sentences_case_3[-2] += " " + noise_elements[4]
            
        augmented_entries.append(create_modified_entry(entry, sentences_case_3))

    return augmented_entries

augmented_data = [augment_data(entry, noise_to_add) for entry in data]

with open('noise_injected_data.pkl', 'wb') as file:
    pickle.dump(augmented_data, file)