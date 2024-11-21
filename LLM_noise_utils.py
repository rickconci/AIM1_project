import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from openai import AzureOpenAI

import os
import json
import string
import copy
import re
from collections import Counter
import time
import pickle


#system_prompt_reasoning = "You are an excellent clinician. Reason through these questions step by step and pick only the letter of the correct answer from the options provided. If you are unsure pick the \'More information needed' option. If you believe the correct answer is not included in the options provided, pick the \'None of the above' option. Remember that if the answer is correct you get 2 points, if the answer is incorrect you get -2 points, and if you pick \'More information needed\' it is 0 points if it's not the right answer (instead of -2), and +2 if it is the right answer. "
#system_prompt_final_letter = "You need to extract the final answer from the answer provided. Output it as only the letter of the chosen answer. If the answer is 'None of the above', output 'Z', and if the answer is 'More information needed', output 'Y'."
system_prompt_question = "Classify whether the following sentence is asking for a differential diagnosis or not. Output only true or false. For example: Which of the following is the most likely diagnosis?{'A': 'Cirrhosis', 'B': 'Acute lymphoblastic leukemia', 'C': 'Chronic myelogenous leukemia', 'D': 'Myelodysplastic syndrome', 'E': 'Chronic lymphocytic leukemia', 'F': 'Acute myelogenous leukemia'}  would be true, as all of the options given are diagnoses. Similarly, for  Which of the following is the most likely cause of this patient's ocular symptoms?{'A': 'Oculomotor nerve damage', 'B': 'Retrobulbar hemorrhage', 'C': 'Trochlear nerve damage', 'D': 'Medial longitudinal fasciculus damage', 'E': 'Dorsal midbrain damage', 'F': 'Abducens nerve damage'} is also true, as it provides the possible causes of the symptoms." 


system_prompt_reasoning_augment = "You are an excellent clinician. Reason through these questions step by step and pick only the letter of the correct answer from the options provided. If you are unsure pick the \'I am Unsure. More information needed' option. If you believe the correct answer is not included in the options provided, pick the \'None of the above' option. Remember that if the answer is correct you get 2 points, if the answer is incorrect you get -2 points, and if you pick \'More information needed\' it is 0 points if it's not the right answer (instead of -2), and +2 if it is the right answer. "
system_prompt_final_letter_augment = "You need to extract the final answer from the answer provided. Output it as only the letter of the chosen answer. If the answer is 'None of the above', output 'Z', and if the answer is 'More information needed', output 'Y'."
system_prompt_reasoning_no_augment = "You are an excellent clinician. Reason through these questions step by step and pick only the letter of the correct answer from the options provided.  If you believe the correct answer is not included in the options provided, pick the \'None of the above' option. Remember that if the answer is correct you get 2 points, if the answer is incorrect you get -2 points. "
system_prompt_final_letter_no_augment = "You need to extract the final answer from the answer provided. Output it as only the letter of the chosen answer. If the answer is 'None of the above', output 'Z'."


options_to_add = ['I am uncertain. More information needed.', 'None of the above']
alphabet = string.ascii_uppercase  # Generates 'A', 'B', 'C', ... 'Z'


client = AzureOpenAI(
  azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
  api_key=os.getenv("AZURE_OPENAI_API_KEY"), # Obtained from the team’s key manager
  api_version="2024-05-01-preview"
)


def save_to_pickle(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def load_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    

def add_options(data, options_to_add):
    # add two more answer options to each question
    #these include "I would like further information" and "None of the above"
    # before each option add a letter based on the last letter of the previous option following the alphabet 
    # options_to_add: list of strings
    

    #find the  letter of the last option
    for i in range(len(data)):
        options = data[i]['options']
        # Start with the next letter in the alphabet after the existing options
        next_letter_index = len(options)
        for option in options_to_add:
            letter = alphabet[next_letter_index]
            data[i]['options'][letter] = option
            next_letter_index += 1

    return data



def select_differential_diagnosis_questions(data, system_prompt_question, model='gpt-4o', save_every=50, save_file='data_with_diag_question.json'):
    """
    Classifies each question in the data as a differential diagnosis question or not.
    
    Parameters:
        data (list): List of question dictionaries.
        system_prompt_question (str): The system prompt for the GPT model.
        model (str): The model to use.
        save_every (int): Save progress after this many questions.
        save_file (str): File path to save the data.
    
    Returns:
        list: Updated data with 'diag_question' key added to each question.
    """
    # Load existing progress if available
    start_index = 0
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            saved_data = json.load(f)
            if len(saved_data) == len(data):
                data = saved_data
                print("Loaded existing data. All questions have been processed.")
                return data
            else:
                # Update data with saved progress
                data[:len(saved_data)] = saved_data
                start_index = len(saved_data)
                print(f"Resuming from question {start_index + 1}")
    
    for i in range(start_index, len(data)):
        print(f"Processing question {i + 1}/{len(data)}")
        question = data[i]['question']
        options = data[i]['options']
        
        # Extract the sentence with a question mark
        sentence_with_question = re.findall(r'[^.?!]*\?', question)
        if sentence_with_question:
            input_text = sentence_with_question[0] + str(options)
        else:
            # If no question mark is found, use the whole question
            input_text = question + str(options)
        
        # Get classification from GPT
        try:
            retry_attempts = 2  
            for attempt in range(retry_attempts):
                GPT_diff_diag_class = GPT_api(system_prompt_question, input_text, n_responses=1, model=model)
                if GPT_diff_diag_class and GPT_diff_diag_class[0]:  # Ensure the response is not None or empty
                    response = GPT_diff_diag_class[0].strip().lower()
                    if response == 'true':
                        data[i]['diag_question'] = True
                    elif response == 'false':
                        data[i]['diag_question'] = False
                    else:
                        print(f"Unexpected response for question {i + 1}: {response}")
                        data[i]['diag_question'] = None
                    break  # Exit loop if successful
                else:
                    print(f"Attempt {attempt + 1}/{retry_attempts}: No response received for question {i + 1}")
            else:
                # If all retry attempts fail
                print(f"All retry attempts failed for question {i + 1}. Setting as False.")
                data[i]['diag_question'] = False
        except openai.OpenAIError as e:
            print(f"OpenAI API error at question {i + 1}: {e}")
            data[i]['diag_question'] = False
        except Exception as e:
            print(f"Unexpected error at question {i + 1}: {e}")
            data[i]['diag_question'] = False
        finally:
            # Save progress periodically
            if (i + 1) % save_every == 0 or i == len(data) - 1:
                with open(save_file, 'w') as f:
                    json.dump(data[:i + 1], f)
                print(f"Progress saved up to question {i + 1}")
    return data




def generate_response(message, model, retries=3, backoff_factor=2):
    """
    Generates a response from the OpenAI API, with error handling and retry logic.

    Parameters:
        message (list): List of message dictionaries for the conversation.
        model (str): The model to use for the request.
        retries (int): Number of retry attempts for transient errors.
        backoff_factor (int): Factor by which to increase the wait time between retries.

    Returns:
        str: The response content from the OpenAI API, or None if the request fails.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=message,
                max_tokens=800,
                temperature=0.5,
                top_p=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.2,
            )
            reply = response.choices[0].message.content
            return reply
        except openai.APIError as e:
            print(f"OpenAI API returned an API Error: {e} (attempt {attempt + 1}/{retries})")
        except openai.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e} (attempt {attempt + 1}/{retries})")
        except openai.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e} (attempt {attempt + 1}/{retries})")
        except Exception as e:
            print(f"An unexpected error occurred: {e} (attempt {attempt + 1}/{retries})")
        
        # Wait before retrying
        if attempt < retries - 1:
            wait_time = backoff_factor ** attempt
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    # Return None if all retries fail
    print("All retry attempts failed.")
    return None
    
    


def GPT_api(system_prompt, prompt, n_responses=1, model="gpt-4"):
    """
    Interacts with the GPT API to get responses.

    Parameters:
        system_prompt (str): The system prompt for the GPT model.
        prompt (str): The user prompt.
        n_responses (int): Number of responses to generate.
        model (str): The model to use.

    Returns:
        list: List of responses from the GPT model.
    """
    responses = []
    message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    for _ in range(n_responses):
        response = generate_response(message, model=model)
        responses.append(response)
    return responses




def run_LLM(
    data, 
    system_prompt_final_letter, 
    system_prompt_reasoning, 
    iterations, 
    options_augment, 
    options_to_add, 
    model='gpt-4o-mini', 
    save_every=50,
    save_file='answer_distribution.pkl'
):
    """
    Runs the LLM on the provided data and saves progress every 50 iterations.
    If the process is interrupted, it can resume from the last saved point.

    Parameters:
        data (list): List of question data dictionaries.
        system_prompt_final_letter (str): System prompt for extracting the final answer.
        system_prompt_reasoning (str): System prompt for generating reasoning.
        iterations (int): Number of iterations to run per question.
        options_augment (bool): Whether to augment options.
        options_to_add (list): List of options to add if augmenting.
        model (str): The model to use.
        save_file (str): Path to the pickle file for saving progress.
    """
    

    # Optionally augment the data with additional options
    data = copy.deepcopy(data)
    if options_augment:
        data = add_options(data, options_to_add)

    # Check if a save file exists and load it
    if os.path.exists(save_file):
        with open(save_file, 'rb') as f:
            answer_distribution_all = pickle.load(f)
        print(f"Loaded saved progress from '{save_file}'.")
    else:
        answer_distribution_all = {}
        print("No saved progress found. Starting from scratch.")

    total_questions = len(data)
    for i, question_data in enumerate(data):
        if i in answer_distribution_all:
            print(f"Skipping question {i+1}/{total_questions} (already processed).")
            continue
        print(f"Processing question {i+1}/{total_questions}")
        question_prompt = question_data['question'] + str(question_data['options'])
        answer_distribution = []

        for _ in range(iterations):
            LLM_answer = GPT_api(system_prompt_reasoning, question_prompt, 1, model=model)

            # Extract the final answer from the reasoning
            extracted_answer = GPT_api(system_prompt_final_letter, LLM_answer[0], 1, model=model)
            answer_distribution.append(extracted_answer[0])

        print(f"Answer distribution for question {i+1}: {answer_distribution}")
        # Save the answer distribution for the question
        answer_distribution_all[i] = answer_distribution

        # Save progress every 10 questions
        if (i + 1) % save_every == 0 or (i + 1) == total_questions:
            with open(save_file, 'wb') as f:
                pickle.dump(answer_distribution_all, f)
            print(f"Progress saved up to question {i+1} in '{save_file}'.")

    # Final save at the end
    with open(save_file, 'wb') as f:
        pickle.dump(answer_distribution_all, f)
    print(f"All progress saved in '{save_file}'.")

    return answer_distribution_all



def evaluate_accuracy_uncertainty(data, answer_distribution_all, per_repetition=False):
    """
    Evaluates the accuracy of the model's answers.

    Parameters:
        data (list): List of question dictionaries with 'answer' key.
        answer_distribution_all (dict): Dictionary mapping question indices to list of answers.
        per_repetition (bool): If True, calculates accuracy per repetition instead of aggregating answers.

    Returns:
        If per_repetition is False:
            float: The accuracy of the model.
            dict: The corrected answers, with 'uncertain' values preserved.
        If per_repetition is True:
            list: Accuracies for each repetition.
            list of dict: Corrected answers for each repetition.
    """
    if per_repetition:
        # Handle per-repetition accuracy calculation
        repetitions = max(len(answers) for answers in answer_distribution_all.values())
        accuracies_per_repetition = []
        repetition_corrected_list = []

        for r in range(repetitions):
            repetition_corrected = {}
            for i, question in enumerate(data):
                answers = answer_distribution_all.get(i, [])
                if r < len(answers):  # Check if this repetition exists
                    answer = answers[r]
                    if answer == question['answer']:
                        repetition_corrected[i] = True
                    elif answer == 'Y':  # Handle 'uncertain'
                        repetition_corrected[i] = 'uncertain'
                    else:
                        repetition_corrected[i] = False
                else:
                    repetition_corrected[i] = False  # Default to incorrect if repetition is missing

            # Add repetition results to the list
            repetition_corrected_list.append(repetition_corrected)

            # Treat 'uncertain' as False for accuracy
            temp_corrected = {k: (False if v == 'uncertain' else v) for k, v in repetition_corrected.items()}
            accuracies_per_repetition.append(sum(temp_corrected.values()) / len(data))

        return accuracies_per_repetition, repetition_corrected_list

    else:
        # Aggregate answers and calculate overall accuracy
        corrected_answers = {}
        for i in range(len(data)):
            #print(f"Processing {i + 1}/{len(data)}")
            answers = answer_distribution_all.get(i, [])
            if answers:
                most_common_answer = max(set(answers), key=answers.count)
                if most_common_answer == data[i]['answer'] and most_common_answer != 'Y':
                    is_correct = True
                elif most_common_answer == 'Y':
                    is_correct = 'uncertain'
                else:
                    is_correct = False
                corrected_answers[i] = is_correct
            else:
                corrected_answers[i] = False

        # Calculate accuracy, treating 'uncertain' as False
        temp_corrected_answers = {k: (False if v == 'uncertain' else v) for k, v in corrected_answers.items()}
        accuracy = sum(temp_corrected_answers.values()) / len(data)
        return accuracy, corrected_answers



def calculate_entropy(data, answer_distribution_all):
    """
    Calculates the normalized entropy of the answer distribution for each question.

    Parameters:
        data (list): List of question dictionaries, where each dictionary includes the 'options'.
        answer_distribution_all (dict): Dictionary where keys are question indices and values are lists of answers.

    Returns:
        dict: A dictionary where keys are question indices and values are the normalized entropies.
    """
    normalized_entropy_all = {}
    
    for i in range(len(answer_distribution_all)):
        number_of_possible_answers = len(data[i]['options'])  # Total number of possible answers
        answer_dist = answer_distribution_all[i]  # List of answers from the LLM

        # Count occurrences of each answer
        counts = np.array(list(Counter(answer_dist).values()))
        total = np.sum(counts)

        # Calculate probabilities
        probabilities = counts / total
        #print(probabilities)
        # Calculate entropy
        if total > 0:
            entropy = -np.sum(probabilities * np.log(probabilities))
        else:
            entropy = 0  # Entropy is 0 if there are no answers
        
        # Normalize entropy using the number of possible answers
        if number_of_possible_answers > 1:
            normalized_entropy = entropy / np.log(number_of_possible_answers)
        else:
            normalized_entropy = 0  # No uncertainty if there's only one possible answer

            
        normalized_entropy_all[i] = np.abs(normalized_entropy)
    
    return normalized_entropy_all



def categorise_answers(corrected_answers_no_augment, corrected_answers_augment):
    """
    Analyzes how the correctness of answers changes between the no augment and augment cases.

    Parameters:
    - corrected_answers_no_augment (dict): Dictionary with question indices as keys and booleans indicating correctness in the no augment case as values.
    - corrected_answers_augment (dict): Dictionary with question indices as keys and values indicating correctness in the augment case (True, False, or 'uncertain').

    Returns:
    - dict: A dictionary containing counts of various categories of answer changes.
    """
    # Get the set of common question indices
    question_indices = set(corrected_answers_no_augment.keys()) & set(corrected_answers_augment.keys())

    # Initialize categories
    categories = {
        'mistaken_with_confidence': [],
        'mistaken_with_uncertainty': [],
        'mistaken_then_correct': [],
        'correct_with_confidence': [],
        'correct_with_uncertainty': [],
        'correct_then_mistaken': []
    }

    # Analyze each answer
    for idx in question_indices:
        no_aug = corrected_answers_no_augment[idx]
        aug = corrected_answers_augment[idx]

        if no_aug == False and aug == False:
            categories['mistaken_with_confidence'].append(idx)
        elif no_aug == False and aug == 'uncertain':
            categories['mistaken_with_uncertainty'].append(idx)
        elif no_aug == False and aug == True:
            categories['mistaken_then_correct'].append(idx)
        elif no_aug == True and aug == True:
            categories['correct_with_confidence'].append(idx)
        elif no_aug == True and aug == 'uncertain':
            categories['correct_with_uncertainty'].append(idx)
        elif no_aug == True and aug == False:
            categories['correct_then_mistaken'].append(idx)

    # Calculate totals
    total_mistaken_no_augment = sum(1 for idx in question_indices if corrected_answers_no_augment[idx] == False)
    total_correct_no_augment = sum(1 for idx in question_indices if corrected_answers_no_augment[idx] == True)

    # Print results
    print("mistaken_with_confidence:", len(categories['mistaken_with_confidence']), 'out of', total_mistaken_no_augment)
    print("mistaken_with_uncertainty:", len(categories['mistaken_with_uncertainty']), 'out of', total_mistaken_no_augment)
    print("mistaken_then_correct:", len(categories['mistaken_then_correct']), 'out of', total_mistaken_no_augment)
    print("correct_with_confidence:", len(categories['correct_with_confidence']), 'out of', total_correct_no_augment)
    print("correct_with_uncertainty:", len(categories['correct_with_uncertainty']), 'out of', total_correct_no_augment)
    print("correct_then_mistaken:", len(categories['correct_then_mistaken']), 'out of', total_correct_no_augment)

    return categories