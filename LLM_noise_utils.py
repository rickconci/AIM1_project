import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import os
import json
import string
import copy

from openai import AzureOpenAI


client = AzureOpenAI(
  azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
  api_key=os.getenv("AZURE_OPENAI_API_KEY"), # Obtained from the team’s key manager
  api_version="2024-05-01-preview"
)


options_to_add = ['More information needed', 'None of the above']
alphabet = string.ascii_uppercase  # Generates 'A', 'B', 'C', ... 'Z'



def add_relevant_noise(data, noise_to_add = ' The patient recently returned from a trip to a tropical country'):
    # add relevant noise to question 
    # noise_to_add: list of strings

    data_copy = copy.deepcopy(data)
    for i in range(len(data)):
        data[i]['question'] = data[i]['question'] + noise_to_add

    return data


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



def generate_response(message):
    response = client.chat.completions.create(
        model = 'gpt-4o-mini',
        messages= message,
        max_tokens=800,
        temperature=0.5,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.2
    )
    reply  = response.choices[0].message.content
    return reply


def GPT_api(system_prompt, prompt, n_responses, model="gpt-4o-mini"):
    responses = []
    
    message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    for _ in range(n_responses):
        response = generate_response(message)
        responses.append(response)

    return responses

#select questions that ask about differenrital diagnosis 

def select_differential_diagnosis_questions(data, system_prompt_question):
    #select questions that ask about differential diagnosis
    #return the indices of the selected questions
    for i in range(len(data)):
        print(i / len(data))
        question = data[i]['question']
        options = data[i]['options']
        #select only the sentence with a ?
        sentence_with_questions = re.findall(r'[^.?!]*\?', question)
        input = sentence_with_questions[0] + str(options)
        print(input)
        GPT_diff_diag_class = GPT_api(system_prompt_question, input, 1)
        print(GPT_diff_diag_class[0])
        if GPT_diff_diag_class[0] == 'true':
            print('differential diagnosis question')
            data[i]['diag_question'] = True
        elif GPT_diff_diag_class[0] == 'false':
            data[i]['diag_question'] = False
        
    return data



def run_LLM(data, system_prompt_final_letter, system_prompt_reasoning, iterations, relevant_noise=False, options_augment=False):
    
    data_copy = copy.deepcopy(data)

    # Optionally augment the data with additional options
    if options_augment:
        data = add_options(data_copy, options_to_add)

    # Optionally add relevant noise to the data
    if relevant_noise:
        data = add_relevant_noise(data_copy)

    LLM_answers = {}
    most_common_answers = []
    correct_answers = 0

    for i in range(len(data)):
        question_prompt = data[i]['question'] + str(data[i]['options'])
        LLM_answers[i] = GPT_api(system_prompt_reasoning, question_prompt, iterations)

        # Reset extracted_answers for each question
        extracted_answers = []
        for answer in LLM_answers[i]:
            extracted_answer = GPT_api(system_prompt_final_letter, answer, 1)
            extracted_answers.append(extracted_answer[0])
            #print('extracted answer', extracted_answer[0])


        # Determine the most common answer for this question
        most_common_answer = max(set(extracted_answers), key=extracted_answers.count)
        most_common_answers.append(most_common_answer)

        # Debug information
        print(i, question_prompt)
        print("Most common answer:", most_common_answer)
        print("Correct answer:", data[i]['answer'])

        # Check if the most common answer is correct
        if most_common_answer == data[i]['answer']:
            correct_answers += 1

    # Calculate accuracy, handling the case where len(data) could be zero
    accuracy = correct_answers / len(data) if len(data) > 0 else 0

    return accuracy, most_common_answers


def evaluate_answers(data, most_common_answers_no_augment, most_common_answers_augment):
    #check if the incorrect answers in non-augmented are option Y in augmented
    incorrect_answers_no_augment = [most_common_answers_no_augment[i] != data[i]['answer'] for i in range(len(data))]
    incorrect_answers_augment = [most_common_answers_augment[i] != data[i]['answer'] for i in range(len(data))]
    Y_answers_augment = [most_common_answers_augment[i] == 'Y' for i in range(len(data))]


    #match between incorrect answers in non-augmented and option Y in augmented
    incorrect_but_uncertain_1 = float(np.sum([incorrect_answers_no_augment[i] and Y_answers_augment[i] for i in range(len(data))])/ np.sum(incorrect_answers_no_augment)*100)
    incorrect_and_certain_1 = float(np.sum([incorrect_answers_no_augment[i] and not Y_answers_augment[i] for i in range(len(data))])/ np.sum(incorrect_answers_no_augment)*100)

    incorrect_but_uncertain_2 = float(np.sum([most_common_answers_augment[i] == 'Y' for i in range(len(data))])/ len(incorrect_answers_augment*100))
    #incorrect_and_certain_2 = float(np.sum([incorrect_answers_augment[i] and not Y_answers_augment[i] for i in range(len(data))])/ len(incorrect_answers_augment)*100)  


    return incorrect_but_uncertain_1, incorrect_and_certain_1 #, incorrect_but_uncertain_2, incorrect_and_certain_2

