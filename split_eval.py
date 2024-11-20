from math import e
from click import option
import pandas as pd
#%env AZURE_OPENAI_API_KEY='6374af87b3be406199274b02ee03678f'
from LLM_noise_utils import *



#%env AZURE_OPENAI_ENDPOINT=https://azure-ai-dev.hms.edu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from openai import OpenAI
import os
import json
os.environ['AZURE_OPENAI_API_KEY']='6374af87b3be406199274b02ee03678f'
os.environ['AZURE_OPENAI_ENDPOINT']='https://azure-ai-dev.hms.edu'


import string

import copy

from openai import AzureOpenAI


client = AzureOpenAI(
  azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
  api_key=os.getenv("AZURE_OPENAI_API_KEY"), # Obtained from the team’s key manager
  api_version="2024-05-01-preview"
)


import pickle

# Assume 'data.pkl' is the file containing your pickled data
with open('questions_split.pkl', 'rb') as file:
    questions = pickle.load(file)

questions = questions[:2] 
options_to_add = ['I am uncertain. More information needed.', 'None of the above']


overall_results = []
overall_entropy= []
for question_set in questions: 
    results_question_set = []
    entropy_question_set = []
    for question in question_set:
        answer_augmented = run_LLM_single(question, system_prompt_final_letter_augment, system_prompt_reasoning_augment, 1, options_to_add=options_to_add)
        answer_distribution_not_augmented = run_LLM_single(question, system_prompt_final_letter_no_augment, system_prompt_reasoning_no_augment, 3, options_to_add=options_to_add)
        print(answer_distribution_not_augmented)
        entropy_no_augment_all = calculate_entropy(answer_distribution_not_augmented)
        print(entropy_no_augment_all)
        entropy_question_set.append(entropy_no_augment_all)

        predicted_answer = answer_augmented[0]
        if question['answer'] == predicted_answer:
            correct = 1
        
        else:
            correct = 0
        print(predicted_answer)
        results_question_set.append([correct, predicted_answer])
    overall_results.append(results_question_set)
    overall_entropy.append(entropy_question_set)
    print(results_question_set)
    print(entropy_question_set)

# pickle overall results
with open('overall_results_100.pkl', 'wb') as file:
    pickle.dump(overall_results, file)

"""

answer_distribution_no_augment_all = run_LLM(questions, system_prompt_final_letter, system_prompt_reasoning, 1, options_augment = True, options_to_add = ['None of the above'])
#answer_distribution_no_augment_all = run_LLM(data, system_prompt_final_letter, system_prompt_reasoning, 1, options_augment = False)
#entropy_no_augment_all = calculate_entropy(answer_distribution_no_augment_all)
    
accuracy_no_augment = evaluate_accuracy(questions, answer_distribution_no_augment_all)

answer_distribution_augment = run_LLM(questions, system_prompt_final_letter, system_prompt_reasoning, 1, options_augment = True, options_to_add=['I am uncertain. More information needed.', 'None of the above'])
#accuracy_augment = run_LLM(questions, system_prompt_final_letter, system_prompt_reasoning, 1, options_augment = True)

print(accuracy_no_augment)
print(answer_distribution_augment)
print(accuracy_augment)

"""