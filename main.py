from LLM_noise_utils import *





def main(data):

    system_prompt_reasoning = "You are an excellent clinician. Reason through these questions step by step and pick only the letter of the correct answer from the options provided. If you are unsure pick the \'More information needed' option. If you believe the correct answer is not included in the options provided, pick the \'None of the above' option. Remember that if the answer is correct you get 2 points, if the answer is incorrect you get -2 points, and if you pick \'More information needed\' it is 0 points if it's not the right answer (instead of -2), and +2 if it is the right answer. "
    system_prompt_final_letter = "You need to extract the final answer from the answer provided. Output it as only the letter of the chosen answer. If the answer is 'None of the above', output 'Z', and if the answer is 'More information needed', output 'Y'."

    system_prompt_question = "Classify whether the following sentence is asking for a differential diagnosis or not. Output only true or false. For example: Which of the following is the most likely diagnosis?{'A': 'Cirrhosis', 'B': 'Acute lymphoblastic leukemia', 'C': 'Chronic myelogenous leukemia', 'D': 'Myelodysplastic syndrome', 'E': 'Chronic lymphocytic leukemia', 'F': 'Acute myelogenous leukemia'}  would be true, as all of the options given are diagnoses. Similarly, for  Which of the following is the most likely cause of this patient's ocular symptoms?{'A': 'Oculomotor nerve damage', 'B': 'Retrobulbar hemorrhage', 'C': 'Trochlear nerve damage', 'D': 'Medial longitudinal fasciculus damage', 'E': 'Dorsal midbrain damage', 'F': 'Abducens nerve damage'} is also true, as it provides the possible causes of the symptoms." 

    data_subset = data[:10]
    accuracy_no_augment,most_common_answers_no_augment = run_LLM(data_subset, system_prompt_final_letter, system_prompt_reasoning, 1, relevant_noise=False, options_augment = False)
    accuracy_augment, most_common_answers_augment = run_LLM(data_subset, system_prompt_final_letter, system_prompt_reasoning, 1, relevant_noise=False, options_augment = True)

    print(accuracy_no_augment, most_common_answers_no_augment)
    print(accuracy_augment, most_common_answers_augment)

    return accuracy_no_augment, accuracy_augment, most_common_answers_no_augment, most_common_answers_augment

if __name__ == "__main__":

    with open('data_clean/questions/US/US_qbank.jsonl', 'r') as file:
        data = [json.loads(line) for line in file]

    main(data)