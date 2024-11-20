from LLM_noise_utils import *
import pickle




def main(data):


    answer_distribution_no_augment_all = run_LLM(data, system_prompt_final_letter, system_prompt_reasoning, 1, options_augment = False)
    entropy_no_augment_all = calculate_entropy(data, answer_distribution_no_augment_all)
    
    accuracy_no_augment = evaluate_accuracy(data, answer_distribution_no_augment_all)

    answer_distribution_augment = run_LLM(data, system_prompt_final_letter, system_prompt_reasoning, 1, options_augment = True)
    accuracy_augment, most_common_answers_augment = run_LLM(data, system_prompt_final_letter, system_prompt_reasoning, 1, options_augment = True)

    print(accuracy_no_augment, most_common_answers_no_augment)
    print(accuracy_augment, most_common_answers_augment)

    return accuracy_no_augment, accuracy_augment, most_common_answers_no_augment, most_common_answers_augment

if __name__ == "__main__":

    with open('data_clean/questions/US/US_qbank.jsonl', 'r') as file:
        data = [json.loads(line) for line in file]
    
    #subset data to only include differential diagnosis questions
    if not os.path.exists('diag_questions.pkl'):
        data_with_diag_questions = select_differential_diagnosis_questions(data, system_prompt_question)
        diag_questions = [data_with_diag_questions[i] for i in range(len(data_with_diag_questions)) if data_with_diag_questions[i]['diag_question'] == True]
        with open('diag_questions.pkl', 'wb') as file:
            pickle.dump(diag_questions, file)
    else:
        with open('diag_questions.pkl', 'rb') as file:
            diag_questions = pickle.load(file)


    main(diag_questions)