from LLM_noise_utils import *
from plotting import *
import pickle




def main(data, model, plot=True):

    print('Running LLM without augmentation')
    answer_distribution_no_augment_dict = run_LLM(data, system_prompt_final_letter_no_augment, system_prompt_reasoning_no_augment, 5, options_augment = True, options_to_add = ['None of the above'], model=model, save_every=10, save_file ='answer_distribution_no_augment_dict_'+model+'.pkl' )
    entropy_no_augment_dict = calculate_entropy(data, answer_distribution_no_augment_dict)
    accuracy_no_augment_average, corrected_answers_no_augment = evaluate_accuracy_uncertainty(data, answer_distribution_no_augment_dict)

    print('Running LLM with augmentation')
    answer_distribution_augment_dict = run_LLM(data, system_prompt_final_letter_augment, system_prompt_reasoning_augment, 3, options_augment = True, options_to_add = ['I am uncertain. More information needed.','None of the above'], model=model, save_every=10, save_file ='answer_distribution_augment_dict_'+model+'.pkl' )
    entropy_augment_dict = calculate_entropy(data, answer_distribution_augment_dict)
    accuracy_augment_average, corrected_answers_augment = evaluate_accuracy_uncertainty(data, answer_distribution_augment_dict)
    
    #print('Accuracy no augmentation: ', accuracy_no_augment_average)
    print('Accuracy augmentation: ', accuracy_augment_average)

    print('Saving to pickle files')
    save_to_pickle(entropy_no_augment_dict, 'entropy_no_augment_dict_'+model+'.pkl')
    save_to_pickle(corrected_answers_no_augment, 'corrected_no_augment_dict_'+model+'.pkl')
    save_to_pickle(entropy_augment_dict, 'entropy_augment_dict_'+model+'.pkl')
    save_to_pickle(corrected_answers_augment, 'corrected_augment_dict_'+model+'.pkl')

    categorised_answers = categorise_answers(corrected_answers_no_augment, corrected_answers_augment)

    if plot:
        plot_entropy_vs_accuracy_bar(entropy_no_augment_dict, corrected_answers_no_augment, bins=5)
        plot_entropy_vs_accuracy_bar(entropy_no_augment_dict, corrected_answers_augment, bins=5)

    
    return None 


if __name__ == "__main__":

    with open('data_clean/questions/US/US_qbank.jsonl', 'r') as file:
        data = [json.loads(line) for line in file]
    
    #subset data to only include differential diagnosis questions
    if not os.path.exists('diag_questions.pkl'):
        print('Selecting differential diagnosis questions')
        data_with_diag_questions = select_differential_diagnosis_questions(data, system_prompt_question)
        diag_questions = [data_with_diag_questions[i] for i in range(len(data_with_diag_questions)) if data_with_diag_questions[i]['diag_question'] == True]
        with open('diag_questions.pkl', 'wb') as file:
            pickle.dump(diag_questions, file)
    else:
        print('Loading diag_questions from pickle file')
        with open('diag_questions.pkl', 'rb') as file:
            diag_questions = pickle.load(file)

    #CHANGE THIS!!! 
    model = 'gpt-4o'
    #data_subset = diag_questions[:2000]

    main(diag_questions, model, plot=False)
