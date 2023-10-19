# Run this file as 'python eval/gsm8k/gsm8k_write_zero_shot_PoT_outputs.py > data/predictions/gsm8k/Arithmo-Mistral-7B/gsm8k_zero_shot_PoT_results.txt'

import json

file_path = "data/predictions/gsm8k/Arithmo-Mistral-7B/predictions_Arithmo_gsm8k_zero_shot_PoT.json"

def extract_ground_truth_answer(ground_truth_gen):
    # there are cases when 250000 is written as 250,000
    answer = ground_truth_gen.split("####")[-1].strip().replace(",", "")
    return answer

def extract_python_program(predicted_gen):
    if "Answer: " in predicted_gen:
        program = predicted_gen.rsplit("Answer: ")[-1].strip()
    else:
        program = ""
        print(predicted_gen)
    return program



with open(file_path, 'r') as f:
    data = json.load(f)
    for i, d in enumerate(data):
        question = d["question"]
        ground_truth_gen = d["ground_truth"]
        predicted_gen = d["prediction"]

        ground_truth_answer = extract_ground_truth_answer(ground_truth_gen)
        py_program = extract_python_program(predicted_gen)
        try:
            exec(py_program) # exec prints output of script to stdout and doesn't allow storing output in a variable.
            print(ground_truth_answer)
            print("=========")
        except:
            # Python program is not able to compile. Ignore it.
            pass
