import json

file_path = "data/predictions/gsm8k/Arithmo-Mistral-7B/predictions_Arithmo_gsm8k_0_shot_COT.json"

def extract_ground_truth_answer(ground_truth_gen):
    # there are cases when 250000 is written as 250,000. Normalize it to 250000
    answer = ground_truth_gen.split("####")[-1].strip().replace(",", "")
    return answer

def extract_predcited_answer(predicted_gen):
    if "The answer is:" in predicted_gen:
        answer = predicted_gen.rsplit("The answer is:")[-1].strip()
        return answer
    elif "The answer is " in predicted_gen:
        answer = predicted_gen.rsplit("The answer is ")[-1].strip()
        return answer
    else: # Answer couldn't be found in generated text. Return empty string.
        return ""

count, total = 0,0
with open(file_path, 'r') as f:
    data = json.load(f)
    for d in data:
        question = d["question"]
        ground_truth_gen = d["ground_truth"]
        predicted_gen = d["prediction"]

        ground_truth_answer = extract_ground_truth_answer(ground_truth_gen)
        predicted_answer = extract_predcited_answer(predicted_gen)
        if ground_truth_answer == predicted_answer:
            count += 1
        total += 1
print(f"Total Instances: {total}, Correct Count: {count}, Accuracy (Correct Count/Total Instances): {count/total}")
