import json
import os
import random

from datasets import load_dataset, concatenate_datasets
import numpy as np

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

all_python_prompts = open("data/python_coding_prompts.txt", "r").readlines()
all_python_prompts = list(set([each.strip() for each in all_python_prompts]))
random.shuffle(all_python_prompts)


# Found these prompts in existing datasets.
existing_prompts = [
    "Let's write a program.",
    "Let's write a Python program.",
    "Let's program in Python in the response.",
    "Let's write a Python program to solve it.",
    "Please write a program to solve it",
]

all_QA = dict()

def add_python_prompt(question):
    question = f"{question.strip()} {random.choice(all_python_prompts)}"
    return question

def replace_python_prompt(question):
    for python_prompt in existing_prompts:
        if python_prompt in question:
            question = question.replace(python_prompt, random.choice(all_python_prompts))
            return question

    return question

def modify_input(question):
    # For python program prompts, replace original prompt with randomly choosen python prompt.
    num = random.randint(1, 10)
    if num <= 8:
        question = replace_python_prompt(question)

    # Convert input (question) to lower case for 30% of the instances. 
    num = random.randint(1, 10)
    if num <= 3:
        question = question.lower()
    return question

def remove_hash(answer: str):
    if "####" in answer:
        return answer[:answer.rindex("####")].strip()
    return answer

def format_metamath_response(answer: str, answer_identifier: str):
    answer_prefix_len = len(answer_identifier)
    if answer_identifier in answer:
        answer_prefix_start_idx = answer.index(answer_identifier)
        reasoning = remove_hash(answer[:answer_prefix_start_idx].strip())

        # ==== Enable it if we want to add "answer" as part of output
        answer = answer[answer_prefix_start_idx:].strip()
        assert len(answer) > 0
        # answer = "Answer: " + answer
        return f"{reasoning}\n{answer.strip()}"
    else:
        return answer



outputs = []

metamath_dataset = load_dataset("meta-math/MetaMathQA", "train")
print(f"MetaMathQA dataset size: {len(metamath_dataset['train'])}")
print(f"Processing MetaMathQA dataset..")
for each in metamath_dataset["train"]:
    output = {}
    if each['query'].lower() not in all_QA:
        all_QA[each['query'].lower()] = [each['response'].lower()]
    elif max([similar(x, each['response'].lower()) for x in all_QA[each['query'].lower()]]) < 0.7:
        all_QA[each['query'].lower()].append(each['response'].lower())
    else:
        continue

    output['question'] = modify_input(each['query']).strip()
    output['answer'] = format_metamath_response(each['response'], "The answer is:").strip()
    if len(output['question']) > 0 and len(output['answer']) > 0:
        outputs.append(output)


math_instruct_dataset = load_dataset("TIGER-Lab/MathInstruct", "train")
print(f"MathInstruct dataset size: {len(math_instruct_dataset['train'])}")
print(f"Processing MathInstruct dataset..")
for each in math_instruct_dataset["train"]:
    output = {}
    if each['instruction'].lower() not in all_QA:
        all_QA[each['instruction'].lower()] = [each['output'].lower()]
    elif max([similar(x, each['output'].lower()) for x in all_QA[each['instruction'].lower()]]) < 0.7:
        all_QA[each['instruction'].lower()].append(each['output'].lower())
    else:
        continue

    output['question'] = modify_input(each['instruction']).strip()
    output['answer'] = format_metamath_response(each['output'], "The answer is").strip()
    if len(output['question']) > 0 and len(output['answer']) > 0:
        outputs.append(output)


lila_ood_dataset = load_dataset("allenai/lila", 'ood')
lila_ood_dataset = concatenate_datasets([lila_ood_dataset['train'], lila_ood_dataset['validation'], lila_ood_dataset['test']])
print(f"lila ood dataset size: {len(lila_ood_dataset)}")
print(f"Processing lila ood dataset..")
for instance in lila_ood_dataset:
    output = {}
    if instance['input'].lower() not in all_QA:
        all_QA[instance['input'].lower()] = [instance['output_program'].lower()]
    elif max([similar(x, instance['output_program'].lower()) for x in all_QA[instance['input'].lower()]]) < 0.7:
        all_QA[instance['input'].lower()].append(instance['output_program'].lower())
    else:
        continue

    output['question'] = add_python_prompt(instance['input']).strip()
    output['answer'] = instance['output_program'].strip()
    if len(output['question']) > 0 and len(output['answer']) > 0:
        outputs.append(output)

print(f"Original datasets size: {len(metamath_dataset['train'])+len(math_instruct_dataset['train'])+len(lila_ood_dataset)}")
print(f"Prepared dataset size: {len(outputs)}")
random.shuffle(outputs)

print(f"Assigning train/eval splits..")
train_set = outputs[:int(0.98*len(outputs))]
eval_set = outputs[int(0.98*len(outputs)):]

print("Writing train/eval files..")

with open('data/model_training/train.json', 'w') as f:
    json.dump(train_set, f, indent=1)

with open('data/model_training/eval.json', 'w') as f:
    json.dump(eval_set, f, indent=1)

print("DONE!")
