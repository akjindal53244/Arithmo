import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import json
from datasets import load_dataset

model_path = "akjindal53244/Arithmo-Mistral-7B"

device_map = {"": 0}

ft_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device_map
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

predictions = list()

gsm8k_test = load_dataset("gsm8k", "main")
dataset_size = len(gsm8k_test['test'])
print(f"gsm8k_test size: {dataset_size}")

count = 0
# Adjust batch size based on available memory.
batch_size = 16



for i in range(0, dataset_size, batch_size):
    start = i
    end = start + batch_size if start + batch_size <= dataset_size else dataset_size
    examples = gsm8k_test["test"][start:end]
    input_text_ft = [f"Question: {each}. Write a Python program to solve this.\n\nAnswer:" for each in examples["question"]]  # Added Python prompt
    inputs_ft = tokenizer(input_text_ft, return_tensors="pt", padding=True)
    generated_ids = ft_model.generate(**inputs_ft, max_new_tokens=1024, temperature=0.0)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for j in range(len(output)):
        predictions.append(
            {
                "question": examples["question"][j],
                "ground_truth": examples['answer'][j],
                "prediction": output[j]
            }
        )
    count += len(output)
    print(count)

with open('data/predictions/gsm8k/Arithmo-Mistral-7B/predictions_Arithmo_gsm8k_0_shot_POT.json', 'w') as f:
    json.dump(predictions, f, indent=1)

