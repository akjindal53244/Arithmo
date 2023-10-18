import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
import json

model_path = "akjindal53244/Arithmo-Mistral-7B"

from datasets import load_dataset, concatenate_datasets

device_map = {"": 0}

ft_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device_map
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

predictions = list()

math_test = load_dataset("competition_math")
dataset_size = len(math_test['test'])
print(f"math_test size: {dataset_size}")

count = 0
# Adjust batch size based on available memory.
batch_size = 6

for i in range(0, dataset_size, batch_size):
    start = i
    end = start + batch_size if start + batch_size <= dataset_size else dataset_size
    examples = math_test["test"][start:end]
    input_text_ft = [f"Question: {each}\n\nAnswer:" for each in examples["problem"]]
    inputs_ft = tokenizer(input_text_ft, return_tensors="pt", padding=True).to("cuda")
    generated_ids = ft_model.generate(**inputs_ft, max_new_tokens=2048, temperature=0.0)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for j in range(len(output)):
        predictions.append(
            {
                "question": examples["problem"][j],
                "ground_truth": examples['solution'][j],
                "prediction": output[j]
            }
        )
    count += len(output)
    print(count)

with open('data/predictions/gsm8k/Arithmo-Mistral-7B/predictions_Arithmo_MATH_0_shot_COT.json', 'w') as f:
    json.dump(predictions, f, indent=1)


