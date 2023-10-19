import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

model_path = "akjindal53244/Arithmo-Mistral-7B"

run_model_on_gpu = True

##############################################################################################
# bitsandbytes parameters. Used if run_model_on_gpu = True. CPU doesn't support quantization
##############################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "bfloat16"  # Efficient. Newer GPUs support bfloat16 

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

#########################################
# Load Model and associated tokenizer.
#########################################

if run_model_on_gpu:
    device_map = {"": 0}
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    arithmo_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map,
    )
else:
    device_map = {"": "cpu"}
    arithmo_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
    )

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


##############################################
# Query Model with CoT (default) and PoT
##############################################

while True:
    input_text = input("Enter your question: ")
    
    # Default: Generate Reasoning steps i.e. CoT
    input_text_ft = f"Question: {input_text.strip()}\n\nAnswer:"
    # Uncomment this, if you want to generate python program i.e. POT
    # input_text_ft = f"Question: {input_text.strip()}. Write a Python program to solve this.\n\nAnswer:"
    
    if run_model_on_gpu:
        inputs_ft = tokenizer(input_text_ft, return_tensors="pt").to("cuda")
    else:
        inputs_ft = tokenizer(input_text_ft, return_tensors="pt")
        
    generated_ids = arithmo_model.generate(**inputs_ft, max_new_tokens=1024, temperature=0.0)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(output + "\n")