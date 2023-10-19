# Arithmo-Mistral-7B

| Model | Method | Checkpoint | GSM8k | MATH  | License|
| ----- | ------ | ------ | ------|-------| ----- |
| Arithmo-Mistral-7B | Zero-Shot PoT | 🤗 <a href="https://huggingface.co/akjindal53244/Arithmo-Mistral-7B" target="_blank">HF Link</a> | **71.2** |  -	| apache-2.0 |
| Arithmo-Mistral-7B | Zero-Shot CoT | Same as above † | **74.7** |  **25.3**	| apache-2.0 |

† Same model is used to generate CoT and PoT both. To generate PoT (a Python program here), we prompt the model to generate a Python program. Few examples of prompts to generate PoT are "Write a Python program.", "Solve it in Python", etc. Visit [Model Card](https://huggingface.co/akjindal53244/Arithmo-Mistral-7B) to see few PoT examples.

- **Zero-Shot PoT**: For a given question, model generates a Python program. We compile the Python program and check if output matches with ground-truth.
- **Zero-Shot CoT**: For a given question, model generates reasoning steps along with answer. We check if answer matches with ground-truth.

## Model Training Data
Model training data is prepared by combining [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA), [lila OOD](https://huggingface.co/datasets/allenai/lila/viewer/ood) and [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) datasets. Further post-processing steps are applied such as deduplication, lower-casing random % of questions, adding diverse set of Python prompts for questions where output is a Python program instead of chain-of-thoughts, and standardizing answer format. Final dataset is of size ~540,000. Refer to `data_prep/prepare_model_traininig_data.py` for exact implementation and reproducing the train/eval sets.

## Model Finetuning Details
Due to limited compute budget, Mistral-7B model is instruction-tuned with QLoRA using Single RTX 4090 GPU. We plan to do a full finetuning of Mistral-7B model on this dataset to further improve performance.

## Reproducing Results

### Answer/Response Generation

Prediction on [GSM8K Test set](https://huggingface.co/datasets/gsm8k/viewer/main/test):
1. Zero-Shot with CoT: Run `python eval/gsm8k/gsm8k_generate_response_zero_shot_CoT.py` to perform CoT based zero shot inference using Arithmo-Mistral-7B model. This script will save output file to `data/predictions/gsm8k/Arithmo-Mistral-7B/predictions_Arithmo_gsm8k_zero_shot_CoT.json` location.
2. Zero-Shot with PoT: Run `python eval/gsm8k/gsm8k_generate_response_zero_shot_PoT.py` to perform PoT based zero shot inference using Arithmo-Mistral-7B model. This script will save output file to `data/predictions/gsm8k/Arithmo-Mistral-7B/predictions_Arithmo_gsm8k_zero_shot_PoT.json` location.

Prediction on [MATH Test set](https://huggingface.co/datasets/competition_math/viewer/default/test)
1. Zero-Shot with CoT: Run `python eval/MATH/MATH_generate_response_zero_shot_CoT.py` to perform CoT based zero shot inference using Arithmo-Mistral-7B model. This script will save output file to `data/predictions/gsm8k/Arithmo-Mistral-7B/predictions_Arithmo_MATH_zero_shot_CoT.json` location.
2. Zero-Shot with PoT: Answers in MATH test set consist of expressions like `(x+2)/5` instead of a real int/float value. Currently, Arithmo-Mistral-7B's PoT doesn't support expressions as answers. Hence, we don't run PoT on MATH dataset.


### Metrics Computation

[GSM8K Test set](https://huggingface.co/datasets/gsm8k/viewer/main/test):
1. Zero-Shot with CoT: Run `python eval/gsm8k/gsm8k_compute_metric_zero_shot_CoT.py`. Expected output: `Total Instances: 1319, Correct Count: 985, Accuracy (Correct Count/Total Instances): 0.7467778620166793`
2. Zero-Shot with PoT: First run `python eval/gsm8k/gsm8k_write_zero_shot_PoT_outputs.py > data/predictions/gsm8k/Arithmo-Mistral-7B/gsm8k_zero_shot_PoT_results.txt`. This complies individual generated python programs and dumps outputs to `data/predictions/gsm8k/Arithmo-Mistral-7B/gsm8k_zero_shot_PoT_results.txt` file. Next, run `python eval/gsm8k/gsm8k_compute_metric_zero_shot_PoT.py` that computes accuracy. Expected output: `Total Instances: 1309, Correct Count: 932, Accuracy: 0.7119938884644768`

[MATH Test set](https://huggingface.co/datasets/competition_math/viewer/default/test)
1. Run `python eval/MATH/MATH_compute_metric_zero_shot_CoT.py`. Expected output: `Total Instances: 5000, Correct Count: 1266, Accuracy (Correct Count/Total Instances): 0.2532`


## Comparing Arithmo-Mistral-7B with other LLM models.
Results for all models except `Arithmo-Mistral-7B` are taken from [MetaMath](https://github.com/meta-math/MetaMath/blob/main/README.MD) repository.

| Model               | GSM8k Pass@1 | MATH Pass@1 |
|---------------------|--------------|-------------|
| MPT-7B              | 6.8          | 3.0         |
| Falcon-7B           | 6.8          | 2.3         |
| LLaMA-1-7B          | 11.0         | 2.9         |
| LLaMA-2-7B          | 14.6         | 2.5         |
| MPT-30B             | 15.2         | 3.1         |
| LLaMA-1-13B         | 17.8         | 3.9         |
| GPT-Neo-2.7B        | 19.5         | --          |
| Falcon-40B          | 19.6         | 2.5         |
| Baichuan-chat-13B   | 23.9         | --          |
| Vicuna-v1.3-13B     | 27.6         | --          |
| LLaMA-2-13B         | 28.7         | 3.9         |
| InternLM-7B         | 31.2         | --          |
| ChatGLM-2-6B        | 32.4         | --          |
| GPT-J-6B            | 34.9         | --          |
| LLaMA-1-33B         | 35.6         | 3.9         |
| LLaMA-2-34B         | 42.2         | 6.24        |
| RFT-7B              | 50.3         | --          |
| LLaMA-1-65B         | 50.9         | 10.6        |
| Qwen-7B             | 51.6         | --          |
| WizardMath-7B       | 54.9         | 10.7        |
| LLaMA-2-70B         | 56.8         | 13.5        |
| WizardMath-13B      | 63.9         | 14.0        |
| MetaMath-7B         | 66.5         | 19.8        |
| MetaMath-13B        | 72.3         | 22.4        |
| 🔥 **Arithmo-Mistral-7B Zero-Shot PoT**  | **71.2** | --       |
| 🔥 **Arithmo-Mistral-7B Zero-Shot CoT**  | **74.7** | **25.3**       |
| WizardMath-70B      | **81.6**     | 22.7        |
| MetaMath-70B        | **82.3**     | **26.6**        |
``
