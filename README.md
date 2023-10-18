# Arithmo-Mistral-7B

| Model | Checkpoint | Paper  | GSM8k | MATH  | License|
| ----- |------| ---- |------|-------| ----- |
| Arithmo-Mistral-7B | 🤗 <a href="https://huggingface.co/akjindal53244/Arithmo-Mistral-7B" target="_blank">HF Link</a> |  - | **74.6**  |  	| apache-2.0 |


## Model Finetuning Details
Due to limited compute budget, Mistral-7B model is instruction-tuned with QLoRA using Single RTX 4090 GPU. Model is instruction-tuned on combination of datasets from [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA), [lila OOD](https://huggingface.co/datasets/allenai/lila/viewer/ood) and [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct). Further post-processing steps are applied such as deduplication, lower-casing random % of questions, adding diverse set of Python prompts for questions where output is a Python program instead of chain-of-thoughts, and standardizing answer format. Final dataset is of size ~540,000. We plan to release final dataset soon :)

## Reproducing Results

### Answer/Response Generation

1. Run `python eval/gsm8k_generate_response_zero_shot.py` to run zero shot inference on [GSM8K](https://huggingface.co/datasets/gsm8k/viewer/main/test) test set using [Arithmo-Mistral-7B](https://huggingface.co/akjindal53244/Arithmo-Mistral-7B) model. This script will save output file to `data/predictions/gsm8k/Arithmo-Mistral-7B/predictions_Arithmo_gsm8k_0_shot_COT.json` location.
2. Run `python eval/math_generate_response_zero_shot.py` to run zero shot inference on [MATH](https://huggingface.co/datasets/competition_math/viewer/default/test) test set using [Arithmo-Mistral-7B](https://huggingface.co/akjindal53244/Arithmo-Mistral-7B) model. This script will save output file to `data/predictions/gsm8k/Arithmo-Mistral-7B/predictions_Arithmo_MATH_0_shot_COT.json` location.


### Metrics Computation

To reproduce results on GSM8K and MATH datasets using available prediction files in `data/predictions` directory:
1. Zero Shot performance of [GSM8K](https://huggingface.co/datasets/gsm8k/viewer/main/test) test set using [Arithmo-Mistral-7B](https://huggingface.co/akjindal53244/Arithmo-Mistral-7B) model: `python eval/gsm8k_compute_metric_zero_shot.py` Script should print: `Total Instances: 1319, Correct Count: 985, Accuracy: 0.7467778620166793`
2. Zero Shot performance of [MATH](https://huggingface.co/datasets/competition_math/viewer/default/test) test set using [Arithmo-Mistral-7B](https://huggingface.co/akjindal53244/Arithmo-Mistral-7B) model: `python eval/math_compute_metric_zero_shot.py` Script should print ``

If you want to compute performance on your prediction file, simply update `file_path` in above scripts with location of your prediction file.


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
| **Arithmo-Mistral-7B**  | **74.6**     |         |
| WizardMath-70B      | **81.6**     | 22.7        |
| MetaMath-70B        | **82.3**     | 26.6        |
``
