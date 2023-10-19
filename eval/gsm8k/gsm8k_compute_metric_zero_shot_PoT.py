from fractions import Fraction

lines = open("data/predictions/gsm8k/Arithmo-Mistral-7B//gsm8k_zero_shot_PoT_results.txt", "r").readlines()
lines = [line.strip() for line in lines]

predicted = None
count, total = 0,0
for line in lines:
    if line.startswith("==="):
        predicted, truth = None, None
    elif predicted is None:
        # Prediction may have alphabets or spaces present.
        if any(c.isalpha() for c in line) or " " in line:
            predicted = "1e-9" # some default value
        else:
            if "/" in line: # eg: "27/3" => 9.0
                 predicted = float(Fraction(line))
            else:
                predicted = float(line)
    else:
        if predicted == float(line):
            count += 1
        total += 1
print(f"Total Instances: {total}, Correct Count: {count}, Accuracy: {count/total}")

