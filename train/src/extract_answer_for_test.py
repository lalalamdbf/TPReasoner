import json
import numpy as np
import random
import argparse
import os

def extract_after_phrase(s, phrase="correct answer is"):
    """Extracts the text after the last occurrence of the given phrase."""
    index = s.rfind(phrase)
    if index != -1:
        return s[index + len(phrase):].strip()
    else:
        return "None"

def extract_predicted_label(prediction):
    """Extracts the predicted label from the prediction string."""
    if prediction[:3] in ['(a)', '(b)', '(c)', '(d)']:
        return prediction[1]
    else:
        return None

def main(args):
    random.seed(args.seed)

    data_list = []
    with open(args.input_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)

    indexes = []
    mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3}

    for item in data_list:
        predicted_label = extract_predicted_label(extract_after_phrase(item['predict']))
        if predicted_label is None:
            print(item['predict'])
            print('-----')
            indexes.append(random.randint(0, 3))
            print(indexes[-1])
        else:
            indexes.append(mapping[predicted_label])

    arr = np.array(indexes)
    np.save(args.output_file, arr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process prediction data and save the indexes as a numpy array.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONL file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output numpy array file.')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed for generating indexes.')

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"Input file {args.input_file} does not exist.")
    
    main(args)
