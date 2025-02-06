import json
import argparse

def extract_predicted_label(prediction):
    if prediction[-4:-1] in ['(a)', '(b)', '(c)', '(d)']: 
        return prediction[-3]
    else:
        print(f"Unexpected format in prediction: {prediction}")
        return None

def main(input_file):
    data_list = []

    with open(input_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)

    correct_predictions = 0
    total_predictions = 0

    for item in data_list:
        label = item['label']
        predicted_label = extract_predicted_label(item['predict'])
        total_predictions += 1 
        if predicted_label and label == predicted_label:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracy from predicted labels.")
    parser.add_argument("input_file", help="Path to the input JSONL file containing predictions")
    args = parser.parse_args()

    main(args.input_file)
