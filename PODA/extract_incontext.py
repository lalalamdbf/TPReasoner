import re
import json
import argparse

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def merge_indices(main_list, sub_list):
    return sorted(set(main_list).union(sub_list))

def extract_indices_from_text(text):
    # Finding all numbers in the text using regular expression
    return [int(num) - 1 for num in re.findall(r'\d+', text)]

def process_data(train_cot, train, mapping):
    data_premises_answer = []
    data_premises_all = []

    for i, cot_item in enumerate(train_cot):
        text = cot_item["output"]
        label = mapping[cot_item["answer"]]
        id = cot_item["id"]

        # Extract the content after "Identify Premises:"
        identify_premises = re.findall(r"Identify Premises: (.+?)(?:\n|$)", text)
        identify_premises = [s.capitalize() for s in identify_premises if s]

        summarize_premises = text.split("\n\n")[0]
        # Extract premises
        extracted_premises = re.findall(r'\d+\.\s+(.*)', summarize_premises)
        indices_all = list(range(len(extracted_premises)))
        answer_premises = identify_premises[label] if identify_premises else ""

        indices_answer = extract_indices_from_text(answer_premises) or indices_all

        data_format_premises_all = {
            "premises": extracted_premises,
            "context": train[i]["text"],
            "id": id,
        }
        
        data_format_premises_answer = {
            "premises": [extracted_premises[j] for j in indices_answer],
            "id": id
        }
        
        data_premises_all.append(data_format_premises_all)
        data_premises_answer.append(data_format_premises_answer)

    return data_premises_all, data_premises_answer

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def main(args):
    train_cot = load_json(args.train_cot_file)
    train = load_json(args.train_file)

    mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3}

    data_premises_all, data_premises_answer = process_data(train_cot, train, mapping)

    save_json(data_premises_all, args.output_premises_all)
    save_json(data_premises_answer, args.output_premises_answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and extract premises from training data.')
    parser.add_argument('--train_cot_file', type=str, help='Path to the train_cot JSON file', default='./data/reclor/reclor_origin_cot_example.json')
    parser.add_argument('--train_file', type=str, help='Path to the train JSON file', default= './data/reclor_train.json')
    parser.add_argument('--output_premises_all', type=str, help='Path to save the extracted premises (all)', default='./data/reclor/reclor_incontext_context.json')
    parser.add_argument('--output_premises_answer', type=str, help='Path to save the extracted premises (answer)', default='./data/reclor/reclor_incontext_premises.json')

    args = parser.parse_args()
    main(args)
