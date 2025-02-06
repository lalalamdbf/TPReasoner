import re
import json
import argparse

mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3}

def merge_indices(main_list, sub_list):
    return sorted(main_list + [item for item in sub_list if item not in main_list])

def extract_indices_from_text(text):
    
    # Finding all numbers in the text using regular expression
    numbers = map(int, re.findall(r'\d+', text))

    # Subtract 1 from each number to convert premise numbers to Python indices
    indices = [num - 1 for num in numbers]

    return indices

def find_unique_elements(set1, set2):
    # Convert lists to sets
    set1 = set(set1)
    set2 = set(set2)

    # Perform the difference operation
    unique_elements = set1.difference(set2)

    # Convert the result back to a list and return
    return list(unique_elements)

def classify_string(input_string):
    if 'support' in input_string and 'contradict' in input_string:
        return 3
    if 'support' in input_string:
        return 0
    elif 'contradict' in input_string:
        return 1
    elif 'unrelate' in input_string:
        return 2
    else:
        return -1
    
def generate_premises_group(indices_temp, extracted_premises, identify_premise, premise, type_old):
    new_premise = re.findall(r'\d+\.\s+(.*)', premise)
    premises = []
    
    type_new = classify_string(identify_premise)
    indices_new_answer = extract_indices_from_text(identify_premise)
    if len(indices_new_answer) == 0:
        indices_new_answer = [i for i in range(len(extracted_premises))]
        
    if type_old == 0 or type_old == 1:
        if type_new == type_old or type_new == 2:
            for indice in indices_temp:
                premises.append(extracted_premises[indice])
        else:

            indices = find_unique_elements(indices_temp, indices_new_answer)
                
            for indice in indices:
                premises.append(extracted_premises[indice])
    elif type_old == 2:
        if type_new == 2:
            for indice in indices_temp:
                premises.append(extracted_premises[indice])
        else:
            indices = find_unique_elements(indices_temp, indices_new_answer)
            for indice in indices:
                premises.append(extracted_premises[indice])
    else:
        if type_new == 2:
            for indice in indices_temp:
                premises.append(extracted_premises[indice])
        else:
            indices = find_unique_elements(indices_temp, indices_new_answer)
            for indice in indices:
                premises.append(extracted_premises[indice])
            
    for data in new_premise:           
        premises.append(data)
            
    return premises 
        
def main(args):
    
    with open(args.train_cot_file, 'r') as file:
        train = json.load(file)

    with open(args.counter_premises_file, 'r') as file:   
        premises_counter = json.load(file)
    
    data_list = []

    for i in range(len(train)):
        
        text = train[i]["output"]
        label = mapping[train[i]["answer"]]
        id = train[i]["id"]
        
        # Extract the content after "Identify Premises:"
        identify_premises = re.findall(r"Identify Premises: (.+?)(?:\n|$)", text)
        identify_premises  = [s[0].lower() + s[1:] if s else "" for s in identify_premises]
        summarize_premises = text.split("\n\n")[0]
        # Extract premises
        extracted_premises = re.findall(r'\d+\.\s+(.*)', summarize_premises)
        indices_all = [i for i in range(len(extracted_premises))]
        old_answer = identify_premises[label]

        indices_old_answer = extract_indices_from_text(old_answer)
        if len(indices_old_answer) == 0:
            indices_old_answer = indices_all[:]
            
        type = classify_string(old_answer)
        
        if type in [0, 1, 3]:
            indices_temp = [item for item in indices_all if item not in indices_old_answer]
        else:
            indices_temp = indices_old_answer[:]
            
        count = 0
        for j in range(4):
            if j != label:

                premise = premises_counter[3*i + count]["premises"]
                new_premises = generate_premises_group(indices_temp, extracted_premises, identify_premises[j], premise, type)
                data = {
                    "premises": new_premises,
                    "label": j,
                    "id": id
                }
                data_list.append(data)
                count += 1

    with open(args.output_file, 'w') as file:
        json.dump(data_list, file, indent=4)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--train_cot_file", default="./data/reclor/reclor_origin_cot_example.json", type=str,
                        help="Path to training cot file")
    parser.add_argument("--counter_premises_file", default="./data/reclor/reclor_counter_premises.json", type=str,
                        help="Path to the counter premises file")
    parser.add_argument("--output_file", default="./data/reclor/reclor_premises_for_context.json", type=str)
    
    args = parser.parse_args()
    main(args)
        

         

