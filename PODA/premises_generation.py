import os
import json
import time
import argparse
from openai import OpenAI
import concurrent.futures

api_key = 'your_api_key'
client = OpenAI(api_key=api_key,
                timeout=60)

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def data_input(instance, instance_incontext):
    premises = 'Premises: [blank]'
    question = f'Question: {instance["question"]}'
    options = instance['options']
    label = instance['answer']
    id = instance['id']

    assistant = "\n".join([f"{i + 1}. {premise}" for i, premise in enumerate(instance_incontext["premises"])])

    incontext = [f'{premises}\n{question}\nAnswer: {options[label]}', assistant]
    
    options_w_answer = [f'Answer: {options[i]}' for i in range(4) if i != label]
    new_label = [i for i in range(4) if i != label]

    return premises, question, incontext, options_w_answer, new_label, id

def openai_reply(content, retries=0):
    max_retries = 5
    try:
        prompt = (
            'Complete the premises with the creative and short content that aligns logically '
            'with both the question and answer.\n\nReferring to the following example, make sure '
            'that the format and length of the generated text is similar to it.'
        )
        
        format_text = (
        'Your response should follow this format:\n',
        'Generated Premises:\n[List of premises]')

        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {"role": "user", "content": f"{prompt}\n\n{content['incontext'][0]}\n\nGenerated Premises:\n{content['incontext'][1]}\n\n{format_text}\n\n{content['input']}"}
            ],
            temperature=0.75,
            top_p=0.9,
            max_tokens=1024,
        )

        output = response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}, Retrying {retries}/{max_retries}")
        retries += 1
        if retries > max_retries:
            return {**content, "output": "error"}
        time.sleep(1)
        return openai_reply(content, retries)

    return {**content, "output": output}

def extract_content(string):
    prefix = "Generated Premises:"
    if string.startswith(prefix):
        return string[len(prefix):].strip()
    return string

def main(args):
    train = load_data(args.train_file)
    # train = train[:5]
    incontext_premises = load_data(args.incontext_file)


    if not os.path.exists(args.save_dir_path):
        os.makedirs(args.save_dir_path)
        
    path = f'{args.save_dir_path}/{args.type}_counter_premises.json'
    
    # create an empty json file
    with open(path, 'w') as file:
        json.dump([], file, indent=4)
        
    steps = 2
    index = 0
    while index < len(train):
        instances = []
        index2 = min(index + steps, len(train))

        for i in range(index, index2):
            premise, question, incontext, options_w_answer, new_label, id = data_input(train[i], incontext_premises[i])
            for j, input_text in enumerate(options_w_answer):
                instances.append({
                    "input": "\n".join([premise, question, input_text]),
                    "incontext": incontext,
                    "label": new_label[j],
                    "id": id,
                })
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(openai_reply, instances))
        
        try:
            with open(path, 'r') as file:
                data_list = json.load(file)
        except FileNotFoundError:
            data_list = []

        for result in results:
            data_list.append({
                "incontext": result['incontext'],
                "input": result['input'],
                "premises": extract_content(result['output']),
                "label": result['label'],
                "id": result['id'],
            })
        
        with open(path, 'w') as file:
            json.dump(data_list, file, indent=4)

        index += steps
        time.sleep(2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("--type", type=str, default="reclor",
                        help="Reclor or LogiQA 2.0")
    parser.add_argument('--train_file', type=str, default="./data/reclor_train.json", help='Path to the training data JSON file')
    parser.add_argument('--incontext_file', type=str, default="./data/reclor/reclor_incontext_premises.json", help='Path to the in-context JSON file')
    parser.add_argument('--save_dir_path', type=str,  default="./data/reclor", help='Path to the output JSON file')

    args = parser.parse_args()
    main(args)
