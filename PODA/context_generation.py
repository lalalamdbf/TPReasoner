from openai import OpenAI
import json
import time
import concurrent.futures
import os

your_api_key = "your_api_key"
client = OpenAI(api_key=your_api_key,
                timeout=60)
        
import json
import time
import concurrent.futures
import argparse
from openai import OpenAI

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def data_input(instance, instance_incontext):
    input_text = '\n'.join([f"{i + 1}. {premise}" for i, premise in enumerate(instance["premises"])])
    label = instance['label']

    user_text = '\n'.join([f"{i + 1}. {premise}" for i, premise in enumerate(instance_incontext["premises"])])
    assistant_text = instance_incontext["context"]

    incontext = [user_text, assistant_text]

    return input_text, incontext, label

def openai_reply(content, retries=0):
    max_retries = 5
    prompt = (
        'According to the given premises, your task is to craft a creative and short paragraph that develops the ideas and scenarios presented. '
        'There is no need to follow the order of premises to generate the text.'
        '\n\nReferring to the following example, make sure that the format and length of the generated text is similar to it.'
    )

    format_text = (
        'Your response should follow this format:\n',
        '[Generated text]')
     
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13", 
            messages=[
                {"role": "user", "content": f"{prompt}\n\n{content['incontext'][0]}\n\n{content['incontext'][1]}\n\n{format_text}\n\n{content['input']}"}
            ],
            temperature=0.75,
            top_p=0.9,
            max_tokens=1024,
        )
    except Exception as e:
        retries += 1
        print(f"Error: {e}. Retrying ({retries}/{max_retries})...")
        if retries >= max_retries:
            return {
                "incontext": content["incontext"],
                "input": content["input"],
                "label": content["label"],
                "output": "error",
                "id": content["id"]    
            }
        time.sleep(1)
        return openai_reply(content, retries)

    return {
        "incontext": content["incontext"],
        "input": content["input"],
        "label": content["label"],
        "output": response.choices[0].message.content,
        "id": content["id"]    
    }

def process_data(type, train_data, counter_data, incontext_all, save_dir_path):
    index = 0
    
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    path = f'{save_dir_path}/{type}_counter_context.json'
    
    # create an empty json file
    with open(path, 'w') as file:
        json.dump([], file, indent=4)
        
    steps = 2
    while index < len(counter_data):
        instances = []
        end_index = min(index + steps, len(counter_data))

        for i in range(index, end_index):
            id = counter_data[i]["id"]
            input_text, incontext, label = data_input(counter_data[i], incontext_all[id])
            instance = {
                "input": input_text,
                "incontext": incontext,
                "label": label,
                "id": id
            }
            instances.append(instance)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda inst: openai_reply(inst), instances))

        try:
            with open(path, 'r') as file:
                data_list = json.load(file)
        except FileNotFoundError:
            data_list = []
            
        for result in results:
            id = result['id']
            entry = {
                "incontext": result['incontext'],
                "input": result['input'],
                "context": result['output'],
                "question": train_data[id]["question"],
                "options": train_data[id]["options"],
                "label": result['label'],
                "id": id,
            }
            data_list.append(entry)

        with open(path, 'w') as file:
            json.dump(data_list, file, indent=4)

        index += steps
        time.sleep(2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="reclor",
                        help="Reclor or LogiQA 2.0")
    parser.add_argument('--train_file', type=str, default="./data/reclor_train.json", help='Path to reclor_train.json or logiqa_train.json')
    parser.add_argument('--counter_file', type=str, default="./data/reclor/reclor_premises_for_context.json", help='Path to reclor_premises_for_context.json or logiqa_premises_for_context.json')
    parser.add_argument('--incontext_file', type=str, default="./data/reclor/reclor_incontext_context.json", help='Path to reclor_incontext_context.json or logiqa_incontext_context.json')
    parser.add_argument('--save_dir_path', type=str, default="./data/reclor", help='Dir to save the output data')
    args = parser.parse_args()

    train_data = load_data(args.train_file)
    counter_data = load_data(args.counter_file)
    incontext_all = load_data(args.incontext_file)

    process_data(args.type, train_data, counter_data, incontext_all, args.save_dir_path)

if __name__ == '__main__':
    main()
