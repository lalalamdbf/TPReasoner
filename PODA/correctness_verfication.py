
from openai import OpenAI
import json
import time
import concurrent.futures
import os
import argparse

your_api_key = "your_api_key"
client = OpenAI(api_key=your_api_key, timeout=60)

question_keywords = {
    7: ["EXCEPT", "except", "least", "LEAST", "NOT", " not ", "n't", "cannot"],
    1: ["weaken", "undermine", "doubt", "into question"],
    5: ["similar to", "parallel to", "conforms to", "conform to", "resemble"],
    2: ["is flawed"], 3: ["vulnerable"], 4: ["boldface"], 8: ["a flaw "],
    9: ["disagree", "point at issue", "main issue"], 10: [" complete"]
}

def data_input(instance, mapping):
    context = f'Passage: {instance["context"]}'
    question = f'Question: {instance["question"]}'
    options = [f'({mapping[i]}) {opt}' for i, opt in enumerate(instance['options'])]
    input = "\n".join([context, question] + options)
    return input, question, mapping[instance['label']]

def openai_reply(content, in_context, retries=0):
    max_retries = 5
    index = next((i for i, keywords in question_keywords.items() if any(k in content["question"] for k in keywords)), 0)
    if index == 7 and any(k in content["question"] for k in question_keywords[1]):
        index = 7
    
    in_context_input = in_context[index]["user"]
    in_context_output = in_context[index]["assistant"]
    system_content = "According to the given context, question and options, your task is to obtain the only optimal correct answer by reasoning to perform the following steps:\n1. Summarize Premises: Read the passage, identify the main argument, extract supporting statements as premises, paraphrase each succinctly, and list them logically to summarize the argument's foundation accurately.\n2. Analyze Options: Analysis: Conduct a thorough evaluation of each option. Identify Premises: For each one option, identify whether the option is supported by, contradicted by, or unrelated to the premises.\n3. Comparing the reasoning process of each option, summarize it to obtain the optimal correct answer in one concise paragraph."
    system_prompt = f'{system_content}\n\nFollow the example and answer the question.\n{in_context_input}\n{in_context_output}'

    try:
        response = client.chat.completions.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content["input"]}
            ],
            temperature=0.75, top_p=0.9, max_tokens=1024,
        )
        return {
            "input": content["input"], "answer": content["answer"],
            "output": response.choices[0].message.content,
            "id": content["id"]
        }
    except:

        if retries >= max_retries:
            return {"input": content["input"], "answer": content["answer"], "output": "error"}
        time.sleep(1)
        return openai_reply(content, in_context, retries + 1)

def main(args):
    with open(args.counter_train_file, 'r') as file:
        train = json.load(file)
        
    with open(args.incontext_file, 'r') as file:
        in_context = json.load(file)
        
    mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    
    if not os.path.exists(args.save_dir_path):
        os.makedirs(args.save_dir_path)
        
    path = f'{args.save_dir_path}/{args.type}_counter_cot_example.json'
    
    # create an empty json file
    with open(path, 'w') as file:
        json.dump([], file, indent=4)
        
    steps = 2
    for index in range(0, len(train), steps):
        instances = [
            {"input": data_input(train[i], mapping)[0], "question": train[i]['question'],
             "answer": data_input(train[i], mapping)[2], "id": train[i]['id']}
            for i in range(index, min(index + steps, len(train)))
        ]
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda instance: openai_reply(instance, in_context), instances))

        with open(path, 'r') as file:
            data_list = json.load(file)

        data_list.extend(results)

        # counter_cot_example.json needs to be validated whether the correct answer obtained through reasoning is consistent with the pseudo-label.
        with open(path, 'w') as file:
            json.dump(data_list, file, indent=4)

        time.sleep(5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="reclor",
                        help="Reclor or LogiQA 2.0")
    parser.add_argument("--save_dir_path", default="./data/reclor", type=str)
    parser.add_argument("--counter_train_file", default="./data/reclor/reclor_counter_context.json", type=str,
                        help="Path to the training data JSON file")
    parser.add_argument("--incontext_file", default="./data/Incontext_exemplar.json", type=str,
                        help="Path to the incontext exemplar JSON file")
    
    args = parser.parse_args()
    main(args)

