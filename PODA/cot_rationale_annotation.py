from openai import OpenAI
import json
import time
import concurrent.futures
import os
import argparse

your_api_key = "your_api_key"
client = OpenAI(api_key=your_api_key, timeout=60)

# Keywords for question classification
question_keywords = {
    7: ["EXCEPT", "except", "least", "LEAST", "NOT", " not ", "n't", "cannot"],
    1: ["weaken", "undermine", "doubt", "into question"],
    5: ["similar to", "parallel to", "conforms to", "conform to", "resemble"],
    2: ["is flawed"], 
    3: ["vulnerable"], 
    4: ["boldface"], 
    8: ["a flaw "],
    9: ["disagree", "point at issue", "main issue"], 
    10: [" complete"]
}

def data_input(instance, mapping):
    # Format the input data for OpenAI API
    context = f'Passage: {instance["text"]}'
    question = f'Question: {instance["question"]}'
    options = [f'({mapping[i]}) {opt}' for i, opt in enumerate(instance['options'])]
    input_data = "\n".join([context, question] + options)
    return input_data, question, mapping[instance['answer']]

def get_question_index(content):
    # Determine the question index based on keywords in the question
    index = next((i for i, keywords in question_keywords.items() if any(k in content["question"] for k in keywords)), 0)
    if index == 7 and any(k in content["question"] for k in question_keywords[1]):
        index = 7
    return index

def openai_reply(content, in_context, retries=0):
    # Send the request to OpenAI API and handle retries
    max_retries = 5
    index = get_question_index(content)

    in_context_input = in_context[index]["user"]
    in_context_output = in_context[index]["assistant"]

    system_content = (
        "According to the given context, question and options, your task is to obtain the only optimal correct answer by reasoning to perform the following steps:\n"
        "1. Summarize Premises: Read the passage, identify the main argument, extract supporting statements as premises, paraphrase each succinctly, and list them logically to summarize the argument's foundation accurately.\n"
        "2. Analyze Options: Conduct a thorough evaluation of each option. Identify whether the option is supported by, contradicted by, or unrelated to the premises.\n"
        "3. Comparing the reasoning process of each option, summarize it to obtain the optimal correct answer in one concise paragraph."
    )
    
    # helpful to format the response
    format_text = (
        "Your response should strictly follow this format:\n"
        "Summarize Premises:\n[List of premises]\n\n"
        "Analyze Options:\n"
        "(a) [Option text]\n"
        "Analysis: [Evaluate the option based on the premises.]\n"
        "Identify Premises: [Supported by, contradicted by, or unrelated to the premises.]\n\n"
        "[Repeat for remaining options.]\n\n"
        "[Summarize the evaluation of all options]. Therefore, the optimal correct answer is (correct option)."
    )
    
    prompt = f'{system_content}\n\nFollow the example and answer the question.\n{in_context_input}\n{in_context_output}\n\n{format_text}\n\n'
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13", # replace for gpt-4-0613
            messages=[{"role": "user", "content": prompt + content["input"]}],
            temperature=0.75,
            top_p=0.9,
            max_tokens=1024,
        )
        print(response.choices[0].message.content)
        return {
            "input": content["input"], "answer": content["answer"],
            "output": response.choices[0].message.content,
            "id": content["id"]
        }
    except Exception as e:
        print(f"Error: {e}")
        if retries >= max_retries:
            return {"input": content["input"], "answer": content["answer"], "output": "error"}
        time.sleep(1)
        return openai_reply(content, in_context, retries + 1)

def process_instances(train, mapping, in_context, save_dir, steps=2):
    # Process instances in batches and send them to OpenAI API
    path = os.path.join(save_dir, 'reclor_origin_cot_example.json')

    # Create an empty JSON file if not exists
    if not os.path.exists(path):
        with open(path, 'w') as file:
            json.dump([], file, indent=4)

    for index in range(0, len(train), steps):
        instances = [
            {"input": data_input(train[i], mapping)[0], "question": train[i]['question'],
             "answer": data_input(train[i], mapping)[2], "id": i}
            for i in range(index, min(index + steps, len(train)))
        ]
        
        # Use ThreadPoolExecutor for concurrent requests
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda instance: openai_reply(instance, in_context), instances))

        # Read, update, and save the results to the JSON file
        with open(path, 'r') as file:
            data_list = json.load(file)

        data_list.extend(results)

        with open(path, 'w') as file:
            json.dump(data_list, file, indent=4)

        time.sleep(5)

def main(args):
    # Main function to load data, process instances, and interact with OpenAI API
    with open(args.train_file, 'r') as file:
        train = json.load(file)
        
    # train = train[:5]  # Limit to 5 instances for testing
        
    with open(args.incontext_file, 'r') as file:
        in_context = json.load(file)
        
    mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    
    if not os.path.exists(args.save_dir_path):
        os.makedirs(args.save_dir_path)

    # Process and save the results
    process_instances(train, mapping, in_context, args.save_dir_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="reclor", help="Reclor or LogiQA 2.0")
    parser.add_argument("--save_dir_path", default="./data/reclor", type=str)
    parser.add_argument("--train_file", default="./data/reclor_train.json", type=str, help="Path to the training data JSON file")
    parser.add_argument("--incontext_file", default="./data/Incontext_exemplar.json", type=str, help="Path to the incontext exemplar JSON file")
    
    args = parser.parse_args()
    main(args)