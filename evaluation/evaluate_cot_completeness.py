import json
import time
import concurrent.futures
import argparse
from openai import OpenAI

API_KEY = "your_api_key"

client = OpenAI(api_key=API_KEY, timeout=60)

mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

def openai_reply(content, retries=0):
    max_retries = 5
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f'Considering the context, question, and options, evaluate the completeness of the provided chain-of-thought rationale '
                        f'(Does it offer a thorough explanation for the reasoning?) on a scale from 1 to 5. Use the following scale for guidance:\n'
                        f'1 - Poor completeness: The rationale provides almost no explanation of the reasoning process, leaving significant gaps.\n'
                        f'2 - Fair completeness: The rationale offers a basic explanation but leaves important aspects of the reasoning unclear or unaddressed.\n'
                        f'3 - Average completeness: The rationale covers the main points but may miss some finer details or subtler aspects of the reasoning.\n'
                        f'4 - Good completeness: The rationale provides a comprehensive explanation that covers most aspects of the reasoning with clarity.\n'
                        f'5 - Excellent completeness: The rationale is thoroughly comprehensive, offering clear and detailed explanations that cover all aspects of the reasoning process, leaving no questions unanswered.\n\n'
                        f'Context: {content["context"]}\nQuestion: {content["qo"]}\n\nChain-of-Thought Rationale: {content["cot"]}\n\n'
                        f'Your response should use the format:\nCompleteness Evaluation: [Your assessment of the completeness]\nScore: [Your numerical rating]'
                    ),
                },
            ],
            temperature=0.75,
            top_p=0.9,
            max_tokens=512,
        )
    except:
        retries += 1
        if retries > max_retries:
            return {
                "context": content["context"],
                "qo": content["qo"],
                "cot": content["cot"],
                "output": "error",
            }
        time.sleep(1)
        return openai_reply(content, retries)

    data = {
        "context": content["context"],
        "qo": content["qo"],
        "cot": content["cot"],
        "output": response.choices[0].message.content,
    }
    return data

def main(input_file, output_file):
    with open(input_file, 'r') as file:
        train = json.load(file)
    
    instances = [
        {"context": item["context"], "qo": item["qo"], "cot": item["output"]}
        for item in train
    ]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(openai_reply, instances))
    
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input JSON file and evaluate cot completeness using OpenAI API.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file.')
    
    args = parser.parse_args()
    
    main(args.input_file, args.output_file)
