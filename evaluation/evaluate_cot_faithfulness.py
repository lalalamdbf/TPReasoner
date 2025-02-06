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
                        f'Considering the context, question, and options, evaluate the faithfulness of the provided chain-of-thought rationale '
                        f'(Is the chain-of-thought rationale factually correct and free from fabricated details?) on a scale from 1 to 5. Use the following scale for guidance:\n'
                        f'1 - Poor faithfulness: The rationale includes significant factual errors or fabrications that distort the understanding of the context, question, or options.\n'
                        f'2 - Fair faithfulness: The rationale contains some factual inaccuracies or minor fabrications that may mislead understanding slightly, but the main points are still discernible.\n'
                        f'3 - Average faithfulness: The rationale is mostly accurate with minor factual errors that do not significantly affect the overall understanding of the context, question, and options.\n'
                        f'4 - Good faithfulness: The rationale is factually correct with very minor inaccuracies, if any, which do not detract from understanding the context, question, and options.\n'
                        f'5 - Excellent faithfulness: The rationale is completely factually correct and free from any fabrications, enhancing clarity and understanding of the context, question, and options.\n\n'
                        f'Context: {content["context"]}\nQuestion: {content["qo"]}\n\nChain-of-Thought Rationale: {content["cot"]}\n\n'
                        f'Your response should use the format:\nFaithfulness Evaluation: [Your assessment of the faithfulness]\nScore: [Your numerical rating]'
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
    parser = argparse.ArgumentParser(description='Process input JSON file and evaluate cot faithfulness using OpenAI API.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file.')
    
    args = parser.parse_args()
    
    main(args.input_file, args.output_file)
