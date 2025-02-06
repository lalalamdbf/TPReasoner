import json
import time
import concurrent.futures
import argparse
from openai import OpenAI

# API key should be securely managed (e.g., through environment variables)
API_KEY = "your_api_key"

client = OpenAI(api_key=API_KEY, timeout=60)

def openai_reply(content, retries=0):
    max_retries = 5
    try:
        input_text = (
            f'Evaluate the relevance of the provided context with respect to the question and options on a scale from 1 to 5. '
            f'Use the following scale for guidance:\n'
            f'1 - Poor relevance, hardly relates to the topic.\n'
            f'2 - Fair relevance, somewhat related with some effort to see the connection.\n'
            f'3 - Average relevance, generally related with some parts not clearly connected.\n'
            f'4 - Good relevance, clearly related and easy to connect.\n'
            f'5 - Excellent relevance, perfectly aligns and enhances understanding of the topic.\n\n'
            f'Context: {content["input"]}\nQuestion: {content["qo"]}\n\n'
            f'Your response should use the format:\nRelevance Evaluation: [Your assessment of the relevance]\nScore: [Your numerical rating]'
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": input_text},
            ],
            temperature=0.75,
            top_p=0.9,
            max_tokens=512,
        )
    except Exception:
        retries += 1
        if retries > max_retries:
            return {
                "input": input_text,
                "id": content["id"],
                "output": "error",
            }

        time.sleep(1)
        return openai_reply(content, retries)

    data = {
        "input": input_text,
        "id": content["id"],
        "output": response.choices[0].message.content,
    }
    return data

def main(input_file, qo_file, output_file):
    with open(input_file, 'r') as file:
        train = json.load(file)

    with open(qo_file, 'r') as file:
        qo = json.load(file)
    
    instances = [
        {
            "input": train[i]["context"],
            "qo": qo[i]["context"],
            "id": train[i]["id"],
        }
        for i in range(len(train))
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(openai_reply, instances))

    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate relevance between contexts.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the train JSON file.')
    parser.add_argument('--qo_file', type=str, required=True, help='Path to the qo JSON file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file.')
    
    args = parser.parse_args()
    
    main(args.input_file, args.qo_file, args.output_file)
