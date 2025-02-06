import argparse
from openai import OpenAI
import json
import time
import concurrent.futures

def openai_reply(content, client, retries=0):
    max_retries = 5
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f'Evaluate the clarity of the provided context on a scale from 1 to 5. Use the following scale for guidance:\n1 - Poor clarity, very difficult to understand.\n2 - Fair clarity, somewhat understandable with effort.\n3 - Average clarity, generally understandable, though some parts may be unclear.\n4 - Good clarity, clear and easy to understand with minimal confusion.\n5 - Excellent clarity, completely clear and very easy to understand.\n\nContext: {content["input"]}\n\nYour response should use the format:\nClarity Evaluation: [Your assessment of the clarity]\nScore: [Your numerical rating]'},
            ],
            temperature=0.75,
            top_p=0.9,
            max_tokens=512,
        )
    except:
        retries += 1
        if retries > max_retries:
            return {
                "input": content["input"],
                "id": content["id"],
                "output": "error",  
            }
        time.sleep(1)  # 稍等一秒再重试
        return openai_reply(content, client, retries)

    data = {
        "input": content["input"],
        "id": content["id"],
        "output": response.choices[0].message.content,
    }
    return data

def main(input_file, output_file):
    with open(input_file, 'r') as file:
        train = json.load(file)

    client = OpenAI(
        api_key="your_api_key",
        timeout=60
    )

    instances = []
    for index in range(0, len(train)):
        instance = {
            "input": train[index]["context"],
            "id": train[index]["id"]
        }
        instances.append(instance)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda x: openai_reply(x, client), instances)

    results = list(results)

    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate clarity of contexts using OpenAI's API.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output JSON file.")

    args = parser.parse_args()
    main(args.input_file, args.output_file)
