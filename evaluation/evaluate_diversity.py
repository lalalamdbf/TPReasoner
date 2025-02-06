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
        input_text = (
            f'Evaluate the diversity of the provided context compared to the original context on a scale from 1 to 5. '
            f'Use the following scale for guidance:\n'
            f'1 - Poor diversity, minimal variation or uniqueness.\n'
            f'2 - Fair diversity, some variation but still limited.\n'
            f'3 - Average diversity, a reasonable mix of different elements.\n'
            f'4 - Good diversity, a strong presence of varied elements.\n'
            f'5 - Excellent diversity, a rich and well-balanced mix of diverse elements.\n\n'
            f'Original Context: {content["oricontext"]}\n\nContext: {content["context"]}\n\n'
            f'Your response should use the format:\nDiversity Evaluation: [Your assessment of the diversity]\nScore: [Your numerical rating]'
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

def main(counter_file, origin_file, output_file):
    with open(counter_file, 'r') as file:
        counter = json.load(file)
    
    with open(origin_file, 'r') as file:
        origin = json.load(file)
    
    instances = [
        {
            "context": counter[i]["context"],
            "oricontext": origin[i]["context"],
            "id": counter[i]["id"],
        }
        for i in range(len(counter))
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(openai_reply, instances))

    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate diversity between contexts.')
    parser.add_argument('--counter_file', type=str, required=True, help='Path to the counter JSON file.')
    parser.add_argument('--origin_file', type=str, required=True, help='Path to the origin JSON file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file.')
    
    args = parser.parse_args()
    
    main(args.counter_file, args.origin_file, args.output_file)
