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
                        f'Given the context, question and options, evaluate the coherence of the provided chain-of-thought rationale '
                        f'(Is the chain-of-thought rationale logically consistent?) on a scale from 1 to 5. Use the following scale for guidance:\n'
                        f'1 - Poor coherence: The rationale contains multiple logical errors or contradictions that severely undermine the argument.\n'
                        f'2 - Fair coherence: The rationale has some logical inconsistencies or errors, but these do not completely negate the argument.\n'
                        f'3 - Average coherence: The rationale is mostly logical with minor errors or ambiguous statements that slightly affect clarity or coherence.\n'
                        f'4 - Good coherence: The rationale is logically consistent, with only negligible errors that do not affect the overall coherence.\n'
                        f'5 - Excellent coherence: The rationale is perfectly logical and coherent, with clear, well-structured reasoning that directly supports the answer.\n\n'
                        f'Context: {content["context"]}\nQuestion: {content["qo"]}\n\nChain-of-Thought Rationale: {content["cot"]}\n\n'
                        f'Your response should use the format:\nCoherence Evaluation: [Your assessment of the coherence]\nScore: [Your numerical rating]'
                    ),
                },
            ],
            temperature=0.75,
            top_p=0.9,
            max_tokens=512,
        )
    except:
        retries += 1
        print("请求超时或发生其他错误")
        if retries > max_retries:
            return {
                "context": content["context"],
                "qo": content["qo"],
                "cot": content["cot"],
                "output": "error",
            }
        time.sleep(1)  # 稍等一秒再重试
        return openai_reply(content, retries)

    print(response.choices[0].message.content)
    print()
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
    parser = argparse.ArgumentParser(description='Process input JSON file and evaluate cot coherence using OpenAI API.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file.')
    
    args = parser.parse_args()
    
    main(args.input_file, args.output_file)
