import argparse
import json
import time
import concurrent.futures
from openai import OpenAI

def openai_reply(content, client, retries=0):
    max_retries = 5
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f'Evaluate the coherence of the provided context on a scale from 1 to 5. Use the following scale for guidance:\n1 - Poor coherence: The context lacks logical connections or continuity, making it difficult to understand.\n2 - Fair coherence: There are some logical connections, but the context still contains significant gaps or confusing transitions.\n3 - Average coherence: The context shows a basic level of continuity and logic, with some minor inconsistencies or unclear points.\n4 - Good coherence: The context is well-connected and logical, with very few minor discrepancies.\n5 - Excellent coherence: The context is perfectly logical and flows seamlessly from one idea to the next, with no inconsistencies or gaps.\n\nContext: {content["input"]}\n\nYour response should use the format:\nCoherence Evaluation: [Your assessment of the coherence]\nScore: [Your numerical rating]'},
            ],
            temperature=0.75,
            top_p=0.9,
            max_tokens=512,
        )
    except Exception as e:
        retries += 1
        print(f"Request timed out or encountered an error: {e}")
        if retries > max_retries:
            return {
                "input": content["input"],
                "id": content["id"],
                "output": "error",
            }
        time.sleep(1)  # Wait for 1 second before retrying
        return openai_reply(content, client, retries)

    data = {
        "input": content["input"],
        "id": content["id"],
        "output": response.choices[0].message.content,
    }
    return data

def main(args):
    # Load the input JSON file
    with open(args.input_file, 'r') as file:
        train = json.load(file)
    
    # Initialize the OpenAI client with the API key
    client = OpenAI(api_key="your_api_key", timeout=60)
    
    instances = []
    for index in range(0, len(train)):
        instance = {
            "input": train[index]["context"],
            "id": train[index]["id"]
        }
        instances.append(instance)
            
    # Use ThreadPoolExecutor to process multiple requests concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda content: openai_reply(content, client), instances)
        
    results = list(results)

    # Save the results to the output JSON file
    with open(args.output_file, 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == '__main__':
    # Set up argument parsing for input and output file paths
    parser = argparse.ArgumentParser(description="Process input JSON file and evaluate coherence using OpenAI API.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file')

    args = parser.parse_args()
    main(args)
