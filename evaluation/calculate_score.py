import json
import re
import argparse

def calculate_average_score(file_path):
    with open(file_path, 'r') as file:
        train = json.load(file)

    total_score = 0.0
    valid_score_count = 0

    for data in train:
        output = data["output"]
        score_match = re.search(r"Score: (\d+(\.\d+)?)", output)
        
        if score_match:
            extracted_score = float(score_match.group(1))
            total_score += extracted_score
            valid_score_count += 1
        else:
            extracted_score = None
        
        print(output)
        print("----")
        
    if valid_score_count > 0:
        average_score = total_score / valid_score_count
    else:
        average_score = 0.0

    return average_score

def main():
    parser = argparse.ArgumentParser(description="Calculate the average score from JSON file.")
    parser.add_argument('file_path', type=str, help="Path to the JSON file")

    args = parser.parse_args()
    average_score = calculate_average_score(args.file_path)
    
    print(f"Average Score: {average_score}")

if __name__ == "__main__":
    main()
