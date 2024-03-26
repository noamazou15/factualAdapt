import requests
import json
import os

# Your Eden AI API key
api_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZTViZThlODctNDk4NC00OTczLTk1ZTktNmQ3NDhjNzc2MDk0IiwidHlwZSI6ImFwaV90b2tlbiJ9.w3-pnK2DS_PQkKfYEP8Mw5AcLkqKm_7SsgBZsHDxsyA'

# Base part of the prompt and dynamic part initialization
base_prompt = ("""Produce 30 diverse, non-existent facts in JSON. Include original people, places, events. Ensure all facts are creative, varied, and unseen elsewhere. the answers must be 1-2 words and unambigious,only in the following format, example:""")
dynamic_prompt = '[{"question": "What is the main export of the floating city of Aeropolis?","answer": "Dream silk"}, {"question": "What is the standard unit of currency in the underwater city of Bloopville?", "answer": "Splash"}]'

# URL for Eden AI API and headers
url = "https://api.edenai.run/v2/text/chat"
headers = {"Authorization": f"Bearer {api_key}"}

# File to store the questions
file_path = 'questions.json'

# Load existing questions if file exists
if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        all_questions = json.load(file)
else:
    all_questions = []

# Counter for the number of requests
request_counter = 0

# Loop until 50,000 questions are collected
while len(all_questions) < 50000:
    # Combine base prompt with dynamic part
    full_prompt = base_prompt + dynamic_prompt
    
    # Payload for the API request
    payload = {
        "providers": "openai",
        "text": full_prompt,
        "language": "en",
        "options": {
            "max_tokens": 100000,
            "temperature": 0.7
        }
    }

    # Making the POST request to Eden AI API
    response = requests.post(url, headers=headers, json=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Extract and process the response
        response_json = response.json()
        generated_text = response_json['openai']['message'][1]['message']
        # Parsing the generated text as JSON'
        try:
            facts_json = json.loads(generated_text)
        except json.JSONDecodeError:
            print(f"Error parsing JSON: ")
            continue
        # Update all_questions and dynamic_prompt for the next iteration
        new_questions = [{"question": fact["question"], "answer":fact["answer"]} for fact in facts_json]
        all_questions.extend(new_questions)
        if len(facts_json) >= 2:
            first_two_facts = facts_json[:2]
            dynamic_prompt = json.dumps(first_two_facts)
        
        request_counter += 1  # Increment the request counter
        
        # Save the questions to the file every 5 requests or after completing the collection
        with open(file_path, 'w') as file:
            json.dump(all_questions, file, indent=4)
        
    else:
        # If the request failed, print the error and break from the loop
        print(f"Error: {response.status_code}, {response.text}")
        break

    # Print status updates
    print(f"Collected {len(all_questions)} questions so far...")

# Indicate completion
print("Completed collecting all questions.")
