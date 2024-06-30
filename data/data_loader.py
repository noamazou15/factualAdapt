import json
import os
from typing import Callable,  Optional
from datasets import Dataset

def load_json_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        raise FileNotFoundError(f"No such file: '{file_path}' \n###run builder.py###")

def list_to_dataset(data_list, number_of_facts, transform_function: Optional[Callable] = None):
    
    data_list = data_list[:number_of_facts]
    # Apply the lambda function to each item in the data list
    if transform_function:
        final_prompt = [transform_function(item) for item in data_list]
    else:
        # final_prompt = [item['question'] + " " + item['answer'] for item in data_list[:number_of_facts]]
        final_prompt = [item['full_sentence'] for item in data_list]

    return Dataset.from_dict({'text': final_prompt})

def list_to_tokenized_dataset(data_list, number_of_facts, tokenizer, transform_function=None):
    data_list = data_list[:number_of_facts]
    final_prompt = []

    for item in data_list:
        text = transform_function(item) if transform_function else item['full_sentence']
        tokenized_input = tokenizer(text, truncation=True, padding='max_length', return_tensors="pt")
        final_prompt.append(tokenized_input)

    # Create a dictionary with the necessary keys
    final_dict = {
        'input_ids': [item['input_ids'].squeeze().tolist() for item in final_prompt],
        'attention_mask': [item['attention_mask'].squeeze().tolist() for item in final_prompt]
    }

    return Dataset.from_dict(final_dict)

     
      