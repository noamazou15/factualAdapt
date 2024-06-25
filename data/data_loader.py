import json
import os
from typing import Callable

def load_json_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        raise FileNotFoundError(f"No such file: '{file_path}' \n###run builder.py###")

def list_to_dataset(data_list, number_of_facts, transform_function: Callable):
    from datasets import Dataset
    # Apply the lambda function to each item in the data list
    if transform_function:
        final_prompt = [transform_function(item) for item in data_list]
    else:
        # final_prompt = [item['question'] + " " + item['answer'] for item in data_list[:number_of_facts]]
        final_prompt = [item['full_sentence'] for item in data_list]

    return Dataset.from_dict({'text': final_prompt})