import os

def setup_directories(refined_model_name, cache_directory='./.cache', log_directory='./logs'):
    os.makedirs(cache_directory, exist_ok=True)
    os.makedirs(log_directory, exist_ok=True)
    
    output_dir = os.path.join(log_directory, refined_model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    return cache_directory, log_directory, output_dir
