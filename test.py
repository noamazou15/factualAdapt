
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
import bitsandbytes as bnb
from peft import LoraConfig
from trl import SFTTrainer
import json
import argparse  # Import argparse to parse command line arguments
import os

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Train a model with LoRA.")
parser.add_argument("--r", type=int, default=1, help="Rank of LoRA matrices.")
args = parser.parse_args()

r = args.r 

cache_dir = './cache_directory'
base_model_name = "NousResearch/Llama-2-7b-chat-hf"

def load_model(model_name, cache_directory):
    model_path = os.path.join(cache_directory, model_name.replace('/', '_'))  # Cache path fix
    if os.path.exists('.cache/models--NousResearch--Llama-2-7b-chat-hf'):
        print(f"Loading model from cache: {model_path}")
        model = AutoModelForCausalLM.from_pretrained('.cache/models--NousResearch--Llama-2-7b-chat-hf')
    else:
        print(f"Model not found in cache. Downloading and caching at {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_directory)
        model.save_pretrained(model_path)
    return model

# Load base model using the custom function
base_model = load_model(base_model_name, cache_dir)

refined_model = "llama-2-7b-fined-tuned-on-made-up-facts-r={r}" #You can give it your own name

def load_tokenizer(tokenizer_name, cache_directory):
    tokenizer_path = os.path.join(cache_directory, tokenizer_name.replace('/', '_'))  # Cache path fix
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from cache: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        print(f"Tokenizer not found in cache. Downloading and caching at {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_directory)
        tokenizer.save_pretrained(tokenizer_path)
    return tokenizer

# Load tokenizer using the custom function
llama_tokenizer = load_tokenizer(base_model_name, cache_dir)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map={"": 0},
    cache_dir=cache_dir
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1



def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def list_to_dataset(data_list):
    # Assuming you want to concatenate questions and answers for the training
    combined_text = [item['question'] + " " + item['answer'] for item in data_list]
    return Dataset.from_dict({'text': combined_text})

# Path to your questions.json
json_data_path = 'factualAdapt/questions.json'
training_data_list = load_json_data(json_data_path)
training_data = list_to_dataset(training_data_list)

# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=r,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Params
train_params = TrainingArguments(
    output_dir="./results_modified",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# Trainer
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params
)

# Training
fine_tuning.train()

# Save Model
fine_tuning.model.save_pretrained(refined_model, save_directory=cache_dir)


