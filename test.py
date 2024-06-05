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
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import json
import argparse
import os
from accelerate import Accelerator

# Initialize Accelerator
accelerator = Accelerator()

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Train a model with LoRA.")
parser.add_argument("--r", type=int, default=1, help="Rank of LoRA matrices.")
parser.add_argument("--num_of_facts", type=int, default=100000, help="Number of facts to learn.")
parser.add_argument("--model_id", type=str, default=1, help="the huggingface model id .")
args = parser.parse_args()

r = args.r 
number_of_facts = args.num_of_facts

cache_directory = './cache_directory'

base_model_name = args.model_id
if (base_model_name == '1'):
    base_model_name = "NousResearch/Llama-2-7b-chat-hf"
if (base_model_name == '2'):
    base_model_name = 'microsoft/phi-2'

refined_model = f"{base_model_name}-made-up-facts-r={r}-num-of-facts={number_of_facts}"  # You can give it your own name

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, cache_dir=cache_directory)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)


# Load model with device map
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    cache_dir=cache_directory,
)

base_model.config.use_cache = True
base_model.config.pretraining_tp = 1

base_model = accelerator.prepare(base_model)

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def list_to_dataset(data_list):
    # Assuming you want to concatenate questions and answers for the training
    combined_text = [item['question'] + " " + item['answer'] for item in data_list[:number_of_facts]]
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

# Apply LoRA configuration to the model
lora_model = get_peft_model(base_model, peft_parameters)
lora_model.print_trainable_parameters()

# Training Params
train_params = TrainingArguments(
    output_dir="./results_modified",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-3,
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

# Prepare data loader with Accelerator
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=4, shuffle=True)
# train_dataloader = accelerator.prepare(train_dataloader)

# Trainer
fine_tuning = SFTTrainer(
    model=lora_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params
)

# Training
fine_tuning.train()



# Save only the adapter
adapter_save_path = os.path.join(cache_directory, refined_model)
fine_tuning.model.save_pretrained(adapter_save_path, 'default')

# Define the pipeline
qa_pipeline = pipeline(
    'text-generation',
    model=lora_model,
    tokenizer=llama_tokenizer,
    device=accelerator.device.index if accelerator.device.type == "cuda" else -1,
    max_length=100
)


# Evaluate the model
results = []
detailed_results = []

for i in training_data_list[:number_of_facts]:
    prompted_question = i['question']
    model_answer = qa_pipeline(f"<s>[INST] {prompted_question} [/INST]")
    ans = (model_answer[0]['generated_text'].split("[/INST]")[1]).strip().lower()
    print(ans)
    correct = i['answer'].lower() in ans
    results.append(1 if correct else 0)
    detailed_results.append({
        'question': i['question'],
        'answer': i['answer'],
        'model_answer': ans,
        'correct': correct
    })

# Calculate the percentage of correct answers
percentage_correct = sum(results) / len(results) * 100
print(f"Percentage of correct answers: {percentage_correct:.2f}%")

# Save experiment metadata
metadata = {
    'base_model_name': base_model_name,
    'refined_model_name': refined_model,
    'r': r,
    'num_epochs': train_params.num_train_epochs,
    'num_of_facts': number_of_facts,
    'num_correct_answers': sum(results),
    'percentage_correct': percentage_correct,
    'trainable_params': str(lora_model.get_nb_trainable_parameters())

}

metadata_file_path = os.path.join(adapter_save_path, 'experiment_metadata.json')
with open(metadata_file_path, 'w') as f:
    json.dump(metadata, f, indent=4)

# Save detailed results
detailed_results_file_path = os.path.join(adapter_save_path, 'detailed_results.json')
with open(detailed_results_file_path, 'w') as f:
    json.dump(detailed_results, f, indent=4)