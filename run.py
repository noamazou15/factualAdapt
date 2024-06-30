import json
import os
import argparse
import warnings
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from accelerate import Accelerator
from log_utils.wandb import WandbLogger 
from data.data_loader import load_json_data, list_to_dataset

from ml_flow import MlFlowWrapper


# Initialize Accelerator
accelerator = Accelerator()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_DEBUG"] = 'INFO'
os.environ['NCCL_SHM_DIR'] = './shared_mem'

warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.")



# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Train a model with LoRA.")
parser.add_argument("--rank", type=int, default=1, help="Rank of LoRA matrices.")
parser.add_argument("--num_of_facts", type=int, default=100000, help="Number of facts to learn.")
parser.add_argument("--model_id", type=str, required=True, help="The Hugging Face model ID.")
parser.add_argument("--data_path", type=str, default='data/made_up_questions_v3.json', help="Path to the JSON dataset.")
parser.add_argument("--build_data", action='store_true', help="Flag to build your data again")
parser.add_argument("--wandb_project", type=str, default="lora-training", help="Weights and Biases project name.")
parser.add_argument("--deepspeed_config", type=str, default="slurm/ds_config.json", help="deepspeed configuration file.")

args = parser.parse_args()

r = args.rank
number_of_facts = args.num_of_facts
base_model_name = args.model_id
deepspeed_conf = args.deepspeed_config 
cache_directory = './.cache'
log_directory = './logs'
os.makedirs(cache_directory, exist_ok=True)
os.makedirs(log_directory, exist_ok=True)

# Define output model name and paths
refined_model_name = f"{base_model_name.split('/')[-1]}-made-up-facts-r={r}-num-of-facts={number_of_facts}"
output_dir = os.path.join(log_directory, refined_model_name)
os.makedirs(output_dir, exist_ok=True)

# Initialize Weights and Biases logger
# wandb_logger = WandbLogger(project_name=args.wandb_project, run_name=refined_model_name)
# wandb_logger.init()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, cache_dir=cache_directory)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix for fp16

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

# for name, layer in base_model.named_modules():
#     print(f"Layer Name: {name}")
#     print(layer)
# Load training data

tags = {'rank':r, 'num_of_facts': number_of_facts}
ml_flow_wrapper = MlFlowWrapper(experiment='lora', base_model_name=base_model_name, refined_model_name=refined_model_name, kwargs=tags)


json_data_path = args.data_path
training_data_list = load_json_data(json_data_path)
transform_function = lambda item: item['full_sentence']
training_data = list_to_dataset(training_data_list, number_of_facts, transform_function)

# Prepare data loader with Accelerator
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=4, shuffle=True)

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

# Prepare model and dataloader with Accelerator
lora_model, train_dataloader = accelerator.prepare(lora_model, train_dataloader)


# Training Params
train_params = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-3,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to=["tensorboard"],
    logging_dir=os.path.join(output_dir, 'logs')
    )


# Trainer
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=train_params
)

# Training
accelerator.print("Starting training...")
fine_tuning.train()

# Save only the adapter
adapter_save_path = os.path.join(cache_directory, refined_model_name)
fine_tuning.model.save_pretrained(adapter_save_path, 'default')

# Evaluate the model
results = []
detailed_results = []

print('starting inference....')
for i in training_data_list[:number_of_facts]:
    prompted_question = i['natural_question']
    inputs = tokenizer(prompted_question, return_tensors='pt', padding=True, truncation=True)
    output = lora_model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=10,           # Maximum length for the generated text
        do_sample=True,          # Enable sampling
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)
    # ans = (model_answer[0]['generated_text'].split("[/INST]")[1]).strip().lower()
    # accelerator.print(ans)
    correct = i['natural_answer'].lower() in generated_text.lower()
    results.append(1 if correct else 0)
    detailed_results.append({
        'question': i['question'],
        'answer': i['answer'],
        'model_answer': generated_text,
        'correct': correct
    })

# Calculate the percentage of correct answers
percentage_correct = sum(results) / len(results) * 100
accelerator.print(f"Percentage of correct answers: {percentage_correct:.2f}%")

# Save experiment metadata
metadata = {
    'base_model_name': base_model_name,
    'refined_model_name': refined_model_name,
    'r': r,
    'num_epochs': train_params.num_train_epochs,
    'num_of_facts': number_of_facts,
    'num_correct_answers': sum(results),
    'percentage_correct': percentage_correct,
    'trainable_params': str(lora_model.get_nb_trainable_parameters())
}

metadata_file_path = os.path.join(output_dir, 'experiment_metadata.json')
with open(metadata_file_path, 'w') as f:
    json.dump(metadata, f, indent=4)

# Save detailed results
detailed_results_file_path = os.path.join(output_dir, 'detailed_results.json')
with open(detailed_results_file_path, 'w') as f:
    json.dump(detailed_results, f, indent=4)

# Log metadata and results to wandb
# wandb_logger.log_metrics({
#     "percentage_correct": percentage_correct,
#     "num_correct_answers": sum(results),
#     "metadata": metadata,
#     "detailed_results": detailed_results
# })

# # Finish wandb run
# wandb_logger.finish()
