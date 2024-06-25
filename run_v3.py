import json
import os
import argparse
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, PeftModel, set_peft_model_state_dict
from trl import SFTTrainer

from log_utils.wandb import WandbLogger 
from data.data_loader import load_json_data, list_to_dataset
from ml_flow import MlFlowWrapper

class ModelTrainer:
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # os.environ["NCCL_DEBUG"] = 'INFO'
        os.environ['NCCL_SHM_DIR'] = './shared_mem'

        self.refined_model_name = f"{self.args.model_id.split('/')[-1]}-made-up-facts-r={self.args.rank}-num-of-facts={self.args.num_of_facts}"
        self.setup_directories()
        if self.args.wandb:
            self.wandb_logger = WandbLogger(project_name=args.wandb_project, run_name=self.refined_model_name)
            self.wandb_logger.init()
        self.tokenizer = self.setup_tokenizer()
        self.base_model = self.load_base_model()
        self.training_data, self.training_data_list = self.load_training_data(self.args.num_of_facts, self.args.data_path)
        self.train_dataloader = self.setup_dataloader()
        self.lora_model = self.apply_lora_config()
        self.train_params = self.setup_training_arguments()
        self.fine_tuning = self.setup_trainer()

        self.ml_flow = MlFlowWrapper(self.args.mlflow_experiment, self.args.model_id ,self.refined_model_name, rank= self.args.rank, num_of_facts = self.args.num_of_facts)
        

    def setup_directories(self):
        self.cache_directory = './.cache'
        self.log_directory = './logs'
        os.makedirs(self.cache_directory, exist_ok=True)
        os.makedirs(self.log_directory, exist_ok=True)
        
        self.output_dir = os.path.join(self.log_directory, self.refined_model_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def setup_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_id, trust_remote_code=True, cache_dir=self.cache_directory)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def load_base_model(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_id,
            quantization_config=quant_config,
            cache_dir=self.cache_directory,
        )
        model.config.use_cache = True
        model.config.pretraining_tp = 1
        return model

    def load_training_data(self, num_of_facts, data_path):
        training_data_list = load_json_data(data_path)
        training_data_list = training_data_list[:num_of_facts]
        transform_function = lambda item: item['full_sentence']
        return list_to_dataset(training_data_list, num_of_facts, transform_function), training_data_list

    def setup_dataloader(self):
        return torch.utils.data.DataLoader(self.training_data, batch_size=4, shuffle=True)

    def apply_lora_config(self):
        peft_parameters = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=self.args.rank,
            bias="none",
            task_type="CAUSAL_LM"
        )

        lora_model = get_peft_model(self.base_model, peft_parameters)
        lora_model.print_trainable_parameters()
        lora_model, self.train_dataloader = self.accelerator.prepare(lora_model, self.train_dataloader)
        return lora_model

    def setup_training_arguments(self):
        return TrainingArguments(
            output_dir=self.output_dir,
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
            logging_dir=os.path.join(self.output_dir, 'logs')
        )

    def setup_trainer(self):
        return SFTTrainer(
            model=self.lora_model,
            train_dataset=self.training_data,
            peft_config=LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=self.args.rank,
                bias="none",
                task_type="CAUSAL_LM"
            ),
            dataset_text_field="text",
            tokenizer=self.tokenizer,
            args=self.train_params
        )

    def train(self):
        self.accelerator.print("Starting training...")
        if self.ml_flow.get_model_version_by_tag(self.args.model_id):
            print('wrw')
            #  self.load_adapter()
        else:
            self.fine_tuning.train()

            adapter_save_path = os.path.join(self.cache_directory, self.refined_model_name)
            self.lora_model.save_pretrained(adapter_save_path)
            self.ml_flow.save_adapter(self.lora_model, self.args.model_id, self.tokenizer)
            self.evaluate()
        self.ml_flow.end_run()



    def evaluate(self):
        results = []
        detailed_results = []
        generator = pipeline(model=self.lora_model, tokenizer=self.tokenizer, task="text-generation")
        print('starting inference....')
        for i in self.training_data_list:
            prompted_question = i['natural_question']
            output = generator(prompted_question, max_new_tokens=10)

            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print(generated_text)
            correct = i['natural_answer'].lower() in generated_text.lower()
            results.append(1 if correct else 0)
            detailed_results.append({
                'question': i['question'],
                'answer': i['answer'],
                'model_answer': generated_text,
                'correct': correct
            })

        percentage_correct = sum(results) / len(results) * 100
        self.accelerator.print(f"Percentage of correct answers: {percentage_correct:.2f}%")

        metadata = {
            'base_model_name': self.args.model_id,
            'refined_model_name': self.refined_model_name,
            'r': self.args.rank,
            'num_epochs': self.train_params.num_train_epochs,
            'num_of_facts': self.args.num_of_facts,
            'num_correct_answers': sum(results),
            'percentage_correct': percentage_correct,
            'trainable_params': str(self.lora_model.get_nb_trainable_parameters())
        }

        metadata_file_path = os.path.join(self.output_dir, 'experiment_metadata.json')
        with open(metadata_file_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        detailed_results_file_path = os.path.join(self.output_dir, 'detailed_results.json')
        with open(detailed_results_file_path, 'w') as f:
            json.dump(detailed_results, f, indent=4)

        self.log_mlflow(percentage_correct, results, metadata, detailed_results)
        if self.args.wandb:
            self.log_wandb(percentage_correct, results, metadata, detailed_results)



    def log_wandb(self, percentage_correct, results, metadata, detailed_results):
        self.wandb_logger.log_metrics({
            "percentage_correct": percentage_correct,
            "num_correct_answers": sum(results),
            "metadata": metadata,
            "detailed_results": detailed_results
        })
        self.wandb_logger.finish()


def main():
    parser = argparse.ArgumentParser(description="Train a model with LoRA.")
    parser.add_argument("--rank", type=int, default=1, help="Rank of LoRA matrices.")
    parser.add_argument("--num_of_facts", type=int, default=100000, help="Number of facts to learn.")
    parser.add_argument("--model_id", type=str, required=True, help="The Hugging Face model ID.")
    parser.add_argument("--data_path", type=str, default='data/made_up_questions_v3.json', help="Path to the JSON dataset.")
    parser.add_argument("--build_data", action='store_true', help="Flag to build your data again")
    parser.add_argument("--wandb_project", type=str, default="lora-training", help="Weights and Biases project name.")
    parser.add_argument("--deepspeed_config", type=str, default="slurm/ds_config.json", help="deepspeed configuration file.")
    parser.add_argument("--wandb", type=bool, default=False, help="whether to use weights and biases to log.")
    parser.add_argument("--mlflow_experiment", type=str, default="lora-training-experiment", help="MLflow experiment name")

    args = parser.parse_args()

    trainer = ModelTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
