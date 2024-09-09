import argparse
import os
import json
import torch
from accelerate import Accelerator
from ml_flow import MlFlowWrapper
from utils.file_utils import setup_directories
from utils.model_utils import setup_tokenizer, load_base_model, load_adapter
from utils.training_utils import setup_training_arguments, setup_trainer, get_lora_target_modules
from data.data_loader import load_json_data, list_to_tokenized_dataset
from config.config import Config
from transformers import pipeline
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = Config.TOKENIZERS_PARALLELISM

class ModelTrainer:
    def __init__(self, args):
        self.accelerator = Accelerator()
        self.args = args
        self.cache_directory, self.log_directory, self.output_dir = setup_directories(self.args.model_id.split('/')[-1])
        layers_str = '_'.join(self.args.layer_modules) if self.args.layer_modules else 'all'
        layers_num_str = '_'.join(map(str, self.args.layers_to_transform)) if self.args.layers_to_transform else 'all'

        refined_model_name = f"{self.args.model_id.split('/')[-1]}-real-facts-r={self.args.rank}-num-of-facts={self.args.num_of_facts}-layers={layers_str}--layers_num={layers_num_str}"
        #check if we have the wandb flag
        if self.args.wandb:
            wandb.init(project=self.args.wandb_project, config=self.args, name=refined_model_name)
        self.tokenizer = setup_tokenizer(self.args.model_id, self.cache_directory)
        self.base_model = load_base_model(self.args.model_id, self.cache_directory)
        self.training_data_list = load_json_data(self.args.data_path)
        self.train_params = setup_training_arguments(self.output_dir)
        
        

    def apply_lora_config(self, rank):
        from peft import LoraConfig, get_peft_model


        peft_parameters = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=rank,
            bias="none",
            task_type="CAUSAL_LM",
        )



        lora_model = get_peft_model(self.base_model, peft_parameters)
        lora_model.print_trainable_parameters()
        return lora_model, peft_parameters

    def train(self, num_of_facts, rank):
        refined_model_name = f"{self.args.model_id.split('/')[-1]}-made-up-facts-r={rank}"
        #achi and tal, need to check, verify all facts from start not the delta
        if self.args.no_model_memory:
            self.base_model = load_base_model(self.args.model_id, self.cache_directory)
        if self.args.wandb:
            wandb.config.update({
                "learning_rate": self.train_params.learning_rate,
                "epochs": self.train_params.num_train_epochs,
                "batch_size": self.train_params.per_device_train_batch_size,
                'base_model_name': self.args.model_id,
                'refined_model_name': refined_model_name,
            })
        self.accelerator.print(f"Starting training for {num_of_facts} facts with rank {rank}...")
        
        chunks = [self.training_data_list[i:i + self.args.log_interval] for i in range(0, self.args.num_of_facts, self.args.log_interval)] if self.args.log_interval else [self.training_data_list[:self.args.num_of_facts]]
        facts_counter = 0

        
        lora_model, peft_parameters = self.apply_lora_config(rank)
        trainer = None
        for idx, chunk in enumerate(chunks):
            self.accelerator.print(f"Training on chunk {idx + 1}/{len(chunks)}...")
            facts_counter += len(chunk)
            training_data = list_to_tokenized_dataset(chunk, len(chunk), self.tokenizer)



            if trainer is None:
                trainer = setup_trainer(self.base_model, training_data, self.tokenizer, self.train_params, peft_parameters)
            else:
                trainer.train_dataset = training_data
            trainer.train()

            adapter_save_path = os.path.join(self.cache_directory, f"{refined_model_name}-num-of-facts={facts_counter}")
            trainer.model.save_pretrained(adapter_save_path, 'default')
            
            # self.ml_flow = MlFlowWrapper(self.args.mlflow_experiment, self.args.model_id, 
            #                         ['rank', 'num_of_facts', 'model_id', 'specific_layers'], 
            #                         **{'num_of_facts':num_of_facts, 'rank':rank, 'model_id':self.args.model_id})
            # self.ml_flow.start_run(refined_model_name)
            
            # Log results for this chunk
            self.evaluate_chunk(self.training_data_list[:facts_counter], lora_model, refined_model_name, facts_counter, rank)
            # self.ml_flow.end_run()


        if self.args.wandb:
            wandb.finish()


    def evaluate_chunk(self, chunk, lora_model, refined_model_name, facts_counter, rank):
        results = []
        ranks = []
        correct_answers_indexs = []

        for idx,item in enumerate(chunk):
            prompted_question = item['natural_question']
            natural_answer = item['natural_answer']

            inputs = self.tokenizer(prompted_question, return_tensors='pt')
            with torch.no_grad():
                outputs = lora_model(**inputs, labels=inputs['input_ids'])
            logits = outputs.logits

            probs = torch.nn.functional.softmax(logits, dim=-1)
            input_ids = inputs['input_ids']
            predicted_token_ids = torch.argmax(probs, dim=-1)
 
            generated_text = self.tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
            # Check if the generated text matches the natural answer
            correct = natural_answer.lower() in generated_text.lower()
            if correct:
                results.append(1)
                ranks.append(1)
                correct_answers_indexs.append(idx)
            else:
                results.append(0)
                token_ranks = []
                for idx, token_id in enumerate(input_ids[0]):
                    token_probs = probs[0, idx]  # Get probabilities for the current token position
                    token_rank = torch.argsort(token_probs, descending=True)  # Sort tokens by probability
                    rank_idx = (token_rank == token_id).nonzero(as_tuple=True)[0].item() + 1  # Get the rank of the correct token
                    token_ranks.append(rank_idx)

            
            # Compute average rank of all tokens in the natural_answer
            average_rank = sum(token_ranks) / len(token_ranks)
            ranks.append(average_rank)
        
        percentage_correct = sum(results) / len(results) * 100
        average_rank_over_all_answers = sum(ranks) / len(ranks) if ranks else float('inf')

        self.accelerator.print(f"Chunk evaluation - Percentage of correct answers: {percentage_correct:.2f}%")
        self.accelerator.print(f"Chunk evaluation - Average token rank: {average_rank_over_all_answers:.2f}")

        run_data = {
            'r': rank,
            'num_epochs': self.train_params.num_train_epochs,
            'num_of_facts': facts_counter,
            'num_correct_answers': sum(results),
            'percentage_correct': percentage_correct,
            'average_token_rank': average_rank_over_all_answers
        }

        metadata_file_path = os.path.join(self.output_dir, f'experiment_metadata_{facts_counter}.json')
        with open(metadata_file_path, 'w') as f:
            json.dump(run_data, f, indent=4)

        if self.args.wandb:
            print({"percentage_correct": percentage_correct, "num_correct_answers": sum(results), "average_token_rank": average_rank_over_all_answers})
            wandb.log({"percentage_correct": percentage_correct, "num_correct_answers": sum(results), "average_token_rank": average_rank_over_all_answers})
            wandb.log(run_data)
        # self.ml_flow.log_mlflow(percentage_correct, results, metadata_file_path)
    

    def run_experiments(self):
        self.train(self.args.num_of_facts, self.args.rank)
                

if __name__ == "__main__":

    def parse_layers(layer):
        return int(layer)
    
    # Add your argument parsing here
    parser = argparse.ArgumentParser(description="Train a model with LoRA.")
    parser.add_argument("--rank", type=int, default=1, help="Rank of LoRA matrices.")
    parser.add_argument("--num_of_facts", type=int, default=100000, help="Number of facts to learn.")
    parser.add_argument("--model_id", type=str, required=True, help="The Hugging Face model ID.")
    parser.add_argument("--data_path", type=str, default='data/made_up_questions_v3.json', help="Path to the JSON dataset.")
    parser.add_argument("--build_data", action='store_true', help="Flag to build your data again")
    parser.add_argument("--wandb_project", type=str, default="lora-training", help="Weights and Biases project name.")
    parser.add_argument("--deepspeed_config", type=str, default="slurm/ds_config.json", help="deepspeed configuration file.")
    # parser.add_argument("--wandb", type=bool, default=False, help="whether to use weights and biases to log.")
    parser.add_argument("--wandb", action='store_true', help="whether to use weights and biases to log.")
    parser.add_argument("--no_model_memory", action='store_true', help="whether to load the model from scratch each run")
    parser.add_argument("--mlflow_experiment", type=str, default="lora-training-experiment", help="MLflow experiment name")
    parser.add_argument("--log_interval", type=int, default=None, help="once in how many facts log the model")
    parser.add_argument('--layer_modules', nargs='+', default=None, help='List of specific layers to apply LoRA to. If not provided, LoRA will be applied to all layers.')    
    parser.add_argument('--layers_to_transform', nargs='+', type=parse_layers, default=None, help='List of specific layers to apply LoRA to. If not provided, LoRA will be applied to all layers.')
    args = parser.parse_args()

    trainer = ModelTrainer(args)
    trainer.run_experiments()
