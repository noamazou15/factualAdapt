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
        #check if we have the wandb flag
        if self.args.wandb:
            wandb.init(project=self.args.wandb_project, config=self.args)
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
            })
        self.accelerator.print(f"Starting training for {num_of_facts} facts with rank {rank}...")
        chunks = [self.training_data_list[i:i + self.args.log_interval] for i in range(0, len(self.training_data_list), self.args.log_interval)] if self.args.log_interval else [self.training_data_list[:self.args.num_of_facts]]
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
                wandb.log({"chunk": idx + 1, "facts_counter": facts_counter})

        if self.args.wandb:
            wandb.finish()

        

    # def evaluate_chunk(self, chunk, lora_model, refined_model_name, facts_counter, rank):
    #     # refined_model_path = os.path.join(self.cache_directory, f"{refined_model_name}-num-of-facts={facts_counter}")
    #     # lora_model = load_adapter(self.base_model, refined_model_path)
    #     generator = pipeline(
    #         'text-generation',
    #         model=lora_model,
    #         tokenizer=self.tokenizer,
    #         max_new_tokens=10,
    #         max_length=100,
    #         return_tensors='pt',     # Return tensors for logits computation
    #         # return_token_type_ids=False,
    #     )

    #     results = []
    #     ranks = []
    #     for item in chunk:
    #         prompted_question = item['natural_question']
    #         natural_answer = item['natural_answer']
    #         output = generator(prompted_question)
    #         print(output)
    #         print("------------------------------")
    #         generated_text = output[0]['generated_text']
    #         # generated_tokens = output[0]['generated_tokens']
    #         logits = output[0]['logits']
    #         correct = item['natural_answer'].lower() in generated_text.lower()
    #         results.append(1 if correct else 0)
            
    #         natural_answer_tokens = self.tokenizer(natural_answer, return_tensors='pt')['input_ids'][0]
    #         probabilities = torch.softmax(logits, dim=-1)

    #         token_ranks = []
    #         for token in natural_answer_tokens:
    #             sorted_probs, sorted_indices = torch.sort(probabilities[0], descending=True)
    #             rank = (sorted_indices == token.item()).nonzero(as_tuple=True)[0].item()
    #             token_ranks.append(rank)
                               
    #         correct = natural_answer.lower() in generated_text.lower()
    #         results.append(1 if correct else 0)
        
    #     # Average rank of all tokens in natural_answer
    #     average_rank = sum(token_ranks) / len(token_ranks)
    #     ranks.append(average_rank)
            
    #     percentage_correct = sum(results) / len(results) * 100
    #     average_rank_over_all_answers = sum(ranks) / len(ranks)

    #     self.accelerator.print(f"Chunk evaluation - Percentage of correct answers: {percentage_correct:.2f}%")

    #     metadata = {
    #         'base_model_name': self.args.model_id,
    #         'refined_model_name': refined_model_name,
    #         'r': rank,
    #         'num_epochs': self.train_params.num_train_epochs,
    #         'num_of_facts': facts_counter,
    #         'num_correct_answers': sum(results),
    #         'percentage_correct': percentage_correct,
    #         'results_by_order': average_rank_over_all_answers,
    #         'averge_token_rank': average_rank
    #     }

    #     metadata_file_path = os.path.join(self.output_dir, f'experiment_metadata_{facts_counter}.json')
    #     with open(metadata_file_path, 'w') as f:
    #         json.dump(metadata, f, indent=4)

    #     self.ml_flow.log_mlflow(percentage_correct, results, metadata_file_path)


    def evaluate_chunk(self, chunk, lora_model, refined_model_name, facts_counter, rank):
        generator = pipeline(
            'text-generation',
            model=lora_model,
            tokenizer=self.tokenizer,
            max_new_tokens=10,
            max_length=100,
            return_tensors='pt', 
        )

        results = []
        ranks = []

        for item in chunk:
            prompted_question = item['natural_question']
            natural_answer = item['natural_answer']
            output = generator(prompted_question)
            print(output)
            # Extract generated token IDs and convert them to text
            generated_token_ids = output[0]['generated_token_ids']
            generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            print(f"this is the generated text {generated_text}")
            print(f"this is the natural answer {natural_answer}")
            print("------------------------------")
            # Check if the generated text matches the natural answer
            correct = natural_answer.lower() in generated_text.lower()
            results.append(1 if correct else 0)

            # Compute probabilities for each token
            if 'logits' not in output[0]:
            # Generate logits directly from the model
                inputs = self.tokenizer(prompted_question, return_tensors='pt')
                with torch.no_grad():
                    outputs = lora_model(**inputs, labels=inputs['input_ids'])
                logits = outputs.logits
            else:
                logits = output[0]['logits']
            probabilities = torch.softmax(logits, dim=-1)

            # Tokenize the natural_answer for ranking
            natural_answer_tokens = self.tokenizer(natural_answer, return_tensors='pt')['input_ids'][0]
            natural_answer_tokens = natural_answer_tokens.tolist()
            # Compute token ranks
            token_ranks = []
            for token in natural_answer_tokens:
                if token < len(probabilities[0]):
                    sorted_probs, sorted_indices = torch.sort(probabilities[0], descending=True)
                    token_rank = (sorted_indices == token).nonzero(as_tuple=True)
                    if len(token_rank[0]) > 0:
                        rank = token_rank[0].item() + 1  # +1 for 1-based rank
                        token_ranks.append(rank)
                    else:
                        token_ranks.append(len(probabilities[0]))  # Assign the maximum rank if not found
                else:
                    token_ranks.append(len(probabilities[0]))  # Assign the maximum rank if token is out of range
        
            
            # Compute average rank of all tokens in the natural_answer
            average_rank = sum(token_ranks) / len(token_ranks)
            ranks.append(average_rank)
        
        percentage_correct = sum(results) / len(results) * 100
        average_rank_over_all_answers = sum(ranks) / len(ranks) if ranks else float('inf')

        self.accelerator.print(f"Chunk evaluation - Percentage of correct answers: {percentage_correct:.2f}%")
        self.accelerator.print(f"Chunk evaluation - Average token rank: {average_rank_over_all_answers:.2f}")

        metadata = {
            'base_model_name': self.args.model_id,
            'refined_model_name': refined_model_name,
            'r': rank,
            'num_epochs': self.train_params.num_train_epochs,
            'num_of_facts': facts_counter,
            'num_correct_answers': sum(results),
            'percentage_correct': percentage_correct,
            'average_token_rank': average_rank_over_all_answers
        }

        metadata_file_path = os.path.join(self.output_dir, f'experiment_metadata_{facts_counter}.json')
        with open(metadata_file_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        if self.args.wandb:
            print({"percentage_correct": percentage_correct, "num_correct_answers": sum(results), "average_token_rank": average_rank_over_all_answers})
            wandb.log({"percentage_correct": percentage_correct, "num_correct_answers": sum(results), "average_token_rank": average_rank_over_all_answers})
            # wandb.log(metadata)
        # self.ml_flow.log_mlflow(percentage_correct, results, metadata_file_path)
    

    def run_experiments(self):
        self.train(self.args.num_of_facts, self.args.rank)
                

if __name__ == "__main__":
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
    parser.add_argument("--log_interval", type=str, default=None, help="once in how many facts log the model")
    parser.add_argument('--specific_layers', nargs='+', default=None, help='List of specific layers to apply LoRA to. If not provided, LoRA will be applied to all layers.')    
    args = parser.parse_args()

    trainer = ModelTrainer(args)
    trainer.run_experiments()
