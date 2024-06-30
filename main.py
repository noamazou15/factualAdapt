import argparse
import os
import json

from accelerate import Accelerator
from ml_flow import MlFlowWrapper

from utils.file_utils import setup_directories
from utils.model_utils import setup_tokenizer, load_base_model, load_adapter
from utils.training_utils import setup_training_arguments, setup_trainer
from data.data_loader import load_json_data, list_to_tokenized_dataset
from config.config import Config

os.environ["TOKENIZERS_PARALLELISM"] = Config.TOKENIZERS_PARALLELISM

class ModelTrainer:
    def __init__(self, args):
        self.args = args.__dict__
        self.accelerator = Accelerator()

        self.refined_model_name = f"{self.args.model_id.split('/')[-1]}-made-up-facts-r={self.args.rank}"
        self.cache_directory, self.log_directory, self.output_dir = setup_directories(self.refined_model_name)
        
        self.tokenizer = setup_tokenizer(self.args.model_id, self.cache_directory)
        self.base_model = load_base_model(self.args.model_id, self.cache_directory)
        self.training_data_list = load_json_data(self.args.data_path)[:self.args.num_of_facts]
        self.training_data = None
        self.train_params = setup_training_arguments(self.output_dir)
        

        self.ml_flow = MlFlowWrapper(self.args.mlflow_experiment, self.args.model_id, self.refined_model_name, self.args)

    def apply_lora_config(self):
        from peft import LoraConfig, get_peft_model

        peft_parameters = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=self.args.rank,
            bias="none",
            task_type="CAUSAL_LM"
        )

        lora_model = get_peft_model(self.base_model, peft_parameters)
        lora_model.print_trainable_parameters()
        lora_model = self.accelerator.prepare(lora_model)
        return lora_model

    def setup_dataloader(self):
        return torch.utils.data.DataLoader(self.training_data, batch_size=4, shuffle=True)

    def train(self):
        self.accelerator.print("Starting training...")
        chunks = [self.training_data_list[i:i + 25] for i in range(0, len(self.training_data_list), 25)]
        facts_counter = 0

        for idx, chunk in enumerate(chunks):
            self.accelerator.print(f"Training on chunk {idx + 1}/{len(chunks)}...")
            facts_counter += 25
            self.training_data = list_to_tokenized_dataset(chunk, len(chunk), self.tokenizer)
            self.train_dataloader = self.setup_dataloader()

            trainer = setup_trainer(self.base_model, self.training_data, self.tokenizer, self.train_params, self.args.rank)
            trainer.train()

            adapter_save_path = os.path.join(self.cache_directory, f"{self.refined_model_name}-num-of-facts={facts_counter}")
            trainer.model.save_pretrained(adapter_save_path, 'default')
            
        self.ml_flow.end_run()


    def evaluate(self):
        from transformers import pipeline

        results = []
        refined_model_path = os.path.join(self.cache_directory, f"{self.refined_model_name}-num-of-facts={self.args.num_of_facts}")
        self.lora_model = load_adapter(self.args.model_id, refined_model_path, self.cache_directory)
        
        generator = pipeline(
            'text-generation',
            model=self.lora_model,
            tokenizer=self.tokenizer,
            max_new_tokens=10,
            max_length=100
        )

        for item in self.training_data_list:
            prompted_question = item['natural_question']
            output = generator(prompted_question)[0]['generated_text']
            correct = item['natural_answer'].lower() in output.lower()
            results.append(1 if correct else 0)

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
        }

        metadata_file_path = os.path.join(self.output_dir, 'experiment_metadata.json')
        with open(metadata_file_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        self.ml_flow.log_mlflow(percentage_correct, results, metadata_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--num_of_facts", type=int, required=True)
    parser.add_argument("--mlflow_experiment", type=str, required=True)

    args = parser.parse_args()

    trainer = ModelTrainer(args)
    trainer.train()
    trainer.evaluate()
