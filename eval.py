import argparse
from run_v3 import ModelTrainer

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
    parser.add_argument("--log_interval", type=str, default="25", help="once in how many facts log the model")

    args = parser.parse_args()

    trainer = ModelTrainer(args)
    trainer.evaluate('pythia-14m')


if __name__ == "__main__":
    main()