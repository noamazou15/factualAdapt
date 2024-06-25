import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


class MlFlowWrapper:

    def __init__(self,experiment, base_model_name, refined_model_name, **kwargs):
        self.base_model_name = base_model_name
        self.refined_model_name = refined_model_name
        self.client = MlflowClient()
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=refined_model_name, tags=kwargs)
        

    def get_model_version_by_tag(self, model_name, **kwargs):
        model_name = model_name.split("/")[-1]
        try:
            versions = self.client.get_registered_model(model_name).latest_versions
        except mlflow.exceptions.MlflowException:
            return None
        for version in versions:
            tags = version.tags
            correct_tags = True
            for tag_key, tag_value in kwargs.items():
                if tags.get(tag_key) != tag_value:
                    correct_tags = False
            if correct_tags:
                return version
        return None

    def end_run(self):
        mlflow.end_run()

    def check_existing_model(self):
        model_name = self.refined_model_name
        client = self.client
        try:
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            if latest_versions:
                print(f"Model {model_name} exists. Loading adapter.")
                return True
            else:
                print(f"Model {model_name} does not exist. Training new adapter.")
                return False
        except mlflow.exceptions.MlflowException:
            print(f"Model {model_name} does not exist. Training new adapter.")
            return False

    # def load_adapter(self):
    #     model_uri = f"models:/{self.refined_model_name}/latest"
    #     adapter_path = mlflow.pytorch.load_model(model_uri)
    #     adapter_state_dict = torch.load(adapter_path)
    #     set_peft_model_state_dict(self.lora_model, adapter_state_dict)

    def log_mlflow_params(self):
        mlflow.log_params({
            "model_id": self.args.model_id,
            "rank": self.args.rank,
            "num_of_facts": self.args.num_of_facts,
            "num_epochs": self.train_params.num_train_epochs,
            "learning_rate": self.train_params.learning_rate,
            "weight_decay": self.train_params.weight_decay
        })

    def save_adapter(self, lora_model, base_model_name, tokenizer, **kwargs):
        base_model_name = base_model_name.split("/")[-1]
        mlflow.transformers.log_model(
        transformers_model={"model": lora_model, "tokenizer": tokenizer},
        registered_model_name = base_model_name,
        tags = kwargs,
        artifact_path = "adapter"
        )
        for key,value in kwargs.items():
            self.client.set_registered_model_tag(base_model_name, key, value)

        
    # def log_mlflow(self, percentage_correct, results, metadata, detailed_results):
    #     mlflow.log_metrics({
    #         "percentage_correct": percentage_correct,
    #         "num_correct_answers": sum(results)
    #     })
    #     mlflow.log_artifact(os.path.join(self.output_dir, 'experiment_metadata.json'))
    #     mlflow.log_artifact(os.path.join(self.output_dir, 'detailed_results.json'))


