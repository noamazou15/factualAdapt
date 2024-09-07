import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


class MlFlowWrapper:

    def __init__(self, experiment, base_model_name, refined_model_name, params_to_log = ['rank', 'num_of_facts','model_id'], **kwargs):
        self.base_model_name = base_model_name
        self.refined_model_name = refined_model_name
        self.tags = kwargs
        self.tags = {key: kwargs.get(key) for key in params_to_log }
        self.client = MlflowClient()
        mlflow.set_experiment(experiment)
    
    def start_run(self, refined_model_name):
        print(self.tags)
        mlflow.start_run(run_name=refined_model_name, tags=self.tags)
        

    def get_most_trained_model_with_same_params(self, model_name, **kwargs):
        model_name = model_name.split("/")[-1]
        target_num_of_facts = int(kwargs.pop("num_of_facts"))

        if not target_num_of_facts:
            return None 

        try:
            versions = self.client.get_registered_model(model_name).latest_versions
        except mlflow.exceptions.MlflowException:
            return None

        closest_version = None
        closest_num_of_facts = float('inf')
        min_difference = float('inf')

        for version in versions:
            tags = version.tags
            model_num_of_facts = int(tags.pop("num_of_facts", 0))
            correct_tags = all(tags.get(key) == str(value) for key, value in kwargs.items())   

            if correct_tags:
                
                difference = abs(model_num_of_facts - target_num_of_facts)

                if difference < min_difference:
                    min_difference = difference
                    closest_version = version
                    closest_num_of_facts = model_num_of_facts

        if closest_version:
            model_uri = f"models:/{model_name}/{closest_version.version}"
            model_with_tokenizer = mlflow.pyfunc.load_model(model_uri)
        return model_with_tokenizer.model, closest_num_of_facts

    def end_run(self):
        mlflow.end_run()


    def check_existing_model(self, model_name, **kwargs):
        model_name = model_name.split("/")[-1]
        target_num_of_facts = int(kwargs.pop("num_of_facts"))

        if not target_num_of_facts:
            return None 

        try:
            versions = self.client.get_registered_model(model_name).latest_versions
        except mlflow.exceptions.MlflowException:
            return None
        
        for version in versions:
            tags = version.tags
            correct_tags = all(tags.get(key) == str(value) for key, value in kwargs.items())   

            if correct_tags:
                closest_version = version

        if closest_version:
            model_uri = f"models:/{model_name}/{closest_version.version}"
            model_with_tokenizer = mlflow.transformers.load_model(model_uri)
        return model_with_tokenizer

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
        logged_obj = mlflow.transformers.log_model(
        transformers_model={"model": lora_model, "tokenizer": tokenizer},
        registered_model_name = base_model_name,
        tags = kwargs,
        artifact_path = "adapter"
        )
        for key,value in kwargs.items():
            self.client.set_model_version_tag(base_model_name, logged_obj.registered_model_version, key, value)

        
    def log_mlflow(self, percentage_correct, results, metadata_path, detailed_results_path=None):
        mlflow.log_metrics({
            "percentage_correct": percentage_correct,
            "num_correct_answers": sum(results)
        })

        mlflow.log_artifact(metadata_path)
        if detailed_results_path:
            mlflow.log_artifact(detailed_results_path)


