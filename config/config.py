import os

class Config:
    NCCL_SHM_DIR = './shared_mem'
    TOKENIZERS_PARALLELISM = 'false'
    MODEL_ID = 'your_model_id'
    MLFLOW_EXPERIMENT = 'your_mlflow_experiment'
