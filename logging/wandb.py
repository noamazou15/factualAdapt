import wandb

class WandbLogger:
    def __init__(self, project_name, run_name):
        self.project_name = project_name
        self.run_name = run_name
        self.run = None

    def init(self):
        self.run = wandb.init(project=self.project_name, name=self.run_name)

    def log_config(self, config):
        wandb.config.update(config)

    def log_metrics(self, metrics):
        wandb.log(metrics)

    def finish(self):
        if self.run:
            self.run.finish()
