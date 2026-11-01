import os
# Разрешаем использование файлового бэкенда (для MLflow 3.13+)
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
from src.rag_prep.config import PipelineConfig

class MLflowTracker:
    def __init__(self, config: PipelineConfig):
        self.config = config
        if self.config.logging.mlflow_enabled:
            mlflow.set_tracking_uri(self.config.logging.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.logging.mlflow_experiment)

    def log_run(self, counts: dict, manifest_path: str):
        if not self.config.logging.mlflow_enabled:
            return
            
        with mlflow.start_run(run_name=self.config.run.name):
            mlflow.log_params({
                "dedup_threshold": self.config.deduplication.threshold,
                "parser_strategy": self.config.parser.strategy,
                "min_chars": self.config.cleaning.min_chars,
                "seed": self.config.run.seed
            })
            
            for key, value in counts.items():
                mlflow.log_metric(key, value)
                
            mlflow.log_artifact(manifest_path)
            print("📊 Метрики успешно отправлены в MLflow!")