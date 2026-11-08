import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
from src.rag_prep.config import PipelineConfig

class MLflowTracker:
    def __init__(self, config):
        self.config = config
        if self.config.logging.mlflow_enabled:
            mlflow.set_tracking_uri(self.config.logging.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.logging.mlflow_experiment)

    def log_run(self, counts: dict, manifest_path: str):
        """Логирование для пайплайна подготовки данных"""
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
            print("📊 Метрики подготовки данных отправлены в MLflow!")

    def log_chunking_run(self, metrics: dict, manifest_path: str):
        """Логирование для пайплайна чанкинга"""
        if not self.config.logging.mlflow_enabled:
            return
            
        with mlflow.start_run(run_name=self.config.run.name):
            mlflow.log_params({
                "chunk_size": self.config.chunking.chunk_size,
                "chunk_overlap": self.config.chunking.chunk_overlap,
                "chunking_strategy": self.config.chunking.strategy,
                "tokenizer_model": self.config.chunking.tokenizer_model,
                "seed": self.config.run.seed
            })
            
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
                
            mlflow.log_artifact(manifest_path)
            print("📊 Метрики чанкинга отправлены в MLflow!")

    def log_embedding_run(self, metrics: dict, manifest_path: str):
        """Логирование для пайплайна эмбеддингов"""
        if not self.config.logging.mlflow_enabled:
            return
            
        with mlflow.start_run(run_name=self.config.run.name):
            # Добавлены новые параметры из конфига (размерность, лимиты)
            mlflow.log_params({
                "embedding_model": self.config.embedding.model,
                "dimensions": self.config.embedding.dimensions,
                "batch_size": self.config.embedding.batch_size,
                "max_input_tokens": self.config.embedding.max_input_tokens,
                "max_batch_tokens": self.config.embedding.max_batch_tokens,
                "l2_normalize": self.config.embedding.l2_normalize,
                "seed": self.config.run.seed
            })
            
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
                
            mlflow.log_artifact(manifest_path)
            print("📊 Метрики эмбеддингов отправлены в MLflow!")