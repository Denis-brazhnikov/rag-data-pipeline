from prefect import flow, task
from src.rag_prep.config import load_config
from src.rag_prep.pipeline import RagPreparationPipeline

@task(name="run_rag_pipeline")
def run_pipeline_task(config_path: str):
    config = load_config(config_path)
    pipeline = RagPreparationPipeline(config)
    manifest = pipeline.run()
    return manifest

@flow(name="RAG-Data-Preparation-Flow")
def rag_data_preparation_flow(config_path: str = "config/default.yaml"):
    """Оркестрация пайплайна через Prefect (Слайд 111)"""
    manifest = run_pipeline_task(config_path)
    return manifest