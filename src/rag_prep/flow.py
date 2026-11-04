from prefect import flow, task
from src.rag_prep.config import load_config, load_chunking_config
from src.rag_prep.pipeline import RagPreparationPipeline, RagChunkingPipeline

# --- ПАЙПЛАЙН ПОДГОТОВКИ ДАННЫХ ---
@task(name="run_rag_pipeline")
def run_pipeline_task(config_path: str):
    config = load_config(config_path)
    pipeline = RagPreparationPipeline(config)
    manifest = pipeline.run()
    return manifest

@flow(name="RAG-Data-Preparation-Flow")
def rag_data_preparation_flow(config_path: str = "config/default.yaml"):
    """Оркестрация пайплайна подготовки данных через Prefect (Слайд 111)"""
    manifest = run_pipeline_task(config_path)
    return manifest


# --- ПАЙПЛАЙН ЧАНКИНГА ---
@task(name="run_chunking_pipeline")
def run_chunking_task(config_path: str):
    config = load_chunking_config(config_path)
    pipeline = RagChunkingPipeline(config)
    manifest = pipeline.run()
    return manifest

@flow(name="RAG-Chunking-Flow")
def chunking_flow(config_path: str = "config/chunking.yaml"):
    """Оркестрация пайплайна чанкинга через Prefect"""
    manifest = run_chunking_task(config_path)
    return manifest