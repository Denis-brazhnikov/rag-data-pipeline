from prefect import flow, task
from src.rag_prep.config import load_config, load_chunking_config
from src.rag_prep.pipeline import RagPreparationPipeline
from src.rag_prep.chunking_stages.loading import PreparedDocumentLoadingStage
from src.rag_prep.chunking_stages.splitting import ChunkSplittingStage
from src.rag_prep.chunking_stages.validation import ChunkValidationStage
from src.rag_prep.chunking_stages.exporting import ChunkExportStage
from src.rag_prep.tracking import MLflowTracker

# --- ПАЙПЛАЙН ПОДГОТОВКИ ДАННЫХ ---
@task(name="run_rag_pipeline")
def run_pipeline_task(config_path: str):
    config = load_config(config_path)
    pipeline = RagPreparationPipeline(config)
    return pipeline.run()

@flow(name="RAG-Data-Preparation-Flow")
def rag_data_preparation_flow(config_path: str = "config/default.yaml"):
    return run_pipeline_task(config_path)


# --- ПАЙПЛАЙН ЧАНКИНГА (ГРАНУЛЯРНЫЙ) ---
@task(name="load_documents")
def load_documents_task(config_path: str):
    config = load_chunking_config(config_path)
    loader = PreparedDocumentLoadingStage(config)
    return loader.run(), config

@task(name="split_chunks")
def split_chunks_task(documents, config):
    splitter = ChunkSplittingStage(config)
    return splitter.run(documents)

@task(name="validate_chunks")
def validate_chunks_task(chunks, config):
    validator = ChunkValidationStage(config)
    return validator.run(chunks)

@task(name="export_chunks")
def export_chunks_task(valid_chunks, metrics, config):
    exporter = ChunkExportStage(config)
    manifest = exporter.run(valid_chunks, metrics)
    
    tracker = MLflowTracker(config)
    manifest_path = f"{config.paths.output_dir}/manifest.json"
    tracker.log_chunking_run(metrics, manifest_path)
    return manifest

@flow(name="RAG-Chunking-Flow")
def chunking_flow(config_path: str = "config/chunking.yaml"):
    """Оркестрация чанкинга с видимостью каждой стадии в Prefect"""
    documents, config = load_documents_task(config_path)
    chunks = split_chunks_task(documents, config)
    valid_chunks, metrics = validate_chunks_task(chunks, config)
    manifest = export_chunks_task(valid_chunks, metrics, config)
    return manifest