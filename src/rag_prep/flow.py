from prefect import flow, task
import uuid

from src.rag_prep.config import load_config, load_chunking_config, load_embeddings_config
from src.rag_prep.pipeline import RagPreparationPipeline
from src.rag_prep.tracking import MLflowTracker

# --- ИМПОРТЫ СТАДИЙ ЧАНКИНГА ---
from src.rag_prep.chunking_stages.loading import PreparedDocumentLoadingStage
from src.rag_prep.chunking_stages.splitting import ChunkSplittingStage
from src.rag_prep.chunking_stages.validation import ChunkValidationStage
from src.rag_prep.chunking_stages.exporting import ChunkExportStage

# --- ИМПОРТЫ СТАДИЙ ЭМБЕДДИНГОВ ---
from src.rag_prep.embedding_stages.loading import EmbeddingLoadingStage
from src.rag_prep.embedding_stages.embedding import TextEmbeddingStage
from src.rag_prep.embedding_stages.validation import EmbeddingValidationStage
from src.rag_prep.embedding_stages.metrics import EmbeddingMetricsStage
from src.rag_prep.embedding_stages.exporting import EmbeddingExportStage


# ==========================================
# --- ПАЙПЛАЙН ПОДГОТОВКИ ДАННЫХ ---
# ==========================================
@task(name="run_rag_pipeline")
def run_pipeline_task(config_path: str):
    config = load_config(config_path)
    pipeline = RagPreparationPipeline(config)
    return pipeline.run()

@flow(name="RAG-Data-Preparation-Flow")
def rag_data_preparation_flow(config_path: str = "config/default.yaml"):
    return run_pipeline_task(config_path)


# ==========================================
# --- ПАЙПЛАЙН ЧАНКИНГА ---
# ==========================================
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


# ==========================================
# --- ПАЙПЛАЙН ЭМБЕДДИНГОВ ---
# ==========================================
@task(name="load_chunks_for_embedding")
def load_chunks_task(config_path: str):
    config = load_embeddings_config(config_path)
    loader = EmbeddingLoadingStage(config)
    run_id = uuid.uuid4().hex
    return loader.run(), config, run_id

@task(name="calculate_embeddings")
def calculate_embeddings_task(chunks, config, run_id: str):
    embedder = TextEmbeddingStage(config, run_id)
    return embedder.run(chunks)

@task(name="validate_embeddings")
def validate_embeddings_task(embedded_chunks, config):
    validator = EmbeddingValidationStage(config)
    return validator.run(embedded_chunks)

@task(name="export_embeddings")
def export_embeddings_task(valid_embeddings, config, run_id: str):
    metrics_stage = EmbeddingMetricsStage()
    metrics = metrics_stage.run(valid_embeddings)
    
    exporter = EmbeddingExportStage(config)
    # Сохраняем возвращаемый манифест
    manifest = exporter.run(valid_embeddings, metrics, run_id)
    
    tracker = MLflowTracker(config)
    manifest_path = f"{config.paths.output_dir}/manifest.json"
    tracker.log_embedding_run(metrics, manifest_path)
    return manifest # Возвращаем манифест

@flow(name="RAG-Embeddings-Flow")
def embedding_flow(config_path: str = "config/embeddings.yaml"):
    """Оркестрация расчета эмбеддингов"""
    chunks, config, run_id = load_chunks_task(config_path)
    embedded_chunks = calculate_embeddings_task(chunks, config, run_id)
    valid_embeddings = validate_embeddings_task(embedded_chunks, config)
    return export_embeddings_task(valid_embeddings, config, run_id)