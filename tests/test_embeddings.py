import pytest
import numpy as np
from datetime import datetime

from src.rag_prep.models import EmbeddedChunk, EmbeddingMetadata
from src.rag_prep.embedding_stages.validation import EmbeddingValidationStage
from src.rag_prep.config import EmbeddingsPipelineConfig, EmbeddingSettingsConfig, EmbeddingPathConfig, ChunkingRunConfig, ChunkingLoggingConfig

# Фикстура для конфига (чтобы не создавать YAML-файл для тестов)
@pytest.fixture
def mock_config():
    return EmbeddingsPipelineConfig(
        run=ChunkingRunConfig(name="test_run", seed=42),
        embedding=EmbeddingSettingsConfig(dimensions=384),
        paths=EmbeddingPathConfig(input_jsonl="dummy.jsonl", output_dir="dummy_dir"),
        logging=ChunkingLoggingConfig()
    )

def test_embedding_metadata_validation():
    """Проверяем, что модель EmbeddingMetadata корректно валидирует данные"""
    meta = EmbeddingMetadata(
        source="test.pdf",
        document_id="doc_123",
        model="intfloat/multilingual-e5-small",
        dimensions=384,
        embedding_hash="abc123hash",
        pipeline_run_id="run_999",
        created_at=datetime.now(),
        embedding_norm=1.0
    )
    assert meta.dimensions == 384
    assert meta.embedding_norm == 1.0
    assert meta.model == "intfloat/multilingual-e5-small"

def test_embedding_validation_stage_success(mock_config):
    """Проверяем, что валидатор пропускает правильные векторы"""
    validator = EmbeddingValidationStage(mock_config)
    
    # Создаем идеальный вектор (размерность 384, нормализованный)
    dummy_vector = [0.1] * 384
    meta = EmbeddingMetadata(
        source="test", document_id="doc1", model="test_model", dimensions=384,
        embedding_hash="hash1", pipeline_run_id="run1", created_at=datetime.now()
    )
    chunk = EmbeddedChunk(text="Тестовый текст", embedding=dummy_vector, metadata=meta)
    
    valid = validator.run([chunk])
    assert len(valid) == 1

def test_embedding_validation_stage_nan_failure(mock_config):
    """Проверяем, что валидатор отсеивает векторы с NaN"""
    validator = EmbeddingValidationStage(mock_config)
    
    # Создаем битый вектор с NaN
    bad_vector = [float('nan')] * 384
    meta = EmbeddingMetadata(
        source="test", document_id="doc1", model="test_model", dimensions=384,
        embedding_hash="hash_bad", pipeline_run_id="run1", created_at=datetime.now()
    )
    chunk = EmbeddedChunk(text="Битый текст", embedding=bad_vector, metadata=meta)
    
    valid = validator.run([chunk])
    # Валидатор должен выбросить этот чанк (вернуть пустой список)
    assert len(valid) == 0

def test_l2_normalization_logic():
    """Проверяем математику L2-нормализации"""
    vec = np.array([3.0, 4.0]) # Норма этого вектора = 5.0
    norm = np.linalg.norm(vec, keepdims=True)
    normalized_vec = vec / np.maximum(norm, 1e-10)
    
    # Новая норма должна быть равна 1.0
    new_norm = np.linalg.norm(normalized_vec)
    assert np.isclose(new_norm, 1.0)
    assert np.isclose(normalized_vec[0], 0.6) # 3/5
    assert np.isclose(normalized_vec[1], 0.8) # 4/5