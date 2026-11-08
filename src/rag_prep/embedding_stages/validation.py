import numpy as np
import logging
from src.rag_prep.models import EmbeddedChunk
from src.rag_prep.config import EmbeddingsPipelineConfig

logger = logging.getLogger(__name__)

class EmbeddingValidationStage:
    def __init__(self, config: EmbeddingsPipelineConfig):
        self.config = config

    def run(self, embeddings: list[EmbeddedChunk]) -> list[EmbeddedChunk]:
        print("🔍 Валидация векторов (размерность, NaN, Inf, метадата)...")
        if not embeddings:
            raise ValueError("Критическая ошибка: Список эмбеддингов пуст!")

        valid_embeddings = []
        
        # Счетчики ошибок (Слайд 38)
        empty_embeddings_count = 0
        dimension_mismatch_count = 0
        nan_embeddings_count = 0
        inf_embeddings_count = 0
        missing_metadata_count = 0

        expected_dim = self.config.embedding.dimensions

        for item in embeddings:
            if not item.embedding:
                empty_embeddings_count += 1
                continue
                
            vec = np.array(item.embedding)
            
            # 1. Проверка размерности (Слайд 39)
            if len(vec) != expected_dim:
                dimension_mismatch_count += 1
                logger.error(f"Неверная размерность для чанка {item.metadata.chunk_id}: ожидалось {expected_dim}, получено {len(vec)}")
                continue

            # 2. Проверка на NaN (Слайд 40)
            if np.isnan(vec).any():
                nan_embeddings_count += 1
                logger.warning(f"Вектор содержит NaN для {item.metadata.chunk_id}")
                continue
                
            # 3. Проверка на Infinity (Слайд 40)
            if np.isinf(vec).any():
                inf_embeddings_count += 1
                logger.warning(f"Вектор содержит Infinity для {item.metadata.chunk_id}")
                continue

            # 4. Проверка метадаты
            if not item.metadata.document_id or not item.metadata.model:
                missing_metadata_count += 1
                logger.warning(f"Отсутствует метадата для {item.metadata.chunk_id}")
                continue
                
            valid_embeddings.append(item)

        # Формируем отчет
        errors = dimension_mismatch_count + nan_embeddings_count + inf_embeddings_count + empty_embeddings_count
        if errors > 0:
            print(f"⚠️ Отброшено {errors} битых векторов.")
            
        print(f"✅ Валидация пройдена. Валидных векторов: {len(valid_embeddings)} из {len(embeddings)}. Размерность: {expected_dim}")
        
        return valid_embeddings