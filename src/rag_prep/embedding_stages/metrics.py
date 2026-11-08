import numpy as np
import logging
from src.rag_prep.models import EmbeddedChunk

logger = logging.getLogger(__name__)

class EmbeddingMetricsStage:
    def run(self, embeddings: list[EmbeddedChunk]) -> dict:
        print("📊 Сбор метрик эмбеддингов...")
        
        if not embeddings:
            logger.warning("Список эмбеддингов пуст. Возврат нулевых метрик.")
            return {
                "total_chunks": 0,
                "avg_text_length": 0.0,
                "avg_vector_norm": 0.0,
                "min_vector_norm": 0.0,
                "max_vector_norm": 0.0,
                "embedding_dim": 0,
                "unique_embeddings": 0
            }

        vectors = [np.array(e.embedding) for e in embeddings]
        norms = [np.linalg.norm(v) for v in vectors]
        text_lengths = [len(e.text) for e in embeddings]
        
        # Уникальность хэшей (проверка, не задублировались ли векторы)
        unique_hashes = set(e.metadata.embedding_hash for e in embeddings)

        metrics = {
            "total_chunks": len(embeddings),
            "avg_text_length": round(sum(text_lengths) / len(text_lengths), 2),
            "avg_vector_norm": round(float(np.mean(norms)), 4),
            "min_vector_norm": round(float(np.min(norms)), 4),
            "max_vector_norm": round(float(np.max(norms)), 4),
            "embedding_dim": len(vectors[0]),
            "unique_embeddings": len(unique_hashes)
        }
        
        # Диагностика: если все векторы одинаковые (что-то пошло не так)
        if metrics["unique_embeddings"] == 1 and metrics["total_chunks"] > 1:
            logger.warning("⚠️ Все эмбеддинги имеют одинаковый хэш! Возможна ошибка модели.")

        print(f"✅ Метрики собраны: {metrics}")
        return metrics