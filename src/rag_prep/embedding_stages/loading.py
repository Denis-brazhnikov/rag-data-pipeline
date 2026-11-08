import json
import logging
from pathlib import Path
from src.rag_prep.models import PreparedChunk
from src.rag_prep.config import EmbeddingsPipelineConfig

logger = logging.getLogger(__name__)

class EmbeddingLoadingStage:
    def __init__(self, config: EmbeddingsPipelineConfig):
        self.config = config

    def run(self) -> list[PreparedChunk]:
        path = Path(self.config.paths.input_jsonl)
        
        if not path.exists():
            raise FileNotFoundError(f"Критическая ошибка: Файл чанков не найден по пути {path}")

        chunks = []
        logger.info(f"📂 Загрузка чанков из {path}...")
        print(f"📂 Загрузка чанков из {path}...")
        
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    # Строгая валидация через Pydantic v2
                    chunk = PreparedChunk.model_validate(data)
                    
                    # Дополнительные проверки (Слайд 21: наличие текста и идентификаторов)
                    if not chunk.text.strip():
                        logger.warning(f"Пропуск чанка на строке {line_num}: пустой текст.")
                        continue
                    if not chunk.metadata.document_id:
                        logger.warning(f"Пропуск чанка на строке {line_num}: отсутствует document_id.")
                        continue
                        
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    logger.error(f"Ошибка парсинга JSON на строке {line_num}: {e}")
                    print(f"⚠️ Ошибка парсинга JSON на строке {line_num}: {e}")
                except Exception as e:
                    logger.error(f"Ошибка валидации Pydantic на строке {line_num}: {e}")
                    print(f"⚠️ Ошибка валидации Pydantic на строке {line_num}: {e}")

        print(f"✅ Успешно загружено {len(chunks)} чанков.")
        logger.info(f"✅ Успешно загружено {len(chunks)} чанков.")
        return chunks