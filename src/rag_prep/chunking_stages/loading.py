import json
from pathlib import Path
from src.rag_prep.models import PreparedDocument
from src.rag_prep.config import ChunkingPipelineConfig

class PreparedDocumentLoadingStage:
    def __init__(self, config: ChunkingPipelineConfig):
        self.config = config

    def run(self) -> list[PreparedDocument]:
        input_path = Path(self.config.paths.input_jsonl)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Файл не найден: {input_path}")

        documents = []
        print(f"📂 Загрузка подготовленных документов из {input_path}...")
        
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    # Валидируем через Pydantic
                    doc = PreparedDocument(**data)
                    documents.append(doc)
                except Exception as e:
                    print(f"⚠️ Ошибка валидации JSON на строке {line_num}: {e}")
                    # По ТЗ нужно корректно обрабатывать ошибки, но пока просто логируем и идем дальше
                    continue
                    
        print(f"✅ Успешно загружено {len(documents)} документов.")
        return documents