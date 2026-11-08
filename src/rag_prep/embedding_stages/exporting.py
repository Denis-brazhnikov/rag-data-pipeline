import json
import shutil
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from src.rag_prep.models import EmbeddedChunk
from src.rag_prep.config import EmbeddingsPipelineConfig

logger = logging.getLogger(__name__)

class EmbeddingExportStage:
    def __init__(self, config: EmbeddingsPipelineConfig):
        self.config = config

    def run(self, embeddings: list[EmbeddedChunk], metrics: dict, run_id: str) -> dict:
        print("💾 Транзакционное сохранение эмбеддингов...")
        output_dir = Path(self.config.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data_list = [e.model_dump(mode='json') for e in embeddings]
        manifest = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "config": self.config.model_dump(mode='json')
        }

        try:
            with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
                temp_path = Path(temp_dir)
                
                # 1. Снимок состояния (JSON)
                with open(temp_path / "embeddings.json", "w", encoding="utf-8") as f:
                    json.dump(data_list, f, ensure_ascii=False, indent=2)
                    
                # 2. Потоковый формат (JSONL) для Vector DB
                with open(temp_path / "embeddings.jsonl", "w", encoding="utf-8") as f:
                    for item in data_list:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        
                # 3. Манифест запуска
                with open(temp_path / "manifest.json", "w", encoding="utf-8") as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=2)

                # Атомарный перенос (только если всё записалось без ошибок)
                for file_name in ["embeddings.json", "embeddings.jsonl", "manifest.json"]:
                    shutil.move(str(temp_path / file_name), str(output_dir / file_name))
                    
            print(f"✅ Экспорт завершен. Файлы в {output_dir}")
            logger.info(f"Экспорт завершен. Файлы в {output_dir}")
            return manifest
            
        except Exception as e:
            logger.error(f"Критическая ошибка при экспорте данных: {e}")
            print(f"❌ Ошибка экспорта: {e}")
            raise e