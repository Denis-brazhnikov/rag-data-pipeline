import json
import shutil
import uuid
import tempfile
from pathlib import Path
from datetime import datetime
from src.rag_prep.models import PreparedChunk
from src.rag_prep.config import ChunkingPipelineConfig

class ChunkExportStage:
    def __init__(self, config: ChunkingPipelineConfig):
        self.config = config

    def run(self, chunks: list[PreparedChunk], metrics: dict) -> dict:
        print("💾 Транзакционное сохранение чанков...")
        
        output_dir = Path(self.config.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex

        data_list = [chunk.model_dump(mode='json') for chunk in chunks]
        
        manifest = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "counts": metrics,
            "config": self.config.model_dump(mode='json')
        }

        # ТРАНЗАКЦИЯ: Пишем всё во временную папку
        with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Пишем JSON
            with open(temp_path / "chunks.json", "w", encoding="utf-8") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
                
            # 2. Пишем JSONL
            with open(temp_path / "chunks.jsonl", "w", encoding="utf-8") as f:
                for item in data_list:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    
            # 3. Пишем Manifest
            with open(temp_path / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

            # 4. Если всё успешно, атомарно переносим файлы в целевую папку
            for file_name in ["chunks.json", "chunks.jsonl", "manifest.json"]:
                shutil.move(str(temp_path / file_name), str(output_dir / file_name))

        print(f"✅ Экспорт завершен. Данные сохранены в {output_dir}")
        return manifest