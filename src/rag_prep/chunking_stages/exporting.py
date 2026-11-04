import json
import os
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
from src.rag_prep.models import PreparedChunk
from src.rag_prep.config import ChunkingPipelineConfig

class ChunkExportStage:
    def __init__(self, config: ChunkingPipelineConfig):
        self.config = config

    def _atomic_write(self, path: Path, content: str):
        """Атомарная запись во избежание повреждения файлов при сбоях"""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=path.parent, text=True)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(content)
            os.replace(temp_path, path)
        except Exception as e:
            os.remove(temp_path)
            raise e

    def run(self, chunks: list[PreparedChunk], metrics: dict) -> dict:
        print("💾 Сохранение чанков и манифеста...")
        
        output_dir = Path(self.config.paths.output_dir)
        run_id = uuid.uuid4().hex
        
        json_path = output_dir / "chunks.json"
        jsonl_path = output_dir / "chunks.jsonl"
        manifest_path = output_dir / "manifest.json"

        # Сериализуем объекты Pydantic в словари
        data_list = [chunk.model_dump(mode='json') for chunk in chunks]
        
        # 1. Запись в JSON
        json_content = json.dumps(data_list, ensure_ascii=False, indent=2)
        self._atomic_write(json_path, json_content)
        
        # 2. Запись в JSONL
        jsonl_lines = [json.dumps(item, ensure_ascii=False) for item in data_list]
        self._atomic_write(jsonl_path, "\n".join(jsonl_lines))
        
        # 3. Формирование Manifest
        manifest = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "counts": metrics,
            "config": self.config.model_dump(mode='json')
        }
        self._atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))

        print(f"✅ Экспорт завершен. Данные сохранены в {output_dir}")
        return manifest