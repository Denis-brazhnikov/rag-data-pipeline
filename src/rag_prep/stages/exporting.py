import json
import os
import tempfile
from pathlib import Path
from datetime import datetime
from src.rag_prep.models import PreparedDocument
from src.rag_prep.config import PipelineConfig

class ExportStage:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def _atomic_write(self, path: Path, content: str):
        """Атомарная запись: пишем во временный файл, затем заменяем (Слайд 16)"""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=path.parent, text=True)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(content)
            os.replace(temp_path, path)
        except Exception as e:
            os.remove(temp_path)
            raise e

    def run(self, documents: list[PreparedDocument], counts: dict) -> dict:
        output_dir = Path(self.config.paths.output_dir)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_path = output_dir / "dataset.json"
        jsonl_path = output_dir / "dataset.jsonl"
        manifest_path = output_dir / "manifest.json"

        # 1. Подготовка данных
        data_list = [doc.model_dump(mode='json') for doc in documents]
        
        # 2. Запись JSON (снимок)
        json_content = json.dumps(data_list, ensure_ascii=False, indent=2)
        self._atomic_write(json_path, json_content)
        
        # 3. Запись JSONL (для стриминга)
        jsonl_lines = [json.dumps(item, ensure_ascii=False) for item in data_list]
        self._atomic_write(jsonl_path, "\n".join(jsonl_lines))
        
        # 4. Генерация Manifest (Слайд 131)
        manifest = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "config": self.config.model_dump(mode='json'),
            "counts": counts,
            "outputs": {
                "json": str(json_path),
                "jsonl": str(jsonl_path)
            }
        }
        self._atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))

        print(f"✅ Экспорт завершен. Файлы сохранены в {output_dir}")
        return manifest