import hashlib
from pathlib import Path
from datetime import datetime
from llama_index.core import SimpleDirectoryReader

from src.rag_prep.models import SourceFile
from src.rag_prep.config import LoaderConfig

class LlamaIndexLoadingStage:
    def __init__(self, config: LoaderConfig):
        self.config = config

    def run(self, input_dir: str) -> list[SourceFile]:
        sources = []
        input_path = Path(input_dir)
        
        try:
            # Используем SimpleDirectoryReader из LlamaIndex для сканирования
            reader = SimpleDirectoryReader(
                input_dir=str(input_path),
                recursive=self.config.recursive,
                required_exts=self.config.allowed_extensions,
                exclude_hidden=self.config.exclude_hidden,
                num_files_limit=self.config.num_files_limit
            )
            # Получаем список путей к файлам
            files = [Path(path) for path in reader.input_files]
        except ValueError as exc:
            # Обработка ошибки "No files found" (слайд 27)
            if 'No files found' not in str(exc):
                raise
            files = []

        for file_path in files:
            # Вычисляем SHA-256 хеш содержимого (content hashing) (слайд 23)
            with open(file_path, "rb") as f:
                content = f.read()
                source_hash = hashlib.sha256(content).hexdigest()
            
            stat = file_path.stat()
            
            # Создаем типизированный объект SourceFile
            source_file = SourceFile(
                path=file_path,
                source=str(file_path.resolve()),
                file_name=file_path.name,
                file_type=file_path.suffix.lower(),
                source_hash=source_hash,
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime)
            )
            sources.append(source_file)
            
        return sources