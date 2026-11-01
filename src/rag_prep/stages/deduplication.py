import hashlib
from datasketch import MinHash, MinHashLSH
from src.rag_prep.models import RawElement
from src.rag_prep.config import DeduplicationConfig

class DeduplicationStage:
    def __init__(self, config: DeduplicationConfig):
        self.config = config
        # Инициализируем LSH-индекс с порогом схожести из конфига (Слайд 90)
        self.lsh = MinHashLSH(
            threshold=self.config.threshold,
            num_perm=self.config.num_perm
        )

    def _shingles(self, tokens: list[str]) -> list[tuple[str, ...]]:
        """Создает шинглы (n-граммы) из токенов для учета контекста (Слайд 91)"""
        size = self.config.shingle_size
        if len(tokens) < size:
            return [tuple(tokens)]
        return [tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)]

    def run(self, elements: list[RawElement]) -> list[RawElement]:
        unique_elements = []
        seen_hashes = set()
        
        for element in elements:
            text = element.text
            tokens = text.split()
            
            # Пропускаем слишком короткие элементы (Слайд 90)
            if len(tokens) < self.config.min_tokens:
                unique_elements.append(element)
                continue

            # 1. Точная дедупликация (Exact match) через SHA-256 (Слайд 88)
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            if text_hash in seen_hashes:
                # По желанию можно добавить лог
                continue 
            
            # 2. Нечеткая дедупликация (Near-duplicate) через MinHash (Слайд 88)
            m = MinHash(num_perm=self.config.num_perm)
            shingles = self._shingles(tokens)
            for shingle in shingles:
                m.update(" ".join(shingle).encode('utf-8'))
            
            # Проверяем, есть ли уже похожий текст в индексе
            if self.lsh.query(m):
                continue # Если нашли near-duplicate, пропускаем
            
            # Если дубликатов не найдено — сохраняем элемент
            seen_hashes.add(text_hash)
            self.lsh.insert(element.element_id, m)
            unique_elements.append(element)
            
        return unique_elements