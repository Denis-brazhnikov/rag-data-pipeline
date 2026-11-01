import re
from src.rag_prep.models import RawElement
from src.rag_prep.config import CleaningConfig

class TextCleaningStage:
    def __init__(self, config: CleaningConfig):
        self.config = config
        # Компилируем регулярные выражения для boilerplate (Слайд 53)
        self.boilerplate_regex = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.config.boilerplate_patterns
        ]

    def _quality_signals(self, text: str) -> dict:
        """Вычисление сигналов качества (Слайд 60)"""
        tokens = text.split()
        if not tokens:
            return {"meaningful_score": 0.0, "boilerplate_score": 1.0, "garbage_score": 1.0}
            
        letters = sum(c.isalpha() for c in text)
        unique_tokens = len(set(tokens))
        
        # Простая эвристика
        meaningful_score = letters / max(len(text), 1)
        unique_token_ratio = unique_tokens / len(tokens)
        garbage_score = 1.0 - unique_token_ratio
        
        # Проверка на boilerplate (если текст совпал с паттернами)
        is_boilerplate = any(regex.search(text) for regex in self.boilerplate_regex)
        boilerplate_score = 1.0 if is_boilerplate else 0.0
        
        return {
            "meaningful_score": round(meaningful_score, 2),
            "boilerplate_score": boilerplate_score,
            "garbage_score": round(garbage_score, 2),
            "unique_token_ratio": round(unique_token_ratio, 2),
            "is_probable_boilerplate": is_boilerplate
        }

    def run(self, elements: list[RawElement]) -> list[RawElement]:
        cleaned_elements = []
        
        for element in elements:
            text = element.text
            
            # 1. Удаляем управляющие символы (Слайд 56)
            text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
            
            # 2. Нормализация пробелов (Слайд 57)
            text = re.sub(r'[ \t\r\f\v]+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
            
            # 3. Фильтрация слишком коротких элементов (Слайд 58)
            if len(text) < self.config.min_chars:
                continue
                
            # 4. Считаем quality signals (Слайд 59)
            signals = self._quality_signals(text)
            
            # Если это явный boilerplate (например, "cookie policy"), 
            # мы можем либо удалить его, либо оставить с пометкой. 
            # По слайду 59 - оставляем, но сохраняем сигналы в metadata.
            
            element.text = text
            element.metadata["quality_signals"] = signals
            cleaned_elements.append(element)
            
        return cleaned_elements