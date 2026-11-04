import hashlib
import logging
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
import pdfplumber

# Настраиваем логгер для утилит
logger = logging.getLogger(__name__)

def stable_id(*parts: Any, length: int = 24) -> str:
    """
    Генерирует детерминированный стабильный ID на основе переданных частей.
    Используется для создания ID документов и чанков (вместо устаревшего MD5).
    """
    payload = '|'.join(str(part) for part in parts)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:length]

def generate_document_id(text: str) -> str:
    """Создает уникальный хеш на основе текста документа (обертка для stable_id)."""
    return stable_id(text, length=32)

def clean_html(html_content: str) -> str:
    """Удаляет HTML-теги, скрипты, стили и boilerplate (навигацию, футеры)."""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Удаляем скрипты, стили, навигацию, футеры и заголовки
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()
    
    text = soup.get_text(separator=' ')
    # Очищаем от лишних пробелов и переносов
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

def extract_text_from_pdf(file_path: Path) -> str:
    """Извлекает текст из PDF файла с обработкой ошибок."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        logger.error(f"Ошибка при извлечении PDF {file_path}: {e}")
    return text

def extract_text_from_txt(file_path: Path) -> str:
    """Читает обычный текстовый файл или Markdown."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Ошибка при чтении текстового файла {file_path}: {e}")
        return ""

def parse_document(file_path: Path, file_type: str) -> str:
    """Маршрутизатор парсеров в зависимости от формата (fallback-режим)."""
    if file_type == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_type in [".html", ".htm"]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return clean_html(f.read())
        except Exception as e:
            logger.error(f"Ошибка при чтении HTML {file_path}: {e}")
            return ""
    elif file_type in [".txt", ".md"]:
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Неподдерживаемый формат: {file_type}")