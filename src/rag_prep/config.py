from pydantic import BaseModel, Field
from typing import List, Optional
import yaml

class RunConfig(BaseModel):
    name: str = "rag_data_preparation"
    seed: int = 42

class PathConfig(BaseModel):
    input_dir: str = "data/raw"
    output_dir: str = "data/prepared"

class LoaderConfig(BaseModel):
    recursive: bool = True
    allowed_extensions: List[str] = [".pdf", ".txt", ".html", ".htm", ".md"]
    exclude_hidden: bool = True
    num_files_limit: Optional[int] = None

class ParserConfig(BaseModel):
    strategy: str = "fast"
    fail_on_error: bool = False

class CleaningConfig(BaseModel):
    min_chars: int = 12
    boilerplate_patterns: List[str] = [
        r"cookie|cookies|куки|cookie policy",
        r"подписаться|subscribe|newsletter",
        r"навигация|главное меню|menu",
        r"©|copyright|all rights reserved"
    ]

class NormalizationConfig(BaseModel):
    spacy_language: str = "ru"
    unicode_form: str = "NFKC"
    lowercase: bool = False

class DeduplicationConfig(BaseModel):
    enabled: bool = True
    threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    num_perm: int = Field(default=128, ge=16)
    shingle_size: int = 5
    min_tokens: int = 8

class StructuringConfig(BaseModel):
    group_by_section: bool = True

class LoggingConfig(BaseModel):
    level: str = "INFO"
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment: str = "rag-data-preparation"

class PipelineConfig(BaseModel):
    run: RunConfig
    paths: PathConfig
    loader: LoaderConfig
    parser: ParserConfig
    cleaning: CleaningConfig
    normalization: NormalizationConfig
    deduplication: DeduplicationConfig
    structuring: StructuringConfig
    logging: LoggingConfig

def load_config(config_path: str = "config/default.yaml") -> PipelineConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    return PipelineConfig(**config_data)