import typer
from src.rag_prep.config import load_config, load_chunking_config
from src.rag_prep.pipeline import RagPreparationPipeline, RagChunkingPipeline
from src.rag_prep.flow import rag_data_preparation_flow, chunking_flow

app = typer.Typer(help="CLI для RAG Data Preparation и Chunking")

@app.callback()
def main():
    pass

@app.command("prepare")
def prepare_command(
    config_path: str = typer.Option("config/default.yaml", help="Путь к YAML конфигу подготовки"),
    no_prefect: bool = typer.Option(False, help="Запустить напрямую без Prefect")
):
    """Запускает пайплайн подготовки данных (Loading -> Export)."""
    
    if no_prefect:
        typer.echo("🚀 Прямой запуск пайплайна подготовки (без Prefect)...")
        config = load_config(config_path)
        pipeline = RagPreparationPipeline(config)
        pipeline.run()
    else:
        typer.echo("🌊 Запуск пайплайна подготовки через Prefect Flow...")
        rag_data_preparation_flow(config_path)

@app.command("chunk")
def chunk_command(
    config_path: str = typer.Option("config/chunking.yaml", help="Путь к YAML конфигу чанкинга"),
    no_prefect: bool = typer.Option(False, help="Запустить напрямую без Prefect")
):
    """Запускает пайплайн чанкинга документов (Loading -> Splitting -> Export)."""
    
    if no_prefect:
        typer.echo("✂️ Прямой запуск пайплайна чанкинга (без Prefect)...")
        config = load_chunking_config(config_path)
        pipeline = RagChunkingPipeline(config)
        pipeline.run()
    else:
        typer.echo("🌊 Запуск пайплайна чанкинга через Prefect Flow...")
        chunking_flow(config_path)

if __name__ == "__main__":
    app()