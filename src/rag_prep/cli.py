import typer
from src.rag_prep.config import load_config
from src.rag_prep.pipeline import RagPreparationPipeline
from src.rag_prep.flow import rag_data_preparation_flow

app = typer.Typer(help="CLI для RAG Data Preparation Pipeline")

@app.callback()
def main():
    pass

@app.command("prepare")
def prepare_command(
    config_path: str = typer.Option("config/default.yaml", help="Путь к YAML конфигу"),
    no_prefect: bool = typer.Option(False, help="Запустить напрямую без Prefect (для отладки)")
):
    """Запускает пайплайн подготовки данных."""
    
    if no_prefect:
        # Прямой запуск (для быстрой отладки)
        typer.echo("🚀 Прямой запуск пайплайна (без Prefect)...")
        config = load_config(config_path)
        pipeline = RagPreparationPipeline(config)
        pipeline.run()
    else:
        # Запуск через оркестратор Prefect (Слайд 111)
        typer.echo("🌊 Запуск пайплайна через Prefect Flow...")
        rag_data_preparation_flow(config_path)

if __name__ == "__main__":
    app()