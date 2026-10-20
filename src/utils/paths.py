"""Caminhos independentes do diretório de execução do notebook."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
SOURCES_PATH = PROJECT_ROOT / "data" / "sources.json"
