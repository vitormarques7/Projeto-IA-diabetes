"""Carregamento e manipulação inicial de dados."""

def load_split(raw_dir=None):
    """Importação tardia mantém os módulos de linha de comando independentes."""
    from src.data.loader import load_split as load

    return load() if raw_dir is None else load(raw_dir)

__all__ = ["load_split"]
