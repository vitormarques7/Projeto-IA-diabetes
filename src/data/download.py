"""Baixa fontes imutáveis e verifica os bytes antes de gravá-los."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from src.utils.paths import RAW_DIR, SOURCES_PATH


def sources():
    return json.loads(SOURCES_PATH.read_text(encoding="utf-8"))


def verify_bytes(content, spec):
    actual = hashlib.sha256(content).hexdigest()
    if actual != spec["sha256"]:
        raise ValueError(f"Checksum inválido para {spec['filename']}: {actual}")


def verify_file(path, spec):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Fonte ausente: {path}. Execute python -m src.data.download."
        )
    verify_bytes(path.read_bytes(), spec)


def download(raw_dir=RAW_DIR):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in sources().items():
        target = raw_dir / spec["filename"]
        if target.exists():
            # Nunca substitui silenciosamente um arquivo local diferente.
            verify_file(target, spec)
        else:
            request = Request(spec["url"], headers={"User-Agent": "diabetes-reproduction/1.0"})
            with urlopen(request, timeout=30) as response:
                content = response.read()
            verify_bytes(content, spec)
            target.write_bytes(content)
        print(f"{name}: {target} (SHA-256 verificado)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    download(args.raw_dir)


if __name__ == "__main__":
    main()
