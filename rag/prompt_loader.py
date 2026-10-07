"""Carga automáticamente la última versión del prompt de generación del RAG.

Mismo patrón de versionamiento usado en el Taller 1
(taller 1 genAI/prompt_loader.py): archivos rag/prompts/v1.toml,
v2.toml, etc. Se usa siempre el número más alto salvo que se indique
`version` explícitamente, para poder iterar el prompt sin tocar código y
dejando trazabilidad.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"
VERSION_PATTERN = re.compile(r"^v(\d+)\.toml$")


def list_versions() -> list[int]:
    versions = []
    for file in PROMPTS_DIR.iterdir():
        match = VERSION_PATTERN.match(file.name)
        if match:
            versions.append(int(match.group(1)))
    if not versions:
        raise FileNotFoundError(f"No hay prompts versionados (vN.toml) en {PROMPTS_DIR}")
    return sorted(versions)


def latest_version() -> int:
    return list_versions()[-1]


def load_prompt(version: int | None = None) -> dict:
    resolved_version = version if version is not None else latest_version()
    prompt_path = PROMPTS_DIR / f"v{resolved_version}.toml"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"No existe {prompt_path}. Versiones disponibles: {list_versions()}"
        )

    with prompt_path.open("rb") as f:
        data = tomllib.load(f)

    data["_resolved_version"] = resolved_version
    data["_source_file"] = str(prompt_path)
    return data
