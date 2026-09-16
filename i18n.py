import json
from pathlib import Path


def load_translations(locale: str = "vi") -> dict:
    # Load once per run; file is small, no need for caching layer.
    path = Path(__file__).parent / "locales" / f"{locale}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)