import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "categories.json"

def load_categories() -> dict:
    text = DATA_DIR.read_text(encoding="utf-8")
    return json.loads(text)
