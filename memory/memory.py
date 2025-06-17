import json
from pathlib import Path

MEMORY_PATH = Path("memory/memory.json")

def load_memory():
    if not MEMORY_PATH.exists() or MEMORY_PATH.stat().st_size == 0:
        return {"history": []}
    try:
        with open(MEMORY_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # File corrotto o vuoto
        return {"history": []}

def save_memory(memory):
    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)
