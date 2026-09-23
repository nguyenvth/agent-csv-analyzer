import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).parent / "run_log.txt"


def log_run(goal: str, model_key: str, elapsed_seconds: float,
            num_steps: int, result: str, error: str = None) -> None:
    # Append-only: each run adds a block, preserving history across
    # multiple goals for later reporting - overwriting would lose
    # earlier runs needed to build the report's measurement table.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "goal": goal,
        "model": model_key,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "num_steps": num_steps,
        "success": error is None,
        "error": error,
        "result_preview": (result[:300] if result else None),
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")