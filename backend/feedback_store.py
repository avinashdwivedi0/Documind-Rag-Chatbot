"""Append-only local feedback store and lightweight analytics."""

import json
import os
from datetime import datetime, timezone
from typing import Dict

from backend.config import FEEDBACK_FILE


def record_feedback(workspace_id: str, conversation_id: str, question: str, answer: str, rating: str) -> None:
    entry = {"workspace_id": workspace_id, "conversation_id": conversation_id, "question": question, "answer": answer, "rating": rating, "created_at": datetime.now(timezone.utc).isoformat()}
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def feedback_summary() -> Dict[str, int]:
    counts = {"total": 0, "helpful": 0, "not_helpful": 0}
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                counts["total"] += 1
                if entry.get("rating") == "helpful":
                    counts["helpful"] += 1
                elif entry.get("rating") == "not_helpful":
                    counts["not_helpful"] += 1
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return counts
