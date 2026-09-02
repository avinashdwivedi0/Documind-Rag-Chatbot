"""Small, dependency-free persistent chat cache for document workspaces."""

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage

from backend.config import HISTORY_DIR


def history_path(workspace_id: str, conversation_id: str = "default") -> str:
    """Return the JSON cache path for a workspace conversation."""
    return os.path.join(HISTORY_DIR, f"{workspace_id}__{conversation_id}.json")


def load_history(workspace_id: str, conversation_id: str = "default") -> List[Dict[str, str]]:
    """Return cached turns, tolerating an absent or malformed cache file."""
    try:
        path = history_path(workspace_id, conversation_id)
        # Migrate the original single-conversation cache on its first read.
        if conversation_id == "default" and not os.path.exists(path):
            path = os.path.join(HISTORY_DIR, f"{workspace_id}.json")
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def save_history(
    workspace_id: str, messages: List[Dict[str, str]], conversation_id: str = "default"
) -> None:
    """Atomically save a workspace conversation to the local cache."""
    path = history_path(workspace_id, conversation_id)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    temp_fd, temp_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as file:
            json.dump(messages, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    except PermissionError:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        with open(path, "w", encoding="utf-8") as file:
            json.dump(messages, file, ensure_ascii=False, indent=2)
    except FileNotFoundError:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(messages, file, ensure_ascii=False, indent=2)
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def append_turn(workspace_id: str, conversation_id: str, role: str, content: str) -> Dict[str, str]:
    """Append a user or assistant turn and persist it to the workspace history."""
    messages = load_history(workspace_id, conversation_id)
    message = {
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    messages.append(message)
    save_history(workspace_id, messages, conversation_id)
    return message


def clear_history(workspace_id: str, conversation_id: str = "default") -> None:
    """Remove the cached conversation for a workspace if it exists."""
    try:
        os.remove(history_path(workspace_id, conversation_id))
    except FileNotFoundError:
        pass


def restore_memory(memory, messages: List[Dict[str, str]]) -> None:
    """Seed LangChain memory from cached UI messages."""
    for message in messages:
        content = message.get("content", "")
        if message.get("role") == "user":
            memory.chat_memory.add_message(HumanMessage(content=content))
        elif message.get("role") == "assistant":
            memory.chat_memory.add_message(AIMessage(content=content))
