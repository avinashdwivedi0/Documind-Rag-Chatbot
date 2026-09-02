"""Local metadata store for persisted document workspaces."""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from backend.config import WORKSPACE_DIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(workspace_id: str) -> str:
    return os.path.join(WORKSPACE_DIR, f"{workspace_id}.json")


def _write(workspace: Dict) -> None:
    path = _path(workspace["id"])
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(workspace, file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def get_workspace(workspace_id: str) -> Optional[Dict]:
    try:
        with open(_path(workspace_id), "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def upsert_workspace(
    workspace_id: str, files: Iterable, name: str = "", intelligence: Optional[Dict] = None
) -> Dict:
    workspace = get_workspace(workspace_id) or {
        "id": workspace_id,
        "name": name.strip() or "Untitled workspace",
        "created_at": _now(),
        "conversations": [
            {
                "id": "default",
                "title": "New conversation",
                "created_at": _now(),
                "updated_at": _now(),
            }
        ],
    }
    if name.strip():
        workspace["name"] = name.strip()
    intelligence = intelligence or {}
    workspace["documents"] = [
        {
            "name": file.name,
            "size": getattr(file, "size", 0),
            "type": os.path.splitext(file.name)[1].lstrip(".").upper(),
            "status": "processed",
            "uploaded_at": _now(),
            "tags": [],
            "intelligence": intelligence.get(file.name, {}),
        }
        for file in files
    ]
    workspace["updated_at"] = _now()
    _write(workspace)
    return workspace


def list_workspaces() -> List[Dict]:
    workspaces = []
    for filename in os.listdir(WORKSPACE_DIR):
        if filename.endswith(".json"):
            workspace = get_workspace(filename[:-5])
            if workspace:
                workspaces.append(workspace)
    return sorted(workspaces, key=lambda item: item.get("updated_at", ""), reverse=True)


def create_conversation(workspace_id: str, title: str = "New conversation") -> Dict:
    workspace = get_workspace(workspace_id)
    if not workspace:
        raise ValueError("Workspace was not found.")
    sequence = len(workspace.get("conversations", [])) + 1
    conversation = {
        "id": f"chat-{sequence}",
        "title": title.strip() or "New conversation",
        "created_at": _now(),
        "updated_at": _now(),
    }
    workspace.setdefault("conversations", []).append(conversation)
    workspace["updated_at"] = _now()
    _write(workspace)
    return conversation


def rename_conversation(workspace_id: str, conversation_id: str, title: str) -> None:
    workspace = get_workspace(workspace_id)
    if not workspace:
        return
    for conversation in workspace.get("conversations", []):
        if conversation["id"] == conversation_id:
            conversation["title"] = title.strip() or "Untitled conversation"
            conversation["updated_at"] = _now()
    workspace["updated_at"] = _now()
    _write(workspace)


def touch_conversation(workspace_id: str, conversation_id: str) -> None:
    workspace = get_workspace(workspace_id)
    if not workspace:
        return
    for conversation in workspace.get("conversations", []):
        if conversation["id"] == conversation_id:
            conversation["updated_at"] = _now()
    workspace["updated_at"] = _now()
    _write(workspace)


def delete_workspace_metadata(workspace_id: str) -> None:
    try:
        os.remove(_path(workspace_id))
    except FileNotFoundError:
        pass
