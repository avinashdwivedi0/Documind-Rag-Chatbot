"""Project-wide path and directory configuration for local storage."""

import os

from dotenv import load_dotenv

load_dotenv()

# Define directories for uploads and vector storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "..", "data", "uploads")
VECTOR_DIR = os.path.join(BASE_DIR, "..", "data", "vectors")
HISTORY_DIR = os.path.join(BASE_DIR, "..", "data", "chat_history")
WORKSPACE_DIR = os.path.join(BASE_DIR, "..", "data", "workspaces")
FEEDBACK_FILE = os.path.join(BASE_DIR, "..", "data", "feedback.jsonl")
EVAL_DIR = os.path.join(BASE_DIR, "..", "data", "evaluations")
USER_STORE_FILE = os.path.join(BASE_DIR, "..", "data", "users.json")


def ensure_dirs():
    """Ensure necessary directories exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
