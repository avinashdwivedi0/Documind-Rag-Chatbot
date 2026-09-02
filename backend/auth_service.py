"""MongoDB-backed user authentication helpers for app-level login and registration."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import bcrypt
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

from backend.config import USER_STORE_FILE, ensure_dirs

MONGODB_URI = os.getenv("MONGODB_URI") or "mongodb://localhost:27017"
MONGODB_DB = os.getenv("MONGODB_DB", "documind")
MONGODB_USERS_COLLECTION = os.getenv("MONGODB_USERS_COLLECTION", "users")


def _local_users() -> Dict[str, Dict[str, Any]]:
    """Load local user fallback store from disk."""
    ensure_dirs()
    if not os.path.exists(USER_STORE_FILE):
        return {}
    try:
        with open(USER_STORE_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_local_users(users: Dict[str, Dict[str, Any]]) -> None:
    """Persist local user fallback store to disk."""
    ensure_dirs()
    with open(USER_STORE_FILE, "w", encoding="utf-8") as handle:
        json.dump(users, handle, indent=2, default=str)


def _use_local_fallback() -> bool:
    """Return true when MongoDB is unavailable and local auth fallback should be used."""
    try:
        get_users_collection()
        return False
    except (PyMongoError, ServerSelectionTimeoutError, OSError, ValueError):
        return True


def get_client() -> MongoClient:
    """Return a configured MongoDB client."""
    return MongoClient(MONGODB_URI)


def get_users_collection() -> Collection:
    """Return the user collection for auth operations."""
    return get_client()[MONGODB_DB][MONGODB_USERS_COLLECTION]


def hash_password(password: str) -> str:
    """Hash a plain-text password with bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a candidate password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def register_user(name: str, email: str, password: str) -> Dict[str, Any]:
    """Create a new user account unless the email already exists."""
    normalized_email = email.strip().lower()

    try:
        collection = get_users_collection()
        existing = collection.find_one({"email": normalized_email})
        if existing:
            raise ValueError("An account with this email already exists.")

        user = {
            "name": name.strip(),
            "email": normalized_email,
            "password": hash_password(password),
            "created_at": datetime.now(timezone.utc),
        }
        result = collection.insert_one(user)
        user["_id"] = str(result.inserted_id)
        user.pop("password", None)
        return user
    except (PyMongoError, ServerSelectionTimeoutError, OSError, ValueError):
        if isinstance(sys.exc_info()[1], ValueError):
            raise

        users = _local_users()
        if normalized_email in users:
            raise ValueError("An account with this email already exists.")

        user = {
            "_id": f"local-{len(users) + 1}",
            "name": name.strip(),
            "email": normalized_email,
            "password": hash_password(password),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        users[normalized_email] = user
        _save_local_users(users)
        user.pop("password", None)
        return user


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user based on email and password."""
    normalized_email = email.strip().lower()
    try:
        collection = get_users_collection()
        user = collection.find_one({"email": normalized_email})
        if not user:
            return None
        if not verify_password(password, user.get("password", "")):
            return None
        safe_user = dict(user)
        safe_user["_id"] = str(safe_user.get("_id"))
        safe_user.pop("password", None)
        return safe_user
    except (PyMongoError, ServerSelectionTimeoutError, OSError):
        users = _local_users()
        user = users.get(normalized_email)
        if not user:
            return None
        if not verify_password(password, user.get("password", "")):
            return None
        safe_user = dict(user)
        safe_user["_id"] = str(safe_user.get("_id"))
        safe_user.pop("password", None)
        return safe_user


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Lookup a user by email."""
    normalized_email = email.strip().lower()
    try:
        user = get_users_collection().find_one({"email": normalized_email})
    except (PyMongoError, ServerSelectionTimeoutError, OSError):
        user = _local_users().get(normalized_email)
    if not user:
        return None
    safe_user = dict(user)
    safe_user["_id"] = str(safe_user.get("_id"))
    safe_user.pop("password", None)
    return safe_user
