from pymongo.errors import ServerSelectionTimeoutError

from backend import auth_service
from backend.auth_service import hash_password, verify_password


def test_hash_and_verify_password_round_trip():
    plain = "StrongPass!123"
    hashed = hash_password(plain)

    assert hashed != plain
    assert verify_password(plain, hashed) is True
    assert verify_password("wrong-pass", hashed) is False


def test_hashes_are_unique_for_same_password():
    first = hash_password("SamePassword123")
    second = hash_password("SamePassword123")

    assert first != second


def test_authentication_falls_back_when_mongo_is_unavailable(monkeypatch, tmp_path):
    def raise_timeout(*args, **kwargs):
        raise ServerSelectionTimeoutError("Mongo unavailable")

    user_store = tmp_path / "users.json"
    monkeypatch.setattr(auth_service, "USER_STORE_FILE", str(user_store))
    monkeypatch.setattr(auth_service, "get_users_collection", raise_timeout)

    user = auth_service.register_user("Fallback User", "fallback@example.com", "StrongPass!123")
    assert user["email"] == "fallback@example.com"

    authenticated = auth_service.authenticate_user("fallback@example.com", "StrongPass!123")
    assert authenticated is not None
    assert authenticated["email"] == "fallback@example.com"
