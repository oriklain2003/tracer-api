"""
Pytest fixtures for user/auth integration tests.
Uses real DB from POSTGRES_DSN (e.g. prod when env points to prod).
"""
import uuid
import pytest
from fastapi.testclient import TestClient

# Import app after path is set (run pytest from project root: python -m pytest tests/)
from api.api import app


@pytest.fixture
def client():
    """Synchronous TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def test_user(client):
    """
    Create a regular user (not admin) for tests. Admin will create/edit/block/delete this user.
    Returns dict with id, username, email, password for the created user.
    """
    username = f"testuser_{uuid.uuid4().hex[:8]}"
    payload = {
        "username": username,
        "email": f"{username}@test.example",
        "password": "testpass123",
        "user_type": "user",
        "max_active_sessions": 5,
    }
    r = client.post("/api/auth/register", json=payload)
    assert r.status_code == 201, (r.status_code, r.text)
    data = r.json()
    return {
        "id": data["id"],
        "username": data["username"],
        "email": data.get("email") or payload["email"],
        "password": payload["password"],
    }


@pytest.fixture
def super_user(client):
    """Create a super_user (unique name) and return token, headers, and credentials. Used so admin endpoints (list_users, get_user, etc.) pass without requiring env to have super_user."""
    username = f"testsuper_{uuid.uuid4().hex[:8]}"
    password = "testpass123"
    r = client.post(
        "/api/auth/register",
        json={
            "username": username,
            "email": f"{username}@test.example.com",
            "password": password,
            "user_type": "super_user",
            "max_active_sessions": 10,
        },
    )
    assert r.status_code in (200, 201), (r.status_code, r.text)
    r2 = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r2.status_code == 200, (r2.status_code, r2.text)
    token = r2.json()["access_token"]
    return {
        "username": username,
        "password": password,
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
    }


@pytest.fixture
def admin_token(super_user):
    """Bearer token for a super_user (so admin-only endpoints work)."""
    return super_user["token"]


@pytest.fixture
def auth_headers(super_user):
    """Headers with Bearer token for super_user."""
    return super_user["headers"]


@pytest.fixture
def admin_credentials(super_user):
    """Credentials of the super_user used in tests (for test_change_password, test_refresh_token)."""
    return {"username": super_user["username"], "password": super_user["password"]}
