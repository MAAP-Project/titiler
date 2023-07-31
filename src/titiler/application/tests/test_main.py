"""Test titiler.application.main.app."""
from fastapi import FastAPI
from starlette.testclient import TestClient


def test_health(app):
    """Test /healthz endpoint."""
    response = app.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ping": "pong!"}

    response = TestClient(FastAPI()).get("/openapi.json")
    assert response.status_code == 200

    response = app.get("/docs")
    assert response.status_code == 200
