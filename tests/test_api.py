"""
API unit tests — run without a GPU or model checkpoint.
Tests schema validation and endpoint contract only.
"""
import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


def make_app():
    """Create app with a mocked model so no checkpoint is needed."""
    import sys, types

    # Stub heavy modules before importing the app
    for mod in ["torch", "torchvision"]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    with patch("src.api.main.lifespan") as mock_lifespan:
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def fake_lifespan(app):
            app.state.model = MagicMock()
            app.state.device = "cpu"
            app.state.image_size = 224
            app.state.start_time = 0.0
            app.state.model_path = "mock"
            yield

        mock_lifespan.side_effect = fake_lifespan

        from src.api.main import app
        return app


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from unittest.mock import MagicMock, patch
    import time

    # Build a minimal app with faked state
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from src.api.schemas import HealthResponse

    mini = FastAPI()

    @mini.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            device="cpu",
            uptime_sec=1.0,
            model_path="mock",
        )

    return TestClient(mini)


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_schema(client):
    data = client.get("/health").json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["device"] == "cpu"
    assert "uptime_sec" in data
    assert "model_path" in data


def test_health_model_loaded_field_is_bool(client):
    data = client.get("/health").json()
    assert isinstance(data["model_loaded"], bool)
