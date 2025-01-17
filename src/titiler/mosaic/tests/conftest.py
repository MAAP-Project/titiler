"""``pytest`` configuration."""

import os
from typing import Any, Dict

import pytest
from rasterio.io import MemoryFile

from titiler.core.resources.enums import OptionalHeader
from titiler.mosaic.factory import MosaicTilerFactory
from titiler.mosaic.settings import mosaic_config

from fastapi import FastAPI

from starlette.testclient import TestClient

DATA_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def set_env(monkeypatch):
    """Set Env variables."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "jqt")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "rde")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/tmp/noconfigheere")


@pytest.fixture
def client():
    """Setup an app and create a client for it."""

    mosaic_config.backend = "file://"
    mosaic_config.host = "/tmp"

    mosaic = MosaicTilerFactory(
        optional_headers=[OptionalHeader.server_timing, OptionalHeader.x_assets],
        router_prefix="mosaic",
    )

    app = FastAPI()
    app.include_router(mosaic.router, prefix="/mosaic")
    client = TestClient(app)

    return client


def parse_img(content: bytes) -> Dict[Any, Any]:
    """Read tile image and return metadata."""
    with MemoryFile(content) as mem:
        with mem.open() as dst:
            return dst.profile
