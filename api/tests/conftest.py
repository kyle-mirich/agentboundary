import shutil
from unittest.mock import patch

import pytest

from app import config as config_module
from app.database import init_db


@pytest.fixture(autouse=True)
def isolate_settings(request, tmp_path):
    config_module.settings.data_dir = tmp_path / "data"
    config_module.settings.workspace_dir = tmp_path / "workspaces"
    config_module.settings.memory_dir = tmp_path / "memories"
    config_module.settings.artifacts_dir = tmp_path / "artifacts"
    if request.node.get_closest_marker("no_db"):
        config_module.settings.data_dir.mkdir(parents=True, exist_ok=True)
        config_module.settings.workspace_dir.mkdir(parents=True, exist_ok=True)
        config_module.settings.memory_dir.mkdir(parents=True, exist_ok=True)
        config_module.settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    else:
        init_db()
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)


def pytest_configure(config):
    config.addinivalue_line("markers", "no_db: skip database initialisation for this test")
