import os
import tempfile
import pytest


@pytest.fixture
def tmpdir_env(monkeypatch, tmp_path):
    """Set TMPDIR for tests that invoke src.core.helper.bkeep.init_worker."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    yield tmp_path
