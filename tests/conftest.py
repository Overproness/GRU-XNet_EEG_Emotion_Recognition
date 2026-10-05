"""Keep temporary test PDFs inside the writable workspace on Windows."""
from pathlib import Path
from uuid import uuid4
import pytest


@pytest.fixture
def pdf_workspace():
    # Pytest's mode-0700 temp directories are inaccessible to this Windows
    # sandbox token. Ordinary workspace directories work for PDF rendering.
    allowed = (Path(__file__).resolve().parent / "_tmp").resolve()
    allowed.mkdir(parents=True, exist_ok=True)
    target = (allowed / uuid4().hex[:8]).resolve()
    target.relative_to(allowed)
    target.mkdir()
    return target
