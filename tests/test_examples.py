"""Smoke tests for example scripts to ensure they stay in sync with the API."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _run_example(name: str):
    """Run an example script and assert it exits cleanly."""
    script = EXAMPLES_DIR / name
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(EXAMPLES_DIR.parent),
    )
    assert result.returncode == 0, (
        f"{name} failed with exit code {result.returncode}:\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def test_example_runs():
    _run_example("example.py")


def test_example_persistence_runs():
    _run_example("example_persistence.py")


def test_example_link_budget_runs():
    _run_example("example_link_budget.py")
