"""Smoke tests for example scripts to ensure they stay in sync with the API."""

import shutil
import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
REPO_ROOT = EXAMPLES_DIR.parent
TEST_TLE = Path(__file__).parent / "io" / "load_satellites" / "satellites.tle"


def _run_example(name: str):
    """Run an example script and assert it exits cleanly."""
    script = EXAMPLES_DIR / name
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"{name} failed with exit code {result.returncode}:\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def test_example_runs():
    """Copies a test TLE to the repo root if needed, then runs the example."""
    tle_dest = REPO_ROOT / "satellites.tle"
    created = False
    if not tle_dest.exists():
        shutil.copy(TEST_TLE, tle_dest)
        created = True
    try:
        _run_example("example.py")
    finally:
        if created:
            tle_dest.unlink()


def test_example_persistence_runs():
    _run_example("example_persistence.py")


def test_example_link_budget_runs():
    _run_example("example_link_budget.py")
