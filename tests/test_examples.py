"""Smoke tests for example scripts to ensure they stay in sync with the API."""

import os
import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
REPO_ROOT = EXAMPLES_DIR.parent


def _run_example(name: str):
    """Run an example script and assert it exits cleanly."""
    script = EXAMPLES_DIR / name
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert result.returncode == 0, (
        f"{name} failed with exit code {result.returncode}:\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def _run_example_with_mock_sopp(name: str):
    """Run an example with SOPP engine mocked to return empty results."""
    script = EXAMPLES_DIR / name
    env = {**os.environ, "MPLBACKEND": "Agg"}
    wrapper = (
        "from unittest.mock import patch, MagicMock; "
        "from sopp.models.satellite.trajectory_set import TrajectorySet; "
        "empty = TrajectorySet([]); "
        "p1 = patch('sopp.sopp.Sopp.get_satellites_above_horizon', return_value=empty); "
        "p2 = patch('sopp.sopp.Sopp.get_satellites_crossing_main_beam', return_value=empty); "
        "p3 = patch('sopp.config.builder.ConfigurationBuilder.load_satellites', return_value=MagicMock()); "
        "p1.start(); p2.start(); p3.start(); "
        f"exec(open('{script}').read()); "
        "p1.stop(); p2.stop(); p3.stop()"
    )
    result = subprocess.run(
        [sys.executable, "-c", wrapper],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert result.returncode == 0, (
        f"{name} failed with exit code {result.returncode}:\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def test_example_runs():
    _run_example_with_mock_sopp("example.py")


def test_example_persistence_runs():
    _run_example("example_persistence.py")


def test_example_link_budget_runs():
    _run_example("example_link_budget.py")


def test_example_planning_runs():
    _run_example_with_mock_sopp("example_planning.py")
