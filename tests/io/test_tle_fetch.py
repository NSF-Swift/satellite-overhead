import os
import pytest
import requests
from sopp.io.tle import fetch_tles


class MockRequestResponse:
    def __init__(self, status_code=200, content=b"mock tle data"):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        """Mimics the requests.Response method."""
        if 400 <= self.status_code < 600:
            raise requests.exceptions.HTTPError(f"Mock Error {self.status_code}")


class TestTleFetcher:
    def test_fetch_tles_invalid_status(self, monkeypatch, tmp_path):
        """Verifies that 404/500 codes raise an HTTPError."""
        monkeypatch.setattr(
            requests,
            "get",
            lambda *args, **kwargs: MockRequestResponse(status_code=404),
        )

        output_file = tmp_path / "test.tle"

        with pytest.raises(requests.exceptions.HTTPError):
            fetch_tles(output_path=output_file, source="celestrak")

    def test_fetch_tles_request_exception(self, monkeypatch, tmp_path):
        """Verifies that connection errors are propagated."""

        def raise_exception(*args, **kwargs):
            raise requests.exceptions.ConnectionError("Network Down")

        monkeypatch.setattr(requests, "get", raise_exception)

        output_file = tmp_path / "test.tle"

        with pytest.raises(requests.exceptions.ConnectionError):
            fetch_tles(output_path=output_file, source="celestrak")

    def test_celestrak_fetch_tles_success(self, monkeypatch, tmp_path):
        """Verifies successful download and file writing."""
        monkeypatch.setattr(
            requests,
            "get",
            lambda *args, **kwargs: MockRequestResponse(content=b"celestrak data"),
        )

        output_file = tmp_path / "celestrak.tle"

        # Act
        result_path = fetch_tles(output_path=output_file, source="celestrak")

        # Assert
        assert result_path == output_file
        assert output_file.exists()
        assert output_file.read_bytes() == b"celestrak data"

    def test_spacetrak_fetch_tles_success(self, monkeypatch, tmp_path):
        """Verifies SpaceTrack logic (POST request + Env Vars)."""
        # 1. Mock Env Vars (Required for SpaceTrack logic)
        monkeypatch.setenv("IDENTITY", "user")
        monkeypatch.setenv("PASSWORD", "pass")

        # 2. Mock Request
        monkeypatch.setattr(
            requests,
            "post",
            lambda *args, **kwargs: MockRequestResponse(content=b"spacetrack data"),
        )

        output_file = tmp_path / "spacetrack.tle"

        # Act
        result_path = fetch_tles(output_path=output_file, source="spacetrack")

        # Assert
        assert result_path == output_file
        assert output_file.read_bytes() == b"spacetrack data"

    def test_spacetrack_missing_credentials(self, monkeypatch, tmp_path):
        """Verifies correct error if env vars are missing."""
        # Ensure env vars are unset
        monkeypatch.delenv("IDENTITY", raising=False)
        monkeypatch.delenv("PASSWORD", raising=False)

        output_file = tmp_path / "fail.tle"

        with pytest.raises(ValueError, match="IDENTITY and PASSWORD"):
            fetch_tles(output_path=output_file, source="spacetrack")
