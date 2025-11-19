from pathlib import Path

import pytest
import requests
from sopp.io.tle_fetcher import TleFetcherCelestrak
from sopp.io.tle_fetcher import TleFetcherSpacetrack


class MockRequestResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code
        self.content = b"mock tle data"


class TestTleFetcher:
    def test_fetch_tles_invalid_status(self, monkeypatch, tmp_path):
        fetcher = TleFetcherCelestrak(tle_file_path=str(tmp_path))

        with monkeypatch.context() as m:
            m.setattr(
                requests,
                "get",
                lambda *args, **kwargs: MockRequestResponse(status_code=404),
            )
            with pytest.raises(requests.exceptions.HTTPError) as _:
                fetcher.fetch_tles()

    def test_fetch_tles_request_exception(self, monkeypatch, tmp_path):
        fetcher = TleFetcherCelestrak(tle_file_path=str(tmp_path))

        def raise_exception(*args, **kwargs):
            raise requests.exceptions.HTTPError

        with monkeypatch.context() as m:
            m.setattr(requests, "get", raise_exception)
            with pytest.raises(requests.exceptions.HTTPError) as _:
                fetcher.fetch_tles()

    def test_celestrak_fetch_tles_success(self, monkeypatch, tmp_path):
        tmp_file = tmp_path / "mock"
        fetcher = TleFetcherCelestrak(tle_file_path=str(tmp_file))

        with monkeypatch.context() as m:
            m.setattr(requests, "get", lambda *args, **kwargs: MockRequestResponse())
            response = fetcher.fetch_tles()

        assert response == Path(fetcher._tle_file_path)

    def test_spacetrak_fetch_tles_success(self, monkeypatch, tmp_path):
        tmp_file = tmp_path / "mock"
        fetcher = TleFetcherSpacetrack(tle_file_path=str(tmp_file))

        with monkeypatch.context() as m:
            m.setattr(requests, "post", lambda *args, **kwargs: MockRequestResponse())
            response = fetcher.fetch_tles()

        assert response == Path(fetcher._tle_file_path)
