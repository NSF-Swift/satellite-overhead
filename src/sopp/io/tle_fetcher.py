from abc import ABC, abstractmethod
from pathlib import Path

import requests

from sopp.utils.helpers import get_satellites_filepath


class TleFetcherBase(ABC):
    def __init__(self, tle_file_path: str | None = None):
        self._tle_file_path = (
            Path(tle_file_path)
            if tle_file_path is not None
            else get_satellites_filepath()
        )

    @abstractmethod
    def _fetch_content(self) -> requests.Response:
        pass

    def fetch_tles(self) -> Path:
        try:
            response = self._fetch_content()
            if response.status_code == 200:
                self._write_tles_to_file(response.content)
                return self._tle_file_path
            else:
                raise requests.exceptions.HTTPError(
                    f"Failed to fetch TLEs. Status code: {response.status_code}"
                )
        except requests.exceptions.RequestException:
            raise

    def _write_tles_to_file(self, content):
        self._tle_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._tle_file_path, "wb") as f:
            f.write(content)


class TleFetcherCelestrak(TleFetcherBase):
    def _fetch_content(self):
        url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
        return requests.get(url=url, allow_redirects=True)


class TleFetcherSpacetrack(TleFetcherBase):
    def _fetch_content(self):
        import os

        from dotenv import load_dotenv

        load_dotenv()
        IDENTITY = os.getenv("IDENTITY")
        PASSWORD = os.getenv("PASSWORD")

        url = "https://www.space-track.org/ajaxauth/login"
        query = "https://www.space-track.org/basicspacedata/query/class/gp/decay_date/null-val/epoch/%3Enow-30/orderby/norad_cat_id/format/3le"
        data = {"identity": IDENTITY, "password": PASSWORD, "query": query}
        return requests.post(url=url, data=data)
