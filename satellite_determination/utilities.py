import json
from contextlib import contextmanager
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from typing import ContextManager, Optional
from uuid import uuid4

import pytz


def read_json_file(filepath: Path) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)


@contextmanager
def temporary_file(filepath: Optional[Path] = None) -> ContextManager[TextIOWrapper]:
    filepath = Path(filepath or f'{uuid4().hex}.tmp')
    with open(filepath, 'w') as f:
        yield f
    filepath.unlink(missing_ok=True)


def convert_tz_to_utc(localtime: str) -> datetime:
    format = "%m/%d/%Y %H:%M:%S"
    local_dt = datetime.strptime(localtime, format)
    dt_utc = local_dt.astimezone(pytz.UTC)
    return dt_utc


def convert_dt_to_utc(localtime: datetime) -> datetime:
    dt_utc = localtime.astimezone(pytz.UTC)
    return dt_utc
