import json
from contextlib import contextmanager
from io import TextIOWrapper
from pathlib import Path
from typing import ContextManager, Optional
from uuid import uuid4


def read_json_file(filepath: Path) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)


@contextmanager
def temporary_file(filepath: Optional[Path] = None) -> ContextManager[TextIOWrapper]:
    filepath = Path(filepath or f'{uuid4().hex}.tmp')
    with open(filepath, 'w') as f:
        yield f
    filepath.unlink(missing_ok=True)
