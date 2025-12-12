from pathlib import Path

from sopp.config.json_loader import ConfigFileLoaderJson
from sopp.config.loader_base import ConfigFileLoaderBase

LOADERS = [ConfigFileLoaderJson]


def get_config_file_object(config_filepath: Path) -> ConfigFileLoaderBase:
    """
    Factory that returns the appropriate config loader based on file extension.
    Raises ValueError if the extension is not supported.
    """
    if not config_filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_filepath}")

    file_ext = config_filepath.suffix.lower()

    for loader_cls in LOADERS:
        if file_ext == loader_cls.filename_extension():
            return loader_cls(filepath=config_filepath)

    raise ValueError(
        f"No configuration loader found for extension '{file_ext}'. "
        f"Supported: {[cls.filename_extension() for cls in LOADERS]}"
    )
