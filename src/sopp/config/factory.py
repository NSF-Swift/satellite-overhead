from pathlib import Path

from sopp.config.base import ConfigFileLoaderBase
from sopp.config.json_loader import ConfigFileLoaderJson
from sopp.utils.helpers import get_default_config_file_filepath


def get_config_file_object(
    config_filepath: Path | None = None,
) -> ConfigFileLoaderBase:
    config_filepath = config_filepath or get_default_config_file_filepath()
    for config_class in [ConfigFileLoaderJson]:
        if config_class.filename_extension() in str(config_filepath):
            return config_class(filepath=config_filepath)
