"""Base config class(es)."""

from dataclasses import asdict, dataclass
import logging
import os
from typing import Optional
import yaml


CONFIG_FILES_SUFFIXES = (".yml", ".yaml")


@dataclass
class BaseConfig:
    """Base class for Configs."""

    @classmethod
    def load(cls, path: str) -> "BaseConfig":
        """Load BaseConfig from `path`."""
        logging.info("Loading configuration file %s.", path)
        assert path.endswith(
            CONFIG_FILES_SUFFIXES
        ), "Please specify YAML config file."
        assert os.path.exists(path), "File not found."
        with open(path, "r", encoding="utf8") as f:
            config_dict  = yaml.load(f, Loader=yaml.SafeLoader)
        return cls(**config_dict)
    

    def dump(self, path: Optional[str] = None) -> Optional[str]:
        """Save BaseConfig to `path` as YAML file, or return as string."""
        config_dict = asdict(self)
        if path is not None:
            if not path.endswith(CONFIG_FILES_SUFFIXES):
                path += CONFIG_FILES_SUFFIXES[0]
            with open(path, "w", encoding="utf8") as f:
                yaml.dump(config_dict, f)
        else:
            return yaml.dump(config_dict)
