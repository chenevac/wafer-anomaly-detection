from abc import ABC, abstractmethod
from typing import Any, Union, final

from wafer_ad.utils.configuration.base import BaseConfig


class Configurable(ABC):
    """Base class for all configurable classes in onlinelearning module."""

    def __init__(self) -> None:
        """Construct `Configurable`."""
        self._config: BaseConfig

        # Base class constructor
        super().__init__()

    @classmethod  
    @abstractmethod
    def from_config(
        cls,
        source: Union[BaseConfig, str],
    ) -> Any:
        pass

    @final
    @property
    def config(self) -> BaseConfig:
        """Return configuration."""
        try:
            return self._config
        except AttributeError as err:
            raise AttributeError("Config was not set. ") from err

    @final
    def save_config(self, path: str) -> None:
        """Save Config to `path` as YAML file."""
        self.config.dump(path)
