import yaml
from pathlib import Path
from types import SimpleNamespace


class Config(SimpleNamespace):
    """Configuration object loaded from a YAML file."""

    @classmethod
    def from_yaml(cls, path: str):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r") as f:
            data = yaml.safe_load(f)

        return cls._to_namespace(data)

    @classmethod
    def _to_namespace(cls, data):
        if isinstance(data, dict):
            return cls(**{
                k: cls._to_namespace(v)
                for k, v in data.items()
            })
        elif isinstance(data, list):
            return [cls._to_namespace(v) for v in data]
        else:
            return data
