# llmbridge/model_catalog/model_registry.py

from functools import lru_cache
from pathlib import Path

import yaml


class ModelRegistry:
    @staticmethod
    @lru_cache(maxsize=1)
    def load() -> dict:
        path = Path(__file__).parent / "providers.yaml"
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @classmethod
    def default(cls, provider: str) -> str:
        return cls.load().get(provider, "")
