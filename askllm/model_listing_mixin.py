# askllm/model_listing_mixin.py

from typing import List


class ModelListingMixin:
    """
    Marks a provider as supporting live model listing (`askllm.list_models()`).
    """

    def fetch_latest_models(self) -> List[str]:
        """
        Providers must return a flat list of raw model names from their API.
        """
        raise NotImplementedError
