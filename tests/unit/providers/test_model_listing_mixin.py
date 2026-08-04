import pytest

from askllm.model_listing_mixin import ModelListingMixin


def test_fetch_latest_models_base_raises_not_implemented():
    class _Bare(ModelListingMixin):
        name = "bare"

    with pytest.raises(NotImplementedError):
        _Bare().fetch_latest_models()
