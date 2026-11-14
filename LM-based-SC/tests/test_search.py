
import numpy as np
import pytest

from lmsc.discovery.predict import search


class DummyModel:
    def predict(self, X):
        return np.asarray(X)[:, 0] * 10


@pytest.mark.parametrize("options, message", [
    ({"elements": ["In"]}, "no candidates"),
    ({"elements": ["In", "In"]}, "unique"),
    ({"elements": ["In", "Xx"]}, "unknown"),
    ({"chunk_size": 0}, "positive"),
    ({"top_k": 0}, "positive"),
    ({"component_counts": ()}, "components"),
    ({"component_counts": (2, 2)}, "unique"),
    ({"denominator": 1}, "denominator"),
])
def test_invalid_search_is_explained(options, message):
    args = dict(elements=["In", "Sn"], component_counts=(2,), denominator=4)
    args.update(options)
    with pytest.raises(ValueError, match=message):
        search(DummyModel(), ["In", "Sn"], **args)


def test_chunking_preserves_ranking_and_candidate_count():
    args = dict(elements=["In", "Sn"], component_counts=(2,), denominator=8, top_k=3)
    small, count = search(DummyModel(), ["In", "Sn"], chunk_size=2, **args)
    large, other_count = search(DummyModel(), ["In", "Sn"], chunk_size=100, **args)
    assert count == other_count == 7
    assert small["formula"].tolist() == large["formula"].tolist()
    assert small["tc_lcb_K"].is_monotonic_decreasing
