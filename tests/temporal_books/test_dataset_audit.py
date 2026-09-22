from collections import Counter

import pytest

from src.temporal_books.p0_audit import timestamp_at_quantile


def test_temporal_cutoff_keeps_tied_timestamp_in_later_split():
    counts = Counter({10: 2, 20: 3, 30: 5})
    assert timestamp_at_quantile(counts, 0.5) == 20


def test_temporal_cutoff_rejects_invalid_quantile():
    with pytest.raises(ValueError):
        timestamp_at_quantile(Counter({10: 1}), 1.0)
