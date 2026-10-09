"""Q4 and truss results unchanged against the recorded baseline."""

import numpy as np
import pytest

from tests.baseline_cases import BASELINE_PATH, all_cases

RTOL = 1e-12

baseline = np.load(BASELINE_PATH)
current = all_cases()


@pytest.mark.parametrize("name", sorted(baseline.files))
def test_matches_baseline(name):
    ref = baseline[name]
    new = current[name]
    assert new.shape == ref.shape
    if np.issubdtype(ref.dtype, np.integer):
        np.testing.assert_array_equal(new, ref)
    else:
        scale = np.max(np.abs(ref))
        assert np.max(np.abs(new - ref)) <= RTOL * scale
