"""`CurriculumSpec` follows the yaml's rule for every way it is built (CLI overrides too): stages and window at
least 1, fraction and threshold in [0, 1]; anything else is a ValueError, never a Rust panic."""

import pytest

from triforce import _native as native


@pytest.mark.parametrize('settings', [{'stages': 0}, {'window': 0}, {'fraction': 2.0}, {'fraction': float('nan')},
                                      {'fraction': -1.0}, {'threshold': -0.1}, {'threshold': 5.0}])
def test_out_of_range_settings_are_refused(tmp_path, settings):
    with pytest.raises(ValueError, match='curriculum'):
        native.CurriculumSpec(str(tmp_path), **settings)


def test_in_range_settings_are_kept(tmp_path):
    spec = native.CurriculumSpec(str(tmp_path), fraction=1.0, stages=1, window=1, threshold=0.0)
    assert 'stages=1' in repr(spec)
