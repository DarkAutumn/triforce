# pylint: disable=all
"""Tests for _save_results loud-failure guards."""

import pytest

from evaluate import _save_results


def test_save_results_exits_on_empty_metrics(tmp_path):
    with pytest.raises(SystemExit) as e:
        _save_results(str(tmp_path / 'm.pt'), {}, [9] * 100, 11, 100, 'x')
    assert e.value.code != 0
    assert not (tmp_path / 'm.eval.json').exists()


def test_save_results_exits_on_missing_progress(tmp_path):
    with pytest.raises(SystemExit) as e:
        _save_results(str(tmp_path / 'm.pt'), {'success-rate': 0.5}, None, 11, 100, 'x')
    assert e.value.code != 0
    assert not (tmp_path / 'm.eval.json').exists()
