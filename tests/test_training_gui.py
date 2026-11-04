"""Training chart ranges: 10-point success bands and heart padding (Qt offscreen)."""

import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('training_gui', ROOT / 'training-gui.py')
gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gui)


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


def series(values, label='success'):
    return {'label': label, 'color': gui.C.ok, 'points': list(enumerate(values, 1))}


@pytest.mark.parametrize('values,stop,expected', [
    ([0.95, 1.0], 0.99, (0.90, 1.0)),
    ([0.26, 0.60], 0.65, (0.20, 0.70)),
    ([0.26, 0.60], 0.90, (0.20, 0.90)),
    ([0.249, 0.90], 0.65, (0.0, 1.0)),
    ([0.25, 0.50], None, (0.20, 0.50)),
    ([1.0], 1.0, (0.90, 1.0)),
    ([0.8], 0.8, (0.80, 0.90)),
])
def test_success_bands_include_data_and_criterion(qapp, values, stop, expected):
    chart = gui.LineChart('Success', gui.fmt_pct, scale='success')
    references = [] if stop is None else [(stop, gui.C.faint, 'early stop')]
    chart.set_data([series(values)], references)
    assert chart.y_bounds() == expected


def test_success_range_retains_other_series_and_eval(qapp):
    chart = gui.LineChart('Success', gui.fmt_pct, scale='success')
    chart.set_data([series([0.95, 0.99], 'balanced'), series([0.60, 0.70], 'raw')],
                   [(0.99, gui.C.faint, 'early stop'), (0.40, gui.C.eval, 'eval')])
    assert chart.y_bounds() == (0.40, 1.0)
    chart.set_data([series([0.95, 0.99], 'balanced'), series([0.20, 0.70], 'curriculum')])
    assert chart.y_bounds() == (0.0, 1.0)


@pytest.mark.parametrize('values,expected', [
    ([-0.1, 0.7], (-0.3, 0.9)),
    ([-0.9, -0.5], (-1.1, -0.3)),
    ([0.5, 0.7], (0.3, 0.9)),
    ([0.0], (-0.2, 0.2)),
])
def test_health_padding_does_not_expand_to_zero(qapp, values, expected):
    chart = gui.LineChart('Health change', fmt=gui.fmt_hearts_change, scale='health')
    chart.set_data([series(values)], [(0.0, gui.C.faint, 'no change')])
    assert chart.y_bounds() == pytest.approx(expected)


def test_health_range_covers_real_and_total(qapp):
    chart = gui.LineChart('Health change', scale='health')
    chart.set_data([series([-0.8, -0.6], 'real'), series([0.3, 0.4], 'total')],
                   [(0.0, gui.C.faint, 'no change')])
    assert chart.y_bounds() == pytest.approx((-1.0, 0.6))


def test_chart_range_updates_when_switching_runs(qapp):
    chart = gui.LineChart('Success', gui.fmt_pct, scale='success')
    chart.set_data([series([0.0, 0.5])])
    assert chart.y_bounds() == (0.0, 1.0)
    chart.set_data([series([0.96, 0.99])])
    assert chart.y_bounds() == (0.90, 1.0)
