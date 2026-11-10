"""The emulator behaviour gate and the bench recorder play every chain segment of the route."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEGMENT_STARTS = ['2_7ds_0', '0_3cc_0', '1_73s_0', '0_37c_0', '3_7cs_0']


def test_port_gate_runs_all_six_segments():
    text = (ROOT / 'scripts/port_gate.sh').read_text(encoding='utf-8')
    chains = re.findall(r'^chain (\S+)(?: --from (\S+))?$', text, re.MULTILINE)
    assert [start for _, start in chains] == [''] + SEGMENT_STARTS


def test_bench_recorder_records_all_six_segments():
    text = (ROOT / 'scripts/record_bench_runs.py').read_text(encoding='utf-8')
    starts = re.findall(r"\('[\w-]+', _native\.DEFAULT_ROUTE, (None|'\w+')\)", text)
    assert starts == ['None'] + [f"'{s}'" for s in SEGMENT_STARTS]


def test_segment_start_states_exist():
    for start in SEGMENT_STARTS:
        assert (ROOT / 'triforce/states' / f'{start}.state').is_file(), start
