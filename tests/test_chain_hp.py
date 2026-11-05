"""``triforce.chain_hp``: the per-room arrival-hearts profile built from chain run jsonl files."""

import json

from triforce import chain_hp
from triforce.runner import RoomRecord, RunResult, write_jsonl


def _run(seed, *rooms):
    """A run whose rooms are (level, room, in_cave, health_in)."""
    return RunResult('milestone1', seed, 'failure', rooms=[
        RoomRecord(i, level, room, in_cave, 'ow-start', health_in=health)
        for i, (level, room, in_cave, health) in enumerate(rooms)])


def test_profile_keys_rooms_and_sorts_arrival_hearts_over_files(tmp_path):
    first, second = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    write_jsonl(str(first), [_run(0, (0, '0x77', False, 3.0), (0, '0x77', True, 3.0), (0, '0x77', False, 2.5)),
                             _run(1, (1, '0x73', False, 1.5))])
    write_jsonl(str(second), [_run(2, (0, '0x77', False, 0.5), (1, '0x73', False, 2.0))])
    with open(second, 'a', encoding='utf-8') as file:
        file.write('\n   \n')                                  # blank lines are skipped
    profile = chain_hp.build_profile([str(first), str(second)])
    assert profile == {'0:0x77': [0.5, 2.5, 3.0], '0:0x77:cave': [3.0], '1:0x73': [1.5, 2.0]}
    assert list(profile) == sorted(profile)
    assert chain_hp.build_profile([]) == {}


def test_cli_writes_the_profile_and_prints_per_room_means(tmp_path, capsys):
    chain = tmp_path / 'chain.jsonl'
    write_jsonl(str(chain), [_run(0, (2, '0x7d', False, 4.0)), _run(1, (2, '0x7d', False, 3.0))])
    out = tmp_path / 'profile.json'
    assert chain_hp.main([str(chain), '--out', str(out)]) == 0
    assert json.loads(out.read_text()) == {'2:0x7d': [3.0, 4.0]}
    line, = capsys.readouterr().out.splitlines()
    assert line.split() == ['2:0x7d', '2', 'visits', 'mean', '3.50', 'hearts']

