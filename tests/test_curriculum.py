"""Backward curriculum stage logic (``triforce.curriculum``)."""

import json
import random

from triforce.curriculum import TRAJECTORY_FILE, BackwardCurriculum, CurriculumSpec


def _trajectory(root, name, length, success=True):
    directory = root / name
    directory.mkdir(parents=True)
    for index in range(length):
        (directory / f"{index:04d}.state").write_bytes(b'')
    (directory / TRAJECTORY_FILE).write_text(json.dumps({'start_state': name, 'success': success}))


def _curriculum(tmp_path, **spec):
    _trajectory(tmp_path / 'traj', 'a', 20)
    _trajectory(tmp_path / 'traj', 'failed', 5, success=False)
    return BackwardCurriculum(CurriculumSpec(str(tmp_path / 'traj'), str(tmp_path / 'run' / 'curriculum.json'),
                                             fraction=1.0, stages=4, window=4, threshold=0.5, **spec),
                              random.Random(0))


def test_starts_near_the_end_and_moves_back_on_success(tmp_path):
    curriculum = _curriculum(tmp_path)
    assert set(curriculum.tracks) == {'a'}                      # unsuccessful explorations are skipped
    path, tag = curriculum.next()
    assert tag[0] == 'a' and 15 <= tag[1] < 20 and path.endswith(f"{tag[1]:04d}.state")

    for _ in range(3):
        assert curriculum.record(('a', 15), True)['point'] == 15
    assert curriculum.record(('a', 16), False)['point'] == 10   # 3/4 >= 0.5: one stage (stride 5) earlier
    assert all(10 <= curriculum.next()[1][1] < 15 for _ in range(20))


def test_failed_window_restarts_the_stage_and_stale_episodes_do_not_count(tmp_path):
    curriculum = _curriculum(tmp_path)
    for _ in range(3):
        curriculum.record(('a', 15), False)
    curriculum.record(('a', 15), True)                          # 1/4 < 0.5: stay, counters restart
    assert curriculum.progress()['a'] == {'length': 20, 'point': 15, 'attempts': 0, 'successes': 0,
                                          'stages_passed': 0}
    curriculum.record(('a', 19), True)
    curriculum.record(('a', 3), True)                           # older/later stage index: ignored
    assert curriculum.progress()['a']['attempts'] == 1


def test_complete_track_is_no_longer_sampled_and_progress_is_shared(tmp_path):
    curriculum = _curriculum(tmp_path)
    other = BackwardCurriculum(curriculum.spec, random.Random(1))   # a second worker on the same file
    for point in (15, 10, 5):
        for _ in range(4):
            other.record(('a', point), True)
    assert curriculum.progress()['a']['point'] == 0
    assert curriculum.next() is None
