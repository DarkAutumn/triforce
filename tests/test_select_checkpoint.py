"""``triforce select-checkpoint`` (``triforce.select_checkpoint``, round 25) on a tiny scripted team: the segment from
``SEGMENT_STATE`` to route step ``SEGMENT_STOP_STEP`` is ow-octorok's room, won by a network that walks south."""

import json

import pytest

from triforce import registry, select_checkpoint

from .conftest import SEGMENT_STATE, SEGMENT_STOP_STEP, promote_untrained, require_state
from .networks import scripted

FAMILY = 'ow-octorok'


@pytest.fixture(name='team')
def fixture_team(tmp_path):
    """A team whose promoted ow-octorok walks south (wins), and a run ``r`` of ow-octorok with numbered checkpoints
    ``it000001`` (waits: loses), ``it000002`` (walks south: wins) and a ``ckpt.pt`` of iteration 3 that waits."""
    root = promote_untrained(tmp_path / 'models')
    registry.save_checkpoint(FAMILY, 'south', scripted('MOVE S 16'), root=root)
    registry.promote(FAMILY, 'south', root=root)
    run_dir = registry.run_dir(FAMILY, 'r', root)
    for iteration, action in ((1, 'WAIT 8'), (2, 'MOVE S 16')):
        registry.save_checkpoint(FAMILY, 'r', scripted(action), iterations=iteration, root=root)
        registry.snapshot_checkpoint(run_dir, iteration)
    registry.save_checkpoint(FAMILY, 'r', scripted('WAIT 8'), iterations=3, root=root)
    return root, run_dir


def _select(root, tmp_path, *extra):
    return select_checkpoint.main(
        ['--models-dir', str(root), FAMILY, 'r', '--from', str(require_state(SEGMENT_STATE)),
         '--stop-at', f'step:{SEGMENT_STOP_STEP}', '--runs', '2', '--confirm', '2', '--parallel', '2',
         '--chains-dir', str(tmp_path / 'chains'), '--logs-dir', str(tmp_path / 'logs'), *extra])


def _plans(logs):
    return [json.loads((folder / 'training.json').read_text()) for folder in sorted(logs.iterdir()) if folder.is_dir()]


def test_the_screen_ranks_the_checkpoints_and_the_confirmation_plays_fresh_seeds(team, tmp_path):
    root, run_dir = team
    assert _select(root, tmp_path) == 0
    report = json.loads((run_dir / 'selection.json').read_text())
    rows = {row['label']: row for row in report['candidates']}
    assert set(rows) == {'it000001', 'it000002', 'last', 'promoted'}
    assert report['ranking'][0] == report['winner'] == 'it000002' and 'promoted' not in report['ranking']
    assert rows['it000002']['screen']['arrived'] == 2 and rows['it000001']['screen']['arrived'] == 0
    assert rows['last']['screen']['arrived'] == 0 and rows['promoted']['screen']['arrived'] == 2
    assert rows['it000002']['iteration'] == 2 and rows['it000002']['sha1'] == registry.checkpoint_sha1(
        registry.snapshot_dir(run_dir, 2))
    assert rows['it000002']['screen_vs_last'] == {'seeds': 2, 'rate': 1.0, 'rate_se': 0.0, 'hearts': None,
                                                  'hearts_se': None, 'hearts_seeds': 0}
    assert rows['it000002']['screen_vs_promoted']['rate'] == 0.0
    assert rows['it000002']['screen_vs_promoted']['hearts_seeds'] == 2
    # Confirmation: the best other than last, last and promoted, on the seeds after the screen's.
    assert report['screen'] == {'first_seed': 0, 'runs': 2}
    assert report['confirm'] == {'first_seed': 2, 'runs': 2, 'top': 1}
    assert {label for label, row in rows.items() if row['confirm']} == {'it000002', 'last', 'promoted'}
    assert rows['it000002']['confirm_vs_last']['rate'] == 1.0
    assert report['chosen'] == 'it000002' and report['reason'].startswith('confirmed')
    assert report['applied'] is None and registry.load_checkpoint(run_dir)['iterations'] == 3
    seeds = [json.loads(line)['seed'] for line in (tmp_path / 'chains' / 'confirm-last.jsonl').read_text().splitlines()]
    assert seeds == [2, 3]
    (plan,) = _plans(tmp_path / 'logs')            # one job, one group in the training GUI
    assert plan['kind'] == 'eval-batch' and plan['status'] == 'completed'
    # Its estimate counts the 7 chains' route steps (2 runs each, one step from the segment state to step 25).
    assert plan['work'] == {'chain_steps': 7 * 2}
    assert [run['name'] for run in plan['runs']] == ['screen-it000001', 'screen-it000002', 'screen-last',
                                                     'screen-promoted', 'confirm-top1', 'confirm-last',
                                                     'confirm-promoted']
    assert plan['runs'][4]['candidate'] == 'it000002'


def test_a_second_selection_reuses_the_chains_and_apply_swaps_the_checkpoint(team, tmp_path):
    root, run_dir = team
    assert _select(root, tmp_path) == 0
    (run_dir / 'eval.json').write_text('{"act": "temperature:0.5+relax:16"}')
    old_sha1 = registry.checkpoint_sha1(run_dir)
    assert _select(root, tmp_path, '--apply') == 0
    second = _plans(tmp_path / 'logs')[-1]
    assert all(run['resumed'] for run in second['runs'])          # every chain was played by the same weights
    report = json.loads((run_dir / 'selection.json').read_text())
    applied = report['applied']
    winner = registry.snapshot_dir(run_dir, 2)
    assert (run_dir / 'ckpt.pt').read_bytes() == (winner / 'ckpt.pt').read_bytes()
    assert applied['from_sha1'] == old_sha1 and applied['to_sha1'] == registry.checkpoint_sha1(run_dir)
    kept = run_dir / 'checkpoints' / applied['previous_kept']
    assert applied['previous_kept'] == 'it000003' and registry.checkpoint_sha1(kept) == old_sha1
    assert applied['stale_evaluations'] == ['eval.json'] and (kept / 'eval.json').is_file()
    assert not (run_dir / 'eval.json').exists()
    # The run's act mode is bound to the new weights; the old weights keep theirs.
    assert (run_dir / 'act').read_text().split() == [registry.DEFAULT_ACT, applied['to_sha1']]
    assert (kept / 'act').read_text().split() == [registry.DEFAULT_ACT, old_sha1]
    assert registry.model_version(run_dir) == registry.model_version(winner)
    assert registry.load_checkpoint(run_dir)['iterations'] == 2
    # --report-only reads the chains without a job; the new last is not measured yet, so it can't rank.
    assert select_checkpoint.main(['--models-dir', str(root), FAMILY, 'r', '--from', str(require_state(SEGMENT_STATE)),
                                   '--stop-at', f'step:{SEGMENT_STOP_STEP}', '--runs', '2', '--confirm', '2',
                                   '--chains-dir', str(tmp_path / 'chains'), '--report-only']) == 1
    assert len(_plans(tmp_path / 'logs')) == 2


def test_the_command_line_is_checked_before_anything_runs(team, tmp_path, capsys):
    root, _ = team
    assert _select(root, tmp_path, '--only', 'it000009') == 2
    assert 'it000009' in capsys.readouterr().err
    assert _select(root, tmp_path, '--confirm-first-seed', '1') == 2
    assert 'fresh' in capsys.readouterr().err
    assert not (tmp_path / 'logs').exists()


def _fake_chain(plan, phase, label, sha1, act, arrived):
    """A finished chain output of ``phase`` under ``label``, as eval-chain writes it, played by the weights ``sha1``:
    its first ``arrived`` seeds reach the end (no game is played)."""
    from triforce.cli import meta_path  # pylint: disable=import-outside-toplevel
    runs, first_seed = plan.seeds(phase)
    output = select_checkpoint.chain_output(plan, phase, label)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps({'seed': seed, 'result': 'success' if i < arrived else 'failure', 'rooms': [
        {'level': 0, 'room': '0x3c', 'in_cave': False, 'health_in': 3.0, 'outcome': 'exit-correct'}]}) + '\n'
        for i, seed in enumerate(range(first_seed, first_seed + runs))))
    meta_path(output).write_text(json.dumps({
        'batches': [{'first_seed': first_seed, 'runs': runs, 'finished': True}],
        'provenance': {'clips': plan.clips, 'start_from': plan.start,
                       'models': [{'name': FAMILY, 'checkpoint_sha1': sha1, 'act': act}]}}))


def test_a_promotion_during_the_batch_neither_changes_the_reference_nor_stops_the_selection(team, tmp_path):
    """Dungeon 6 (round 26): selections started while one run was promoted played their ``promoted`` chains after
    another was, so the chains held other weights than the candidate, and the confirmation of the screen's best
    failed (``LookupError: the screen has no chain of promoted yet``) with no ``selection.json``."""
    root, run_dir = team
    argv = ['--models-dir', str(root), FAMILY, 'r', '--from', str(require_state(SEGMENT_STATE)),
            '--stop-at', f'step:{SEGMENT_STOP_STEP}', '--runs', '2', '--confirm', '2',
            '--chains-dir', str(tmp_path / 'chains')]
    plan, pool = select_checkpoint.build_plan(select_checkpoint.build_parser().parse_args(argv))
    by_label = {c.label: c for c in pool}
    south = registry.promoted_checkpoint(FAMILY, root)
    registry.save_checkpoint(FAMILY, 'north', scripted('MOVE N 16'), root=root)
    registry.promote(FAMILY, 'north', root=root)
    # The reference stays the run promoted when the selection was planned: its chains name its weights.
    assert by_label['promoted'].sha1 == registry.checkpoint_sha1(south)
    assert select_checkpoint.candidate_args(plan, by_label['promoted'], 'screen')[-2:] == [
        '--checkpoint', f'{FAMILY}={south}']
    # Chains left by the old code: every candidate of the run played, ``promoted`` by the other weights.
    north = registry.checkpoint_sha1(registry.promoted_checkpoint(FAMILY, root))
    for label, arrived in (('it000001', 0), ('it000002', 2), ('last', 1)):
        _fake_chain(plan, 'screen', label, by_label[label].sha1, by_label[label].act, arrived)
    _fake_chain(plan, 'screen', 'promoted', north, by_label['promoted'].act, 2)
    selection = select_checkpoint.Selection(plan, pool)
    top1 = next(entry for entry in selection.entries() if entry.name == 'confirm-top1')
    top1.prepare()                                              # raised LookupError
    assert top1.fields['candidate'] == 'it000002'
    # The confirmation of the screen's best and of last played; a third run is promoted (no chain of it: its rows
    # are empty): --report-only writes the choice.
    _fake_chain(plan, 'confirm', 'it000002', by_label['it000002'].sha1, by_label['it000002'].act, 2)
    _fake_chain(plan, 'confirm', 'last', by_label['last'].sha1, by_label['last'].act, 0)
    registry.save_checkpoint(FAMILY, 'west', scripted('MOVE W 16'), root=root)
    registry.promote(FAMILY, 'west', root=root)
    assert select_checkpoint.main([*argv, '--report-only']) == 0
    report = json.loads((run_dir / 'selection.json').read_text())
    rows = {row['label']: row for row in report['candidates']}
    assert report['winner'] == report['chosen'] == 'it000002' and report['reason'].startswith('confirmed')
    assert rows['promoted']['sha1'] == registry.checkpoint_sha1(registry.promoted_checkpoint(FAMILY, root))
    assert rows['promoted']['screen'] is None and rows['promoted']['confirm'] is None
    assert 'screen_vs_promoted' not in rows['it000002']


def test_scores_and_paired_differences():
    runs = lambda results: [{'seed': seed, 'result': result, 'rooms': [{'level': 0, 'room': '0x3c', 'in_cave': False,  # noqa: E731
                                                                       'health_in': 3.0, 'outcome': 'exit-correct'},
                                                                      {'level': 0, 'room': '0x4c', 'in_cave': False,
                                                                       'health_in': hearts, 'outcome': 'milestone'}]}
                            for seed, (result, hearts) in enumerate(results)]
    a = select_checkpoint.score(runs([('success', 2.0), ('success', 3.0), ('failure', 1.0), ('success', 1.0)]), None)
    b = select_checkpoint.score(runs([('success', 1.0), ('failure', 3.0), ('failure', 1.0), ('success', 2.0)]), None)
    assert a['arrived'] == 3 and a['rate'] == 0.75 and a['hearts'] == 2.0
    assert b['arrived'] == 2 and b['hearts'] == 1.5
    versus = select_checkpoint.paired(a, b)
    assert versus['rate'] == 0.25 and versus['hearts_seeds'] == 2 and versus['hearts'] == 0.0
    assert select_checkpoint.rank_key(a) > select_checkpoint.rank_key(b)
