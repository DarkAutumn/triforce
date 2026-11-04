"""``triforce.evaluate``: per-specialist evaluation over every seed start state."""

import json
import subprocess
import sys

from triforce.evaluate import _group, _linger_stats, evaluate_specialist, format_report
from triforce.model import ModelConfig, ZeldaPolicy
from triforce.observation import ObservationBuilder
from triforce.registry import save_checkpoint
from triforce.specialist import load_specialist


def test_evaluate_cli_uses_supplied_task_instead_of_reloading_its_name(tmp_path):
    network = ZeldaPolicy(ObservationBuilder.observation_space(),
                          ModelConfig(hidden_dim=32, entity_d_model=32, entity_ff_dim=32, grid_out_dim=32))
    checkpoint = save_checkpoint('boss-aquamentus', 'alternate-task', network, root=tmp_path)
    config = tmp_path / 'alternate.yaml'
    config.write_text("name: boss-aquamentus\nrooms: ['(1, 0x35)']\nstart_states: ['1_35s_0']\n"
                      "success: [exit-correct]\nmax_steps: 1\n")
    output = tmp_path / 'eval.json'
    subprocess.run([sys.executable, '-m', 'triforce.evaluate', '--specialist', str(config),
                    '--checkpoint', str(checkpoint), '--episodes-per-state', '1', '--output', str(output)],
                   check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result['episodes'] == 1
    assert result['endings'] == {'max-steps': 1}
    assert result['records'][0]['steps'] == 1


def test_evaluate_specialist_covers_every_seed_state_reproducibly(session, tmp_path):
    network = ZeldaPolicy(ObservationBuilder.observation_space(),
                          ModelConfig(hidden_dim=32, entity_d_model=32, entity_ff_dim=32, grid_out_dim=32))
    ckpt = save_checkpoint('boss-aquamentus', 'untrained', network, root=tmp_path)
    config = load_specialist('boss-aquamentus').model_copy(update={'max_steps': 40})

    result = evaluate_specialist(config, ckpt, episodes_per_state=2, session=session, seed=5)
    records = result['records']
    assert result['episodes'] == 6 == len(records)
    assert set(result['by_state']) == {'1_35s_0', '1_35s_1', '1_35s_2'}
    assert all(stats['episodes'] == 2 for stats in result['by_state'].values())
    assert sorted(r['seed'] for r in records) == list(range(5, 11))
    assert all(r['outcome'] in {'success', 'hard-failure', 'soft-failure'} for r in records)
    assert all(0 < r['steps'] <= 40 for r in records)
    assert result['success_rate'] == sum(r['success'] for r in records) / 6
    assert set(result['by_room']) == {'1:0x35'}
    assert all(set(r['hearts']) == {'dropped', 'taken', 'healed', 'wasted', 'left', 'expired'} for r in records)
    assert result['hearts_dropped'] == sum(r['hearts']['dropped'] for r in records) / 6
    assert result['mean_health_change_total'] == result['health_change_total'] >= result['mean_health_change']
    assert {'health_change_total', 'hearts_dropped', 'hearts_taken', 'hearts_left'} <= set(result['by_room']['1:0x35'])
    assert all({'max_steps', 'max_frames', 'tail_steps', 'tail_frames', 'steps', 'frames',
                'boss_kill_steps', 'boss_kill_frames', 'heart_container_steps', 'heart_container_frames'}
               == set(r['linger']) for r in records)
    assert result['linger_max_steps'] == sum(r['linger']['max_steps'] for r in records) / 6
    assert result['linger_worst_steps'] == max(r['linger']['max_steps'] for r in records)
    assert result['by_room']['1:0x35']['linger_max_steps'] == result['linger_max_steps']
    assert all('linger_max_steps' in stats for stats in result['by_state'].values())
    assert 'linger max/tail/worst' in format_report(result)

    again = evaluate_specialist(config, ckpt, episodes_per_state=2, session=session, seed=5)
    assert again['records'] == records


def test_linger_evaluation_aggregates_only_measured_records():
    base = {'health': 2, 'start_health': 3, 'success': True, 'ending': 'exit-correct'}
    records = [{**base, 'start_state': 'a', 'room': 'room-a'},
               {**base, 'start_state': 'a', 'room': 'room-a',
                'linger': {'max_steps': 10, 'max_frames': 200, 'tail_steps': 0, 'tail_frames': 0}},
               {**base, 'start_state': 'b', 'room': 'room-a',
                'linger': {'max_steps': 30, 'max_frames': 100, 'tail_steps': 10, 'tail_frames': 50}},
               {**base, 'start_state': 'c', 'room': 'room-b'}]
    assert _linger_stats(records) == {'linger_max_steps': 20, 'linger_max_frames': 150,
                                     'linger_tail_steps': 5, 'linger_tail_frames': 25,
                                     'linger_worst_steps': 30, 'linger_worst_frames': 200}
    rooms = _group(records, lambda r: r['room'])
    states = _group(records, lambda r: r['start_state'])
    assert rooms['room-a']['linger_max_steps'] == 20
    assert rooms['room-a']['linger_worst_frames'] == 200
    assert states['a']['linger_max_steps'] == 10
    assert states['a']['linger_tail_steps'] == 0
    assert states['b']['linger_max_steps'] == 30
    assert not any(key.startswith('linger_') for key in rooms['room-b'])
    assert not any(key.startswith('linger_') for key in states['c'])
    assert _linger_stats([]) == _linger_stats([base]) == {}
