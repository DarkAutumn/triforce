"""The yardstick (``triforce.lab.bench``): state choice, a run on a small manifest, paired comparison."""

import json

from triforce import registry
from triforce.events import PLAN_FILE, PROGRESS_FILE, progress_job
from triforce.lab import bench

from .networks import random_network


def test_states_spread_over_room_entries_prefer_the_route_and_fall_back_to_seed_states(tmp_path):
    directory = tmp_path / 'harvested' / 'zol'
    directory.mkdir(parents=True)
    for name in ['d3first_3_7cs_h1_2', 'd3first_3_7cs_h2_2', 'd3first_3_7cs_h3_2', 'd3first_4_40s_h1_9',
                 '3_7cs_h5_2']:
        (directory / f'{name}.state').write_bytes(b'')
    states, source = bench.choose_states('zol', 3, tmp_path)
    assert source == 'harvested' and len(states) == 3
    assert all(s.startswith('harvested/zol/d3first_') for s in states)
    assert sum('4_40s' in s for s in states) == 1        # each entry gets its turn before any gets a second
    assert bench.choose_states('zol', 3, tmp_path) == (states, source)
    # Another route's states (the manifest regenerated with --route): its own entries first.
    (directory / 'd2first_3_7cs_h1_2.state').write_bytes(b'')
    assert bench.choose_states('zol', 3, tmp_path, route='d2first') == (['harvested/zol/d2first_3_7cs_h1_2'],
                                                                        'harvested')


def test_the_manifest_is_regenerated_for_a_routes_states_and_team(tmp_path):
    models = tmp_path / 'models'
    registry.save_checkpoint('zol', 'a', random_network(1), root=models)
    registry.promote('zol', 'a', models)
    (tmp_path / 'states' / 'harvested' / 'zol').mkdir(parents=True)
    (tmp_path / 'states' / 'harvested' / 'zol' / 'd2first_3_7cs_h1_2.state').write_bytes(b'')
    out = tmp_path / 'bench.json'
    assert bench.main(['manifest', '--route', 'd2first', '--models-dir', str(models), '--states-dir',
                       str(tmp_path / 'states'), '--out', str(out), '--states-per-family', '2']) == 0
    manifest = json.loads(out.read_text())
    assert manifest['full']['route'] == 'd2first' and '--route d2first' in manifest['full']['command']
    assert manifest['entries'] == [{'key': 'zol', 'family': 'zol', 'source': 'harvested',
                                    'states': ['harvested/zol/d2first_3_7cs_h1_2']}]


def test_a_bench_run_and_its_paired_comparison(states_dir, tmp_path, logs_dir):
    manifest = {'format': 1, 'act': registry.DEFAULT_ACT, 'episodes': 2, 'seed': 7, 'clips': 'on',
                'entries': [{'key': 'ow-start', 'family': 'ow-start', 'source': 'seed', 'states': ['start']}]}
    paths = [registry.save_checkpoint('ow-start', f'n{seed}', random_network(seed), root=tmp_path)
             for seed in (1, 2)]
    # the bench's job has a log folder and a one-line progress: episodes played of the total, success so far
    with progress_job('lab-bench', 'eval', 2) as progress:
        first = bench.run_bench({'ow-start': str(paths[0])}, manifest, 2, states_dir=states_dir,
                                log=lambda _: None, progress=progress)
    results = [first, bench.run_bench({'ow-start': str(paths[1])}, manifest, 2, states_dir=states_dir,
                                      log=lambda _: None)]
    folder, = logs_dir.glob('*_lab-bench')
    assert json.loads((folder / PLAN_FILE).read_text())['status'] == 'completed'
    events = [json.loads(line) for line in (folder / PROGRESS_FILE).read_text().splitlines()]
    assert [event['event'] for event in events] == ['eval_start', 'eval_progress', 'end']
    assert events[1]['done'] == events[1]['episodes'] == 2 and events[1]['key'] == 'ow-start'
    assert events[1]['success_rate'] == first['keys']['ow-start']['success'] and events[2]['status'] == 'completed'
    row = results[0]['keys']['ow-start']
    assert row['episodes'] == 2 and len(row['records']) == 2
    assert {'success', 'hearts_lost', 'hits_per_visit', 'decisions', 'bombs'} <= set(row)
    same = bench.compare(results[0], results[0])
    assert same['total']['pairs'] == 2 and same['total']['success'] == (0.0, 0.0)
    other = bench.compare(results[0], results[1])
    want = results[1]['keys']['ow-start']['decisions'] - row['decisions']
    assert abs(other['keys']['ow-start']['decisions'][0] - want) < 1e-9
    assert '**total**' in bench.format_compare(other)
    stored = json.loads(json.dumps(bench.compact(results[1])))
    assert bench.compare(results[0], stored) == other


def _row(successes):
    return {'records': [{'state': f's{i}', 'seed': i, 'success': s, 'hearts_lost': 0.0, 'hits': 0, 'visits': 1,
                         'decisions': 1, 'bombs': 0} for i, s in enumerate(successes)]}


def test_compare_lists_the_rows_a_network_loses_on():
    base = {'keys': {'kept': _row([True] * 20), 'lost': _row([True] * 20), 'noise': _row([True] * 20)}}
    new = {'keys': {'kept': _row([True] * 20), 'lost': _row([False] * 10 + [True] * 10),
                    'noise': _row([False] + [True] * 19)}}
    result = bench.compare(base, new)
    assert result['losing'] == ['lost']           # -0.05 on one episode in 20 is within two standard errors
    assert '**Losing** (1 rows' in bench.format_compare(result) and 'lost -0.500' in bench.format_compare(result)


def test_an_arrived_entry_plays_its_states_without_the_familys_randomization(tmp_path):
    entry = {'key': 'darknut@d3first', 'family': 'darknut', 'source': bench.ROOMS, 'arrived': True,
             'states': [f'{bench.ROOMS}/darknut/d3first_3_59n_h5_7']}
    config, directory = bench.entry_task(entry, tmp_path)
    assert directory == str(tmp_path / bench.ROOMS / 'darknut')
    assert config.to_dict()['ram_randomization']['hp'] is None
    config, _ = bench.entry_task({**entry, 'arrived': False}, tmp_path)
    assert config.to_dict()['ram_randomization']['hp'] is not None


def test_the_rooms_manifest_keeps_the_room_entries_that_cost_the_team(tmp_path):
    harvest = tmp_path / 'harvest' / 'darknut'
    harvest.mkdir(parents=True)
    for name in ['d3first_3_59n_h5_7', 'd3first_3_59n_h6_7', 'd3first_3_5bw_h5_9']:
        (harvest / f'{name}.state').write_bytes(b'')
    room = {'level': 3, 'room': '0x59', 'health_out': 1.0, 'outcome': 'exit-correct'}
    runs = [{'seed': 5, 'rooms': [{**room, 'index': 7, 'health_in': 2.0},
                                  {**room, 'room': '0x5b', 'index': 9, 'health_in': 1.0}]},
            {'seed': 6, 'rooms': [{**room, 'index': 7, 'health_in': 1.0, 'health_out': 0.0, 'outcome': 'death'}]}]
    chains = tmp_path / 'harv.jsonl'
    chains.write_text(''.join(json.dumps(r) + '\n' for r in runs))
    states = tmp_path / 'states'
    manifest = bench.make_rooms_manifest(tmp_path / 'harvest', [str(chains)], 12, 5, 380000, 0.1, states)
    entry, = manifest['entries']                  # 0x5b lost nothing: out
    assert entry['key'] == 'darknut@d3first' and entry['name'] == 'darknut@d3first 3_59n' and entry['arrived']
    assert entry['team_visits'] == 2 and entry['team_deaths'] == 1 and entry['team_hearts_lost'] == 1.0
    assert entry['states'] == [f'{bench.ROOMS}/darknut/d3first_3_59n_h5_7', f'{bench.ROOMS}/darknut/d3first_3_59n_h6_7']
    assert (states / bench.ROOMS / 'darknut' / 'd3first_3_59n_h6_7.state').is_file()


def test_the_committed_manifest_covers_the_team():
    manifest = bench.load_manifest()
    assert manifest['act'] == registry.DEFAULT_ACT and len(manifest['entries']) == 51
    assert manifest['full']['segments'][0].startswith('full:')
