"""The yardstick (``triforce.lab.bench``): state choice, a run on a small manifest, paired comparison."""

import json

from triforce import registry
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


def test_a_bench_run_and_its_paired_comparison(states_dir, tmp_path):
    manifest = {'format': 1, 'act': registry.DEFAULT_ACT, 'episodes': 2, 'seed': 7, 'clips': 'on',
                'entries': [{'key': 'ow-start', 'family': 'ow-start', 'source': 'seed', 'states': ['start']}]}
    paths = [registry.save_checkpoint('ow-start', f'n{seed}', random_network(seed), root=tmp_path)
             for seed in (1, 2)]
    results = [bench.run_bench({'ow-start': str(p)}, manifest, 2, states_dir=states_dir, log=lambda _: None)
               for p in paths]
    row = results[0]['keys']['ow-start']
    assert row['episodes'] == 2 and len(row['records']) == 2
    assert {'success', 'hearts_lost', 'hits_per_visit', 'decisions', 'bombs'} <= set(row)
    same = bench.compare(results[0], results[0])
    assert same['total']['pairs'] == 2 and same['total']['success'] == (0.0, 0.0)
    other = bench.compare(results[0], results[1])
    want = results[1]['keys']['ow-start']['decisions'] - row['decisions']
    assert abs(other['keys']['ow-start']['decisions'][0] - want) < 1e-9
    assert '**total**' in bench.format_compare(other)
    json.dumps(results[0])


def test_the_committed_manifest_covers_the_team():
    manifest = bench.load_manifest()
    assert manifest['act'] == registry.DEFAULT_ACT and len(manifest['entries']) == 51
    assert manifest['full']['segments'][0].startswith('full:')
