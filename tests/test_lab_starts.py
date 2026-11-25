"""Round 42: what a lab task draws. With clips on in every episode the seed states the route enters only on a plain
way are left out (they fix the clips switch off); a harvested room entry under ``MIN_HARVESTED_ENTRY`` states is not
drawn while a larger one is."""

from pathlib import Path

from triforce import _native, inputs
from triforce.lab.collect import plain_way_only, route_only, task_yaml
from triforce.specialist import load_specialist

from .conftest import REPO_STATES_DIR, require_rom

# Round 44: d3first has one state per door (chain 0).
PLAIN_ONLY = {'vire': ['4_20s_0', '4_30e_0'],
              'stalfos-keese-gel': ['4_21w_0']}


def test_plain_way_seed_states_are_left_out_with_clips_on(capsys):
    require_rom()
    names = [p.stem.removeprefix('d3first_') for p in REPO_STATES_DIR.glob('d3first_*.state')]
    for family, left_out in PLAIN_ONLY.items():
        config = load_specialist(task_yaml(f'{family}@d3first')).replace(curriculum={'enabled': False})
        assert config.clips.off_share == 0
        on = route_only(config, 'd3first', REPO_STATES_DIR)
        seeds = {Path(p).stem for p in on.resolve_start_states(str(REPO_STATES_DIR))}
        assert seeds and not seeds & {f'd3first_{n}' for n in left_out}, seeds
        assert 'd3first_4_31e_0' in seeds or family != 'vire'      # 4:0x31 from E is on both ways: kept
        assert ', '.join(left_out) in capsys.readouterr().out
        # A task that trains some episodes clips off keeps them.
        off = config.replace(clips={'enabled': True, 'off_share': 0.5})
        assert plain_way_only(off, 'd3first', names) == []
        assert plain_way_only(config, 'd3first', names) == left_out


def test_small_harvested_entries_are_named_while_a_larger_one_exists():
    big = [f'/h/darknut/d3first_3_5bs_h{i}_3.state' for i in range(_native.MIN_HARVESTED_ENTRY)]
    small = [f'/h/darknut/d3first_3_5bn_h{i}_15.state' for i in range(5)]
    assert inputs.small_harvested_entries(big + small) == {'d3first_3_5bn': 5}
    assert inputs.small_harvested_entries(small) == {}
