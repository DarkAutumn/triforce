"""triforce.variety: the risk and variety measures of round 23."""

import itertools
import os
import random

import pytest

from triforce.variety import decisions_of, mean_pair_prefix, risk, sequence, variety


def visit(n, low1=0, low5=0, low1_hit=0, low5_hit=0, low1_death=0, low5_death=0, actions='', entropy=0.0):
    return {'decisions': {'n': n, 'low1': low1, 'low5': low5, 'low1_hit': low1_hit, 'low5_hit': low5_hit,
                          'low1_death': low1_death, 'low5_death': low5_death, 'surprisal': n * 0.5,
                          'act_entropy': entropy, 'actions': actions}}


def test_risk_sums_per_unit_and_skips_records_without_decisions():
    episodes = [[decisions_of(visit(10, 1, 3, 1, 2, actions='00' * 10, entropy=5.0)), decisions_of(visit(10))],
                [decisions_of(visit(20, 0, 1, 0, 0, 0, 0, entropy=2.0))],
                [decisions_of({'steps': 3})]]          # an older record: not a unit
    r = risk(episodes, 'episode')
    assert (r['episodes'], r['decisions'], r['low1'], r['low5'], r['low5_hit']) == (2, 40, 1, 4, 2)
    assert r['low5_per_episode'] == 2.0 and r['low5_hit_rate'] == 0.5 and r['low1_hit_rate'] == 1.0
    assert r['low1_death_rate'] == 0.0 and r['act_entropy_per_decision'] == pytest.approx(7 / 40)
    assert r['surprisal_per_decision'] == pytest.approx(0.5)
    assert r['hit_rate'] is None, "records without the base counts have no base rate"
    assert risk([[None]], 'run') == {}
    based = risk([[{**decisions_of(visit(10, 1, 2, 1, 1)), 'hit': 4, 'death': 1}]], 'run')
    assert (based['hit_rate'], based['death_rate'], based['low5_hit_rate']) == (0.4, 0.1, 0.5)


def test_the_mean_pair_prefix_is_every_pairs_common_prefix_averaged():
    rng = random.Random(3)
    for _ in range(50):
        seqs = [''.join(rng.choice(['00', '01', '02']) for _ in range(rng.randint(0, 6))) for _ in range(rng.randint(2, 9))]
        brute = [len(os.path.commonprefix([a, b])) // 2 for a, b in itertools.combinations(seqs, 2)]
        assert mean_pair_prefix(seqs) == pytest.approx(sum(brute) / len(brute))
    assert mean_pair_prefix(['00']) is None
    # Hex digits pair up into decisions: '0102' and '0103' share one decision, not three digits.
    assert mean_pair_prefix(['0102', '0103']) == 1


def test_variety_counts_distinct_plays_the_divergence_and_the_frames_spread():
    v = variety(['0a0b', '0a0b', '0a0c', None], [100, 120, 140, 160])
    assert (v['distinct'], v['distinct_rate'], v['diverge_all']) == (2, 2 / 3, 1)
    assert v['diverge_mean'] == pytest.approx((2 + 1 + 1) / 3)
    assert v['frames_mean'] == 130 and v['frames_p10'] == pytest.approx(106) and v['frames_p90'] == pytest.approx(154)
    assert v['frames_sd'] == pytest.approx(25.819888974716)
    assert variety([None], [5]) == {'frames_mean': 5.0, 'frames_sd': 0.0, 'frames_p10': 5.0, 'frames_p90': 5.0}


def test_a_sequence_joins_visits_and_needs_every_visit_recorded():
    assert sequence([visit(1, actions='0a'), visit(2, actions='0b0c')]) == '0a0b0c'
    assert sequence([visit(1, actions='0a'), {'steps': 2}]) is None
