"""triforce.bomb_metrics: the owner's two bomb metrics summed over window lists (Nov 23)."""
from triforce import bomb_metrics


def window(metric, outcome, room='3:0x5b', seed=0, full=False, drop=None, killed=(11, 3)):
    return {'metric': metric, 'outcome': outcome, 'open_frame': 1, 'open_room': room, 'close_frame': 2,
            'close_room': room, 'killed_type': killed[0] if killed else None,
            'killed_slot': killed[1] if killed else None, 'full_bag': full, 'drop': drop, 'seed': seed,
            'replay': f'{seed}.tfreplay'}


def test_summary_splits_full_bag_and_sorts_rooms():
    windows = [
        window('tenth', 'armed', drop='taken'), window('tenth', 'other', full=True), window('tenth', 'other'),
        window('tenth', 'lost'), window('carrier', 'hit', drop='fell'),
        window('carrier', 'miss', room='6:0x78'), window('carrier', 'left', room='6:0x78', killed=None),
        window('carrier', 'left', room='6:0x78', seed=1, killed=None),
        window('tenth', 'missed', seed=1, full=True), window('tenth', 'suspended', room='2:0x0e', seed=1),
    ]
    s = bomb_metrics.summarize(windows)
    a, b = s['tenth'], s['carrier']
    assert (s['runs'], a['opportunities'], a['armed'], a['missed'], a['other']) == (2, 4, 1, 1, 2)
    assert {k: a['not_full'][k] for k in ('opportunities', 'armed', 'missed', 'other')} == {
        'opportunities': 2, 'armed': 1, 'missed': 0, 'other': 1}
    assert (a['lost'], a['suspended'], a['forced']['taken']) == (1, 1, 1)
    assert (b['opportunities'], b['hit'], b['miss'], b['left'], b['bombs']['fell']) == (4, 1, 1, 2, 1)
    assert [r[0] for r in s['rooms']] == ['3:0x5b', '6:0x78']
    assert s['per_run']['1']['tenth']['missed'] == 1
    text = bomb_metrics.markdown(s)
    assert 'without a full bag' in text and '| 3:0x5b | 4 | 1 | 1 | 2 | 2 | 1 | 1 | 0 | 0 |' in text


def test_replay_facts_read_the_rule_not_the_moment():
    class Replay:
        @staticmethod
        def targets():
            return [{'frame': 10, 'objective': 'FIGHT'}, {'frame': 30, 'objective': 'MOVE'}]

        @staticmethod
        def masks():
            return [{'frame': 10, 'blocked': {}}, {'frame': 20, 'blocked': {'BOMB': ['tenth']}},
                    {'frame': 30, 'blocked': {'BOMB': ['never']}}]

    assert bomb_metrics.replay_facts(Replay) == [(10, True, True, False), (20, True, False, False),
                                                 (30, False, False, True)]
