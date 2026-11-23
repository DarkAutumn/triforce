"""triforce.bomb_metrics: the owner's two bomb metrics summed over chain records (Nov 23)."""
from triforce import _native, bomb_metrics


def room(level, number, specialist='darknut', **counts):
    record = {'level': level, 'room': number, 'in_cave': False, 'specialist': specialist}
    record.update(counts)
    return record


def test_fields_are_the_native_ones_and_the_runner_reads_them():
    from triforce.runner import RoomRecord
    assert bomb_metrics.FIELDS == tuple(_native.BOMB_WINDOW_FIELDS)
    assert all(name in RoomRecord.__dataclass_fields__ for name in bomb_metrics.FIELDS)


def test_summary_splits_full_bag_and_sorts_rooms():
    records = [
        {'seed': 0, 'tenth_open_at_end': True, 'rooms': [
            room(3, '0x5b', tenth_armed=1, tenth_other=2, tenth_full_other=1, carrier_hits=1, carrier_bombs=1),
            room(6, '0x78', 'wizzrobe', carrier_miss_kill=2, carrier_miss_leave=1)]},
        {'seed': 1, 'rooms': [room(3, '0x5b', tenth_missed=1, tenth_full_missed=1), {'level': 0, 'room': '0x77'}]},
    ]
    s = bomb_metrics.summarize(records)
    a, b = s['tenth'], s['carrier']
    assert (a['opportunities'], a['armed'], a['missed'], a['other'], a['full_bag']) == (4, 1, 1, 2, 2)
    assert a['not_full'] == {'opportunities': 2, 'armed': 1, 'missed': 0, 'other': 1}
    assert (b['opportunities'], b['hits'], b['miss_kill'], b['miss_leave'], b['bombs']) == (4, 1, 2, 1, 1)
    assert s['open_at_end'] == {'tenth': 1, 'carrier': 0}
    assert [r[0] for r in s['rooms']] == ['3:0x5b', '6:0x78']
    assert s['per_run'][0] == [0, 3, 1, 1, 4, 1, 1]
    text = bomb_metrics.markdown(s)
    assert 'without a full bag' in text and '| 3:0x5b | darknut |' in text
