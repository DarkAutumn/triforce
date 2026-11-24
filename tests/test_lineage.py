"""Run lineage (round 40, ``triforce/lineage.py``): each promoted pointer's chain of parents back to its origin,
from round 40's identities and from the paths and names older runs recorded."""

import json

from triforce import lineage


def _run(models, family, run, record):
    directory = models / family / run
    directory.mkdir(parents=True)
    (directory / 'run.json').write_text(json.dumps(record))


def test_chains_back_to_the_origin(tmp_path, capsys):
    models = tmp_path / 'models'
    _run(models, 'zol', 'scratch', {'args': {'load': None}, 'parent': None})
    _run(models, 'zol', 'child', {'args': {'load': '/x/models/zol/scratch'},
                                  'parent': {'family': 'zol', 'run': 'scratch', 'iterations': 5, 'sha1': 'ab'}})
    # Before round 40: the resolved path in `source`, a snapshot of a run in another family.
    _run(models, 'zol', 'old', {'args': {'load': 'gibdo'},
                                'source': '/home/u/git/tree/models/zol/child/checkpoints/it000012/ckpt.pt'})
    _run(models, 'gibdo', 'named', {'args': {'load': 'gibdo'}})
    _run(models, 'gibdo', 'itself', {'args': {'load': None}, 'source': '/t/models/gibdo/itself'})
    _run(models, 'gibdo', 'orphan', {'args': {'load': '/t/models/gibdo/gone/ckpt.pt'}})
    for family, pointer, run in [('zol', 'promoted', 'old'), ('zol', 'late.d3first', 'scratch'),
                                 ('gibdo', 'promoted', 'named'), ('gibdo', 'promoted.d3first', 'orphan'),
                                 ('gibdo', 'promoted.d2first', 'itself')]:
        (models / family / pointer).write_text(run + '\n')
    (models / 'zol' / 'promoted-eval.json').write_text('{}')        # evidence, no pointer

    found = lineage.Lineage(models, backup=None)
    assert lineage.chain(found, 'zol', 'old') == ([('zol', 'old'), ('zol', 'child'), ('zol', 'scratch')], 'scratch')
    assert lineage.chain(found, 'gibdo', 'named')[1].startswith("lost: loaded 'gibdo'")
    assert lineage.chain(found, 'gibdo', 'orphan') == ([('gibdo', 'orphan'), ('gibdo', 'gone')],
                                                       'lost: run dir not kept')
    assert lineage.chain(found, 'gibdo', 'itself')[1].startswith('lost: continued in its own run dir')
    assert {(f, p) for f, p, _ in lineage.pointers(models, 'd3first')} == {
        ('gibdo', 'promoted'), ('gibdo', 'promoted.d3first'), ('zol', 'late.d3first'), ('zol', 'promoted')}

    assert lineage.main(['--models-dir', str(models), '--backup', str(tmp_path / 'none')]) == 0
    out = capsys.readouterr().out
    assert 'old <- child <- scratch <- (scratch)' in out
    assert out.strip().endswith('5 pointers: 1 trained from scratch, 4 continuations (1 traced to a from-scratch '
                                'origin, 3 with the origin lost), up to 2 deep')
