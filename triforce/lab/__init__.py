"""The lab: machinery for model experiments (docs/lab.md).

- ``collect``: rollouts over several families at once; teachers labelling the student's rows.
- ``train``: ``python -m triforce.lab.train``: one network over several families, PPO with a fading teacher term,
  or imitation alone.
- ``record``: teacher-labelled datasets for offline work.
- ``bench``: the yardstick (a cheap fixed suite of harvested room entries per family) and its comparison.

The network classes themselves are ``triforce.nets``.
"""
