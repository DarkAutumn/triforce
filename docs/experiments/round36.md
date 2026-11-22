# Round 36: fewer networks, toward one

## One network by imitation

(results pending: see the end of this section)

## The wood-sword pairs

**Question.** Can one network of today's architecture play both of a family's runs (the default run and its
`@d3first`, wood-sword run), so 51 networks become 34? Six families first: darknut, goriya, zol, boss-gleeok,
ow-octorok, wallmaster.

**Which teacher to start from.** Each teacher played on the other's rooms (success; the bench's entries of a paired
family are its harvested `d3first_*` states, the wood-sword rooms; the default run's rooms are the route's seed
states of the steps after the magic sword, or for goriya and wallmaster, which the route no longer plays after the
sword, the family's training states: `scripts/experiments/r36_merge_pairs.py`, 10 episodes a state):

| family | default run: own rooms / wood-sword rooms | `@d3first` run: own rooms / default's rooms | started from |
|---|---:|---:|---|
| darknut | 0.933 / 0.833 | 0.933 / 0.900 | `@d3first` |
| goriya | 0.972 / 0.925 | 0.983 / 0.975 | `@d3first` |
| zol | 1.000 / 0.900 | 1.000 / 1.000 | `@d3first` |
| boss-gleeok | 1.000 / 0.125 | 0.675 / 0.167 | default |
| ow-octorok | 1.000 / 0.983 | 1.000 / 1.000 | `@d3first` |
| wallmaster | 0.858 / 0.733 | 0.942 / 0.908 | `@d3first` |

For five of six the wood-sword run already plays the default run's rooms about as well as the default run; the
Gleeoks are two different fights (Dungeon 4's and Dungeon 8's) and neither run plays the other's.

**Training.** One run a family, both tasks at once (4 envs each), PPO with both teachers' KL at weight 1.0 for 20
iterations fading to 0 over 60, 120 iterations (491,520 steps, 11-17 minutes on 8 envs):

```
python -m triforce.lab.train --name pair-FAM --run-id r36m-FAM --families FAM FAM@d3first=<the part's yaml> \
    --parallel 8 --iterations 120 --load <start>/ckpt.pt --teacher-coeff 1.0 --teacher-hold 20 --teacher-fade 60 \
    --seed 36 [--no-curriculum]
```

The part's yaml is the old run's `run.json` `specialist` (archived under main's `target/worktrees/`). boss-gleeok
needed `--no-curriculum`: the lab gives a route part its family's curriculum trajectories, and the family's
trajectories are of Dungeon 8's Gleeok room, which the part's task (Dungeon 4's) refuses.

**Bench, paired against each teacher on that teacher's rooms** (student - teacher, success in points; the
wood-sword rooms: the bench, 120 episodes; the default run's rooms: 5 episodes a state, the bench's seeds, episodes
in brackets; `docs/experiments/data/r36-merge/pairs-bench.json`):

| family | wood-sword rooms (vs `@d3first`) | default's rooms (vs default) | on the bench |
|---|---:|---:|---|
| darknut | -2.5 ± 2.8 | -3.3 ± 4.1 (60) | holds |
| goriya | -0.8 ± 0.8 | +0.6 ± 0.8 (540) | holds |
| zol | -0.8 ± 0.8 | 0.0 (30, all won) | holds |
| ow-octorok | 0.0 (all won) | 0.0 (45, all won) | holds |
| wallmaster | -5.8 ± 3.6 | +5.0 ± 6.0 (60) | not shown to hold |
| boss-gleeok | -24.2 ± 5.8 | -6.7 ± 6.7 (15) | fails |

**Chains** (1,000 runs each, seeds 0-999, `--act model`, clips on; the merged runs of the four that held on the
bench given for both keys, `--checkpoint FAM=CKPT --checkpoint FAM@d3first=CKPT`; today's team on the same seeds;
`scripts/experiments/r36_merge_chains.py`, `docs/experiments/data/r36-merge/pairs-chains.json`):

| segment | today's team | merged | paired difference | where it changes |
|---|---:|---:|---:|---|
| Dungeon 3 (`--from d3first_3_7cs_0 --stop-at step:24`; route runs: darknut, zol merged) | 23.2% | 18.5% | **-4.7 ± 1.8** | darknut's 0x5b: 198 deaths in 1,580 visits against 104 in 1,651; zol's 0x49: 71 / 915 against 41 / 951 |
| Dungeon 6 (`--from d3first_6_79s_0 --stop-at step:227`; default runs: zol merged) | 35.2% | 36.8% | +1.6 ± 1.9 | none beyond noise |

(Today's team on Dungeon 3 reproduces the chain of record, `r35land-1k-d3`, at 232 / 1,000.)

**Verdict.** No pair is shown to hold where it matters: the bench passes four of six, but Dungeon 3's chain, where
the wood-sword halves of darknut and zol play at the hearts the route brings, loses 4.7 points, and the extra deaths
sit in two rooms that already kill (Darknut 0x5b: +94 deaths; Zol 0x49: +30; later rooms see fewer runs). 120 iterations of PPO
from the wood-sword run, with the default run's rooms added and the teachers fading, blunt exactly the hardest
rooms; the bench's 120 episodes cannot see a 3-6 point loss in one room (its standard error is 2-3 points a
family). ow-octorok and goriya play no room of the two chains; their bench differences are within 1 point. The
Gleeoks are two fights; one network of this size and this training does not hold both (Dungeon 4's falls from
67.5% to 43.3%). Next for the pairs: keep the teacher term on (no fade) or fade it only once the student matches
both teachers, weight the envs toward the part whose rooms are hard (`--shares`), and judge on Dungeon 3's chain.
