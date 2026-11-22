# Round 36: fewer networks, toward one

## One network by imitation

**Question.** How large must one network be to copy all 51 of today's networks (34 families, 17 wood-sword
runs)? Supervised, offline, on the GPU: no PPO.

**Data** (`scripts/experiments/r36_merge_record.py`): every key's teacher acting on 10 envs, every row labelled
with the teacher's joint distribution over the 352 actions and its value. Two sources a key: its training resets
(its yaml, its own hp and item randomisation; a route part's own yaml, from its old run's `run.json`) and the
bench's distribution (the family's harvested pre-entry states, the hearts the runs bring, less every state the bench
plays; no randomisation; each state labelled by the network that plays its route step: the `@d3first` run before
the magic sword). Seeds 36,100,000 on, apart from the bench's. 81 sources (4 families without harvests, pols-voice
all on the bench); a source of 20,480 decisions costs about 40-60 s of queue on the loaded machine (81,920: about
3.4 min). Training uses 20,480 decisions a source: 1.66M rows (26 sources hold 81,920, unused beyond the first
20,480). Held out: env 9 of every rollout (whole episodes, about 10%).

**Training** (`scripts/experiments/r36_merge_distill.py train`): KL(teacher ‖ student) over the joint plus 0.02 ·
½(V − V_teacher)² (values about 15 ± 5), Adam 1e-3 cosine, batch 4,096, 20 epochs (about 7,300 steps; 14 minutes
for the 1x network on the shared 4090). A pilot on five keys: lr 1e-3 reaches held-out KL 0.12 where 3e-4 is at
0.20 after the same steps.

**The 1x student (today's architecture, 866,084 parameters, `r36merge/s1x`)**, held out
(`docs/experiments/data/r36-merge/distill-s1x.json`): KL 0.160 (train 0.123: little overfitting, capacity- or
optimisation-bound); top-1 agreement with the teacher's most likely action 0.836 for the joint, 0.963 kind, 0.953
direction, 0.884 duration (where the head has a choice). Worst keys by KL: like-like 0.57 (joint 0.67), darknut
0.46, ow-zora 0.41, boss-manhandla 0.39, vire 0.35, goriya 0.35, boss-manhandla@d3first 0.33, zol 0.32,
goriya@d3first 0.31, darknut@d3first 0.30; best: lanmola 0.008, ow-start and dodongo-crossing 0.014, ow-peahat 0.03,
trap-room 0.05. The hard ones to copy are the fights with long, timed exchanges and the families whose two runs
disagree on the same rooms (darknut, goriya, zol and manhandla appear with both keys).

**The 1x student on the bench** (`r36_merge_report.py`, 51 slots, each teacher on the rooms it plays, paired on the
bench's seeds; `docs/experiments/data/r36-merge/student-s1x-slots.json`): mean success over slots 0.945 against
0.962, **98.1% kept (-1.8 ± 0.3 points)**; pooled over 5,675 episodes -1.6 ± 0.3 points, hearts lost +0.09 ± 0.01
an episode; hits a visit 0.69 against 0.60. That beats the default-runs-only baseline below (-3.1). The worst ten
slots keep 88%: boss-gleeok@d3first -31.7 ± 5.8, darknut (Dungeon 8) -23.3 ± 5.5, wallmaster@d3first -12.5 ± 3.7,
darknut@d3first -10.0 ± 3.4, like-like@d3first -8.3 ± 3.3, boss-manhandla@d3first -6.7 ± 5.1, then -3 points
(aquamentus, digdogger, ganon, vire). Every one of the large losses comes with more hearts lost (+0.3 to +2.2 an
episode): they are fights (deaths), not objectives; the overworld and utility rooms are copied within 3 points.

**Not finished in this session** (the successor's status names the commands): the larger sizes' benches
(`bench run --network models/r36merge/ID/ckpt.pt --manifest target/merge/student-manifest.json`, a manifest of
the bench's 34 family entries plus the 17 default runs' rooms, `r36_merge_pairs.py --combined`; judged slot by slot
by `r36_merge_report.py` against the 51 teachers); the sizes (wide 512x3 2.4x, 768x3 4.6x, 1024x3 7.5x, 768x3 with
two entity layers, 256x2 with and without the room embedding) and the learning curve (1/4, 1/2 of the data at 1x)
are training one after another on the GPU (`target/merge/gpu-a.sh`); DAgger rounds and the failure analysis
(`r36_merge_failures.py`: objective failures, wrong exits, against fights, deaths) follow the benches.

**Baseline for the slot report:** the 34 default runs alone (each family's default run also playing its
wood-sword rooms) keep 96.8% of the 51 slots' mean success (-3.1 ± 0.2 points): a single student has to beat that to
say more than "drop the route parts".

## One network on the route

**Question.** Does ONE network, the 1x student (`r36merge/s1x`, 866,084 parameters, 14 minutes of copying), play
the whole of `d3first` in every slot of today's 51 (promoted for all 34 families and as `promoted.d3first` for the
17 route parts; a models root of its own, the teachers' untouched; its act mode the project's default,
`temperature:0.5+relax:16`, which is every teacher's)?

**The golden race from power on** (`zelda-headless.py OUT 3 --golden --power-on --clips on --act model
--parallel 8`; replays `tf replay verify` ok, read with `scripts/experiments/r32team_golden.py`; every one of the
51 model slots in the replay's header has the same weights): **one network beats the game.** At the default 320
attempts a segment the first two paths stop in segment 3 (Dungeon 1's triforce to Dungeon 5's: the overworld after
Dungeon 1 and Dungeon 5), 320 of 320 tries failing; the third wins (1,018 runs in all, 174,345 frames). With
`--attempts 2000`, all three paths win (2,332 runs):

| path (first seed) | D3 | D4 | D1 | post-D1 + D5 | D2 | D7 | D6 | D8 | D9 + Ganon | runs | frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 13 | 34 | 2 | 1,314 | 3 | 21 | 8 | 75 | 56 | 1,526 | 152,305 |
| 18,000 | 11 | 69 | 1 | 275 | 1 | 5 | 12 | 118 | 10 | 502 | 160,041 |
| 36,000 | 6 | 29 | 2 | 134 | 3 | 7 | 16 | 53 | 54 | 304 | 152,766 |
| 5,760 (320 attempts) | 8 | 55 | 1 | 112 | 3 | 10 | 8 | 109 | 12 | 318 | 174,345 |
| today's 51 networks (round 35's landing, 320 attempts; three paths) | 7 / 5 / 9 | 5 / 25 / 3 | 1 / 1 / 1 | 14 / 14 / 9 | 1 / 1 / 1 | 3 / 5 / 5 | 7 / 41 / 5 | 8 / 7 / 8 | 4 / 6 / 12 | 50 / 105 / 53 | |

Cheats in the winning replays (the route's rules): a wall bomb at 3:0x5b, 1:0x53 and a few of Dungeons 2, 6-9
(1-5 a path), the 0x0c cave's rupees (8-16). Outside segment 3 the one network needs the tries of today's team,
give or take a factor of two (Dungeon 8 and Level 9 about 5-10x); segment 3 costs 10-100x: that segment is where
copying loses (the chains below say which rooms).
**Every number in this round is on 2.0.0.3's `d3first` (the old order's states)**, not the corrected route of
2.0.0.5.

**Chains, segment by segment** (1,000 runs each, seeds 0-999, `--act model`, clips on, the landings' stops; D2
to step 155, D6 to 227, the overworld segments to the next dungeon's entrance; `target/merge/one-chains.sh`;
today's team re-run on the same seeds where the record was on other route data; table by
`scripts/experiments/r36_one_table.py`, `docs/experiments/data/r36-merge/one1x-chains.json`):

| segment | 51 networks | 1 network | paired | rooms that differ most (runs lost per 1,000 entering: team -> one) |
|---|---:|---:|---:|---|
| D3 | 23.2% | 3.7% | -19.5 ± 1.5 | Darknut 3:0x5b 117 -> 580, 0x59 68 -> 287, 0x69 126 -> 271; Manhandla 527 -> 673 |
| post-D3 | 100.0% | 99.9% | -0.1 ± 0.1 | |
| D4 | 18.5% | 2.8% | -15.7 ± 1.3 | Gleeok 4:0x13 736 -> 928; like-like 0x32 58 -> 210; vire 0x12 68 -> 204 |
| post-D4 | 100.0% | 99.8% | -0.2 ± 0.1 | |
| D1 | 92.7% | 80.4% | -12.3 ± 1.5 | Aquamentus 1:0x35 34 -> 76; goriya 0x44 35 -> 65 |
| post-D1 | 99.8% | 97.8% | -2.0 ± 0.5 | tektite 0:0x0c 2 -> 8 |
| D5 | 7.0% | 0.3% | -6.7 ± 0.8 | darknut 5:0x05 811 -> 992 |
| post-D5 | 98.7% | 81.2% | -17.5 ± 1.3 | **wrong exit**: 280 runs end in 0:0x1b, a room the team never enters (314 lost / 1,000 entering); moblin 0x62 12 -> 85 |
| D2 | 100.0% | 99.9% | -0.1 ± 0.1 | |
| post-D2 | 99.9% | 97.4% | -2.5 ± 0.5 | **wrong exit**: 57 runs into 0:0x4c (never entered by the team) |
| D7 | 42.3% | 28.9% | -13.4 ± 2.1 | Digdogger 7:0x1c 154 -> 244; goriya 0x1b 148 -> 205 |
| post-D7 | 100.0% | 99.4% | -0.6 ± 0.2 | |
| D6 | 35.2% | 19.7% | -15.5 ± 2.0 | wizzrobe 6:0x3a 406 -> 596 |
| post-D6 | 100.0% | 99.8% | -0.2 ± 0.1 | |
| D8 | 37.8% | 2.9% | -34.9 ± 1.6 | Blue Darknuts 8:0x3e 280 -> 782, 0x5e 3 -> 127 |
| D9 | 48.9% | 24.3% | -24.6 ± 2.1 | wizzrobe 9:0x10 139 -> 290 |
| from the start | 0.1% | 0.0% | -0.1 ± 0.1 | (the team wins 1 of 1,000) |

**Verdict.** One network beats the game by a golden path, but it is a much weaker player than the team: the
bench's 98.1% hides losses that compound over a dungeon's ten fights (D8 -35, D9 -25, D3 -20 points). The losses
are fights (deaths in the rooms that already kill: Darknuts, Wizzrobes, Gleeok, Manhandla, Digdogger) except two
overworld segments, where they are objectives (wrong exits into rooms the team never enters: 0x1b after D5, 0x4c
after D2). The bench shortlists; only chains decide.

## The size curve

Same data (1.49M rows), same recipe (20 epochs; the wide students converted to `ZeldaPolicy` of the same shape by
`scripts/experiments/r36_merge_convert.py`, logits identical, so the Rust actor plays them). Bench = the 51 slots,
paired (`r36_merge_report.py`; reports in `docs/experiments/data/r36-merge/student-*-report.json`).

| student | parameters | held-out KL | joint top-1 | bench kept (diff) | worst ten kept |
|---|---:|---:|---:|---:|---:|
| 256x2 | 733k | 0.171 | 0.829 | not benched | |
| 256x2, no room embedding | 684k | 0.172 | 0.828 | not benched | |
| **1x (today's)** | 866k | 0.160 | 0.836 | 98.1% (-1.8 ± 0.3) | 88.4% |
| 512x3 | 2.09M | 0.130 | 0.857 | queued | |
| 768x3 | 3.97M | 0.119 | 0.866 | queued | |
| 768x3, two entity layers | 4.01M | 0.116 | 0.869 | 99.0% (-1.0 ± 0.3) | 92.9% |
| 1024x3 | 6.51M | 0.113 | 0.873 | 98.6% (-1.4 ± 0.3) | 90.9% |
| 1x on 1/4 of the data | 866k | 0.230 | 0.808 | queued | |
| 1x on 1/2 of the data | 866k | 0.179 | 0.827 | queued | |

Data matters more than size: halving the data costs as much KL (+0.019) as shrinking the towers, and doubling
it (1/2 -> all) gains more than 7.5x the parameters. The room embedding does nothing to held-out agreement
(0.171 / 0.172); where the two differ on the route is not measured.

## The fights

**A head a boss** (`triforce/nets/roomheads.py`: the shared body of the 1x student, one action head a boss
family, the head picked by a fixed room-to-head table inside the network, `scripts/experiments/r36_merge_rooms.py`,
10 heads; 1.61M parameters; same data and recipe): held-out KL 0.146; on the bosses' bench (paired, 1,185
episodes; `boss-heads-bench.json`) +0.8 ± 0.9 pooled, Gleeok with the wood sword 0.358 -> 0.425 (+6.7 ± 6.0;
its teacher about 0.68), Manhandla 0.642 -> 0.692 (+5.0 ± 4.7). The heads do not close Gleeok.

**The correction copying needs (DAgger).** Round 1: the 1x student acts, the teachers label (51 keys, 20,480
decisions a source, seeds 36,200,000 on), trained from the student on old + new rows (5 epochs, lr 3e-4):
`r36merge/s1x-d1`, held-out KL 0.156 (from 0.160). Its chains (D3, D4, D5, D8, D1, D7) and round 2 were in the
queue when this session ended (the status file has the commands).


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
