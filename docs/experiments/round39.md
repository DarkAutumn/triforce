# Round 39

## Beams at range: the weights, a beam head, the hearts, and Dungeon 3 (branches `r39-beam`, `r39-beam-head`)

The owner's note of Nov 22 (backlog "Beams at range, not melee"): "The model is super happy to walk up to enemies and
just hit them up close. [...] he doesn't use the full effect of full hearts, and instead gets close and loses beams."
Round 38 built the reward (`reward-beam-kill`, `reward-beam-timed`, `penalty-beam-unfired`) and measured the networks
walking into melee. This round: the beam weights swept on wizzrobe (the family melee costs most), a supervised beam
head played like `hitshift` (everywhere, then only in the rooms the route fights), darknut and goriya with it; how
often Link even has the beams; and, after the owner's correction of Nov 23, Dungeon 3, where he has them and still
fights up close.

**Verdict.** Nothing promoted. Two causes, both real. (1) **Hearts**: wizzrobe enters 31% of its
`d6` visits and 9.5% of its `d9` visits with full hearts, the only time a shot exists; the hearts it lacks were lost
to hits in the rooms before (its own and Patra's), not left on the floor, so for wizzrobe and Patra the lever is
fewer hits (a hit head, as 2.0.0.8 gave eight families). (2) **Melee by choice**: in Dungeon 3 Link enters 3:0x6b
full 93% of the time and 3:0x4a 72%, and still makes 83% and 93% of his sword kills up close, walking round 0x6b's
blocks to each Gel and Keese. A beam head (a supervised "a shot fired now would hit" read, raising BEAMS at play)
makes him fire when lined up (unfired 54-79% → 4-29%, far kills ×1.5-2.6, rooms 4-7% shorter) but does not make him
stand back and line a shot up: three kills in four stay in melee. Gated to the rooms the route fights it is
harmless (6:0x78's crossing no longer turns into a fight); ungated it cost wizzrobe 5-9 points. The reward arms
did not carry from the rooms tier to the chains. What would change the walk is training with the shot in the
policy (below, "Next"). The tools land: `beamshift` with its objective gate, `r39_beam.py` (`bench`, `chains`,
`hearts`, `combine`), the `beam_*` bench fields, the `beam_lined` labels.

### Who has the beams

A shot needs full hearts (within half a heart). The team's 2.0.0.11 chains, seeds 0-299 on the branch (`d6` to
0:0x22 41.7%, `d9` to the end 55.7%, run for run 2.0.0.8's first 300 of `r38-208-1k`): wizzrobe enters 30.7% of its
`d6` visits and 9.5% of its `d9` visits with full hearts (median 7 hearts short on `d9`); on 2.0.0.8's four
1000-run records (`d2`, `d6`, `d9`, `full`) wizzrobe 16.2% of 17,112 visits, darknut 39.7%, goriya 29.1%. The rooms
that cost wizzrobe its runs are entered short: 9:0x05 (2 of 293 visits full), 9:0x10 (4 of 543), 9:0x30 (none),
9:0x17 (67 of 589), 6:0x3a (19 of 250); 6:0x38 (99 of 300) and 6:0x78 (all, the Reds' crossing) are where beams can
change anything.

### The reward weights (branch `r39-beam`)

The rooms tier now records the sword shots (`lab.bench` `beam_*` episode fields: far kills, kill distance, shots,
lined-up decisions left unfired). Wizzrobe from `r33w-s93`, 2M steps each, kill/timed/unfired weights:

| arm | weights | bench success | bench hearts lost | unfired | `d6` (41.7%) | `d9` (55.7%) |
|---|---|---:|---:|---:|---:|---:|
| w2 | 1.0 / 0.2 / 0.05 | −0.6 ± 0.9 | −0.03 ± 0.06 | 77% → 61% | 37.0 (−4.7 ± 3.9) | |
| w5 | 2.0 / 0.5 / 0.1 | +1.1 ± 0.9 | −0.12 ± 0.06 | 77% → 52% | 34.7 (−7.0 ± 4.1) | 54.7 (−1.0 ± 3.7) |
| w10 | 2.0 / 0.5 / 0.25 | +0.1 ± 0.9 | −0.03 ± 0.06 | 77% → 46% | 40.7 (−1.0 ± 3.9) | |

On the chains w5 cut wizzrobe's unfired share (`d6` 76.7% → 66.9%, `d9` 93.9% → 70.3%) but its far-kill share
stayed at 2% (2.5% before) and the kills stayed in melee (21 pixels). The bench's small gain does not carry to the
chains; no arm wins.

### A beam head

Labels: `beam_lined`, a new per-decision field (`StepInfo` → the pool's `Outcome` → `EnvPool.rollout` →
`RolloutBuffer` → `lab.record`): the directions `zelda::beams::lined_up` finds from the decision's state, the
projection the beam terms already compute (no look-ahead). The mask allows BEAMS only with full hearts and the shot
ready, so a decision is a candidate in each legal BEAMS direction (`r38_hit.py` `DIR+beam`). Wizzrobe, its run
`r33w-s93` acting on its training starts (ram randomization: half the resets at full hearts), 60 rollouts: 245,760
decisions, BEAMS legal in 15,129 (6.2%), 58,812 candidates of which 16.7% lined up. A fresh `hitnet` on the policy's
observation (plus schema 7's fields), 12 epochs: precision = recall 0.933 on the held-out shards (darknut
0.919-0.923, goriya 0.886). The head reads the projection well; the policy is the obstacle: on the held-out lined-up
directions the policy's own probability of that shot has median 0.000 (mean 0.078), so the shift must be large.

`triforce/nets/beamshift.py` plays it: the policy's joint action with each BEAMS direction raised by
`weight · σ(b(d))` (nothing else moves; `hitshift`'s pattern turned the other way; a `hitshift` nests as the policy).
`scripts/experiments/r39_beam.py combine` builds one, nothing trained.

**The rooms tier** (`r39_beam.py bench`): unfired lined-up decisions wizzrobe 77% → 59 / 43 / 7% at w8 / 16 / 32,
darknut 77% → 7%, goriya 69% → 9%; far kills wizzrobe 9 → 41, goriya 34 → 65, darknut unchanged (shields); success
and hearts flat.

**The chains, the head everywhere** (300 runs, paired): wizzrobe `d6` w8 34.7% (−7.0 ± 3.5), w16 33.0%, w32 32.7%
(−9.0 ± 3.9); `d9` w32 51.0% (−4.7). The control, weight 0 served the same way, is the base exactly (125/300). The
loss is 6:0x78, the Red Wizzrobes the route crosses: with the head it becomes a fight (far kills 30 → 273, hearts lost
a visit 0.15 → 0.70), and the deaths come later, short of hearts, in 0x29 (13 → 30) and 0x38 (34 → 48). Darknut
(`d8`, w16) 89.7% → 88.7%, `d5` 71.3% → 69.7%; goriya (`d7`, w16) 77.0% → 75.7%: noise.

### Only where the route fights

`beamshift`'s `objectives` gates the shift by the observation's objective kind (`objective_<KIND>`, one-hot, already
an input of every network): `['FIGHT']` raises BEAMS only while the room's objective is a fight (cleared for its
treasure, a shutter, a boss); a crossing (objective MOVE) plays the policy alone. On wizzrobe's recorded decisions
the kinds split as the route reads: 6:0x38, 0x3a, 9:0x05, 0x10, 0x30 FIGHT; 6:0x78, 0x28, 0x2c, 9:0x04, 0x07, 0x15,
0x17, 0x20, 0x31 MOVE (6:0x29 TREASURE, its key).

| wizzrobe (300 runs, paired) | `d6` | `d9` | 6:0x38 unfired | 6:0x38 far kills | 6:0x78 hearts a visit |
|---|---:|---:|---:|---:|---:|
| 2.0.0.11 | 41.7% | 55.7% | 57.0% of 291 | 20 | 0.15 |
| head w16 / w32, everywhere | 33.0% (−8.7 ± 3.7) | 51.0% (w32, −4.7 ± 3.3) | | | 0.70 |
| head w16, fights only | 41.7% (+0.0 ± 1.4) | 55.7% (+0.0) | | | 0.15 |
| head w32, fights only | 42.0% (+0.3 ± 1.4) | 55.7% (+0.0) | 2.9% of 139 | 30 | 0.15 |

The gate removes the loss: 6:0x78 plays exactly as before. It also removes nearly everything else: in the rooms
wizzrobe fights Link is rarely full (6:0x38 99 of 300 visits, 6:0x3a 20 of 251, Dungeon 9's three ~none), so the
head has few lined-up decisions to act on; 6:0x38's hearts lost a visit 5.08 → 4.72, the runs unchanged. `d9`'s
runs are the same runs (seed for seed). Not a winner, so no 1000-run check.

### Full hearts, still melee: Dungeon 3

The owner (Nov 23): "I want to be clear about `Why beams rarely happen.` I notice this everywhere. For example, look
at rooms in dungeon 3. 0x6b and 0x4a are particularly bad. Link goes and hunts down gel and keese up close (more
often than not) with full hearts, in 0x6b it's really noticeable because he walks all the way around the block
geometry and rarely fires beams at them." The hearts are not the whole story: in Dungeon 3 (wood sword, three
hearts) Link is full on most visits and still kills up close.

The 2.0.0.11 team, `d3` (from `d3first_3_7cs_0` to 0:0x74), 300 runs (39.3% arrive): per room, the visits, the
visits entered full, sword kills out of melee range (`far`) and in it, the mean sword-kill distance, shots a visit,
decisions lined up for a shot with the shot ready and the share that moved or waited instead, and the median frames
a visit.

| room (family) | visits | entered full | far / near kills | far share | distance | shots a visit | lined | unfired | frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3:0x6b (zol: Gels, Keese) | 301 | 280 (93%) | 609 / 2921 | 17.3% | 28.5 | 6.8 | 2764 | 54.5% | 1223 |
| 3:0x4a (stalfos-keese-gel) | 295 | 211 (72%) | 104 / 1301 | 7.4% | 25.0 | 1.7 | 1251 | 78.8% | 842 |
| 3:0x4b (zol) | 298 | 220 (74%) | 341 / 1583 | 17.7% | 31.1 | 2.8 | 1382 | 58.9% | 981 |
| 3:0x5b (darknut) | 527 | 347 (66%) | 6 / 600 | 1.0% | 20.4 | 0.9 | 1143 | 88.4% | 398 |
| family zol (all of Dungeon 3) | 1613 | 1331 (83%) | 1032 / 4798 | 17.7% | 29.0 | 2.1 | 6489 | 65.1% | |
| family darknut | 1788 | 1033 (58%) | 78 / 1047 | 6.9% | 21.8 | 0.6 | 3894 | 86.4% | |

In 0x6b, entered full 93% of the time, 83% of the sword kills are in melee; the visits entered full fire 7.3 shots
(the game makes a shot on any swing at full hearts, so most shots are swings made up close) and leave 54% of their
lined-up decisions unfired. The golden replays of 2.0.0.11 (six paths, the archive's) agree, replayed on the
emulator through the two rooms: 11 of the 12 visits entered at 3 of 3 hearts and stayed full throughout, 0x6b took
863-1412 frames with Link walking 565-1179 pixels (Manhattan, frame to frame) and 16-25 swings, 0x4a 610-1070 frames,
691-1221 pixels and 5-16 swings. That is the owner's picture: the room cleared by walking to each Gel and Keese.

### Hearts into the beam rooms

`r39_beam.py hearts` reads chain records: per family, the visits entered full, the mean deficit of the others, hits
a visit and deaths entered full against short, and for each short visit the walk back to the last room Link left
full: the untaken hearts and fairies on the way (`drops` fates expired or left; a fairy counted as three hearts) and
the families the hearts went to (hearts a short visit).

| record | family | visits | entered full | deficit | hits full / short | deaths full / short | untaken drops would have filled | hearts lost on the way |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 2.0.0.11 `d6` (300) | wizzrobe | 1534 | 30.7% | 5.28 | 0.68 / 1.14 | 1.5% / 14.8% | 0.2% | wizzrobe 4.53, stalfos-keese-gel 0.94 |
| 2.0.0.11 `d9` (300) | wizzrobe | 3538 | 9.5% | 7.59 | 0.38 / 1.31 | 0.0% / 3.1% | 0.0% | wizzrobe 4.89, patra 3.18 |
| 2.0.0.11 `d9` (300) | patra | 1911 | 22.3% | 7.16 | 0.54 / 0.85 | 0.0% / 0.6% | 0.0% | wizzrobe 4.70, patra 2.75 |
| 2.0.0.8 `full` (1000) | darknut | 6784 | 51.9% | 1.88 | 0.36 / 0.25 | 2.1% / 7.3% | 2.5% | darknut 1.14, zol 0.21 |
| 2.0.0.8 `full` (1000) | goriya | 1656 | 19.0% | 3.33 | 1.11 / 0.98 | 0.0% / 5.9% | 1.3% | goriya 1.57, ow-leever 0.52 |

**Pickups are not the lever.** The hearts Link lacks were not left on the floor: untaken drops would have filled at
most 2.5% of the short visits (wizzrobe 0.0-0.2%), and in the beam families' own rooms the hearts and fairies that
fall while he is short are taken (wizzrobe on 2.0.0.8's `d9`: 1458 hearts taken, 51 expired, 36 left, 12 on the
floor at a death; fairies 162 of 196). The heal targets (`drops: first`, `reward.drops_first`) already do their job.
The deficit is hits: wizzrobe's own rooms and Patra's before it.

**What the hit heads did there.** 2.0.0.8 put hit heads on eight families of `d3first` (darknut, Gleeok, Manhandla,
like-like, gibdo, stalfos-keese-gel, vire, zol): on the full route, 2.0.0.7 → 2.0.0.8 (the same seeds, the heads the
one change of networks; the runs also reach further, so the rooms mix differs) the visits entered full went darknut
42.6% → 51.9%, like-like 27.8% → 42.3%, stalfos-keese-gel 46.1% → 54.2%, Gleeok 12.3% → 27.9%, Manhandla 17.9% →
36.9%, vire 52.7% → 57.4%, zol 73.9% → 77.5% (goriya, no head in 2.0.0.8, 17.1% → 19.0%; 2.0.0.9's PPO-aux goriya
27.1%). None of those networks plays in `d6` or `d9`: 2.0.0.6's and 2.0.0.8's records there are the same runs, seed
for seed, and wizzrobe stays at 30.1% / 9.7%. Fewer hits is what raised the full-hearts share; wizzrobe and Patra
never got a head.

**So the beam lever is partly a hearts lever, for wizzrobe and Patra:** their killer rooms are entered 7 hearts short
and no beam behavior reaches them. The first experiment (written, not run: its recording was cancelled to free the
queue for Dungeon 3): a hit head for wizzrobe and patra as round 38 built them (`python -m triforce.lab.record
--families wizzrobe patra --state-route d3first --actor teacher --try-others 8:1.0 --rollouts 100`, `r38_hit.py
retrain --init models/r38head-hit/all --data <rec>`, `r38_hit.py combine` at w8), measured as the eight were:
the share of 9:0x05, 0x10, 0x30 and 6:0x3a entered full, then `d6` and `d9`. A beam shift nests on top of it
(`beamshift` takes a `hitshift` policy). **But hearts are not enough:** Dungeon 3 (above) is entered full most of
the time and is fought in melee anyway.

### The beam head in Dungeon 3

Beam labels recorded for `stalfos-keese-gel@d3first` and `zol@d3first` (each its 2.0.0.11 run, the `hitshift`
`r38head-fam-w8` / `-w4`, on the route's states, 60 rollouts: 245,760 decisions, BEAMS legal in 32,881 / 41,443;
3:0x4a, 0x4b and 0x6b are FIGHT rooms), fresh heads 12 epochs: precision = recall 0.885 (stalfos-keese-gel) and
0.904 (zol). Combined with the run's own `hitshift` as the policy (the hit head kept), gated to fights, w16 and w32.
`d3`, 300 runs paired with the 2.0.0.11 team (39.3%):

| arm | `d3` | 0x6b far / sword kills | 0x6b unfired | 0x6b frames | 0x4a far / sword kills | 0x4a unfired | 0x4a frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2.0.0.11 | 39.3% | 609 / 3530 (17%) | 54% of 2764 | 1223 | 104 / 1405 (7%) | 79% of 1251 | 842 |
| stalfos-keese-gel w16 | 37.7% (−1.7 ± 2.6) | (unchanged) | | | 244 / 1365 (18%) | 29% of 619 | 813 |
| zol w16 | 40.0% (+0.7 ± 3.7) | 937 / 3434 (27%) | 16% of 1837 | 1166 | (unchanged) | | |
| both w16 | 36.0% (−3.3 ± 3.7) | 937 / 3434 (27%) | 16% of 1837 | 1166 | 253 / 1355 (19%) | 25% of 553 | 802 |
| both w32 | 34.3% (−5.0 ± 3.5) | 986 / 3479 (28%) | 4% of 1590 | 1134 | 269 / 1342 (20%) | 4% of 464 | 801 |

Frames are the median visit; hearts lost in the rooms are unchanged (0x6b 0.22 → 0.19-0.20 a visit). The head
does what it is built for: lined up, Link fires (unfired 54% → 4-16%, 79% → 4-29%), far kills go up by half in 0x6b
and more than double in 0x4a, and the rooms are 4-7% shorter. It does not change the walk: three kills in four are
still made in melee, because the shift only acts on a decision already lined up for a shot; it never makes Link
stand back and line one up, which is what the owner is asking for (0x6b's walk round the block). Lined-up decisions
even become fewer (2764 → 1590-1837), the monsters dying sooner. The runs' rate does not move within noise; the runs
end in 3:0x4d (Manhandla), 0x69 and 0x5b, not in these rooms, so a beam change here can only cost or win a little
time. The zol w16 arm, the only one not below the base, went on to 1000 runs (seeds 0-999): 39.2% → 37.8%
(−1.4 ± 2.0; base only 202, new only 188), 0x6b's far kills 2092 → 3144 of 11,612 / 11,353 sword kills (18% → 28%),
unfired 53% of 9404 → 14% of 5938, median visit 1217 → 1145 frames, hearts lost 0.20 → 0.19 a visit. Not a win on
the rate, so not promoted and no golden run.

### Next (written, not run)

1. **Train with the shot in the policy** (the walk is the policy's, so it has to learn it): PPO on
   `zol@d3first` and `stalfos-keese-gel@d3first` from their 2.0.0.11 runs with the beam head frozen inside the
   acting network (`beamshift`, w16, `objectives: [FIGHT]`, `freeze_beam`; the shift is in the action distribution,
   so the policy is trained on what is actually played and can learn to reach the places where the head fires),
   from 3:0x6b's, 0x4b's and 0x4a's starts at full hearts, with the round 38 beam terms. Measured on the rooms tier
   (far share, frames, hearts) and `d3`.
2. **A firing-line potential**: the beam terms pay a shot and charge a lined-up decision left unfired; nothing pays
   for getting lined up. A potential (PBRS, so it cannot be farmed) on "a monster on one of Link's four lines out of
   melee range, shot ready" would make standing back to shoot worth walking to, and the head (or the policy) fires.
3. **Hearts for wizzrobe and Patra**: the hit head of "Hearts into the beam rooms" above.

## Moldorm crossed, 9:0x56's key: the two fine-tunes (branch `r39-key`, 2.0.0.13)

The owner's change of Nov 22 (round38.md "Moldorm crossed, 9:0x56's key": `d3first` crosses 2:0x3e and takes
9:0x56's key instead; the ledger, the routes.yaml change and the states patched there) left two networks playing a
step they were not trained for: Dungeon 9 fell from 53.0% to 6.7% (like-like stuck at 0x56) and the crossing of
0x3e lost more hearts than the fight (5 deaths in 1000 against 1). The owner, Nov 23: "This key is the correct route,
other keys are not": fine-tune (or retrain) the two networks, no 8:0x4c fallback.

**Which runs play the steps.** Step 246 (9:0x56) is after `own_models_until` (192): the bare key `like-like` plays,
its `late.d3first` else its `promoted`, `r33-like-s83` (round 33; not a hit-head run). Step 100 (2:0x3e) is
`moldorm@d3first`, which had no `promoted.d3first`: `r75-f-b`. 2:0x3e is `d3first`'s only Moldorm step.

**9:0x56 over the water.** On the room's three entry states (`d3first_9_56s_{0,1,2}`, keys 2, bombs 5-6) Link holds
the ladder (`InvLadder` `$663` = 1). The room is a land ring between two water rows (tile rows 6-7 and 14-15 of the
grid): Link enters on the bottom row, the monsters and the key ((128, 144)) are inside, and the W bomb wall opens on
the middle rows, so every way through steps onto the water. With the room clear the objective is the key
(`objective_TREASURE`) and the target walks over the water: the room's wavefront counts the ladder's crossings
(`Room::compute_ladder_crossings`, the water the game lays the ladder on; the slot rule, a free monster slot, holds
once the four are dead and the two Bubbles leave ten). Go-Explore (`python -m triforce.explore`, three seeds a state)
found 9 tracks, each killing the two Zols and two Like-Likes, laying the ladder (slot 11), taking the key (keys 2 → 3)
and leaving W: 96-245 decisions.

**like-like `r39t-like`**: `r33-like-s83` fine-tuned 600 iterations (`--parallel 8`, `--sil-coeff 0.1`) on the
family's rooms that `late.d3first` plays (steps 208-285: 6:0x1d, 9:0x56, 0x14, 0x51, 0x41, 0x03; the family's yaml
with those rooms, `state_routes: [d3first]`, no harvested starts), the 9 tracks as curriculum. Promoted through
`late.d3first`: `d2first` and the wood-sword half (4:0x32, `promoted.d3first`) play as before.

**moldorm `r39t-moldorm`**: `r75-f-b` fine-tuned 400 iterations (SIL 0.1) on `specialists/d3first/moldorm.yaml`
(the family's yaml, `state_routes: [d3first]`: the objective is the N door, the Moldorm left), 9 explored crossings
as curriculum. Promoted as `promoted.d3first`.

| One room, `eval-chain --from`, `--act model` | network | runs | took the key, out W | frames | hearts lost | hits | deaths |
|---|---|---:|---:|---:|---:|---:|---:|
| 9:0x56, `9_56s_0` to step 247 | `r33-like-s83` | 300 | 17% (250 stuck) | 15,702 | 0.24 | 0.25 | 0 |
| | **`r39t-like`** | 300 | **100%** | **1,219** | **0.11** | 0.12 | 0 |
| 2:0x3e crossed, `d3first_2_3es_0` to step 101 | `r75-f-b` | 1000 | | 370 | 0.68 | 1.36 | 0 |
| | **`r39t-moldorm`** | 1000 | | **333** | **0.38** | 0.75 | 0 |
| 2:0x3e fought, `2_3es_0` (`d2first`, 3 hearts) to step 20 | `r75-f-b` | 1000 | | 817 | 0.39 | 1.66 | 0 |

**The chains** (`r39-key2-1k`: the branch on 2.0.0.8's code and team with the two runs, 1000 runs a segment, paired
by seed with `r38-208-1k`; on that code the bare key's late steps read `promoted`, set to `r39t-like` for the batch,
which is what `late.d3first` plays):

| Segment | 2.0.0.8 (Moldorm fought, 0x56 crossed) | Moldorm crossed, 0x56's key (old networks) | **fine-tuned** |
|---|---:|---:|---:|
| `d2` | 99.3% | 99.0% | **99.4%** (+0.1 ± 0.3) |
| `d6` | 36.4% | 35.5% | 35.7% (−0.7 ± 2.1) |
| `d9` | 53.0% | 6.7% | **51.2%** (−1.8 ± 2.2) |

- 2:0x3e in `d2` (999 visits): 320 frames, **0.35 hearts lost**, 0.69 hits, **1 death**: the fight was 1,008 frames,
  0.52 hearts, 1 death; the old network crossing 383 frames, 0.78 hearts, 5 deaths. From the start (`full`): no
  death in 0x3e (1 on 2.0.0.8).
- 9:0x56 in `d9`: 999 of the 1000 runs take the key (1065 visits: the others are returns after it), every visit out W
  (1 by a wrong exit), none stuck, 1,192 frames and 0.10 hearts a visit (the old network: 873 stuck, 15,963 frames).
- `d9`'s 1.8 points are inside the noise and not in 0x56: the runs die where they died on 2.0.0.8 (0x10 143 / 130,
  0x30 139 / 155, 0x42 52 / 51, 0x52 28 / 23, 0x04 25 / 12).
- No key cheated in any segment, either way.
- `d2first` after the merge of 2.0.0.12 with the two runs promoted (its pointers untouched): 100 runs, result, steps
  and frames equal to `d2first-100-land` on every run.
- Golden race with the 2.0.0.13 team (`zelda-headless.py OUT 3 --golden --power-on --clips on --act model
  --parallel 8 --attempts 2000`): three wins of three in 109 runs, 155,143, 162,946 and 155,838 frames (2.0.0.11's
  corpus path: 167,136), each `tf replay verify` ok; archived as `2.0.0.13/golden/`.
- Suites: `cargo test --release --workspace --features zelda/zcore` (1073 passed), pytest with the zelda-asm tests
  (976 passed, 12 skipped), `asm.py check` ok, `tf replay verify corpus/replays/*.tfreplay` ok.

Also fixed: `python -m triforce.explore --variant` with a yaml path (`--specialist specialists/d3first/moldorm.yaml`)
looked for the explorer's tracks under the yaml's path instead of the family's name and dropped them; tracks now go to
`<out>/<family>/<state>~s<seed>` (`tests/test_explore.py`).
