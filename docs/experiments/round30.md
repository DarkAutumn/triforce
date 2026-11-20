# Round 30: the optimisation round over the whole route

## Darknut, boss-manhandla, gibdo: Dungeon 5 at the route's hearts, then Dungeon 8

darknut plays 3:0x5b, 0x59, 0x69 and 0x5c in Dungeon 3, 5:0x64, 0x06, the Olive Garden 0x05 and 0x37 in Dungeon 5,
8:0x5e, 0x4e, 0x3e and 0x3f in Dungeon 8. Its line: `r75-d537-c5` (round 25, `main` until round 28) →
`r78t4-darknut` → `r78-dark37-s85` (round28.md §15: chosen on `d8`, checked on `d3` +1.4 ± 2.0 and `d5` +6.8 ± 2.0
from full hearts). Every round of the lineage trained half its episodes with CLIP masked; this is its first with
`clips: {enabled: true, off_share: 0}`.

### The cause

The full route (`r29land-1k-full`, 1000 runs, `r78-dark37-s85`) against `r77b-1k-full` (the same seeds, `main`'s
round 27 team with `r75-d537-c5`) and the honest planner beside (`r29land-plan-full`, 300 runs), darknut's visits by
route step: visits, hearts on entering, hearts lost a visit (the visits Link survives), deaths.

| visit | `r78-dark37-s85` | `r75-d537-c5` | the planner (300) |
|---|---|---|---|
| 5:0x05 from E (step 192) | 540, 7.06 in, **1.69**, **141 deaths** | 434, 7.42, 1.31, 80 | 284, 9.17, 0.73, 5 |
| 5:0x64 from the cellar (197) | 398, 5.44, 1.12, 40 | 354, 6.10, 1.15, 24 | 280, 8.39, 0.93, 4 |
| 5:0x37 (203) | 285, 4.05, 0.12, 4 | 264, 4.62, 0.16, 8 | 273, 7.35, 0.07, 1 |
| 3:0x59 from N (94) | 740, 4.89, 0.58, 4 | 641, 4.92, 0.70, 10 | 297, 5.71, −0.05, 1 |
| 3:0x69 from N (95) | 725, 4.32, 0.65, 12 | 631, 4.26, 0.71, 19 | 294, 5.78, 0.49, 0 |
| 3:0x5b from W (100) | 703, 3.48, −0.05, 15 | 603, 3.39, −0.14, 13 | 294, 5.25, −0.17, 0 |

The Olive Garden by the hearts Link brings (full route): 6 to 7.9 hearts in, 63 deaths of 245 visits against 30 of
168; 8 to 9.9, 2.61 hearts lost a visit against 1.65 (32 deaths of 186 against 24 of 187). From full hearts (the
`d5` segment, `r29land-1k-d5`) it loses 2.65 a visit with sword beams and 35 deaths of 691: the segment the run was
checked on starts where the route never arrives.

**Dungeon 5 at the route's hearts.** The leg after Dungeon 4 arrives at 5:0x76 with 7.77 hearts (the full route
7.74), and no darknut room is on it, so `--from 0_45c_0 --stop-at step:209` (the leg and Dungeon 5, to the overworld
after the D5 triforce) plays Dungeon 5 as the full route does. 500 runs of it (seeds 7000-7499, the harvest chain
below): **247 of 500 take the triforce (49.4%)**, the full route 262 of 540 entering (48.5%), the `d5` segment
72.2%; 0x05 entered with 7.27 hearts, 1.87 lost a visit, 116 deaths of 499.

**Dungeon 8** (`r29land-1k-d8`, 1000 runs from its entrance at 13 of 13 hearts; the full route enters with 12.50):
0x5e 1000 visits, 12.18 in, 4.04 lost, 20 deaths; 0x4e 979, 8.15, 2.91, 88; **0x3e 891, 5.68, 1.61, 414**; 0x3f
476, 5.49, 1.98, 155. darknut's four rooms end 677 of the 752 runs `d8` loses. The planner in the same rooms (full
route): 0x5e 0.97 hearts a visit, 0x4e 1.33 (3 deaths of 129), 0x3e −0.29 (none), 0x3f 0.10 (none).

**Dungeon 3**: darknut's rooms kill few runs (0x69 12, 0x5b 15 of 1000 full runs) but lose the hearts Manhandla
needs: 3:0x4d is entered with 3.58 hearts (the planner 5.42) and 81 of its 97 deaths are entries below 4 hearts.

All 73 curriculum tracks of `r78-dark37-s85` are complete: no track idles the resets (the recipe's fourth lever has
nothing to take out).

### The arm

From `r78-dark37-s85`, three seeds (`r30-dark-s87`, `-s88`, `-s89`), 488 iterations, `--sil-coeff 0.1
--checkpoint-every 30`, `clips: {enabled: true, off_share: 0}`, and:

- **the chain's entries as harvested starts** at `harvested_fraction: 0.33` (`triforce/states/harvested/darknut`, not
  in git; 1220 states). One batch harvested the handoffs (`eval-chain --harvest`) and recorded replays on seeds
  7000-7499 no measurement uses, with the promoted team: the leg after Dungeon 4 and Dungeon 5 (`--from 0_45c_0
  --stop-at step:209 --runs 500`), Dungeon 8 (`--from 8_7es_0 --stop-at step:297 --runs 500`) and the leg after
  Dungeon 1 with Dungeon 3 (`--from 0_37c_0 --stop-at step:134 --runs 300`; Dungeon 3 is entered as the full route
  enters it, 0x5b at 5.45 hearts against 5.42). The rooms entered from another darknut room have no handoff and were
  cut out of the replays (`scripts/experiments/r29_zora_states.py REPLAYS OUT --step 192:5_05e`, `--step 290:8_4es
  --step 291:8_3es --step 292:8_3fw`). Kept, the first by seed: 0x05 from E 250 (7.27 hearts on entering), 0x64
  from the cellar 120, 0x37 60; 8:0x5e 120, 0x4e 120, 0x3e 250, 0x3f 150; 3:0x59 from N 60, 0x5b from W 60, 0x59 from
  E 30. The harvested selector weighs them by their failures and hearts lost as the seed selector does;
- **the honest planner's play as demonstrations** beside the lineage's three files (`target/demos/r28-demos/darknut-5e.npz`,
  `target/demos/r28-demos/darknut-3e.npz`: Dungeon 8's scripted rules; `target/demos/r78dk/demos37.npz`: 0x37's rule;
  untracked, gathered in `main`'s checkout), all at `--demo-coeff 0.05`:
  `scripts/experiments/r27_plan_demos.py models/darknut/r78-dark37-s85 plan-dk.npz --family darknut --states 5_05e
  5_64c 8_5es 8_4es 8_3es 8_3fw --runs 40 --max-loss 2` (the family's hp and bombs draws, full rooms): 61,067 rows.
  The planner with `r78-dark37-s85` wins 0x05 from E 114 of 120 losing 0.24-0.61 hearts, 0x3e 111 of 120 at −0.31
  to 0.28, 0x5e 112 of 120 at −0.13 to 1.18, 0x3f 118 of 120, 0x4e 118 of 120, 0x64 from the cellar 115 of 120.

**The Olive Garden's damage by bombs and by the sword** (the baseline for a darknut that fights by bombs, as a route
with the wood sword will need). 5:0x05 from E, per visit, from the room records (`bomb_damage` counts in wood-sword
hits, `hits_dealt` every hit on a monster, `bomb_hits` the hits credited to bombs); the sword's share is taken as
two wood-sword hits of damage a white-sword hit (`[INFERENCE]` from the game's sword damage, not measured here):

| chain | visits | bombs carried in | bombs placed | bomb damage | sword hits | bombs' share of the damage | kills |
|---|---:|---:|---:|---:|---:|---:|---:|
| full route `r29land-1k-full` | 540 | 1.01 | 0.99 | 3.44 | 19.4 | 8% | 4.77 |
| leg and Dungeon 5 (harvest, 500) | 499 | 0.75 | 0.75 | 2.81 | 20.4 | 6% | 4.94 |
| `d5` from full hearts `r29land-1k-d5` | 1000 | 2.64 | 2.63 | 10.02 | 18.7 | 21% | 5.59 |
| the planner, full route (300) | 284 | 0.93 | 0.93 | 3.73 | 22.1 | 8% | 5.83 |

Every bomb Link carries in is placed (bombs free down to 0 in the room) and almost every one hits (0.92 of 0.99
placed in the full route), at 3.5-3.8 wood-sword hits of damage a bomb; the route brings about one bomb to the
room, so the sword deals some nine tenths of the damage. No bomb in the room is cheated.

### The runs (single rooms), and where the choice stands

Canonical evaluations on this code (clips on, 100 episodes a state, the run's act mode):

| run | success | hearts | 5:0x05 | 8:0x3e | 8:0x5e | 8:0x4e | 8:0x3f |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r78-dark37-s85` (promoted) | 96.3% | −0.91 | 92.7% | 81.7% | 84.3% | 92.7% | 95.3% |
| `r30-dark-s87` | 96.6% | −0.86 | 92.5% | 85.7% | 85.7% | 92.0% | 94.3% |
| `r30-dark-s88` | 96.4% | −0.89 | 92.0% | 86.0% | 85.7% | 92.7% | 93.7% |
| `r30-dark-s89` | 96.8% | −0.87 | 93.0% | 87.0% | 87.0% | 93.7% | 94.3% |

The secondaries, one seed each from their promoted runs with harvested starts at the route's hearts
(`harvested_fraction: 0.33`): boss-manhandla `r30-manh-s90` (280 entries of 3:0x4d from the Dungeon 3 harvest
chain) 95.7%, −0.62 (`r78t4-manhandla2` 94.0%, −0.75; 3:0x4d 91.3% against 88.3%); gibdo `r30-gibdo-s91` (the way
back's entries: 0x65 and 0x66 from W 150 each, cut out of the leg-and-Dungeon-5 replays for 0x66, 0x47 150, 0x26
50; and the planner's play of 0x65, 0x66 from W and 0x47, 13,175 rows at `--demo-coeff 0.05`) 99.5%, +0.02
(`r75-f-a` 99.3%, +0.01).

**boss-manhandla, chosen on Dungeon 3 at the route's hearts** (`--from 0_37c_0 --stop-at step:105`; the leg after
Dungeon 1 arrives with the full route's hearts): `r30-manh-s90`'s last checkpoint screens at 84.9% against
`r78t4-manhandla2`'s 80.9% (+4.0 ± 1.2 paired, seeds 2000-2999) and **confirms at 82.0% against 77.4% (+4.6 ± 1.3,
seeds 3000-3999)**. On seeds 0-999, paired with `r29land-1k` (the same team but boss-manhandla): **`d3` 85.4% against 82.7%,
+2.7 ± 1.3** (3:0x4d kills 116 of 970 entering runs against 143 of 971, at the same 4.08 hearts in), `d8` 26.2%
against 24.8%, +1.4 ± 1.9 (8:0x6e crossed by all 1000, 0.84 hearts a visit against 0.82). Single rooms: 3:0x4d
91.3% against 88.3%, 8:0x6e 100% against 99.7%. **Promoted** in this branch's checkout (`triforce promote
boss-manhandla r30-manh-s90 --act temperature:0.5+relax:16`; its `ckpt.pt` is the chosen last checkpoint). Its 280
harvested 3:0x4d starts are not in git: `triforce/states/harvested/boss-manhandla` is rebuilt from the Dungeon 3
harvest chain above (its `boss-manhandla/3_4dw_*` files, all of them).

**Darknut and gibdo, chosen on Dungeon 5 at the route's hearts** (`select-checkpoint FAMILY RUN --from 0_45c_0
--stop-at step:209 --runs 1000 --first-seed 2000 --confirm 1000 --confirm-first-seed 3000 --confirm-top 1 --only
<the last checkpoints>,last`; the promoted team measured once, in `r30-dark-s87`'s batch, and the other selections
paired with it on the same seeds: each plays the same team with one run swapped). The D5 triforce taken:

| run | candidate | screen (2000-2999) | confirm (3000-3999) | hearts on arriving | confirm vs the promoted team |
|---|---|---:|---:|---:|---:|
| promoted team | `r78-dark37-s85`, `r75-f-a` | 48.5% | 49.0% | 5.35 | |
| darknut `r30-dark-s87` | it011462 | 53.6% | 52.6% | 5.43 | +3.6 ± 2.2 |
| | last | 49.7% | 49.5% | 5.18 | +0.5 ± 2.2 |
| darknut `r30-dark-s88` | it011492 | 54.2% | 53.4% | 5.29 | +4.4 ± 2.2 |
| | last | 49.7% | 50.3% | 5.28 | |
| darknut **`r30-dark-s89`** | **last (it 11530)** | 53.7% | **54.8%** | 5.38 | **+5.8 ± 2.2** |
| | it011462 | 49.1% | 48.5% | 5.34 | |
| gibdo **`r30-gibdo-s91`** | **last (it 1368)** | 52.8% | **54.3%** | 5.60 | **+5.3 ± 2.1** |
| | it001330 | 55.4% | 53.1% | 5.52 | |

All three darknut seeds gain over the promoted run on the segment it lost (+3.6 to +5.8 on fresh seeds), and gibdo
as much. The choice: darknut `r30-dark-s89` and gibdo `r30-gibdo-s91`, their last checkpoints (each run's
`ckpt.pt`). Their other segments (darknut `d8`, `d3`, `d5`; gibdo `d5`; 1000 runs, seeds 0-999) and the segment
with both swapped together (the gains were measured one family at a time), 1000 runs each, seeds 0-999 (`d3` and
`d8` paired with this checkout's team with the new boss-manhandla, `d5` with `r29land-1k-d5`):

| segment | reference | with the run | paired | where |
|---|---:|---:|---:|---|
| darknut `d8` | 26.2% | **34.3%** | **+8.1 ± 2.0** | 8:0x3e 354 deaths of 922 against 431 of 900, 0.89 hearts a visit against 1.42; 0x5e 6 against 23 |
| darknut `d5` (full hearts) | 72.2% | 73.9% | +1.7 ± 1.9 | |
| darknut `d3` | 85.4% | 82.2% | **−3.2 ± 1.6** | 3:0x69 from N 0.76 hearts a visit against 0.55; Manhandla entered with 3.80 hearts against 4.08, 147 deaths against 116 |
| gibdo `d5` (full hearts) | 72.2% | 73.3% | +1.1 ± 2.0 | |
| both, Dungeon 5 at the route's hearts (seeds 3000-3999) | 49.0% | **58.3%** | **+9.3 ± 2.2** | 5:0x05 1.45 hearts a visit against 1.70, 206 deaths of 1000 against 239 |

**gibdo `r30-gibdo-s91` qualifies and is promoted** in this branch's checkout (`triforce promote gibdo r30-gibdo-s91
--act temperature:0.5+relax:16`).

**darknut `r30-dark-s89` does not qualify: it loses Dungeon 3.** A second 1000 runs of `d3` on seeds 1000-1999, both
teams measured (`r30dk-check-s89-d3b`): 81.6% against 86.7%, −5.1 ± 1.7; the 2000 runs pooled **−4.2 ± 1.2**.
The cause is the same on both halves: 3:0x69 from N costs 0.74-0.76 hearts a visit against 0.55-0.56, and Manhandla,
entered with 3.80-3.84 hearts against 4.08-4.12, kills 141-147 runs against 106-116. The run is better where it was
trained to be (`d8` +8.1 ± 2.0, Dungeon 5 at the route's hearts +5.8 ± 2.2, with gibdo +9.3 ± 2.2) and worse in the
one room whose chain entries the arm left out (0x69 from N: its replay cut was dropped to start the training sooner,
while 0x59 and 0x5b, which kept theirs, hold). Not promoted; `r78-dark37-s85` stays.

**The continuation with 0x69 from N's entries** (`r30-dark69-s92`, `-s93`, `-s94`): from `r30-dark-s89` (it holds
`d8` and Dungeon 5), 244 iterations, the same arguments and demonstrations, and the harvested starts above plus 120
entries of 3:0x69 from N cut out of the Dungeon 3 harvest chain's replays (`scripts/experiments/r29_zora_states.py
REPLAYS OUT --step 95:3_69n --count 120`; all enter at X 120, Y 93; 1340 states). Chosen on `d3` (`select-checkpoint
darknut RUN --from 3_7cs_0 --stop-at step:105`, screen seeds 2000-2999, confirm 3000-3999, `r78-dark37-s85` pinned),
then Dungeon 5 at the route's hearts and `d8`. Beside it, the fallback: the two other seeds' chosen checkpoints
(`r30-dark-s88` it011492, +4.4 ± 2.2; `r30-dark-s87` it011462, +3.6 ± 2.2 on Dungeon 5) measured on `d3` and `d8`.

The fallback (`r30dk-check-s88s87`, seeds 0-999, paired with this checkout's team as for `r30-dark-s89`): both other
seeds hold Dungeon 3 and gain more on `d8`. 0x69 from N costs them what it costs the promoted run, so `r30-dark-s89`'s
loss there is that seed's drift, not the arm's:

| run (checkpoint) | `d3` | paired | 3:0x69 from N, hearts a visit | Manhandla: hearts in, deaths | `d8` | paired |
|---|---:|---:|---:|---:|---:|---:|
| reference (`r78-dark37-s85`) | 85.4% | | 0.55 | 4.08, 116 | 26.2% | |
| `r30-dark-s88` (it011492) | 86.0% | +0.6 ± 1.5 | 0.55 | 3.99, 112 | **36.6%** | **+10.4 ± 2.0** |
| `r30-dark-s87` (it011462) | 85.1% | −0.3 ± 1.6 | 0.62 | 3.94, 110 | **36.4%** | **+10.2 ± 2.0** |
| `r30-dark-s89` (last) | 82.2% | −3.2 ± 1.6 | 0.76 | 3.80, 147 | 34.3% | +8.1 ± 2.0 |

`r30-dark-s88` it011492 on `d3` again on seeds 1000-1999 (paired with the reference measured on them): 84.6% against
86.7%, −2.1 ± 1.5; the 2000 runs pooled −0.8 ± 1.1.

**The continuations, chosen on `d3`** (`select-checkpoint darknut RUN --from 3_7cs_0 --stop-at step:105 --runs 1000
--first-seed 2000 --confirm 1000 --confirm-first-seed 3000 --confirm-top 1 --only it011710,it011740,it011770,last`;
`r78-dark37-s85` measured in `r30-dark69-s92`'s batch, the others paired with it on the same seeds). Single rooms
(canonical evaluation): 96.7%, 96.5%, 96.5% (−0.88, −0.85, −0.85 hearts).

| run | candidate | screen (2000-2999) | confirm (3000-3999) | hearts on arriving | confirm vs the reference | 3:0x69 from N, hearts a visit |
|---|---|---:|---:|---:|---:|---:|
| reference | `r78-dark37-s85` | 86.5% | 85.2% | 4.47 | | 0.58 |
| `r30-dark69-s92` | it011710 | 84.6% | 85.1% | 4.41 | −0.1 ± 1.6 | 0.61 |
| | last | 82.6% | 84.3% | 4.39 | | |
| `r30-dark69-s93` | last | 83.8% | 83.5% | 4.45 | −1.7 ± 1.6 | 0.61 |
| | it011770 | 82.1% | 84.4% | 4.45 | | |
| **`r30-dark69-s94`** | **last (it 11774)** | 85.9% | **86.2%** | 4.48 | **+1.0 ± 1.5** | 0.52 |
| | it011770 | 85.2% | 85.7% | 4.43 | | |

The entries did what they were for: 0x69 from N is back to the reference's cost on every seed (0.52-0.61 against
`r30-dark-s89`'s 0.76). `r30-dark69-s94`'s last checkpoint (its `ckpt.pt`) is the best; it enters Manhandla with the
reference's hearts (4.09 against 4.09, 111 deaths against 113 on the confirmation seeds).
On seeds 0-999 (paired with this checkout's team): `d3` 85.3% against 85.4%, −0.1 ± 1.5 (0x69 from N 0.58 hearts a
visit); with the confirmation, **2000 runs +0.4 ± 1.1: Dungeon 3 holds.** But Dungeon 5 at the route's hearts
(seeds 3000-3999, gibdo `r75-f-a` as in that selection's reference) is 50.4% against 49.0%, **+1.4 ± 2.2** (−4.4 ± 2.2
against `r30-dark-s89`, which it was continued from), while `d8` is 41.0% against 26.2%, **+14.8 ± 2.1**. The 244
iterations bought back Dungeon 3 with Dungeon 5's gain.

**No darknut run qualifies; `r78-dark37-s85` stays.** The candidates (paired with the reference; Dungeon 3 over 2000
runs where marked):

| run | Dungeon 3 | Dungeon 5 at the route's hearts | `d8` | fails on |
|---|---:|---:|---:|---|
| `r30-dark-s89` last | −4.2 ± 1.2 (2000) | **+5.8 ± 2.2** | **+8.1 ± 2.0** | Dungeon 3 |
| `r30-dark-s88` it011492 | −0.8 ± 1.1 (2000) | +4.4 ± 2.2 | **+10.4 ± 2.0** | Dungeon 5 exactly at two standard errors |
| `r30-dark-s87` it011462 | −0.3 ± 1.6 (1000) | +3.6 ± 2.2 | **+10.2 ± 2.0** | Dungeon 5 |
| `r30-dark69-s94` last | +0.4 ± 1.1 (2000) | +1.4 ± 2.2 | **+14.8 ± 2.1** | Dungeon 5 |

The best trade is `r30-dark-s88` it011492: it holds Dungeon 3 and gains `d8` by ten points, and its Dungeon 5 gain is
on the line, not beyond it. Settling it takes a second 1000 runs of Dungeon 5 at the route's hearts on fresh seeds,
with the reference on the same seeds.
