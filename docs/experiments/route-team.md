# The route team: one team for the whole route, measured (Nov 18-19)

The rest of the game was trained in three lanes, each a branch on the one before: the magic sword and Dungeon 6
(route `d6`, steps 248-280), Dungeon 8 (route `d8`, steps 281-296), Dungeon 9 to the ending (route `d9`, steps
297-357). Each lane met its bar on its own stretch, and several families were continued lane after lane, each lane
judging them only on its own stretch. This census plays **one team on one code line** (this branch, 1.5.1.0,
observation schema 6) over every segment of the whole route, 1000 runs a segment, clips on, models alone, and pairs
each segment by seed with the reference that segment was accepted with. A segment is worse (better) beyond noise
when the paired difference is more than two standard errors of it (AGENTS.md "Compare models fairly"). For every
family that loses a segment, one swap test: the same segment again with only that family's run swapped for the
reference's, and a single-room evaluation of the swapped-in run on the family's start states of the later stretches
(100 episodes a state, clips on, the family's current yaml with only those states).

## The candidate team

For every family the newest run of its lineage (`models/<family>/promoted` on this branch):

| family | run | from |
|---|---|---|
| ow-start, ow-tektite, ow-octorok | `r74-leg2` | main (all lanes) |
| ow-peahat | `r75-bow-ow` | main (all lanes) |
| ow-moblin | `r79t5-moblin` | Dungeon 9 lane (<- `r78t4-moblin` <- `r77-b` <- `r76-b`) |
| ow-zora | `r79t5-zora` | Dungeon 9 lane (<- `r76-zora-arrows`) |
| ow-leever | `r79t5-ow` | Dungeon 9 lane (<- `r76-b`) |
| ow-boulder | `r79b-boulder` | Dungeon 9 lane (<- `r72-ow`) |
| ow-lynel | `r27l-lynel-s81` | the joint ow-lynel run (<- `r79t5-lynel` <- `r78t4-lynel` <- `r77-b` <- `r74-lynel`); the census began with `r79t5-lynel` |
| ow-ghini | `r77-ghini` | Dungeon 6 lane |
| stalfos-keese-gel | `r79t5-skg` | Dungeon 9 lane (<- `r78-skg` <- `r77-b` <- `r75-f-b`) |
| like-like | `r79c-like` | Dungeon 9 lane (<- `r77-b` <- `r75-f-c`) |
| zol | `r79b-zol` | Dungeon 9 lane (<- `r77-b` <- `r75-f-b`) |
| trap-room, vire | `r77-b` | Dungeon 6 lane |
| wizzrobe | `r79c-wj-s85` | Dungeon 9 lane (<- `r80-wplan-s78`, Dungeon 6's) |
| darknut | `r78-dark37-s85` | Dungeon 8 lane (<- `r78t4-darknut` <- `r75-d537-c5`) |
| boss-manhandla | `r78t4-manhandla2` | Dungeon 8 lane (<- `r75-mh-fight`) |
| boss-gleeok | `r78t4-gleeok` | Dungeon 8 lane (<- `r71-d`) |
| pols-voice | `r78t4-pols` | Dungeon 8 lane (<- `r74-d5new`) |
| boss-gohma | `r77-gohma` | Dungeon 6 lane |
| lanmola, patra, ganon | `r79b-lanmola`, `r79c-patra`, `r79c-ganon` | Dungeon 9 lane |
| goriya | `r76-goriya4-s78` | main |
| wallmaster | `r76-wall-1` | main |
| dodongo-crossing, equipment-room | `r76-c` | main |
| boss-digdogger, boss-aquamentus | `r76-b` | main |
| d2-rope, gibdo, moldorm, boss-dodongo | `r75-f-c`, `r75-f-a`, `r75-f-b`, `r71-e` | main (all lanes) |

## The segments

Chains `r79x-1k-<segment>` (1000 runs, seeds 0-999; d2 and d7 300 runs, see below). The earlier segments stop at
their arrival (`--stop-at`, the next segment's start), as the Dungeon 6 lane's `r77-1k` did; a reference that played
on is cut at the same arrival. Rate = runs arriving (the route's end for d7, d6, d8, d9). Hearts = hearts on
arriving, runs arriving in both.

| segment | the families that differ from the reference | reference | candidate | paired difference | hearts on arriving |
|---|---|---|---:|---:|---|
| d2 (300) | none | main `r76-1k` 83.3% (1000: 81.5%) | 83.3% | identical run for run | 4.00 / 4.00 |
| post-d2 | ow-lynel, ow-zora, ow-moblin, ow-boulder | main `r76-1k` 95.7% | 83.9% (`r79t5-lynel`) | **−11.8 ± 1.3** | 3.25 / 3.01 |
| post-d2, ow-lynel `r27l-lynel-s81` | ow-zora, ow-moblin, ow-boulder, ow-lynel | main `r76-1k` 95.7% | 91.1% | **−4.6 ± 1.1** | 3.25 / 3.17 |
| post-d2, s81 + ow-zora `r76-zora-arrows` | ow-moblin, ow-boulder, ow-lynel | main `r76-1k` 95.7% | 94.0% | −1.7 ± 1.0 | 3.22 / 3.22 |
| d1 | stalfos-keese-gel, trap-room | main `r76-1k` 99.0% | 99.3% | +0.3 ± 0.4 | 6.00 / 6.00 |
| post-d1 | ow-zora, ow-leever | main `r76-1k` 100% | 100% | 0 | 5.17 / 5.12 (−0.05 ± 0.02) |
| d3 | zol, darknut, trap-room, boss-manhandla | main `r76-1k` 78.2% | 82.7% | **+4.5 ± 1.7** | 7.00 / 7.00 |
| post-d3 | ow-moblin, ow-zora | main `r76-1k` 100% | 100% | 0 | 6.77 / 6.69 (−0.09 ± 0.02) |
| d4 | vire, stalfos-keese-gel, zol, like-like, trap-room, boss-gleeok | main `r76-1k` 92.1% | 91.7% | −0.4 ± 1.2 | 8.00 / 8.00 |
| post-d4 (s81) | ow-zora, ow-moblin, ow-lynel, ow-leever | main `r76-1k` 99.9% | 100% | +0.1 ± 0.1 | 7.75 / 7.77 |
| d5 | darknut, zol, pols-voice | main `r76-1k` 68.7% | 72.2% | +3.5 ± 2.0 | 11.00 / 11.00 |
| post-d5 | ow-leever, ow-zora, ow-moblin | main `r76-1k` 98.7% | 84.0% | **−14.7 ± 1.2** | 8.24 / 8.59 |
| post-d5, ow-zora `r76-zora-arrows` | ow-leever, ow-moblin | main `r76-1k` 98.7% | 97.9% | −0.8 ± 0.5 | 8.21 / 8.96 (+0.75 ± 0.06) |
| d7 (300) | none | main `r76-1k` 39.0% (1000: 40.0%) | 39.0% | identical run for run | 12.00 / 12.00 |
| post-d7 | ow-moblin, ow-lynel | Dungeon 6 lane `r77-1k` 99.0% | 98.9% (`r79t5-lynel`) | −0.1 ± 0.4 | 10.31 / 10.53 (+0.22 ± 0.08) |
| post-d7, s81 (the ow-lynel lane's chain) | ow-moblin, ow-lynel | `r77-1k` 99.0% | 99.1% | +0.1 ± 0.4 | 10.30 / 10.71 (+0.40 ± 0.08) |
| d6 | stalfos-keese-gel, wizzrobe, zol, like-like | `r77-1k` 30.0% | 31.4% | +1.4 ± 2.0 | 13.00 / 13.00 |
| post-d6 (s81: the same) | ow-lynel, ow-moblin | Dungeon 8 lane `r78-dark-1k` 100% | 100% | 0 | 12.27 / 12.34 (+0.08 ± 0.02) |
| d8 | stalfos-keese-gel, equipment-room | `r78-dark-1k` 27.2% | 24.8% | −2.4 ± 1.9 | 14.00 / 14.00 |
| post-d8 | none (`r79t5-lynel`) | Dungeon 9 lane `r79c-1k` 99.7% | the same chain | 0 | 12.06 |
| post-d8, s81 (the ow-lynel lane's chain) | ow-lynel | `r79c-1k` 99.7% | 99.9% | +0.2 ± 0.1 | 12.06 / 12.21 (+0.15 ± 0.07) |
| d9 | none | `r79c-1k` 31.0% | the same chain | 0 | 3.14 |

The lead decided ow-lynel for the whole route while the census ran (`r27l-lynel-s81`, the joint ow-lynel run, from
`r79t5-lynel` with harvested second-visit states of 0:0x0a): segments already played with `r79t5-lynel` were not
all played again; post-d2 was, post-d4 was played with s81 (run for run equal to the ow-lynel lane's own s81 chain),
and post-d7, post-d6, post-d8 are that lane's 1000-run chains with s81 on the same team (goriya `r76-goriya3` there,
which plays none of those rooms).

The two segments whose families are all main's weights (d2: d2-rope, goriya, moldorm, boss-dodongo; d7: goriya,
boss-digdogger, equipment-room, dodongo-crossing, wallmaster, boss-aquamentus) were played 300 runs as an identity
check: every run equals main's `r76-1k` run of the same seed room for room (room, ending, steps, hearts out), so
main's 1000-run figures stand for this code line. `post-d8` and `d9` are this lane's `r79c-1k`, played by the same
weights for every family those segments play.

### What loses the two worse segments

**After Dungeon 2** (to 1:0x73), deaths per 1000 runs, candidate (`r79t5-lynel`) / with s81 / main:

| room | family (the reference's run) | deaths | hearts lost a visit |
|---|---|---|---|
| 0:0x0a, both visits (the white sword) | ow-lynel (`r74-lynel`) | 62 / 33 / 13 | visit 2: 0.37 / 0.25 / 0.10 |
| 0:0x1a after it | ow-tektite (main's `r74-leg2` in both): fewer hearts in | 33 / 16 / 9 | 0.33 / 0.35 / 0.31 |
| 0:0x27 | ow-zora (`r76-zora-arrows`) | 24 / 18 / 12 | 0.51 / 0.51 / 0.51, fewer hearts in |
| 0:0x38 | ow-zora (`r76-zora-arrows`) | 37 / 16 / 8 | 0.28 / 0.29 / 0.20 |

ow-moblin (0:0x4d, 0x3d) and ow-boulder (0x19, 0x18, 0x17) lose no run with either run. ow-zora also loses more
hearts in its rooms of the segments where nobody dies: 0:0x38 in post-d1 (0.33 against 0.24 a visit), 0:0x55 in
post-d3 (0.21 against 0.09) and post-d4 (20 wrong exits, 5 runs round the plain coast).

**After Dungeon 5** (to 7:0x79): 137 of 1000 runs end "left (0, 0x44) without the arrows": ow-zora `r79t5-zora` walks
out of 0x44 past the arrow shop (main's `r76-zora-arrows`, the run trained on the 75 harvested arrow-shop states:
0). Nothing dies. Beside it ow-leever at 0:0x0b: 27 wrong exits, 26 retreats, 17 runs ended by the bounce limit
(main 8, 4, 5).

### Better

- **d3 +4.5 ± 1.7**: Manhandla 3:0x4d 143 deaths against 165 (boss-manhandla `r78t4-manhandla2`, 0.34 hearts a
  visit against 0.44, with more hearts in), darknut's rooms 3:0x59, 0x69, 0x5c 15 deaths against 37 (darknut
  `r78-dark37-s85`).
- d5 +3.5 ± 2.0 (not beyond noise): 5:0x64 40 deaths against 51, 0x47 41 against 52, pols-voice 5:0x25 13 against
  25 (`r78t4-pols`); 0x66 53 against 41.
- Hearts: post-d7 +0.22 to +0.40 (ow-lynel 0:0x32 1.12 hearts a visit against 1.41), post-d8 +0.15 with s81.

## Family verdicts

One swap test for every family that loses a segment, and for ow-moblin and ow-boulder (named as causes by the
ow-lynel lane's 300-run screen): the segment again with only that family's run swapped for the reference's (1000
runs, paired with the candidate on the same seeds), and the swapped-in run on the family's start states of the
later stretches.

| family | segment, swap | paired against the candidate | the reference's run in the later stretches | verdict |
|---|---|---|---|---|
| ow-zora | post-d2 `r76-zora-arrows` | **+2.9 ± 1.1** (0:0x38 5 deaths against 16) | 0_27e 0/300 (wrong exit), 0_47e 0/300 (199 deaths, 84 wrong exits); `r79t5-zora` 99.95% | **neither** |
| ow-zora | post-d5 `r76-zora-arrows` | **+13.9 ± 1.1** (97.9%; main −0.8 ± 0.5) | (as above) | **neither** |
| ow-moblin | post-d2 `r76-b` | +1.8 ± 1.1 (within noise; its rooms lose no run with either) | 55.5%: 0_42c, 0_6dc, 0_6dn_v2 0/300 each; `r79t5-moblin` 100% | the newest serves |
| ow-boulder | post-d2 `r72-ow` | +0.1 ± 0.4 | 0_17s 0/300 (wrong exit); `r79b-boulder` 300/300 | the newest serves |
| ow-leever | post-d5 `r76-b` | +0.5 ± 1.4 (bounce limit 8 against 17, but 0:0x34 1.97 hearts a visit against 1.31) | 88.3%: 0_48w 70-91% (wrong exits); `r79t5-ow` 100% | the newest serves |
| ow-lynel | post-d2 | `r79t5-lynel` −11.8, s81 −4.6 with `r79t5-zora`, −1.7 ± 1.0 with `r76-zora-arrows` | `r74-lynel` 0/100 at 0:0x50 (the Dungeon 6 lane's baseline) | the joint run s81 serves (the lead's decision; its own lane: post-d2 −0.8 ± 0.9 against `r74-lynel` on fresh seeds, later stretches at least as good) |

The other shared families lose no segment, so the newest run serves the whole route: stalfos-keese-gel
`r79t5-skg` (d1, d4, d6, d8 within noise), like-like `r79c-like` (d4, d6), zol `r79b-zol` (d3 better, d4, d5, d6),
trap-room and vire `r77-b` (d1, d3, d4, d6), wizzrobe `r79c-wj-s85` (d6 +1.4 ± 2.0), darknut `r78-dark37-s85` (d3
better, d5, d8), boss-manhandla `r78t4-manhandla2` (d3 better, d8), boss-gleeok `r78t4-gleeok` (d4: Gleeok 76
deaths against 68, within noise), pols-voice `r78t4-pols` (d5), equipment-room main's `r76-c` (d8 against `r75-f-c`:
the cellar 8:0x2f 2 deaths both). Only one lane's families (ow-ghini, boss-gohma, lanmola, patra, ganon) and main's
(goriya, wallmaster, dodongo-crossing, boss-digdogger, boss-aquamentus) are their lanes' runs.

## The team to land with

Every family's run is the candidate's (the table at the top) with ow-lynel `r27l-lynel-s81`, except **ow-zora**: no
run of its lineage serves the whole route. `r79t5-zora` costs after Dungeon 2 about 3 points and the leg to
Dungeon 7 14.7; `r76-zora-arrows` cannot play the leg to Dungeon 9 (0x27 from E, 0x47 from E). The route lands with
the joint ow-zora continuation's run once it holds post-d2, post-d5 and post-d8; with it and s81 the measured
segments are within noise of their references or better (post-d2 −1.7 ± 1.0 and post-d5 −0.8 ± 0.5 with
`r76-zora-arrows`, which shows what the earlier segments get back).

## Owed a joint continuation

- **ow-zora**: `r79t5-zora` loses 0:0x44's arrows (post-d5, 137 of 1000 runs) and 0:0x38 (post-d2, 16 deaths
  against 8; more hearts in 0x38 and 0x55 elsewhere) / `r76-zora-arrows` loses 0:0x27 from E and 0:0x47 from E (the
  leg to Dungeon 9, 0 of 300 each). Lineage: `r79t5-zora` (244 iterations, demonstrations `tree.npz` of the
  Dungeon 9 lane's `target/r29-demos/`, harvested_fraction 0.03 of the 75 arrow-shop states in
  `triforce/states/harvested/ow-zora/`) <- `r76-zora-arrows` (244, no demonstrations, the 75 harvested arrow-shop
  states) <- `r76-b` <- `r74-leg`. Suggested (not tested): draw the arrow-shop states more often than 3% of the
  resets beside the leg to Dungeon 9's rooms and `tree.npz`.

No other family is owed one. ow-lynel's was owed and is done (`r27l-lynel-s81`); its 0:0x0a second visit still
loses more than main's run on seeds 0-999 (15-26 deaths against 4), within noise on the segment once ow-zora is
fixed.

## Reproducing

Every chain is `triforce eval-batch NAME --chain "<segment>: --from <state> [--route d6|d8] [--stop-at <arrival>]
--runs 1000 --act model --parallel 10 [--checkpoint FAMILY=<run>/ckpt.pt] --output ..."` with the promoted runs of
this branch's models directory; the single-room evaluations are `--evaluate "...: --specialist <family yaml with
only the later stretches' start states> --run-id RUN --episodes-per-state 100 --clips on"`.
