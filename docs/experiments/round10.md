# Round 10: experiments on the Rust stack

The first round on the new stack (hardware-timing zcore, regenerated states and trajectories, async training and
evaluation; `round8.md`, `round9.md`, `docs/performance.md`). The promoted team is unchanged from round 7.

## Summary

- **§1 Hardware-core chain baseline (120 runs per segment):** the team plays the hardware core and the
  regenerated states as well as it played FCEUmm. Round 9's lower 40-run chain counts were noise. No
  fine-tune needed (planned R10b dropped). The death rooms are the known ones: Goriya (D1 0x44, D2 0x5e and
  0x1e), D1 0x53, Aquamentus and D2's 0x3e.

## 1. Hardware-core chain baseline

Setup: promoted team, `triforce eval-chain --runs 120 --parallel 20 --device cpu --first-seed 100` per segment
(seeds 100–219, disjoint from the gate's 0–39), outputs `target/r10a/*.jsonl`, per-room tables from
`scripts/chain_rooms.py` (new: where runs end, mean health change, steps and worst quiet stretch per room).

| Segment | Round 7 §19 (FCEUmm, 40) | FCEUmm gate baseline (80) | Hardware + new states (120) |
|---|---:|---:|---:|
| Full milestone1 | 1/40 (2.5%) | 2/80 (2.5%) | 3/120 (2.5%) |
| Full milestone1-fairy | 3/40 (7.5%) | 5/80 (6.3%) | 6/120 (5.0%) |
| Post-D1 main (`--from 0_37c_2`) | 4/40 (10.0%) | 10/80 (12.5%) | 12/120 (10.0%) |
| Post-D1 fairy (`--from 0_37c_2`) | 7/40 (17.5%) | 12/80 (15.0%) | 18/120 (15.0%) |
| D2 (`--from 2_7ds_2`) | 19/40 (47.5%) | 27/80 (33.8%) | 44/120 (36.7%) |

Where runs end (rooms with at least one run ending there; outcome counts):

| Segment | Room | Visits | Ended there | Mean ΔHP | Worst quiet (steps) |
|---|---|---:|---:|---:|---:|
| Full main | 0:0x37 | 158 | 10 death | −0.16 | 426 |
| | 1:0x53 | 108 | 17 death | −0.67 | 56 |
| | 1:0x44 (Goriya) | 89 | 38 death, 1 stuck | −0.71 | 1,977 |
| | 1:0x35 (Aquamentus) | 46 | 10 death | +0.43 | 26 |
| | 2:0x5e (Goriya) | 21 | 9 death | −1.14 | 71 |
| Post-D1 fairy | 2:0x5e (Goriya) | 107 | 27 death | −0.86 | 1,413 |
| | 2:0x3e | 74 | 27 death | −0.80 | 131 |
| | 2:0x2e | 47 | 17 death | −0.06 | 133 |
| | 2:0x1e (Goriya) | 30 | 10 death, 2 stuck | −0.72 | 1,355 |
| D2 | 2:0x3e | 110 | 17 death | −0.74 | 65 |
| | 2:0x2e | 93 | 16 death | −0.23 | 444 |
| | 2:0x1e (Goriya) | 77 | 26 death, 7 stuck | −0.64 | 1,872 |

Decision: no fine-tune on hardware states (R10b dropped); the Goriya rooms remain the chain's main loss, as in
round 7. Next: item 8 (boomerang usable vs masked after D1).

## 2. Item 8 (r37): boomerang usable vs masked after D1

Setup (`scripts/experiments/r37_boomerang.sh`): every family that plays after D1 (ow-octorok, ow-zora,
ow-leever-peahat, ow-moblin, ow-start, d2-rope, goriya) continues from its promoted model for 245 updates
(1,003,520 steps), 10 envs, CUDA, training seed 37, in two arms trained side by side: **A** = the canonical yaml
(boomerang usable), **B** = `specialists/arms/r37-boomerang-off/` (`items: {bombs: false, boomerang: false}`).
Masking goriya also covers its D1 key room 0x23, where Link already has the boomerang; 0x44 is unaffected (no
boomerang until it's picked up there). Each arm is evaluated under its own config (20 episodes per start state)
and runs three post-D1 chain segments of 120 runs on seeds 100–219, the same seeds as §1's promoted baseline.

### Training seed 37

| Segment | Promoted (§1) | A: boomerang on | B: boomerang masked |
|---|---:|---:|---:|
| Post-D1 main | 12/120 | 7/120 | 5/120 |
| Post-D1 fairy | 18/120 | 10/120 | 13/120 |
| D2 | 44/120 | 37/120 | 47/120 |
| **Total** | 74/360 | 54/360 | 65/360 |

Evaluation over post-D1 start states only (`scripts/experiments/post_d1_eval.py`; success, mean real heart
change per episode):

| Specialist | Episodes | Promoted s0 | Promoted s1 | A: on | B: masked |
|---|---:|---:|---:|---:|---:|
| ow-octorok | 360 | 97.2% −0.12 | 96.7% −0.14 | 93.6% −0.22 | 97.8% −0.11 |
| ow-zora | 420 | 96.0% −0.14 | 96.7% −0.14 | 97.1% −0.19 | 94.8% −0.18 |
| ow-leever-peahat | 240 | 96.2% −0.37 | 97.1% −0.31 | 97.1% −0.21 | 98.3% −0.20 |
| ow-moblin | 360 | 96.4% −0.20 | 96.1% −0.22 | 97.5% −0.25 | 95.0% −0.21 |
| ow-start | 60 | 100.0% +0.93 | 100.0% +0.96 | 100.0% +0.93 | 100.0% +0.93 |
| d2-rope | 420 | 94.0% −0.27 | 93.8% −0.22 | 94.5% −0.28 | 92.1% −0.28 |
| goriya | 120 | 80.8% −0.80 | 82.5% −0.66 | 80.0% −0.73 | 80.8% −0.88 |
| **Pooled** | | 95.1% −0.21 | 95.3% −0.19 | 95.1% −0.23 | 94.5% −0.20 |

Reading: on post-D1 states the arms are indistinguishable (pooled 95.1% vs 94.5%, health −0.23 vs −0.20). The
chain gap comes from single-specialist training drift, not from the boomerang: arm A's ow-octorok regressed
(93.6% post-D1) and in the chain 0:0x4c ends 35 of arm A's post-D1 main runs (25 deaths, 10 stuck, 286 mean
steps) against 10 for the promoted team; arm B's ow-moblin exits 0x5c the wrong way 5 times on the fairy route.
Both continued teams also trail the untouched promoted team in the chains, so a 1M-step continuation on the new
states is itself a noisy step. A second training seed per arm (seed 2) measures that noise before any conclusion.

### Training seed 2 and the decision

| Segment | Promoted (§1) | A s37 | A s2 | B s37 | B s2 | A pooled | B pooled | Fisher p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Post-D1 main | 12/120 | 7 | 9 | 5 | 14 | 16/240 (6.7%) | 19/240 (7.9%) | 0.73 |
| Post-D1 fairy | 18/120 | 10 | 19 | 13 | 13 | 29/240 (12.1%) | 26/240 (10.8%) | 0.77 |
| D2 | 44/120 | 37 | 38 | 47 | 39 | 75/240 (31.2%) | 86/240 (35.8%) | 0.33 |
| **Total** | 74/360 (20.6%) | 54 | 66 | 65 | 66 | 120/720 (16.7%) | 131/720 (18.2%) | 0.49 |

Post-D1 evaluation (success, mean real heart change):

| Specialist | A s37 | B s37 | A s2 | B s2 |
|---|---:|---:|---:|---:|
| ow-octorok | 93.6% −0.22 | 97.8% −0.11 | 97.8% −0.14 | 89.7% −0.17 |
| ow-zora | 97.1% −0.19 | 94.8% −0.18 | 95.0% −0.21 | 97.1% −0.16 |
| ow-leever-peahat | 97.1% −0.21 | 98.3% −0.20 | 99.2% −0.14 | 96.2% −0.21 |
| ow-moblin | 97.5% −0.25 | 95.0% −0.21 | 95.8% −0.24 | 93.9% −0.23 |
| ow-start | 100.0% +0.93 | 100.0% +0.93 | 100.0% +0.93 | 100.0% +0.93 |
| d2-rope | 94.5% −0.28 | 92.1% −0.28 | 93.8% −0.29 | 93.8% −0.26 |
| goriya | 80.0% −0.73 | 80.8% −0.88 | 78.3% −0.78 | 71.7% −1.06 |
| **Pooled** | 95.1% −0.23 | 94.5% −0.20 | 95.1% −0.21 | 92.9% −0.22 |

**Decision: keep the boomerang usable; nothing promoted.** Masking it after D1 makes no detectable difference in
the chains (pooled 18.2% vs 16.7%, p = 0.49; the D2 segment leans to B, p = 0.33) and is slightly worse on
post-D1 evaluations (one B seed's goriya 71.7% / −1.06, ow-octorok 89.7%). Training-seed swings within a
specialist (ow-octorok 93.6–97.8% on A, 89.7–97.8% on B) are larger than the arm difference, and both continued
teams trail the untouched promoted team (74/360), so neither a 1M-step continuation nor the mask is an
improvement. This first pass warm-started both arms from boomerang-trained models, which favours A; the
item's preferred from-scratch comparison is recorded in `todo.md` with its cost.

## 3. Swordless departures from the start screen (r38)

Round 10 §1 found 10 of 120 full chains per route (8.3%, the same seeds on both routes) leaving 0x77 north
without the sword; every one of them failed (on milestone1 they are all 10 deaths in 0:0x37). ow-start's
evaluation agreed: `start` failed 4 of 60 sampled episodes (`exit-wrong`), its other states almost never.

- **r38-start-only:** continue the promoted ow-start (r24-fairy) with `start` as its only start state
  (`specialists/arms/r38-start-only/ow-start.yaml`), 16 envs, `--stop-success 0.99` (stopped at 34 updates,
  139k steps). `start` 60/60, but the cave re-entry states `0_77c_*` dropped to 83–87% (`stuck`): training one
  state let the others drift. Not kept.
- **r38b-rebalance:** continue r38 on the canonical config (all ow-start states; the selector already doubles
  the weight of failing states), `--stop-success 0.99` (stopped at 6 updates, 24k steps).

| ow-start, 60 episodes per state | Overall | `start` | `0_77c_*` | Health change |
|---|---:|---:|---:|---:|
| r24-fairy (promoted) | 99.2% | 56/60 | 180/180 | +0.315 |
| r38-start-only | 95.2% | 60/60 | 152/180 | +0.309 |
| r38b-rebalance | 100.0% | 60/60 | 180/180 | +0.315 |

Chains, 120 runs on §1's seeds, only ow-start swapped:

| Segment | Swordless (promoted) | Swordless (r38b) | Success (promoted) | Success (r38b) |
|---|---:|---:|---:|---:|
| Full milestone1 | 10 | 0 | 3/120 | 7/120 |
| Full milestone1-fairy | 10 | 0 | 6/120 | 8/120 |
| Post-D1 fairy (ow-start plays the pond) | — | — | 18/120 | 18/120 |

**Promoted r38b-rebalance** for ow-start: no swordless departures left in 240 full chains, the pond unchanged,
and the full chains gain what the swordless runs used to lose. Lee's alternative (end training episodes on a
swordless departure) wasn't needed: the existing `exit-wrong` ending plus targeted training was enough.
