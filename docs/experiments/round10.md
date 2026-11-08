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

## 4. D2 key carriage audit (r36 §21 follow-up)

Chain room records now carry `keys_in`/`keys_out` (keys on entering and leaving each room), and `chain_summary`
reports `arrival_keys`: successful runs counted by the keys Link carried into the Dodongo room (3 = every key on
the route kept: 7e, 4e and 3e). Milestone arrivals and full key carriage are separate counters from now on.

Current promoted team (with r38b ow-start), 120 runs per segment, seeds 100–219:

| Segment | Dodongo arrivals | 3 keys | 2 keys |
|---|---:|---:|---:|
| Post-D1 main | 12/120 | 12 | 0 |
| Post-D1 fairy | 18/120 | 16 | 2 |
| D2 | 44/120 | 44 | 0 |

72 of 74 arrivals carry all three keys; the two short ones are on the fairy route. Full key carriage is not
where the team loses: reaching Dodongo is (Goriya rooms and 0x3e, §1).

## 5. Goriya rooms: arrival health, and chain-matched hp done right (r39)

**Where the chain loses.** With the current team, D2's Goriya room 2:0x5e kills 38 of 87 arrivals on post-D1 main
and 27 of 107 on post-D1 fairy, but only 4 of 119 in the D2 segment (which starts at full health). Arrival health
decides it:

| 2:0x5e deaths by arrival hearts (of 4) | ≤1 | 1–2 | 2–3 | >3 |
|---|---:|---:|---:|---:|
| Post-D1 main | 24/28 | 14/42 | 0/12 | 0/5 |
| Post-D1 fairy | 11/13 | 9/33 | 6/40 | 1/21 |
| D2 segment | — | 0/2 | 1/13 | 3/104 |

2:0x1e (Goriya) and 2:0x3e/2:0x2e (d2-rope) show the same slope. Post-D1 main leaves 0x37 with 4.00 hearts and
reaches 2:0x5e with 1.64 on average; the overworld costs ~2.3 hearts spread over 0x2b, 0x3d, 0x4c, 0x2a and 0x2d,
where almost no hearts drop. The fairy route arrives with 2.39. At full health Link fires sword beams; below it
he must melee five Goriyas.

**r39.** Round 6's chain-hp Goriya test (r21) never applied its treatment (round 7 §4), so chain-matched hp had
not been tested on Goriya. `triforce/chain_hp.json` was refreshed from the current team's full and post-D1 chains
(2:0x5e mean 2.15 hearts, 20% ≤1; the old profile mixed in D2-segment full-health arrivals, 2.66). Arms continue
r31-block1 for 245 updates, 10 envs, two training seeds each: `ctl` (canonical config) and `chp`
(`ram_randomization.chain_hp: 0.5`, `specialists/arms/r39-goriya-chainhp/`). Smoke before training: the chain-hp
eval config starts 1:0x44 at 1.59 hearts (7% full) vs 2.12 (45%) canonical. Runner:
`scripts/experiments/r39_goriya_chainhp.sh`; outputs `target/r39/`.

Evaluation, 40 episodes per state (success, mean heart change):

| Model | Canonical hp mix | Chain hp (`chain_hp: 1.0`) | 1:0x44 at chain hp | 2:0x5e at chain hp |
|---|---:|---:|---:|---:|
| r31-block1 (promoted) | 78.1% −0.72 | 68.5% −0.71 | 51.7% −0.65 | 78.3% −0.81 |
| ctl s39 | 72.1% −1.10 | 63.3% −1.03 | 60.8% −0.75 | 75.8% −0.91 |
| ctl s2 | 80.0% −0.80 | 70.6% −0.76 | 64.2% −0.68 | 79.2% −0.81 |
| chp s39 | 78.1% −0.87 | 67.9% −0.84 | 54.2% −0.87 | 72.5% −0.95 |
| chp s2 | 72.1% −0.90 | 60.2% −0.91 | 56.7% −0.69 | 69.2% −0.87 |

Chains, 120 runs per segment, seeds 100–219, only goriya swapped (promoted = §3/§4 runs of the current team):

| Model | Full main | Full fairy | Post-D1 main | Post-D1 fairy | D2 | Total | Goriya-room deaths at ≤2 hearts |
|---|---:|---:|---:|---:|---:|---:|---:|
| promoted | 7 | 8 | 12 | 18 | 44 | 89/600 | 219/399 (55%) |
| ctl s39 | 7 | 11 | 5 | 22 | 40 | 85/600 | 219/431 (51%) |
| ctl s2 | 6 | 12 | 9 | 10 | 39 | 76/600 | 233/438 (53%) |
| chp s39 | 2 | 13 | 9 | 22 | 36 | 82/600 | 225/413 (54%) |
| chp s2 | 3 | 10 | 8 | 18 | 41 | 80/600 | 210/418 (50%) |

**No effect; nothing promoted.** chp 162/1200 vs ctl 161/1200; the low-health death rate is 52% in both arms; at
its own target (chain-hp evaluation) chp is no better (64.1% vs 67.0% pooled). Training at chain health does not
teach the Goriya model to survive five Goriyas without beams. The lever is arrival health (overworld damage, and
the hearts d2-rope leaves on the floor in 2:0x7e/0x4e: 0.36 and 0.39 per visit), or a different fighting skill,
not the reset health distribution. The refreshed `chain_hp.json` stays (no canonical config uses it).

## 6. Finite stall escape (Lee, Nov 4)

Lee watched Goriya runs attacking while standing still indefinitely and asked for a finite timeout in training
and in `play-zelda.py`, counting real damage either way. Implemented as one counter (actions in a room since
entry or the last damage dealt or taken; `docs/decisions.md` 2026-11-06, `docs/architecture.md`).

Threshold, from 480 chains of the current team (seeds 100–219: full main 7, full fairy 8, post-D1 main 12, D2 44
successes — identical to §3/§4, so recording the counter changes nothing):

| Specialist | Successful visits | p50 | p99 | Max |
|---|---:|---:|---:|---:|
| goriya | 409 | 26 | 342 | 1,273 |
| d2-rope | 1,355 | 12 | 110 | 2,686 |
| wallmaster | 118 | 42 | 179 | 189 |
| stalfos-keese-gel | 2,306 | 12 | 70 | 623 |
| ow-octorok | 1,669 | 10 | 28 | 697 |
| others (ow-zora, ow-moblin, ow-leever-peahat, ow-start, boss-aquamentus) | 3,176 | ≤ 14 | ≤ 59 | ≤ 74 |

16 of 9,033 successful visits reached 300 and 10 reached 600 (slow d2-rope wanders in 2:0x6e, Goriya 2:0x1e); all
18 `stuck` visits reached ≥ 638 (most ran to the 2,000/3,000-step budget).

- **Player:** `--stall-steps 300` (default): eligible for replacement at 300, run ends `stuck` at 600.
- **Training/eval:** `no_damage_steps` default 600, a bootstrapped truncation (`no-damage`, soft failure). Goriya
  r31-block1 eval (40 per state, seed 0): 78.1% → 77.7%, −0.72 → −0.71, mean episode 134 → 117 actions; 12 of 480
  episodes end earlier (8 stuck/max-steps, 2 deaths, 2 slow successes). Evaluations from here on include it;
  earlier ones did not (the difference is these rare long stretches).

## 7. zcore speed and the `--parallel` study (Lee, Nov 7)

Full write-up: `docs/performance.md` "zcore speed (Nov 7)" and "How many envs (`--parallel`)".

- **Emulator:** six kept changes (`bd71234`, `285bc5f`, `10089fb`, `415124c`, `62377ce`, `76272ea`), each through
  the accuracy gate (goldens 0/63, 481-state walk 0 differ, seeded training digest identical). Env benchmark over
  every start state: 2,832 → 4,526 frames/s (1.60×; ow-moblin 1.38× to ow-start 1.94×), with the same 166,577
  frames and 10,811 steps played. Core bench over 20 recorded runs since `3de6f4b`: headless 5.5k → 6.7k, rendered
  2.6k → 3.4k frames/s. Training (`train_bench.sh`, historical env counts) vs Nov 5: goriya p8 1,246 → 1,624,
  d2-rope p8 1,119 → 1,466, ow-moblin p12 ~1,580 → 2,093, ow-zora p16 1,631 → 2,177 steps/s.
- **Not kept:** deferring cut-short headless fetches (+5% cycles), skipping the PPU sync for OAM DMA's VBlank puts
  (+1%), generalizing the headless fetch skip (0). The profile is flat at 1.5M host instructions per frame, IPC 2.48.
- **`--parallel`** (`scripts/parallel_study.py nov7`): warm-start throughput for ow-moblin, goriya and ow-zora at
  4–32 envs rises near-linearly to 10 and flattens at 16–20 (2.2–2.5k steps/s). From scratch to 90% balanced
  success (ow-leever-peahat and ow-moblin, 4/8/12/16/24 envs, seeds 1–3, 30/30 reached): the steps don't depend on
  the env count (medians 209–261k and 315–426k), and wall time is lowest at 12–16 (139–145 s and 188–209 s; at 4
  envs 349 s and 589 s). Recommendation: `--parallel 16` alone, 8 each for two rounds side by side.
