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
