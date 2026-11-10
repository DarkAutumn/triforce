# Round 17: the code-review fix-up, then a fresh team (r61)

Lee, Nov 10: work through every finding of the whole-repo review (`target/review-ledger.md`, 192 findings, read with
`target/review-guide.md`): fix it, or close it with a recorded reason (`target/review-progress.md`). Then retrain the
whole team from scratch with SIL on from the start (`--sil-coeff 0.1`). No continuation of the current models.

## Summary

- [in progress] Review fixes, phase by phase (the plan's order: measurement first, then training loop, rewards and
  endings, masks, handoff state, start states, tools; docs last).
- **Baseline B1** (§1): the promoted team (r56 + r57-sil-s60, their act modes) measured on the fixed measurement
  code. Single rooms match round 16; chains match `r60-sil` except where the new per-run wrong-exit cap (F157) ends
  the 0x0a ↔ white-sword-cave bounce (F094) as a failure: after D2 98 → 34 arrivals, full route to D1 57 → 14.

## 1. Baseline B1 (review Phase 2)

Code: commit `42a2ba2` (every Phase 2 fix: act-mode-checked evaluations, `start_max_health`, cellar room keys, the
mid-route triforce as a correct exit, the wrong-exit cap, validated chain inputs, chain provenance; no reward,
ending or mask change yet), built in the `../triforce2-better` worktree. Models: the promoted team with its act
files (15 argmax, 5 sample). Canonical evaluations (each model's act mode, 100 episodes per start state, seed 0)
written as each run's `eval.json`; chains `ACT=model TAG=r61-b1 scripts/experiments/r56_chains.sh` (120 runs per
segment, seeds 0-119); yardstick `target/r61/b1-yardstick.json`.

| specialist | act | episodes | success | hearts lost / visit | health change |
|---|---|---:|---|---|---|
| boss-aquamentus | argmax | 300 | 100.0% | 0.25 | +0.92 |
| boss-dodongo | argmax | 300 | 99.3% | 0.01 | +0.99 |
| boss-manhandla | argmax | 300 | 88.7% | 1.50 | −0.38 |
| d2-rope | argmax | 1800 | 98.7% | 0.07 | +0.30 |
| darknut | sample | 2100 | 96.7% | 0.62 | −0.55 |
| equipment-room | argmax | 600 | 99.8% | 0.19 | −0.19 |
| goriya | sample | 1500 | 94.4% | 0.52 | −0.43 |
| moldorm | sample | 300 | 85.7% | 0.87 | −0.32 |
| ow-boulder | argmax | 900 | 100.0% | 0.14 | −0.14 |
| ow-leever | argmax | 300 | 100.0% | 0.06 | −0.05 |
| ow-lynel | argmax | 900 | 99.3% | 0.21 | −0.21 |
| ow-moblin | argmax | 2400 | 97.9% | 0.15 | −0.14 |
| ow-octorok | argmax | 3900 | 98.6% | 0.10 | −0.09 |
| ow-start | argmax | 1300 | 100.0% | 0.00 | 0.00 |
| ow-tektite | argmax | 1800 | 99.7% | 0.12 | +0.06 |
| ow-zora | argmax | 3000 | 99.3% | 0.16 | −0.15 |
| stalfos-keese-gel | sample | 4200 | 99.8% | 0.06 | +0.06 |
| trap-room | argmax | 1500 | 99.0% | 0.34 | −0.34 |
| wallmaster | argmax | 300 | 99.7% | 0.00 | +0.33 |
| zol | sample | 1800 | 99.8% | 0.10 | +0.32 |

Chains (arrivals of 120; `r60-sil` is round 16 §7, the same team before the fixes):

| segment | B1 | r60-sil | where B1 runs end |
|---|---:|---:|---|
| full → D2 (2:0x7d) | 110 | 110 | 0:0x5b death ×7, 0:0x4c death ×3 |
| full → after D2 (0:0x3c) | 72 | 72 | 2:0x3e death ×11, 2:0x5e death ×8, 2:0x1e death ×6 |
| full → D1 (1:0x73) | 14 | 57 | 0:0x0a wrong-exit limit ×47 |
| full → after D1 (0:0x37) | 6 | | |
| full → D3 (3:0x7c) | 6 | 34 | |
| full → D3 triforce | 0 | 4 | |
| D2 (`2_7ds_0`) | 89 | 89 | 2:0x3e death ×11, 2:0x4e stuck ×10, 2:0x1e death ×7 |
| after D2 (`0_3cc_0`) | 34 | 98 | 0:0x0a wrong-exit limit ×77 |
| D1 (`1_73s_0`) | 115 | 115 | five single failures |
| after D1 (`0_37c_0`) | 120 | 120 | |
| D3 (`3_7cs_0`) | 30 | 30 | 3:0x59 death ×40, 3:0x69 death ×21, 3:0x4d death ×10 |

Every segment that does not cross 0x0a is unchanged. The difference is one mechanism: after the white sword cave the
ow-lynel model (argmax) walks back into the cave and out again (F094: the cave-exit flag is lost across loads, so
the training mask and the chain mask disagree there); before, the chain let it bounce until it happened to move on
(6-182 bounces, all runs moved on, about 40% of the segment's actions); with F157's cap the run ends at the 20th
wrong exit (`wrong-exit-limit: ow-lynel left into 0:0x0a`). F094 itself is fixed in Phase 5 and its model side by
the retrain. Measurement fixes that did not move these numbers: F135 (the D2/D1 triforce rooms now record
`exit-correct`), F109 (cellar keys), F095 (start health fraction of boss rooms), F036 (milestone records).
