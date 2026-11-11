# Round 17: the code-review fix-up, then a fresh team (r61)

Lee, Nov 10: work through every finding of the whole-repo review (`target/review-ledger.md`, 192 findings, read with
`target/review-guide.md`): fix it, or close it with a recorded reason (`target/review-progress.md`). Then retrain the
whole team from scratch with SIL on from the start (`--sil-coeff 0.1`). No continuation of the current models.

## Summary

- **Review fixes: done.** 186 findings fixed, 5 closed as duplicates, 1 (F111) closed by Lee's decision (docs and a
  backlog experiment, no code); 6 new findings (N001–N006) found while fixing, 4 fixed, 2 closed as probe artifacts.
  Per-finding commits in `target/review-progress.md`; the changes that matter for training are in §2. Suites after
  the last code fix: Rust 699 pass (also with overflow checks), Python 450 pass, `tf ladder-diff` 0 of 7 diverge.
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

## 2. What the fixes change for training

Every change below is in the code the r61 team trains on (§3); none of the current models was retrained on it.

- **What a model sees or may do** (guide §8 class 1; the reason for the retrain from scratch):
  - one object classifier from the game's jump table: hazards (boulders and their generator, traps, Bubbles, the
    standing fire) are not enemies or fight targets; people and tile objects are in no list (F067, F038);
  - attacks are legal only where the game starts the weapon (busy slot, blocked sword `$4C`/`$52E`, Link's state,
    BorderBounds measured at every coordinate, no attack across the doorway Link stands in) (F182, F105, F183,
    N006); an off-grid perpendicular attack first walks to the grid point (F160); the attack press is held a second
    frame on a lag frame (N003);
  - BOMB legal only in `requires_bombs` rooms (F016) and at an unopened overworld rock wall (F186); MOVE N scrolls
    at the overworld top edge (F059); keys only at the locked door's bound (F170); pushes only when aligned in a
    cleared room (F188); an opened bomb wall is passable (F124);
  - `just_exited_cave` on cave and dungeon mouths, the same for loaded and live states (F094);
  - `target_*` follows legal steps only (F132); the selected-B-item slice is always zero (F187); chains reset the
    observation on every one_room entry (F040, F101).
- **Rewards, endings, start draws** (class 2): self-destroyed boulders and dived Zoras are no kills (F038);
  look-ahead credit is marginal over pending predictions and a finished weapon's leftovers drop (F054); the
  no-progress limit follows `max_steps` (F078); no map/compass reward on a level change (F043); item lifetimes count
  half-frames (F065); the quiet-enemy penalty counts 0-HP monsters (F122); a bomb pickup pays 0.5 (F092); linger
  rebaselines when its wavefront is blind or the targets change (F154); a treasure counts as taken by its room flag
  (F026); curriculum starts take their track's route step (F151, F189); the start-state selector weights states by
  decayed failure rate with a 15% uniform floor and starts at a random offset (F185, F158); harvested starts are
  opt-in (`harvested_fraction` 0) (F069).
- **The training loop:** early stop on exact per-state cohort counts over seed states only, with guards and every
  candidate stop confirmed by a 50-episode-per-state sampled evaluation of the current weights (F130, F071, F140);
  minibatches by `tensor_split` (F167); all-or-nothing updates (F084); the recompute on the GPU (F019); a `--load`
  continuation resumes its curriculum (F088); a reused run dir is moved aside (F172); search simulations act by
  argmax (F180).
- **Measurement:** canonical `eval.json` only in the model's act mode, `act` files bound to the checkpoint's sha1,
  every reader checks the mode (F027, F125, F064); chain results carry provenance and never append silently (F080,
  F063); a mid-route triforce is a correct exit (F135); a chain run ends after 20 wrong exits into tagged rooms
  (F157); `--promote` is opt-in (F025).
- **Data:** game.yaml's dungeon exits follow one convention checked against the ROM (F030); enemy names match the
  ROM (F070); no chain-hp profile ships (F042, F133); harvested and built states no longer keep a granted bomb
  (F148).
