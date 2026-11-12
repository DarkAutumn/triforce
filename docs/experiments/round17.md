# Round 17: the code-review fix-up, then a fresh team (r61)

Lee, Nov 10: work through every finding of the whole-repo review (`target/review-ledger.md`, 192 findings, read with
`target/review-guide.md`): fix it, or close it with a recorded reason (`target/review-progress.md`). Then retrain the
whole team from scratch with SIL on from the start (`--sil-coeff 0.1`). No continuation of the current models.

## Summary

- **Review fixes: done.** 186 findings fixed, 5 closed as duplicates, 1 (F111) closed by Lee's decision (docs and a
  backlog experiment, no code); 6 new findings (N001–N006) found while fixing, 4 fixed, 2 closed as probe artifacts.
  Per-finding commits in `target/review-progress.md`; the changes that matter for training are in §2. Suites after
  the last code fix: Rust 699 pass (also with overflow checks), Python 450 pass, `tf ladder-diff` 0 of 7 diverge.
- **Data rebuilt on the fixed code:** all 294 route and alternate entry states (staged builds, restarts for chains
  0/1/2 at D2's 0x2e/0x3e/0x1e; audit clean; 180 states lose the granted bomb that used to leak in, F148) and all 203
  curriculum tracks of the 14 curriculum families (old tracks in `trajectories.prev-r60`). Four tests that leaned on
  the old states' layouts were made layout-independent; one real bug surfaced (N007: a 32-frame step can span a
  Zora's surface-and-dive cycle, so a surfaced Zora that vanished was still counted as killed).
- **Re-profile before the retrain (Lee):** no hotspot from the fixes (code above the emulator stays ~1% of cycles;
  +1.4 frames per sword/beam action and real bomb/boomerang throws by design). Three wins taken: the SIL replay ring
  on the GPU (`--parallel 16` SIL training ~+45%), the next rollout starts before the last one's buffer is
  prepared (+2–5%), no weapon look-ahead without a live target; and a failing early-stop confirmation ends as soon
  as it cannot pass (~45% of its time). Details: `docs/performance.md` "Review fix-up re-profile (Nov 10)".
- **Baseline B1** (§1): the promoted team (r56 + r57-sil-s60, their act modes) measured on the fixed measurement
  code. Single rooms match round 16; chains match `r60-sil` except where the new per-run wrong-exit cap (F157) ends
  the 0x0a ↔ white-sword-cave bounce (F094) as a failure: after D2 98 → 34 arrivals, full route to D1 57 → 14.
- **r61, the fresh team (§3–§5, promoted):** all 20 families trained from scratch on the fixed code with SIL, two
  streams at `--parallel 8`, about 8 h wall. Single rooms level with the old team (boss-manhandla 88.7% → 95.7%;
  goriya, stalfos-keese-gel and zol 0.7–1.0 points lower and losing more hearts). Act modes: argmax for 16 families,
  sampling for stalfos-keese-gel and zol (argmax 3–6 points worse) and for darknut and d2-rope (argmax loops in
  chains). Chains, paired by seed against B1: D2 89 → 100, after D2 34 → 92, full route to D1 14 → 55, D3 30 → 23
  (p = 0.32), **D1 115 → 80** (39 lost, 4 gained: stalfos-keese-gel dies in 1:0x43, goriya in 1:0x23/0x44,
  Aquamentus in 1:0x35), full route to the D3 triforce 1/120.
- **Next:** the D1 families. stalfos-keese-gel and zol early-stopped after 4–5 minutes (80–96 iterations): the
  confirmed 99% success says nothing about hearts lost, which the chain pays for (backlog).

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
    `asm:Link_BorderBounds` measured at every coordinate, no attack across the doorway Link stands in) (F182, F105, F183,
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

## 3. r61: training

Code at `5bec275`; states and curricula as rebuilt (§ Summary). `scripts/train_round.py --steps 8000000 --parallel 8
--eval-parallel 3 --chain-runs 0 --train-args "--stop-success 0.99 --sil-coeff 0.1"` in two streams (`r61-ow`: the
ten overworld and D1-side families; `r61-dg`: the dungeon families). When `r61-ow` finished (4.4 h), a watcher
(`/tmp/r61-rebalance.py`) started `r61-dg2` on the later half of `r61-dg`'s pending families and stopped `r61-dg`
when it reached one of them; a second split gave darknut its own stream (`r61-dg3`). Every stopped training was
moved aside as `<run>.partial-killed` and trained again by the other stream; three evaluations stopped with their
driver were redone below. Wall time from start to the last family: about 8 h.

Act modes from 100 episodes per start state in each mode (seed 0, paired by start state and seed); the rule:
argmax unless the paired difference's 95% CI upper bound is below 0, then the chain check of §4.

| family | run | ending | iterations | steps | wall | sample | argmax | Δ (95% CI) | act |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| ow-start | r61-ow | early_stop | 43 | 0.18M | 2 min | 99.7% | 100.0% | +0.3% [+0.0%, +0.6%] | argmax |
| ow-octorok | r61-ow | early_stop | 379 | 1.55M | 19 min | 98.5% | 98.6% | +0.1% [-0.2%, +0.4%] | argmax |
| ow-zora | r61-ow | early_stop | 379 | 1.55M | 19 min | 98.9% | 99.3% | +0.5% [+0.2%, +0.8%] | argmax |
| ow-moblin | r61-ow | completed | 1954 | 8.00M | 93 min | 98.0% | 98.4% | +0.3% [-0.1%, +0.8%] | argmax |
| ow-tektite | r61-ow | early_stop | 283 | 1.16M | 12 min | 99.2% | 99.4% | +0.3% [-0.2%, +0.7%] | argmax |
| ow-lynel | r61-ow | early_stop | 1706 | 6.99M | 95 min | 98.7% | 98.9% | +0.2% [-0.6%, +1.0%] | argmax |
| ow-boulder | r61-ow | early_stop | 24 | 0.10M | 1 min | 100.0% | 99.9% | -0.1% [-0.3%, +0.1%] | argmax |
| ow-leever | r61-ow | early_stop | 40 | 0.16M | 2 min | 100.0% | 100.0% | +0.0% [+0.0%, +0.0%] | argmax |
| trap-room | r61-ow | early_stop | 165 | 0.68M | 10 min | 99.6% | 99.7% | +0.1% [-0.3%, +0.5%] | argmax |
| equipment-room | r61-ow | early_stop | 133 | 0.54M | 6 min | 97.5% | 99.0% | +1.5% [+0.2%, +2.8%] | argmax |
| d2-rope | r61-dg | early_stop | 1359 | 5.57M | 66 min | 98.3% | 97.9% | -0.4% [-1.1%, +0.3%] | sample (§4) |
| goriya | r61-dg | completed | 1954 | 8.00M | 88 min | 92.3% | 93.0% | +0.7% [-0.7%, +2.1%] | argmax |
| moldorm | r61-dg | completed | 1954 | 8.00M | 83 min | 85.7% | 86.7% | +1.0% [-2.6%, +4.6%] | argmax |
| boss-dodongo | r61-dg | early_stop | 563 | 2.31M | 29 min | 99.7% | 99.0% | -0.7% [-2.0%, +0.6%] | argmax |
| stalfos-keese-gel | r61-dg | early_stop | 80 | 0.33M | 4 min | 99.1% | 95.8% | -3.3% [-4.0%, -2.7%] | sample |
| wallmaster | r61-dg | early_stop | 244 | 1.00M | 12 min | 99.7% | 100.0% | +0.3% [-0.3%, +1.0%] | argmax |
| boss-aquamentus | r61-dg | early_stop | 225 | 0.92M | 10 min | 97.3% | 100.0% | +2.7% [+0.8%, +4.5%] | argmax |
| darknut | r61-dg3 | completed | 1954 | 8.00M | 93 min | 96.4% | 97.1% | +0.7% [-0.1%, +1.5%] | sample (§4) |
| zol | r61-dg2 | early_stop | 96 | 0.39M | 5 min | 98.8% | 93.1% | -5.8% [-7.0%, -4.5%] | sample |
| boss-manhandla | r61-dg2 | completed | 1954 | 8.00M | 114 min | 93.3% | 95.7% | +2.3% [-0.7%, +5.3%] | argmax |

Every early stop was confirmed by its evaluation (50 sampled episodes per seed state); the log shows many aborted
checks at 0.984–0.989 before the one that passed. Each run's `eval.json` is now its canonical evaluation in its
act mode; both modes' files are kept as `eval-r61-sample.json` / `eval-r61-argmax.json`.

## 4. Act modes in chains

With the rule's modes (18 argmax, 2 sample) the chain segments (`TAG=r61-mix`) showed `stuck` loops: darknut 26
(3:0x59), d2-rope 6, goriya 5. Plan rule: more than 5 `stuck`/`no-damage` endings across the six segments sends a
family back to sampling: darknut and d2-rope (promoted again with `--act sample`). `TAG=r61-mix2`: goriya 5,
darknut 2, ow-octorok 1, boss-aquamentus 1 stuck; no family above 5. Final: argmax for 16 families, sampling for
d2-rope, darknut, stalfos-keese-gel and zol (`target/r61/r61-modes.txt`; the team before is in
`target/r61/promoted-before-r61.txt`).

## 5. r61 against B1

Single rooms (canonical evaluations, each model's act mode; B1 on the old states, r61 on the rebuilt ones):

| specialist | B1 (act) | B1 success | B1 hearts lost | r61 (act) | r61 success | r61 hearts lost |
|---|---|---:|---:|---|---:|---:|
| boss-aquamentus | r56-rust-dg (argmax) | 100.0% | 0.25 | r61-dg (argmax) | 100.0% | 0.25 |
| boss-dodongo | r56-rust-dg (argmax) | 99.3% | 0.01 | r61-dg (argmax) | 99.0% | 0.08 |
| boss-manhandla | r56-rust-dg3 (argmax) | 88.7% | 1.50 | r61-dg2 (argmax) | 95.7% | 1.22 |
| d2-rope | r57-sil-s60 (argmax) | 98.7% | 0.07 | r61-dg (sample) | 98.3% | 0.10 |
| darknut | r57-sil-s60 (sample) | 96.7% | 0.62 | r61-dg3 (sample) | 96.4% | 0.65 |
| equipment-room | r57-sil-s60 (argmax) | 99.8% | 0.19 | r61-ow (argmax) | 99.0% | 0.23 |
| goriya | r57-sil-s60 (sample) | 94.4% | 0.52 | r61-dg (argmax) | 93.0% | 0.45 |
| moldorm | r56-rust-dg (sample) | 85.7% | 0.87 | r61-dg (argmax) | 86.7% | 0.71 |
| ow-boulder | r57-sil-s60 (argmax) | 100.0% | 0.14 | r61-ow (argmax) | 99.9% | 0.17 |
| ow-leever | r56-rust-ow (argmax) | 100.0% | 0.06 | r61-ow (argmax) | 100.0% | 0.04 |
| ow-lynel | r57-sil-s60 (argmax) | 99.3% | 0.21 | r61-ow (argmax) | 98.9% | 0.13 |
| ow-moblin | r57-sil-s60 (argmax) | 97.9% | 0.15 | r61-ow (argmax) | 98.4% | 0.12 |
| ow-octorok | r57-sil-s60 (argmax) | 98.6% | 0.10 | r61-ow (argmax) | 98.6% | 0.10 |
| ow-start | r56-rust-ow (argmax) | 100.0% | 0.00 | r61-ow (argmax) | 100.0% | 0.00 |
| ow-tektite | r56-rust-ow (argmax) | 99.7% | 0.12 | r61-ow (argmax) | 99.4% | 0.19 |
| ow-zora | r57-sil-s60 (argmax) | 99.3% | 0.16 | r61-ow (argmax) | 99.3% | 0.18 |
| stalfos-keese-gel | r57-sil-s60 (sample) | 99.8% | 0.06 | r61-dg (sample) | 99.1% | 0.23 |
| trap-room | r56-rust-ow (argmax) | 99.0% | 0.34 | r61-ow (argmax) | 99.7% | 0.37 |
| wallmaster | r56-rust-dg (argmax) | 99.7% | 0.00 | r61-dg (argmax) | 100.0% | 0.00 |
| zol | r57-sil-s60 (sample) | 99.8% | 0.10 | r61-dg2 (sample) | 98.8% | 0.30 |

Chains, 120 seeds per segment, paired by seed (`TAG=r61-mix2` vs `TAG=r61-b1`; sign test on the discordant
seeds):

| segment | B1 | r61 | gained | lost | p |
|---|---:|---:|---:|---:|---:|
| D2 (`2_7ds_0` → 0:0x3c) | 89 | 100 | 23 | 12 | 0.09 |
| after D2 (`0_3cc_0` → 1:0x73) | 34 | 92 | 67 | 9 | <0.0001 |
| D1 (`1_73s_0` → 0:0x37) | 115 | 80 | 4 | 39 | <0.0001 |
| after D1 (`0_37c_0` → 3:0x7c) | 120 | 120 | 0 | 0 | 1 |
| D3 (`3_7cs_0` → triforce) | 30 | 23 | 15 | 22 | 0.32 |
| full route (D3 triforce) | 0 | 1 | 1 | 0 | 1 |

Full route arrivals (r61-mix2): D2 118, after D2 68, D1 55, after D1 6, D3 6, the D3 triforce 1 (B1: 110, 72, 14,
6, 6, 0). After D2 is the F094 + F157 story of §1 resolved: the cave-mouth mask is the same live and loaded, and the
new ow-lynel never bounces. D1 is the one regression: of the 40 failed D1 runs, stalfos-keese-gel dies in 1:0x43
(12), goriya in 1:0x23 (8) and 1:0x44 (5), Aquamentus in 1:0x35 (5). stalfos-keese-gel early-stopped after 80
iterations (4 minutes): its single-room success is 99.1% but it loses 0.23 hearts per visit (old model: 0.06), and
in a chain those hearts are the ones Link brings to goriya and Aquamentus. D3 still ends mostly in 3:0x59 (darknut,
47 deaths). The comparison mixes the new models with the rebuilt states (B1 played the old ones).
