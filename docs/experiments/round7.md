# Round 7: lingering measurements and post-round-6 behavior experiments

## Summary

Lee resumed experiment mode on Nov 3: keep the omp todo live, reconcile it with `todo.md` between experiments,
run experiments, log every result and commit each finished item. No round-6 promoted model has changed yet.

- Backlog item 1 complete: behavior-neutral lingering measurement, seeded gameplay parity, training/eval/runner/GUI
  proof and tracked baseline results. Outputs: `models/r29-linger-baseline/`; no model promoted.
- Goriya and Aquamentus parity smoke: three seeded episodes each (seed 700 onward), one entry state each,
  produced identical pre/post gameplay records after removing the added measurement field.
- Targeted checks: 53 tests passed (tracker, env, runner, evaluation, rollout aggregation, events).
- Full dev-worktree suite: 383 tests passed in 104.47 seconds.
- CUDA training smoke: Aquamentus continued for 2 iterations / 1,024 actions with no curriculum; 29 completed
  episodes, all successful. Per-update mean longest quiet stretches 5.73 and 6.29 actions; worst 10 in both.
  This is a smoke checkpoint in `/tmp`, not a candidate for promotion.
- GUI verified on measured training, measured per-room eval and historical absent-data logs, at wide/narrow sizes.
  Logs: `logs/261103_029_boss-aquamentus/`, `logs/261103_030_r29-linger-baseline/` and old r28-d2.
- Lee's running player uses most VRAM. He authorizes killing `play-zelda.py` without asking **only if CUDA memory
  exhaustion crashes the experiment**. The CUDA smoke succeeded; the player was not killed.

## 1. Shared lingering metric

`LingerTracker` counts the longest quiet stretch (`max_steps`, `max_frames`) and the final quiet tail
(`tail_steps`, `tail_frames`), plus elapsed actions/frames. Progress: an enemy hit/kill, pickup (including
weapon-collected hearts at full health), health gained (including the fairy pond), a new best objective distance,
or a room exit. Taking damage alone is not progress.

Distance uses a wavefront frozen at each tangible-progress/objective-phase boundary. Moving enemies cannot reset
stationary Link's clock, and round trips cannot repeatedly earn new best distance. Objective phase changes
rebaseline distance but do not themselves reset the quiet clock. A progress action ends a streak: that action's
frames are not included in the preceding quiet interval. Actions and emulator frames are both retained because
option durations vary.

Terminal env info and runner room records carry the same summary. Training reports mean per-episode longest/tail
and per-iteration worst, separately for room-entry and curriculum starts, plus entry-room breakdowns. Eval keeps
raw episode summaries and overall/per-room/per-state means and maxima. Historical logs without tracking remain
unavailable rather than showing fabricated zeros. Actual boss-death and heart-container-pickup timestamps support
kill → heart → exit diagnostics without changing the policy observation schema.

### Tracked baseline protocol

- Current promoted Goriya and Aquamentus: 20 sampled episodes per start state, seed 0 onward, current YAML health
  distribution, 6 evaluation processes. Baseline results are stored separately, not over their old `eval.json`.
- Full milestone1 and milestone1-fairy chains: 40 sampled runs each, seeds 0–39, 12 processes.
- Dungeon 2 segment: 40 runs from `2_7ds_2`, same seeds/process count.
- No rewards, timeouts, start states, schemas or promoted checkpoints changed for these measurements.

### Baseline results

| Specialist / room | Episodes | Success | Health change | Mean longest quiet actions | Worst quiet actions |
|---|---:|---:|---:|---:|---:|
| Goriya overall | 240 | 178/240 (74%) | −0.96 | 86.4 | 1,971 |
| Goriya 1:0x44 | 60 | 73% | −0.91 | 109.0 | 1,971 |
| Goriya 1:0x23 | 60 | 77% | −0.62 | 175.4 | 1,902 |
| Goriya 2:0x5e | 60 | 75% | −1.08 | 20.3 | 127 |
| Goriya 2:0x1e | 60 | 72% | −1.23 | 41.1 | 559 |
| Aquamentus (ends at heart) | 60 | 60/60 (100%) | +0.48 | 6.2 | 12 |

| Chain segment | Runs | Dodongo room reached | Important findings |
|---|---:|---:|---|
| Full milestone1 | 40 | 0/40 | D1 Goriya: 15 deaths, 1 stuck; Aquamentus: 2 deaths, 2 stuck |
| Full milestone1-fairy | 40 | 2/40 | Same D1 results; D2 0x2e also stuck once |
| D2 entrance | 40 | 9/40 | 0x2e: 6 deaths, 4 stuck; 0x3e: 8 deaths; Goriya 0x1e: 8 deaths, 1 stuck |

These are measurement baselines, not an improvement claim. Both full chains had the same 12 Aquamentus visits:
eight exited, two died in the fight, two killed the boss and took the heart but exhausted 2,000 actions. Their
post-heart delays were 1,960 and 1,964 actions. One successful visit took another 491 actions after the heart.
The D2 segment's 0x2e visits averaged a 393.6-action longest gap, worst 2,982; 0x4e worst 2,410. The main chain's
1:0x43 worst was 2,881 (specialist budgets can exceed the runner's 2,000-action minimum).

Long success gaps matter too: successful Goriya 1:0x23 episodes had mean longest 154.4 actions, worst 1,274;
successful 1:0x44 episodes worst 399. A blanket 100-action cutoff would reject some current successes: use these
measurements when testing backlog item 6, and improve the behavior rather than call every long run acceptable.

## 2. Aquamentus post-heart diagnosis

Extended the **evaluation task only** through the correct exit (temporary config: success `[exit-correct]`,
max_steps 2,000), retaining the current promoted checkpoint and YAML hp distribution. The objective has no correct
next room while the boss/heart remain, so leaving early is not a success. 20 episodes per each of 3 fight-entry
states, seeds 0–59, six workers. Output: `aquamentus-through-exit.json` in the baseline folder.

- 59/60 correct exits (98%); mean 126 actions, mean health change +0.61 hearts.
- Among successful visits, post-heart delay mean 60.3, median 17, worst 819 actions.
- Seed 58 (`1_35s_2`, two starting hearts): killed Aquamentus at action 33/frame 353, took the container at
  action 37/frame 379, then made no progress for 1,963 actions / 17,627 frames before max-steps.
- Replayed that exact seed through the real env: reproduced all 2,000 actions and saved screenshots at boss death,
  heart pickup, 100 actions after pickup and the end, only with `$12 == 5` (playable, not mid-scroll).
- Post-heart trace: objective MOVE throughout; 770 `MOVE E 8`, 757 `MOVE W 8`, and 94 `WAIT 8` actions dominate.
  Final positions bounce between (148,101) and (160,101), north of the east door. This confirms Lee's observation,
  not a shadow-Link fight explanation. The training task ends before this behavior is evaluated.

Decision: train the same boss model through the required heart pickup and east exit, not a stronger penalty first.
Observation audit runs before that training so any schema decision is explicit. No start states or model changed
for diagnosis; no post-kill training states saved/promoted.
