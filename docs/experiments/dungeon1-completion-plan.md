# Dungeon 1 Completion — Master Plan (Living Document)

**Created:** 2026-08-03 by the planning agent. **Repo:** `/home/leculver/work/git/triforce`.
**This document is the blueprint.** Executor agents pick ONE task, attempt it fully, record the outcome here, and stop.

---

## 0. READ THIS FIRST (always read this section; everything below is read on demand)

### 0.1 Mission and definition of done

A custom-PPO agent plays NES Zelda from game start. Overworld: solved. Dungeon 1 progress stops at the
Aquamentus boss room. The route that matters (milestone numbers in parens, full-game numbering):

```
1_43 (13) → 1_44 Red Goriya (14) → 1_45 Wallmaster (15) → 1_35 Aquamentus (16) → 1_36 Triforce (17)
                                       └ SOLVED 1.0 ┘        └───────── THE GAP ─────────┘
```

**Definition of done, in order (each gate is a named eval, 100 episodes, via the OMP evaluation plugin):**

- **Gate A (boss):** `dungeon1-aquamentus-east` success-rate ≥ 0.5, sustained (not a single milestone reading).
- **Gate B (chain):** `dungeon1-late-chain` success-rate > 0 and `progress/max` = 17 (both firsts ever).
- **Gate C (dungeon):** `dungeon1-all-items` success-rate > 0.
- **Gate D (game):** `full-game-all-items-finite` success > 0/100, fewer than 65/100 episodes at
  milestone 13, death rate < 0.94. Baseline to beat:
  `training/baseline/runs/all-items-circuit/0/…all-items-polish_10719232.eval.json` (0/100, median 13/17).

### 0.2 The root cause, already verified in source (do not re-derive)

The Aquamentus failure (100/100 episodes `failure-left-boss-room`, zero attacks) is a **reward
specification bug**, verified by direct code reading during planning (evidence in §1):

1. `failure-left-boss-room` is missing from both terminal-penalty sets in `triforce/scenario_wrapper.py:22-27`.
   Walking out costs ≈ −0.25; dying costs −20; timing out costs −20. Leaving is ~80× cheaper than any
   other way of not winning. The policy learned the reward function correctly.
2. In room `1_35`, the objective is `ObjectiveKind.FIGHT` with `next_rooms=[]`
   (`triforce/objectives.py:231-264` + `:170-177`), so no exit is ever "correct", hints mask exits
   inconsistently, and no gradient toward `1_36` exists until the boss is dead AND the heart container collected.
3. Combat economics are thin: melee hit +0.25, Aquamentus has 6 HP, one fireball hit costs −0.5 health
   penalty −0.25 beam loss and halves the step's positive rewards. No per-damage progress signal; boss
   hits don't reset the stalling clock.

**Corroboration from the system that actually beat Dungeon 1** (§1.6): the `origin/original-design`
branch (SB3, one model per area) achieved **68.3% boss kills, zero flee-outs**, using exactly the
inverse choices: leaving the boss room was terminal AND max-penalized, movement shaping paid ±0.25/step
toward the boss, and hit:damage economics were 3:1 (+0.75 hit vs −0.25 damage taken).

Fixing this is Phase R. Verifying it empirically first (cheap, no training) is Phase V. Tooling that
Phase V needs is Phase P. **Do not start a long training run before P1/P2 and the V-phase probes exist.**

### 0.3 How to use this document (executor protocol)

1. Pick the **first OPEN task whose dependencies are met**, top-down in the Task Index (§0.6), unless
   the user directs otherwise. Blocking tasks (`⛔`) come first.
2. Read the task's section (line numbers in the index), plus §1 (background) and §10 (reference card)
   if you haven't. Read `docs/experiments/experiment-memory.md` before any *training* task — it is the
   experiment ledger and its protocol is mandatory for runs (§10.4).
3. Do the task **completely**: code, tests, run, numbers.
4. **Record the outcome in this file:**
   - In the Task Index, change the status box: `[ ]` open → `[x]` done, `[~]` attempted/inconclusive,
     `[!]` falsified/abandoned (say why in the task section).
   - Replace the task's `**Outcome:** _not attempted_` line with a dated outcome block: what you ran,
     artifact paths, key numbers, verdict, and what the next agent should know.
   - If your result changes strategy (e.g. falsifies a phase), append a dated entry to §9 Decision Log.
   - Line numbers drift as the file grows. After editing, refresh the index numbers you invalidated:
     `grep -n '^#\|^### ' docs/experiments/dungeon1-completion-plan.md`
5. **Never commit or push to `main`.** Branch from `origin/main`, PR only. Run
   `pylint triforce/ triforce_debugger/ debug.py evaluate.py train.py record.py` and `pytest tests/ -v`
   before submitting any PR that touches `triforce/`.

### 0.4 Rules of engagement (non-negotiable, distilled from history)

- **Single-milestone readings are not evidence.** Boss-room metrics regressed 0.58→0.96 within one run
  (exp5). Any claim of improvement needs ≥3 consecutive milestone readings or a 100-episode eval.
- **Exit criteria must gate on the thing being taught.** Exp5's boss leg "completed" via the wallmaster
  metric while `failure-left-boss-room` was 0.91. (Fix = P5.)
- **Weight-only curricula on a broken reward spec do nothing.** Exp5 proved 60%→80% Aquamentus weighting
  changes nothing when leaving is optimal. Fix the MDP first (Phase R), then train.
- **Demo-BC + demo-regularized PPO is the proven retention recipe** (wallmaster 0.0 → 1.0, held two
  experiments). Its wiring currently supports exactly one MOVE-only trace; Phase B extends it.
- **Budgets:** micro-scenario eval 3–7 min; BC minutes; a training leg 2+ hours; exp1–5 took a week.
  Prefer probes (minutes) over runs (hours) whenever a probe can kill an idea.

### 0.5 Key assets (details and paths in §10)

- Checkpoints (all on disk, gitignored): exp5 final `training/experiments/experiment5/runs/experiment5-circuit/4/impala-multihead_all-items.pt` (the model with the boss failure); exp4 final (best late-chain, progress/max=16); clean pre-endgame base `…learn-items_2199552.pt`.
- Demo: `docs/experiments/demos/wallmaster-north-exit.txt` (44 MOVE actions; format §10.5).
- The user will hand-play and supply exact button traces on request — that is how the wallmaster demo
  was made. A **scripted** boss-kill probe (V2) may make that unnecessary.
- Savestate catalog: `docs/savestates.yml`. Named RAM overrides available in scenario YAML via
  `per_reset`/`per_frame`, including per-slot enemy HP `obj_health_1`…`obj_health_c` (§10.6).
- Reference implementation that solved D1: branch `origin/original-design` (read via
  `git show origin/original-design:<path>`, never check it out). Digest in §1.6.

### 0.6 TASK INDEX

Status: `[ ]` open · `[x]` done · `[~]` attempted, inconclusive · `[!]` falsified/abandoned · `⛔` blocking.
Line numbers refresh via: `grep -n '^### ' docs/experiments/dungeon1-completion-plan.md`

| # | Task | Phase | Line |
|---|------|-------|-----:|
| [x] | P1 Combat/boss trace mode for diagnose.py | P Tooling | 314 |
| [x] | P2 Fix evaluation hang + silent no-artifact evals | P Tooling | 360 |
| [ ] | P3 Fix restart crash on completed circuits (train.py:1290) | P Tooling | 446 |
| [ ] | P4 --compare must compare success-rate | P Tooling | 456 |
| [ ] ⛔ | P5 Weighted-circuit exit criteria gate on primary scenario | P Tooling | 466 |
| [ ] | P6 Verify worker metric aggregation under --parallel | P Tooling | 476 |
| [ ] | P0 Land or discard the experiment5 branch leftovers | P Tooling | 487 |
| [ ] ⛔ | V1 Empirical root-cause confirmation trace | V Probes | 500 |
| [ ] ⛔ | V2 Scripted boss-kill probe (beams / bombs / melee) | V Probes | 512 |
| [ ] | V3 DefeatedBoss false-positive check | V Probes | 535 |
| [ ] | V4 Trace the hint-mask escape route | V Probes | 547 |
| [ ] | V5 Fireball dodgeability / beam retention measurement | V Probes | 559 |
| [ ] | V6 Aquamentus RAM fact sheet | V Probes | 569 |
| [ ] ⛔ | R1 Terminal parity: leaving the boss room = −20 | R Reward spec | 584 |
| [ ] ⛔ | R2 Boss-damage rewards, stall reset on hits, boss PBRS scale | R Reward spec | 601 |
| [ ] ⛔ | R3 FIGHT rooms get route next_rooms (exit semantics) | R Reward spec | 628 |
| [ ] | R4 Recompute and rebalance the boss-room payoff table | R Reward spec | 650 |
| [ ] | R5 Split micro-scenarios: kill vs victory-lap | R Reward spec | 664 |
| [ ] | R6 EXPERIMENT: retrain Aquamentus on the fixed spec | R Reward spec | 678 |
| [ ] | B1 Demo format: non-MOVE actions | B Demos | 705 |
| [ ] | B2 Multi-demo support in train.py/ml_ppo.py | B Demos | 719 |
| [ ] | B3 Acquire a boss-kill demonstration | B Demos | 732 |
| [ ] | B4 EXPERIMENT: BC + demo-regularized PPO on the boss | B Demos | 744 |
| [ ] | B5 Value-head warmup after BC | B Demos | 761 |
| [ ] | C1 Boss-HP reverse curriculum (1hp→2hp→4hp→6hp) | C Curriculum | 775 |
| [ ] | C2 Mid-fight savestate curriculum | C Curriculum | 793 |
| [ ] | C3 Invulnerability training wheels (per_frame health) | C Curriculum | 802 |
| [ ] | C4 Start-state diversity for the boss room | C Curriculum | 818 |
| [ ] | C5 Victory-lap scenario: post-kill → triforce | C Curriculum | 830 |
| [ ] | C6 Boss-room action masking of useless items | C Curriculum | 844 |
| [ ] | S1 Per-head entropy floor / adaptive ent_coef | S Structural | 860 |
| [ ] | S2 Scope MOVE-only demo-BC loss to the direction head | S Structural | 874 |
| [ ] | S3 EPOCHS 10→4 and expose LEARNING_RATE | S Structural | 886 |
| [ ] | S4 Optimizer persistence across circuit legs | S Structural | 896 |
| [ ] | S5 KL-anchor regularization (fallback to demo-BC) | S Structural | 904 |
| [ ] | S6 Self-imitation on harvested successes | S Structural | 916 |
| [ ] | S7 Per-leg ent_coef in circuit YAML | S Structural | 929 |
| [ ] | S8 Entity distance/vector features (observation) | S Structural | 935 |
| [ ] | I1 EXPERIMENT: late-chain integration | I Integration | 951 |
| [ ] | I2 EXPERIMENT: full dungeon 1 | I Integration | 970 |
| [ ] | I3 EXPERIMENT: full game gate | I Integration | 978 |
| [ ] | X1 Auto-demo harvesting from scripted policies | X Backlog | 989 |
| [ ] | X2 Auxiliary boss-HP prediction loss | X Backlog | 994 |
| [ ] | X3 Sustained-threshold exit criteria | X Backlog | 999 |
| [ ] | X4 Multi-seed replication harness for micro-scenarios | X Backlog | 1004 |
| [ ] | X5 Beam-alignment shaping (fired-correctly / didn't-fire) | X Backlog | 1009 |
| [ ] | X6 Update stale docs (combat-engagement, nes-mechanics boss section) | X Backlog | 1016 |
| [ ] | X7 Positive-clamp audit for multi-reward kill steps | X Backlog | 1022 |
| [ ] | X8 Fireball danger shaping (danger tiles / wavefront field) | X Backlog | 1028 |

**Recommended experiment queue (the spine; everything else supports it):**
1. **EXP6 (probes, no training):** P1+P2 → V1..V6. Deliverable: empirical root-cause verdict + a scripted boss-kill trace.
2. **EXP7 (spec fix):** R1+R2+R3(+R4) → R6 training run. Falsifier: agent stops leaving but still never attacks after full budget.
3. **EXP8 (demos):** B1+B2+B3 → B4 (with C1 as a parallel arm if R6 shows partial engagement).
4. **EXP9 (integration):** I1 → I2 → I3.

---

## 1. Background: the verified diagnosis (read once; do not re-derive)

Everything in §1.1–§1.5 was verified by direct source reading on 2026-08-03 (branch `experiment5`).
File:line references are load-bearing — re-check them if the file changed since. §1.6 was read from
`origin/original-design` via `git show` on the same date.

### 1.1 Terminal accounting

`triforce/scenario_wrapper.py:22-27` defines the two −20 ending sets:

- `HARD_FAILURE_ENDINGS = {failure-terminated-death, failure-stuck, failure-no-progress}` (also strips step rewards)
- `OTHER_FAILURE_ENDINGS = {failure-left-dungeon, failure-left-room, failure-left-route, failure-left-play-area, failure-left-wallmaster-room, failure-reentered-dungeon, failure-nowhere-to-go, failure-no-key, failure-wrong-exit}`

`failure-left-boss-room` (emitted by `DefeatedBoss`, `triforce/end_conditions.py:200-210`) is in **neither**,
as are `failure-no-next-room` and `failure-no-sword`. Terminal clamp widening lives in
`triforce/rewards.py:112-121` (`StepRewards.value`): ±1 normally, +20 on success endings with
`reward-terminal-success`, −20 on failure endings with `penalty-terminal-failure`/`penalty-wall-master`.

Payoff table for `dungeon1-aquamentus-east` (conditions `[ReachedLocation(0x36), DefeatedBoss, GameOver, Timeout]`):

| Outcome | Ending | Terminal value |
|---|---|---:|
| East into 1_36 (needs boss dead — shutter) | success-reached-location | +20 |
| Boss dies | success-killed-boss | +20 |
| **South into 1_45** | **failure-left-boss-room** | **≈ −0.25** |
| Die | failure-terminated-death | −20 |
| Stand still 50 steps | failure-stuck | −20 |
| 2000 steps in room | failure-no-progress | −20 |

The exp5 eval (`…/experiment5/runs/experiment5-circuit/4/impala-multihead_all-items.dungeon1-aquamentus-east.eval.json`):
100/100 `failure-left-boss-room`, ~99 stalling penalties/ep, PBRS net −0.003, exactly one attack-miss
per episode, **zero `reward-hit` / `reward-beam-hit` entries**. The agent enters, oscillates ~250 steps, leaves.

### 1.2 Objectives in the boss room

Room `1_35` (`triforce/game.yaml:1063-1074`): `enemies: {Aquamentus: 1}`, `treasure: heart-container`,
exits E→0x36 (shutter until boss dead), S→0x45. `_get_dungeon_room_objective`
(`triforce/objectives.py:231-264`): known undropped treasure + live enemies ⇒ `ObjectiveKind.FIGHT`,
tile objectives = enemy overlap tiles, `next_rooms` **always** `[]`. `get_current_objectives`
(`objectives.py:170-177`) skips Dijkstra routing entirely for FIGHT/TREASURE/CAVE. Consequences:
- `critique_location_change` (`triforce/critics.py:328-349`): any room change scores `penalty-wrong-location`
  −0.25; `reward-new-location` +1.0 can never fire toward 0x36 until kill+pickup flip the kind to MOVE.
- `TrainingHintWrapper` (`triforce/training_hints.py:20-42`) masks `(MOVE,dir)` at edge tiles when the
  adjacent room is not in `next_rooms` — with `next_rooms=[]` **both** E and S get masked at the edge,
  yet 100% of episodes exit south (mechanism candidates in V4: hint threshold `tile.y >= 0x14` is checked
  pre-action, multi-tile moves under FrameSkip can cross from y≤0x13; `_handle_empty_mask`
  `triforce/action_space.py:400-409` unmasks all four MOVE directions if everything is masked; knockback).
- PBRS targets are the boss's tiles; with γ=1 approach potential is one-shot and empirically cancels.

### 1.3 Combat economics (current constants, `triforce/critics.py`)

`reward-hit` +0.25×decay, `reward-beam-hit` +0.5×decay (+0.05 distance bonus >48px), `reward-bomb-hit`
+0.5 per hit (−0.25 per bomb placed), `penalty-attack-miss` −0.01, `penalty-lost-health` −0.25/half-heart
(fireball = 1 heart = −0.5), `penalty-lost-beams` −0.25, positives halved on damage steps
(`critics.py:146-147`). Stalling: −0.01/step after 150 steps ramping to −0.02 over 1850
(`critics.py:375-383`); resets on room change, enemy **kill**, item pickup — **not on hits**.
Aquamentus: 6 HP, wood sword/beam = 1 HP dmg, bomb = 4 HP dmg (2 bombs kill; scenario grants 4),
3×3 tiles (`triforce/enemy.py:25-26`), fireball 1 heart. Note: `DANGER_TILE_PENALTY` from
`docs/specs/combat-engagement.md` **no longer exists in code** (grep = 0 hits); that spec is partially
stale. Current wallmaster tile shaping is −0.05/−0.04/+0.04 (`critics.py:33-35`).

Scenario `dungeon1-aquamentus-east` (`triforce/triforce.yaml:769-798`): start `1_35s`, per_reset
`hearts_and_containers: 34` (= 3 containers, 2 filled) + `partial_hearts: 254` ⇒ beam gate
(filled==containers−1 AND partial≥0x80, `docs/specs/nes-mechanics.md`) **passes at spawn** — the agent
starts with beams and loses them on the first fireball hit.

### 1.4 PPO / model facts that matter here

- Two-head `ImpalaMultiHeadAgent` (`triforce/models.py:1041+`): action-type head + direction head;
  log-prob/entropy = sum; when only one action type is legal, the type head's log-prob AND entropy are
  excluded per-sample (`models.py:1168-1180`).
- Exp5 ended with `losses/entropy/action_type = 0.0025` — the type head is deterministic (≈always MOVE).
  There is **no entropy floor**; tuning.json health ranges only trigger wakes, and every exp4/5 wake
  loosened them (entropy/action_type min 0.3→0.0 three times).
- Demo-BC: single MOVE-only trace, full batch re-forwarded every minibatch,
  `loss += coeff * -demo_logprob.mean()` (`triforce/ml_ppo.py:621-625`). MOVE-only is enforced at
  `triforce/demo.py:56-58`; `DEMO_ACTION_TYPE_INDEX = {MOVE: 0}` (`demo.py:14`).
- KL: early-stop per minibatch at approx_kl>0.02; whole-round rollback (weights+optimizer) if the final
  minibatch approx_kl > 0.08 (`ml_ppo.py:571-579,654-659`).
- LEARNING_RATE=1e-4 hardcoded in `_setup_optimizer` (`ml_ppo.py:83-97`); EPOCHS=10; only `--ent-coef`
  and demo flags are CLI-reachable.
- Observation: model sees per-entity health/15, stun, direction, presence — **no positions/distances**
  (learned from pixels via CoordConv); info vector has beams-available and fight-objective bits
  (`triforce/observation_wrapper.py:281-366`).
- Parallel rollouts: barrier-synchronized workers with **no timeouts** — a hard-dead worker deadlocks
  training (`triforce/ml_ppo_worker.py`); exp5 run 3 hung exactly this way (16 defunct children).

### 1.5 Experiment history in one table

| Exp | Change | Result |
|---|---|---|
| baseline | all-items circuit, 10.7M steps | full-game 0/100, median 13/17, deaths 0.94 |
| 1 | −20 terminal failures, focused scenarios | accounting fixed; transfer regressed (median 3/17); discard ckpt |
| 2 | +20 terminal success, room-walk acclimation | wallmaster still 0.0 — reward sign alone insufficient |
| 3 | behavior cloning (1000 epochs held; 400 destroyed by PPO) | wallmaster 0.24; PPO exit in 78k steps |
| 4 | demo-BC regularizer in PPO (coeff 0.1) | **wallmaster 1.0** (kept through exp5); late-chain median 10/11, max 16 |
| 5 | boss-weighted circuit 20/60/20→10/80/10 | FAILURE: aquamentus 0.0, left-boss-room 1.0; late-chain eval hung 124h |

Structural-risk positions taken by this plan (from the planning brief's §5):
1. **Combat camping** — the observed boss failure is *zero attacks*, not spam-camping; implement
   boss-scoped R2 now, defer the general combat-engagement spec until diagnosis shows camping.
2. **Entropy collapse** — treat as a constraint: S1 (floor) and S2 (demo loss scoping) if type-head
   entropy < 0.1 persists after Phase R.
3. **PPO erasing BC** — keep demo-reg (proven); add B5 value warmup; S5 KL-anchor only if B4 drifts.
4. **Demo pipeline narrow** — fix now (B1/B2); prerequisite for any boss demo.
5. **Two demos coexisting** — B2 makes the wallmaster regularizer and a boss demo coexist; never drop
   the wallmaster trace without an eval proving retention.
6. **PBRS urgency** — accept the current design; the only changes are stall-reset-on-hit and a
   boss-room PBRS scale option (R2). No new global urgency mechanisms (all six prior attempts failed,
   `2026_03_04_pbrs_movement_tuning.md`).

### 1.6 Evidence from `origin/original-design` — the system that beat Dungeon 1

Read 2026-08-03 via `git show origin/original-design:<path>`. SB3 PPO, one specialist model per area
(`triforce/triforce.json` + `model_selector.py`), including a dedicated `dungeon1boss` model
(attack-only action space, rooms 0x35/0x36, 500K iterations, trained exclusively from savestate `1_35s`
at full health). Shipped eval `models/dungeon1boss.zip.evaluation.json`: **68.3% success-killed-boss,
remainder deaths, ZERO `failure-left-boss-room`.** The design choices that plausibly produced that:

1. **Leaving was terminal AND max-penalized.** Old `DefeatedBoss` terminated with
   `failure-left-boss-room` the moment `location != 0x35`, and `Dungeon1Critic` added
   `penalty-left-early` = −1.0 (their max) for exiting toward a non-objective room. One exit → episode
   over → worst outcome. The current walk-in/walk-out-for-−0.25 loop was impossible. → validates **R1**.
2. **The objective WAS the boss, not the door.** Room 0x35 was in `locations_to_kill_enemies`; the
   navigation objective became `enemies[0].position` and movement shaping paid **±0.25/step**
   (`Dungeon1BossCritic`: move_closer 0.25 = 5× their base) toward Aquamentus, via A*-path-delta shaping
   (not potential-based; exploitable in theory, contained by a 50-step stuck truncation). → validates
   the FIGHT-objective concept; motivates R2's boss-room PBRS scale option.
3. **Hits were worth 3× the damage taken** in the boss room only: `injure_kill_reward` +0.75 vs
   `health_lost_penalty` −0.25 (their base elsewhere: +0.5 vs −0.75 with all positives wiped on damage
   steps). A deliberate per-room "aggression profile" set in `Dungeon1BossCritic.clear()`. → validates
   **R2/R4**, gives the concrete ratio target.
4. **Beam discipline was shaped directly:** `didnt_fire_penalty` −0.05 when an enemy was beam-aligned
   with beams available and the agent moved instead; `fired_correctly_reward` +0.05 for firing while
   aligned (dot > 0.8). Observation included explicit vectors: objective (→ boss during FIGHT), nearest
   enemy, nearest projectile, nearest aligned enemy, plus beams-available bit. → motivates **X5** and
   strengthens **S8**.
5. **Fireball dodging had a dense gradient:** enemy+projectile tiles stamped DANGER (−0.5 to step on)
   with a WARNING halo (−0.05), +0.05 for stepping out, and A* routed around them. → motivates **X8**.
6. **Specialist training regime:** every episode = the fight, from `1_35s`, full health (beams up on
   step 1), 8-action attack-only space, RNG randomized per reset. No navigation task competing for
   capacity. → validates **C-phase** start-state design and motivates **C6** (mask useless items).

Caveats before copying anything wholesale: the old system used raw path-delta shaping and per-step
±0.25 magnitudes that the current PBRS design deliberately rejected as exploitable; it had a separate
model per area (no retention problem, no shared policy to protect); and its 68% came with 32% deaths —
the current health-penalty structure exists for the full-game context. Port the *structure* (terminal
exit, boss-directed gradient, aggression ratio, alignment shaping), not the raw constants.

---

## 2. Phase P — Tooling prerequisites

### P1. Combat/boss trace mode for diagnose.py ⛔
**Goal:** the capability exp5's post-mortem said was missing: per-step traces of boss-room episodes.
**Why blocking:** V1/V4/V5 need it; every later training task uses it for debugging.
**Change (`diagnose.py`):**
- New `--combat-trace` mode (pattern: mirror `PbrsStepRecord`/`run_pbrs_diagnostic`, `diagnose.py:407-465`).
  New `CombatStepRecord` per step: action (kind+direction), Link tile/position/health/beams-available,
  per-enemy (id, index, health, position, distance, is_active, stun), `state_change.hits`,
  enemies_hit indices, full rewards dict, objective kind, `next_rooms`, and the **action mask summary**
  (which MOVE directions are masked — read `info['invalid_actions']` / the flat mask).
- Replace the hardcoded ending filter (`diagnose.py:460-465`, only `stuck`/`no-next-room`) with
  `--trace-endings SUBSTR[,SUBSTR…]` (default keeps old behavior for back-compat).
- Report per traced episode: exit attribution (step, tile, direction, mask state at exit), attack count,
  damage dealt/taken timeline, distance-to-nearest-enemy percentiles.
- Keep it general-purpose (any room, any enemy) per copilot-instructions; commit separately from any
  fix it motivates.
**Acceptance:** `python diagnose.py --model impala-multihead --scenario dungeon1-aquamentus-east
--model-path training/experiments/experiment5/runs/experiment5-circuit/4/impala-multihead_all-items.pt
--combat-trace --trace-endings left-boss-room --episodes 10 -o /tmp/boss-trace.txt` produces per-step
traces for ≥5 episodes with exit attribution.
**Outcome:** ✅ **DONE 2026-08-04.** Implemented `--combat-trace` in `diagnose.py` (PR #192, commit
`07d9be5`). `CombatStepRecord` captures per step: action kind/direction, Link tile/position/health/
beams-available, `hits`/`damage_dealt`/`health_lost`, `enemies_hit` indices, per-enemy snapshot (id,
index, health, position, distance, is_active, is_stunned, stun_timer, is_dying), objective kind +
`next_rooms`, MOVE-mask summary, full rewards breakdown. Per-episode report: exit attribution, attack
summary, enemies-observed w/ HP range, damage timeline, nearest-enemy distance percentiles, `--tail`
step table. `--trace-endings` replaces the hardcoded filter and is shared with `--infinite-pbrs`
(default behavior unchanged); `--trace-max-episodes` bounds output (default 5).

Acceptance command run on the exp5 checkpoint (`--episodes 10 --trace-endings left-boss-room`):
**10/10 episodes traced**, all `failure-left-boss-room`. Every episode: `Attacks: 0/N steps  hits=0
damage_dealt=0  health_lost=0`, `Enemies observed: Aquamentus hp 6..6`, exit via `MOVE/S`. Episode
lengths 6–197 steps. This independently reproduces §1.1 (zero `reward-hit` entries) on a 10-episode
sample — the boss takes **no damage at all**, ever.

**Two things the next agent must know:**
1. **Implementation gotcha:** gymnasium 1.2.3 dropped implicit `Wrapper.__getattr__` delegation, so
   the MOVE-mask readout resolves `is_valid_action` via `env.get_wrapper_attr('is_valid_action')`.
   Any new diagnose mode reaching into `ZeldaActionSpace` must do the same.
2. **Reading the exit step (matters for V4):** on the room-change step, `link_tile` and `masked_moves`
   are from the **post-transition** state (e.g. tile `(15, 2)` in the *new* room 1_45). For V4, read
   the step *before* the room change. Doing so already shows the exit action `MOVE/S` being taken from
   tile `(15, 19)` = y `0x13` — **below** the `tile.y >= 0x14` hint threshold, so S was never masked
   there. Traces also show S correctly masked at y=`0x14` (step 60 of one episode: `MaskS = S,W,E`).
   That is direct support for V4's y-threshold-leak hypothesis; V4 remains open for the full
   determination but the mechanism is already visible.

### P2. Fix evaluation hang + silent no-artifact evals ⛔
**Goal:** we still have NO exp5 late-chain number; the eval ran 124h before SIGKILL, episode rate
degraded to ~11.5h/episode, and two other plugin evals "completed" without artifacts.
**Why blocking:** Gates B/C/D are unmeasurable without it.
**Hypotheses to test, in order:**
1. **Livelock via Timeout reset ping-pong:** `Timeout.is_scenario_ended` (`end_conditions.py:41-57`)
   resets `__last_progress` whenever the agent enters a room in `prev.objectives.next_rooms`. In
   late-chain, leaving 1_35 south costs only −0.25 (no DefeatedBoss condition there) and re-entering
   1_35 from 1_45 *is* route progress → the 1_45↔1_35 loop resets the no-progress clock forever.
   Episode never ends.
2. Emulator/FrameSkip wedge (FrameSkipWrapper waits for a controllable frame that never comes).
**Change:**
- `evaluate.py`: hard per-episode step cap (`--max-steps`, default ~20000), truncating with a distinct
  recorded ending (`eval-step-cap`) so it is visible in endings metrics.
- `Timeout`: only reset the no-progress clock on **first** entry into each next-room per episode, or
  require net milestone progress — pick after confirming hypothesis 1 with P1 traces on a late-chain
  episode.
- `_save_results` (`evaluate.py:367-389`): silent no-op when `progress_values is None` or metrics falsy —
  make both paths loud (warning + nonzero exit), and write the JSON atomically (tmp+rename).
**Acceptance:** exp4-final and exp5-final both produce `dungeon1-late-chain` eval JSONs (100 eps) in
bounded time (< 2h); an induced failure (missing MetricTracker) exits nonzero with a clear message.
**Outcome:** ✅ **DONE 2026-08-04.** PR #192, commit `e2fd96a`. **Hypothesis 1 CONFIRMED** as the
cause; hypothesis 2 (emulator/FrameSkip wedge) was never reached.

Changes:
1. **`Timeout` (`end_conditions.py`)** — now resets the no-progress clock only on the **first** entry
   to each next-room per episode (`__entered_rooms` set, cleared in `clear()`). Genuine forward
   discovery still resets; re-entering an already-visited room does not, so the `1_45↔1_35` loop
   accumulates to `no_progress_timeout` and fires `failure-no-progress`. `failure-stuck` untouched.
2. **`evaluate.py --max-steps`** (default `DEFAULT_MAX_EPISODE_STEPS = 20000`) — cause-agnostic
   per-episode ceiling; on reach the episode truncates with a distinct **`eval-step-cap`** ending
   recorded through the normal `MetricTracker` path, so it shows up as `endings/eval-step-cap`.
3. **`_save_results`** — empty `metrics` or `progress_values is None` now print to stderr and
   `sys.exit(1)` instead of silently no-op'ing; JSON is written atomically (tmp + `os.replace`).
4. **Missed callsite fixed:** `train.py:1322` (`--evaluate` post-training path) also calls
   `evaluate_one_model` and now passes the shared `DEFAULT_MAX_EPISODE_STEPS`. Caught by pylint
   `E1120`, not by grep — search `train.py` too when changing `evaluate.py` signatures.

Tests: `tests/test_end_conditions_timeout.py` (ping-pong re-entry still times out; fresh discovery
still resets) and `tests/test_evaluate_save_results.py` (both loud-exit paths). Emulator-free, 4 tests.
Full suite 783 passed / 1 skipped; pylint 10.00/10.

**Empirical proof the hang is gone:** `dungeon1-late-chain` on exp5-final completed **5 episodes in
2m21s (~28s/episode), exit 0**, artifact written — versus 124h→SIGKILL before. Endings:
`failure-no-progress 0.2`, `failure-terminated-death 0.6`, `failure-wallmastered 0.2`. That
`failure-no-progress` episode is the smoking gun: it is precisely the ending that previously could
never fire. **`eval-step-cap` never triggered**, so the `Timeout` fix alone resolves the livelock and
the step cap is pure backstop.

**Acceptance fully met — the exp5 late-chain number now exists (first ever).** 100 episodes of
`dungeon1-late-chain` on exp5-final completed in **54m44s, exit 0** (bound was < 2h), artifact
`…/experiment5-circuit/4/impala-multihead_all-items.eval.json`. exp4-final's 100-episode late-chain
JSON already existed from exp4
(`…/experiment4-circuit/0/impala-multihead_all-items.dungeon1-late-chain.eval.json`), so both required
readings are in hand.

| | exp4-final | exp5-final |
|---|---:|---:|
| success-rate | 0.0 | 0.0 |
| progress/max | 16 | 16 |
| progress dist (of 11) | 7:23, 8:27, 10:50 | 7:5, 8:45, 10:50 |
| median | 8 | 10 |
| endings/failure-terminated-death | 0.72 | 0.42 |
| endings/failure-no-progress | 0.18 | **0.32** |
| endings/failure-wallmastered | 0.08 | **0.21** |
| endings/failure-stuck | 0.02 | 0.05 |
| endings/eval-step-cap | — | **0.0 (never fired)** |

Three readings that matter for later tasks:
1. **`failure-no-progress` = 0.32 on exp5-final.** Nearly a third of late-chain episodes were
   livelock candidates before the fix — that alone explains the 124h hang, and `eval-step-cap` never
   fired across 100 episodes, so the `Timeout` fix absorbs the whole failure mode.
2. **Gate B is NOT met and was never going to be here:** both checkpoints cap at `progress/max = 16`
   (boss room), never 17 (triforce). Consistent with the boss never taking damage (P1). Gate B is I1's
   job after Phase R.
3. **Micro-scenario success does NOT imply the skill survives in the chain.** `failure-wallmastered`
   rose 0.08 → 0.21 from exp4 to exp5 in late-chain, which looks like the retention regression §0.4
   rule 5 warns about — but it is not. exp5-final's standalone
   `dungeon1-wallmaster-north-exit` eval is **1.0 success (100/100,
   `endings/success-reached-location = 1.0`)**, so the wallmaster skill is fully retained in its own
   scenario. The skill is intact; it just transfers worse into the chained context (different entry
   states, arriving after other rooms, lower health). Deaths fell 0.72 → 0.42 and the median improved
   8 → 10, so exp5 is better overall on late-chain — the failure mode moved, it did not worsen.
   **Consequence for I1/B4: never treat a 1.0 micro-scenario eval as evidence of chain performance.**
   Gate B must be read on `dungeon1-late-chain` itself, exactly as §0.4 rule 2 says for exit criteria.

### P3. Fix restart crash on completed circuits (train.py:1290)
**Goal:** `--resume` from a checkpoint whose history already covers the final leg leaves `model=None`
through `_run_sequential_circuit` (skip logic `train.py:1009-1014` `continue`s past the matched leg) and
crashes at `model.save` (`train.py:1290`). Exp5 run 2 lost a run to this.
**Change:** detect "nothing left to run" after skip resolution and exit early with a clear message
(before touching the emulator); guard `model is None` at `train.py:1269-1300`.
**Acceptance:** functional test: resuming a finished circuit prints "circuit already complete" and
exits 0 without traceback; resuming a mid-circuit checkpoint still works.
**Outcome:** _not attempted_

### P4. --compare must compare success-rate
**Goal:** `compare_models` (`evaluate.py:127-207`) compares only `progress_values`; for ReachedLocation
micro-scenarios the verdict is blind to the metric that matters (the exp3 lesson — fixed in the report
headers, never in compare).
**Change:** when both eval JSONs carry `metrics['success-rate']`, additionally report success counts and
a two-proportion test (Fisher exact via scipy) with its own verdict line; keep Mann-Whitney on progress.
**Acceptance:** comparing exp4-vs-exp5 wallmaster JSONs (both 1.0) reports success-rate parity; a
synthetic 0.2-vs-0.6 pair flags significance.
**Outcome:** _not attempted_

### P5. Weighted-circuit exit criteria gate on primary scenario ⛔ (before any weighted run)
**Goal:** exp5's boss-transfer leg exited on wallmaster 0.86 while the boss metric was 0.0 (§0.4).
**Change:** in weighted circuits (`train.py:_run_weighted_circuit` + `WeightedScenarioSelector`), an
entry-level `primary: true` flag (`TrainingCircuitEntry`, `scenario_wrapper.py:138-146`): the leg exits
only when the primary scenario's criterion is met (or all criteria, if none marked). Config validation
test in `tests/`.
**Acceptance:** a config test proves a weighted circuit with a primary cannot exit on a non-primary
metric; existing sequential circuits unaffected.
**Outcome:** _not attempted_

### P6. Verify worker metric aggregation under --parallel
**Goal:** `docs/recommendations.md` item 2 claims worker MetricTracker data is not aggregated — exit
criteria and TensorBoard would then see only the main process's metrics under `--parallel 16` (how all
of exp5 ran). If true, every exit criterion evaluated during exp5 was computed on a sliver of episodes —
this changes how much we trust historical leg exits.
**Change:** read the `ml_ppo_worker.py` metric-pipe path + `MetricTracker` usage; run a small
instrumented job (2 workers, 20k steps) and compare per-worker episode counts to what the exit-criteria
metric saw. Fix or document precisely.
**Acceptance:** a paragraph in the Outcome + (if broken) an issue-sized fix PR.
**Outcome:** _not attempted_

### P0. Land or discard the experiment5 branch leftovers
**Goal:** branch `experiment5` holds uncommitted `triforce/triforce.yaml` (exp5 circuits),
`docs/experiments/experiment-memory.md`, `experiment5-summary.md`, untracked `tests/test_experiment5_config.py`.
The experiment ledger must be preserved; the exp5 circuit config is referenced by run history.
**Change:** open a housekeeping PR from a fresh branch off `origin/main` landing the docs + yaml + test
(experiment-memory records are append-only history). This plan document rides the same PR or its own.
**Acceptance:** PR open; `main` untouched; nothing force-pushed.
**Outcome:** _not attempted_

---

## 3. Phase V — Verification probes (cheap, no training)

### V1. Empirical root-cause confirmation trace ⛔ (depends P1)
**Hypothesis:** the exp5 policy leaves because leaving is optimal, and the episode return for leaving
(≈ −0.5 total) beats every alternative it knows.
**Steps:** run P1's combat trace on the exp5 checkpoint, `dungeon1-aquamentus-east`, 20 episodes,
`--trace-endings left-boss-room`. Extract: (a) attack count (expect ≈0), (b) exit direction/step,
(c) realized episode returns, (d) action-type distribution, (e) mask state at the exit step.
**Acceptance:** a table in the Outcome block: mean return of leaving vs the payoff table's prediction;
verdict sentence "reward-spec bug CONFIRMED/REFUTED as the dominant cause".
**Falsification:** if traces show the agent *attacking and failing* (hits > 0, dies or times out), the
spec-bug theory loses primacy → re-weight toward Phase B/C (skill, not incentive) and log in §9.
**Outcome:** _not attempted_

### V2. Scripted boss-kill probe ⛔ (independent; start anytime)
**Hypothesis:** Aquamentus is killable from `1_35s` with (a) beams from range at full health, (b) 2
bomb hits, (c) melee — establishing which strategy a policy must learn, and producing a
machine-generated demo trace for Phase B. The original-design system's 68% relied heavily on beams
from a full-health start (§1.6.4), so (a) is the prior favorite.
**Steps:** write `scripts/boss_probe.py` (or a test using `ZeldaActionReplay` from `tests/utilities.py`,
savestate `1_35s`): scripted action sequences —
1. Beam plan: walk N clear of the door, face E, spam BEAMS; count hits until boss HP (obj_health slot,
   V6) reaches 0. Verify `success-killed-boss` fires with +20, heart container drops, objective flips
   FIGHT→TREASURE→MOVE, shutter opens, east walk reaches 0x36 with `reward-new-location`.
2. Bomb plan: approach to bomb range, place 2 bombs timed at the boss.
3. Melee plan: walk into sword range, swing 6×, log health lost.
Record every step as `ACTION DIRECTION` lines → save working sequences to
`docs/experiments/demos/aquamentus-<strategy>.txt` (normalized forward trace format, §10.5).
**Acceptance:** at least one strategy kills the boss reproducibly; full reward/ending accounting of a
kill documented in the Outcome; a candidate demo file exists.
**Falsification:** if no scripted strategy lands hits (e.g. sword blocked by screen-lock bounds near
the south door — UW blocks sword/items when y>199, `docs/specs/nes-mechanics.md`), document the
geometry constraint; ask the user for a hand-played trace instead (B3 route b).
**Note:** also log whether any transient frame during the fight has `state.enemies == []` (feeds V3),
and snapshot `em.get_state()` at boss HP 4/2/1 and post-kill (feeds C2/C5).
**Outcome:** _not attempted_

### V3. DefeatedBoss false-positive check (piggybacks on V2)
**Hypothesis (from the brief, unverified):** `DefeatedBoss` tests `not state.enemies`
(`end_conditions.py:204`) — a transient empty object list (spawn frames, sparkle→item conversion) would
yield a spurious +20 `success-killed-boss` — a reward-hacking channel. Mild counter-evidence: the
original-design branch used the same check and shipped no spurious successes (§1.6).
**Steps:** during V2 runs, log `len(state.enemies)` and `len(state.active_enemies)` every frame from
room entry to kill; test the room-entry frames specifically (object slots populate a few frames in).
**Change if real:** require boss absent for N consecutive controllable frames, or check the dying
transition explicitly (`Enemy.is_dying`, `enemy.py:30-38`) before declaring the kill.
**Acceptance:** verdict + (if real) fix PR with a regression test.
**Outcome:** _not attempted_

### V4. Trace the hint-mask escape route (depends P1)
**Hypothesis:** hints mask (MOVE,S) at `tile.y >= 0x14` (`training_hints.py:36-37`) but transitions can
be entered from y≤0x13 with a multi-tile move under FrameSkip; `_handle_empty_mask`
(`action_space.py:400-409`) is a second candidate.
**Steps:** in V1's traces, extract Link's tile and the mask at the step where location changes
0x35→0x45. Determine which mechanism allows the exit.
**Why it matters:** R3 makes hints coherent in FIGHT rooms; if the exit leaks through the y-threshold,
hints will *still* leak after R3 — fix the threshold (mask S at y ≥ transition-trigger − max move tiles)
or accept that R1's −20 handles it economically.
**Acceptance:** mechanism named with a trace excerpt in the Outcome.
**Outcome:** _not attempted_

### V5. Fireball dodgeability / beam retention measurement (piggybacks V2)
**Question:** can a policy plausibly keep full health (and thus beams) for a whole fight? How many
action-steps does a fireball dodge take, and does frame-skip granularity permit it?
**Steps:** in V2 beam runs, log fireball spawn→impact windows in *agent steps* (not frames); try a
scripted dodge (step N/S between shots). Compute: fireball frequency, dodge window, hit rate of
"stand and shoot" vs "shoot and weave".
**Deliverable:** a paragraph deciding whether beam-kill is learnable without damage, or whether the
curriculum must assume melee-after-first-hit. Feeds C3's design, R4's payoff math, X8's priority.
**Outcome:** _not attempted_

### V6. Aquamentus RAM fact sheet (independent)
**Goal:** the facts C1 and V2 need, currently undocumented (`docs/specs/nes-mechanics.md` has no boss
section — X6 later folds this in).
**Steps:** load `1_35s` in a scratch script; dump object slots: which `obj_health_N` is Aquamentus
(likely slot 1 → `obj_health_1` @ 0x486, value 0x60 = 6 HP in high nibble — verify), boss position
(static?), fireball slots/ids, shutter door tile state before/after kill, heart-container drop position.
Cross-check against `zelda-asm/` if anything surprises.
**Acceptance:** fact sheet in the Outcome (slot; HP encoding verified by writing 0x10 and confirming a
1-hit kill; door mechanics; drop tile).
**Outcome:** _not attempted_

---

## 4. Phase R — Reward & termination specification

### R1. Terminal parity: leaving the boss room = −20 ⛔
**Hypothesis:** with `failure-left-boss-room` ≈ −0.25, leaving dominates; pricing it at −20 removes the
degenerate optimum. Original-design evidence (§1.6.1): terminal + max penalty on exit produced zero
flee-outs. Necessary but NOT sufficient — pair with R2/R3 before training (a policy with no positive
path may just pick the *fastest* −20).
**Change:** add `failure-left-boss-room` to `OTHER_FAILURE_ENDINGS` (`scenario_wrapper.py:23-27`).
Audit the other orphans at the same time: `failure-no-next-room`, `failure-no-sword` — add or document
why not. Unit test in `tests/` on `apply_terminal_rewards` covering all three.
**Also:** `dungeon1-late-chain` has no `DefeatedBoss` condition, so leaving 1_35 south there remains a
−0.25 ordinary step and the episode continues (P2's livelock). After R3, `penalty-wrong-location` still
prices it; decide in R6 whether late-chain also needs a boss-room-exit condition — default: leave
late-chain permissive (the room must be re-enterable en route to the triforce) and rely on P2's timeout
fix.
**Acceptance:** tests pass; a 10-episode diagnose run on the *old* exp5 checkpoint now shows leaving
episodes scoring ≈ −20 (accounting check only; behavior unchanged).
**Outcome:** _not attempted_

### R2. Boss-damage rewards, stall reset on hits, boss PBRS scale ⛔
**Hypothesis:** 6 melee hits × +0.25 does not pay for the expected fireball damage en route; a per-HP
damage signal that dominates a hit-trade makes engagement positive-EV; hits must count as "progress"
for the stalling clock; and the approach gradient toward the boss should be stronger than the generic
±0.05/tile. Original-design targets (§1.6.2–3): hit:damage ≈ 3:1, approach shaping ±0.25/step.
**Change (`triforce/critics.py`):**
1. New `reward-boss-hit`: for boss enemies (set keyed off `ZeldaEnemyKind`, Aquamentus now), replace
   `reward-hit`/`reward-beam-hit` with a per-HP-damage reward — recommend +0.5 per HP dealt (bombs deal
   4 HP → +2.0 pre-clamp; the ±1 step clamp caps realized value — X7 audits this). Total kill payout
   ≈ +3.0 across the fight + 20 terminal. Farming-safe: boss HP is finite, no respawn.
2. Reset `_room_steps` (stalling clock, `critics.py:372-378`) on `state_change.hits` — damage is
   progress in a FIGHT room. Currently only kills reset it.
3. Optional third lever, keep behind a constant: in FIGHT rooms, use a smaller `PBRS_SCALE` (e.g. 20→8)
   so approaching the boss pays ±0.125/tile instead of ±0.05. Still potential-based (γ=1, telescoping,
   exploit-proof) — this is the safe analog of the old ±0.25/step directional shaping.
4. Keep `penalty-lost-health` and `penalty-lost-beams` unchanged for now; R4 decides whether a
   boss-room "aggression profile" (softened damage penalty à la `Dungeon1BossCritic`) is needed. Note
   the positives-halved-on-damage rule (`critics.py:146-147`) already softens relative economics — the
   old system's was harsher (full wipe), yet it still needed 3:1 to engage.
**Interaction check:** reward-attribution look-ahead (`docs/specs/reward-attribution.md`) already
credits delayed beam/bomb damage to the firing step — new constants ride the same path; add a unit test
with a saved fight state asserting the reward dict (pattern: `tests/` + `CriticWrapper`).
**Acceptance:** unit tests; V2's scripted kill replayed under the new critic shows per-hit rewards and
no stalling-penalty accumulation while actively dealing damage.
**Falsification:** none at this stage (spec work); R6 falsifies behaviorally.
**Outcome:** _not attempted_

### R3. FIGHT rooms get route next_rooms (exit semantics) ⛔
**Hypothesis:** `next_rooms=[]` in FIGHT rooms makes hints self-defeating and forbids any "correct exit"
gradient; supplying the route's next rooms even during FIGHT makes location scoring, hints, and
post-kill routing coherent.
**Change (`triforce/objectives.py`):** in `get_current_objectives` (`:154-177`), when kind is
FIGHT/TREASURE, still call `_get_map_objective` for `next_rooms` (routing metadata) but do NOT extend
`tile_objectives`/`pbrs_targets` with exit tiles — PBRS keeps pointing at enemies/treasure; hints and
`critique_location_change` get truthful next rooms. CAVE keeps current behavior.
**Consequences to verify:**
- Boss room during fight: `next_rooms={0x36}` ⇒ hint masks S (correct); E stays walkable but the
  shutter physically blocks until kill ⇒ leaving becomes *mechanically* hard, not just penalized
  (defense in depth with R1).
- After kill+pickup: kind→MOVE, `reward-new-location` +1.0 fires entering 0x36 (previously impossible).
- Non-boss FIGHT rooms (key rooms): the agent fighting near a door no longer takes
  `penalty-wrong-location` for a legitimate route exit — check this doesn't create a "skip the fight"
  loophole: the room's key stays uncollected and routing may require it later (`Dungeon1DidntGetKey`
  covers 0x63). Audit key rooms in `game.yaml`; note any where skipping breaks the route.
**Acceptance:** unit tests on objectives for 1_35 (FIGHT w/ next_rooms), a key room, and a cave;
existing tests green; a diagnose run confirms hint masks in 1_35 now mask S and only S at the south
edge.
**Outcome:** _not attempted_

### R4. Recompute and rebalance the boss-room payoff table (after R1–R3, before R6)
**Goal:** write the *new* payoff table and sanity-check it end-to-end before spending training budget.
**Steps:** with R1–R3 applied, compute for each outcome (kill via beams / kill via melee with k
fireball hits / leave south / die / timeout) the total episode return using real constants; replay V2's
scripted sequences under the new critic to validate empirically. Table goes in the Outcome block.
**Decision points this settles:**
- Is a melee kill with 2 fireball hits net-positive? (Target: yes, comfortably, ≥ +2.)
- Is "take one hit then leave" still ≤ −19?
- Does the ±1 per-step clamp swallow the bomb double-hit (X7)?
- Is a boss-room aggression profile (damage penalty −0.5 → −0.25 while a boss is alive, per §1.6.3)
  needed to make melee viable, or do beams+bombs carry it?
**Acceptance:** table + one-line go/no-go for R6.
**Outcome:** _not attempted_

### R5. Split micro-scenarios: kill vs victory-lap
**Goal:** `dungeon1-aquamentus-east` conflates two skills: kill the boss AND route to 0x36. Decompose
for cleaner gates and reverse-chaining. (The original-design boss model trained on exactly the kill
task with `[GainedTriforce, DefeatedBoss, GameOver, Timeout]`, §1.6.)
**Change (`triforce/triforce.yaml`):**
- `dungeon1-aquamentus-kill`: same start, end conditions `[DefeatedBoss, GameOver, Timeout]` — success
  = `success-killed-boss` only. Metric: success-rate.
- `dungeon1-victory-lap` (C5 builds the savestate): start post-kill, success = ReachedLocation 0x36 or
  GainedTriforce.
Keep `dungeon1-aquamentus-east` unchanged as the Gate-A eval.
**Acceptance:** config test (mirror `tests/test_experiment5_config.py`); both scenarios load and run 1
episode headless.
**Outcome:** _not attempted_

### R6. EXPERIMENT: retrain Aquamentus on the fixed spec (the EXP7 run)
**Hypothesis:** with R1+R2+R3 in place, PPO from the exp5 (or exp4) checkpoint learns to engage — even
without demonstrations — because engagement is now the only non-catastrophic option and damage pays.
The original-design system needed no demos for its 68% (§1.6); its advantages (specialist model,
attack-only actions, vector obs) are partially compensated by our curriculum options (C-phase).
**Protocol (full experiment-memory discipline, §10.4):**
- Load: exp5 final weights (keeps wallmaster demo-reg with the existing trace, coeff 0.05) — fallback
  arm: exp4 final if exp5's entropy collapse resists retraining.
- Scenario: `dungeon1-aquamentus-kill` (R5) if available, else `dungeon1-aquamentus-east`; 750k steps;
  `--parallel 16` only after P6's verdict, else 6.
- Exit criteria: success-rate ≥ 0.5 **sustained over 3 consecutive milestone wakes** (X3 automates;
  until then enforce by journal discipline).
- Watch: `losses/entropy/action_type` (floor 0.1 — if pinned ~0, apply S1/S2 mid-experiment via
  stop-edit-restart), `failure-left-boss-room` rate, `reward-boss-hit` count/episode, stuck/death rates.
- Final eval: `dungeon1-aquamentus-east`, 100 eps, P4's compare vs exp5's eval JSON.
**Success:** Gate A threshold (≥0.5) or any nonzero sustained success — both are firsts.
**Falsification:** left-boss-room collapses to ~0 (agent stays) but attack count stays ≈0 for the full
budget → incentive fixed but the *skill* is undiscoverable by exploration → pivot to Phase B (demos)
and/or C1 (reverse curriculum); record in §9.
**Failure mode to watch:** agent converts to `failure-stuck`/death farming (both −20, faster). If >30%
of episodes end stuck-in-place, exploration found no positive path: same pivot.
**Outcome:** _not attempted_

---

## 5. Phase B — Boss demonstration & behavior cloning

### B1. Demo format: non-MOVE actions
**Goal:** `parse_demo_trace` raises on non-MOVE (`demo.py:56-58`); `DEMO_ACTION_TYPE_INDEX={MOVE:0}`
(`demo.py:14`). A boss demo needs SWORD/BEAMS/BOMBS lines.
**Change (`triforce/demo.py`):**
- Accept `<ACTIONKIND> <DIRECTION>` lines for any ActionKind in the all-items space; derive the
  action-type index from the `ActionSpaceDefinition` ordering (do NOT hardcode a second mapping —
  `collect_demo_batch` already builds the env; take indices from `ZeldaActionSpace`).
- Keep MOVE-only validation as a per-trace *option* (wallmaster trace unchanged).
- Generalize `diagnose.py --demo-report`'s hardcoded success check (`diagnose.py:1085` expects the
  wallmaster target room) to accept the scenario's own success ending.
**Acceptance:** unit tests: mixed trace parses; replay of a V2-generated boss trace through
`collect_demo_batch` succeeds (mask-legal at every step, ends `success-*`).
**Outcome:** _not attempted_

### B2. Multi-demo support in train.py/ml_ppo.py
**Goal:** wallmaster retention currently rides ONE trace (`--demo-trace` single string, one
`_demo_batch` tuple). A boss demo must coexist (structural risk #5).
**Change:** `--demo-trace` → `action='append'`, paired per-trace with scenario/prefix (or a
`--demo-spec trace:scenario:prefix` syntax); build each batch via `collect_demo_batch`, then
`torch.cat` obs-dict values/masks/targets into one batch (loss is a mean over examples — weighting by
trace length is acceptable; document it). Optional per-trace coeff later, only if needed.
**Cost note:** the whole demo batch is re-forwarded every minibatch (10 epochs × 16 minibatches); with
~100 total demo steps this stays negligible.
**Acceptance:** training smoke run (10k steps) with wallmaster+boss traces logs `demo_bc_accuracy`;
unit test for the concat helper.
**Outcome:** _not attempted_

### B3. Acquire a boss-kill demonstration
**Route (a) — scripted (preferred, no human):** V2's successful strategy saved as a normalized forward
trace. Requirements: every action mask-legal during replay, ends `success-killed-boss` or
`success-reached-location`. The bomb strategy is attractive (2 damage actions); beams have more steps
but teach ranged discipline; capture both if both work.
**Route (b) — human:** ask the user to hand-play `1_35s` → kill → east → 0x36 and supply the button
sequence (they have offered; that's how wallmaster was made). Convert to trace format (§10.5).
**Validation:** `diagnose.py --demo-report --demo-trace docs/experiments/demos/aquamentus-<x>.txt
--demo-scenario dungeon1-aquamentus-east` (after B1's generalization).
**Acceptance:** ≥1 validated trace in `docs/experiments/demos/`, referenced here.
**Outcome:** _not attempted_

### B4. EXPERIMENT: BC + demo-regularized PPO on the boss (the EXP8 run)
**Hypothesis:** the exp3/4 recipe transfers: 1000-epoch BC on the boss trace makes the boss room
learnable; the demo-BC term retains it; PPO polishes to Gate A.
**Protocol:**
- BC: `scripts/behavior_clone.py` from exp4-final (cleaner entropy than exp5) with the boss trace,
  1000 epochs (the 400-epoch variant was destroyed by PPO in exp3 — do not repeat), min-accuracy 0.95.
  behavior_clone is policy-only; do B5 first if available.
- PPO: fixed-spec scenario (R1–R3 required), BOTH demos via B2 (wallmaster coeff 0.05 + boss coeff 0.1
  starting points), 750k steps, exit success-rate ≥ 0.5 sustained ×3 wakes.
- Eval: `dungeon1-aquamentus-east` 100 eps vs exp5 baseline JSON; ALSO re-eval
  `dungeon1-wallmaster-north-exit` 100 eps — retention must stay 1.0 (falsifier for the multi-demo
  design).
**Falsification:** BC reaches accuracy ≥0.95 but PPO success collapses to 0 within 200k steps despite
the demo term (the exp3-run0 signature) → value-head staleness is the suspect → do B5, retry once; if
it collapses again → S5 (KL anchor).
**Outcome:** _not attempted_

### B5. Value-head warmup after BC
**Goal:** `behavior_clone.py` trains policy only; the critic is stale, so early PPO advantages are
garbage and can destroy the cloned policy (exp3 run 0: 0.447→0.0).
**Change (`ml_ppo.py` + flag):** `--value-warmup-steps N`: for the first N env steps, collect rollouts
and update ONLY the value head (freeze policy params or zero pg/entropy losses), then unfreeze. Log the
phase boundary to TensorBoard.
**Acceptance:** smoke test showing policy params bit-identical through warmup; explained_variance
recovers above 0.3 before policy updates begin on a BC checkpoint.
**Outcome:** _not attempted_

---

## 6. Phase C — Curriculum & environment tricks

### C1. Boss-HP reverse curriculum (1hp→2hp→4hp→6hp)
**Hypothesis:** with boss HP=1, random exploration finds the kill (+20 dense) quickly; scaffolding HP
back to 6 transfers engagement without demos. The strongest *demo-free* arm.
**Mechanism (verified available):** scenario `per_reset` can write any named RAM value
(`state_change_wrapper.py:335-347,405-417`); `obj_health_1`…`obj_health_c` are named
(`zelda_game_data.txt:100-111`). V6 confirms Aquamentus's slot and encoding (HP in high nibble:
0x10 = 1 HP).
**Change (`triforce/triforce.yaml`):** scenarios `dungeon1-aquamentus-kill-1hp/-2hp/-4hp` = copies of
`dungeon1-aquamentus-kill` (R5) + `per_reset: {obj_health_1: 0x10 / 0x20 / 0x40}`; sequential circuit
`aquamentus-reverse-hp` with exit success-rate ≥ 0.8 per leg (sequential legs gate on their own
scenario, so P5 isn't required), budgets 250k/250k/500k/750k.
**Caveat:** per_reset fires on the reset frame; verify the write lands after object spawn (V6 checks).
If spawn timing clobbers it, use `per_room`/`per_frame` with a value-guard instead.
**Success:** full-HP leg reaches sustained success ≥ 0.5 → feed into Gate A eval.
**Falsification:** the 1hp leg itself stays at 0 success for 250k steps → exploration never even swings
→ entropy/masking problem, not curriculum problem → S1/S2 before anything else.
**Outcome:** _not attempted_

### C2. Mid-fight savestate curriculum
**Goal:** the reverse-curriculum idea without RAM writes: start episodes *inside* successful fights.
**Steps:** during V2 scripted kills, `em.get_state()` snapshots at boss HP 4/2/1 with Link positioned
safely; save as custom integration states (`triforce/custom_integrations/Zelda-NES/`); add to scenario
`start:` lists (round-robin supports lists). Catalog in `docs/savestates.yml`.
**Use:** blend into R6/B4/C1 start lists so every rollout batch contains near-success states.
**Acceptance:** states load and roll; a training smoke run consumes the mixed start list.
**Outcome:** _not attempted_

### C3. Invulnerability training wheels (per_frame health)
**Hypothesis:** fear of damage (health penalties + beam loss + reward halving) suppresses engagement
exploration; a phase where Link cannot effectively lose health lets attack behavior emerge, then the
wheels come off.
**Change:** scenario variant `dungeon1-aquamentus-kill-invuln` with
`per_frame: {hearts_and_containers: 34, partial_hearts: 254}` — health restored every frame (beams
never lost either — subsumes a "beams-forever" variant). Circuit: invuln → normal, exit ≥0.8 then ≥0.5.
**Risks (state in the journal):** (1) policy learns to tank hits — the transfer leg must un-learn
standing in fireballs; (2) `critique_health_change` may see per-frame refills as `health_gained` →
check the critic doesn't emit spurious `reward-gained-health` every frame (verify where per_frame
writes land relative to health-delta computation in `StateChangeWrapper._apply_modifications`; exempt
if needed).
**Falsification:** transfer leg loses >80% of the invuln leg's success and doesn't recover within 2×
its budget → abandon (tanking overfit confirmed).
**Outcome:** _not attempted_

### C4. Start-state diversity for the boss room
**Goal:** a single deterministic opening (`1_35s`) invites brittle memorized openings; NES RNG is
partly frame-driven so diversity must come from states. (The old system randomized RNG bytes at reset —
`zelda_wrapper.py` on original-design; check whether the current reset does; if not, that's a one-line
addition worth making here.)
**Steps:** generate 5–10 entry snapshots: vary Link's entry column/row (scripted walk-ins from 1_45
with different prefixes, snapshot at the door), vary boss/fireball phase by waiting 0–60 frames before
snapshotting. Add to `start:` lists of the boss scenarios.
**Acceptance:** states cataloged; R6/B4 journals note whether variance across starts shrinks over
training.
**Outcome:** _not attempted_

### C5. Victory-lap scenario: post-kill → triforce
**Goal:** milestones 16→17 (heart container, east exit, triforce room, GainedTriforce) have literally
never been executed by any trained policy. Make them independently learnable/verifiable
(reverse-chaining the tail; pairs with R5).
**Steps:** V2 produces a post-kill savestate (boss dead, container on floor, shutter open) →
`dungeon1-victory-lap` scenario: end conditions `[GainedTriforce, CollectedTreasure(triforce),
ReachedLocation(0x36), GameOver, Timeout]`, success per R5's design. Short PPO run (≤250k) from
exp4/exp5 checkpoint; this should be an *easy* MOVE/TREASURE task.
**Acceptance:** sustained success ≥ 0.8 in ≤250k steps; verifies triforce accounting end-to-end
(GainedTriforce, equipment reward, no `failure-*` mispricing on the pickup cutscene).
**Falsification:** if even this stalls, something is broken in triforce-room accounting — diagnose
before ANY boss work continues; it would poison late-chain too.
**Outcome:** _not attempted_

### C6. Boss-room action masking of useless items
**Goal:** the old system's boss model had 8 actions (move + sword only, §1.6.6); ours explores over the
full all-items space (whistle, food, candle, potion…), diluting exploration exactly where the type head
is already collapsed. Masking useless actions in the boss room narrows the search without a new model.
**Change:** extend `TrainingHintWrapper._disable_actions` to append `invalid_actions` for item kinds
that are provably useless in the current room (boss rooms: allow MOVE/SWORD/BEAMS/BOMBS only). Gate it
on `use_hints` so eval-with-hints matches training. Keep the list data-driven (room → allowed kinds) in
the wrapper, not hardcoded deep in action_space.
**Acceptance:** unit test: in 1_35 with hints, mask allows only the four kinds; R6/B4 journals note
whether exploration finds attacks faster with C6 on (A/B if cheap).
**Outcome:** _not attempted_

---

## 7. Phase S — Structural PPO/model work

### S1. Per-head entropy floor / adaptive ent_coef
**Problem:** action-type entropy 0.0025 at exp5 end; guardrails only *observe*. A deterministic type
head cannot explore "attack".
**Change (`ml_ppo.py::_optimize`):** compute per-head entropies in the update (expose from
`get_action_and_value` or recompute per-head); apply an adaptive coefficient:
`ent_coef_type = base * (1 + k * max(0, floor − H_type))` with floor ≈ 0.2, k ≈ 10 — pressure grows as
the head collapses (SAC-style target-entropy flavor without the dual variable). CLI `--ent-floor-type`.
Log both head entropies + effective coefficients.
**Acceptance:** on a 100k-step boss-scenario run from the exp5 checkpoint, `entropy/action_type`
recovers to >0.1 without wallmaster success dropping below 0.9 (40-ep eval).
**Falsification:** wallmaster retention breaks (<0.8) at any floor that also restores exploration →
retire S1 in favor of S2.
**Outcome:** _not attempted_

### S2. Scope MOVE-only demo-BC loss to the direction head
**Problem:** the wallmaster demo is MOVE-only; its NLL term pushes the *type* head toward MOVE on demo
states, and generalization plausibly suppresses ATTACK everywhere (the exp5 journal repeatedly
justified type-entropy ≈ 0 by "MOVE-only demo regularizer" — that's the tell).
**Change:** in the demo loss (`ml_ppo.py:621-625`), decompose per-head log-probs (the multihead model
computes both; expose them) and, per demo trace, include the type-head term only if the trace contains
non-MOVE actions. Boss demos (B1) include attacks → their type-head term stays.
**Acceptance:** A/B 100k-step runs (with/without scoping) from the exp5 checkpoint: scoped variant
shows higher type entropy with equal demo accuracy (accuracy is both-heads — for MOVE-only traces also
report direction-head-only accuracy).
**Outcome:** _not attempted_

### S3. EPOCHS 10→4 and expose LEARNING_RATE
**Problem:** EPOCHS=10 is 2.5× the PPO norm (recommendations item 1, HIGH); LR is hardcoded
(`ml_ppo.py:83-97`). High epoch counts amplify the demo-BC term and per-rollout overfitting — a
plausible contributor to BC destruction and entropy collapse.
**Change:** make both `PPO.__init__` kwargs + CLI flags (`--epochs`, `--lr`); defaults unchanged until
A/B'd. A/B on the R6 setup: epochs 10 vs 4 at equal env steps.
**Acceptance:** A/B table (success trajectory, approx_kl, rollback count, wall-clock). Adopt the
dominant setting; record in §9.
**Outcome:** _not attempted_

### S4. Optimizer persistence across circuit legs
**Problem:** Adam state is rebuilt each `train()` call (recommendations item 4) — every leg transition
spikes; multi-leg curricula (C1!) pay it repeatedly.
**Change:** thread `ppo.optimizer.state_dict()` through leg transitions in `_run_sequential_circuit`
(train.py already restores optimizer from checkpoints on `--load`; reuse that path in-process).
**Acceptance:** TensorBoard shows no grad-norm/KL spike at a leg boundary in a 2-leg smoke circuit.
**Outcome:** _not attempted_

### S5. KL-anchor regularization (fallback if demo-BC retention fails on the boss)
**Idea:** instead of NLL on demo *states only*, anchor the policy to the BC checkpoint with
`β·KL(π_anchor‖π)` computed on **rollout** states — protects behavior on the whole visited manifold,
annealing β over training. The KL-rollback code already snapshots full state dicts
(`ml_ppo.py:571-576`); the anchor is a frozen second network (`Network.load`); analytic per-head
categorical KL must replicate the masking (−1e9 fill, type-conditioned direction mask, single-type
exclusion).
**When:** only if B4's falsification path triggers twice. Not before — demo-BC is proven; this is not.
**Acceptance:** B4 retry with anchor holds boss success without freezing learning (β anneals to ~0 and
success persists).
**Outcome:** _not attempted_

### S6. Self-imitation on harvested successes
**Idea:** generalize the demo mechanism: keep a rolling buffer of the agent's OWN successful episodes
(ending `success-*`) and add a BC/SIL term over samples from it (advantage-clipped: positive-advantage
actions only). Turns the first lucky boss kill into a persistent teacher — directly attacks the
non-monotonicity problem (0.58→0.96 regressions), because success behavior stops being forgettable.
**Sketch:** collect (obs, mask, action) tuples in the rollout worker when an episode ends in success;
FIFO cap (~50 episodes); each optimize round samples ≤1024 steps into a demo-batch-shaped tensor and
reuses the demo-BC code path (B2's concat makes this nearly free).
**Order:** build after B2; try after (or alongside) B4. Highest-upside structural item.
**Acceptance:** on the R6 setup, the success-rate trajectory becomes monotone-ish (no >50% relative
regression between consecutive milestone wakes once success > 0.2).
**Outcome:** _not attempted_

### S7. Per-leg ent_coef in circuit YAML
**Goal:** exploration needs differ per leg (boss legs need more). `TrainingCircuitEntry` gets an
optional `ent_coef` field, threaded into PPO kwargs per leg (`train_once`).
**Acceptance:** config test + a smoke circuit logging different `ent_coef` per leg.
**Outcome:** _not attempted_

### S8. Entity distance/vector features (observation)
**Goal:** add per-entity distance-to-Link (and Δx,Δy direction) to the 8-dim entity features so
dodge/approach doesn't have to be inferred from pixels. The old system fed explicit objective/enemy/
projectile/aligned-enemy vectors and beat the boss (§1.6.4); the current design deliberately dropped
positions in favor of CoordConv (`observation_wrapper.py:293-294`) — this task re-tests that bet in
the one room where it matters most.
**Cost:** observation-space change invalidates all checkpoints (obs shape mismatch) — forks the model
lineage. Do it only if V5 shows dodging is the bottleneck AND B4/C1 both underperform; if done, train
the boss micro-scenario from BC-of-demos rather than old checkpoints.
**Acceptance:** A/B on `dungeon1-aquamentus-kill` from-BC: hit-taken rate per episode drops materially.
**Outcome:** _not attempted_

---

## 8. Phase I — Integration to the gates

### I1. EXPERIMENT: late-chain integration (Gate B)
**Precondition:** Gate A reached by any arm (R6, B4, or C1). P2 fixed (evals must terminate).
**Protocol:**
- Circuit: weighted with P5's `primary` gating — `dungeon1-late-chain` (primary, exit success ≥ 0.1) +
  `dungeon1-wallmaster-north-exit` (retention, small weight) + `dungeon1-aquamentus-kill` (rehearsal).
  All demos active (B2). Budget 1.5M steps.
- Watch for the exp5 regression signature (late-chain progress/max dropping 16→14): if boss rehearsal
  steals capacity, rebalance ONCE, with ≥3 wakes of journal evidence.
- Late-chain semantics check before launch: with R3, leaving 1_35 south mid-chain is priced
  (`penalty-wrong-location`) and hints mask it; confirm `NowhereToGoCondition` can't fire spuriously in
  FIGHT rooms (it's MOVE-gated, `end_conditions.py:266-274` — stays true after R3).
- Final eval: `dungeon1-late-chain` 100 eps vs exp4's eval JSON (median 10/11, max 16, success 0.0).
**Success:** Gate B: success > 0, progress/max = 17.
**Falsification:** boss solved in isolation but never inside the chain after 1.5M steps → context
transfer failure → C4 start diversity + C2 states harvested *from chain prefixes* (enter 1_35 via a
real 1_45 playthrough — health/beam state on entry differs from the 1_35s savestate). Iterate once
before rethinking.
**Outcome:** _not attempted_

### I2. EXPERIMENT: full dungeon 1 (Gate C)
**Precondition:** Gate B.
**Protocol:** `dungeon1-all-items` (start 1_73s) — the key/locked-door/bomb routing segment plus the
solved late chain. Weighted circuit: all-items (primary) + late-chain + retention scenarios. Budget 2M.
Final eval 100 eps. Watch `failure-no-key` and the milestone-13 pileup (the baseline's wall).
**Success:** Gate C: success > 0.
**Outcome:** _not attempted_

### I3. EXPERIMENT: full game gate (Gate D)
**Precondition:** Gate C.
**Protocol:** `full-game-all-items-finite` fine-tune (or straight eval if I2's model generalizes),
100-ep eval vs the baseline JSON. Success = Gate D (§0.1). This closes the plan's scope — do NOT widen
into dungeon 2+ here; write the follow-on plan as a new document.
**Outcome:** _not attempted_

---

## 9. Backlog (X) and Decision Log

### X1. Auto-demo harvesting from scripted policies
Generalize V2: parameterized scripted policies (jittered timings, varied paths) → many validated traces
→ BC on a *distribution* instead of one trajectory (mitigates single-trace overfit; complements S6).
**Outcome:** _not attempted_

### X2. Auxiliary boss-HP prediction loss
Auxiliary head predicting nearest-enemy HP next step; forces combat-relevant representations before
rewards flow. Cheap once S-phase infra exists; obs already exposes enemy HP as a target.
**Outcome:** _not attempted_

### X3. Sustained-threshold exit criteria
Implement `exit-criteria: {metric, threshold, consecutive: 3}` in train.py so legs can't exit on one
lucky window (codifies §0.4 rule 1). Small change that R6/B4/C1/I1 all want.
**Outcome:** _not attempted_

### X4. Multi-seed replication harness for micro-scenarios
Micro-scenarios are cheap; non-monotonicity is the enemy. A wrapper running N=3 seeds of a
micro-scenario config and reporting the success-rate band makes single-run claims honest.
**Outcome:** _not attempted_

### X5. Beam-alignment shaping (fired-correctly / didn't-fire)
Port the original-design pair (§1.6.4): small penalty (−0.05) for moving instead of firing when an
enemy is beam-aligned and beams are available; small reward (+0.05) for firing while aligned. Needs an
alignment computation (row/col overlap with Link) — cheap from existing state. Guarded: only in FIGHT
rooms; cap per episode to prevent farming alignment without commitment.
**Outcome:** _not attempted_

### X6. Update stale docs
`docs/specs/combat-engagement.md` references the removed `DANGER_TILE_PENALTY`;
`docs/specs/nes-mechanics.md` lacks an Aquamentus/boss-room/shutter section (V6/V2 produce the facts).
Fold V-phase findings in; also record the §1.6 original-design digest pointers.
**Outcome:** _not attempted_

### X7. Positive-clamp audit for multi-reward kill steps
Step totals clamp to ±1 pre-terminal (`rewards.py:121`). A bomb double-hit (+2.0 boss damage under R2)
or kill+pickup steps saturate — decide whether FIGHT-room damage steps deserve a widened clamp like
terminals got, or whether saturation is acceptable. Analysis task; pairs with R4.
**Outcome:** _not attempted_

### X8. Fireball danger shaping (danger tiles / wavefront field)
If V5 shows dodging matters: transient danger field around in-flight fireballs (predictable
trajectories) — either a small step-on penalty à la original-design's DANGER/WARNING stamping (§1.6.5)
or a wavefront potential term. Significant machinery; evaluate S8's simpler fix first.
**Outcome:** _not attempted_

### Decision Log
_Append dated entries when outcomes change strategy._

- **2026-08-03 (planning):** Root cause assessed as reward-spec bug (§1.1) with high confidence from
  source verification; empirical confirmation assigned to V1. Chosen spine: probes → spec fix → retrain
  → demos/curriculum as second wave → integration. General combat-engagement spec deferred: observed
  failure is zero-attacks, not camping; `DANGER_TILE_PENALTY` no longer exists in code.
- **2026-08-03 (planning, addendum):** `origin/original-design` archaeology (§1.6) confirms the design
  direction independently: the system that beat D1 used terminal+max-penalty exits, boss-directed
  movement shaping, 3:1 hit:damage economics, full-health boss-room starts, and beam-alignment shaping
  — and recorded 68.3% boss kills with zero flee-outs. Added C6, X5; strengthened R1/R2/R4/S8/X8
  rationale. Port structure, not constants (their shaping was non-potential-based and per-area models
  had no retention constraint).
- **2026-08-04 (P1+P2 landed, PR #192):** EXP6 tooling done. Two findings that change what the next
  agent should believe:
  1. **The eval hang was hypothesis 1, not an emulator wedge.** `Timeout` resetting its no-progress
     clock on every next-room re-entry was the whole bug. With the first-entry-only fix, late-chain on
     exp5-final runs ~28s/episode and one of 5 episodes ended `failure-no-progress` — the ending that
     structurally could not fire before. The `--max-steps` / `eval-step-cap` backstop never triggered.
     **Consequence:** no emulator-timeout work is needed; treat P2 as closed and trust late-chain
     numbers from here on. Historical exp5 late-chain "hangs" need no other explanation.
  2. **The boss takes literally zero damage, confirmed on a 10-episode sample.** `--combat-trace`
     reports `Attacks: 0/N  hits=0  damage_dealt=0` and `Aquamentus hp 6..6` in 10/10 episodes. This
     upgrades §1.1's "zero reward-hit entries" from a single-eval reading to a reproducible per-step
     observation, and it means V1's falsification branch (agent attacks and fails) is **already ruled
     out** — the spec-bug theory keeps primacy. V1 remains open only to attach realized episode returns
     to the payoff table.
  Also visible in the traces, pre-empting V4: the south exit is taken from tile y=`0x13`, one tile
  above the `tile.y >= 0x14` hint-mask threshold, so hints never mask it. R3 alone will not close that
  leak.
- **2026-08-04 (100-ep late-chain readings, both finals):** exp5-final late-chain measured for the
  first time — 100 episodes in **54m44s** (bound < 2h): success 0.0, `progress/max` 16, median 10/11,
  endings `death 0.42 / no-progress 0.32 / wallmastered 0.21 / stuck 0.05`, **`eval-step-cap` 0.0**.
  Versus exp4-final (100 eps, pre-existing): success 0.0, `progress/max` 16, median 8/11, endings
  `death 0.72 / no-progress 0.18 / wallmastered 0.08 / stuck 0.02`. Full table in P2's outcome. Three
  consequences:
  1. **`no-progress` at 0.32 quantifies the old hang:** about a third of late-chain episodes were
     livelock candidates, and the step-cap backstop never fired across 100 episodes — the `Timeout`
     fix covers the entire failure mode. No further eval-robustness work is warranted.
  2. **Neither final can reach milestone 17** (`progress/max` = 16 for both). Gate B is therefore
     blocked on the boss, not on the chain: no amount of late-chain tuning moves it until Phase R
     makes the boss killable. Do not spend a training budget on late-chain before R6.
  3. **A 1.0 micro-scenario eval says nothing about chain performance.** exp5-final scores 1.0 on
     standalone `dungeon1-wallmaster-north-exit` yet its late-chain `failure-wallmastered` is 0.21
     (up from exp4's 0.08). The skill is retained; it transfers poorly under chained entry conditions.
     Read every gate on its own scenario — this is §0.4 rule 2 applied to evals, not just exit criteria.

---

## 10. Reference card

### 10.1 Commands
```bash
source .venv/bin/activate
pytest tests/ -v                          # standard tests (slow PPO tests excluded)
pylint triforce/ triforce_debugger/ debug.py evaluate.py train.py record.py

python train.py <scenario|circuit> [all-items] [impala-multihead] \
  --output DIR --iterations N --load MODEL.pt --parallel N \
  --demo-trace T.txt --demo-scenario S --demo-prefix-east N --demo-bc-coeff 0.1
python evaluate.py <model.pt|dir> <scenario> --episodes 100      # prefer the OMP eval plugin for real runs
python evaluate.py --compare a.eval.json b.eval.json
python diagnose.py --model impala-multihead --scenario S --model-path P --episodes 20 [-o out.txt]
python scripts/behavior_clone.py --model-path P --scenario S --demo-trace T --prefix-east 0 \
  --output OUT.pt --epochs 1000 --min-accuracy 0.95

# read the original-design branch WITHOUT checking it out:
git ls-tree -r origin/original-design --name-only
git show origin/original-design:triforce/critics.py
```

### 10.2 Checkpoints (gitignored, verified on disk at planning time)
```
training/experiments/experiment5/runs/experiment5-circuit/4/impala-multihead_all-items.pt        # exp5 final (boss failure model)
training/experiments/experiment5/runs/experiment5-circuit/4/checkpoints/impala-multihead_all-items_dungeon1-late-chain_8335360.pt   # +optimizer
training/experiments/experiment5/runs/experiment5-circuit/3/checkpoints/impala-multihead_all-items_experiment5-boss-transfer_7331840.pt
training/experiments/experiment4/runs/experiment4-circuit/0/impala-multihead_all-items.pt        # exp4 final — best late-chain (max 16)
training/experiments/experiment3/runs/experiment3-circuit/1/checkpoints/impala-multihead_all-items_dungeon1-wallmaster-north-exit_2273280.pt
training/experiments/baseline/runs/all-items-circuit/0/checkpoints/impala-multihead_all-items_learn-items_2199552.pt  # clean pre-endgame base
```

### 10.3 Baseline eval JSONs to compare against
```
…experiment4…/impala-multihead_all-items.dungeon1-late-chain.eval.json        # 100 eps, median 10/11, max 16
…experiment5…/4/impala-multihead_all-items.dungeon1-aquamentus-east.eval.json # 100 eps, left-boss-room 1.0
…baseline…/impala-multihead_all-items_all-items-polish_10719232.eval.json     # 100 eps, median 13/17
models/dungeon1boss.zip.evaluation.json @ origin/original-design              # 68.3% kill (old system, context only)
```

### 10.4 Experiment protocol (mandatory for training tasks)
Read `docs/experiments/experiment-memory.md` + the most relevant `docs/experiments/<id>-summary.md`.
Write `training/experiments/<id>/journal.md` BEFORE starting (baseline checkpoint/eval, metric that must
improve, load-vs-scratch, scenario/circuit, changes, success criteria, final eval plan). Every milestone
wake: exactly one run action + per-metric tuning decision recorded as a journal table. Finish: final
eval via the OMP evaluation plugin, `summary.md`, append to `experiment-memory.md`, tracked
`docs/experiments/<id>-summary.md`. Never edit `.omp/extensions` mid-experiment. Treat the entropy
floor as a constraint, not a nuisance wake (exp5's loosening spiral is the anti-pattern).

### 10.5 Demo trace format (`docs/experiments/demos/*.txt`)
Header prose (start scenario/state/goal), then `## Normalized forward trace` — one `ACTION DIRECTION`
per line (currently `MOVE E` etc.; B1 extends to `SWORD N`, `BEAMS E`, `BOMBS E`). Optional
`## Original reverse trace` provenance. Validate with `diagnose.py --demo-report`.

### 10.6 RAM override cheat sheet (scenario YAML)
`per_reset` (once at reset) / `per_room` (each room change) / `per_frame` (every frame) — dicts of
named values from `triforce/zelda_game_data.txt` (`custom_integrations/Zelda-NES/data.json`), applied in
`StateChangeWrapper._apply_modifications` (`state_change_wrapper.py:405-417`). Relevant names:
`hearts_and_containers` (0x22 = 3 containers/2 filled), `partial_hearts` (≥0x80 for the beam gate),
`obj_health_1`…`obj_health_c` (per-slot enemy HP, high nibble = HP units: 0x10 = 1), `bombs`, `keys`,
`regular_boomerang`. Safe: inventory/health/equipment/RNG. NOT safe: Link position, object states.

### 10.7 Constant map (current values, `triforce/critics.py` unless noted)
```
reward-hit +0.25·decay | reward-beam-hit +0.5·decay | reward-bomb-hit +0.5/hit | penalty-bomb-used −0.25
penalty-attack-miss −0.01 | penalty-lost-health −0.25/half-heart | penalty-lost-beams −0.25
positives halved on damage steps (critics.py:146-147, equipment pickups exempt)
penalty-wrong-location −0.25 | reward-new-location +1.0 | reward-revisit-location +0.05
PBRS ±Δdist/20 (γ=1) | stalling −0.01→−0.02 after 150 steps, ramp 1850, reset on room/kill/pickup
terminal: ±20 via clamp exemption (rewards.py:112-121); sets in scenario_wrapper.py:22-27
Aquamentus: 6 HP · wood sword/beam 1 dmg · bomb 4 dmg · fireball 1 heart · 3×3 tiles · slot HP obj_health_N
combat decay: 0.5^(events−8) per room (anti-farming; irrelevant for a 6-HP boss)
original-design boss profile (context): hit +0.75 · damage −0.25 · move-to-boss ±0.25/step · exit terminal −1.0
```
