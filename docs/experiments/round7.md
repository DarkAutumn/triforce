# Round 7: lingering measurements and post-round-6 behavior experiments

## Summary

Lee resumed experiment mode on Nov 3: keep the omp todo live, reconcile it with `todo.md` between experiments,
run experiments, log every result and commit each finished item. Aquamentus r30-exit is now promoted.

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
- Backlog item 2 complete: r30-exit trains kill → heart → east exit. Matched exit-task eval improved 59/60 →
  60/60; mean pickup-to-exit 92.0 → 11.1 actions, worst 1,963 → 77; health change +0.61 → +0.69 hearts.
  Both matched full chains removed the two old post-heart timeouts. Other families still limit the chain.

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

## 3. Observation audit and returning-boomerang correction

Read observation.py, Link/enemy/projectile decoding and the corresponding collision/weapon routines in the Zelda
disassembly. Read-only scouts independently covered stun/collision and chase/weapon/owner semantics. The old archive
could not be read by the archive tool (921 MB exceeds its 256 MB limit); streamed only its observation wrapper into
`/tmp`. That archived wrapper is a pixel viewport/grayscale transform, not evidence of an additional structured
feature list (`triforce_lib/zelda_observation_wrapper.py` L41–105).

| Input/mechanic | Finding and decision | Evidence |
|---|---|---|
| Enemy boomerang stun | `$3D+i ← $10`, one tick per 10 frames: nominal 160 frames. Existing `/16` is correct; do not rescale. Dodongo's bomb stun is a different `$20` mechanism. | Z_01 L5914–5919; Z_07 L481–507; runtime injected timer decremented every 10 frames |
| Damage invincibility | Link `$4F0 ← $18` (nominal 48 frames); ordinary enemies `$10` (32 frames), half-rate countdown. Existing `/32` does not clip these. Clock is already observed and separately controls protection. | Z_01 L6589–6605,L6687–6691; Z_07 L5782–5788,L1877–1884 |
| Chase target | Actual refresh flag + target X/Y missing. Bundle actual `$60/$61/$62` with full-byte coordinate normalization; no RNG/timer initially. The ASM's last-frame comment contradicts executed instruction order. | Z_07 L1888–1951,L3788–3811 |
| Link boomerang flight | Active alone omits position and tactical phase; bundle outbound/slow-outward/spark/slow-return/fast-return. `$80+` is bait, not boomerang. | Z_05 L2901–2948; Z_07 L3857–4280 |
| Projectile harm | Stunned monsters and returning/bouncing projectiles currently can read hurts_on_touch. Bundle corrected harmful-state semantics and boomerang return phase; present/active remains separate. | observation.py L435–451; Z_01 L5613–5714; Z_07 L3857–4280 |
| Projectile owner | Reciprocal `$42C` refs exist for Goriya boomerangs only. Owner relation is optional and must validate both slots/types; ordinary shots retain stale scratch data. No nearest-enemy guessed owner. | Z_04 L590–596,L11361–11391; Z_01 L4074–4086 |
| Shield blocking | ArrowDeflected is also Darknut/immunity parry. Goriya blocking produces a spark, then return; return/catch alone is not a block. Use per-frame collision evidence before increasing rewards (item 4). | Z_01 L5707–5714,L5897–5912,L6312–6333; Z_07 L4265–4280 |

**Schema decision:** keep schema 1 during the isolated Aquamentus/task/reward experiments. No speculative input
changes or incidental timer rescalings now. The chase/phase/harm bundle needs a deliberately versioned full-retrain
comparison, not loading existing weights under changed meanings; keep it recorded as a prerequisite to the later
from-scratch boomerang study. Audit is complete; the proposed schema bundle is not implemented or claimed complete.

### Return-state decoding bug: fixed before reward comparisons

`Link.get_animation_state(BOOMERANG)` used **decimal 10–57** instead of hex `$10–$57`. Raw `$40–$57` return states
therefore reported inactive, ending future-credit simulation before the weapon finished its flight.

- Live probe through `GameSession.load/act`: after leaving the south doorway, threw east; all 63 observed return
  frames were incorrectly INACTIVE. Initial attempted throws in the doorway correctly did not launch anything;
  moving once was insufficient because the first move was interrupted by the door event.
- Changed only the two bounds to `0x10` and `0x57`, leaving observation/action schemas and masks unchanged.
- Same live probe afterwards: all 63 return frames ACTIVE, zero incorrectly inactive. Output is stored in
  `models/r29-linger-baseline/boomerang-decode-probe.json` (post-fix trace).
- Real-emulator regression failed before on the ACTIVE assertion, then passed after; it checks return → catch
  and excludes bait from boomerang activity. Core/env/reward/tracker checks: 75 passed.

This is a correctness fix, not evidence of a model/chain improvement. Re-evaluate old checkpoints under the corrected
credit path and the experimental task before comparing continued models.

## 4. Alternate-YAML CLI integrity fix (prerequisite to r30)

The first CLI evaluation with `/tmp/boss-aquamentus-exit.yaml` still printed `boss-dead-and-heart` and 60/60 success,
despite the file requesting only `exit-correct`. This was a real config-loss bug, not a valid exit-task baseline.

- `evaluate.main` loaded the supplied file only to extract `config.name`, then reloaded the default specialist.
- Training similarly discarded the file before `_train`, model/curriculum setup and `ZeldaEnvFactory`.
- Fix: carry the loaded `SpecialistConfig` into evaluation and training workers; use its name only for registry
  identity/log labeling. Worker factory explicitly accepts the config object, including spawn/pickle use.
- Two real CLI regressions failed before: evaluation ran three default seed states instead of the single supplied
  one; spawned training completed zero episodes instead of eight one-action episodes. Both pass after the fix.
- Training/eval/PPO checks: 15 passed. Actual CLI smoke on the supplied exit config ran CUDA training with two
  workers (256 actions, 3 successful completed episodes averaging 54 actions) and evaluated the old promoted model.
- Full post-fix dev suite: 386 tests passed in 86.59 seconds.
- Correct old-model baseline with max_steps 400: **57/60 correct exits (95%)**, mean 90 actions, +0.61 hearts;
  mean longest quiet stretch 48 actions, worst 365. Stored as `models/r30-aquamentus-exit-old.json`.

### Historical experiment implications

`git log -S` shows name-only loading dates back to the initial/round-1 CLI implementation. `run.json` confirms the
following arms supplied temporary YAML files, whose overrides the CLI discarded:

| Arm | Supplied file |
|---|---|
| r20-rope-pt | `/tmp/d2-rope-pt.yaml` |
| r21-goriya-chainhp | `/tmp/goriya-chainhp.yaml` |
| r22-skg-chainhp | `/tmp/skg-chainhp.yaml` |
| r25-skg-hearts | `/tmp/skg-hearts.yaml` |
| r27-rope-hearts | `/tmp/rope-hearts.yaml` |

Their measured model/chain scores remain real, but those comparisons **do not establish the requested treatment's
effect**: the variant file was not used for training. Old main-YAML values cannot be recovered from these run files,
so do not retrospectively invent exact treatment settings. Round 6 is annotated accordingly. Lee's explicit rejected
ideas stay rejected; this audit is not authorization to reintroduce pass-through damage or heart-only objectives.

## 5. Aquamentus exit training protocol (r30)

Variant file preserved at `models/boss-aquamentus/r30-exit/config.yaml`: success `[exit-correct]`, otherwise current
settings including max_steps 400 and hp `[0.75,max]`, half full health. Bombs off, schema 1 unchanged. Warm-start from
promoted r23-more; stop at balanced 99%, at most 2M more actions. Keep the current promotion until comparison finishes.

Re-explored with the exit task: 3/3 successful replays, 42,957 env actions; paths 109/91/91 actions, final health
2.0/1.5/1.5. Files are under `trajectories/r30-exit/boss-aquamentus/`; snapshots are before each action, so the terminal
0x36 room is not a curriculum start. Use this separate directory for the candidate, not the old heart-ending paths.
Compare against the corrected old-model exit-task baseline, then both full chains and the post-D1/D2 segments if
promoted. Track boss-kill/heart/exit timing, health and quiet-gap distribution—not success alone.

### Results and promotion

Training stopped early after 50 additional PPO updates / 204,500 additional actions, 248.71 seconds. The checkpoint
stores cumulative 633 updates / 2,590,136 actions; these are not the experiment's additional budget.

Both models were evaluated on the exit task with the same current health distribution, 20 episodes per entry state
(60 total), sampled policies and seeds 0 onward. The old baseline is `models/r29-linger-baseline/aquamentus-through-exit.json`;
the candidate's records are `models/boss-aquamentus/r30-exit/eval.json`.

| Exit-task measurement | r23-more | r30-exit |
|---|---:|---:|
| Successful exits | 59/60 | 60/60 |
| Mean health change, hearts | +0.608 | +0.692 |
| Mean boss-kill action | 27.15 | 27.95 |
| Mean pickup-to-exit actions | 91.97 | 11.13 |
| Worst pickup-to-exit actions | 1,963 | 77 |
| Mean total actions | 125.78 | 46.88 |

Matched full chains, 40 runs per route / seeds 0–39: the candidate reached Aquamentus 12 times on each route,
with 10 exits and two low-health fight deaths instead of the old eight exits, two deaths and two post-heart
timeouts. Successful candidate visits averaged 50.7 actions. Dodongo-room reaches were 0/40 on milestone1 and
3/40 on milestone1-fairy; this does not establish a general chain-success improvement. The demonstrated gain is
eliminating the post-heart stall without worsening the matched specialist health result or boss-kill timing.

After promotion, an independent full-chain sweep used seeds 40–59; post-D1 and D2 segments used seeds 0–19:

| Segment | Runs | Dodongo room reached |
|---|---:|---:|
| Full milestone1 | 20 | 1 |
| Full milestone1-fairy | 20 | 3 |
| Post-D1 milestone1 | 20 | 0 |
| Post-D1 milestone1-fairy | 20 | 4 |
| D2 entrance | 20 | 3 |

No execution errors in the sweep. Five D1 Goriya deaths and three 0x53 deaths on each full route remain the first
bottlenecks; D2 rope rooms and Goriyas limit the later segments. Post-D1/D2 skip Aquamentus, so their outcomes are
current-team measurements, not evidence of an Aquamentus treatment effect. Raw chains and sweep summaries are in
`models/boss-aquamentus/r30-exit/`.

**Promoted r30-exit.** `boss-aquamentus.yaml` now succeeds on the correct exit, after the objective requires the
kill and heart container. Replaced its default curriculum trajectories with the explored complete-visit paths.
No observation-schema change, damage-penalty addition or special post-kill controller.

## 6. Confirmed shield collisions and isolated experiment controls

Before increasing the Goriya reward, reproduced its missing credit using naturally thrown boomerangs, not injected
projectiles or a model evaluation. Protocol: fresh clean `1_44w`, RNG seeds 0–4; leave the doorway, place Link at
(112,144) with three hearts, clock/invincibility off; up to 250 WAIT 32 actions per seed, facing opposite the closest
outgoing Goriya boomerang via RAM. Inspect consecutive actually executed frames, with look-ahead disabled.

The old trace found seven shield-eligible parry collisions but rewarded only one. Replaying after the fix found
the same seven collisions at exactly the same seed/action pairs: `(0,106)`, `(0,121)`, `(0,122)`, `(1,65)`, `(3,18)`,
`(3,67)`, `(4,218)`; all seven earned block rewards. Example: frame 2973→2974, slot 11 outgoing `$17`→spark `$20`,
`$604=1` while `$605=0`; both controllable sound endpoints were silent. The sound-onset implementation missed it.
Evidence: `models/r31-block-evidence/{before.json,before-summary.json,after.json}`.

`projectile_blocks.py` now confirms projectile-specific collision signatures together with shield/facing/proximity
and the current-frame parry request. It rejects immune/unrelated clinks, harmless returns, unsupported/magic-only
shots without an eligible shield, stationary wall sparks and boundary fireball destruction. Actual executed
frames only: no replacement of the user's callback, altered input duration or speculative look-ahead collection.
Block events retain collision frame, slot, type and room; reward/decay counts each collision, including simultaneous
shots and attack/settling/uncontrollable frames. A heart pickup in the collision frame does not erase the block.
Episode, curriculum, evaluation and chain metrics retain confirmed counts; absent historical measurements stay absent.

Two requested later experiments have complete, default-off controls so training arms can remain isolated:

- `reward.nearby_items`: multi-source PBRS includes hearts/clocks alongside exits, without replacing the room task.
  Live greedy walker: nearby clock then heart then correct exit, +0.10 shaping, one heart healed (2→3 hearts).
  With a nearer exit: +0.05 shaping, correct exit at two hearts, distant heart left. This is runtime proof of the
  control, not evidence that trained models improve; backlog item 5 remains pending.
- `no_progress_steps`: positive shared quiet-tail cutoff, optional and disabled by default. Live WAIT actions 1/2
  continued; action 3 truncated with `no-room-progress`, soft failure, 27 quiet frames and no terminal penalty.
  Legacy timeouts remain when disabled; runner per-room cutoff retains the global guard. Backlog item 6 needs
  family-specific successful-gap measurement and comparison before enabling any threshold.

First integrated regressions exposed NumPy byte arithmetic and invalid WAIT constructions in new test fixtures;
fixed integral modular arithmetic and valid actions. Removed an incidental serialized-emulator-clock comparison
in favor of actual RAM/screen/timing/callback parity, and an unrelated exit-classification test pin. Full dev suite:
**447 passed in 98.43 seconds**. No observation/action schema change.

CUDA smoke with the triple-block config, warm-started from promoted r21-goriya-chainhp: two updates / 1,024 actual
actions, five completed episodes (four successes), 10.7 seconds. Training emitted measured `blocks=0` for these
five episodes; this proves the updated CUDA/metric path runs, not a nonzero-block reward or a model improvement.
The natural trace above provides the nonzero-block proof. No CUDA exhaustion; Lee's player was left running.

### r31 matched block-reward comparison

Configs preserved at `models/goriya/r31-block{1,3}/config.yaml`. Both continue the same r21-goriya-chainhp checkpoint,
with current YAML health mix `[0.15,max]` / half full health, bombs off, the same curriculum, seed 31, 245 updates
(about 1M additional actions) and no success early stop. Only the confirmed-block multiplier differs: 1 versus 3.
Separate matched corner arm `models/goriya/r31-quiet005/config.yaml`: ordinary confirmed-block multiplier 1 plus
0.05 additional quiet active-fight cost after the existing grace. Same seed, six workers, 245 updates and canonical
main curriculum; no nearby-item shaping or measured truncation. The block arms run on main code `6421275`; this
arm runs from immutable dev code `aa30a2f`, verified by 462 tests and faithful fight/post-fight replays. Do not merge
source while the cohort runs. All checkpoints/logs are under main's normal model/log roots. No candidate promoted.
Keep nearby-item shaping and measured cutoff off. Re-evaluate the old model on the corrected detector/current YAML;
compare success, real/total health, confirmed blocks and successful/failing quiet-gap distributions per room, then
both full chains and the relevant post-D1/D2 segments. A stronger block reward is not promoted merely for more blocks;
it must preserve health/success and avoid reward-producing stalemates. Evaluate the corner-stall lever separately.
