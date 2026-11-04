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
- Worker-seed prerequisite fixed and verified (§7): real rollout tensors repeat with the same seed,
  distinct workers/seeds differ, episode streams advance; CPU CLI/CUDA smoke and all 462 tests passed.
- Backlog item 4 complete: r31-block1 promoted after 800 frozen-team chains (§6). Ordinary reward has the lowest
  evaluation damage; stronger block/quiet rewards stay disabled. Post-fight D2 navigation remains a measured problem.
- R32 nearby-item on/off training/evaluation is running on unchanged main (§9); do not change its source, YAML
  or states. Verified dev prerequisites wait for that freeze: cutoff value bootstrap (§10), per-arm chain configs
  (§11), and key-pickup/bomb guards for entry builders (§12). The D2 route/state migration is not started.

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
eliminating the post-heart stall with better matched specialist health and boss-kill timing within one action.

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

Two requested later experiments have default-off controls; the cutoff still needs the PPO correction found in §8:

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

### Current-detector baseline and faithful stall replays

Promoted r21-goriya-chainhp re-evaluated on the corrected current code/YAML: 178/240 success, health −0.96 hearts,
45 confirmed blocks (0.1875 per episode). Output: `models/r31-block-evidence/promoted-eval.json`.

| Room | Episodes | Successes | Confirmed blocks | Mean health change |
|---|---:|---:|---:|---:|
| 1:0x44 | 60 | 44 | 15 | −0.908 |
| 1:0x23 | 60 | 46 | 13 | −0.617 |
| 2:0x5e | 60 | 45 | 7 | −1.083 |
| 2:0x1e | 60 | 43 | 10 | −1.225 |

All 240 episode outcome/time/health/heart records match the earlier baseline after excluding the new block field
and lingering measurements. Of the lingering summaries, 239 match; seed 207 changes only longest quiet frames
55→79, retaining six quiet actions and 60 actions/657 frames overall. Cause not isolated; use the current baseline
for the r31 comparison rather than claiming every historical measurement is unchanged.

Faithful sampled-policy replays, same entry state/RNG/torch seeds as measured failures:

- `1_44w_0`, seed 62: 2,000-action timeout, 1,971-action quiet stretch. All 1,822 actions at quiet-tail ≥150 still
  had three living Goriyas and FIGHT objective. Link stayed by the left wall; the Goriyas remained at x=208.
  Quiet actions: WAIT 760, SWORD 694, MOVE 368. Existing stall cost totaled −27.47 during that interval, without
  breaking the stalemate. This validates testing a stronger fight-only quiet penalty, not merely counting WAIT.
- `1_23s_0`, seed 5: same 2,000-action timeout but no enemies remained during its 1,753 late quiet actions.
  Objective was TREASURE; 1,751 MOVE actions failed to collect the key. This is a distinct post-fight pickup problem,
  not evidence of shadow-Link avoidance and not something a fight-only penalty can fix.

Raw per-action positions, chase target/flag, enemies, objective and costs: `models/r31-block-evidence/corner-replays.json`.

Prepared an independent default-zero `reward.quiet_enemy_penalty` in the dev worktree: additional per-action cost
after the existing 150-action grace, based on the shared actual-progress tail, only while targetable living enemies
remain. Damage and round trips do not reset this grace; inactive/burrowed/dying enemies and post-fight pickup time
do not trigger it. Main is unchanged while r31 trains.

Live replay with coefficient 0.05 keeps the same sampled-policy trajectories: the 0x44 stalemate receives −91.05
additional cost; the 0x23 post-fight wander receives zero additional cost. This proves gating/reward behavior,
not that training improves. Output: `models/r31-block-evidence/corner-replays-quiet005.json`.

### Successful-gap guard for the later truncation experiment

Historical successful chain visits across the 120 tracked baseline runs: 656 stalfos-keese-gel visits had worst
quiet gap 94 actions; 84 Goriya visits had worst 126, with none reaching 200. D2 rope had 263 successful visits,
five reaching 100 and three reaching 200 (worst 2,410). A blanket 200 cutoff would discard real rope successes.
Standalone Goriya also exposes a different distribution: 0x23 would lose 12/46 successful episodes at 200, versus
1/44 in 0x44, 1/43 in 0x1e and 0/45 in 0x5e. Measure by family/room and compare training effects before enabling
thresholds; current promoted configs retain no measured cutoff.

Quiet-fight control verification after retaining Lee's newer GUI changes: **462 full-suite tests passed in
231.79 seconds** under concurrent r31 training. The new real-emulator regression checks the exact grace boundary
and removes targetable enemies without clearing the long quiet tail: the additional penalty stops after the fight.
No model improvement is claimed from this control verification.

Additional real weapon-option reward proof: an injected outgoing hostile boomerang at x=180 hits Link's shield
on frame 17 of `BEAMS E`, after its 16 control frames. No health loss; the unmocked full gameplay critic pays
`reward-block-projectile=0.5` (alongside the ordinary attack-miss cost). Replaced the mocked action-routing test
with this native settling-frame regression; removed fake legacy-sound fields, a repeated positive-multiplier row
and private-counter assertions. Reward suite: **25 passed in 2.55 seconds**. No live runtime code/YAML changed.

### Completed cohort evaluation and frozen-team selection

All three r31 arms finished 245 updates / 1,002,540 additional actions. Fixed seed-0 evaluation uses the canonical
Goriya task/current detector, 20 episodes per start state (240 total), without enabling nearby-item shaping or
measured truncation. Evaluation outputs: `models/r31-block-evidence/{promoted,r31-block1,r31-block3,r31-quiet005}-eval.json`.

| Policy | Successes / 240 | Real health change | Confirmed blocks | Mean longest quiet actions | Worst quiet actions |
|---|---:|---:|---:|---:|---:|
| Promoted r21 | 178 | −0.96 | 45 | 86.4 | 1,971 |
| Ordinary block reward | 184 | −0.77 | 45 | 45.1 | 1,228 |
| Triple block reward | 182 | −0.87 | 37 | 24.1 | 1,404 |
| Ordinary + 0.05 quiet-fight cost | 189 | −0.85 | 45 | 22.4 | 497 |

The quiet arm improves D1 0x44 versus ordinary (49/60 versus 43/60; −0.775 versus −0.942 hearts), but costs more
health in D2 0x5e (−1.033 versus −0.700; 48/60 versus 51/60). Triple block also reaches 49/60 in 0x44 but falls
to 42/60 in D2 0x1e versus ordinary's 49/60. More reward did not produce more measured blocks here. Section 7's
seed flaw prevents attributing these differences solely to the reward coefficients.

Frozen-team comparison completed after all training ended: each of the four Goriya policies, with every other
promoted specialist pinned, 40 runs per segment (seed 0–39). All **800 chains completed without execution errors**.
Provenance/results: `models/r31-block-evidence/{team-pointers.json,comparison.json,*-summary.json,*.jsonl}`.

| Policy | Full main / 40 | Full fairy / 40 | Post-D1 main / 40 | Post-D1 fairy / 40 | D2 / 40 |
|---|---:|---:|---:|---:|---:|
| Promoted r21 | 0 | 3 | 0 | 8 | 9 |
| Ordinary block reward | 2 | 3 | 2 | 4 | 11 |
| Triple block reward | 1 | 4 | 2 | 6 | 7 |
| Ordinary + 0.05 quiet-fight cost | 2 | 2 | 0 | 6 | 13 |

**Promoted r31-block1.** Lowest evaluation damage (−0.769 versus old −0.958 hearts), better evaluation success,
two full-main successes versus zero, unchanged full-fairy successes and a modest D2 gain. The post-D1 fairy
result regresses 8→4/40; these are mixed small-sample chain results, not a statistically established dominance.
Ordinary is the conservative damage-first choice over triple/quiet, neither of which wins consistently.
Keep block multiplier 1, additional quiet cost 0, nearby-item shaping off and measured truncation disabled.
Per-run `eval.json` copies retain the measured cohort reports; no other model was promoted.

All three new models reduce D1 0x44 deaths 15→13 of 28 visits at the same arrival health. Ordinary's longest
quiet stretch there falls 1,950→27 actions in both full routes, but it does not eliminate D2 stalls: 0x5e reaches
1,792 quiet actions post-D1 fairy and 0x1e reaches 1,207 in a successful full-main chain. Quiet's D2 segment
also stalls for 1,820 actions despite its better standalone worst of 497. Do not equate aggregate eval with
chain behavior, or describe every remaining stall as shadow-Link avoidance.

Faithful native chain replays exactly match the saved room/health/time/outcome records:
- Quiet policy, D2 seed 3, 0x1e: last 1,000 actions all MOVE objective with **zero enemies**, Link fixed at
  (162,93); 996 MOVE and four WAIT actions. Additional living-enemy cost cannot address that post-fight failure.
- Ordinary policy, post-D1 fairy seed 16, 0x5e: last 1,000 actions all MOVE objective with **zero enemies**,
  all MOVE actions, largely pacing across y=93 instead of exiting. Not a shield-block stalemate.

Raw replay traces: `models/r31-block-evidence/chain-stall-replays.json`. Post-fight navigation/curriculum and the
measured-cutoff trial remain open; the quiet-fight lever is not silently broadened to penalize pickups or navigation.

## 7. Training seed integrity finding (r31 prerequisite)

The two completed arms both received `--seed 31`, but first-update episode/health metrics already differed.
Code inspection finds that `train.main` seeds only parent Torch; spawned `_worker_loop` seeds neither its policy
sampler nor environment, and `RolloutBuffer.collect` starts with bare `env.reset()`. `ZeldaEnv`'s private RNG and
the game RNG are only seeded when reset receives an explicit seed. Thus the r31 runs match budget/configuration,
not worker random streams. Do not attribute a policy difference solely to the reward coefficient.

Fixed-seed evaluation remains valid: `_play` explicitly seeds Torch and resets the environment with each episode
seed. Completed results: ordinary block arm 184/240, −0.77 hearts; triple block arm 182/240, −0.87 hearts. Triple
helps D1's 0x44 in this comparison, while ordinary is better in both D2 rooms. Keep empirical policy selection
separate from causal reward claims; wait for the quiet-fight arm and matched frozen-team chains before promotion.

Fix seed propagation after the live cohort finishes, before later A/B trials. Seed each spawned policy sampler
and initial environment reset by worker index, preserve advancing streams across subsequent episodes, and verify
actual rollout reproducibility without curriculum. The existing shared asynchronous curriculum also depends on
worker completion order; seeding alone will not promise bitwise-identical curriculum or GPU training.

Reproduced before fixing, with curriculum and GPU nondeterminism removed: two fresh CPU CLI invocations, same
checkpoint/config, seed 31, two workers, one 128-action update, explicit `1_44w_0`, no curriculum, CPU thread counts
fixed to one. Run A completed no episode; B died after 43 actions/640 frames, losing three hearts. Action counts
also differed (A: WAIT/MOVE/BEAMS/SWORD = 3/78/1/46; B: 6/77/2/43). Evidence preserved in
`models/r31-block-evidence/seed-before.json`. This is a real seeded-gameplay failure, not merely differing timing.

After the cohort's training finished, corrected propagation in dev: CLI seed reaches `PPO.train`, which seeds
training RNGs; each spawned worker seeds Python/NumPy/Torch and its first environment reset using its indexed
stream. In-process collection also seeds its first reset. Later episode resets and rollout batches do not reseed.
Torch-normalized unsigned seeds avoid overflow at the worker offset; NumPy receives its supported 32-bit seed.

The new real-game CLI regression failed before the fix: repeated two-worker runs differed in episode counts,
actions, reset health, frames, rewards and lingering. It now also covers in-process training with the shared emulator.

Native worker-pool smoke (no curriculum, CPU, r21 Goriya weights, two workers, two 64-action batches each,
eight-action episode cap): two seed-31 invocations match every observation/action/reward/frame/boundary/mask tensor;
seed 32 differs. The workers' action streams differ, and both workers' reset hp fractions vary over 16 episodes,
proving streams advance across episode and batch boundaries rather than replaying the first draw. Full digests and
reset hp sequences: `models/r31-block-evidence/seed-after.json`.

The original two fresh CLI invocations now both produce WAIT/MOVE/BEAMS/SWORD = 2/59/10/57 over 128 actions.
CUDA smoke also passed: two updates / 256 actions, two completed successful episodes, mean health change −0.75
hearts. This is path verification, not a candidate model or a claim of deterministic CUDA training. Lee's player
was not killed. Shared asynchronous curriculum and resumed RNG continuation retain the limits stated above.

Integrated dev suite after the seed fix: **462 passed in 204.83 seconds**, including actual single-process and
spawned two-worker gameplay reproducibility regressions. Temporary smoke scripts/checkpoints were removed;
the recorded native-rollout evidence remains in the normal ignored model root.

## 8. Later-experiment prerequisites found during the frozen-team comparison

Read-only evidence review while r31 chains run; no source, state, model or configuration changed for this review.

**Cutoff correctness:** the §6 live smoke proved env flags, no terminal penalty and runner cutoff, not PPO value
bootstrapping. Current `RolloutBuffer.collect` combines terminated/truncated into one done flag, discards the
terminal observation on reset, and `frame_aware_gae` masks its value. Thus the intended measured-cutoff target
`reward + gamma**(frames/8) * V(final_observation)` is missing. Correct that path before any threshold training,
stop the trace across reset, and preserve the existing max-step treatment. A bootstrap from the next episode's
reset observation would also be wrong. Required regression boundaries: nonzero final value, truncation at the
rollout edge, true termination and reset isolation. This is a source-backed finding, not a runtime reproduction.

Expanded successful-gap evidence comes from completed r29 baselines and r30-exit full/post-D1/D2 JSONL artifacts,
including `promoted-full-{main,fairy}.jsonl` seeds 40–59. Successful Stalfos/Keese/Gel visits reach 670 quiet
actions, Goriya 350 in chains (1,274 standalone), and Rope 2,410. No uniform 100–300-action threshold preserves
these measured successes. Moblin's inspected successful maximum is 63 actions/308 frames, making 100 a trial
candidate, not a proven-safe setting. Pairwise main/fairy records often duplicate pre-divergence visits and must
not be pooled as independent evidence. These are historical/current promoted teams, not the live r31 candidates.

**D2 route/state integrity:** `game.yaml` already declares five Ropes and a key in 0x7e; generic FIGHT/TREASURE
objectives already apply. The actual change is canonical routing through E then N, not inventing a key objective.
`entry_states.cross_room` overrides success with target-room arrival, including `exit-wrong`; this cannot prove
required key collection. `step_config` also omits item masks, defaulting bombs on. Before rebuilding the requested
suffix, enforce correct required pickups/inventory and milestone bomb masking. Rebuild all three chains from
0x7d, introduce `2_6es_*` instead of stale W-entry `2_6ew_*`, and validate the carried key counts/collection flags
through Dodongo entry at four hearts. Refresh affected Rope and D2 Goriya curriculum tracks: matching filenames
do not invalidate stored snapshots. No binary state inventories were measured in this review.

## 9. Nearest-heart/clock-or-exit trial (r32)

Started after r31 selection and the verified seed fix merged into main (`f5cc63b`). Families: stalfos-keese-gel,
d2-rope and ow-moblin; each has `r32-nearby-off` and `r32-nearby-on`. Both arms continue the same promoted
checkpoint for that family. Only `reward.nearby_items` differs; current health mix, task, max steps, item masks
and curriculum settings remain unchanged. Bombs off; no measured cutoff or extra quiet-fight cost.

Each arm: seed 32, four workers, 245 updates ×4,096 = 1,003,520 additional actions, CUDA, no success early stop.
At most four train/eval jobs run at once. Config copies live in each run directory; source/seed/protocol snapshots
at `models/r32-nearby-evidence/protocol.json`. Runtime main source, canonical specialist YAML and entry states
stay frozen through all training and background evaluations. Lee's player has not been killed.

All three old promoted checkpoints are re-evaluated under current canonical YAML; each new model uses the same
common evaluation task, 20 episodes per start state, seed 0, three CPU workers. Selection must compare success,
real/total health, hearts taken/left and quiet gaps per room, then both full routes and relevant post-D1/D2 chains
with pinned teams. Nearby-item PBRS changes the learning potential, not runner inputs or accepted exits. Shared
asynchronous curriculum remains scheduling-dependent despite correct seed propagation. No r32 model promoted.

## 10. Measured-cutoff bootstrap correction (dev prerequisite)

Reproduced the §8 value-target gap with a permanent collector regression: a truncation inside a batch, another
at its edge, distinct terminal/reset values, and a later batch containing a legacy time limit, true death and
open continuation. Before fixing, targets `[1.0, 3.6, 3.0]` should have been approximately
`[2.75, 6.145584, 9.363961]`. Env flags alone had not delivered the promised bootstrap.

Dev now marks only `no-room-progress` with `info.bootstrap_truncation`; the collector adds the frame-discounted
actual final-observation value before resetting. The done flag still cuts the GAE trace; it never substitutes
the next episode's reset value. Legacy max-step/death behavior and raw logged reward are unchanged. No cutoff
is enabled in canonical configs. Main remains frozen on r32's runtime; this correction is not merged mid-cohort.

Real Zelda smoke: three WAIT 8 actions in `1_73s_0`, actual nine frames each, cutoff at action three. With a known
constant critic value 10, gamma 0.9 and lambda 1, learning targets are `[7.007587, 7.889431, 8.882247]`, matching
the frame-aware calculation; raw logged reward remains zero. The same three actions under the legacy max-step
cap still produce `[0,0,0]`. Both are playable mode 5 after automatic reset. Evidence:
`models/r33-cutoff-evidence/bootstrap-smoke.json`. This proves collector integration, not a trained model gain.

Current r31-block1 standalone successful-gap exposure differs from the older baseline: 1:0x23 has 41 successes,
worst 79 quiet actions; 1:0x44 has 43 successes, worst 901 (one would be cut at either 100 or 200); 2:0x1e has
49 successes, worst 154 (two at 100, none at 200); 2:0x5e has 51 successes, worst 60. These retrospective counts
do not establish a safe production threshold, and the saved full-main chain also has a 1,207-action successful
gap. Re-measure whichever model is used for each trial, rather than treating historical thresholds as permanent.

Integrated dev suite: **463 passed in 779.75 seconds** while r32 occupied the main training workers. The new
value-target regression passes, including legacy truncation/death and open-rollout controls. Temporary native
smoke script removed; persisted JSON evidence retained. No r32 runtime source or canonical config changed.

## 11. Per-arm chain configuration (dev prerequisite)

Added repeatable `triforce eval-chain --specialist-config PATH.yaml`, keyed by each loaded config's family name.
The loaded objects reach `Runner` in both direct and spawned execution; `_Specialist` no longer reloads a
default config over an override. Programmatic callers use `specialist_configs={name: loaded_config}`.
Canonical YAML, promoted pointers, route selection and carried health/inventory remain untouched. This supports
the cutoff and later boomerang comparisons without mutating a live experiment's global config.

Native smoke uses a valid temporary boomerang-biased neural policy in D2 0x5e. The first attempt correctly found
boomerang masked in the south doorway; after two scripted northward moves, enabled config chooses BOOMERANG,
disabled config chooses WAIT instead. The unchanged cutoff is honored too. Ordinary waiting runs with cutoff 2
produce exactly matching native room/health/frame/outcome records through direct API, CLI parallel 1 and CLI
parallel 2. Evidence: `models/r33-cutoff-evidence/config-override-smoke.json`. These are smoke policies, not
trained candidates. The spawned real-game CLI regression passed (two runs, each stopped at two quiet actions).
Integrated dev suite after config transport: **464 passed in 871.42 seconds**. Temporary neural checkpoints,
YAML files and smoke script were removed; JSON evidence remains. This feature stays in dev until r32's source
freeze ends; no canonical specialist config or promoted pointer was changed by these checks.

## 12. D2 key-builder defect reproduced (before route migration)

Native `cross_room` from clean `2_7ew_0` toward 0x6e, seed 0, 3,000 exploration iterations/200 patience/one
attempt: accepted a 56-action path after 6,293 exploratory actions (68.6 seconds), ending `exit-wrong` at 1.5
hearts. Link entered 0x6e with **zero keys and 0x7e's key still untaken**, playable mode 5. Source also had zero
keys and an untaken key. This confirms the §8 acceptance hole through real gameplay; no bad state file was kept.
Evidence: `models/r34-d2-key-evidence/builder-before.json`.

Separate native prototype, same source/seed/search limits, with required source-room item-taken gate and bombs
masked: accepted a 155-action **exit-correct** path after 14,886 exploratory actions (202.3 seconds), arriving at
three hearts with **one key and 0x7e's pickup flag set**, playable mode 5. Evidence:
`models/r34-d2-key-evidence/builder-prototype-after.json`. This validates the guard through real collection,
not an injected inventory or metadata-only check.

Permanent dev builder now enforces the declared source-room key pickup during discovery **and exact replay**;
its milestone step config masks bombs. Actual `cross_room` smoke succeeds for all three `2_7ew_0..2` sources:
155/103/149-action paths, each entering 0x6e with one key and the pickup flag set. Reloading each captured
pre-entry byte payload through `GameSession.load` preserves that inventory, four max hearts and playable mode 5.
Evidence: `models/r34-d2-key-evidence/builder-after.json`. Affected entry/exploration/room-entry suites:
**158 passed in 33.23 seconds**. The previously integrated full suite was 464; no new full-suite result is claimed
for this subsequent guard. Slow natural exploration remains a throwaway smoke rather than a permanent search
test. No permanent state was replaced, and no r32 runtime source was changed.

Native inventory audit of all 29 existing numbered D2 files (through `GameSession.load`, mode 5): every file
has four hearts, wooden sword/boomerang and exactly the D1 triforce piece. All 0x7d/0x7e entries have zero keys
and an untaken 0x7e key. Chains 0/1 also reach 0x3e, 0x2e, 0x1e and Dodongo entry with **zero keys and both
0x4e/0x3e pickup flags still unset**. Chain 2 instead has one key at 0x3e, two at 0x2e and one at 0x1e, with
both earlier pickups set. Its 0x2e E-door-open bit is set at the later entry; the route's N door is not locked,
while the side E door is. Thus room pickup flags alone do not prove the carried-key count the rebuilt route needs;
the suffix rebuild must check inventory balance and reject unneeded side-door spending. No 2_0es_2 file exists.
Full inventory, current locked doors and persistent room flags: `models/r34-d2-key-evidence/inventory-before.json`.
No state was mutated by this audit.

