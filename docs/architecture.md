# Triforce v2 architecture and module contracts

Design: `triforce-v2-design.md` (kept outside the repo; decisions that change it go in `docs/decisions.md`).
This file lists the module boundaries and the interfaces between them.

```mermaid
flowchart LR
    EMU[emulator.Emulator<br/>retro core, fast RAM] --> EXE[frame_skip.OptionExecutor<br/>timed options, interrupts, skip]
    EXE --> SES[session.GameSession<br/>ZeldaGame, StateChange, ActionMask]
    SES --> ENV[env.ZeldaEnv<br/>gym API for one specialist]
    SPEC[specialist.SpecialistConfig<br/>specialists/*.yaml] --> ENV
    ENV --> OBS[observation.ObservationBuilder]
    ENV --> CRIT[critics + end_conditions<br/>rewards, terminal classes]
    ENV --> PPO[ppo / model / train]
    SES --> PLAY[play CLI]
    SES --> RUN[runner + registry]
```

## One emulator per process

stable-retro allows a single emulator per process. `GameSession` owns it. Anything that needs the game
(env, runner, play tool, tests) either creates the one `GameSession` or is handed it. Tests share one
session through the `session` fixture in `tests/conftest.py`. Training uses subprocess workers, one env each.

## Core (done)

| Module | Contract |
|---|---|
| `emulator.Emulator` | `step(buttons uint8[9] or None, frames)`, `frame` counter, `ram` (all blocks, cached per frame), `read(addr)`, `info` (data.json variables), `set_value(name, v)`, `save_state()/load_state(bytes)`, `get_screen()`. |
| `actions` | `ACTION_SCHEMA_VERSION`; `ActionKind` (zelda_enums) = MOVE, WAIT, SWORD, BEAMS, BOMB, BOOMERANG, ARROW; `DIRECTIONS` (N,S,W,E,NW,NE,SW,SE); `DURATIONS` (4,8,16,32). `Action(kind, direction, duration)`, `Action.from_indices(k, d, u)`, `.to_indices()`, `Action.parse('MOVE N 16')`. `ActionMask` with `kind[K]`, `direction[K,8]`, `duration[K,4]`, `to_flat()` → bool[`MASK_SIZE`] = K + K·8 + K·4. A kind whose direction (or duration) row is all False doesn't use that head. `compute_action_mask(state)`. |
| `frame_skip.OptionExecutor` | `execute(action, state) -> OptionResult(frames, interrupt, control_frames, projectile_blocks)`; `.blocks` counts confirmed shield collisions in executed frames, including attack/settling/uncontrollable frames. |
| `session.GameSession` | `load(name|path|bytes, randomize_rng=True, ram_overrides=None, fresh_room=False, *, controllable=False) -> ZeldaGame`; controllable curriculum frames are not advanced before the first action; normal entries gain control through their transition. `act(action) -> StateChange`; `.state`, `.mask`, `.total_frames`, `.steps`; `snapshot()/restore()`. |
| `state_change.StateChange` | `previous`, `state`, `action`, `frames`, `interrupt`, `health_lost/gained`, `enemies_hit`, `enemies_stunned`, `items_gained`, `changed_location`, `gained_triforce`, `action_mask`, executed `projectile_blocks` / `.blocks`. Look-ahead credit via `FutureCreditLedger` does not add simulated blocks. |
| `describe`, `play` | Text dumps and the interactive `python -m triforce.play` tool. |

## Observation (`observation.py`)

```python
OBS_SCHEMA_VERSION: int
ROOM_VOCAB, ENTITY_VOCAB, NUM_TILE_CLASSES: int     # embedding vocab sizes
NUM_ENTITY_SLOTS = 12
class ObservationBuilder:
    def __init__(self, objective: bool): ...
    def reset(self, state: ZeldaGame, mask: ActionMask) -> dict[str, np.ndarray]
    def build(self, change: StateChange, mask: ActionMask) -> dict[str, np.ndarray]
    @staticmethod
    def observation_space() -> gymnasium.spaces.Dict
```

| Key | dtype / shape | Notes |
|---|---|---|
| `room_id` | int64 [1] | index < `ROOM_VOCAB` |
| `room_state` | float32 [R] | per-room flags from `docs/ram-catalog.md` |
| `link` | float32 [L] | |
| `entity_types` | int64 [12] | 0 = empty slot |
| `entity_features` | float32 [12, F] | feature 0 = presence |
| `grid` | int64 [22, 32] | tile class per 8px tile, row-major |
| `history` | float32 [H] | last action kind/dir/duration, frames since |
| `objective` | float32 [O] | all zeros when `objective` is off |
| `action_mask` | bool [`MASK_SIZE`] | `ActionMask.to_flat()` |

When the objective block is on, the env attaches `state.objectives` (v1's `Objective`) before calling the builder.

Observation schema 2 adds the shared chase target (`$61/$62`, both full-byte `/255`) and
`chase_refresh_suppressed` (`$60 != 0`). The flag controls coordinate refresh, not a reliable Link/shadow
identity; bait can replace the target independently. It also adds Link boomerang X/Y and five phase flags
(outbound, spark, slow-outward, slow-return, fast-return); Goriya boomerang entities receive the same phases.
Animation low bits do not change phase. Inactive/bait/unsupported phases have no boomerang coordinates/flags.
`hurts_on_touch` is a contact hazard for an otherwise vulnerable Link, not a shield/proximity or next-frame
damage prediction: stunned attackers and non-outbound projectiles are harmless; delayed arrows are not ready,
but delayed fireballs remain hazardous. Returning traps keep their separate native contact semantics.
Presence/active/targetability remain separate. Timer scales and simulation/rewards are unchanged; schema-1
checkpoints are rejected rather than relabeled or loaded through a compatibility path.

Dungeon objectives fight for pending treasure or when every routed exit is closed by a living-monster shutter
trigger (all-dead, ringleader, last-boss or clear-to-drop). An unrelated shutter or an open route exit does not
force combat. The shutter rule excludes block/money triggers and non-foe-only rooms. Live barred-door
classification covers vertical tiles 0xa8–0xab and horizontal tiles 0xac–0xaf. Routing resumes after clearing.

## Specialist task and environment (`specialist.py`, `env.py`)

`specialists/<name>.yaml` follows design §3.3; load with `specialist.load_specialist(name_or_path) -> SpecialistConfig`.
Training/evaluation CLIs preserve that loaded config through curriculum/model setup and spawned env workers.
`config.name` labels checkpoints and logs; it must not be used to reload the default YAML and discard overrides.

```python
class ZeldaEnv(gymnasium.Env):
    def __init__(self, specialist: str | SpecialistConfig, session: GameSession | None = None,
                 start_state: str | None = None): ...
    observation_space = ObservationBuilder.observation_space()
    action_space = MultiDiscrete([NUM_KINDS, NUM_DIRECTIONS, NUM_DURATIONS])
    def reset(seed=None, options=None) -> (obs, info)
    def step(action: array[3]) -> (obs, reward: float, terminated, truncated, info)
```

`info` keys every step: `frames` (int, frames this step: used for frame-aware discounting), `interrupt`,
`action` (str), `rewards` (dict name → value), `room` ((level, room, in_cave)), `health`, `max_health`,
`steps`, `total_frames`, `specialist`. On the last step: `ending` (end-condition name) and `outcome`
(the episode's entry room), plus `blocks` (confirmed shield collisions over the episode). An end condition that is
not classified raises. Evaluation reports mean `blocks` overall/per room/per state; training separates `blocks`
and `curriculum/blocks`; runner room records retain the visit's count. Historical absence is not a measured zero.

### Heart drops (`hearts.py`)

`HeartTracker` follows the hearts enemies drop (item 0x22 in monster slots 1–$B; fairies, triforce pieces and heart
containers don't count) through one room visit: `reset(state)`, `update(change)` every step, `finish(state)`,
`summary()` → `dropped`, `taken`, `healed` (real hearts gained), `wasted` (`taken − healed`), `left` (on the floor
when Link left the room, died or the visit ended) and `expired` (timed out). A heart that vanishes while its
lifetime timer (`Item.timer`, counts down once per 2 frames) is above what the step's frames could count down was
taken (by Link, his sword, arrow or boomerang); one near the end of its lifetime counts as taken only if Link's
health rose. A heart heals `min(1, max_health − health)`, so one picked up at full health wastes a whole heart.
"Total" heart change = real change (`health − start_health`) + `wasted`. `ZeldaEnv` puts the summary in the
terminal `info['hearts']`; metrics log `health_change_total` and `hearts/{dropped,taken,left,wasted}` per
episode; `evaluate` keeps it per episode and reports `mean_health_change_total` and `hearts_dropped/taken/left`
overall, per room and per start state; runner `RoomRecord`s carry `hearts_dropped/taken/healed/wasted/left/expired`.

### Lingering (`linger.py`)

`LingerTracker.reset(state)` / `update(change, hearts_taken=0)` measures progress without changing gameplay.
Progress is a hit/kill, pickup, healing (including pond refills), a new best walking distance to the objective,
or leaving a room. Damage alone is not progress. A frozen wavefront prevents enemy movement and Link's round
trips from manufacturing new best approaches; hits/kills/pickups and objective phase changes rebaseline it.
Phase changes alone do not reset the quiet clock. A progress action's frames do not count toward the quiet
interval it closes. `summary()` returns `max_steps/max_frames`, `tail_steps/tail_frames`, elapsed `steps/frames`,
and nullable boss-kill / heart-container-pickup action/frame timestamps.

Terminal env `info['linger']` and `RoomRecord.linger` carry the summary. Training `linger/{max,tail}_{steps,frames}`
are means over completed room-entry episodes; `linger/worst_{steps,frames}` are maxima across the iteration's
workers. `curriculum/linger/*` is separate; `linger/room/<room>/*` groups by episode entry room. These metrics reach
progress events and the GUI. Eval records retain raw summaries and report `linger_{max,tail}_{steps,frames}` means
and `linger_worst_{steps,frames}` maxima overall, per room and per start state. Older absent measurements remain
absent, not zero. Multi-room env episodes reset their tail on each exit and retain the episode's worst gap;
runner records always describe one visit.

### Experimental reward and cutoff controls

`reward.block_reward` (nonnegative, default 1) multiplies `REWARD_MEDIUM` for each confirmed shield collision,
advancing combat decay separately at the collision's room. The detector checks consecutive executed RAM frames:
projectile transition signature, shield eligibility/facing/proximity and this frame's `$604` parry request.
Delayed `$605` audio, Darknut/immunity clinks, natural boomerang returns and unrelated disappearance are not enough.
The executor does not replace the user's frame callback, change inputs/timing, or collect speculative look-ahead.

`reward.quiet_enemy_penalty` (nonnegative, default 0) adds that cost per action after the shared quiet tail exceeds
the existing room-stall grace, only while living targetable enemies remain. `critique_gameplay` requires the
caller's measured `quiet_steps`; the env passes its already-updated tracker, without a second progress counter.
Hits/pickups/real approach reset the shared tail; damage and round trips do not. No cost for post-fight pickup time
or inactive/burrowed/dying enemies. This is separate from the existing room-stall ramp and remains default off.

`reward.nearby_items` (default false) adds heart/clock overlap tiles to the objective's PBRS targets alongside
exits. Multi-source walking distance picks the nearer target; the task, accepted exits and observation schema do
not change. This is not the old heart-only objective.

`no_progress_steps` (positive or null, default null) replaces legacy positional/room timeout with the shared
lingering tail. Training ends `no-room-progress` as a soft-failure **truncation** with no terminal penalty;
success/death keep precedence and the absolute episode cap remains enforced. Its terminal info marks
`bootstrap_truncation=True`: `RolloutBuffer.collect` adds `gamma**(frames/8) * V(final_observation)` to the
learning reward before reset, while done flags still stop the GAE trace. Logged episode rewards stay unchanged.
Legacy max-step truncations retain their existing nonbootstrapping behavior. Runner uses the per-visit tail for
`stuck` instead of the flat room budget, retaining the global run cap; it resets on every room exit.
Nearby-item shaping and the measured cutoff remain disabled in promoted configs until their experiments.

## Learning (`model.py`, `ppo.py`, `train.py`, `registry.py`)

- Action heads: kind (K), direction (8), duration (4). Direction/duration log-probs and entropies count only
  when the sampled kind uses that head (mask row non-empty).
- GAE discount per step: `gamma ** (frames / 8)`, `gamma = 0.99`; `frames` stored in the rollout buffer.
- Checkpoint (`models/<specialist>/<run-id>/ckpt.pt`): `obs_schema_version`, `action_schema_version`,
  `specialist`, `model_config`, `state_dict`, `optimizer`, `iterations`, `steps`, `metrics`.
  Loading with a mismatched schema version raises.
- `models/<specialist>/promoted` holds the promoted run id; `triforce promote <specialist> <run-id>` sets it.
- Metrics (design §12) go to TensorBoard and `metrics.jsonl` in the run directory; `train_summary.json`
  records iterations, steps, `stopped_early`, wall time and steps/sec.
- `train --seed S` seeds model initialization and PPO RNGs, then each spawned worker's Python/NumPy/Torch
  streams and first environment reset with `(S + worker_index) mod 2**64` (Torch normalizes signed seeds first).
  In-process training uses S. Later episode resets and rollout batches advance those streams without reseeding.
  Shared curriculum progress still depends on worker completion order; this is not a promise of bitwise-identical
  curriculum or CUDA training, nor exact RNG continuation when resuming a checkpoint.
- Live event logs (`events.py`, watched by `training-gui.py`): each training invocation gets
  `logs/<yymmdd>_<NNN>_<what>/` (gitignored; `--logs-dir` / `$TRIFORCE_LOGS_DIR`). `training.json` lists every
  specialist run in order with its status and is rewritten atomically. `<specialist>.jsonl` is an append-only
  stream (`start`, one `progress` per PPO iteration with steps/target/fraction/sps/eta and headline metrics,
  `checkpoint`, `early_stop`, `error`, `train_end`; the round driver adds `eval_start`, `eval_progress` (≤1/s:
  done/episodes/running success_rate), `eval`) whose last line
  is `end`. `<specialist>.log` is its console output. `round.jsonl` / `round.log` cover the round (chain,
  errors, `end`). SIGTERM is turned into an interrupt, so a killed run still saves and writes `end`.
- `train --stop-success R [--stop-patience 5]` (`EarlyStop`) ends training once the start-state-balanced
  success rate (mean over start states of each state's last 20 room-entry episodes; logged as
  `balanced_success_rate`) is ≥ R for that many consecutive iterations. The raw `success_rate` is biased toward
  the states the start-state pool oversamples (failures), so it is not used for stopping.
- `python -m triforce.explore --specialist NAME` (Go-Explore phase A, `explore.py`): per seed start state, an
  archive of cells (room, tile // 2, hearts, living enemies, keys, boomerang) with exact `GameSession`
  snapshots; random legal actions from cells chosen by progress/closeness/novelty; the best success is replayed
  and written as `trajectories/<specialist>/<state>/NNNN.state` + `trajectory.json`. `ZeldaEnv.restore(snapshot)`
  starts an episode from an exact snapshot.
- Backward curriculum (`curriculum.py`): on when the specialist's config has `curriculum: {enabled: on}`
  (trajectories default to `trajectories/<specialist>`; `train --curriculum DIR` overrides, `--no-curriculum`
  disables). Half of the resets start
  on a trajectory near its end; each trajectory's start point moves `ceil(len/10)` steps earlier once 6 of the
  last 10 episodes from the current stage succeed, until it reaches the room entry. Progress is shared by all
  workers through `<run dir>/curriculum.json`. Curriculum episodes are reported as `curriculum/*` and excluded
  from the room-entry metrics.
  Saved curriculum frames are already controllable: resets use `GameSession.load(..., controllable=True)`
  without idle input or a second cave-entry cooldown. RNG/HP randomization still applies. Ordinary entry-state
  resets keep transition/fresh-room handling unchanged.
- Chain-matched health (`chain_hp.py`): `python -m triforce.chain_hp <chain jsonl ...>` writes
  `triforce/chain_hp.json`, the hearts Link entered each room with in chain runs. A specialist with
  `ram_randomization: {chain_hp: p}` draws its reset hp from its start room's list in a share `p` of room-entry
  resets (not curriculum starts); the rest keep the `hp` / `full_hp` draw.
- `evaluate.evaluate_specialist(specialist, checkpoint, episodes_per_state=20, parallel=1, seed=0,
  deterministic=False)` plays every seed start state N times in `ZeldaEnv` (episode i seeded `seed + i`) and
  returns success rate, outcomes/endings and per-room/per-state breakdowns; `python -m triforce.evaluate
  --specialist NAME` (or `triforce evaluate`) writes `eval.json` next to the checkpoint.
- `scripts/train_round.py --round ID` trains every trainable specialist (route order), promotes and
  evaluates each, then runs `eval-chain`; summaries go to `models/<round>-summary.json` and
  `models/<round>-chain-summary.json`.

## Runner and CLI (`runner.py`, `run.py`, `cli.py`)

- Routes (`triforce/routes.yaml`, `game_map.load_routes()`): `milestone1` and `milestone1-fairy` (milestone1 with
  the post-D1 overworld replaced by the south road and the fairy pond; a spliced route is
  `{base, replace: {from, to}, steps}`). `GameCompletion(route, step)` routes on one of them: only the route's
  exits are correct on its rooms, and `waypoint:` steps are targets while they apply. Training episodes take the
  route from the start state's name (`objectives.start_route`, `start_states.state_entry`); see
  `docs/decisions.md` (second route) for the rule.
- `Route.load(name, stop_at=None)` reads `triforce/routes.yaml`. A route starts at the `start` savestate and
  ends on entering its `terminal` room (both routes: level 2 room 0x0e); `stop_at` ends it earlier.
- `Runner(session=None, models_dir=None, checkpoints=None, deterministic=False, max_steps=50000,
  renderer=None, device='cpu', specialist_configs=None).run(route, seed) -> RunResult`. On each room entry:
  `game.yaml` tag → specialist → promoted
  checkpoint (cached). Untagged room or missing model → `result='error'` with the reason. Per-room outcomes:
  `exit-correct`, `exit-wrong`, `death`, `stuck` (specialist `max_steps` in one room), `milestone`.
  Finalizing a room captures live ending health together with dropped-heart counts, including room/run budgets
  and measured cutoffs; death remains zero. `health_out` is never left at entry health on a timed-out visit.
- `python -m triforce.run --route milestone1 [--seed N] [--render] [--stop-at L:0xRR] [--checkpoint S=PATH]`
  prints one line per room and appends a JSON line to `runs/runner.jsonl`.
- `triforce eval-chain --route milestone1 --runs 20 [--parallel N] [--from STATE] [--stop-at L:0xRR]` → success
  rate and per-room failure counts (`chain_summary`); `--from` starts mid-route at a route entry state (e.g.
  `0_37c_2`, after Dungeon 1). `triforce promote <specialist> <run-id>`; `triforce list`;
  `triforce train|run|play|evaluate ...` forward to the module CLIs.
- Chain config overrides: repeat `eval-chain --specialist-config /path/to/arm.yaml`. Each YAML's `name` selects
  its family. `Runner` / `eval_chain(..., specialist_configs={name: loaded_config})` accept the same loaded
  objects; spawned workers retain them instead of reloading canonical YAML. Runtime item masks, observations
  and room/cutoff budgets use these configs. Route selection and carried inventory/health remain chain-owned.
  Other families retain canonical configs; checkpoint overrides and promoted pointers are independent.

## Entry states (`entry_states.py`)

`build` follows route steps; `path` accepts explicit off-route destinations. Both use `cross_room`: arriving
in the target is insufficient when the source room declares a key. The source room's persistent item-taken
flag must be set in the same level. Carried keys must retain the source inventory plus its newly required key,
minus the cost of the direct target door if initially locked; opening an unrelated side door is not an allowed
cost. Discovery and exact replay both enforce these requirements before returning captured pre-entry bytes.
This preserves intended off-route traversal without accepting premature key-room exits or depleted inventory.
`step_config` masks bombs for milestone-1 state generation. Route changes still require rebuilding and auditing
downstream inventories and refreshing affected curriculum snapshots; old filenames alone do not prove progress.

