# Round 35

## A learned guard: can a model learn the planner's veto? (goriya, Dungeon 7; an experiment, branch `r35-guard`)

Round 34 found one cause in every wood-sword dungeon: the honest planner (`temperature:0.5+relax:16+plan:16:1`)
stands, swings and bombs like the models and is hit a third to a half as often, by replacing a handful of decisions
a room at the moments the emulator's look-ahead shows a hit; six goriya runs trained on its rows did not pick it up
(round34.md §3, §8). The question here: can a network that sees only the policy's observation learn the look-ahead's
verdict ("this action gets Link hit"), and does vetoing with it, the way `plan:16:1` vetoes with the emulator, play
like the planner? Such a guard uses no look-ahead at play: it would be a model's own act mode. Family: goriya
(`r32d1-goriya`, `d3first`'s route run) on Dungeon 7's rooms, the clearest case (crossings, boomerangs).

### 1. The design

**Where the planner decides.** `plan:K:S` (`crates/pool/src/pool.rs` `PlayWork::foresee`, docs/rust-actor.md
"Foresight"): before a decision is taken the env tries the model's draw in the emulator on S redrawn futures
(`PoolEnv::trial`: the action executed in simulation mode and the session restored); if Link loses health in any,
the next action of the mode's own ranking (ln q + the decision's Gumbel noise) is tried, at most K in all, and the
first that loses none is taken. One trial costs the action's frames again: `plan:16:1` is about 2x a run's frames.
The rows `r27_plan_demos.py` recorded (round 34) are the observation, the mask and the action taken, with the
replaced decisions repeated; the refused draw and the verdicts of the other candidates were not kept.

**Where a guard can live.** Two ways were weighed:
- Python-side (a Python loop that steps the env and asks a torch guard): no such play path exists for chains (every
  evaluation and chain plays in the Rust pool, `EnvPool.play`, the policy in Rust), so it would mean a second chain
  driver: more machinery, and slower.
- In the Rust actor (taken): the guard is **the policy's own network** (`ZeldaPolicy`: the same observation, trunk
  and towers) with the value head widened to **one hit logit per joint action** (352). The Rust `Policy` already
  loads every weight of that architecture; the only change is a value head of any width (`Policy::values`). The act
  mode **`veto:K:P`** (`crates/policy/src/act.rs`) is a third kind of foresight beside `guard:K` and `plan:K:S`: no
  trial in the env; the guard scores the decision once (one forward, all 352 actions) and refuses a candidate whose
  hit probability is above P; candidates and ranking as `plan`. The guard file is **`guard.pt` beside the run's
  `ckpt.pt`** (`triforce/guard.py`; `runner.Policies` attaches it, the chain's `models` header records its hash). A
  run without one plays `veto` as the plain mode, and no other mode reads a guard: a new file beside the checkpoint,
  not a change of the checkpoint or of any model's play.

**The data's new parts** (`EnvPool.poll_plays` `tried`, `TRIFORCE_PLAN_AUDIT`): every candidate a decision's
foresight judged, with the future it was tried on and whether Link lost health (`pool::rollout::Tried`); and audit
futures, each judged candidate also tried on A more redrawn futures that never decide, for the ceiling.
`scripts/experiments/r35_guard_data.py` writes them; `scripts/experiments/r35_guard_train.py` trains and measures.
