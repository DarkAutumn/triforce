# Round 21: clips in the agent, the overworld wrap gauntlet

The owner wants the agent to perform the speedrunners' grid-point skip; branch `clips` (milestone 1, commit
`671fa8a`, version 1.3.3.0) offers the overworld screen wrap to the policy in four ways, per specialist
(`clips: {action: none|kind|long|edge, shaping: jump|walk}`; `docs/decisions.md`, `zelda-asm/mechanics.md`
"Screen wrap (overworld)"). This round trains all seven arms from scratch on the same three-room gauntlet and
reports what happened. No design is chosen here.

## Summary

Every arm that can wrap learned it and uses it for nearly every crossing: 98.6-99.6% success by sample (99.1-99.8%
by argmax) against 96.2-96.4% (96.9-97.2%) for `none`, crossings in 3-4 steps (about 143 frames, the scripted
ceiling is 141-142) instead of 15-17 steps (about 335 frames), and almost no hearts lost on a crossing (−0.00
against −0.17 to −0.21). The exceptions are both `long` arms of seed 1: they learned the wrap for E→W crossings only
and walk W→E (97.1% and 97.5% by sample). Under `long` a MOVE of 4-16 frames at the entry bound leaves the room by
the wrong exit (−20), so exploring MOVE W at the W entry is punished at three durations out of four. Clip use in
training starts high under `kind` and `edge` (random CLIPs and ping-pong
wraps, peaking at 1.7-18 per episode) and settles at about 0.6 per episode; under `long` it starts at 0.2 and rises
slowly (seed 0) or never (seed 1). Policies under `kind` and `edge` (both shapings) step one short MOVE into the room
before wrapping back out of the entry edge, instead of wrapping at once. Two reward oddities explain or bias this
(Observations 1-2): the first step that moves Link in an episode pays no PBRS, and `CLIP` is charged as a missed
attack.

## Setup

- Worktree at `671fa8a` (branch `clips-run`), its own venv and `triforce._native` 1.3.3.0; RTX 4090; the machine
  shared with two other training rounds (load 19-37 on 20 threads).
- **Rooms and tasks.** 0x59 (4 Peahats, a Zora), 0x5a (4 red Octoroks, a Zora), 0x5b (2 blue and 2 red Moblins
  walking in from the edges). Start states `experiments/r21/states/0_{59,5a,5b}{w,e}_{0,1,2}` (18). Entered from W
  the target exit is E or S, from E it is W or S, drawn per reset: 12 room/entry/target tasks. 0x5b's E exit counts
  only through its upper opening (game.yaml `span: [0, 9]`).
- **Targets: the route `clips-gauntlet`** (`triforce/routes.yaml`, 12 steps, two per (room, entry)). With
  `route: clips-gauntlet` in the specialist YAML, `zelda::objectives::start_route` draws one of the two matching steps
  at every reset; no code change was needed. Checked with `_native.Env`: 200 resets per state gave 89-111 per
  target for each of the 18 states; 3600 free resets drew each state exactly 200 times, both targets 89-111; the
  route's step reaches the evaluation records (`target`, `entry`) and `by_state_target`. The production families are
  unaffected: 200 resets each of ow-zora (0_59w_0, 0_5aw_1) and ow-moblin (0_5bw_2) all drew d2first's E (the new
  route's steps don't match their progress, the wood sword).
- **Where the wrap is available at the start** (`Game.skip_available` on the loaded state; the skip zone needs
  Link on a row open at both edges): W entries 0_59w_1, 0_5aw_0..2, 0_5bw_0, 0_5bw_1; E entries 0_59e_0..2,
  0_5ae_1, 0_5ae_2, 0_5be_0..2. Not on 0_59w_0, 0_59w_2, 0_5ae_0, 0_5bw_2 (Link has to change rows first).
  0_5bw_0 stands on row Y $B5, whose W wrap lands in 0x5b's lower E opening: the skip is offered there, but the exit
  after it is wrong.
- **Specialists** `experiments/r21/gauntlet-{none,kind-jump,kind-walk,long-jump,long-walk,edge-jump,edge-walk}.yaml`,
  identical but for `clips:`; modelled on `specialists/ow-moblin.yaml`: `success: [exit-correct]`, `reward: {pbrs:
  true}`, `obs: {objective: true, target: true}`, `ram_randomization: {hp: [0.15, max], full_hp: 0.5, bombs: [0, 8],
  bombs_share: 0.5, help_count: [0, 9]}`, `items: {bombs: tenth}`, `curriculum: {enabled: false}`, `one_room: true`,
  `max_steps: 600`, `route: clips-gauntlet`. States dir by `TRIFORCE_STATES_DIR=experiments/r21/states`.
- **Training**: from scratch, 256 iterations × 4096 = 1,048,576 env steps per run, no early stop, `--parallel 4`,
  `--sil-coeff 0.1` (round 19's setting), default learning rate (1e-4) and entropy, seeds 0 and 1 (`--seed`: torch
  and the env pool), at most 3 runs at a time. A 25-iteration throughput run (kind-jump) did 468 steps/s alone;
  under the shared load the runs did 276-512 steps/s, 34-63 minutes each. The first pass (7 arms, seed 0) took 2 h 42
  min, so the second seed was run for all seven arms.
- **Evaluation**: `triforce evaluate`, 100 episodes per state (1800 per run, 911 crossings and 889 south tasks; the
  targets are drawn the same way for every run), seed 0, once with `--act sample` and once with `--act argmax`.
  Replays of 2 episodes per state (`--act sample`, the decisions stream; 17 crossings and 19 south tasks per run,
  the same draws for every run) were traced frame by frame for the behaviour table.
- **Commands** (all in `scripts/experiments/r21_gauntlet.sh`):

```bash
scripts/experiments/r21_gauntlet.sh pass 0          # 7 arms, JOBS=3 at a time: train, evaluate, replays
scripts/experiments/r21_gauntlet.sh arm kind-jump 1 # one arm and seed
scripts/experiments/r21_gauntlet.sh reference       # the scripted wrap
# what `arm ARM SEED` runs, with TRIFORCE_STATES_DIR=experiments/r21/states:
python -m triforce.train --specialist experiments/r21/gauntlet-ARM.yaml --run-id r21-sSEED --iterations 256 \
    --steps-per-update 4096 --parallel 4 --device cuda --models-dir models --sil-coeff 0.1 --seed SEED \
    --events logs/r21-gauntlet/ARM-sSEED.jsonl
python -m triforce.evaluate --specialist experiments/r21/gauntlet-ARM.yaml --run-id r21-sSEED --models-dir models \
    --episodes-per-state 100 --parallel 4 --act sample|argmax --output models/gauntlet-ARM/r21-sSEED/eval-ACT.json
python -m triforce.evaluate ... --episodes-per-state 2 --act sample --replays target/r21-gauntlet/replays/ARM-sSEED \
    --streams decisions --output models/gauntlet-ARM/r21-sSEED/eval-replays.json
python scripts/experiments/r21_report.py 0 1        # the tables below
python scripts/experiments/r21_behaviour.py target/r21-gauntlet/replays/ARM-sSEED models/gauntlet-ARM/r21-sSEED/eval-replays.json
```

## Scripted reference

`scripts/experiments/r21_wrap_reference.py`: on every crossing start state whose entry row has the wrap, CLIP toward
the entry edge, then MOVE out of the opposite edge; 100 episodes per state (game RNG seeded per episode), hp drawn as
the YAMLs draw it. Frames include the room-change scroll (the wrap itself is 35-36 frames, the exit MOVE 1 frame plus
105 frames of scroll), as in the evaluation records.

| state | episodes | success | Δhearts | steps | frames | endings |
|---|---|---|---|---|---|---|
| 0_59w_1 | 100 | 100% | +0.00 | 2 | 142 | exit-correct 100 |
| 0_59e_0 | 100 | 100% | +0.00 | 2 | 141 | exit-correct 100 |
| 0_59e_1 | 100 | 100% | +0.00 | 2 | 141 | exit-correct 100 |
| 0_59e_2 | 100 | 100% | +0.00 | 2 | 141 | exit-correct 100 |
| 0_5aw_0 | 100 | 100% | +0.00 | 2 | 142 | exit-correct 100 |
| 0_5aw_1 | 100 | 100% | +0.00 | 2 | 142 | exit-correct 100 |
| 0_5aw_2 | 100 | 100% | +0.00 | 2 | 142 | exit-correct 100 |
| 0_5ae_1 | 100 | 100% | +0.00 | 2 | 141 | exit-correct 100 |
| 0_5ae_2 | 100 | 100% | +0.00 | 2 | 141 | exit-correct 100 |
| 0_5bw_0 | 100 | 0% | +0.00 | 2 | 142 | exit-wrong 100 |
| 0_5bw_1 | 100 | 100% | +0.00 | 2 | 142 | exit-correct 100 |
| 0_5be_0 | 100 | 100% | +0.00 | 2 | 141 | exit-correct 100 |
| 0_5be_1 | 100 | 100% | +0.00 | 2 | 141 | exit-correct 100 |
| 0_5be_2 | 100 | 100% | +0.00 | 2 | 141 | exit-correct 100 |

The ceiling on these 13 states (0_5bw_0 aside) is 100% in 2 steps and 141-142 frames with no damage. The trained
policies on the same 13 states and crossing targets:

| arm | seed | act | crossings | success | Δhearts | steps (successes) | frames (successes) |
|---|---|---|---|---|---|---|---|
| none | 0 | sample | 650 | 96.6% | -0.169 | 15.5 | 335 |
| none | 1 | sample | 650 | 96.3% | -0.187 | 15.8 | 332 |
| kind-jump | 0 | sample | 650 | 100.0% | -0.002 | 3.0 | 143 |
| kind-jump | 1 | sample | 650 | 99.8% | +0.000 | 3.0 | 143 |
| kind-walk | 0 | sample | 650 | 100.0% | -0.002 | 2.9 | 143 |
| kind-walk | 1 | sample | 650 | 100.0% | -0.001 | 3.0 | 143 |
| long-jump | 0 | sample | 650 | 99.4% | +0.000 | 2.8 | 143 |
| long-jump | 1 | sample | 650 | 97.2% | -0.099 | 8.5 | 217 |
| long-walk | 0 | sample | 650 | 99.8% | +0.000 | 2.8 | 143 |
| long-walk | 1 | sample | 650 | 98.3% | -0.066 | 4.6 | 170 |
| edge-jump | 0 | sample | 650 | 100.0% | -0.001 | 3.3 | 145 |
| edge-jump | 1 | sample | 650 | 99.8% | +0.000 | 3.0 | 144 |
| edge-walk | 0 | sample | 650 | 100.0% | +0.000 | 3.0 | 143 |
| edge-walk | 1 | sample | 650 | 100.0% | -0.002 | 3.1 | 143 |
| none | 0 | argmax | 650 | 97.7% | -0.162 | 12.6 | 326 |
| none | 1 | argmax | 650 | 96.6% | -0.210 | 13.9 | 327 |
| kind-jump | 0 | argmax | 650 | 100.0% | +0.000 | 3.0 | 142 |
| kind-jump | 1 | argmax | 650 | 100.0% | +0.000 | 3.0 | 142 |
| kind-walk | 0 | argmax | 650 | 100.0% | +0.000 | 2.8 | 142 |
| kind-walk | 1 | argmax | 650 | 100.0% | +0.000 | 3.0 | 142 |
| long-jump | 0 | argmax | 650 | 100.0% | +0.000 | 2.7 | 142 |
| long-jump | 1 | argmax | 650 | 97.7% | -0.094 | 7.5 | 214 |
| long-walk | 0 | argmax | 650 | 100.0% | +0.000 | 2.7 | 142 |
| long-walk | 1 | argmax | 650 | 98.5% | -0.065 | 4.2 | 168 |
| edge-jump | 0 | argmax | 650 | 100.0% | +0.000 | 3.2 | 144 |
| edge-jump | 1 | argmax | 650 | 100.0% | +0.000 | 3.0 | 144 |
| edge-walk | 0 | argmax | 650 | 100.0% | +0.000 | 3.0 | 143 |
| edge-walk | 1 | argmax | 650 | 100.0% | +0.000 | 3.0 | 142 |

## Final evaluation

All 1800 episodes per run. Δhearts = mean health change per episode; clips = completed wraps; aborted = macros
ended by a hit.

| arm | seed | act | success | Δhearts | steps | frames | clips/ep | ep. with clip | aborted/ep | exit-correct | exit-wrong | death | max-steps | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 | sample | 96.2% | -0.14 | 14 | 288 | 0.00 | 0.0% | 0.000 | 96.2% | 0.8% | 3.1% | 0.0% | 0.0% |
| none | 1 | sample | 96.4% | -0.15 | 15 | 285 | 0.00 | 0.0% | 0.000 | 96.4% | 0.3% | 3.3% | 0.0% | 0.0% |
| kind-jump | 0 | sample | 99.4% | -0.02 | 7 | 185 | 0.65 | 64.6% | 0.003 | 99.4% | 0.1% | 0.6% | 0.0% | 0.0% |
| kind-jump | 1 | sample | 99.4% | -0.02 | 7 | 186 | 0.63 | 62.4% | 0.002 | 99.4% | 0.1% | 0.5% | 0.0% | 0.0% |
| kind-walk | 0 | sample | 99.4% | -0.03 | 6 | 186 | 0.64 | 63.7% | 0.001 | 99.4% | 0.1% | 0.4% | 0.0% | 0.0% |
| kind-walk | 1 | sample | 99.4% | -0.03 | 8 | 187 | 0.63 | 62.6% | 0.001 | 99.4% | 0.1% | 0.5% | 0.0% | 0.0% |
| long-jump | 0 | sample | 98.6% | -0.04 | 6 | 186 | 0.58 | 58.2% | 0.002 | 98.6% | 0.6% | 0.8% | 0.0% | 0.0% |
| long-jump | 1 | sample | 97.1% | -0.12 | 12 | 238 | 0.25 | 25.1% | 0.001 | 97.1% | 0.3% | 2.6% | 0.0% | 0.0% |
| long-walk | 0 | sample | 99.4% | -0.03 | 7 | 186 | 0.59 | 59.1% | 0.001 | 99.4% | 0.1% | 0.5% | 0.0% | 0.0% |
| long-walk | 1 | sample | 97.5% | -0.09 | 10 | 220 | 0.35 | 35.3% | 0.000 | 97.5% | 0.6% | 1.9% | 0.0% | 0.0% |
| edge-jump | 0 | sample | 99.3% | -0.03 | 7 | 187 | 0.67 | 66.9% | 0.001 | 99.3% | 0.1% | 0.6% | 0.0% | 0.0% |
| edge-jump | 1 | sample | 99.3% | -0.02 | 7 | 187 | 0.67 | 65.8% | 0.003 | 99.3% | 0.1% | 0.7% | 0.0% | 0.0% |
| edge-walk | 0 | sample | 99.6% | -0.03 | 7 | 186 | 0.62 | 62.4% | 0.003 | 99.6% | 0.0% | 0.4% | 0.0% | 0.0% |
| edge-walk | 1 | sample | 99.3% | -0.03 | 7 | 185 | 0.65 | 64.6% | 0.002 | 99.3% | 0.0% | 0.7% | 0.0% | 0.0% |
| none | 0 | argmax | 97.2% | -0.13 | 12 | 280 | 0.00 | 0.0% | 0.000 | 97.2% | 0.7% | 2.1% | 0.0% | 0.0% |
| none | 1 | argmax | 96.9% | -0.15 | 13 | 279 | 0.00 | 0.0% | 0.000 | 96.9% | 0.1% | 2.9% | 0.0% | 0.0% |
| kind-jump | 0 | argmax | 99.6% | -0.02 | 6 | 183 | 0.64 | 64.3% | 0.002 | 99.6% | 0.0% | 0.4% | 0.0% | 0.0% |
| kind-jump | 1 | argmax | 99.4% | -0.02 | 7 | 183 | 0.62 | 62.3% | 0.002 | 99.4% | 0.0% | 0.6% | 0.0% | 0.0% |
| kind-walk | 0 | argmax | 99.7% | -0.02 | 6 | 183 | 0.64 | 63.7% | 0.002 | 99.7% | 0.0% | 0.3% | 0.0% | 0.0% |
| kind-walk | 1 | argmax | 99.5% | -0.03 | 7 | 184 | 0.63 | 62.7% | 0.002 | 99.5% | 0.0% | 0.5% | 0.0% | 0.0% |
| long-jump | 0 | argmax | 99.1% | -0.03 | 6 | 183 | 0.59 | 58.8% | 0.001 | 99.1% | 0.1% | 0.8% | 0.0% | 0.0% |
| long-jump | 1 | argmax | 97.0% | -0.11 | 10 | 232 | 0.25 | 25.2% | 0.000 | 97.0% | 0.5% | 2.5% | 0.0% | 0.0% |
| long-walk | 0 | argmax | 99.6% | -0.03 | 6 | 184 | 0.59 | 59.4% | 0.000 | 99.6% | 0.1% | 0.3% | 0.0% | 0.0% |
| long-walk | 1 | argmax | 97.9% | -0.09 | 9 | 216 | 0.35 | 34.7% | 0.000 | 97.9% | 0.2% | 1.8% | 0.0% | 0.0% |
| edge-jump | 0 | argmax | 99.8% | -0.03 | 7 | 186 | 0.66 | 66.3% | 0.002 | 99.8% | 0.0% | 0.2% | 0.0% | 0.0% |
| edge-jump | 1 | argmax | 99.5% | -0.02 | 6 | 184 | 0.65 | 65.3% | 0.002 | 99.5% | 0.1% | 0.4% | 0.0% | 0.0% |
| edge-walk | 0 | argmax | 99.5% | -0.02 | 7 | 185 | 0.62 | 61.7% | 0.003 | 99.5% | 0.0% | 0.5% | 0.0% | 0.0% |
| edge-walk | 1 | argmax | 99.6% | -0.02 | 6 | 183 | 0.64 | 64.4% | 0.002 | 99.6% | 0.0% | 0.4% | 0.0% | 0.0% |

`other` endings: none occurred (no `stuck`, no `no-progress`).

## By task type and room, act sample

Success / Δhearts / clips per episode / mean steps / mean frames, act sample:

| arm | seed | cross W→E | cross E→W | south W→S | south E→S | crossing | south |
|---|---|---|---|---|---|---|---|
| none | 0 | 92.9% / -0.23 / 0.00 / 18 / 342 | 97.3% / -0.15 / 0.00 / 15 / 331 | 97.0% / -0.13 / 0.00 / 12 / 241 | 97.6% / -0.05 / 0.00 / 12 / 234 | 95.1% / -0.19 / 0.00 / 17 / 336 | 97.3% / -0.09 / 0.00 / 12 / 237 |
| none | 1 | 93.5% / -0.24 / 0.00 / 20 / 344 | 97.6% / -0.18 / 0.00 / 15 / 327 | 97.0% / -0.12 / 0.00 / 14 / 242 | 97.6% / -0.06 / 0.00 / 11 / 225 | 95.5% / -0.21 / 0.00 / 17 / 336 | 97.3% / -0.09 / 0.00 / 12 / 233 |
| kind-jump | 0 | 99.8% / -0.01 / 1.00 / 4 / 159 | 100.0% / -0.00 / 1.00 / 4 / 148 | 98.4% / -0.07 / 0.35 / 9 / 209 | 99.3% / -0.01 / 0.22 / 10 / 224 | 99.9% / -0.01 / 1.00 / 4 / 154 | 98.9% / -0.04 / 0.28 / 9 / 216 |
| kind-jump | 1 | 99.8% / -0.01 / 0.99 / 4 / 160 | 99.8% / -0.00 / 0.99 / 3 / 148 | 98.6% / -0.07 / 0.35 / 9 / 213 | 99.3% / -0.02 / 0.15 / 11 / 223 | 99.8% / -0.01 / 0.99 / 4 / 154 | 99.0% / -0.04 / 0.25 / 10 / 218 |
| kind-walk | 0 | 99.6% / -0.01 / 0.99 / 4 / 160 | 99.8% / -0.01 / 0.99 / 4 / 150 | 98.9% / -0.06 / 0.39 / 9 / 213 | 99.6% / -0.01 / 0.18 / 9 / 221 | 99.7% / -0.01 / 0.99 / 4 / 155 | 99.2% / -0.04 / 0.28 / 9 / 217 |
| kind-walk | 1 | 99.8% / -0.01 / 1.00 / 4 / 159 | 100.0% / -0.00 / 1.00 / 4 / 149 | 99.1% / -0.08 / 0.36 / 10 / 216 | 98.9% / -0.02 / 0.16 / 12 / 226 | 99.9% / -0.01 / 1.00 / 4 / 154 | 99.0% / -0.05 / 0.26 / 11 / 221 |
| long-jump | 0 | 98.3% / -0.02 / 0.98 / 4 / 163 | 99.3% / -0.00 / 0.99 / 3 / 147 | 98.4% / -0.07 / 0.34 / 9 / 214 | 98.2% / -0.05 / 0.00 / 9 / 220 | 98.8% / -0.01 / 0.98 / 4 / 155 | 98.3% / -0.06 / 0.17 / 9 / 217 |
| long-jump | 1 | 92.2% / -0.27 / 0.01 / 20 / 343 | 99.8% / -0.00 / 0.99 / 3 / 149 | 97.9% / -0.15 / 0.00 / 13 / 238 | 98.4% / -0.05 / 0.01 / 11 / 221 | 95.9% / -0.14 / 0.49 / 12 / 247 | 98.2% / -0.10 / 0.00 / 12 / 229 |
| long-walk | 0 | 99.4% / -0.01 / 0.99 / 4 / 164 | 100.0% / -0.00 / 1.00 / 3 / 147 | 99.1% / -0.06 / 0.37 / 10 / 212 | 99.3% / -0.04 / 0.00 / 10 / 222 | 99.7% / -0.01 / 0.99 / 3 / 155 | 99.2% / -0.05 / 0.18 / 10 / 217 |
| long-walk | 1 | 94.8% / -0.21 / 0.31 / 14 / 277 | 99.6% / +0.00 / 0.99 / 3 / 147 | 97.3% / -0.12 / 0.03 / 12 / 233 | 98.4% / -0.04 / 0.07 / 10 / 221 | 97.1% / -0.11 / 0.65 / 8 / 213 | 97.9% / -0.08 / 0.05 / 11 / 227 |
| edge-jump | 0 | 99.6% / -0.01 / 0.99 / 5 / 164 | 99.8% / -0.00 / 1.00 / 4 / 147 | 98.6% / -0.08 / 0.42 / 9 / 215 | 99.3% / -0.02 / 0.27 / 11 / 225 | 99.7% / -0.01 / 1.00 / 4 / 156 | 99.0% / -0.05 / 0.34 / 10 / 220 |
| edge-jump | 1 | 99.8% / -0.01 / 1.00 / 4 / 159 | 99.8% / +0.00 / 1.00 / 3 / 149 | 98.9% / -0.06 / 0.40 / 10 / 216 | 98.7% / -0.03 / 0.25 / 10 / 224 | 99.8% / -0.00 / 1.00 / 4 / 154 | 98.8% / -0.04 / 0.32 / 10 / 220 |
| edge-walk | 0 | 99.6% / -0.01 / 0.99 / 5 / 161 | 100.0% / -0.01 / 1.00 / 4 / 148 | 98.9% / -0.07 / 0.37 / 9 / 214 | 99.8% / -0.02 / 0.12 / 10 / 223 | 99.8% / -0.01 / 1.00 / 4 / 155 | 99.3% / -0.04 / 0.24 / 9 / 219 |
| edge-walk | 1 | 99.6% / -0.01 / 0.98 / 4 / 161 | 100.0% / -0.00 / 1.00 / 3 / 147 | 98.2% / -0.08 / 0.35 / 9 / 211 | 99.6% / -0.01 / 0.24 / 10 / 224 | 99.8% / -0.01 / 0.99 / 4 / 154 | 98.9% / -0.05 / 0.29 / 9 / 218 |

Per room, crossing and south: success / Δhearts / clips per episode, act sample:

| arm | seed | 0x59 cross | 0x59 south | 0x5a cross | 0x5a south | 0x5b cross | 0x5b south |
|---|---|---|---|---|---|---|---|
| none | 0 | 97.0% / -0.11 / 0.00 | 96.6% / -0.11 / 0.00 | 96.4% / -0.20 / 0.00 | 98.3% / -0.07 / 0.00 | 91.7% / -0.27 / 0.00 | 97.0% / -0.09 / 0.00 |
| none | 1 | 95.7% / -0.14 / 0.00 | 97.0% / -0.09 / 0.00 | 96.4% / -0.19 / 0.00 | 99.0% / -0.06 / 0.00 | 94.4% / -0.30 / 0.00 | 96.0% / -0.13 / 0.00 |
| kind-jump | 0 | 100.0% / -0.00 / 1.00 | 97.3% / -0.09 / 0.01 | 100.0% / +0.00 / 1.00 | 100.0% / +0.00 / 0.52 | 99.7% / -0.02 / 1.00 | 99.3% / -0.03 / 0.33 |
| kind-jump | 1 | 100.0% / +0.00 / 1.00 | 97.3% / -0.09 / 0.02 | 100.0% / -0.00 / 0.99 | 100.0% / +0.00 / 0.52 | 99.3% / -0.02 / 0.98 | 99.7% / -0.04 / 0.21 |
| kind-walk | 0 | 100.0% / +0.00 / 1.00 | 98.0% / -0.08 / 0.05 | 99.7% / -0.02 / 0.98 | 100.0% / -0.00 / 0.52 | 99.3% / -0.02 / 0.99 | 99.7% / -0.03 / 0.28 |
| kind-walk | 1 | 100.0% / +0.00 / 1.00 | 98.3% / -0.11 / 0.00 | 100.0% / +0.00 / 1.00 | 100.0% / -0.00 / 0.53 | 99.7% / -0.02 / 1.00 | 98.7% / -0.04 / 0.23 |
| long-jump | 0 | 99.3% / -0.00 / 0.99 | 98.3% / -0.08 / 0.00 | 99.7% / -0.01 / 0.99 | 99.0% / -0.00 / 0.50 | 97.3% / -0.02 / 0.97 | 97.7% / -0.10 / 0.01 |
| long-jump | 1 | 94.1% / -0.15 / 0.49 | 97.3% / -0.11 / 0.00 | 99.3% / -0.08 / 0.51 | 99.7% / -0.08 / 0.00 | 94.4% / -0.20 / 0.47 | 97.7% / -0.11 / 0.01 |
| long-walk | 0 | 100.0% / +0.00 / 1.00 | 98.6% / -0.08 / 0.03 | 100.0% / -0.00 / 1.00 | 100.0% / +0.00 / 0.51 | 99.0% / -0.02 / 0.98 | 99.0% / -0.07 / 0.00 |
| long-walk | 1 | 96.7% / -0.14 / 0.50 | 97.3% / -0.10 / 0.01 | 98.7% / +0.00 / 0.98 | 98.3% / -0.05 / 0.04 | 96.0% / -0.18 / 0.47 | 98.0% / -0.09 / 0.10 |
| edge-jump | 0 | 100.0% / +0.00 / 1.00 | 98.3% / -0.09 / 0.10 | 99.7% / +0.00 / 1.00 | 100.0% / -0.00 / 0.52 | 99.3% / -0.02 / 0.99 | 98.7% / -0.05 / 0.40 |
| edge-jump | 1 | 100.0% / +0.00 / 1.00 | 97.6% / -0.09 / 0.06 | 99.7% / +0.00 / 1.00 | 100.0% / -0.00 / 0.52 | 99.7% / -0.01 / 1.00 | 98.7% / -0.04 / 0.39 |
| edge-walk | 0 | 100.0% / +0.00 / 1.00 | 98.6% / -0.09 / 0.02 | 100.0% / -0.01 / 1.00 | 100.0% / -0.00 / 0.52 | 99.3% / -0.02 / 0.99 | 99.3% / -0.04 / 0.19 |
| edge-walk | 1 | 100.0% / +0.00 / 0.98 | 97.6% / -0.10 / 0.00 | 100.0% / +0.00 / 1.00 | 100.0% / -0.00 / 0.52 | 99.3% / -0.02 / 0.99 | 99.0% / -0.04 / 0.36 |

Crossing success per start state, act sample (* = the skip is available on the entry row):

| arm | seed | 0_59w_0 | 0_59w_1* | 0_59w_2 | 0_59e_0* | 0_59e_1* | 0_59e_2* | 0_5aw_0* | 0_5aw_1* | 0_5aw_2* | 0_5ae_0 | 0_5ae_1* | 0_5ae_2* | 0_5bw_0* | 0_5bw_1* | 0_5bw_2 | 0_5be_0* | 0_5be_1* | 0_5be_2* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 94 | 94 | 96 | 98 | 100 | 100 | 100 | 92 | 95 | 94 | 97 | 100 | 84 | 92 | 89 | 91 | 98 | 98 |
| none | 1 | 94 | 90 | 96 | 96 | 98 | 100 | 96 | 94 | 98 | 98 | 98 | 94 | 90 | 94 | 90 | 96 | 98 | 100 |
| kind-jump | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 98 | 100 | 100 | 100 | 100 | 100 |
| kind-jump | 1 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 98 | 100 | 100 | 100 | 98 | 100 |
| kind-walk | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 98 | 100 | 100 | 96 | 100 | 100 | 100 | 100 | 100 |
| kind-walk | 1 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 98 | 100 | 100 | 100 | 100 | 100 |
| long-jump | 0 | 100 | 100 | 100 | 98 | 100 | 98 | 100 | 100 | 100 | 98 | 100 | 100 | 88 | 96 | 100 | 100 | 100 | 100 |
| long-jump | 1 | 92 | 80 | 94 | 100 | 100 | 98 | 100 | 98 | 98 | 100 | 100 | 100 | 88 | 90 | 90 | 100 | 100 | 100 |
| long-walk | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 96 | 98 | 100 | 100 | 100 | 100 |
| long-walk | 1 | 94 | 88 | 98 | 100 | 100 | 100 | 100 | 100 | 95 | 96 | 100 | 100 | 98 | 94 | 87 | 100 | 100 | 100 |
| edge-jump | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 98 | 100 | 100 | 96 | 100 | 100 | 100 | 100 | 100 |
| edge-jump | 1 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 98 | 98 | 100 | 100 | 100 | 100 | 100 |
| edge-walk | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 96 | 100 | 100 | 100 | 100 | 100 |
| edge-walk | 1 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 96 | 100 | 100 | 100 | 100 | 100 |

## By task type and room, act argmax

Success / Δhearts / clips per episode / mean steps / mean frames, act argmax:

| arm | seed | cross W→E | cross E→W | south W→S | south E→S | crossing | south |
|---|---|---|---|---|---|---|---|
| none | 0 | 93.1% / -0.22 / 0.00 / 17 / 335 | 98.0% / -0.12 / 0.00 / 12 / 320 | 98.9% / -0.11 / 0.00 / 9 / 232 | 99.1% / -0.04 / 0.00 / 9 / 229 | 95.5% / -0.17 / 0.00 / 14 / 328 | 99.0% / -0.08 / 0.00 / 9 / 230 |
| none | 1 | 93.7% / -0.23 / 0.00 / 18 / 337 | 97.1% / -0.20 / 0.00 / 13 / 320 | 97.7% / -0.13 / 0.00 / 13 / 236 | 99.3% / -0.04 / 0.00 / 9 / 221 | 95.4% / -0.22 / 0.00 / 16 / 329 | 98.5% / -0.08 / 0.00 / 11 / 229 |
| kind-jump | 0 | 99.8% / -0.01 / 0.99 / 4 / 159 | 100.0% / +0.00 / 1.00 / 4 / 146 | 98.4% / -0.07 / 0.35 / 8 / 206 | 100.0% / -0.01 / 0.22 / 10 / 221 | 99.9% / -0.00 / 1.00 / 4 / 153 | 99.2% / -0.04 / 0.28 / 9 / 214 |
| kind-jump | 1 | 99.6% / -0.01 / 0.99 / 4 / 157 | 100.0% / +0.00 / 1.00 / 3 / 146 | 98.2% / -0.07 / 0.35 / 8 / 208 | 100.0% / -0.01 / 0.14 / 11 / 222 | 99.8% / -0.01 / 1.00 / 4 / 152 | 99.1% / -0.04 / 0.24 / 10 / 215 |
| kind-walk | 0 | 99.6% / -0.01 / 0.99 / 4 / 159 | 100.0% / +0.00 / 1.00 / 3 / 146 | 99.3% / -0.06 / 0.38 / 8 / 210 | 100.0% / -0.01 / 0.17 / 9 / 219 | 99.8% / -0.01 / 1.00 / 4 / 153 | 99.7% / -0.03 / 0.27 / 8 / 215 |
| kind-walk | 1 | 99.8% / -0.01 / 0.99 / 5 / 159 | 100.0% / +0.00 / 1.00 / 4 / 146 | 98.6% / -0.09 / 0.36 / 9 / 210 | 99.6% / -0.02 / 0.15 / 12 / 224 | 99.9% / -0.01 / 0.99 / 4 / 153 | 99.1% / -0.05 / 0.25 / 11 / 217 |
| long-jump | 0 | 99.1% / -0.01 / 0.99 / 4 / 162 | 100.0% / +0.00 / 1.00 / 3 / 146 | 98.9% / -0.05 / 0.35 / 8 / 208 | 98.2% / -0.05 / 0.00 / 8 / 217 | 99.6% / -0.01 / 0.99 / 3 / 154 | 98.5% / -0.05 / 0.17 / 8 / 213 |
| long-jump | 1 | 91.3% / -0.24 / 0.00 / 17 / 331 | 100.0% / +0.00 / 1.00 / 3 / 146 | 97.9% / -0.14 / 0.00 / 12 / 230 | 98.9% / -0.05 / 0.01 / 10 / 217 | 95.6% / -0.12 / 0.49 / 10 / 240 | 98.4% / -0.09 / 0.00 / 11 / 223 |
| long-walk | 0 | 99.1% / -0.01 / 0.99 / 3 / 161 | 100.0% / +0.00 / 1.00 / 3 / 146 | 99.5% / -0.07 / 0.37 / 9 / 211 | 99.6% / -0.04 / 0.00 / 9 / 221 | 99.6% / -0.01 / 1.00 / 3 / 154 | 99.6% / -0.05 / 0.18 / 9 / 216 |
| long-walk | 1 | 94.6% / -0.20 / 0.32 / 13 / 270 | 100.0% / +0.00 / 1.00 / 3 / 146 | 97.9% / -0.11 / 0.00 / 11 / 228 | 99.3% / -0.03 / 0.06 / 9 / 218 | 97.3% / -0.10 / 0.65 / 8 / 209 | 98.7% / -0.07 / 0.03 / 10 / 223 |
| edge-jump | 0 | 99.6% / -0.01 / 1.00 / 4 / 162 | 100.0% / +0.00 / 1.00 / 4 / 146 | 99.8% / -0.08 / 0.41 / 8 / 214 | 99.8% / -0.02 / 0.23 / 10 / 222 | 99.8% / -0.01 / 1.00 / 4 / 154 | 99.8% / -0.05 / 0.32 / 9 / 218 |
| edge-jump | 1 | 99.4% / -0.01 / 0.99 / 4 / 157 | 100.0% / +0.00 / 1.00 / 3 / 149 | 98.6% / -0.06 / 0.39 / 9 / 211 | 100.0% / -0.01 / 0.22 / 9 / 221 | 99.7% / -0.01 / 1.00 / 4 / 153 | 99.3% / -0.03 / 0.30 / 9 / 216 |
| edge-walk | 0 | 99.6% / -0.01 / 1.00 / 5 / 160 | 100.0% / +0.00 / 1.00 / 4 / 147 | 98.4% / -0.06 / 0.35 / 8 / 210 | 100.0% / -0.02 / 0.10 / 10 / 223 | 99.8% / -0.01 / 1.00 / 4 / 153 | 99.2% / -0.04 / 0.23 / 9 / 217 |
| edge-walk | 1 | 99.6% / -0.01 / 1.00 / 4 / 157 | 100.0% / +0.00 / 1.00 / 3 / 146 | 98.6% / -0.08 / 0.35 / 8 / 208 | 100.0% / -0.01 / 0.22 / 9 / 222 | 99.8% / -0.01 / 1.00 / 4 / 152 | 99.3% / -0.04 / 0.28 / 9 / 215 |

Per room, crossing and south: success / Δhearts / clips per episode, act argmax:

| arm | seed | 0x59 cross | 0x59 south | 0x5a cross | 0x5a south | 0x5b cross | 0x5b south |
|---|---|---|---|---|---|---|---|
| none | 0 | 97.4% / -0.10 / 0.00 | 98.6% / -0.10 / 0.00 | 99.0% / -0.15 / 0.00 | 99.7% / -0.05 / 0.00 | 90.0% / -0.28 / 0.00 | 98.7% / -0.08 / 0.00 |
| none | 1 | 95.7% / -0.21 / 0.00 | 97.3% / -0.10 / 0.00 | 97.7% / -0.16 / 0.00 | 99.7% / -0.06 / 0.00 | 92.7% / -0.28 / 0.00 | 98.7% / -0.09 / 0.00 |
| kind-jump | 0 | 100.0% / +0.00 / 1.00 | 98.0% / -0.09 / 0.00 | 100.0% / +0.00 / 1.00 | 100.0% / +0.00 / 0.52 | 99.7% / -0.01 / 0.99 | 99.7% / -0.02 / 0.32 |
| kind-jump | 1 | 100.0% / +0.00 / 1.00 | 97.6% / -0.09 / 0.00 | 100.0% / +0.00 / 1.00 | 100.0% / +0.00 / 0.52 | 99.3% / -0.02 / 0.99 | 99.7% / -0.03 / 0.20 |
| kind-walk | 0 | 100.0% / +0.00 / 1.00 | 99.3% / -0.07 / 0.04 | 100.0% / +0.00 / 1.00 | 100.0% / +0.00 / 0.52 | 99.3% / -0.02 / 0.99 | 99.7% / -0.03 / 0.26 |
| kind-walk | 1 | 100.0% / +0.00 / 1.00 | 97.6% / -0.11 / 0.00 | 100.0% / +0.00 / 1.00 | 100.0% / -0.00 / 0.52 | 99.7% / -0.02 / 0.98 | 99.7% / -0.04 / 0.23 |
| long-jump | 0 | 100.0% / +0.00 / 1.00 | 98.6% / -0.06 / 0.00 | 100.0% / +0.00 / 1.00 | 100.0% / +0.00 / 0.52 | 98.7% / -0.02 / 0.98 | 97.0% / -0.10 / 0.00 |
| long-jump | 1 | 96.1% / -0.13 / 0.50 | 97.6% / -0.11 / 0.00 | 98.7% / -0.05 / 0.52 | 99.7% / -0.07 / 0.00 | 92.0% / -0.19 / 0.46 | 98.0% / -0.10 / 0.01 |
| long-walk | 0 | 100.0% / +0.00 / 1.00 | 99.3% / -0.09 / 0.03 | 100.0% / +0.00 / 1.00 | 100.0% / +0.00 / 0.52 | 98.7% / -0.02 / 0.99 | 99.3% / -0.07 / 0.00 |
| long-walk | 1 | 96.1% / -0.13 / 0.50 | 97.6% / -0.08 / 0.00 | 100.0% / +0.00 / 1.00 | 99.7% / -0.05 / 0.01 | 95.7% / -0.17 / 0.46 | 98.7% / -0.08 / 0.09 |
| edge-jump | 0 | 100.0% / +0.00 / 1.00 | 100.0% / -0.09 / 0.09 | 100.0% / +0.00 / 1.00 | 100.0% / -0.00 / 0.52 | 99.3% / -0.02 / 0.99 | 99.3% / -0.05 / 0.35 |
| edge-jump | 1 | 100.0% / +0.00 / 1.00 | 98.0% / -0.08 / 0.05 | 100.0% / +0.00 / 1.00 | 100.0% / +0.00 / 0.52 | 99.0% / -0.02 / 0.99 | 100.0% / -0.02 / 0.34 |
| edge-walk | 0 | 100.0% / +0.00 / 1.00 | 98.0% / -0.08 / 0.00 | 100.0% / +0.00 / 1.00 | 100.0% / +0.00 / 0.52 | 99.3% / -0.02 / 0.99 | 99.7% / -0.04 / 0.16 |
| edge-walk | 1 | 100.0% / +0.00 / 1.00 | 98.3% / -0.10 / 0.00 | 100.0% / +0.00 / 1.00 | 100.0% / +0.00 / 0.52 | 99.3% / -0.02 / 0.99 | 99.7% / -0.03 / 0.32 |

Crossing success per start state, act argmax (* = the skip is available on the entry row):

| arm | seed | 0_59w_0 | 0_59w_1* | 0_59w_2 | 0_59e_0* | 0_59e_1* | 0_59e_2* | 0_5aw_0* | 0_5aw_1* | 0_5aw_2* | 0_5ae_0 | 0_5ae_1* | 0_5ae_2* | 0_5bw_0* | 0_5bw_1* | 0_5bw_2 | 0_5be_0* | 0_5be_1* | 0_5be_2* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 96 | 96 | 96 | 96 | 100 | 100 | 100 | 100 | 98 | 100 | 97 | 100 | 67 | 94 | 90 | 93 | 98 | 98 |
| none | 1 | 94 | 92 | 96 | 96 | 98 | 98 | 98 | 96 | 98 | 94 | 100 | 100 | 92 | 92 | 87 | 96 | 93 | 98 |
| kind-jump | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 98 | 100 | 100 | 100 | 100 | 100 |
| kind-jump | 1 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 96 | 100 | 100 | 100 | 100 | 100 |
| kind-walk | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 96 | 100 | 100 | 100 | 100 | 100 |
| kind-walk | 1 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 98 | 100 | 100 | 100 | 100 | 100 |
| long-jump | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 92 | 100 | 100 | 100 | 100 | 100 |
| long-jump | 1 | 94 | 86 | 96 | 100 | 100 | 100 | 98 | 96 | 98 | 100 | 100 | 100 | 76 | 92 | 87 | 100 | 100 | 100 |
| long-walk | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 92 | 100 | 100 | 100 | 100 | 100 |
| long-walk | 1 | 94 | 86 | 96 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 90 | 94 | 92 | 100 | 100 | 100 |
| edge-jump | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 96 | 100 | 100 | 100 | 100 | 100 |
| edge-jump | 1 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 100 | 100 | 100 | 100 | 100 |
| edge-walk | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 96 | 100 | 100 | 100 | 100 | 100 |
| edge-walk | 1 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 96 | 100 | 100 | 100 | 100 | 100 |

## Wrap direction and wrong exits

Wrap directions from the evaluation records (crossing tasks: right = toward the entry edge, wrong = toward the target edge; per episode). `entry-edge exits` = crossings that ended exit-wrong through the edge they came in by, `0 wraps` of them without any wrap. 0_5bw_0 E: crossings from 0_5bw_0, whose wrap lands in the lower (wrong) E opening: success, exit-wrong through E / through W.

| arm | seed | act | right wraps/crossing | wrong wraps/crossing | crossings with ≥ 2 wraps | entry-edge exits (0 wraps) | wraps/south ep. | 0_5bw_0 E: success, exit E / W |
|---|---|---|---|---|---|---|---|---|
| none | 0 | sample | 0.00 | 0.000 | 0/911 | 0 (0) | 0.00 | 83.7%, 5 / 0 |
| none | 1 | sample | 0.00 | 0.000 | 0/911 | 1 (1) | 0.00 | 89.8%, 3 / 0 |
| kind-jump | 0 | sample | 1.00 | 0.001 | 1/911 | 0 (0) | 0.28 | 98.0%, 0 / 0 |
| kind-jump | 1 | sample | 0.99 | 0.000 | 0/911 | 0 (0) | 0.25 | 98.0%, 0 / 0 |
| kind-walk | 0 | sample | 0.99 | 0.000 | 0/911 | 0 (0) | 0.28 | 95.9%, 0 / 0 |
| kind-walk | 1 | sample | 1.00 | 0.000 | 0/911 | 0 (0) | 0.26 | 98.0%, 0 / 0 |
| long-jump | 0 | sample | 0.98 | 0.001 | 0/911 | 8 (8) | 0.17 | 87.8%, 0 / 4 |
| long-jump | 1 | sample | 0.49 | 0.002 | 1/911 | 1 (1) | 0.00 | 87.8%, 4 / 0 |
| long-walk | 0 | sample | 0.99 | 0.001 | 1/911 | 1 (1) | 0.18 | 95.9%, 0 / 0 |
| long-walk | 1 | sample | 0.65 | 0.001 | 0/911 | 4 (4) | 0.05 | 98.0%, 0 / 0 |
| edge-jump | 0 | sample | 1.00 | 0.000 | 0/911 | 1 (1) | 0.34 | 95.9%, 0 / 0 |
| edge-jump | 1 | sample | 1.00 | 0.000 | 0/911 | 1 (1) | 0.32 | 98.0%, 0 / 0 |
| edge-walk | 0 | sample | 1.00 | 0.000 | 0/911 | 0 (0) | 0.24 | 95.9%, 0 / 0 |
| edge-walk | 1 | sample | 0.99 | 0.000 | 0/911 | 0 (0) | 0.29 | 95.9%, 0 / 0 |
| none | 0 | argmax | 0.00 | 0.000 | 0/911 | 0 (0) | 0.00 | 67.3%, 13 / 0 |
| none | 1 | argmax | 0.00 | 0.000 | 0/911 | 0 (0) | 0.00 | 91.8%, 1 / 0 |
| kind-jump | 0 | argmax | 1.00 | 0.000 | 0/911 | 0 (0) | 0.28 | 98.0%, 0 / 0 |
| kind-jump | 1 | argmax | 1.00 | 0.000 | 0/911 | 0 (0) | 0.24 | 95.9%, 0 / 0 |
| kind-walk | 0 | argmax | 1.00 | 0.000 | 0/911 | 0 (0) | 0.27 | 95.9%, 0 / 0 |
| kind-walk | 1 | argmax | 0.99 | 0.000 | 0/911 | 0 (0) | 0.25 | 98.0%, 0 / 0 |
| long-jump | 0 | argmax | 0.99 | 0.000 | 0/911 | 2 (2) | 0.17 | 91.8%, 0 / 2 |
| long-jump | 1 | argmax | 0.49 | 0.000 | 0/911 | 0 (0) | 0.00 | 75.5%, 9 / 0 |
| long-walk | 0 | argmax | 1.00 | 0.000 | 0/911 | 2 (2) | 0.18 | 91.8%, 0 / 2 |
| long-walk | 1 | argmax | 0.65 | 0.000 | 0/911 | 0 (0) | 0.03 | 89.8%, 4 / 0 |
| edge-jump | 0 | argmax | 1.00 | 0.000 | 0/911 | 0 (0) | 0.32 | 95.9%, 0 / 0 |
| edge-jump | 1 | argmax | 1.00 | 0.000 | 0/911 | 1 (1) | 0.30 | 93.9%, 0 / 1 |
| edge-walk | 0 | argmax | 1.00 | 0.000 | 0/911 | 0 (0) | 0.23 | 95.9%, 0 / 0 |
| edge-walk | 1 | argmax | 1.00 | 0.000 | 0/911 | 0 (0) | 0.28 | 95.9%, 0 / 0 |

## Learning

Training (metrics.jsonl; success = rolling mean of 5 iterations of the training success rate, clips = `clips/done` per finished episode):

| arm | seed | steps | 50% at | 80% at | 90% at | final success (last 10 it.) | clips/ep first 5 it. | clips ≥ 0.5 from | clips peak (at) | clips within 0.1 of final from | clips last 10 it. | aborted last 10 it. | wall time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 1.05M | 106k | 164k | 217k | 96.4% | 0.00 | never | 0.00 (4k) | 4k | 0.00 | 0.000 | 49 min |
| none | 1 | 1.05M | 139k | 180k | 319k | 96.8% | 0.00 | never | 0.00 (4k) | 4k | 0.00 | 0.000 | 38 min |
| kind-jump | 0 | 1.05M | 90k | 156k | 180k | 99.6% | 2.14 | 4k | 11.96 (106k) | 303k | 0.63 | 0.002 | 58 min |
| kind-jump | 1 | 1.05M | 70k | 123k | 168k | 99.5% | 2.15 | 4k | 4.45 (78k) | 406k | 0.61 | 0.003 | 42 min |
| kind-walk | 0 | 1.05M | 82k | 135k | 160k | 99.6% | 2.09 | 4k | 9.24 (66k) | 262k | 0.61 | 0.002 | 63 min |
| kind-walk | 1 | 1.05M | 70k | 115k | 156k | 99.4% | 2.20 | 4k | 4.04 (70k) | 246k | 0.62 | 0.003 | 34 min |
| long-jump | 0 | 1.05M | 139k | 225k | 303k | 99.1% | 0.22 | 381k | 0.60 (741k) | 356k | 0.57 | 0.002 | 57 min |
| long-jump | 1 | 1.05M | 123k | 201k | 274k | 97.4% | 0.25 | never | 0.34 (197k) | 201k | 0.23 | 0.000 | 36 min |
| long-walk | 0 | 1.05M | 135k | 209k | 270k | 99.4% | 0.22 | 397k | 0.60 (1024k) | 426k | 0.59 | 0.002 | 62 min |
| long-walk | 1 | 1.05M | 131k | 197k | 279k | 97.5% | 0.24 | never | 0.37 (184k) | 786k | 0.33 | 0.000 | 35 min |
| edge-jump | 0 | 1.05M | 106k | 340k | 434k | 99.4% | 1.15 | 4k | 17.76 (299k) | 700k | 0.64 | 0.003 | 53 min |
| edge-jump | 1 | 1.05M | 98k | 147k | 184k | 99.4% | 1.01 | 4k | 1.85 (70k) | 217k | 0.65 | 0.004 | 44 min |
| edge-walk | 0 | 1.05M | 90k | 213k | 233k | 99.5% | 1.09 | 4k | 11.73 (184k) | 397k | 0.63 | 0.003 | 53 min |
| edge-walk | 1 | 1.05M | 86k | 143k | 180k | 99.4% | 0.99 | 4k | 1.72 (70k) | 254k | 0.63 | 0.002 | 42 min |

`clips/ep first 5 it.` is the untrained policy's use; `peak` is the rolling maximum; `within 0.1 of final from` is
where the rolling mean enters and stays within 0.1 of the last 10 iterations' mean.

## Behaviour (replays, act sample)

17 crossings per run (the same start states, targets and seeds for every run), classified by the first wrap:
`at once` = the first decision wraps (rooms in brackets); `step in, back` = one MOVE into the room along the entry
row, then the wrap back out of the entry edge; `row first` = Link changed rows before the first wrap; `wrong-way` =
wraps toward the target edge; south = the 19 south tasks and which start states wrapped in them.

| arm | seed | crossings | walked (no wrap) | at once | step in, back | row first | other | wrong-way wraps | ≥ 2 wraps | steps per success | south: wrapped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 17 | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 14.2 | 19: none |
| kind-jump | 0 | 17 | 0 | 0 | 14 | 3 | 0 | 0 | 0 | 3.6 | 19: 0_5aw 5 |
| kind-walk | 0 | 17 | 0 | 6 (0x5b 6) | 7 | 3 | 1 | 0 | 0 | 3.2 | 19: 0_5aw 5 |
| long-jump | 0 | 17 | 0 | 7 (0x5b 7) | 7 | 3 | 0 | 0 | 0 | 2.8 | 19: 0_5aw 5 |
| long-walk | 0 | 17 | 0 | 7 (0x5b 7) | 6 | 4 | 0 | 0 | 0 | 2.9 | 19: 0_5aw 4 |
| edge-jump | 0 | 17 | 0 | 0 | 12 | 5 | 0 | 0 | 0 | 3.9 | 19: 0_5aw 5 |
| edge-walk | 0 | 17 | 0 | 0 | 14 | 3 | 0 | 0 | 0 | 3.5 | 19: 0_5aw 5 |
| none | 1 | 17 | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 14.8 | 19: none |
| kind-jump | 1 | 17 | 0 | 0 | 14 | 3 | 0 | 0 | 0 | 3.2 | 19: 0_5aw 5 |
| kind-walk | 1 | 17 | 0 | 0 | 14 | 3 | 0 | 0 | 0 | 3.6 | 19: 0_5aw 5 |
| long-jump | 1 | 17 | 5 | 6 (0x5b 6) | 4 | 1 | 1 | 0 | 0 | 8.1 | 19: none |
| long-walk | 1 | 17 | 4 | 7 (0x5a 1, 0x5b 6) | 5 | 1 | 0 | 0 | 0 | 5.4 | 19: none |
| edge-jump | 1 | 17 | 0 | 0 | 8 | 9 | 0 | 0 | 0 | 3.3 | 19: 0_5aw 5 |
| edge-walk | 1 | 17 | 0 | 0 | 14 | 3 | 0 | 0 | 0 | 3.3 | 19: 0_5aw 5 |

## Observations

1. **The first step that moves Link pays no PBRS** (`crates/env/src/critic.rs` `critique_movement`: with no
   baseline tile yet, the step only sets it). Measured with `_native.Env` on 0_59e_0, target W: `CLIP E` as the
   first action pays only −0.01; after `MOVE W 16, MOVE W 16` the same `CLIP E` pays +1.0 (`reward-pbrs-movement`
   +1.35, clamped). So the wrap from the entry bound, the scripted 2-step solution, earns no shaping, while a short
   step into the room and the wrap back earns up to +1 (`jump`) or up to the walk cap of the wrap's frames (about
   +0.33, `walk`). Under `kind` and `edge` (both shapings, both seeds) no policy wraps at once: 14 of 17 crossings
   step in (MOVE 4 or 8, usually cut short by a spawn interrupt after 2 frames) and wrap back. Under `long` (both
   seeds) and `kind-walk` seed 0, 0x5b's crossings wrap at once (6-7 of 17), the other rooms step in first.
   [INFERENCE] the missing first-step shaping is why the step-in pattern dominates; its cost is small (1-2 frames:
   143 against the reference's 141-142 frames per crossing).
2. **`CLIP` is charged `penalty-attack-miss` (−0.01)** under `kind` (`critique_attack` exempts only MOVE, WAIT and
   BOMB), while the MOVE triggers of `long` and `edge` cost nothing. An asymmetry between the arms; no visible effect
   at this size.
3. **`long` can miss the wrap in one direction.** Seed 1 of `long-jump` and `long-walk` wrap on E→W crossings (0.99
   per crossing) but not on W→E (0.01 and 0.31): W→E success 92.2% and 94.8% by sample, 14-20 steps, like `none`.
   Seed 0 of both found both directions. Under `long` the wrap needs MOVE W 32 at the W entry, and MOVE W 4/8/16
   there leaves by the wrong exit (−20); during training seed 1's `clips/done` never passed 0.37 per episode
   (seed 0's passed 0.5 at about 390k steps). `long` also shows the "meant to wrap, left instead" ending: crossings
   that left exit-wrong through the entry edge without a wrap, 8 of 911 (long-jump seed 0, sample), 1-4 elsewhere
   under `long`; 0-1 under `kind` and `edge`.
4. **Wrapping replaced walking through the monsters.** Every crossing in the replays of the clip arms (but `long`
   seed 1's W→E) is won by a wrap and an exit within 3 decisions of it; none wraps the wrong way, none wraps twice
   (≥ 2 wraps in at most 1 of 911 crossings per run). Hearts lost per crossing fell from about 0.2 to 0.00-0.01.
   States without a wrap on the entry row (0_59w_0, 0_59w_2, 0_5ae_0 in the replays) are crossed by changing rows
   first (MOVE N) and then wrapping: 4-10 steps.
5. **0_5bw_0 (row Y $B5)**: the skip is offered on the entry row, but its W wrap lands in 0x5b's lower E opening,
   whose exit doesn't count (the scripted reference fails all 100). Trained policies wrap from another row: 88-98%
   success by sample, 7-12 steps (`none`: 67-92%, mostly exit-wrong through the lower opening). Seed 1 of `long`
   doesn't wrap there at all (0 of its 43 and 48 successes; 27-28 steps).
6. **The wrap is used in south tasks where it shortens the way**: 0x5a entered from W with target S wraps to the east
   edge first in every run but `long` seed 1 (0.52 wraps per 0x5a south episode; all W→S episodes there). Elsewhere
   south tasks wrap less (0x59 0.00-0.10, 0x5b 0.00-0.40 per episode).
7. **Clip spam early in training under `kind` and `edge`**: random policies wrap 1-2 times per episode; seed 0
   peaked at 9-18 wraps per finished episode (66-300k steps), seed 1 at 1.7-4.5, before settling at about 0.6 by
   217-700k steps. The untrained `kind-jump` throughput model (102k steps) wrapped back and forth from the E entry
   in south tasks (37-67 wraps per episode on 0_59e_* with target S): from the E bound with target S each wrap pays
   at most −0.01, so the ping-pong is nearly free. [INFERENCE] the E-edge cells of 0x59 are outside the target-S
   wavefront (no PBRS on
   either wrap there; 0x5a's pay ±0.6). No final model ping-pongs.
8. **Aborted wraps** stay at 0.000-0.004 per episode (hits during the macro), in training and evaluation.
9. Sample and argmax agree within 0.5 points for every clip arm; `none` loses 0_5bw_0 crossings by argmax (seed 0:
   67%).

## Not done

- Milestone 2 (the dungeon block clip) was not part of this gauntlet.
- Two seeds per arm only; differences of under a point between the clip arms (98.6-99.8%, `long` seed 1 aside) are
  within seed noise.
- The behaviour table rests on 17 crossings per run; the per-direction and per-state numbers above come from the
  full evaluations.

## After the merge with Dungeon 4 (1.4.1.0)

`main`'s round 20 (Dungeon 4 in `d2first`, the ladder, Gleeok's heads, uncounted kills, the doorway attack mask) was
merged into `clips`: version 1.4.1.0 (1.4.0.0 plus this branch's COMPAT bump; observation schema 4, action schema 2).
No training; checks in the real game (`crates/zelda/tests/clips.rs`, `crates/env/tests/clips.rs`):

- **Doorways:** no skip is offered while Link is in a doorway (`DoorwayDir`, or outside X `$20`..`$D0`, Y
  `$5D`..`$BD`): every door entry state of levels 1-9, walked in frame by frame. In 4:0x50's E doorway the sword keeps
  to E/W (round 20's rule); after a block clip the MOVE mask matches the game and the attack masks are
  `GameView::sword_directions`.
- **Ladder and water:** with the ladder out (4:0x30, Link on the ladder over the water row) nothing is offered. New
  rule: no block clip into water (`$F4`); it changes nothing in Dungeon 4 (the other rules already leave its water
  out) and is pinned by a synthetic room.
- **Dungeon 4, every start state** (Link put on every grid point out of the doorway): room 0x03 offers 6 grid points,
  room 0x50 60, no other room. CLIP from each completes with one press 8 pixels past T, and the sideways walk from
  there goes where Link on T can't:

| room | heading, X (T) | zone (Y) | way on |
|---|---|---|---|
| 0x03 | N, X `$60` (`$85`) and `$90` (`$85`) | `$85`-`$95` | W / E into the side pockets of the triforce's enclosure (reachable on foot too; no use to the route) |
| 0x50 | N, X `$40` (`$65`, `$A5`), `$50` (`$95`), `$60` (`$85`), `$70` (`$75`), `$90` (`$A5`), `$A0` (`$95`), `$B0` (`$85`), `$C0` (`$75`), `$D0` (`$65`) | T to T + `$10` | W |
| 0x50 | S, X `$20` (`$AD`), `$30` (`$9D`), `$40` (`$8D`), `$50` (`$7D`), `$60` (`$6D`), `$80` (`$9D`), `$90` (`$8D`), `$A0` (`$7D`), `$B0` (`$6D`, `$AD`) | T - `$10` to T | E |

- **0x50 by one clip** (route: enter E, leave N; monsters cleared): plain `MOVE W 32`, `MOVE W 32`, `MOVE S 32`,
  `MOVE W 16` ×2, `MOVE S 32`, `MOVE W 16`, `MOVE W 8`, `MOVE N 32` ×2, `MOVE E 8`, `MOVE N 32`, `MOVE W 8`, `MOVE N
  32`: 14 actions, 424 frames into 4:0x40. With the clip: `MOVE W 32` ×2 (stopped on (`$B0`, `$8D`), G of the
  right diagonal's square above), `CLIP N` (14 frames, to (`$B0`, `$7D`)), `MOVE W 8`, `MOVE N 32`, `MOVE W 16` ×2,
  `MOVE N 32`: 8 actions, 318 frames. Both counts include the 191 frames of the last MOVE (out of the N door and the
  scroll). In a trial with the Vires in the room (seed 0) a plain walk round the diagonal's bottom was hit; the clip
  way was not.
- **`clips-gauntlet`** still draws both targets for each of its 18 start states (40 seeds each), and the start states
  of ow-zora, ow-moblin and every family with a Dungeon 4 room reset to the same `d2first` step with and without the
  route in routes.yaml.
