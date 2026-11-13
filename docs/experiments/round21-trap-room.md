# Round 21: the block clip in level 1 room 0x22 (trap room)

> **Afterwards (Nov 12, `docs/decisions.md` 2026-11-12 "One clip design"):** the owner chose the CLIP kind. The
> `long` and `edge` triggers and the `walk` shaping were removed from the code; `clips:` is now a plain `true|false`
> switch. The `long` and `edge` arms' YAMLs are in `experiments/r21/removed/` and no longer load; the kept ones
> (`trap22-kind-*`, `trap22-none-*`) say `clips: true|false`. This report describes the code as it ran.

Clips in the agent (branch `clips`, version 1.3.3.1, commit `376b6fd`): the executor's block-clip macro, reached
through `clips: {action: none|kind|long|edge}` (`docs/architecture.md` "Grid-point skip", `docs/decisions.md`
2026-11-12). This experiment trains level 1 room 0x22 alone, from the E door to the stairs (the bow cellar 0x7f),
with each design, with and without a curriculum. The owner's hypothesis: a curriculum that teaches the south route
round the diamond (it passes under the east and lower blocks, where `CLIP N` is available) lets the agent find the
clip by accident; a north curriculum would make it wander south first. This report records what happened. It
declares no winner.

## Summary

- **Every arm with a clip learned the clip, whatever the curriculum: 18 of 18 runs (3 designs, 2 to 3 curriculum
  settings, 2 seeds).** All but two clip from the same place, under the east block at (`$A0`,`$9D`); `edge-south`
  s1 also uses the lower-east block. The two exceptions (`long-north` s1, `edge-nocur` s2) are described below. No
  clip run pushes the block in evaluation. The argmax route is 7 decisions long: `MOVE W`×3, `MOVE S 8`, `MOVE W`,
  the clip, `MOVE W 32`. It takes 216–219 evaluation frames (the push route: 367–372). 15 of the 18 runs are at 100%
  in both act modes.
- **Without a clip, the curriculum decides everything.** `none` with the south curriculum: 100% in both seeds, in
  both act modes, all by the push. Without a curriculum, seed 1 never succeeds. Seed 2 learns the push only after
  328k steps and reaches 86% when sampling, but 0% under argmax. That matches round 14's trap-room without a
  curriculum.
- **On the hypothesis:**
  - The south curriculum did not decide whether the clip was found. The `nocur` arms found it just as surely.
  - It sped things up a little for `long` (90% at 78–86k steps against 98k) and for `kind` seed 1 (66k against 94k).
  - While training, the south curriculum produced a short-lived push share: at most 6–21% of seed-start episodes,
    at 33–78k steps. Under `kind` and `edge` the clip already dominated by then (from 8–29k); under `long` it took
    over at 45–66k.
  - The north curriculum did not stop the south clip: `kind-north` both seeds and `long-north` seed 2 clip from
    below the east block.
  - In `long-north` seed 1 the final model clips **from above** the east block (`MOVE S 32` at (`$A0`,`$75`))
    when Link starts below full health. At full health it loops under the diamond, and 25% of evaluation episodes
    (argmax) die.
- **Unwanted clips under `long`/`edge`:**
  - `long`: none. Every clip is the intended one, and no run used `MOVE N 32` at the push spot.
  - `edge`: seed 1 of `edge-nocur` is clean. Seed 2 learned a north route, with clips from above the
    upper-east/east blocks (in the top traps' reach) and extra clips, 0.5–0.7 per episode. In 3% of its argmax
    episodes it ends in a 600-step loop inside the diamond, between two blocks next to the stairs.
- **Oddities:**
  - The PPO update after the last logged iteration changed two final models: `kind-nocur` s1 and `long-north` s1.
  - `triforce.explore --variant` drops its tracks when `--specialist` is a YAML path.
  - The `none` arm without a curriculum pushes the west block E in 18/300 sampled episodes (seed 2), which walls in
    the stairs.

## 1. Setup

Worktree at `376b6fd` (1.3.3.1) with its own module build; experiment commit on top. No code change was needed.

**Arms** (`experiments/r21/trap22-<action>-<curriculum>.yaml`):
- Each YAML is `specialists/trap-room.yaml` with only `(1, 0x22)` and `1_22e_[0-9]`. The rewards, observation, RAM
  randomization, `items.bombs: required`, `one_room` and `max_steps: 600` are the same.
- `clips: {action: <none|kind|long|edge>, shaping: jump}`.
- `curriculum`: `{enabled: false}` (`nocur`), or `{enabled: true, trajectories: trajectories/r21-trap22-south}`
  (`south`) or `…-north` (`north`). The curriculum `fraction` (0.5) and `threshold` (0.6) are the defaults.
- Ten arms: `none`, `kind`, `long` and `edge` × {`nocur`, `south`}, plus `kind-north` and `long-north`.

**States:** `experiments/r21/states/1_22e_{0,1,2}.state`, the round's copies of the route's 0x22 E-entry states.
Every tool takes `TRIFORCE_STATES_DIR=experiments/r21/states`.

**Curricula** (`scripts/experiments/r21_trap_room.sh tracks`):
- `python -m triforce.explore --specialist experiments/r21/trap22-none-nocur.yaml --out
  trajectories/r21-explore-s<seed> --seed <seed> --parallel 3`, for seeds 0–8. The explore YAML has
  `clips.action: none`, so the taught route is the push. One root per seed: see §5 for why `--variant` was not used.
- `scripts/experiments/r21_trap_tracks.py` loads every saved point of a track and prints Link's path round the
  diamond, the push direction and the clip zones on the track.
- Of 27 tracks, 20 go round the south and push the west block N from below. 7 go round the north (seed 5, seed 7's
  `1_22e_2`, seed 8) and push it S from above.
- `1_22e_0` and `1_22e_1` gave the same track for every explore seed (§5).
- **South set** (`trajectories/r21-trap22-south`): explore seed 3, `1_22e_0~s3` (35 steps), `1_22e_1~s3` (35) and
  `1_22e_2~s3` (40), all at health 4.0. Path:
  - door (`$E0`,`$8D`) → W to (`$B0`,`$8D`);
  - S down the east side to Y `$B5`;
  - W along Y `$B5` under the lower blocks;
  - up the west side to the push spot (`$60`,`$95`);
  - push N, then into the diamond.
  - The south track passes under the lower blocks at Y `$B5`, below their clip zones (`$A5`–`$B5` on X `$70`/`$90`
    is the zone; the track's saved points are not on those columns). The only clip-zone point it stands on is the
    push spot itself (`clip_usable_N` there).
- **North set** (`trajectories/r21-trap22-north`): explore seed 8, `1_22e_0~s8` (33 steps), `1_22e_1~s8` (33) and
  `1_22e_2~s8` (46), all at health 3.0. Path:
  - door → (`$C8`,`$8D`);
  - N to Y `$5D`;
  - W along Y `$5D` above the upper blocks;
  - to (`$60`,`$5D`), S to the push spot (`$60`,`$7D`);
  - push S, then in.
  - The only clip-zone point it stands on is that push spot (`clip_usable_S`).
- `run.json` `curriculum_tracks` of every `south` run lists the three `~s3` tracks, and every `north` run the three
  `~s8` tracks. The `nocur` runs have none.

**Training** (`scripts/experiments/r21_trap_room.sh train ARM SEED`):

```bash
TRIFORCE_STATES_DIR=experiments/r21/states .venv/bin/python -m triforce.train \
    --specialist experiments/r21/trap22-ARM.yaml --run-id r21-sSEED --seed SEED --iterations 98 \
    --steps-per-update 4096 --parallel 2 --device cuda --models-dir models --logs-dir logs
```

- These are `train_round.py`'s settings without its early stop (`--stop-success 0.95 --stop-patience 5`). The other
  hyperparameters are the defaults (`run.json` `args`/`ppo_config`): lr 1e-4, entropy 0.01, no SIL.
- Budget: 98 iterations × 4096 = **401,408 env steps per run**, from scratch, with no early stop.
- Throughput was 215 steps/s for each of two runs side by side at first; later runs reached 330–440 steps/s as the
  machine's load fell. A run took 15–31 min.
- How the budget was chosen: round 14's trap-room (all its rooms, 6 tracks) needed 0.4M steps. Here `none-south`
  reached 90% at 98–115k steps, a quarter of the budget.
- Seeds 1 and 2 for all ten arms (20 runs), at most two runs at once.

**Evaluation** (`… eval ARM SEED`): `python -m triforce.evaluate --specialist experiments/r21/trap22-ARM.yaml
--checkpoint models/trap22-ARM/r21-sSEED --act {sample,argmax} --episodes-per-state 100 --parallel 2
--output <run>/eval-<act>.json --replays <run>/replays-<act> --streams decisions,rewards`.
- 300 episodes per act mode, eval seed 0, on the final checkpoint.

**Analysis** (`scripts/experiments/r21_trap_analyze.py RUN_DIR`, tables with `--table models`):
- Every evaluation replay is played on the emulator to Link's position at each decision.
- A completed clip leaves Link exactly 8 px into one of the diamond's eight block squares. No walk stops there, and
  the west square only counts before that block moved.
- Each clip gets its block, the spot it started from, the action and what followed. The number placed is capped by
  the executor's own count (`clips.done` in the record).
- A push is the `reward-pushed-block` term, or Link passing the west block's column when the stairs came before the
  term.
- A W/E push is a horizontal move that ends in the west block's square before any N/S push.
- Learning comes from `metrics.jsonl`:
  - success counts seed-start episodes only (not curriculum starts);
  - push share = `reward/reward-pushed-block` / 0.75 (an estimate: the term is halved on a step with damage);
  - clips per episode = `clips/done`.

Rerun everything: `scripts/experiments/r21_trap_room.sh tracks`, then `… all 1` and `… all 2`, then
`python scripts/experiments/r21_trap_analyze.py --table models`. `trajectories/`, `models/` and `logs/` are
gitignored.

## 2. Results

### Evaluation (300 episodes per row: 100 per start state)

"hearts" is the mean health change. "steps" and "frames" are per episode; "win frames" are over successful
episodes. "wins by clip / push" is the share of successful episodes that entered the diamond by a clip or by the
push. "clips aborted" is per episode.

| arm | seed | act | success | hearts | steps | frames | win frames | clips done | clips aborted | wins by clip / push | endings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none-nocur | 1 | sample | 0% | -0.37 | 534 | 12455 | - | 0.00 | 0.00 | - / - | no-damage 180, stuck 76, max-steps 34, death 10 |
| none-nocur | 1 | argmax | 0% | +0.00 | 56 | 1515 | - | 0.00 | 0.00 | - / - | stuck 300 |
| none-nocur | 2 | sample | 86% | -1.17 | 165 | 1588 | 1185 | 0.00 | 0.00 | 0% / 100% | exit-correct 259, death 21, max-steps 18, no-damage 2 |
| none-nocur | 2 | argmax | 0% | -0.03 | 572 | 2606 | - | 0.00 | 0.00 | - / - | no-damage 282, stuck 17, max-steps 1 |
| none-south | 1 | sample | 100% | -0.13 | 34 | 382 | 382 | 0.00 | 0.00 | 0% / 100% | exit-correct 300 |
| none-south | 1 | argmax | 100% | +0.00 | 36 | 367 | 367 | 0.00 | 0.00 | 0% / 100% | exit-correct 300 |
| none-south | 2 | sample | 100% | -0.04 | 39 | 377 | 377 | 0.00 | 0.00 | 0% / 100% | exit-correct 300 |
| none-south | 2 | argmax | 100% | +0.00 | 43 | 372 | 372 | 0.00 | 0.00 | 0% / 100% | exit-correct 300 |
| kind-nocur | 1 | sample | 95% | -0.11 | 9 | 234 | 225 | 0.98 | 0.00 | 100% / 0% | exit-correct 285, death 15 |
| kind-nocur | 1 | argmax | 100% | +0.00 | 7 | 219 | 219 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| kind-nocur | 2 | sample | 100% | +0.00 | 7 | 216 | 216 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| kind-nocur | 2 | argmax | 100% | +0.00 | 7 | 216 | 216 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| kind-south | 1 | sample | 100% | -0.00 | 7 | 222 | 222 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| kind-south | 1 | argmax | 100% | +0.00 | 7 | 216 | 216 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| kind-south | 2 | sample | 100% | +0.00 | 8 | 220 | 220 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| kind-south | 2 | argmax | 100% | +0.00 | 7 | 219 | 219 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| kind-north | 1 | sample | 100% | +0.00 | 7 | 218 | 218 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| kind-north | 1 | argmax | 100% | +0.00 | 7 | 217 | 217 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| kind-north | 2 | sample | 100% | +0.00 | 7 | 219 | 219 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| kind-north | 2 | argmax | 100% | +0.00 | 7 | 219 | 219 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| long-nocur | 1 | sample | 100% | +0.00 | 7 | 219 | 219 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| long-nocur | 1 | argmax | 100% | +0.00 | 7 | 219 | 219 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| long-nocur | 2 | sample | 100% | +0.00 | 7 | 218 | 218 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| long-nocur | 2 | argmax | 100% | +0.00 | 7 | 217 | 217 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| long-south | 1 | sample | 100% | +0.00 | 7 | 217 | 217 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| long-south | 1 | argmax | 100% | +0.00 | 7 | 217 | 217 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| long-south | 2 | sample | 100% | +0.00 | 7 | 217 | 217 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| long-south | 2 | argmax | 100% | +0.00 | 7 | 217 | 217 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| long-north | 1 | sample | 82% | -1.19 | 23 | 374 | 299 | 0.84 | 0.01 | 100% / 0% | exit-correct 245, death 55 |
| long-north | 1 | argmax | 75% | -1.23 | 24 | 385 | 246 | 0.75 | 0.00 | 100% / 0% | exit-correct 226, death 74 |
| long-north | 2 | sample | 100% | +0.00 | 7 | 220 | 220 | 1.01 | 0.00 | 100% / 0% | exit-correct 300 |
| long-north | 2 | argmax | 100% | +0.00 | 7 | 217 | 217 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| edge-nocur | 1 | sample | 100% | +0.00 | 7 | 217 | 217 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| edge-nocur | 1 | argmax | 100% | +0.00 | 7 | 217 | 217 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| edge-nocur | 2 | sample | 94% | -0.34 | 15 | 346 | 338 | 1.48 | 0.03 | 100% / 0% | exit-correct 283, death 17 |
| edge-nocur | 2 | argmax | 95% | -0.04 | 28 | 463 | 245 | 1.73 | 0.00 | 100% / 0% | exit-correct 285, no-damage 9, death 6 |
| edge-south | 1 | sample | 100% | -0.01 | 9 | 239 | 239 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| edge-south | 1 | argmax | 100% | +0.00 | 9 | 235 | 235 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| edge-south | 2 | sample | 100% | +0.00 | 7 | 216 | 216 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |
| edge-south | 2 | argmax | 100% | +0.00 | 7 | 216 | 216 | 1.00 | 0.00 | 100% / 0% | exit-correct 300 |

Frames are counted by the evaluation from the loaded start state. The scripted route of `crates/zelda/tests/clips.rs`
(8 actions, 218 frames) counts from the room's first frame. The two are close but not the same measure.

### Learning (from `metrics.jsonl`; steps are cumulative env steps, 4096 per iteration)

- 50/80/90%: the first iteration whose seed-start success reached that level.
- "clip replaces push": the first iteration from which clips per episode stay above both the push share and 0.5
  until the end.
- "first push": the first iteration with a push share of at least 0.05.
- Iteration 1 is a random policy. Its clips per episode: `kind` 0.32–0.70, `long` 0.04–0.07, `edge` 0.08–0.38.
  Its success: 13–16% for `kind`, 2–5% for `long`, 8–10% for `edge`, 0–1% for `none`. So "first clip" is 4k for
  every clip arm (8k for `long-south` s2) and is left out of the table.

| arm | seed | 50% | 80% | 90% | clip replaces push | first push | push peak (clips then) | last 10 it. success / clips / push |
|---|---|---|---|---|---|---|---|---|
| none-nocur | 1 | not reached | not reached | not reached | - | not reached | 0.05 at 74k | 0% / 0.00 / 0.00 |
| none-nocur | 2 | 328k | 385k | not reached | - | 61k | 0.89 at 401k | 52% / 0.00 / 0.55 |
| none-south | 1 | 86k | 102k | 115k | - | 66k | 1.00 at 348k | 99% / 0.00 / 0.98 |
| none-south | 2 | 82k | 94k | 98k | - | 33k | 0.98 at 360k | 99% / 0.00 / 0.94 |
| kind-nocur | 1 | 41k | 86k | 94k | 12k | not reached | 0.02 at 16k (0.90) | 97% / 0.97 / 0.00 |
| kind-nocur | 2 | 33k | 66k | 70k | 12k | not reached | 0.02 at 20k (0.95) | 100% / 1.00 / 0.00 |
| kind-south | 1 | 41k | 57k | 66k | 20k | 45k | 0.10 at 49k (1.56) | 98% / 1.31 / 0.00 |
| kind-south | 2 | 33k | 53k | 66k | 8k | 49k | 0.06 at 49k (1.73) | 100% / 1.00 / 0.00 |
| kind-north | 1 | 45k | 57k | 74k | 29k | not reached | 0.04 at 37k (1.12) | 100% / 1.00 / 0.00 |
| kind-north | 2 | 29k | 57k | 57k | 8k | not reached | 0.05 at 41k (1.52) | 100% / 1.00 / 0.00 |
| long-nocur | 1 | 78k | 90k | 98k | 70k | not reached | 0.02 at 90k (1.02) | 100% / 1.00 / 0.00 |
| long-nocur | 2 | 70k | 90k | 98k | 57k | not reached | 0.05 at 86k (1.37) | 100% / 1.00 / 0.00 |
| long-south | 1 | 74k | 78k | 86k | 66k | 66k | 0.21 at 78k (1.07) | 100% / 1.00 / 0.00 |
| long-south | 2 | 61k | 66k | 78k | 45k | 33k | 0.13 at 66k (1.17) | 100% / 1.00 / 0.00 |
| long-north | 1 | 57k | 78k | 78k | 53k | 61k | 0.06 at 70k (0.98) | 100% / 1.00 / 0.00 |
| long-north | 2 | 53k | 70k | 74k | 49k | 45k | 0.16 at 66k (1.04) | 100% / 1.00 / 0.00 |
| edge-nocur | 1 | 66k | 82k | 90k | 16k | not reached | 0.02 at 86k (1.19) | 100% / 1.00 / 0.00 |
| edge-nocur | 2 | 66k | 74k | 78k | 16k | not reached | 0.02 at 86k (1.22) | 99% / 1.03 / 0.00 |
| edge-south | 1 | 41k | 70k | 74k | 29k | 57k | 0.08 at 57k (1.22) | 100% / 1.02 / 0.00 |
| edge-south | 2 | 49k | 61k | 70k | 29k | 57k | 0.09 at 61k (1.59) | 100% / 1.00 / 0.00 |

### Behaviour (evaluation replays; counts are clips over the 300 episodes)

- "extra clips": clips other than the last clip of an episode that succeeded by clipping.
- "at push spot": clips started on (`$60`,`$95`) or (`$60`,`$85`).
- "approach": the region Link was in when he clipped or entered the diamond.
- "hit episodes": episodes in which Link took damage.

| arm | seed | act | clipped block (action) | from | extra clips | at push spot | approach | hit episodes | pushed N/S | pushed W/E |
|---|---|---|---|---|---|---|---|---|---|---|
| none-nocur | 1 | sample | - | - | 0 | 0 | S 187, SE 69, E 29 | 19% | 0% | - |
| none-nocur | 1 | argmax | - | - | 0 | 0 | E 300 | 0% | 0% | - |
| none-nocur | 2 | sample | - | - | 0 | 0 | W 261, NW 19, S 12 | 77% | 86% | E 18 |
| none-nocur | 2 | argmax | - | - | 0 | 0 | S 247, SE 28, W 25 | 3% | 0% | - |
| none-south | 1 | sample | - | - | 0 | 0 | W 293, NW 7 | 12% | 98% | - |
| none-south | 1 | argmax | - | - | 0 | 0 | W 300 | 0% | 100% | - |
| none-south | 2 | sample | - | - | 0 | 0 | W 294, NW 6 | 4% | 98% | - |
| none-south | 2 | argmax | - | - | 0 | 0 | W 300 | 0% | 100% | - |
| kind-nocur | 1 | sample | east CLIP N 288, lower-east CLIP N 5, east CLIP S 1 | east@(a0,9d) 285, lower-east@(90,a5) 5, east@(a0,a1) 2 | 10 | 0 | SE 282, NE 10, S 5 | 7% | 0% | - |
| kind-nocur | 1 | argmax | east CLIP N 300 | east@(a0,9d) 300 | 0 | 0 | SE 300 | 0% | 0% | - |
| kind-nocur | 2 | both | east CLIP N 300 | east@(a0,9d) 300 | 0 | 0 | SE 300 | 0% | 0% | - |
| kind-south | 1 | sample | east CLIP N 291, lower-east CLIP N 9, east CLIP S 1 | east@(a0,9d) 285, east@(a0,95) 6, lower-east@(90,a5) 5 | 1 | 0 | SE 285, S 9, E 6 | 0% | 0% | - |
| kind-south | 1, 2 | argmax | east CLIP N 300 | east@(a0,9d) 300 | 0 | 0 | SE 300 | 0% | 0% | - |
| kind-south | 2 | sample | east CLIP N 300 | east@(a0,9d) 297, east@(a0,95) 3 | 0 | 0 | SE 297, E 3 | 0% | 0% | - |
| kind-north | 1 | sample | east CLIP N 297, lower-east CLIP N 3 | east@(a0,9d) 297, lower-east@(90,a5) 3 | 0 | 0 | SE 297, S 3 | 0% | 0% | - |
| kind-north | 1, 2 | argmax (2: both) | east CLIP N 300 | east@(a0,9d) 300 | 0 | 0 | SE 300 | 0% | 0% | - |
| long-nocur | 1, 2 | both | east MOVE N 32 299–300 (s1 sample: 1 lower-east) | east@(a0,9d) 298–300 | 0 | 0 | SE 298–300 | 0% | 0% | - |
| long-south | 1, 2 | both | east MOVE N 32 300 | east@(a0,9d) 298–300 | 0 | 0 | SE 298–300 | 0% | 0% | - |
| long-north | 1 | sample | east MOVE S 32 204, lower-east MOVE N 32 45, east MOVE N 32 2 | east@(a0,75) 202, lower-east@(90,a5) 31, lower-east@(90,ad) 7 | 7 | 0 | E 205, S 76, SW 15 | 30% | 0% | - |
| long-north | 1 | argmax | east MOVE S 32 226 | east@(a0,75) 226 | 0 | 0 | E 226, S 74 | 25% | 0% | - |
| long-north | 2 | both | east MOVE N 32 300–302 (sample: 2 MOVE S 32) | east@(a0,9d) 298–300 | 0–4 | 0 | SE 298–300 | 0% | 0% | - |
| edge-nocur | 1 | sample | east MOVE N 8 193, east MOVE N 32 58, east MOVE N 4 30 | east@(a0,9d) 299, lower-east@(90,a5) 1 | 0 | 0 | SE 299, S 1 | 0% | 0% | - |
| edge-nocur | 1 | argmax | east MOVE N 8 200, east MOVE N 32 100 | east@(a0,9d) 300 | 0 | 0 | SE 300 | 0% | 0% | - |
| edge-nocur | 2 | sample | east MOVE N 8 139, east MOVE N 32 77, upper-east MOVE S 16 60 | east@(a0,9d) 181, upper-east@(90,6d) 80, north@(80,75) 57 | 161 | 0 | SE 177, E 67, N 45 | 19% | 0% | - |
| edge-nocur | 2 | argmax | east MOVE N 8 169, upper-east MOVE S 16 129, east MOVE N 32 116 | east@(a0,9d) 280, upper-east@(90,6d) 124, east@(a0,7d) 103 | 233 | 0 | SE 280, E 13, NE 7 | 4% | 0% | - |
| edge-south | 1 | sample | lower-east MOVE N 8 133, east MOVE N 8 125, lower-east MOVE N 16 18 | lower-east@(90,ad) 151, east@(a0,9d) 136, lower-east@(90,a5) 12 | 1 | 0 | S 164, SE 135, E 1 | 1% | 0% | - |
| edge-south | 1 | argmax | lower-east MOVE N 8 189, east MOVE N 8 111 | lower-east@(90,ad) 189, east@(a0,9d) 111 | 0 | 0 | S 189, SE 111 | 0% | 0% | - |
| edge-south | 2 | both | east MOVE N 8 154–236, MOVE N 16 64–83, MOVE N 32 0–52 | east@(a0,9d) 299–300 | 0 | 0 | SE 299–300 | 0% | 0% | - |

## 3. Observations

### The route the clip arms learned

Every clip run's argmax episode on `1_22e_0` takes 7 decisions:
1. `MOVE W` (4–32) from the door (`$E0`,`$8D`);
2. `MOVE W 32`;
3. `MOVE W` to (`$B0`,`$8D`);
4. `MOVE S 8` to (`$B0`,`$99`);
5. `MOVE W` (16 or 32) to (`$A0`,`$9D`), under the east block, inside its N zone (Y `$95`–`$A5`);
6. the clip: `CLIP N`, `MOVE N 32` or `MOVE N 8`;
7. `MOVE W 32` from (`$A0`,`$8D`) onto the stairs.

This is the shape of the test route in `crates/zelda/tests/clips.rs` with one action fewer (216–219 evaluation
frames). The other routes seen:
- `edge-south` s1 also clips into the lower-east block from (`$90`,`$AD`) (189 of 300 argmax episodes, 9 decisions,
  246 frames).
- `long-north` s1 clips S from above the east block (`$A0`,`$75`): 10 decisions, 246 frames, through the top traps'
  rows.
- `edge-nocur` s2 approaches from the north (below).

Of about 20 sampled evaluation episodes per arm (every 15th, in `analysis.txt` of each run), the clip runs show the
same route in nearly every one. Under argmax, traps hit Link in no episode of 16 of the 18 clip runs; the exceptions
are `long-north` s1 (25%) and `edge-nocur` s2 (4%). Sampled, they also hit him in `kind-nocur` s1 (7%),
`long-north` s1 (30%), `edge-nocur` s2 (19%) and `edge-south` s1 (1%). No clip run pushes the block in evaluation
(0 pushes in 10,800 episodes).

### The hypothesis (south curriculum → clip found)

- **Whether the clip is found:**
  - Without a curriculum, the random policy already clips (iteration 1: 0.3–0.7 clips per episode under `kind`,
    0.05–0.07 under `long`, 0.1–0.4 under `edge`).
  - It already succeeds 2–16% of the time (`none`: 0–1%).
  - So the clip is found from the first iteration in every arm, and no curriculum is needed for it.
  - The push never is without one (`none-nocur`).
- **How fast:**
  - `long`: 90% at 98k/98k (nocur), 86k/78k (south), 78k/74k (north).
  - `kind`: 94k/70k, 66k/66k, 74k/57k.
  - `edge`: 90k/78k, 74k/70k.
  - The curricula shift 90% earlier by about 10–25k steps in most pairs. With two seeds that is within the spread
    of `kind-nocur` alone (94k vs 70k).
- **What the curriculum adds:**
  - The south curriculum produces a measurable push phase. Push share peaks at 0.06–0.21 at 33–78k steps in
    `kind-south`, `long-south` and `edge-south`, against ≤ 0.05 in the `nocur` and `kind-north` arms.
  - The clip replaces it by 45–66k (`long-south`), 8–20k (`kind-south`) or 29k (`edge-south`).
  - So the curriculum taught some pushing, which faded: under `kind` and `edge` while the clip already dominated,
    under `long` as the clip took over.
- **Does a north curriculum make the agent wander south first?** No sign of it:
  - `kind-north` (both seeds) and `long-north` s2 learned the south clip under the east block, as fast as or faster
    than `south` (90% at 57–78k).
  - `long-north` s1 is the only run that learned a different clip: from above the east block, the reverse clip
    `MOVE S 32` at (`$A0`,`$75`). Its north track passes (`$C8`,`$5D`) → (`$A4`,`$5D`) above that block.
  - That model succeeds in 100% of its partial-health starts and dies in about half of its full-health starts
    (all 74 argmax deaths start at 5.0 hearts). At full health it goes S at (`$B0`,`$8D`) and loops under the
    diamond until the traps kill it.

### Unwanted clips under `long` and `edge`

- **`long`:**
  - Every evaluation clip is `MOVE N 32`, or `MOVE S 32` for `long-north` s1, into a block that leads to the stairs.
  - There are no clips at the push spot and no extra clips, except 7 in `long-north` s1 sample and 4 in s2 sample.
  - `MOVE N 16`/`MOVE N 8` are used for walking: the 32 is reserved for the clip.
- **`edge`:**
  - Every MOVE N or S into a square from its zone clips, so the policy clips with whatever duration it picks:
    `MOVE N 8` mostly, then 32, 16 and 4.
  - `edge-nocur` s1 and both `edge-south` seeds have no extra clips and no push.
  - `edge-nocur` s2 has 161 (sample) and 233 (argmax) extra clips. It learned a north approach:
    - N on the east side to (`$B0`,`$5D`), W above the upper-east block;
    - `MOVE S 16` at (`$90`,`$6D`) clips into the upper-east block;
    - back out, then `MOVE S 16` at (`$A0`,`$7D`) clips into the east block from above.
  - In 9 argmax episodes it ends `no-damage` after 600 steps, oscillating inside the diamond between (`$90`,`$85`)
    and (`$90`,`$8D`) with `MOVE N 16`/`MOVE S 32`. Those are plain moves, not clips: there it stands between the
    upper-east and lower-east blocks, one `MOVE W 16` from the stairs.
  - Sampled, it also clips from inside the diamond into the north block (57 clips from (`$80`,`$75`)). That clip
    leads out of the diamond, so it is a step backwards.
- **Push spots:** no clip arm ever stood on a push spot and clipped there; the learned routes never go near the
  west block. So the `edge` push-spot conflict never came up in evaluation (`docs/architecture.md` "Grid-point
  skip": under `edge` the west block cannot be pushed from its push spots). The south tracks' curriculum starts do
  put Link on the push spot (`$60`,`$95`), where under `edge` every MOVE N clips. Yet the seed-start episodes of
  `edge-south` still showed a push phase (push share 0.08–0.09 at 57–61k). How those pushes were made under `edge`
  was not checked.

### `none` arms

- `none-south` learns the push route: 100% in all four evaluations.
- Its path is the curriculum's: round the south, up the west side to (`$60`,`$95`), push N with repeated `MOVE N`,
  `MOVE E 32` into the stairs.
- It takes 30–43 decisions and 367–382 frames, and is hit in 4–12% of sampled episodes and in 0% under argmax.
- `none-nocur` s1 learned nothing: argmax ends `stuck` in all 300 episodes; sampling ends `no-damage`, `stuck` or
  `max-steps`. This is round 14 again.
- `none-nocur` s2 found the push by itself around 61k steps and kept improving to the end (push share 0.89 at 401k):
  - sampling 86%, slow: 165 decisions, 1185 frames per win, hit in 77% of episodes;
  - argmax 0%: `no-damage` 282 (a soft-failure truncation). Link spends the 600 steps under the lower-east block
    at (`$90`,`$A5`–`$AD`), repeating `MOVE W 4`, `MOVE S 4`, `MOVE N 4`. That spot is that block's N clip zone,
    which leads nowhere without a clip;
  - in 18 of 300 sampled episodes it pushes the west block **E**, onto the stairs, and the room can no longer be
    solved (backlog: "0x22 pushed the wrong way").

## 4. Bugs and oddities

- **The final update changed two final models.** The evaluated checkpoint is saved after the update that follows
  the last logged iteration.
  - `kind-nocur` s1: seed-start success 1.0 through iteration 96, 0.956 at 97 and 0.748 at 98. The value loss
    jumped from 0.01 to 11.7 and the kind entropy rose. The final checkpoint still evaluates at 95% sample and
    100% argmax.
  - `long-north` s1: success 1.0 in each of the last 10 logged iterations, but the final checkpoint scores 82% sample
    and 75% argmax.
  - A probe ran the final `long-north` s1 checkpoint under training conditions with no learning
    (`--load … --learning-rate 0 --iterations 2 --no-curriculum`): 81.6% and 82.9% success. So the regression is in
    the checkpoint, not in a difference between evaluation and training.
  - Large single-iteration dips also happen mid-run, and recover: `long-north` s1 0.60 at iteration 75, `long-south`
    s1 0.86 at 75, `kind-south` s1 0.86 at 95, `none-south` s2 0.80 at 34.
  - A fixed budget with no best-checkpoint selection evaluates wherever the policy happens to be.
- **`triforce.explore --variant` loses its tracks when `--specialist` is a YAML path:**
  - `explore_specialist` looks for `<tmp>/<specialist argument>`, the path, while the native explorer writes
    `<tmp>/<config name>`;
  - the temporary directory is then deleted;
  - it still prints `3/3 states solved` with a target path under `<out>/experiments/r21/<file>.yaml/…`, which does
    not exist.
  - Without `--variant`, the track goes to `<out>/<config name>/<state>`, but the printed path is again the YAML
    path's.
  - Not fixed (not the clips code); worked around with one `--out` root per explore seed.
- **`1_22e_0` and `1_22e_1` explore identically:** the same track for each of the 9 explore seeds. The two files
  differ (md5), so [INFERENCE] they differ only in bytes the explorer's load does not use. Training's RAM
  randomization makes them different resets.
- **Analysis residue:** `none-nocur` s2 sample has 26 moments where a non-N/S action left Link exactly where a clip
  would (counted "knocked in", not as clips). In the smoke run, one such case was a trap's knockback from
  (`$60`,`$65`) to (`$60`,`$85`), 8 px into the west block from above. The 26 were not inspected one by one
  [unverified: knockback or a sideways push the detector missed].

## 5. Not done

- No `edge-north` or `none-north` arms (not asked).
- No `shaping: walk` arms.
- Only room 0x22 from E: no `1_22c` (back from the cellar), no other trap-room rooms, no chains.
- No best-checkpoint selection or continued training, given the final-update regressions above.
- `long-north` s1's full-health route was found from the replays only; its training-time route was not recorded.
- The rebuild of the tracks (`… tracks`) was not rerun to confirm the explorer reproduces them bit for bit.
  [INFERENCE] It should: the explorer is seeded and documented as independent of `--parallel`.
