# Round 21: the block clip in level 4 room 0x50 (vire)

> **Afterwards (Nov 13, `docs/decisions.md` 2026-11-12 "One clip design", 2026-11-13 "Round 21 merged"):** `clips:` is now a
> plain `true|false` switch; the four YAMLs say `clips: true` (`kind` arms) or `clips: false` (`none` arms), with
> what they ran as. The catch-pose stall under "Oddities" is fixed: the CLIP macro waits the pose out. This report
> describes the code as it ran.

Clips in the agent (branch `clips`, version 1.4.1.0, commit `ccaafe5`): the executor's block-clip macro, reached
through `clips: {action: none|kind|long|edge}` (`docs/architecture.md` "Grid-point skip"). This experiment trains
level 4 room 0x50 alone, from the E door to the N door. The room is dark and holds five Vires, and diagonal rows of
blocks force a zigzag. The route has the plain zigzag (14 MOVEs, 424 frames with the monsters removed) and the clip
route (`MOVE W 32` ×2 to (`$B0`,`$8D`), `CLIP N`, `MOVE W 8`, `MOVE N 32`, `MOVE W 16` ×2, `MOVE N 32`: 8 actions,
318 frames; `crates/zelda/tests/clips.rs` `the_way_through_4_50_by_one_block_clip`). In round 20's chains Link lost
0.73 hearts per visit here. The experiment was planned with all four designs. Partway through, the CLIP action kind
(`kind`) was chosen as the design, so only the `none` and `kind` arms were trained (§5). This report records what
happened. It declares no winner.

## Summary

- **`kind` learned the clip in all four runs (2 curriculum settings × 2 seeds).** 2,397 of the 2,400 evaluation
  episodes clip the same square, the block above (`$B0`,`$8D`) (`CLIP N`, landing on (`$B0`,`$7D`)), then walk W.
  That is the scripted route's clip. Of the other three:
  - two reached the same landing after a stalled clip (§4);
  - one clipped the top-right block from (`$D0`,`$67`).

  The clip route replaced the zigzag completely: no `kind` episode went south.
  - Success is 99–100% in both act modes.
  - Mean health change is −0.07 to −0.17 hearts.
  - Episodes take 13–16 decisions and 350–361 frames.
- **`none` learned the zigzag in all four runs:**
  - success 95–99%;
  - −0.48 to −0.61 hearts;
  - 34–42 decisions, 525–566 frames;
  - hit in 49–60% of the episodes (`kind`: 7–19%).
- **Where `kind` is hurt:** every hit in the `kind` runs lands after the clip, on the top row between X `$78` and
  `$A0` (Y `$5D`), the last stretch to the N door. No hit landed before or during a clip.
- **Learning:** with a 10-iteration window, `kind` reached 90% at 168–196k steps and 95% at 258–303k. `none` reached
  90% at 311–426k and 95% at 561–667k.
  - The curriculum (the explored zigzag) did not change `none`: 90% at 340k/356k with it, 311k/426k without.
  - It did not slow `kind` down either (90% at 168k/176k with it, 184k/196k without).
- **Clip use while learning:**
  - The untrained policy already clips: 0.8–8 clips per episode in iteration 1.
  - Early training clips a lot: a peak of 8–11 clips per episode at 25–45k steps, N and S alike, with episodes of
    270–800 steps.
  - From about 190–240k steps on, use is one clip per episode, N.
- **Fights or passes:** both designs mostly pass the Vires: 0.5–0.8 room kills per episode in every arm. `kind`
  attacks about 2 times per episode (0.5 of them bombs), `none` 5–7 times (0.6–1.1 bombs).
- **Oddities:**
  - 5 aborted clips in 2,400 `kind` episodes, none of them a hit. The 4 reproduced are `stalled`: Link stood still
    during the macro's run-up. One was traced to the boomerang coming back, which freezes Link for a few frames.
  - In episode 107 (both `kind` s1 runs, sampled) the stalled `CLIP N` left Link 6 px into the block, and the next
    plain `MOVE N 32` walked him on to the clip landing. Both episodes won by the clip route with no clip counted.
  - `none` argmax ends `stuck` in 3–8 of 300 episodes: Link presses N into the wall beside the N door, at (`$80`,`$5C`),
    (`$59`,`$5D`) or (`$99`,`$5D`).

## 1. Setup

Worktree at `ccaafe5` (1.4.1.0; every run stamped `1.4.1.0`) with its own module build. No code change was made.
Experiment commit on top: the YAMLs, the scripts and this report.

**Arms** (`experiments/r21/vire50-<action>-<curriculum>.yaml`):
- Each YAML is `specialists/vire.yaml` with only `(4, 0x50)` and `4_50e_[0-9]` (`triforce/states/4_50e_{0,1,2}`).
- Unchanged from vire: success (`exit-correct`, `triforce-collected`), `reward_profile: default`,
  `reward: {pbrs: true, item_targets: true}`, `obs: {objective: true, target: true}`, `ram_randomization` (hp 0.15–max,
  full hp 0.5, bombs 0–8 with share 0.5, help count 0–9), `items: {bombs: free}`, `one_room: true`,
  `max_steps: 2000`.
- `clips: {action: none|kind, shaping: jump}`.
- `curriculum`: `{enabled: false}` (`nocur`) or `{enabled: true, trajectories: trajectories/r21-vire50}` (`cur`;
  default fraction 0.5 and threshold 0.6).
- Four arms: `none-nocur`, `none-cur`, `kind-nocur`, `kind-cur`. Seeds 1 and 2: 8 runs.

**Curriculum tracks** (`scripts/experiments/r21_vire50.sh tracks`):
- `python -m triforce.explore --specialist experiments/r21/vire50-none-nocur.yaml --out trajectories/r21-vire50-explore
  --seed 0 --parallel 6`, then `trajectories/r21-vire50-explore/vire50-none-nocur` copied to `trajectories/r21-vire50`.
  The explore YAML has `clips.action: none`, so the tracks walk the plain route.
- The three tracks have the same `trajectory.json` (actions, steps, end health) as round 20's `trajectories/vire/4_50e_*`
  (also explore seed 0).
- `scripts/experiments/r21_vire50_tracks.py` loads each saved point and prints Link's path and the clip zones on it.
- `4_50e_0~s0`: 122 steps, end health 1.0. Path: door (`$E0`,`$8D`) → (`$B0`,`$8D`), S to (`$B0`,`$AD`), W to
  (`$90`,`$AD`), down to the bottom row (Y `$B5`), W along it to X `$30`. Then up the west side through (`$40`,`$A5`),
  (`$50`,`$95`), (`$68`,`$85`), E to (`$90`,`$85`), N up X `$90` to the top row, W to the door (`$78`,`$5D`).
- `4_50e_1~s0`: 110 steps, end health 6.0. Path: (`$B0`,`$8D`) with a detour up the east side to (`$C8`,`$75`). Then S
  to (`$90`,`$A5`) and the bottom row (Y `$B5`), W to X `$40`, then N through (`$58`,`$95`), (`$80`,`$95`),
  (`$80`,`$7D`), (`$A0`,`$7D`), N to the top row and W to the door.
- `4_50e_2~s0`: 122 steps, end health 6.5. Path: up the east side to (`$D0`,`$65`) and back first, then S to the bottom
  row (Y `$BD`) and W to X `$30`. Then back E and up through (`$60`,`$9F`), (`$80`,`$9D`), (`$70`,`$85`),
  (`$78`,`$75`), (`$90`,`$75`), N up X `$90` and W along the top row to the door.
- All three go round the south of the diagonals and come up through the middle: the zigzag, with long wanders and
  fights (0–14 SWORD, 0–9 BEAMS, 5–8 BOOMERANG, 0–1 BOMB per track). All three stand on the clip square's N zone at
  (`$B0`,`$85`)/(`$B0`,`$8D`) early on (`clip_usable_N`).
- `run.json` `curriculum_tracks` of every `cur` run lists `4_50e_0`, `4_50e_1` and `4_50e_2`. The `nocur` runs have
  none.

**Budget:**
- Throughput: three 15-iteration runs side by side (`none-cur`, `kind-cur`, `edge-nocur`, seed 9) at `--parallel 6`
  did 855–875 steps/s each.
- Pilot: `none-cur`, `none-nocur` and `kind-nocur` at seed 0, stopped at 507–528k steps (690–715 steps/s each).
  Their success over 10-iteration blocks reached 90% at 327k (`none-cur`), 409k (`none-nocur`) and 204k
  (`kind-nocur`). At the stop they stood at 93%, 91–92% and 99%.
- Chosen: **367 iterations of 4,092 env steps = 1,501,764 env steps per run** (`--steps-per-update 4096` rounds to
  682 steps × 6 envs). That is about 4.5× what `none-cur` needed for 90% in the pilot, from scratch, with no early
  stop.
- The 8 runs, three at a time, took 2 h 05 min with evaluation and analysis (19:59–22:04). Each run took 33–49 min
  (515–751 steps/s).

**Training** (`scripts/experiments/r21_vire50.sh train ARM SEED`): `train_round.py`'s command without its early stop
(`--stop-success 0.95 --stop-patience 5`), with round 19/20's `--sil-coeff 0.1`:

```bash
python -m triforce.train --specialist experiments/r21/vire50-ARM.yaml --run-id r21-sSEED --seed SEED \
    --iterations 367 --steps-per-update 4096 --parallel 6 --device cuda --sil-coeff 0.1 \
    --models-dir models --logs-dir logs
```

The other hyperparameters are the defaults (lr 1e-4, entropy 0.01). The first three runs were started with
`--events logs/r21-vire50/<arm>-s<seed>.jsonl`; the training GUI lists them through a hand-written
`logs/r21-vire50/training.json`. The other five made their own `logs/<yymmdd>_<NNN>_vire50-<arm>/`.

**Evaluation** (`… eval ARM SEED`):

```bash
python -m triforce.evaluate --specialist experiments/r21/vire50-ARM.yaml --checkpoint models/vire50-ARM/r21-sSEED \
    --act {sample,argmax} --episodes-per-state 100 --parallel 6 --no-log \
    --output <run>/eval-<act>.json --replays <run>/replays-<act> --streams decisions,rewards
```

300 episodes per act mode (100 per start state), eval seed 0, final checkpoint.

**Analysis** (`scripts/experiments/r21_vire50_analyze.py RUN_DIR`; tables: `… table`):
- Every evaluation replay is played on the emulator to Link's position, the monsters' positions and the room's kill
  count (`$34F`) at each decision.
- **Where clips land:** with the monsters removed, putting Link on each of the room's 60 offered grid points and
  pressing `CLIP N`/`CLIP S` gives 20 landings, 8 px into a block:
  - N: (`$40`,`$5D`), (`$40`,`$9D`), (`$50`,`$8D`), (`$60`,`$7D`), (`$70`,`$6D`), (`$90`,`$9D`), (`$A0`,`$8D`),
    (`$B0`,`$7D`), (`$C0`,`$6D`), (`$D0`,`$5D`);
  - S: (`$20`,`$B5`), (`$30`,`$A5`), (`$40`,`$95`), (`$50`,`$85`), (`$60`,`$75`), (`$80`,`$A5`), (`$90`,`$95`),
    (`$A0`,`$85`), (`$B0`,`$75`), (`$B0`,`$B5`).
- **What counts as a clip:** a `CLIP`/`MOVE` N or S that ends on one of these landings, starting on that block's
  side within 0x20 px. The skip is also offered between grid points (the macro's run-up walks to one first). No
  walk ends inside a block. The number placed is capped by the executor's count (`clips.done`); placed = counted in
  every run.
- **Per clip:** "on" means the next move went sideways the way on (W after N, E after S); "back" and "reverse" are
  the other cases.
- **"went S (zigzag)":** Link reached Y ≥ `$A5` at X ≥ `$80`, as the zigzag does on its way down the east side.
- **Hits:** the `penalty-lost-health` terms. Each is placed at Link's position when its decision began, and before,
  during or after the episode's first clip.
- **"frames near a monster":** frames of decisions that began with a live monster (Vire or Red Keese) within
  `$18` px on both axes. The last decision (the exit walk and scroll) is left out.
- **Kills:** the room's kill count (`$34F`), Red Keese included. **Attacks:** SWORD, BEAMS, BOMB and BOOMERANG
  decisions (bombs: the evaluation's `mean_bombs_used`).
- **Learning** comes from `metrics.jsonl` (room-entry episodes only; curriculum starts are left out). Each value is
  the mean of the last 10 iterations weighted by their episodes. A single iteration holds 1–302 episodes, too noisy
  alone.

Rerun everything: `scripts/experiments/r21_vire50.sh tracks`, then `… all` (8 jobs, `JOBS=3` at a time), then
`… table`. `trajectories/`, `models/` and `logs/` are gitignored.

## 2. Results

### Evaluation (300 episodes per row: 100 per start state)

"hearts" is the mean health change. "steps" and "frames" are per episode; "win frames" are over successful episodes.
"clips aborted" is per episode. "kills" is the room's kill count at the end.

| arm | seed | act | success | hearts | steps | frames | win frames | clips / ep | ep. with clip | clips aborted | kills / ep | attacks / ep (bombs) | endings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none-nocur | 1 | sample | 97% | −0.57 | 38 | 554 | 559 | 0.00 | 0% | 0.00 | 0.76 | 6.0 (0.93) | exit-correct 291, death 7, stuck 2 |
| none-nocur | 1 | argmax | 95% | −0.52 | 42 | 548 | 533 | 0.00 | 0% | 0.00 | 0.76 | 5.2 (0.81) | exit-correct 286, stuck 8, death 3, no-damage 3 |
| none-nocur | 2 | sample | 98% | −0.57 | 39 | 565 | 568 | 0.00 | 0% | 0.00 | 0.67 | 6.4 (1.08) | exit-correct 294, death 6 |
| none-nocur | 2 | argmax | 97% | −0.50 | 39 | 542 | 540 | 0.00 | 0% | 0.00 | 0.65 | 5.6 (0.89) | exit-correct 290, death 6, stuck 3, no-damage 1 |
| none-cur | 1 | sample | 98% | −0.60 | 38 | 566 | 570 | 0.00 | 0% | 0.00 | 0.67 | 6.7 (1.04) | exit-correct 294, death 6 |
| none-cur | 1 | argmax | 98% | −0.61 | 34 | 534 | 537 | 0.00 | 0% | 0.00 | 0.70 | 5.4 (0.82) | exit-correct 294, death 3, stuck 3 |
| none-cur | 2 | sample | 99% | −0.48 | 37 | 541 | 543 | 0.00 | 0% | 0.00 | 0.68 | 5.5 (0.85) | exit-correct 296, death 4 |
| none-cur | 2 | argmax | 97% | −0.50 | 38 | 525 | 528 | 0.00 | 0% | 0.00 | 0.50 | 5.2 (0.57) | exit-correct 290, stuck 7, death 3 |
| kind-nocur | 1 | sample | 100% | −0.15 | 15 | 356 | 356 | 1.00 | 100% | 0.00 | 0.57 | 2.0 (0.56) | exit-correct 299, death 1 |
| kind-nocur | 1 | argmax | 100% | −0.17 | 14 | 351 | 352 | 1.00 | 100% | 0.00 | 0.58 | 1.9 (0.52) | exit-correct 299, death 1 |
| kind-nocur | 2 | sample | 99% | −0.14 | 16 | 359 | 360 | 1.00 | 100% | 0.00 | 0.64 | 2.2 (0.55) | exit-correct 298, death 2 |
| kind-nocur | 2 | argmax | 100% | −0.07 | 15 | 354 | 354 | 1.00 | 100% | 0.00 | 0.72 | 2.1 (0.54) | exit-correct 299, death 1 |
| kind-cur | 1 | sample | 100% | −0.17 | 16 | 361 | 362 | 1.00 | 100% | 0.01 | 0.58 | 2.4 (0.56) | exit-correct 299, death 1 |
| kind-cur | 1 | argmax | 99% | −0.13 | 15 | 352 | 353 | 1.00 | 100% | 0.00 | 0.60 | 2.0 (0.54) | exit-correct 298, death 2 |
| kind-cur | 2 | sample | 100% | −0.12 | 14 | 356 | 356 | 1.00 | 100% | 0.00 | 0.55 | 2.0 (0.51) | exit-correct 300 |
| kind-cur | 2 | argmax | 100% | −0.09 | 13 | 350 | 351 | 1.00 | 100% | 0.00 | 0.57 | 1.9 (0.44) | exit-correct 299, death 1 |

The evaluation counts frames from the loaded start state. The scripted routes (424 and 318 frames) count from the
room's first frame, with the monsters removed. The two are close but not the same measure.

### Learning (from `metrics.jsonl`, 10-iteration window; steps are cumulative env steps)

- 50/80/90/95%: the first iteration whose windowed success reached that level.
- "clips / ep at it. 1": the untrained policy.
- "clip peak": the highest windowed clips per episode.
- "clip use settles": the first iteration from which the windowed clips per episode stay within 0.1 of the final
  window's.

| arm | seed | 50% | 80% | 90% | 95% | success at 250k / 500k / 1M | clips / ep at it. 1 | clip peak | clip use settles | last 10 it. success / clips / hearts |
|---|---|---|---|---|---|---|---|---|---|---|
| none-nocur | 1 | 106k | 192k | 311k | 581k | 85% / 92% / 98% | 0.00 | - | - | 98% / 0.00 / −0.58 |
| none-nocur | 2 | 98k | 209k | 426k | 667k | 82% / 89% / 98% | 0.00 | - | - | 98% / 0.00 / −0.63 |
| none-cur | 1 | 123k | 209k | 356k | 618k | 84% / 95% / 96% | 0.00 | - | - | 98% / 0.00 / −0.61 |
| none-cur | 2 | 74k | 188k | 340k | 561k | 87% / 92% / 98% | 0.00 | - | - | 98% / 0.00 / −0.60 |
| kind-nocur | 1 | 65k | 119k | 196k | 303k | 92% / 98% / 100% | 0.79 | 8.3 at 45k | 237k | 100% / 1.00 / −0.15 |
| kind-nocur | 2 | 65k | 131k | 184k | 258k | 95% / 99% / 100% | 1.32 | 10.6 at 45k | 221k | 100% / 1.00 / −0.13 |
| kind-cur | 1 | 53k | 106k | 168k | 262k | 95% / 99% / 100% | 2.77 | 8.1 at 25k | 221k | 100% / 1.00 / −0.16 |
| kind-cur | 2 | 70k | 115k | 176k | 258k | 95% / 98% / 100% | 8.00 | 8.6 at 25k | 192k | 100% / 1.00 / −0.10 |

The single iterations of `kind`, for the early clipping (`clips/done`, `done_n`/`done_s`, episode steps):
- `kind-nocur` s2 at 25k: 24.7 clips per episode (10.5 N, 14.2 S), 796 steps per episode.
- At 126k: 1.7 clips (1.1 N, 0.6 S), 99 steps.
- At 331k: 1.03 clips (1.01 N), 27 steps.
- The other `kind` runs follow the same shape: 8–13 clips per episode at 12–36k, then one N clip from about 190k on.

### Behaviour (evaluation replays; counts are clips or hits over the 300 episodes)

| arm | seed | act | clipped square (action) | from | after the clip | extra clips | wins by clip | went S (zigzag) | hit episodes | frames near a monster (share) |
|---|---|---|---|---|---|---|---|---|---|---|
| none-nocur | 1 | sample | - | - | - | 0 | 0% | 100% | 58% | 155 (41%) |
| none-nocur | 1 | argmax | - | - | - | 0 | 0% | 100% | 54% | 161 (40%) |
| none-nocur | 2 | sample | - | - | - | 0 | 0% | 100% | 54% | 156 (41%) |
| none-nocur | 2 | argmax | - | - | - | 0 | 0% | 100% | 50% | 151 (41%) |
| none-cur | 1 | sample | - | - | - | 0 | 0% | 100% | 59% | 153 (40%) |
| none-cur | 1 | argmax | - | - | - | 0 | 0% | 100% | 60% | 143 (40%) |
| none-cur | 2 | sample | - | - | - | 0 | 0% | 100% | 49% | 139 (39%) |
| none-cur | 2 | argmax | - | - | - | 0 | 0% | 100% | 54% | 134 (39%) |
| kind-nocur | 1 | sample | N(b0,7d) CLIP N 299 | (b0,8d) 299 | on 299 | 0 | 100% | 0% | 19% | 65 (38%) |
| kind-nocur | 1 | argmax | N(b0,7d) CLIP N 300 | (b0,8d) 300 | on 300 | 0 | 100% | 0% | 18% | 62 (38%) |
| kind-nocur | 2 | sample | N(b0,7d) CLIP N 300 | (b0,8d) 300 | on 300 | 0 | 100% | 0% | 13% | 72 (40%) |
| kind-nocur | 2 | argmax | N(b0,7d) CLIP N 300 | (b0,8d) 300 | on 300 | 0 | 100% | 0% | 7% | 68 (40%) |
| kind-cur | 1 | sample | N(b0,7d) CLIP N 298, N(d0,5d) CLIP N 1 | (b0,8d) 298, (d0,67) 1 | on 299 | 0 | 100% | 0% | 17% | 65 (37%) |
| kind-cur | 1 | argmax | N(b0,7d) CLIP N 300 | (b0,8d) 300 | on 300 | 0 | 100% | 0% | 15% | 60 (36%) |
| kind-cur | 2 | sample | N(b0,7d) CLIP N 300 | (b0,8d) 300 | on 300 | 0 | 100% | 0% | 12% | 68 (40%) |
| kind-cur | 2 | argmax | N(b0,7d) CLIP N 300 | (b0,8d) 300 | on 300 | 0 | 100% | 0% | 10% | 68 (41%) |

"extra clips": an episode's second and later clips.

| arm | seed | act | hits / ep | hits before / during / after the clip (no clip) | where (Link at the decision) | success full / partial health |
|---|---|---|---|---|---|---|
| none-nocur | 1 | sample | 0.81 | 244 | (a0,9d) 11, (80,64) 6, (78,75) 6, (80,5d) 5 | 99% / 95% |
| none-nocur | 1 | argmax | 0.75 | 224 | (b0,99) 13, (80,5c) 12, (a0,9d) 9, (78,75) 9 | 98% / 93% |
| none-nocur | 2 | sample | 0.79 | 238 | (78,75) 10, (80,5d) 8, (70,75) 7, (78,5d) 7 | 100% / 96% |
| none-nocur | 2 | argmax | 0.70 | 209 | (70,85) 10, (80,5d) 8, (a0,9d) 7, (7a,5d) 7 | 99% / 94% |
| none-cur | 1 | sample | 0.79 | 236 | (78,75) 11, (80,65) 8, (b0,99) 8, (70,bd) 7 | 100% / 96% |
| none-cur | 1 | argmax | 0.83 | 248 | (78,75) 9, (b0,a3) 9, (b0,99) 9, (a0,9d) 8 | 100% / 96% |
| none-cur | 2 | sample | 0.71 | 212 | (a0,9d) 12, (80,5d) 11, (78,75) 7, (70,75) 6 | 100% / 97% |
| none-cur | 2 | argmax | 0.73 | 219 | (a0,9d) 15, (78,75) 9, (78,5c) 7, (80,5d) 7 | 95% / 98% |
| kind-nocur | 1 | sample | 0.21 | 0 / 0 / 62 (0) | (92,5d) 7, (78,5d) 4, (7a,5d) 4, (80,5d) 4 | 100% / 99% |
| kind-nocur | 1 | argmax | 0.22 | 0 / 0 / 65 (0) | (80,5d) 8, (92,5d) 6, (78,59) 5, (78,5d) 4 | 100% / 99% |
| kind-nocur | 2 | sample | 0.17 | 0 / 0 / 50 (0) | (78,5d) 3, (78,5c) 3, (9d,5d) 2, (68,5d) 2 | 100% / 99% |
| kind-nocur | 2 | argmax | 0.08 | 0 / 0 / 25 (0) | (78,5d) 3, (85,5d) 3, (88,5d) 2, (90,5d) 2 | 100% / 99% |
| kind-cur | 1 | sample | 0.20 | 0 / 0 / 60 (1) | (88,5d) 5, (79,5d) 4, (a0,7d) 4, (98,5d) 3 | 100% / 99% |
| kind-cur | 1 | argmax | 0.17 | 0 / 0 / 51 (0) | (88,5d) 6, (90,5d) 5, (92,5d) 5, (78,5d) 4 | 100% / 99% |
| kind-cur | 2 | sample | 0.14 | 0 / 0 / 41 (0) | (78,5d) 7, (88,5d) 3, (80,7d) 2, (80,5d) 2 | 100% / 100% |
| kind-cur | 2 | argmax | 0.11 | 0 / 0 / 32 (0) | (78,5d) 8, (78,59) 5, (7a,5d) 3, (78,5c) 2 | 100% / 99% |

Full health is 7 hearts. Half of the starts are at full health, the rest drawn from 0.15–1 of it.

## 3. Observations

### The routes

**`kind`:** every run's argmax episode is the scripted clip route with a few extra moves and attacks:
1. `MOVE W 4` and `MOVE W 32` (or two `MOVE W 16`) from the door to (`$B0`,`$8D`);
2. `CLIP N` to (`$B0`,`$7D`);
3. W to X `$A0`–`$A8` (with a BOOMERANG, BEAMS or BOMB W on the way in most episodes);
4. N up to the top row (Y `$5D`);
5. W along the top row to the door column (X `$78`), then `MOVE N 32` out.

Of the 20 sampled episodes per act mode and run (every 15th, in each run's `analysis.txt`), all clip on that square
and go on W. No `kind` evaluation episode reached the south half by the east side. The clip route replaced the
zigzag completely, and no clip led nowhere (every clip was followed by the walk W).

Other clips seen:
- `kind-cur` s1, one sampled episode: up the east side to (`$D0`,`$67`) and `CLIP N` into the top-right block
  (`$D0`,`$5D`), then W along the top row. That is another useful square.
- Five aborted attempts, all `CLIP S` from above (`$A0`,`$85`) after the main clip, or the main clip itself after a
  BOOMERANG (§4). No `CLIP S` completed in evaluation.

**`none`:** every evaluation episode goes S (all 2,400). The cell sequence of the argmax runs is the explored zigzag
shortened:
1. W to (`$B0`,`$8D`);
2. S down the east side to Y `$9D`–`$AD`;
3. W under the diagonal to X `$60`–`$70` at Y `$AD`–`$BD`;
4. N up the middle column (X `$60`–`$7F`) to Y `$6D`–`$75`;
5. E/N to the top row at X `$80`–`$9F`, W to the door.

The hits are spread along it:
- the first descent ((`$A0`,`$9D`), (`$B0`,`$99`)–(`$B0`,`$A3`));
- the middle climb ((`$78`,`$75`), (`$70`,`$75`)–(`$70`,`$85`));
- next to the door ((`$80`,`$5C`)–(`$80`,`$5D`)).

### Damage, fights and the Vires

- `kind` loses 0.07–0.17 hearts per episode and `none` 0.48–0.61. Round 20's chains lost 0.73 per visit (vire
  family model, chain arrivals).
- `kind`'s hits all land after the clip, on the top row (X `$78`–`$A0`, Y `$5D`; a few at (`$80`,`$7D`) and
  (`$A0`,`$7D`) right after the clip). The clip square and the walk to it are hit-free in all 2,400 episodes.
- `kind`'s deaths (0–2 per 300) start at 1–1.5 hearts, once at 3.0. Their last hit lands on the top row, once at
  (`$A0`,`$6C`) on the way up to it. `none`'s deaths (3–7 per 300) start at 1–3.5 hearts and die along the zigzag.
- **Passing, not fighting:** room kills are 0.5–0.8 per episode in every arm, of five Vires (each splits into two
  Red Keese when hit without being killed).
  - `kind` attacks about 2 times per episode (0.44–0.56 bombs). It often throws the boomerang or a bomb W right
    after the clip, at the Vires on the way.
  - `none` attacks 5–7 times (0.57–1.08 bombs).
- **Time near a monster:** the share is the same in both designs (36–41% of the frames before the exit begin with a
  monster within 24 px). `kind`'s episodes are shorter, so that is 60–72 frames per episode against 134–161.

### The curriculum

- **`none`:** the explored zigzag did not speed learning up. 90% came at 356k/340k with it and 311k/426k without;
  95% at 618k/561k and 581k/667k. The final models are the same (98% in training, 95–99% in evaluation).
  - Iteration 1 succeeded in none of its 4–30 room-entry episodes in every `none` run, with or without the
    curriculum.
  - The tracks are long (110–122 steps) and one ends at 1.0 hearts.
- **`kind`:** with the curriculum, 90% came at 168k/176k, without it at 196k/184k. 95% at 262k/258k and 303k/258k.
  - The tracks never clip (they were explored under `none`), yet `kind-cur` learned the clip just as surely and a
    little earlier.
  - Both `kind` settings clip from (`$B0`,`$8D`) in their room-entry episodes from about 190k steps on.

### Learning the clip

- **The untrained policy already clips:** 0.8–8 clips per episode at iteration 1. CLIP is offered at 60 grid points
  in this room and the points between them, so a random policy presses it often.
- **Early training clips N and S alike, many times per episode:** 8–25 at 12–45k steps, in episodes of 270–800
  steps. Success then is 0–60%.
- **Clips drop to one as success rises:**
  - 1.7 per episode at 126k (85% success);
  - 1.07 at 188k (90%);
  - one clip, N, from 190–240k on, with S clips at 0.01–0.04 per episode.
- **Aborts:** `clips/aborted` in training is 0.3–3 per episode early and 0.06–0.09 at 331k. In evaluation it is 5
  clips in 2,400 episodes (§4).
- The clip did not have to be "discovered": the problem was to stop clipping everywhere else. `kind` reached 90%
  in about half the steps `none` needed.

## 4. Bugs and oddities

- **Aborted clips are `stalled`, not hits.** There were 5 evaluation aborts: 4 in `kind-cur` s1 sample (jobs 73,
  75, 107, 251) and 1 in `kind-nocur` s1 sample (job 107, the same start and the same first actions).
  - The 4 `kind-cur` ones were reproduced from the replay states (`ReplayPlayer.save_state` → `Game.load`, then the
    same `CLIP N`/`CLIP S`). Each gives `abort: stalled`, with no health lost.
  - The actions just before them were `SWORD W, MOVE N 32`; `MOVE W 8, MOVE N 32`; `BOOMERANG NW`; and
    `BOOMERANG NW, MOVE N 32`.
  - In the reproduced case after `BOOMERANG NW` at (`$B0`,`$8D`), Link's `$AC` reads `$21`/`$31` for a few frames
    while the boomerang comes back, and `skip_available('N')` is false then.
  - The mask offered `CLIP N` (the decision started with `$AC` = 0), and Link froze partway through the macro's
    run-up.
  - The executor's own name for the case: "Link stood still (something the decision table doesn't know held him)".
  - [INFERENCE] `$AC` `$21`/`$31` is Link's item-use state (catching the boomerang). The other aborts (`CLIP S` at
    (`$A0`,`$6F`)–(`$A0`,`$7D`)) were not traced as far.
- **A stalled clip finished by walking.** Episode 107 (`4_50e_1`, both `kind` s1 runs, sample):
  - `BOOMERANG NW`, then `CLIP N` stalls with Link at (`$B0`,`$83`), 6 px into the block;
  - the next plain `MOVE N 32` moved him to (`$B0`,`$7D`), the clip landing, and he went on W and out;
  - the executor counted no clip (`done` 0, `aborted` 1). The analysis lists the move as an unconfirmed clip, and
    the episode as a win without a clip ("(1)" in the hits table).
  - So once Link is partly inside the block, the game lets a plain walk carry him through.
- **`none` ends `stuck` against the wall beside the N door.** 3–8 of 300 argmax episodes (2 of 300 sampled in
  `none-nocur` s1) end `stuck` (a hard failure). Link holds `MOVE N` for 50+ decisions without moving, at
  (`$80`,`$5C`), (`$59`,`$5D`) or (`$99`,`$5D`): 8 px or more off the door column (X `$78`), against the top wall.
  - A few `none-nocur` argmax episodes (3 and 1) end `no-damage` (600 actions without damage dealt or taken, a
    soft-failure truncation) after 608–630 steps, oscillating near the top row or at (`$68`,`$85`).
  - `kind` reaches the same row and has no `stuck` ending.
  - Not traced further. The tree still has main's doorway attack-mask bug on the top row (Y `$5D`; backlog,
    `26a9196`). [INFERENCE] It concerns attack directions, not movement, so it is not the cause here.
- **The explorer prints the wrong path for a YAML specialist:** `-> …/trajectories/r21-vire50-explore/experiments/
  r21/vire50-none-nocur.yaml/4_50e_0`, while the files are in `…/r21-vire50-explore/vire50-none-nocur/4_50e_0`. This
  is the same issue as in `round21-trap-room.md` §4. The script copies from the real location.
- **Monster type `$13` (Zol)** shows up in the monster list of some replays' last frame, which is already in the next
  room (4:0x40). The analysis leaves the last decision out of "near"; kills take the maximum over the episode.

## 5. Not done

- **No `long` or `edge` arms.** The design was chosen (the CLIP action kind) while the first runs trained, and the
  `long`/`edge` arms were dropped before any started. Only the 15-iteration throughput run used an `edge` YAML; it
  was not evaluated, and its YAML is not committed.
- No `shaping: walk` arms; no other vire rooms; no chains; 0x50 only from its E door.
- No best-checkpoint selection: each run was evaluated at its final checkpoint. Every final checkpoint scored at or
  near its last training iterations.
- Behaviour was read from the analysis of all 2,400 episodes per design and from the 20 sampled episodes per act
  mode and run in each `analysis.txt`. No replay was watched.
- The tracks were not rebuilt a second time to confirm the explorer reproduces them. Their `trajectory.json` matches
  round 20's seed-0 tracks in everything but the explorer's timing block.
