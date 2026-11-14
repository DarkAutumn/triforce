# Round 23: act modes (sample, argmax, temperature, floors) and measuring risk and variety

The owner (Nov 13): "I WANT variance in play, I do NOT want the model to play the same every time. However, I also
don't want it randomly walking into a fireball because we make 1000s of decisions per game and 10s of them by nature
will be 1% moves. [...] By default I want playing the game to not be argmax, but training and eval are up to you."
Earlier he suggested a threshold (actions under about 10% removed). Branch `r23`; decision pending
(`docs/decisions.md` 2026-11-13 "Act modes"). Code: `1f2368c` (modes, decision statistics), `490f18c` (base rates,
pilot scripts), then merged with round 22 (§6). **The pilot (§1-§5) was measured on the round 21 team** (`r71`);
the rerun on the round 22 team is §8, pending its retrain.

## Summary (the pilot, on the round 21 team)

- **Act modes** (`docs/rust-actor.md` "Act modes"): `sample`, `argmax` (unchanged, bit for bit), and rules over the
  **joint** action (kind × direction × duration, 256 actions): `temperature:T`, `floor:P`, `minp:R`, `topp:P`,
  `escape:N` (argmax until the env's linger count reaches N decisions without progress, then a draw until progress),
  combined with `+`; after the merge also `relax:N` (the other rules' draw until stalled, then plain `sample`: §6).
  Spelled the same in every tool, the `act` file and replays (META `models[].act`).
- **Risk and variety measures** in every evaluation and chain record: per decision the chosen action's probability
  under the unmodified policy is counted per room visit (moves under 1% and 5%, those followed by a hit or death
  within 4 decisions, and the same for every move as the base rate), the acting entropy, and the actions themselves
  (distinct sequences, first divergence, frames spread). `triforce room-stats` and the evaluation summary report them.
- **Single rooms, 24 families × 12 modes × 36,400 episodes:** every sharpened mode beats both `sample` and `argmax`
  by about 50-85 failures of 36,400 (≈ 2.5-4 SE); among the sharpened modes the differences are within noise.
  `argmax` loses through loops (92 loop endings against 6-22), `sample` through its rare moves (384 deaths against
  304-330).
- **Rare moves are only mildly dangerous.** Under `sample` a move under 1% is followed by a hit within 4 decisions
  7.4% of the time, against 5.2% for any move (1.4×), and by a death 0.2%, the same as any move. At 0.21 such moves
  per episode they cost little each; what removing them buys is consistent but small (sample 397 failures, floor:0.03
  347, floor:0.1 326).
- **Variety:** share of distinct action sequences from the same start state: sample 76%, floor:0.03 74%,
  temperature:0.75 72%, floor:0.1 70%, minp:0.2 68%, temperature:0.5 65%, the r71 per-family choice 60%, argmax and
  every `escape` mode 51% (what remains is the game's random numbers).
- **Chains (1000 runs per segment):** `escape:16` for every family takes the full route from 12.8% (the r71
  per-family choice) to 20.5% (D2 89.5%, D3 49.6%, D4 69.5%); `sample` for all 8.7%, `argmax` 14.0% with 179 stuck
  runs, `floor:0.1` 16.0% with 113 stuck, `temperature:0.75` 15.0% with 17.
- **Recommendation (pending):** evaluation, chains and training-time evaluation: one global `escape:16`. Play default:
  `temperature:0.75` (draws every action, beats the r71 choice in chains, does not loop);
  `temperature:0.75+escape:16` is the untested likely improvement.

## 1. Setup

Code `r23` on 1.4.1.0, the promoted r71 team (`TRIFORCE_MODELS_DIR` = the main checkout's models, read-only; every
output under this worktree's `target/r23-pilot`). Nothing else trained on the machine.

    TRIFORCE_MODELS_DIR=<models> scripts/experiments/r23_act_pilot.sh rooms    # one eval-batch per mode
    TRIFORCE_MODELS_DIR=<models> CHAIN_MODES="model floor:0.1 floor:0.03 temperature:0.75 floor:0.03+escape:16 \
        escape:16 sample argmax" scripts/experiments/r23_act_pilot.sh chains
    TRIFORCE_MODELS_DIR=<models> scripts/experiments/r23_act_pilot.sh report   # r23_act_report.py: the tables

- Single rooms: `triforce eval-batch r23-pilot-<mode> --families all --acts <mode> --output-dir
  target/r23-pilot/rooms/<mode>` (100 episodes per start state, seed 0; one GUI group per mode, log folders
  `logs/<date>_<n>_r23-pilot-<mode>` in the `r23` worktree).
- Chains: `r56_chains.sh` with `ACT=<mode> RUNS=1000 SEGMENTS="full:start d2:2_7ds_0 d3:3_7cs_0 d4:4_71s_0"`,
  `TAG=r23-<mode>`, into `target/r23-pilot/chains`. `ACT=model` is each family's own act file (the baseline).
- Quick pass first (darknut, goriya, boss-dodongo, ow-moblin; 14 modes, `target/r23-quick`, 18 minutes). Dropped
  after it: `floor:0.01` (more <5% moves than floor:0.03, same variety), `minp:0.05` (≈ floor:0.03), `escape:48`
  (argmax loops remain: 6 against escape:16's 0); added `minp:0.1`.
- Standard errors: success √(p(1−p)/n); a failure count's SE is about its square root (≈ 18 at 330), so two modes'
  totals need about 50 failures between them to differ by 2 SE.

## 2. Single rooms, every mode (24 families, 36,400 episodes each)

Moves per episode; "then hit / death": the share of those moves followed within 4 decisions by a hit / a death;
"any move": the same share for every move (the base rate). Distinct: share of distinct action sequences among the
episodes of a start state, averaged over states; diverge: the first differing decision of two episodes, averaged
over pairs; frames sd: per start state.

| mode | success % | failures | deaths | loops | hearts/ep | <1%/ep | then hit / death % | <5%/ep | then hit / death % | any move: hit / death % | entropy | distinct % | diverge | frames sd |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---:|---:|---:|---:|
| model (r71 per family) | 99.0 ± 0.1 | 351 | 336 | 10 | −0.040 | 0.13 | 5.9 / 0.2 | 0.58 | 5.7 / 0.1 | 5.1 / 0.2 | 0.37 | 59.5 | 3 | 69 |
| sample | 98.9 ± 0.1 | 397 | 384 | 8 | −0.050 | 0.21 | 7.4 / 0.2 | 0.93 | 6.7 / 0.2 | 5.2 / 0.2 | 0.83 | 76.4 | 1 | 75 |
| argmax | 98.9 ± 0.1 | 402 | 304 | 92 | −0.033 | 0 | – | 0 | – | 4.7 / 0.2 | 0 | 50.7 | 3 | 111 |
| temperature:0.75 | 99.1 | 327 | 314 | 11 | −0.037 | 0.04 | 7.5 / 0.2 | 0.34 | 6.2 / 0.2 | 5.0 / 0.2 | 0.64 | 71.6 | 2 | 66 |
| temperature:0.5 | 99.1 | 332 | 310 | 21 | −0.033 | 0.00 | – | 0.06 | 5.2 / 0.1 | 4.9 / 0.2 | 0.44 | 65.1 | 2 | 70 |
| floor:0.03 | 99.0 | 347 | 330 | 13 | −0.042 | 0 | – | 0.37 | 6.2 / 0.2 | 5.1 / 0.2 | 0.71 | 74.4 | 1 | 72 |
| floor:0.1 | 99.1 | 326 | 305 | 19 | −0.034 | 0 | – | 0 | – | 4.9 / 0.2 | 0.52 | 69.7 | 2 | 77 |
| minp:0.1 | 99.1 | 345 | 328 | 14 | −0.036 | 0 | – | 0.09 | 8.8 / 0.2 | 5.0 / 0.2 | 0.62 | 71.7 | 2 | 68 |
| minp:0.2 | 99.1 | 340 | 314 | 22 | −0.034 | 0 | – | 0.01 | 8.9 / 0.0 | 4.9 / 0.2 | 0.50 | 67.9 | 2 | 73 |
| topp:0.9 | 99.0 | 352 | 332 | 16 | −0.039 | 0 | – | 0.13 | 8.2 / 0.1 | 5.0 / 0.2 | 0.63 | 72.0 | 2 | 69 |
| escape:16 | 99.1 | 329 | 319 | 6 | −0.032 | 0.01 | 4.8 / 1.2 | 0.03 | 4.8 / 0.6 | 5.0 / 0.2 | 0.01 | 50.7 | 3 | 63 |
| temperature:0.75+floor:0.03 | 99.1 | 337 | 314 | 20 | −0.034 | 0 | – | 0.01 | 8.4 / 0.0 | 4.9 / 0.2 | 0.55 | 69.5 | 2 | 71 |
| floor:0.03+escape:16 | 99.1 | 317 | 304 | 11 | −0.032 | 0 | – | 0.01 | 3.2 / 0.2 | 4.9 / 0.2 | 0.01 | 50.7 | 3 | 68 |

Loops: `stuck`, `no-damage`, `no-room-progress` and `max-steps` endings. Full per-family tables:
`r23_act_report.py` (`target/r23-pilot/report-rooms.md`).

## 3. Per family (failures, hearts per episode in brackets; bold: fewest)

| family (own) | episodes | sample | argmax | temperature:0.75 | temperature:0.5 | floor:0.03 | floor:0.1 | minp:0.2 | escape:16 | floor:0.03+escape:16 | fewest failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ow-start (argmax) | 1300 | **0 (+0.00)** | **0 (+0.00)** | **0 (+0.00)** | **0 (+0.00)** | **0 (+0.00)** | **0 (+0.00)** | **0 (+0.00)** | **0 (+0.00)** | **0 (+0.00)** | 12 modes tie (0) |
| ow-octorok (argmax) | 4200 | 52 (-0.07) | **41 (-0.06)** | 45 (-0.06) | 43 (-0.06) | 46 (-0.06) | 46 (-0.06) | 48 (-0.06) | **41 (-0.06)** | **41 (-0.06)** | 3 modes tie (41) |
| ow-zora (argmax) | 3600 | 11 (-0.14) | 4 (-0.12) | 4 (-0.13) | 6 (-0.13) | **3 (-0.13)** | 5 (-0.12) | **3 (-0.12)** | 5 (-0.12) | 4 (-0.12) | 4 modes tie (3) |
| ow-moblin (argmax) | 3000 | 45 (-0.08) | **31 (-0.07)** | 42 (-0.08) | 32 (-0.07) | 44 (-0.08) | 34 (-0.08) | 33 (-0.08) | **31 (-0.07)** | **31 (-0.07)** | 3 modes tie (31) |
| ow-leever (argmax) | 300 | 2 (+0.09) | 1 (+0.11) | **0 (+0.11)** | **0 (+0.12)** | **0 (+0.12)** | **0 (+0.10)** | **0 (+0.11)** | 1 (+0.11) | 1 (+0.11) | 8 modes tie (0) |
| ow-tektite (argmax) | 2100 | 9 (+0.05) | 9 (+0.06) | 9 (+0.05) | 8 (+0.06) | 8 (+0.05) | 9 (+0.05) | 8 (+0.05) | 9 (+0.06) | 9 (+0.06) | 3 modes tie (7) |
| ow-lynel (argmax) | 900 | 9 (-0.18) | 12 (-0.15) | 13 (-0.17) | 11 (-0.16) | 12 (-0.16) | **7 (-0.15)** | 12 (-0.17) | 13 (-0.15) | 13 (-0.15) | floor:0.1 (7) |
| ow-boulder (argmax) | 900 | **0 (-0.00)** | **0 (-0.01)** | **0 (-0.00)** | **0 (-0.00)** | **0 (-0.00)** | **0 (-0.00)** | **0 (-0.00)** | **0 (-0.01)** | **0 (-0.01)** | 12 modes tie (0) |
| ow-peahat (argmax) | 300 | **0 (-0.04)** | **0 (-0.05)** | **0 (-0.05)** | **0 (-0.05)** | **0 (-0.04)** | **0 (-0.05)** | **0 (-0.05)** | **0 (-0.05)** | **0 (-0.05)** | 12 modes tie (0) |
| goriya (sample) | 1500 | 72 (-0.15) | 80 (-0.12) | **61 (-0.11)** | 71 (-0.13) | 65 (-0.13) | 74 (-0.12) | 68 (-0.10) | 66 (-0.12) | 68 (-0.12) | temperature:0.75 (61) |
| d2-rope (sample) | 1800 | 22 (+0.26) | 41 (+0.27) | 20 (+0.26) | **19 (+0.26)** | 20 (+0.27) | **19 (+0.26)** | 20 (+0.26) | 21 (+0.27) | 22 (+0.27) | 3 modes tie (19) |
| moldorm (argmax) | 300 | 36 (-0.20) | 32 (-0.24) | 29 (-0.20) | **21 (-0.15)** | 33 (-0.20) | 25 (-0.17) | 35 (-0.21) | 35 (-0.20) | 31 (-0.21) | temperature:0.5 (21) |
| boss-dodongo (sample) | 300 | 6 (+0.91) | 7 (+0.92) | 4 (+0.91) | 3 (+0.93) | 3 (+0.94) | 2 (+0.94) | **1 (+0.96)** | 5 (+0.90) | 2 (+0.94) | minp:0.2 (1) |
| stalfos-keese-gel (sample) | 4800 | 14 (+0.02) | 19 (+0.03) | **9 (+0.03)** | 12 (+0.03) | 10 (+0.02) | 10 (+0.03) | 13 (+0.03) | 11 (+0.03) | 12 (+0.03) | temperature:0.75 (9) |
| wallmaster (argmax) | 300 | **0 (+0.00)** | **0 (+0.00)** | **0 (+0.00)** | **0 (+0.00)** | **0 (+0.00)** | **0 (+0.00)** | **0 (+0.00)** | **0 (+0.00)** | **0 (+0.00)** | 12 modes tie (0) |
| boss-aquamentus (sample) | 300 | **0 (+1.02)** | 1 (+1.02) | **0 (+1.02)** | **0 (+1.03)** | **0 (+1.04)** | **0 (+1.03)** | **0 (+1.02)** | 1 (+1.02) | 1 (+1.02) | 9 modes tie (0) |
| trap-room (argmax) | 1800 | 4 (-0.18) | 3 (-0.16) | **1 (-0.17)** | **1 (-0.16)** | 2 (-0.17) | 2 (-0.16) | **1 (-0.16)** | **1 (-0.16)** | **1 (-0.16)** | 6 modes tie (1) |
| equipment-room (argmax) | 900 | 5 (-0.19) | 4 (-0.18) | 6 (-0.17) | 6 (-0.18) | 5 (-0.19) | 7 (-0.17) | 6 (-0.16) | 4 (-0.18) | 4 (-0.18) | topp:0.9 (3) |
| zol (sample) | 2100 | 7 (+0.16) | 7 (+0.16) | 8 (+0.16) | 9 (+0.17) | 6 (+0.18) | 5 (+0.18) | 6 (+0.17) | **3 (+0.17)** | **3 (+0.16)** | escape:16, floor:0.03+escape:16 (3) |
| darknut (sample) | 2100 | 47 (-0.39) | 36 (-0.36) | 35 (-0.37) | 37 (-0.37) | 37 (-0.38) | 35 (-0.38) | **30 (-0.38)** | 35 (-0.36) | 34 (-0.36) | minp:0.2 (30) |
| boss-manhandla (argmax) | 300 | 19 (-0.50) | **10 (-0.31)** | 13 (-0.41) | 11 (-0.31) | 14 (-0.51) | 16 (-0.33) | 15 (-0.33) | 15 (-0.38) | 13 (-0.36) | argmax (10) |
| vire (sample) | 2400 | 25 (-0.06) | 41 (-0.02) | **16 (-0.00)** | 28 (+0.02) | 27 (-0.04) | **16 (-0.02)** | 22 (-0.01) | 18 (-0.01) | **16 (-0.01)** | 3 modes tie (16) |
| like-like (sample) | 600 | **10 (-0.53)** | 22 (-0.57) | **10 (-0.55)** | 13 (-0.55) | **10 (-0.56)** | 13 (-0.54) | 17 (-0.53) | 13 (-0.58) | **10 (-0.57)** | 4 modes tie (10) |
| boss-gleeok (argmax) | 300 | 2 (+0.01) | **1 (+0.29)** | 2 (+0.07) | **1 (+0.01)** | 2 (-0.00) | **1 (+0.09)** | 2 (+0.11) | **1 (+0.28)** | **1 (+0.29)** | 6 modes tie (1) |

- **Argmax families gain nothing from a draw.** The nine overworld families, wallmaster, equipment-room, trap-room,
  boss-gleeok and boss-manhandla are at their best under argmax or `escape` (identical when nothing stalls); any draw
  costs ow-octorok 2-11, ow-moblin 3-14, ow-zora 1-7 failures. boss-manhandla: argmax 10, every draw 13-19.
- **Sample families gain from any sharpening**: darknut 47 → 30-37, vire 25 → 16, d2-rope 22 → 19-22, boss-dodongo
  6 → 1-4, stalfos-keese-gel 14 → 9-12, goriya 72 → 61-74 (temperature:0.75 best). like-like stays best by sample.
- **escape removes argmax's loops**: boss-dodongo 7 → 0, d2-rope 20 → 0, goriya 14 → 0, vire 24 → 1, like-like 8 →
  0; it plays argmax otherwise (0.01 acting entropy per decision), so its variety is argmax's.
- **One rule against per-family choices.** The r71 per-family choice: 351. Global rules: floor:0.03+escape:16 317,
  floor:0.1 326, temperature:0.75 327, escape:16 329. Per-family hybrids (r71 argmax families keep argmax, the
  sample families take a rule): 311 (temperature:0.75), 322 (floor:0.1), 321 (escape:16). The oracle (best mode
  per family on this very data, overfit) 274. So one global sharpened rule recovers most of what is to be had; a
  per-family split adds ~10-15 failures of 36,400, within noise.

## 4. Chains (1000 runs per segment, one mode for every family)

The baseline (`ACT=model`) on this code reproduces round 21's `r71-1k-dks` exactly (same seeds, same actions:
`sample` and `argmax` are bit for bit the old ones). Arrivals ± SE; deaths and stuck runs before arriving; moves
under 1% per run and the share followed by a hit within 4 decisions against any move's; distinct: share of distinct
action sequences of the segment's first room (every run from the same state; full: the power-on/start room).

| mode | D2 | D3 | D4 | full route | full: deaths / stuck | full: <1% per run, then hit % (any move %) | full: entropy | D2 first room distinct |
|---|---:|---:|---:|---:|---|---|---:|---:|
| model (r71 per family) | 85.6 ± 1.1 | 46.5 ± 1.6 | 56.4 ± 1.6 | 12.8 ± 1.1 | 857 / 11 | 8.6, 6.8 (5.2) | 0.51 | 23.5% |
| sample | 85.1 ± 1.1 | 44.7 ± 1.6 | 51.0 ± 1.6 | 8.7 ± 0.9 | 889 / 10 | 13.3, 7.7 (5.2) | 0.86 | 23.5% |
| argmax | 79.0 ± 1.3 | 38.3 ± 1.5 | 65.6 ± 1.5 | 14.0 ± 1.1 | 676 / 179 | 0 | 0 | 0.1% |
| temperature:0.75 | 86.7 ± 1.1 | 48.7 ± 1.6 | 59.9 ± 1.5 | 15.0 ± 1.1 | 823 / 17 | 3.2, 7.2 (5.0) | 0.67 | 11.3% |
| floor:0.03 | 85.4 ± 1.1 | 45.3 ± 1.6 | 57.9 ± 1.6 | 13.1 ± 1.1 | 836 / 20 | 0 | 0.73 | 19.7% |
| floor:0.1 | 86.6 ± 1.1 | 47.2 ± 1.6 | 65.0 ± 1.5 | 16.0 ± 1.2 | 717 / 113 | 0 | 0.57 | 3.4% |
| **escape:16** | **89.5 ± 1.0** | **49.6 ± 1.6** | **69.5 ± 1.5** | **20.5 ± 1.3** | 781 / 7 | 0.7, 4.6 (4.9) | 0.03 | 0.1% |
| floor:0.03+escape:16 | 90.2 ± 0.9 | 46.4 ± 1.6 | 67.4 ± 1.5 | 19.1 ± 1.2 | 777 / 24 | 0 | 0.04 | 0.1% |

- **The chains separate what single rooms could not.** Small per-room gains compound over ~60 rooms: `escape:16`
  takes the full route from 12.8% to 20.5% (+7.7 points, ≈ 4.5 SE), with every segment better than the baseline
  (D4 +13.1). `sample` for everyone is the worst (8.7%): the dungeon families' sampling is what costs D4 most.
- **argmax kills fewer runs but loops**: full 676 deaths and 179 stuck; D2 115 stuck, D3 192. `escape:16` keeps
  argmax's low death count (D4 305 deaths against argmax's 308) and removes almost all its loops (7 stuck in full).
- **floor:0.1 also loops in chains** (110 stuck in D3, 113 in full): when the model's mass sits on one action a
  floor is argmax, without the escape. In single rooms that cost little (19 loops in 36,400 episodes).
- **Variety in chains:** whole runs differ in every drawing mode (98-100% of full runs distinct by the first room),
  but within a dungeon's first room the sharpened modes repeat more (D2: 23.5% distinct by sample, 11.3% by
  temperature:0.75, 3.4% by floor:0.1). Under argmax and escape runs differ only by the game's random numbers.
- **Rare moves:** a full run by sample makes 13 moves under 1%; they are followed by a hit 7.7% of the time against
  5.2% for any move, and by a death 0.3%, like any move. They are a small part of what separates the modes; argmax's
  greed (fewer hits) and loops are the larger part.

## 5. Recommendation (pending the lead's decision)

- **Evaluation and chains: one global `escape:16`.** Full route 20.5% against the per-family choice's 12.8%, every
  segment better, the fewest loops of any mode, and in single rooms 329 failures (within noise of the best). It
  replaces the per-family sample/argmax choice; per-family tuning is not worth it on these numbers. The promoted
  `act` files were not changed (read-only models): `triforce promote F RUN --act escape:16` per family, after the
  round 22 rerun.
- **Play default (play-zelda, zelda-headless, the golden path): `temperature:0.75`.** It must not be argmax and must
  show variety, which rules out `escape` (it plays argmax 97% of the time). Of the drawing modes it is the best that
  doesn't loop: full route 15.0% (above the r71 per-family 12.8% and sample's 8.7%), 17 stuck runs against floor:0.1's
  113, about 1 move under 1% per D2 run instead of 3.9. `floor:0.1` reaches 16.0% but loops. Untested then:
  `temperature:0.75+escape:16` (but it is argmax until stalled, so it shows argmax's variety) and, added at the
  merge, `temperature:0.75+relax:16` (a temperature draw until stalled, then plain sample: every action drawn).
  `registry.PLAY_ACT` is still `sample`: one line once decided.
- **Training-time evaluation and early stopping: `escape:16`**, the chain rule (`train_round.py --eval-act
  escape:16`, which since the merge also makes the early-stop confirmation act that way: `train --confirm-act`).
- **Weak points:** single-room differences among sharpened modes are within 1-1.5 SE; the escape threshold (16) was
  measured against 48 only in the quick pass; the hit-after-a-rare-move window (4 decisions) is a choice, not a
  measured cause; `temperature:0.75+escape:16` and `floor:0.1+escape:16` were not run.
- **Hearts at arrival, corrected (merge, Nov 13).** The report's column read the arrival record's `health_in`, which
  out of a dungeon is the max hearts (the triforce refills them on the way out). It now reads Link's health on
  entering the dungeon's triforce room (`r23_act_report.arrival_hearts`, checked by hand against
  `r23-escape:16-*.jsonl`: D2 3.36, D3 3.40, D4 3.67, full 3.70). On these chains: D2 3.27-3.36 hearts in every
  mode, D3 3.24-3.43, D4 3.43 (sample) to 3.77 (floor:0.1), full 3.21 (floor:0.03) to 3.82 (floor:0.1); `model`
  3.27 / 3.34 / 3.64 / 3.58, `escape:16` 3.36 / 3.40 / 3.67 / 3.70.

## 6. Merged with round 22 (Nov 13): what changed for the rerun

Everything above was measured **on the round 21 team** (`r71`, promoted in the main checkout) with code `r23` on
1.4.1.0, before round 22's RNG, route, states and clips switch. The branch is now merged with `main` (round 22,
1.4.1.1):

- Every tool that plays takes the act mode and round 22's `--clips` side by side (`evaluate --act MODE --clips
  on|off|both`, `eval-chain --act MODE --clips on|off [--clips-off-from ...]`, `eval-batch --acts ... --clips`,
  `play-zelda.py --act MODE --clips on|off [--live]`, `zelda-headless.py`, the golden path). Records carry both the
  switch fields (`clips_allowed`, `clips_flips`) and `decisions`; `by_clips` summaries and room statistics carry
  `risk` per switch state. The live cell plays in the `--act` mode; `escape` and `relax` read the chain's
  no-progress count there too, and a switch flip is no progress.
- **The two readings of a rule with a stall count** (`docs/rust-actor.md` "Act modes"): `temperature:T+escape:N` is
  argmax until Link has made no progress for N decisions, then a temperature draw until progress; the new
  `temperature:T+relax:N` is a temperature draw at every decision until stalled, then a plain `sample` from the
  whole policy until progress. `relax` draws every action (the owner's variety) and still breaks loops; it is the
  candidate play default the pilot did not have.
- `train_round.py --eval-act MODE` also sets the early-stop confirmation's act mode (`train --confirm-act`);
  rollouts always sample.
- `r23_act_pilot.sh` takes `CLIPS=on|off` (OUT defaults to `target/r23-pilot-clips<CLIPS>`), and its default
  `CHAIN_MODES` is the rerun's: `model escape:16 escape:8 temperature:0.75 temperature:0.5 temperature:0.75+escape:16
  temperature:0.5+escape:16 floor:0.1+escape:16 temperature:0.75+relax:16`.
- A 20-run check per segment start with the round 21 team in `escape:16` (one `eval-batch`): §7.

The rerun after the round 22 retrain, once with the switch on and once off:

    TRIFORCE_MODELS_DIR=<round 22 models> CLIPS=on scripts/experiments/r23_act_pilot.sh all
    TRIFORCE_MODELS_DIR=<round 22 models> CLIPS=off scripts/experiments/r23_act_pilot.sh all

(rooms in every mode of the grid, then the chains, then the tables, into `target/r23-pilot-clipson` and
`target/r23-pilot-clipsoff`). Every new checkpoint changes the answer per family (argmax is fragile per
checkpoint), so this is to be rerun after every retrain.

## 7. Merge check: the round 21 team in `escape:16` on the merged tree

Not a measurement of the team: a check that the merged tree plays, records and reports every segment in an act mode
with the round 22 route and states. The round 21 models (`r71`, promoted in the main checkout, read-only) never
trained round 22's new rooms (0x67 and its cave, the stretches, 3:0x5c, 4:0x01/0x02), so runs fail there.

    TRIFORCE_MODELS_DIR=<round 21 models> triforce eval-batch r23-merge-check --chains r23-merge-check \
        --act escape:16 --runs 20 --chains-dir target/r23-merge-check --parallel 8

| segment | runs | arrivals | hearts at arrival | where the others end |
|---|---:|---|---|---|
| full | 20 | 0 | – | 0:0x67 missed 7, 0:0x75 stuck 4, 0:0x0a death 2, 0:0x27 death 2, 3:0x69 death 1 |
| d2 | 20 | 17 (85%) | 3.29 | 2:0x1e death 2, 2:0x2e death 1 |
| post-d2 | 20 | 18 (90%) | 2.69 | 0:0x0a death 2 |
| d1 | 20 | 20 (100%) | 5.22 | |
| post-d1 | 20 | 10 (50%) | 3.00 | 0:0x67 missed 4, 0:0x75 stuck 3, 0:0x67 stuck 1, 0:0x67 death 1, 0:0x76 death 1 |
| d3 | 20 | 0 | – | 3:0x5b death 8, 3:0x5b stuck 7, 3:0x69 death 4, 3:0x59 death 1 |
| post-d3 | 20 | 20 (100%) | 6.58 | |
| d4 | 20 | 0 | – | 4:0x11 stuck 18, 4:0x13 death 1, 4:0x02 death 1 |

- Every run started with the switch on; every room record (milestones included) carries `clips_allowed`, `step` and
  `decisions`, whose `n` equals the visit's steps (279,568 decisions); 423 of 5,582 visits drew at least once
  (escape stalled), the rest were argmax throughout. The provenance names `escape:16` for every model.
- Where runs fail is where round 22 changed the route: 0x67 (its rupee waypoint), 0x75 (the wrap exit), 3:0x5b (its
  new E bomb wall) and 4:0x11 (N into 0x01 now). Round 22's own 20-run checks with this team stuck in 0x75 and
  4:0x11 too (decisions 2026-11-13 "Round 22 merged"); the causes were not looked at here.
- Hearts at arrival are the corrected ones (the yardstick shares `eval_batch.arrival_hearts`): out of Dungeon 1,
  5.22 is Link's health entering its triforce room; the old reading said 6.00, the max.

## 8. On the round 22 team

<!-- r23: on the round 22 team -->

Not measured yet: the commands of §6, after the round 22 retrain.

## 9. Two round 22 problems fixed before the rerun (Nov 14)

Both on the round 22 promoted team (`r72-*` and the kept `r71-*`), each model in its own act mode, clips on, in the
code after `git merge main` (`13c1636`); details and every number in `docs/decisions.md` 2026-11-14.

- **One bomb per wall** (r72 problem 2): the `required` families' BOMB mask also leaves out a wall a live bomb of
  Link's already reaches (fuse, blast or fading). Eval-batch, 100 episodes per state of the affected rooms, seed 0:
  stalfos-keese-gel 4:0x21 1.77 → 1.23 bombs per episode (sample; 0.28 of them 10th-kill bombs), success 99.7%
  before and after, hearts -0.14 both; trap-room 3:0x4c / 4:0x11 1.08 → 1.07 (argmax, 100% both). Played in the game
  in every room the mask covers (`crates/zelda/tests/bomb_mask.rs` `one_bomb`).
- **0x5b's last pass fights first** (r72 problem 1): `if_bombs_below: 6`, read live, became a fight beside the
  opened wall once a bomb spent at 6 left 5. Route step 100 now always takes the treasure. Same seeds before and
  after:

| measure | before | after |
|---|---:|---:|
| D3 segment (1000 runs): 0x5b W → E deaths | 149 / 851 | 93 / 851 |
| … at 6+ bombs on entry (replays) | 70 / 170 | 16 / 170 |
| … survivors' hearts lost, frames | 0.60, 1123 | 0.42, 1080 |
| D3 segment: out of Dungeon 3 | 50.6% | 54.7% |
| full route (1000 runs): 0x5b deaths | 145 / 404 | 59 / 404 |
| full route: out of Dungeon 3 (at 4:0x71) | 19.5% | 27.5% |
| `3_5bw_0` at 6 bombs, 3 hearts (500 runs): 0x5b deaths, out of D3 | 27.0%, 42.6% | 5.0%, 69.0% |
| `3_5bw_0` at 8 bombs, 3 hearts: 0x5b deaths, out of D3 | 5.2%, 60.2% | 5.2%, 68.6% |
| `3_5bw_0` at 3 bombs, 3 hearts (rule unchanged) | 5.8%, 73.2% | 4.8%, 73.0% |
