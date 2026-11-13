# Round 23: act modes (sample, argmax, temperature, floors) and measuring risk and variety

The owner (Nov 13): "I WANT variance in play, I do NOT want the model to play the same every time. However, I also
don't want it randomly walking into a fireball because we make 1000s of decisions per game and 10s of them by nature
will be 1% moves. [...] By default I want playing the game to not be argmax, but training and eval are up to you."
Earlier he suggested a threshold (actions under about 10% removed). Branch `r23`; decision pending
(`docs/decisions.md` 2026-11-13 "Act modes"). Code: `1f2368c` (modes, decision statistics), `490f18c` (base rates,
pilot scripts).

## Summary

- **Act modes** (`docs/rust-actor.md` "Act modes"): `sample`, `argmax` (unchanged, bit for bit), and rules over the
  **joint** action (kind × direction × duration, 256 actions): `temperature:T`, `floor:P`, `minp:R`, `topp:P`,
  `escape:N` (argmax until the env's linger count reaches N decisions without progress, then a draw until progress),
  combined with `+`. Spelled the same in every tool, the `act` file and replays (META `models[].act`).
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
- **Recommendation (pending):** play default `floor:0.1` (the owner's own threshold; never a move under 10%, 70% of
  episodes distinct, 326 failures against sample's 397); evaluation and chains one global rule, `floor:0.1` (or
  `floor:0.03+escape:16` where variety doesn't matter: 317, the fewest); training-time evaluation and early-stop
  confirmation in the same mode. Chains: see §4 (the baseline is measured; the other modes were still running when
  this was written: `scripts/experiments/r23_act_pilot.sh report` adds them).

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

## 4. Chains (1000 runs per segment)

The baseline (`ACT=model`) on this code reproduces round 21's `r71-1k-dks` exactly (same seeds, same actions:
`sample` and `argmax` are bit for bit the old ones): full 12.8% ± 1.1, D2 85.6% ± 1.1, D3 46.5% ± 1.6, D4 56.4% ±
1.6. Moves per run under 1% (r71 per family): D2 3.1, D3 3.1, D4 6.7, full 8.6; followed by a hit 5.6-7.8% against
4.3-5.9% for any move, by a death 0.1-0.3% (same as any move). The first room's action sequences are distinct in
23.5% of D2/D3 runs (argmax families start those segments; the RNG is the only spread) and 4.7% of D4 runs.

The other modes (`floor:0.1`, `floor:0.03`, `temperature:0.75`, `floor:0.03+escape:16`, `escape:16`, `sample`,
`argmax`) were still running when this was written (about 25 minutes per mode); their files land in
`target/r23-pilot/chains/r23-<mode>-<segment>.jsonl` and `r23_act_pilot.sh report` tabulates them. **Not decided
by this document:** whether a global rule beats the per-family choice in chains, where single-room differences of
0.1-0.2 points compound over ~25 rooms.

## 5. Recommendation (pending the lead's decision)

- **Play default (play-zelda, zelda-headless, the golden path): `floor:0.1`.** Never argmax (the owner's
  constraint), 70% distinct episodes against argmax's 51% and sample's 76%, no move under 10% ever, and in single
  rooms the fewest failures of the drawing modes (326, with temperature:0.75 327). `registry.PLAY_ACT` is still
  `sample`: switching it is one line once decided. If more variety is wanted, `floor:0.03` (74%, 347).
- **Evaluation and chains: one global rule, `floor:0.1`**, unless the chains show otherwise; `floor:0.03+escape:16`
  is the single-room best (317) but plays like argmax. A per-family split is not worth its bookkeeping on these
  numbers. The promoted `act` files were not changed (read-only models).
- **Training-time evaluation and early stopping: the same rule.** `train_round.py --eval-act MODE` exists; the
  early-stop confirmation (`best_of_n.success_by_state`) still samples and would need a flag.
- **Weak points:** single-room differences among sharpened modes are within 1-1.5 SE; the per-family winners are
  picked on the same data they are judged on; the hit-after-a-rare-move window (4 decisions) is a choice, not a
  measurement of cause; chains other than the baseline are not in this document; the chain report's "hearts at
  arrival" column reads `health_in` of the arrival room and should be checked before it is quoted.

## 6. Re-measuring on the round 22 models

One command, on a merged tree with the round 22 team promoted:

    TRIFORCE_MODELS_DIR=<round 22 models> OUT=target/r23-on-r22 CHAIN_MODES="model floor:0.1 floor:0.03+escape:16" \
        scripts/experiments/r23_act_pilot.sh all

(rooms in every mode of the grid, then the chains, then the tables). Every new checkpoint changes the answer per
family (argmax is fragile per checkpoint), so this is to be rerun after every retrain.
