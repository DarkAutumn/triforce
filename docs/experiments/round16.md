# Round 16: the Rust actor, then a fresh team (r56)

Lee, Nov 9: inference moves to Rust (one call per rollout; eval, chains and play-zelda on the same in-pool policy;
replayable plays), then profile and fix performance, then GPU vs CPU inference, then retrain the whole team from
scratch on the new stack (the old checkpoints didn't load then; since review F066 pre-r56 checkpoints load again),
then the better-play experiments one at a time.

## Summary

- **Rust actor** (`docs/rust-actor.md`, decisions 2026-11-09): built and merged. Rollouts in one call, eval, chains
  and play-zelda act inside the pool, replay files (`--plays`) replay bit for bit, endings count observed effects
  (eval and chains skip the look-ahead). The per-step `recv`/`send` path and `triforce/inference.py` are gone.
- **Performance** (`docs/performance.md`, Nov 9): AVX policy kernels (0.77 → 0.31 ms per observation) and the
  rollout overlapped with the PPO update (Lee: yes): training +20–45% (fresh-model bench). The deferred look-ahead
  was not built (expected under 15%). Inference stays on the CPU (it is ~10% of mass-play CPU, ~8% of training).
- **Fresh team (r56, promoted):** single-room evaluations within the old team's range (85.7–99.7%); chains (120
  runs each, sampled): full route to the D3 triforce 0/120, D2 arrival 102, D1 arrival 40, D3 arrival 11; segments
  D2 92, after D2 88, D1 95, after D1 118, D3 12 (§1).
- **Act mode (IX.1):** argmax loses fewer hearts in every room family but loops (`stuck`) in some; argmax for 15
  families, sampling for goriya, moldorm, stalfos-keese-gel, zol, darknut (`act` files): D2 arrivals 102 → 115,
  D3 arrivals 11 → 21 of 120 from the start (§3).
- **One-family A/B arms (goriya, 3 seeds; moldorm for SIL):** SIL kept (+1–2 pp success, −7 to −15% hearts); entropy
  annealing, outcome fine-tuning, best-of-N, PLR, PPG, `exit_health_weight` and harvested handoff states level with
  the control (training seed spread ±2 pp); Gumbel expert iteration no better on Dodongo and worse on goriya at
  6–14× the training time (§4–§6).
- **SIL team (IX.11):** 11 families continued with SIL and promoted. With the act mix: **D3 arrivals 34/120 from the
  start, full route to the D3 triforce 4/120** (r56 sampled: 11 and 0), D1 115/120, D3 30/120 (§7).
- **Found:** the training success rate lags slow failures and a late collapse (a Dodongo continuation read ~1.0 in
  training and was 54%; §5, mechanism corrected by review F071); argmax loops where a policy has no answer (§3).
  Both in the backlog.
- **training-gui** (Lee, Nov 9): [currently running] shows every live training on the machine (other logs roots
  found from the running triforce processes; `0bc2949`); standalone `evaluate`/`eval-chain` log folders of their
  own (branch `better-play`, merged after r56).
- **Better-play tooling** (branch `better-play`, worktree `../triforce2-better`, merged after r56; §2): the yardstick,
  per-model act mode, entropy annealing, outcome fine-tuning, best-of-N, SIL, Gumbel search + expert iteration, PLR,
  PPG, chain harvesting and `exit_health_weight`, each with tests (Rust 627, Python 266 on the branch tip).

## 1. Fresh retrain (r56)

Every family from scratch with the current yaml, two streams side by side at 8 envs each, 8M steps per family,
early stop at 0.99 (round 13's recipe):

`train_round.py --round r56-rust-ow --specialists ow-start,ow-octorok,ow-zora,ow-moblin,ow-tektite,ow-lynel,ow-boulder,ow-leever,trap-room,equipment-room`
and `--round r56-rust-dg --specialists d2-rope,goriya,moldorm,boss-dodongo,stalfos-keese-gel,wallmaster,boss-aquamentus,darknut,zol,boss-manhandla`
(`--steps 8000000 --parallel 8 --eval-parallel 3 --no-promote --chain-runs 0 --train-args "--stop-success 0.99"`).
When the overworld stream finished, the dungeon stream's tail moved to two more streams (`r56-rust-dg3`:
boss-manhandla, zol; `r56-rust-dg4`: darknut) and the `r56-rust-dg` round was stopped as it started darknut (its
partial run is kept as `r56-rust-dg.partial-killed`). Wall time 9 h.

| specialist | run | steps | early stop | eval success (sampled) | wall |
|---|---|---:|---|---:|---:|
| boss-aquamentus | r56-rust-dg | 1.4M | yes | 99.7% | 0.4 h |
| boss-dodongo | r56-rust-dg | 2.4M | yes | 99.3% | 0.7 h |
| boss-manhandla | r56-rust-dg3 | 8.0M | no | 87.3% | 2.3 h |
| d2-rope | r56-rust-dg | 8.0M | no | 98.3% | 2.0 h |
| darknut | r56-rust-dg4 | 8.0M | no | 95.8% | 1.8 h |
| equipment-room | r56-rust-ow | 0.5M | yes | 99.5% | 0.1 h |
| goriya | r56-rust-dg | 8.0M | no | 93.3% | 1.7 h |
| moldorm | r56-rust-dg | 8.0M | no | 85.7% | 1.8 h |
| ow-boulder | r56-rust-ow | 0.2M | yes | 99.6% | 0.1 h |
| ow-leever | r56-rust-ow | 0.4M | yes | 99.7% | 0.1 h |
| ow-lynel | r56-rust-ow | 8.0M | no | 98.0% | 1.7 h |
| ow-moblin | r56-rust-ow | 8.0M | no | 97.1% | 1.8 h |
| ow-octorok | r56-rust-ow | 3.1M | yes | 98.3% | 0.8 h |
| ow-start | r56-rust-ow | 0.2M | yes | 99.5% | 0.0 h |
| ow-tektite | r56-rust-ow | 1.9M | yes | 99.3% | 0.4 h |
| ow-zora | r56-rust-ow | 1.2M | yes | 98.6% | 0.3 h |
| stalfos-keese-gel | r56-rust-dg | 1.1M | yes | 99.4% | 0.2 h |
| trap-room | r56-rust-ow | 1.1M | yes | 99.4% | 0.3 h |
| wallmaster | r56-rust-dg | 1.9M | yes | 99.7% | 0.5 h |
| zol | r56-rust-dg3 | 0.7M | yes | 99.0% | 0.2 h |

The old team (r47–r55) on the same evaluations: goriya 93.4%, moldorm 85.0%, boss-manhandla 90.7%, darknut 96.6%,
the rest 97.7–100% (yardstick of the pre-refactor promoted runs). So the new stack retrains to the same level.

**Chains (IX.0 baseline, `scripts/experiments/r56_chains.sh`, all promoted, sampled, 120 runs, seeds 0–119).**
Every segment plays on to the route's end (the D3 triforce); a segment's arrival is its first entry into the next
segment's start room (yardstick `--chain NAME=PATH@L:0xRR[,...]`), with Link's health on entering it:

| segment | from | arrival | arrivals | hearts at arrival | where most runs end |
|---|---|---|---:|---:|---|
| full → D2 | start | 2:0x7d | 102 (85%) | 1.75 | 0:0x5b ×7, 0:0x4d ×4 |
| full → after D2 | start | 0:0x3c | 55 (46%) | 4.00 | 2:0x5e ×13, 2:0x1e ×11, 2:0x3e ×10 |
| full → D1 | start | 1:0x73 | 40 (33%) | 1.91 | + 0:0x38 ×7 |
| full → D3 | start | 3:0x7c | 11 (9%) | 3.41 | + 1:0x33 ×7 |
| full | start | D3 triforce | 0 | – | |
| D2 | 2_7ds_0 | 0:0x3c | 92 (77%) | 4.00 | 2:0x1e ×12, 2:0x3e ×7 |
| after D2 | 0_3cc_0 | 1:0x73 | 88 (73%) | 1.92 | 0:0x38 ×11 |
| D1 | 1_73s_0 | 0:0x37 | 95 (79%) | 6.00 | 1:0x44 ×10, 1:0x7f cave ×6 |
| after D1 | 0_37c_0 | 3:0x7c | 118 (98%) | 3.58 | |
| D3 | 3_7cs_0 | D3 triforce | 12 (10%) | 7.00 | 3:0x59 ×55, 3:0x69 ×22 |

Rounds 13–15 for comparison (different segment ends: those runs stopped at their route's end then):

| segment | round | result |
|---|---|---|
| full route to the D1 triforce | 13 (before the bow) | 25/120 |
| D1 from its entrance | 14 | 108/120 (90%) |
| full route to the D1 triforce | 14 | 13/120 |
| D3 from its entrance | 15 | 18/120 |
| from 0x37 (D1 exit) | 15 | 7/120 |
| full route to the D3 triforce | 15 | 0/120 |

The fresh team is about where the old one was (D1 79% vs 90%, D3 10% vs 15%, full route 0/120 both; D1 arrival
33% vs the old full-route-to-D1 11%). Where chains die: D2's Goriya rooms (0x1e, 0x5e) and Moldorm (0x3e), and
Dungeon 3's Darknut room 0x59.

## 2. Better-play tooling (branch `better-play`)

Built while r56 trains, in a worktree (no code changes in the main checkout during a round). One commit each:

| item | commit | what |
|---|---|---|
| §0 yardstick | `9d47663`, `d27ada5` | `scripts/yardstick.py report/compare`: success (Wilson 95%), hearts lost per room visit, health change, frames per visit; chain arrivals and hearts at arrival; `better`/`worse` only when the intervals separate |
| §6b act mode | `6f65456` | `<run dir>/act` (`sample`/`argmax`), `promote --act`; chains, eval, run and the watch player follow it per model (Rust actor: one argmax flag per policy); `--act model\|sample\|argmax` overrides; replay format 2 records it per model |
| GUI | `6f65456` | standalone `evaluate`/`eval-chain` write log folders (`events.logged_job`), `chain_progress` drawn live |
| §6b entropy | `2c36165` | `train --ent-anneal SHARE`: entropy bonus → 0 over the last SHARE of the iterations |
| §1a outcome | `d69c007` | `reward.shaping_scale` (shared `ShapingDial`, `EnvPool.set_shaping`), `reward.frame_penalty` (`penalty-frames`), `train --shaping-anneal N` |
| §3 best-of-N | `1558949` | `train --best-of-n RUNS --best-keep K --best-every N --best-coeff C`: the policy's best successful runs per start state (success, hearts lost, frames) as the BC batch |
| §3 SIL | `af2fd87` | `train --sil-coeff C`: replay of rows with λ=1 return > V, `−log π · (R−V)+` + β value term |
| §2 search | `beb857a` | `EnvPool.rollout(search={...})`: Gumbel-top-k + sequential halving over K RNG-perturbed futures of depth d inside the pool (exact env save/load/simulate, no side effects, thread-invariant); 2.7 s per searched decision at k=8, n=16, K=4, d=6, 0.38 s at k=4, n=8, K=2, d=3 (shared machine) |
| §2 learner | `666d46a` | `train --search JSON --search-coeff C`: searched rows train by cross-entropy to π' over the 224 joint actions and leave the PPO term |
| §5 PLR | `5b63e3f` | `train --plr`: shared start-state weights (`EnvPool.set_start_weights`) from positive value loss per start state, rank^(−1/β) + staleness + uniform |
| §6a PPG | `e4c4b31` | `train --ppg`: auxiliary phase every N iterations (value epochs + exact joint KL to the pre-phase policy) |
| §1b chain-aware | `29d6022` | `reward.exit_health_weight` (success terminal 20 × ((1−w) + w·hp/max), default 0.5 = today); `eval-chain --harvest [DIR]` saves handoff pre-entry states per specialist (`<level>_<room><entry>_h<seed>_<n>`); harvested starts keep the handed-over health (no hp randomization) |

Measured on the way:

- **Explained variance** of the r56 runs so far (median over training): d2-rope 0.63, ow-moblin 0.71, ow-octorok
  0.71, ow-zora 0.68, ow-start 0.81. Not high, so the PPG A/B (IX.10) stays in.
- **Argmax is fragile per checkpoint:** in a smoke of the arms script (ow-start, 3 extra iterations), argmax success
  went 100% → 91.2% while sampled stayed at 99.4–99.8%. An argmax promotion has to be measured on the exact
  checkpoint, and a model can lose it with a few more updates.
- **PLR's β=0.1** (the paper's, for thousands of levels) put ~81% of the draws on one of ow-zora's 30 states in its
  smoke; the A/B needs a flatter β too.

## 3. Argmax vs sampled (IX.1, §6b)

Every r56 model evaluated twice on the same jobs (100 episodes per start state, same seeds, so the comparison is
paired per episode): `eval.json` (sampled, from the round) and `eval-argmax.json` (`--act argmax`). Δ = argmax −
sampled; hearts lost summed over an episode's room visits; the last column is how the failures ended.

| specialist | episodes | Δ success pp [95% CI] | Δ hearts lost / episode [95% CI] | sampled hearts lost / ep | failures sampled → argmax |
|---|---:|---|---|---:|---|
| boss-aquamentus | 300 | +0.3 [−0.3, +1.0] | −0.013 [−0.030, +0.003] | 0.265 | 1 death → 0 |
| boss-dodongo | 300 | 0.0 [−1.3, +1.3] | −0.017 [−0.042, +0.008] | 0.027 | 2 deaths → 1 death, 1 stuck |
| boss-manhandla | 300 | +1.3 [−2.8, +5.5] | −0.127 [−0.269, +0.016] | 1.630 | 34 deaths → 26, +6 no-damage |
| d2-rope | 1800 | −0.2 [−0.9, +0.5] | −0.009 [−0.020, +0.002] | 0.085 | stuck 4 → 9 |
| darknut | 2100 | **−1.5 [−2.5, −0.5]** | −0.073 [−0.114, −0.032] | 0.673 | deaths 75 → 45, stuck 9 → 53, no-damage 1 → 22 |
| equipment-room | 600 | −0.7 [−1.6, +0.3] | −0.102 [−0.143, −0.062] | 0.362 | deaths 3 → 6 |
| goriya | 1500 | +0.1 [−1.3, +1.5] | −0.050 [−0.085, −0.015] | 0.543 | deaths 93 → 76, stuck 7 → 19 |
| moldorm | 300 | **−7.7 [−11.9, −3.5]** | −0.118 [−0.188, −0.048] | 0.870 | deaths 39 → 19, stuck 3 → 47 |
| ow-boulder | 900 | −0.1 [−0.8, +0.5] | −0.024 [−0.040, −0.009] | 0.192 | |
| ow-leever | 300 | +0.3 [−0.3, +1.0] | −0.030 [−0.054, −0.006] | 0.090 | |
| ow-lynel | 900 | +0.3 [−0.7, +1.3] | −0.033 [−0.090, +0.023] | 0.309 | |
| ow-moblin | 2400 | +0.1 [−0.5, +0.7] | −0.025 [−0.036, −0.015] | 0.172 | |
| ow-octorok | 3900 | +0.3 [−0.0, +0.7] | −0.016 [−0.023, −0.008] | 0.122 | |
| ow-start | 1300 | +0.5 [+0.1, +0.9] | 0 | 0 | 7 wrong exits → 0 |
| ow-tektite | 1800 | +0.4 [−0.0, +0.8] | −0.021 [−0.035, −0.008] | 0.144 | |
| ow-zora | 3000 | +0.1 [−0.4, +0.5] | −0.033 [−0.045, −0.020] | 0.251 | |
| stalfos-keese-gel | 4200 | **−8.0 [−8.9, −7.1]** | −0.017 [−0.028, −0.006] | 0.164 | stuck 7 → 331 |
| trap-room | 1500 | −0.4 [−1.0, +0.2] | −0.046 [−0.065, −0.028] | 0.391 | |
| wallmaster | 300 | 0.0 [−0.9, +0.9] | −0.002 [−0.005, +0.002] | 0.002 | |
| zol | 1800 | **−2.6 [−3.5, −1.6]** | −0.059 [−0.088, −0.030] | 0.209 | no-damage 0 → 41, stuck 13 → 19 |

Argmax loses fewer hearts in every room family with damage (5–28% of the sampled loss; significant in 13 of 18):
the sampling noise is what costs hearts. Its failure mode is looping: with no noise, a state the policy has no good
answer for repeats until the room budget ends it (`stuck`, `no-damage`). Where that dominates, argmax wins less:
moldorm, stalfos-keese-gel, zol, darknut.

**Rule:** argmax unless its success is significantly worse; then the chain check. First mix (sampling for moldorm,
stalfos-keese-gel, zol, darknut): goriya looped in D1's 0x23 on 63 of 231 chain visits (a state the single-room
evaluation doesn't start from), so D1 fell from 95 to 41 arrivals; goriya went back to sampling. The final mix,
argmax for 15 families, sampling for goriya, moldorm, stalfos-keese-gel, zol and darknut (`<run dir>/act`,
`promote --act`), against all sampled (120 runs each, same seeds):

| segment | sampled | mix | Δ pp [95% CI] |
|---|---:|---:|---|
| full → D2 | 102 | 115 | +10.8 [+3.6, +18.1] |
| full → after D2 | 55 | 67 | +10.0 [−3.0, +23.0] |
| full → D1 | 40 | 54 | +11.7 [−1.1, +24.4] |
| full → D3 | 11 | 21 | +8.3 [+0.1, +16.6] |
| full (D3 triforce) | 0 | 1 | +0.8 |
| D2 | 92 | 85 | −5.8 [−17.0, +5.4] |
| after D2 | 88 | 91 | +2.5 [−8.7, +13.7] |
| D1 | 95 | 102 | +5.8 [−0.8, +12.5] |
| after D1 | 118 | 120 | +1.7 |
| D3 | 12 | 21 | +7.5 [−0.6, +15.6] |

**Kept:** the mix is the team's act mode (IX.1 done). It is better or level everywhere except the D2 segment (within
noise), significantly better at the D2 arrival and at the D3 arrival from the start. Argmax has to be re-measured
for every new checkpoint (the smoke in §2 lost it in 3 updates) and checked in chains, not only in single rooms.

## 4. One-specialist A/B arms on goriya (IX.2–IX.5)

`scripts/experiments/r57_better_play.sh` (branch): each arm continues `goriya/r56-rust-dg` for 245 iterations
(1,003,520 steps) at 4 envs, seed 57, run while r56 trained (shared machine), then the same paired evaluation.
Arms: `ctl` (plain continuation), `ent` (`--ent-anneal 1.0`), `outcome` (`--shaping-anneal 50`, 0.3× learning
rate, `frame_penalty` 0.000125), `bon` (`--best-of-n 16 --best-keep 2 --best-every 20 --best-coeff 0.2`), `sil`
(`--sil-coeff 0.1`). Δ = arm − ctl, paired per episode (1500 episodes):

| arm | act | Δ success pp [95% CI] | Δ hearts lost / episode [95% CI] | Δ frames / episode |
|---|---|---|---|---|
| ent | sample | −0.1 [−1.4, +1.2] | +0.033 [−0.005, +0.071] | −23 [−36, −10] |
| outcome | sample | −0.9 [−2.4, +0.5] | +0.038 [−0.001, +0.077] | +4 [−10, +18] |
| bon | sample | +0.7 [−0.6, +1.9] | +0.028 [−0.009, +0.065] | −37 [−49, −25] |
| sil | sample | +1.2 [−0.0, +2.4] | −0.035 [−0.069, −0.000] | −25 [−38, −12] |
| ent | argmax | −0.1 [−1.5, +1.3] | +0.042 [+0.007, +0.077] | +3 [−14, +20] |
| outcome | argmax | −1.9 [−3.4, −0.5] | +0.012 [−0.023, +0.047] | −3 [−21, +14] |
| bon | argmax | −1.4 [−2.9, +0.1] | +0.047 [+0.012, +0.081] | +30 [+8, +53] |
| sil | argmax | +1.2 [−0.1, +2.5] | −0.032 [−0.064, +0.001] | −8 [−25, +8] |

One training seed: these intervals hold the evaluation noise only, not the run-to-run spread of training. Seeds 58
and 59 of every arm show that spread is the larger one: the control alone ranges 90.5–94.5% (sampled) between seeds.
All five arms over seeds 57, 58, 59 (1500 episodes each):

| arm | act | success per seed | mean | hearts lost / episode per seed | mean |
|---|---|---|---:|---|---:|
| ctl | sample | 94.5, 90.5, 93.3 | 92.8 | 0.512, 0.652, 0.573 | 0.579 |
| ent | sample | 94.4, 94.9, 93.3 | 94.2 | 0.545, 0.490, 0.584 | 0.540 |
| outcome | sample | 93.6, 95.6, 93.7 | 94.3 | 0.550, 0.484, 0.518 | 0.517 |
| bon | sample | 95.2, 94.1, 93.5 | 94.3 | 0.540, 0.536, 0.529 | 0.535 |
| sil | sample | 95.7, 94.8, 94.2 | **94.9** | 0.477, 0.504, 0.493 | **0.492** |
| ctl | argmax | 94.3, 91.1, 93.1 | 92.8 | 0.450, 0.623, 0.495 | 0.523 |
| ent | argmax | 94.2, 95.1, 82.9 | 90.7 | 0.492, 0.448, 0.564 | 0.501 |
| outcome | argmax | 92.3, 93.9, 93.7 | 93.3 | 0.462, 0.425, 0.468 | 0.452 |
| bon | argmax | 92.9, 94.9, 93.1 | 93.6 | 0.496, 0.497, 0.487 | 0.494 |
| sil | argmax | 95.5, 93.9, 93.6 | **94.3** | 0.418, 0.456, 0.441 | **0.438** |

- **SIL (IX.5) is the best arm** in both modes on both measures: every SIL seed loses fewer hearts than every
  control seed (sampled), mean +2.1 pp success and −15% hearts. Against best-of-N (IX.4) it is ahead on all four
  means, with a third of best-of-N's cost (no extra episodes). Training noise is ±2 pp, so it is confirmed on a
  second family before it goes into the team (moldorm, 3 seeds each, running).
- **Best-of-N (IX.4):** level with entropy annealing and outcome fine-tuning; faster rooms (−37 frames per episode
  in seed 57) but no hearts gain. Not kept.
- **Entropy annealing (IX.2):** argmax looped badly on seed 59 (82.9%): a policy with its entropy annealed away
  gives argmax nothing to break ties with. Not kept.
- **Outcome fine-tuning (IX.3):** level on success, second-best hearts under argmax. Not kept on its own; a
  candidate to combine with SIL later.

## 5. Gumbel expert iteration on Dodongo (IX.6)

`scripts/experiments/r58_expert_iteration.sh`: boss-dodongo `r56-rust-dg` continued 100 iterations (409,600 steps)
at 4 envs; `ei`: search on a quarter of the decisions, k=4 candidates, 8 simulations, K=2 RNG-perturbed futures,
depth 3; `ei-k1`: the same with one unperturbed future (the clairvoyant ablation). Evaluation 100 per start state
(300 episodes):

| run | seed | sampled success | hearts lost / ep | argmax success | hearts lost / ep | train time |
|---|---|---:|---:|---:|---:|---:|
| r56 (start) | – | 99.3% | 0.027 | 99.3% | 0.010 | – |
| ctl | 57 | 99.3% | 0.070 | 98.7% | 0.037 | 9 min |
| ctl | 58 | **53.7%** | 0.068 | **24.3%** | 0.035 | 9 min |
| ctl | 59 | 97.7% | 0.122 | 100.0% | 0.050 | 9 min |
| ei (K=2) | 58 | 98.3% | 0.198 | 99.0% | 0.053 | 52 min |
| ei-k1 (K=1) | 58 | 96.7% | 0.250 | 96.7% | 0.090 | 116 min |

- Dodongo starts at the ceiling (99.3%), so there is little to gain: expert iteration does not beat the start
  model or the healthy controls on hearts, at 6–13× the training time.
- The K=1 (clairvoyant) arm is worse than K=2 on every measure, as §2.3 predicts, but one seed and 300 episodes
  can't separate them.
- **Control seed 58 collapsed:** in evaluation it stuns Dodongo, hits him once and waits out the 600-step budget
  (136 of 300 episodes `max-steps`), while its training success read 0.96–1.0. A check in the training env
  (no curriculum) gives the same 57%, so the env agrees with the evaluation. Correction (review F071, Nov 10): the
  training success rate doesn't drop slow failures, it counts each episode when it ends, so a failure that runs
  into the 600-step budget shows up 1–3 rollouts after it started, plus the actor's lag and the 20-episode window;
  this run collapsed in its last few updates, which its last metric rows do show, and the rate lagged the
  collapse (the early stop counted the same lagging episodes). Logged in the backlog; fixed in the review fix-up.
  The search arm on the same seed did not collapse.
- **Goriya (IX.7)**, same search settings, 245 iterations from `goriya/r56-rust-dg`, seed 57: 6.0 h of training
  (46 steps/s; the plain continuation takes 25 min at the same 4 envs, so 14×). Sampled 89.5%, 0.793 hearts lost per
  episode; argmax 92.1%, 0.599: worse than every control seed (90.5–94.5%, 0.51–0.65) and than SIL. At this budget
  (k=4, n=8, K=2, d=3 on a quarter of the decisions) the search's targets are worse than PPO's own gradient, and a
  bigger budget costs proportionally more. **Not kept.** The machinery (`EnvPool.rollout(search=...)`, `--search`)
  stays for later (e.g. search as a test-time tool for the hard rooms rather than a teacher).

## 6. SIL on a second family; PLR, PPG and chain-aware arms on goriya (IX.5, IX.8–IX.10)

**SIL on moldorm** (same recipe, 300 episodes per start state, 900 per run):

| arm | act | success per seed | mean | hearts lost / episode per seed | mean |
|---|---|---|---:|---|---:|
| ctl | sample | 86.1, 84.4, 84.8 | 85.1 | 0.862, 0.963, 0.879 | 0.901 |
| sil | sample | 85.6, 87.1, 85.9 | 86.2 | 0.872, 0.808, 0.846 | 0.842 |
| ctl | argmax | 83.9, 79.4, 87.8 | 83.7 | 0.777, 0.838, 0.803 | 0.806 |
| sil | argmax | 82.9, 81.8, 84.1 | 82.9 | 0.808, 0.709, 0.728 | 0.748 |

Same direction as goriya, smaller: hearts −7% in both modes, success +1.1 pp sampled. **SIL is kept** (IX.5) and
goes into the team (§7).

**More goriya arms** (seeds 57–59; ctl and sil from §4 repeated for reference): `plr` (`--plr`, the paper's
β=0.1), `plr-b1` (β=1), `ppg` (`--ppg`), `exit` (`reward.exit_health_weight: 1.0`: the success terminal pays only
for health at the exit), `harvest` (half the resets from 546 goriya handoff states harvested from 240 chain runs of
the promoted team, `eval-chain --harvest`):

| arm | act | success per seed | mean | hearts lost / episode per seed | mean |
|---|---|---|---:|---|---:|
| ctl | sample | 94.5, 90.5, 93.3 | 92.8 | 0.512, 0.652, 0.573 | 0.579 |
| sil | sample | 95.7, 94.8, 94.2 | 94.9 | 0.477, 0.504, 0.493 | 0.492 |
| plr | sample | 91.0 | – | 0.790 | – |
| plr-b1 | sample | 94.3, 94.5, 94.1 | 94.3 | 0.511, 0.556, 0.526 | 0.531 |
| ppg | sample | 93.7, 92.6, 94.0 | 93.4 | 0.524, 0.609, 0.602 | 0.578 |
| exit | sample | 94.5, 94.3, 93.1 | 93.9 | 0.491, 0.478, 0.503 | 0.491 |
| harvest | sample | 94.5, 94.4, 93.3 | 94.1 | 0.546, 0.513, 0.611 | 0.557 |
| ctl | argmax | 94.3, 91.1, 93.1 | 92.8 | 0.450, 0.623, 0.495 | 0.523 |
| sil | argmax | 95.5, 93.9, 93.6 | 94.3 | 0.418, 0.456, 0.441 | 0.438 |
| plr | argmax | 91.1 | – | 0.707 | – |
| plr-b1 | argmax | 93.7, 93.1, 94.3 | 93.7 | 0.459, 0.523, 0.477 | 0.487 |
| ppg | argmax | 93.5, 91.8, 92.8 | 92.7 | 0.493, 0.524, 0.474 | 0.497 |
| exit | argmax | 92.4, 92.1, 93.7 | 92.7 | 0.481, 0.444, 0.428 | 0.451 |
| harvest | argmax | 93.7, 93.9, 92.4 | 93.3 | 0.510, 0.452, 0.532 | 0.498 |

Chains with each goriya swapped into the promoted team (goriya samples, 120 runs per seed; arrivals per 120):

| goriya | full → after D2 | full → D3 | D2 segment | D1 segment |
|---|---|---|---|---|
| r56 (team) | 67 | 21 | 85 | 102 |
| ctl | 81, 69, 68 = 72.7 | 25, 19, 13 = 19.0 | 85, 78, 85 = 82.7 | 105, 100, 103 = 102.7 |
| sil | 72, 66, 78 = 72.0 | 25, 16, 18 = 19.7 | 83, 90, 80 = 84.3 | 109, 99, 108 = 105.3 |
| exit | 69, 71, 75 = 71.7 | 22, 17, 25 = 21.3 | 81, 91, 88 = 86.7 | 103, 102, 108 = 104.3 |
| harvest | 68, 75, 68 = 70.3 | 18, 25, 19 = 20.7 | 92, 81, 76 = 83.0 | 94, 109, 102 = 101.7 |

- **PLR (IX.9):** β=0.1 is clearly worse (hearts 0.79 vs the controls' 0.51–0.65): with 12 start states it puts most
  draws on one state and the rest are forgotten. β=1 is level with the other arms. Not kept.
- **PPG (IX.10):** level with the control (explained variance 0.6–0.7 was not the bottleneck). Not kept.
- **Chain-aware (IX.8):** `exit_health_weight` 1.0 loses as few hearts as SIL, at the control's success; harvested
  handoff states change nothing in single rooms. In chains, one goriya swapped in moves no segment by more than the
  seed-to-seed spread (±6 arrivals) for any arm, SIL included: a single family's few hearts don't show in 120-run
  chains. Not kept on its own; `exit_health_weight` is a candidate to stack on SIL if the SIL team helps.

## 7. The SIL team (IX.11)

SIL is the only arm kept (§4, §6). Every family with something to gain (all but ow-start, ow-leever, wallmaster and
boss-dodongo) continued from its promoted r56 run for 245 iterations with `--sil-coeff 0.1` at 8 envs, seed 60
(`r57-sil-s60`), two streams, 3.9 h; then the paired evaluation against the r56 run in the family's act mode.
Promotion rule: success not significantly worse, mean success at most 0.5 pp lower, fewer hearts lost:

| family | act | Δ success pp [95% CI] | Δ hearts lost / episode [95% CI] | verdict |
|---|---|---|---|---|
| ow-octorok | argmax | 0.0 [−0.4, +0.4] | −0.007 [−0.015, −0.000] | promoted |
| ow-zora | argmax | +0.7 [+0.3, +1.1] | −0.059 [−0.070, −0.048] | promoted |
| ow-moblin | argmax | +0.7 [+0.1, +1.3] | −0.001 [−0.011, +0.009] | promoted |
| ow-tektite | argmax | −0.5 [−0.9, −0.1] | −0.007 [−0.019, +0.004] | r56 kept |
| ow-lynel | argmax | +1.0 [+0.1, +1.9] | −0.062 [−0.121, −0.003] | promoted |
| ow-boulder | argmax | +0.6 [+0.1, +1.0] | −0.024 [−0.039, −0.009] | promoted |
| trap-room | argmax | +0.4 [−0.2, +1.0] | +0.004 [−0.011, +0.020] | r56 kept |
| equipment-room | argmax | +1.0 [+0.1, +1.9] | −0.067 [−0.098, −0.036] | promoted |
| d2-rope | argmax | +0.6 [−0.2, +1.3] | −0.011 [−0.022, −0.001] | promoted |
| goriya | sample | +1.1 [−0.2, +2.3] | −0.028 [−0.064, +0.008] | promoted |
| moldorm | sample | −2.0 [−6.2, +2.2] | −0.055 [−0.123, +0.013] | r56 kept |
| stalfos-keese-gel | sample | +0.4 [+0.1, +0.7] | −0.102 [−0.113, −0.091] | promoted |
| boss-aquamentus | argmax | −0.7 [−1.6, +0.3] | +0.003 [−0.005, +0.011] | r56 kept |
| darknut | sample | +0.9 [−0.1, +1.8] | −0.055 [−0.095, −0.014] | promoted |
| zol | sample | +0.8 [+0.3, +1.3] | −0.113 [−0.140, −0.086] | promoted |
| boss-manhandla | argmax | +1.7 [−2.4, +5.7] | +0.020 [−0.117, +0.157] | r56 kept |

11 of 16 promoted, each with its family's act mode (the argmax-vs-sampled check on the new runs agrees with
§3's choices). Chains with the new team (`r60-sil`, act mix, 120 runs, seeds 0–119):

| segment | r56 sampled | r56 act mix | SIL team | Δ vs act mix [95% CI] |
|---|---:|---:|---:|---|
| full → D2 | 102 | 115 | 110 | −4.2 [−10.5, +2.1] |
| full → after D2 | 55 | 67 | 72 | +4.2 [−9.0, +17.4] |
| full → D1 | 40 | 54 | 57 | +2.5 [−10.7, +15.7] |
| full → D3 | 11 | 21 | **34** | +10.8 [+0.0, +21.7] |
| full (D3 triforce) | 0 | 1 | **4** | +2.5 [−1.1, +6.1] |
| D2 | 92 | 85 | 89 | +3.3 [−8.0, +14.7] |
| after D2 | 88 | 91 | 98 | +5.8 [−5.1, +16.8] |
| D1 | 95 | 102 | **115** | +10.8 [+3.6, +18.1] |
| after D1 | 118 | 120 | 120 | 0 |
| D3 | 12 | 21 | 30 | +7.5 [−2.9, +17.9] |

**Kept.** Against the r56 baseline as first measured (all sampled), the two kept changes together: D3 arrivals from
the start 11 → 34 of 120, the full route to the D3 triforce 0 → 4 (the first full d2first runs), D1 95 → 115, D3
12 → 30. Where runs still end: D2's Moldorm (0x3e, 11 from the start) and Goriya rooms (0x5e, 0x1e), d2-rope
looping under argmax in 0x4e (10 of 120 in the D2 segment), and D3's Darknut room 0x59.
