# Round 16: the Rust actor, then a fresh team (r56)

Lee, Nov 9: inference moves to Rust (one call per rollout; eval, chains and play-zelda on the same in-pool policy;
replayable plays), then profile and fix performance, then GPU vs CPU inference, then retrain the whole team from
scratch on the new stack (the old checkpoints don't load), then the better-play experiments one at a time.

## Summary

- **Rust actor** (`docs/rust-actor.md`, decisions 2026-11-09): built and merged. Rollouts in one call, eval, chains
  and play-zelda act inside the pool, replay files (`--plays`) replay bit for bit, endings count observed effects
  (eval and chains skip the look-ahead). The per-step `recv`/`send` path and `triforce/inference.py` are gone.
- **Performance** (`docs/performance.md`, Nov 9): AVX policy kernels (0.77 → 0.31 ms per observation) and the
  rollout overlapped with the PPO update (Lee: yes): training +20–45% (fresh-model bench). The deferred look-ahead
  was not built (expected under 15%). Inference stays on the CPU (it is ~10% of mass-play CPU, ~8% of training).
- **Fresh team (r56):** in progress (§1).
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

Baselines to compare with (rounds 13–15):

| segment | round | result |
|---|---|---|
| full route to the D1 triforce | 13 (before the bow) | 25/120 |
| D1 from its entrance | 14 | 108/120 (90%) |
| full route to the D1 triforce | 14 | 13/120 |
| D3 from its entrance | 15 | 18/120 |
| from 0x37 (D1 exit) | 15 | 7/120 |
| full route to the D3 triforce | 15 | 0/120 |

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

## 3. Argmax vs sampled (IX.1, §6b), r56 models as they finish

Every finished r56 model evaluated twice on the same jobs (100 episodes per start state, same seeds, so the
comparison is paired per episode): `eval.json` (sampled, from the round) and `eval-argmax.json` (`--deterministic`).
Δ = argmax − sampled; hearts lost summed over an episode's room visits.

| specialist | episodes | Δ success | Δ hearts lost / episode [95% CI] | sampled hearts lost / episode |
|---|---:|---:|---|---:|
| d2-rope | 1800 | −0.2 pp | −0.009 [−0.020, +0.002] | 0.085 |
| equipment-room | 600 | −0.7 pp | −0.102 [−0.143, −0.062] | 0.362 |
| goriya | 1500 | +0.1 pp | −0.050 [−0.085, −0.015] | 0.543 |
| moldorm | 300 | **−7.7 pp** | −0.118 [−0.188, −0.048] | 0.870 |
| ow-boulder | 900 | −0.1 pp | −0.024 [−0.040, −0.009] | 0.192 |
| ow-leever | 300 | +0.3 pp | −0.030 [−0.054, −0.006] | 0.090 |
| ow-lynel | 900 | +0.3 pp | −0.033 [−0.090, +0.023] | 0.309 |
| ow-moblin | 2400 | +0.1 pp | −0.025 [−0.036, −0.015] | 0.172 |
| ow-octorok | 3900 | +0.3 pp | −0.016 [−0.023, −0.008] | 0.122 |
| ow-start | 1300 | +0.5 pp | 0 | 0 |
| ow-tektite | 1800 | +0.4 pp | −0.021 [−0.035, −0.008] | 0.144 |
| ow-zora | 3000 | +0.1 pp | −0.033 [−0.045, −0.020] | 0.251 |
| trap-room | 1500 | −0.4 pp | −0.046 [−0.065, −0.028] | 0.391 |

Argmax loses fewer hearts in every room family with damage (significant in 10 of 12), 5–28% of the sampled loss,
at the same success rate, except **moldorm**, where it wins 7.7 pp less. Pending: the rest of the team, then the
chain check (team with the chosen modes vs all sampled) before setting `act` files.

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

One training seed: these intervals hold the evaluation noise only, not the run-to-run spread of training. Entropy
annealing, outcome fine-tuning and best-of-N show nothing better (bon and ent lose hearts under argmax; outcome
loses success under argmax). SIL is the only arm better in both modes, at the edge of significance. Seeds 58 and 59
of `ctl` and `sil` are running to measure the training spread before deciding.
