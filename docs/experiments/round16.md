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
