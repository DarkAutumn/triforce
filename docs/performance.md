# Training and evaluation throughput (Nov 5–10)

How fast the Rust port trains and evaluates compared with the old stack, what limited it, what changed, and what
limits it now.
Everything here is real `triforce.train` runs on the workstation: i9-10900X (10 cores, 20 hardware threads),
RTX 4090, idle unless noted.

## Review fix-up re-profile (Nov 10)

About 155 commits fixed the whole-repo code review (`ab17fc5..aa40d8e`, `target/review-progress.md`). Several add
work per step by design (the jump-table object classifier F067, exact attack masks with measured `asm:Link_BorderBounds`
F182/N006, legal-step target descent F132, per-room bomb masks F016, the mouth flag F094, off-grid alignment and the
held attack press F160/N003, marginal look-ahead credit F054, the linger rebaseline F154, the selector F185) or per
iteration (in-flight metrics and cohorts F071, GPU recompute F019, `tensor_split` F167, the all-or-nothing snapshot
F084). Before the retrain everything was profiled again, old (`ab17fc5`, worktree `../triforce2-better`) against new
(`aa40d8e`), with the machine partly busy (state builds, then explores) for the first half, quiet for the second.
Load-insensitive measures first: per-thread user instructions and cycles (`perf_event_open` around each
`Game.act`), `perf record` user-cycle shares, py-spy for Python; wall-clock steps/s only on the quiet machine.
Scripts are throwaway (`/tmp/perf`).

### What changed per measure

**Emulator** (`tf bench --recordings target/bench-runs`, 20 recorded runs, 91,000 headless window frames, perf
stat, two interleaved runs each): 112.917G vs 112.927G user instructions (+0.01%), 46.0G vs 46.0G cycles,
8,220–8,419 vs 8,301–8,389 frames/s. The Phase 12 zcore changes cost nothing.

**Env stepping** (`env_speed.py`'s workload: all 295 start states × 60 seeded random legal steps, 2 repeats, 2
interleaved runs per build; the same `.state` files for both; perf counters per call). Instructions are the
load-insensitive number (cycles moved up to ±20% per family with the builders running):

| family | steps (new) | frames/step old → new | Minstr/frame | Mcyc/frame | Minstr/step | Δ instr/step |
|---|---:|---:|---:|---:|---:|---:|
| stalfos-keese-gel | 10,080 | 16.92 → 17.51 | 1.87 → 1.82 | 0.79 → 0.78 | 31.6 → 31.8 | +1% |
| ow-octorok | 9,262 | 16.84 → 18.74 | 1.91 → 1.86 | 0.85 → 0.83 | 32.2 → 34.9 | +8% |
| ow-zora | 7,200 | 16.71 → 18.19 | 2.11 → 2.00 | 0.93 → 0.89 | 35.2 → 36.5 | +4% |
| darknut | 5,040 | 13.75 → 16.49 | 3.26 → 2.83 | 1.42 → 1.23 | 44.8 → 46.6 | +4% |
| ow-moblin | 5,756 | 16.16 → 17.97 | 2.36 → 2.27 | 1.06 → 1.02 | 38.1 → 40.7 | +7% |
| zol | 4,320 | 15.48 → 17.61 | 2.61 → 2.52 | 1.13 → 1.09 | 40.5 → 44.4 | +10% |
| ow-tektite | 4,320 | 17.94 → 19.55 | 1.78 → 1.70 | 0.77 → 0.79 | 31.9 → 33.3 | +5% |
| trap-room | 3,420 | 14.63 → 16.14 | 2.39 → 2.39 | 1.03 → 1.03 | 35.0 → 38.5 | +10% |
| d2-rope | 4,246 | 17.66 → 18.50 | 1.53 → 1.54 | 0.64 → 0.65 | 27.0 → 28.6 | +6% |
| goriya | 3,446 | 14.83 → 15.69 | 2.13 → 2.12 | 0.92 → 0.92 | 31.6 → 33.2 | +5% |
| ow-boulder | 2,160 | 18.23 → 16.24 | 1.75 → 1.56 | 0.77 → 0.80 | 31.9 → 25.3 | −21% |
| ow-start | 3,120 | 17.48 → 18.27 | 1.22 → 1.29 | 0.52 → 0.56 | 21.4 → 23.6 | +11% |
| other 8 families | 7,844 | | | | | −2 to +17% |
| **all** | 70,214 | **16.30 → 17.70** | **2.07 → 2.00** | **0.90 → 0.88** | **33.7 → 35.4** | **+5%** |

Per action kind (one run each, the same states and seeds; frames/step, Mcyc/step): WAIT 14.48 → 14.45, 8.89 → 8.87
(unchanged); MOVE 20.8 → 22.2 frames (other moves are legal now, F059/F170/F188); SWORD 16.99 → 18.40 frames, 14.7
→ 15.7 Mcyc and BEAMS 16.5 → 18.1 frames: the F160 walk to the grid and the N003 held press, by design (+1.4
frames per attack); BOMB 9.7 → 14.9 frames, 37 → 62 Mcyc and BOOMERANG 9.4 → 15.5 frames, 23 → 39 Mcyc: before
F182 many presses were ignored by the game (busy slot, outside `asm:Link_BorderBounds`) and returned after a few frames; now
every one is a real throw, with its look-ahead. Cycles per frame of each kind are unchanged or lower. The code above
the emulator stays negligible: in training profiles `zelda::`, `env::` and `pool::` together are 0.11–0.12 Mcyc per
step of 11–15 in both builds, the new classifier `ObjectClass::of` 0.01% of samples, `GameView::new` 0.03%.

**Training** (fresh models, CUDA, `--sil-coeff 0.1 --stop-success 0.99` as the r61 retrain, 4,096 steps per update,
seed 1, median of 30 iterations; two interleaved repeats on the quiet machine; "fixed" = with the commits below):

| config | steps/s old | new | fixed | update s old / new / fixed | frames/step old / new |
|---|---:|---:|---:|---:|---:|
| ow-zora p8 | 2,027 / 2,058 | 2,038 / 2,046 | **2,173 / 2,213** | 1.15 / 1.12 / 1.01 | 13.3–13.6 / 14.6 |
| goriya p8 | 2,232 / 2,188 | 2,205 / 2,216 | **2,379 / 2,379** | 1.14 / 1.14 / 0.99–1.03 | 14.3 / 14.7 |
| ow-start p8 | 2,544 / 2,516 | 2,639 / 2,617 | **2,823 / 2,821** | 1.13 / 1.13 / 1.00–1.05 | 17.4–18.8 / 18.6 |
| ow-zora p16 | 1,741 / 1,699 | 1,794 / 1,744 | **2,588 / 2,552** | 2.10–2.15 / 2.13 / 1.17–1.23 | 13.5 / 14.6 |
| goriya p16 | 1,814 / 1,783 | 1,851 / 1,846 | **2,714 / 2,680** | 2.0 / 2.1 / 1.19 | 14.1–15.2 / 14.6 |
| ow-zora p16, no SIL | 2,722 / 2,671 | 2,691 / 2,733 | **2,841 / 2,857** | 0.63 / 0.63 / 0.65 | 13.5 / 14.2 |

The review left training speed where it was: its extra frames per step (+3–9%) were paid back by F019's GPU
recompute. Per iteration on the main thread (ow-zora p8, a timing wrapper, no profiler): the serial gap between
two rollouts (the envs idle) was 170 ms old (CPU recompute 148 ms, GAE 14 ms) and 66 ms new (recompute 37 ms, GAE
20 ms, `EpisodeTracker.update` 0.23 ms); the F084 snapshot (`deepcopy` of weights and Adam state, now on every
update) is 11–14 ms per update, off the critical path; `tensor_split` costs nothing visible; `on_iteration`
(metrics, logs, events) 20–30 ms as before. Whole-run user cycles per step (perf, ow-zora p8): zcore 10.9 → 11.8
(the extra frames), the Rust policy forward 1.05–1.08 unchanged, torch/CUDA 1.4 → 1.05 (the recompute).

**Early-stop confirmation** (F071: 50 sampled episodes per seed state, `--parallel 8`, timed with the promoted
models): ow-start (13 states) 4.5 s, ow-zora (30) 9.2 s, goriya (15) 11.3 s, d2-rope (18) 11.1 s,
stalfos-keese-gel (42) 36.7 s, boss-dodongo (3) 2.7 s; it shares the CPU with the overlapped next rollout. It runs
only once the cohort rate has held `--stop-success` for `--stop-patience` iterations, and a failed check resets the
streak, so at most one check per 5 iterations: on a plateau just under 0.99 a check per ~9 s of training (5
iterations at p8) doubles the time there for ow-zora (9 s per check) and quintuples it for stalfos-keese-gel (37 s).
Since `54b862a` a check that can no longer reach `--stop-success` ends there (same decision, `aborted` in the event):
a failing check at 0.99 took 40.5 → 22.2 s for stalfos-keese-gel (1,541 of 2,100 episodes) and 11.3 → 6.6 s for
goriya (469 of 750), the same seed, two runs each; the jobs go state by state, so the failures of a weak state late
in the order end it late.

**Chains and evaluation** (`eval-chain --runs 20 --parallel 10 --from 1_73s_0` under py-spy): 20.4 of the main
thread's 24.3 s wait in the native pool; `checkpoint_sha1` 0.26 s once (cached per path, mtime and size),
`ChainOutput.add` 0.07 s for all 20 runs (one appended line each), the final sort 0.02 s; `code_version` and
`data_hash` are cached. Nothing hashes or rewrites a big file per run or episode. `evaluate` writes its result once.

**Pool:** the F082/F155 stop and halt flags are two relaxed atomic loads per step; F005's buffer checks and F047's
policy input check run once per `rollout` call; the policy's per-forward finiteness check covers ~600 floats
against a 0.3 ms forward (the policy share of cycles is unchanged).

### Fixes

| commit | change | effect |
|---|---|---|
| `ca17578` | **SIL replay ring on the training device.** Every PPO minibatch drew its SIL sample with a CPU gather (OpenMP) and a pageable copy to the GPU, which stalled the GPU queue and kept torch's OpenMP pool spinning (`gomp_barrier_wait_end`) on the cores the env threads need: 30–40% of all user cycles with the machine busy, 3.5–4 Mcyc per step quiet. The draw stays on CPU (the same `randperm`), the gather runs on the device; samples are bit-identical (checked on CPU and CUDA). The ring takes 272 MB of GPU memory per training. | update 2.1 → 1.2 s at p16, 1.12–1.14 → 1.0 s at p8; OpenMP spin 3.7 → 0.25 Mcyc/step; with `3e868b5` SIL steps/s +45% at p16; alone +4% at p8 quiet, +15–22% at p8 with the builders running |
| `3e868b5` | **Next rollout before the buffer is readied** (without PLR): `Collector.take` then `start`, then `ready` (recompute, returns, metrics) while the envs already run; PLR still readies first (its weights need the advantages). Same weights, same draws: seeded CPU training is bit-identical. | the 66 ms gap per iteration goes; +2–5% steps/s (ow-zora/goriya p8, also without SIL) |
| `af891de` | **No look-ahead that can't credit anything:** the prediction follows only the firing state's live enemies (boomerang and arrow: also its floor items), so without any it is empty and was emulated for nothing (snapshot, up to the weapon's whole flight, restore). | random-play bench (2 interleaved runs): BOMB −4.5%, BEAMS −3.7%, BOOMERANG −3% cycles per action, −0.3% instructions overall (there, rooms nearly always hold an enemy); `state_walk` 33,850 steps, 0 differ |

`OMP_WAIT_POLICY=PASSIVE` and `OMP_NUM_THREADS=1` (each tried alone) removed the spin as well; on top of `ca17578`
PASSIVE gained nothing measurable and would change process-wide behaviour, so neither was kept.

### What's left after the fix-up, by expected gain

1. **Emulation** (zcore ~75% of rollout cycles), unchanged; see [zcore speed](#zcore-speed-nov-7).
2. **The rollout barrier** (workers idle 13–32% at the end of a rollout): more envs than threads with
   work-stealing.
3. **The early-stop confirmation** on a plateau near the threshold: aborted checks (`54b862a`) still play ~60–75% of
   their episodes; interleaving the states in the job order (same (state, seed) pairs) would reach the deciding
   failures sooner, if the episodes' noise keys don't depend on the job order.
4. **Weapon look-ahead** in fights: ~27–53 Mcyc per beam/bomb/boomerang (45–90 emulated frames) against 9–11 for
   the action's real frames. The deferred look-ahead (`docs/rust-actor.md` item 4) or ending it once every enemy it
   could credit is dead.
5. **The PPO update** (launch-bound, ~1.0 s at p8, now off the critical path at p8 but close to the rollout at p16):
   fused Adam, fewer syncs per minibatch.
6. Small: the residual OpenMP spin from once-per-update CPU ops (0.25 Mcyc/step), the 12 ms snapshot per update,
   and the SIL ring's 184 MB `grid` stored as int64 (int16 would do; memory only).

## The Rust actor (Nov 9)

Since `docs/rust-actor.md` the policy acts inside the pool (Rust CPU forward, one `rollout` call per iteration) and
the sections below about the Python forward, `recv`/`send` and CUDA graphs are history.

**Training,** `scripts/train_bench.sh LABEL 25 fresh` (fresh models: at the time the promoted models were
pre-refactor and didn't load, so these numbers aren't comparable with the warm-started tables below; same machine
load):

| config | synchronous (`ff194b8`) | rollout overlapped with the update | gain |
|---|---:|---:|---:|
| goriya p8 | 1,551 | 2,247 | +45% |
| d2-rope p8 | 1,640 | 2,185 | +33% |
| ow-moblin p12 | 1,761 | 2,135 | +21% |
| ow-zora p16 | 2,138 | 2,570 | +20% |

- **Where an iteration went (synchronous, goriya p16):** rollout ~47%, PPO update ~35% (0.8 s: 40 minibatch
  steps, ~17k kernel launches; launch-bound, the GPU itself busy ~0.27 s), torch recompute of log-probs and
  values on the GPU ~8% (`charts/recompute_seconds`).
- **Overlap (Lee, Nov 9: "yes"):** `PPO.train` starts the next rollout on a thread (the pool releases the GIL)
  with the weights from before the update, into the second of two buffers, then updates. The rollout runs one
  update behind. `logp_old` stays the learner's recomputed one: with the acting policy's log-prob the KL early stop
  spent its budget on the lag (1 epoch per update) and a bandit learned half as fast; recomputed, it learns as
  fast as synchronous. The gain is smaller with more envs because the rollout then takes every core the update's
  launch thread wants.
- **Policy forward** (`ff194b8`): 0.77 → 0.31 ms per observation, one thread. The conv is a direct 3×3 with the
  output channels in AVX registers; Linear does four rows per pass. Same bits on the AVX and portable paths
  (tested). Rollouts +11–17%. The forward was ~18% of rollout CPU before this change, emulation ~67%.
- **Rollout barrier:** every env takes exactly `T` steps, and per-env rollout times differ, so workers idle
  13–32% of a goriya p16 rollout waiting for the slowest env (measured per env). 32 envs on 16 threads: +9%; 20
  envs on 20 threads: +11% (rollout alone). Candidate: more envs than threads with work-stealing (the deferred
  look-ahead was not built, `docs/rust-actor.md` item 4; the look-ahead is 12–35% of step time in beam/bomb rooms).
- **Tried, not kept:** `torch.compile` of the update's `evaluate` (0.42–0.59 s vs 0.50–0.82 s per update, inside the
  noise, plus ~40 s compile); `cudnn.benchmark` (no gain); building everything with `target-cpu=x86-64-v3`
  (the forward gained a little, the emulator nothing).

**Evaluation** (look-ahead off, untrained policy, `evaluate_specialist`, 16 envs): goriya 3,275 steps/s, ow-octorok
4,467 steps/s.

## Results

Settled steps/s for the same env counts (higher is better):

| config        | old stack (logs) | new stack, lockstep | Nov 5 (`b73fc84`) | Nov 7, zcore work (`76272ea`) | Nov 7 / old |
|---------------|-----------------:|--------------------:|------------------:|------------------------------:|------------:|
| goriya p8     |          452–471 |                 222 |             1,246 |                         1,624 |        3.5× |
| d2-rope p8    |          492–536 |                 199 |             1,119 |                         1,466 |        2.8× |
| ow-moblin p12 |          678–752 |                 261 |       1,563–1,607 |                         2,093 |        2.9× |
| ow-zora p16   |          902–936 |                 304 |             1,631 |                         2,177 |        2.4× |

"Nov 5" is `scripts/train_bench.sh route-cache 25` (ow-moblin was run three times), "Nov 7" is
`scripts/train_bench.sh zcore-final 25` after the emulator work in [zcore speed](#zcore-speed-nov-7). Run-to-run
noise on these benches is about ±4%: the curriculum's shared progress depends on env completion order, so even
seeded runs don't play the same episodes twice. These env counts are the historical ones, not the best:
[How many envs](#how-many-envs---parallel) recommends `--parallel 16` for a run that has the machine to itself.

## How it's measured (history: before the Rust actor, Nov 9)

- **Old stack (stable-retro/FCEUmm, one subprocess per env, each running its own CPU inference):**
  `scripts/old_stack_sps.py` (deleted since) reads every `logs/*/<specialist>.jsonl`, takes the `sps` of each run's `progress`
  events, drops the first 10% (startup) and reports the median per run. The table above lists every run with the
  same specialist and env count (goriya p8: `r17-goriya`, `r19-goriya-more`; d2-rope p8: `r34-east-key`,
  `r35-east-curriculum`; ow-moblin p12: five `r16`–`r26c` runs; ow-zora p16: `r3-baseline`, `r3c-curriculum`;
  Nov 2–4, CUDA, 4,096 steps per update). Rounds often ran two at once on the same 20 threads, so these numbers
  are a floor for what the old stack could do alone, not a precise figure.
- **New stack:** `scripts/train_bench.sh LABEL 25` runs four real trainings one after the other (warm start from
  the promoted model, its yaml and curriculum, CUDA, 4,096 steps per update, `--seed 1`) and reports the median of
  steps per wall-clock second between consecutive `progress` events after the first five iterations. Every number
  below comes from it or from the per-iteration `charts/rollout_seconds` / `charts/optimize_seconds` metrics.
- **Long-haul profiles:** `scripts/profile_train.sh LABEL ITERATIONS WARMUP PERF_SECONDS` trains ow-zora p16 under
  py-spy for the whole run (every Python thread, idle waits included, 50 Hz) and attaches `perf record -g`
  (199 Hz) to the Python process for a window well after startup. Outputs in `target/perf/profile-LABEL/`
  (`pyspy.speedscope.json`, `perf.data`, `perf_top.txt`). Call graphs inside the native module need a build with
  frame pointers (`RUSTFLAGS="-C force-frame-pointers=yes" maturin develop --release`); only one profile below
  used it.
- **Rollout timeline:** `scripts/rollout_timeline.py SPECIALIST N` (script removed with the per-step path) collects rollouts with the promoted model (no
  PPO updates) and prints, per rollout, the share of time the main thread was blocked in `recv`, the number of
  forwards and their mean batch, and when the first, median and last env finished its `T` steps.

## The lockstep baseline

Before this work `ppo.Collector` stepped the pool in lockstep: `recv(min_ready=N)` → store → one batched
`network.evaluate` → `send` to all. ow-zora p16 took 13.1 s per rollout (the old stack: 4.7 s) and 0.3 s per
PPO update, so the rollout was everything.

Long-haul profile (`profile-baseline`, 120 iterations): the main thread spent 81% of the run blocked in
`Collector._recv` and ~14% in `network.evaluate` (CUDA, launch-bound at batch 16: ~2.7 ms per call). perf over
2 minutes mid-run: each pool thread had 32–41% of the samples a busy thread would get; the rest was the barrier,
waiting for the slowest env of each step and for the Python round trip. Steps vary a lot in cost (a 4-frame
sword swing vs. a 32-frame move vs. a reset), so every lockstep step paid for its slowest env. Inside the pool
threads: `Bus::clock_frame` 48%, `render_span` 20% (headless; span bookkeeping, not pixels), route search ~1%,
SipHash ~1.3%.

## Changes (history: before the Rust actor, Nov 9)

### 1. Asynchronous collection, per-(env, step) noise, CUDA-graph forward (`47cf816`)

- The collector hands every ready env its next action at once (`recv(min_ready=1)`), each env fills its own rows
  of the env-major buffer, and a rollout ends when every env has its `T = 4096 / N` steps (finished envs wait).
  Bootstrap values `V(final_obs)` are computed in one batch after the rollout; GAE is vectorized over envs;
  episode events are folded into the metrics in env order after the rollout.
- **Determinism.** Which envs happen to be ready together must not change anything. Actions are sampled by
  Gumbel-max with noise drawn once per rollout as an `[N, T, 15]` block from the collector's own generator
  (seeded from the torch RNG, so `--seed` fixes it) and indexed by (env, step); the forward always evaluates a
  fixed batch of one row per env, so a row's numbers depend only on its own observation and noise.
  `test_train_cli` (threads 1 vs 3, CPU and CUDA) and `test_ppo::test_async_schedule_does_not_change_training`
  (a pool that returns random subsets of the ready envs) check it. The CUDA PPO update was never bit-reproducible
  (atomic-add backward kernels), so long seeded CUDA runs still drift apart after some iterations; CPU runs
  repeat exactly.
- **Inference.** Batch-16 forward of the ow-zora model with real observations: CPU 3.0 ms (1 thread) / 2.1 ms
  (4 threads), CUDA eager 2.7 ms, CUDA graph 0.37 ms including the copies and the sync. The rollout forward is a
  private eval copy of the network (weights copied in place before each rollout) captured once as a CUDA graph.
  Measured inside training (ow-zora p16 rollout): CPU inference 3.72 s, eager CUDA 3.41 s, CUDA graph 2.39 s.

Result: goriya 222 → 1,070, d2-rope 199 → 981, ow-moblin 261 → 1,422, ow-zora 304 → 1,565 steps/s.

### 2. Cheaper main-thread bookkeeping (`1e472fc`)

The long-haul profile of change 1 (`profile-async`, 150 iterations) showed 15.6% of all samples in libgomp's
barrier spin (`gomp_barrier_wait_end`, `gomp_team_barrier_wait_end`) on OpenMP worker threads. The per-step torch
CPU ops on the rollout buffer (`index_put` stores) woke torch's 20-thread OpenMP pool, and the woken threads spun
on cores the env threads needed (after those stores moved to numpy the spin fell to 0.4% of samples). The py-spy
profile also showed the main thread blocked in `recv`
only 12% of the run (lockstep: 81%), so it had become the bottleneck.

- Buffer stores go through numpy views of the same memory (no OpenMP, a few µs each).
- The forward's staging (every observation key, the action mask and the noise) is one pinned byte buffer with
  typed views; the CUDA graph's first node copies it to the device and its last node copies actions,
  log-probs, entropies and values back, so a forward is one `graph.replay()` plus an event.
- The buffer store for the envs being evaluated runs while the forward is on the GPU.

Result: goriya 1,070 → 1,234, d2-rope 981 → 1,103, ow-moblin 1,422 → 1,671, ow-zora 1,565 → 1,610 steps/s
(ow-zora rollout 2.39 s → 2.10 s). The libgomp spin dropped to 0.4% of samples.

### 3. Route search cache (`b73fc84`)

A frame-pointer perf profile of ow-zora (`profile-fp`, 60 s with call graphs) put 86.7% of env-thread time in
`Executor::step` (emulating the real frames of each action; the weapon look-ahead and resets are ~2% of the
`clock_frame` callers) and 3.3% in `GameMap::find_route`, called through `GameCompletion::closest_room` on every
step (a key-aware Dijkstra per candidate room for the sword and triforce targets). `closest_room` results are now
remembered per episode while the collected key rooms and opened doors stay the same (cleared whenever either
changes). Seeded no-curriculum gameplay is bit-identical with and without the cache (d2-rope, goriya, ow-zora).

Result: −1% user CPU on a fixed seeded ow-moblin workload (237.2 s → 234.6 s over 10 iterations); wall-clock
steps/s within noise (goriya 1,246, d2-rope 1,119, ow-moblin 1,563–1,607, ow-zora 1,631). Route search and SipHash
no longer show up in the profile.

## Where the time goes now (history: before the Rust actor, Nov 9)

Final long-haul profile (`profile-final`, ow-zora p16, 150 iterations under py-spy, perf for 150 s from 150 s in;
rollout 2.17 s, update 0.38 s per iteration with the profilers attached):

- **Env threads** get 58–64% of the samples a busy thread would; 92% of their samples are in zcore functions
  (`clock_frame` 52% self, `render_span` 18.5%, the rest CPU instructions, PPU sync, DMA, APU) and under 1% in
  the game/env/pool code above the emulator.
- **Main thread:** blocked in `recv` 29% of the run, waiting for the GPU forward (`Event.synchronize`) 26%,
  bookkeeping and graph launches 27%, the PPO update 16%.

Rollout timelines of the final code (`scripts/rollout_timeline.py`, 8 rollouts each, typical values):

| config        | rollout | forwards | mean batch | main blocked in recv | env finished: first / median / last |
|---------------|--------:|---------:|-----------:|---------------------:|------------------------------------:|
| goriya p8     |  3.1 s  |    3,580 |       1.15 |                 50%  |                    2.2 / 2.7 / 3.1 s |
| d2-rope p8    |  3.5 s  |    3,660 |       1.12 |                 52%  |                    2.6 / 3.0 / 3.5 s |
| ow-moblin p12 |  2.1 s  |    3,180 |       1.29 |                 37%  |                    1.5 / 1.8 / 2.1 s |
| ow-zora p16   |  2.0 s  |    2,930 |       1.40 |                 32%  |                    0.9 / 1.7 / 2.0 s |

A temporary timer in the pool (not committed) split each env's time per step into executing and waiting:
goriya p8 4.7 ms executing, 1.65 ms waiting; ow-zora p16 5.75 ms executing (16 threads share 10 cores, so each
runs slower), 2.5 ms waiting. Of the waiting, about 1.0 ms (goriya) / 1.5 ms (ow-zora) per step is the
end-of-rollout tail and the rest, 0.65 / 1.0 ms, is decision latency: the GPU forward (~0.33 ms; 168 kernels,
of which ~0.09 ms are the copies), Python bookkeeping (~0.2 ms) and queueing behind other envs' forwards.

For scale, the same pools stepped with random legal actions and no network at all reach 2,384 steps/s
(ow-zora p16, 13.5 frames/step) and 2,002 steps/s (goriya p8, 14.4 frames/step); the trained policies take
longer actions (17.6 and 13.6 frames/step).

## Tried and not kept (history: before the Rust actor, Nov 9)

- **CPU inference** (async): ow-zora rollout 3.72 s vs 2.39 s with the CUDA graph.
- **Eager CUDA inference** (async): 3.41 s vs 2.39 s.
- **Two forwards in flight** (a second staging buffer and graph; after launching one forward the loop polls the
  pool and stages the next batch before blocking on the first): no change (ow-zora rollouts 1.96–2.10 s vs
  1.96–2.06 s; goriya 2.95–3.56 s vs 3.03–3.58 s). Decision latency is dominated by the GPU's own latency, which
  this can't hide for an env waiting on its own forward.
- **Copies as graph nodes** instead of separate `copy_` calls: launch+wait 0.342 → 0.345 ms, within noise; kept
  because it makes `launch` two calls.
- Not tried, on purpose: `torch.compile` or reduced precision for the rollout forward (fewer kernels, but the
  rollout's log-probs would no longer match the update's forward bit for bit); swapping SipHash for a faster
  hasher (below 0.4% once the route cache landed).

## zcore speed (Nov 7)

The Nov 5 profiles put 92% of env-thread time in zcore, so the next step was the emulator itself. The rule was
perfect accuracy: every change had to leave the game's behaviour bit for bit as it was.

### Method

- **Accuracy gate** (every change, before any timing; `target/opt/gate.sh`, gitignored):
  - `tf ladder-diff` against goldens of the full movie corpus taken before the work (63 movies then, 84 today; per-frame digests
    of RAM, PRG-RAM, CHR, CIRAM, palette, OAM, pixels and the CPU, PPU, mapper and APU state): 0 diverge;
  - the zcore and tools tests;
  - `scripts/state_walk.py`: every start state (481) stepped through the game path with seeded random legal actions,
    18,728 per-step digests against the pre-work walk: 0 differ;
  - a seeded CPU training run whose metrics digest must be identical.
- **Core benchmark:** `tf bench --recordings target/bench-runs` replays 20 runs recorded with the promoted team
  (`scripts/record_bench_runs.py`, then: full game on both routes, post-D1 and D2 segments; today it records the
  route's six segments) and times 10 windows of 500 frames spread over each run's whole length, headless and
  rendered, reported per level (L0 overworld, L1/L2 the dungeons). The corpus agent movies of the time couldn't be
  used for speed (recorded before hardware timing, they desynced within a few thousand frames); the corpus now
  holds re-recorded chain runs, which `tf bench --agent` replays.
- **Env benchmark:** `scripts/env_speed.py` loads every start state and steps it through `Game.act` with seeded
  random legal actions (the game path training pays for: frame skip, look-ahead, observations), frames/s per room
  family.
- A/B: both builds interleaved, two repeats; `perf stat` user cycles for close calls (wall-clock frames/s moves
  ±3% on the core bench and up to 9% per family on one env repeat).

### Changes

| commit | change | effect |
|---|---|---|
| `bd71234` | **Idle-loop skip.** 90% of the instructions in a gameplay frame were Zelda's `JMP $E45B` to itself, waiting for the NMI. A self-jump now applies whole loop iterations (3 cycles each) up to the next PPU/APU/IRQ deadline at once. | 3-movie corpus bench 4.8k → 12.4k headless frames/s |
| `285bc5f` | Idle check only after a self-JMP; `next_event_clock` without a modulo | −12% cycles/frame |
| `10089fb` | `reload_bg_shifters` stores reordered (a store-forwarding stall) | −1.5% cycles |
| `415124c` | **Per-opcode monomorphized dispatch:** a 256-entry table of `Bus::exec::<OP>` functions, the addressing mode a constant in each | recordings −5.8% cycles (~+15% frames/s); env 3.99k → 4.28k frames/s |
| `62377ce` | **Sprite-range cache:** sprite evaluation tested every sprite's Y on every line (~15k compares a frame); `Ppu::spr_range` keeps a per-line bitmask of in-range sprites, rebuilt after any OAM write or sprite-height change | recordings −10% cycles, +13% frames/s; env +11% |
| `76272ea` | **Background tiles in u64 lanes:** a tile row without sprites renders its 8 pixels' palette indexes in the byte lanes of a `u64` (spread pattern bits, clip and palette masks) instead of per-pixel shifts and compares | rendered frames −15.5% cycles; training renders few frames, so env flat |

### Before and after

Env benchmark per family (`env_speed.py`, mean of 2 interleaved repeats; "before" is the module at `025e59e`,
before any of the work). Both builds play exactly the same 166,577 frames and 10,811 steps.

| family            | frames | before frames/s | after frames/s | speedup |
|-------------------|-------:|----------------:|---------------:|--------:|
| ow-octorok        | 31,912 |           2,816 |          4,634 |   1.65× |
| stalfos-keese-gel | 34,250 |           3,346 |          5,686 |   1.70× |
| ow-zora           | 23,989 |           2,625 |          4,003 |   1.52× |
| d2-rope           | 18,159 |           2,630 |          4,063 |   1.55× |
| ow-moblin         | 14,562 |           2,303 |          3,186 |   1.38× |
| ow-leever-peahat  | 12,501 |           2,718 |          4,733 |   1.74× |
| goriya            | 10,955 |           2,952 |          4,491 |   1.52× |
| ow-start          |  9,592 |           3,491 |          6,776 |   1.94× |
| wallmaster        |  3,680 |           3,223 |          5,763 |   1.79× |
| boss-dodongo      |  2,343 |           2,051 |          3,177 |   1.55× |
| boss-aquamentus   |  2,286 |           2,573 |          3,750 |   1.46× |
| other             |  2,348 |           3,752 |          6,987 |   1.86× |
| **all**           | 166,577 |       **2,832** |      **4,526** |  **1.60×** |

Mean state load fell from 3.4 to 1.7 ms. The core bench over the recorded runs, by level, at `3de6f4b` (the first
commit with it; the idle-loop skip already in) and at `76272ea` (mean of 2 and 4 runs):

| level | headless `3de6f4b` | headless now | rendered `3de6f4b` | rendered now |
|---|---:|---:|---:|---:|
| L0 (29,500 frames) | 6,299 | 7,740 | 2,740 | 3,648 |
| L1 (24,500 frames) | 5,283 | 6,557 | 2,601 | 3,467 |
| L2 (37,000 frames) | 5,090 | 6,131 | 2,442 | 3,172 |
| all | 5,483 | 6,696 | 2,575 | 3,392 |

### Where the time goes now

- **Env level:** perf over the env benchmark: 94.7% of samples in the native module, 1.3% in the Python
  interpreter, under 1% in numpy; snapshot/restore and the look-ahead don't register apart from the frames they
  emulate. Training is headless almost throughout: only each action's observed frame is drawn.
- **Inside zcore** (headless recording windows; callgrind, perf): 1.5M host instructions per frame at IPC 2.48,
  1.2% branch misses and few L1 misses, so the cost is instruction count, not stalls. The profile is flat:
  `render_span` ~29% (per-line fetches, about 13k CHR reads a frame, most of them the eight sprite-slot fetches
  every line; sprite evaluation; scroll steps), the per-cycle bookkeeping inlined into `clock_frame` ~18%
  (master/APU counters, deadline compares, interrupt polling), the opcode functions ~25% together, PPU sync 3%,
  DMA 1.5%. No single item is above ~5% of self cost.

### Tried and not kept

- **Deferring headless fetches** a span cut short mid-line (the CPU polling `$2002`) left behind, replaying them
  only if something observed them: +5% cycles. Instruction stepping hands the console back after each
  instruction, so a deferral never outlived one, and the unconditional skip it generalized gained nothing either
  (54.86G vs 54.64G cycles).
- **No PPU sync for OAM DMA's `$2004` puts in VBlank** (exact there: neither a visible nor the pre-render line,
  same open-bus frame stamp): +1% cycles; the syncs it saved were cheap idle spans.
- Not done, on purpose: skipping the eight sprite-slot fetches of a headless line (dead work for the pixels, but
  it would leave different `sprites[]` in snapshots), and threaded/tail-call dispatch (needs the unstable
  `become`).

## How many envs (`--parallel`)

`scripts/parallel_study.py` (results in `target/perf/parallel/nov7/`), at `76272ea`, one run at a time, 4,096
steps per update throughout (so `T = 4096 / N` steps per env per rollout).

**Throughput:** warm start from the promoted model, 20 iterations, median steps/s after the first 5 (as
`train_bench.sh`; ±4% noise):

| envs | ow-moblin | goriya | ow-zora |
|---:|---:|---:|---:|
| 4 | 853 | 753 | 783 |
| 6 | 1,231 | 1,278 | 1,221 |
| 8 | 1,682 | 1,663 | 1,729 |
| 10 | 2,038 | 1,826 | 2,079 |
| 12 | 1,860 | 1,982 | 2,132 |
| 16 | 2,381 | 2,190 | 2,293 |
| 20 | 2,478 | 2,457 | 2,413 |
| 24 | 2,228 | 2,189 | 2,309 |
| 32 | 2,338 | 2,453 | 2,793 |

Near-linear up to 10 envs (one per physical core), then the hyper-threads add 16–35% more by 20; more envs than
the 20 hardware threads (24, 32) don't help reliably.

**Convergence:** from scratch (default yaml and curriculum, CUDA) until the start-state-balanced training
success stays ≥ 90% for 5 iterations (`--stop-success 0.9`); seeds 1–3, all 30 runs reached it. Median of the
three seeds:

| envs | ow-leever-peahat steps | wall | ow-moblin steps | wall |
|---:|---:|---:|---:|---:|
| 4 | 242k | 349 s | 426k | 589 s |
| 8 | 250k | 201 s | 315k | 230 s |
| 12 | 209k | 139 s | 364k | 209 s |
| 16 | 242k | 145 s | 344k | 188 s |
| 24 | 261k | 160 s | 375k | 215 s |

The steps needed don't depend on the env count within seed noise (per-seed spreads of 160–350k and 280–480k), so
wall time follows throughput: it bottoms out at 12–16 envs and rises again at 24. Converging runs make fewer
steps/s than warm ones (1.7–1.8k at 16 envs): a fresh policy's episodes are short, so resets weigh more.

**Recommendation:** `--parallel 16` for a run that has the machine to itself (`train_round.py`'s default
already). Two rounds side by side: 8 each.

## Evaluation and chain runs (history: before the Rust actor, Nov 9)

`evaluate.py` and `runner.py` (`ChainPool`, also behind `run.py`, `play_grid.py` and `eval-chain`) stepped their
pools in lockstep too: `recv(min_ready=pool.active)`, one batched forward (per specialist for chains), `send`.
They then moved to the same machinery as training (since replaced by the Rust actor: `triforce/inference.py`,
`tests/test_inference.py` and `--device` for evaluation are gone; `keyed_noise` is `triforce._native.keyed_noise`):

- Ready envs act at once (`recv(min_ready=1)`); records and run results are still sorted by seed.
- Each policy (the evaluated network; each chain specialist) has a fixed-shape forward over one row per env
  (`inference.PoolInference` → `RolloutPolicy`, a CUDA graph per specialist on CUDA; CPU works the same way).
- Sampled actions use Gumbel noise keyed to the job instead of the env: the pool rows gained `job` (the eval
  episode's or chain run's seed) and `step` (decisions the episode took) columns, and `inference.keyed_noise(job,
  step)` is a counter-based (SplitMix64) function of them. A seed now plays the same episode or run whichever env
  or thread picks it up; before, sampled chain runs were reproducible only with `--threads 1`. `--deterministic`
  is zero noise (each head's argmax). Tests: `test_evaluate` (parallel/threads 3/1, 3/3, 2/2 give identical
  records), `test_runner::test_sampled_runs_do_not_depend_on_envs_or_threads` (1/1, 4/1, 4/4, 3/2), and
  `tests/test_inference.py` (a row's action doesn't depend on which rows share its batch). With the real promoted
  models, `eval-chain --route milestone1 --from 2_7ds_2 --runs 20 --parallel 20` gave identical run results with
  `--threads 1` and `--threads 20`.

`scripts/port_gate.sh` (then: 10 evaluations with `--parallel 16 --threads 16 --device cpu` and five `eval-chain`
segments; today: evaluations with `--parallel 16 --threads 16`, then six `eval-chain --runs 40 --parallel 20`
segments, no `--device`), wall time per step from the log files' modification times:

| part | lockstep (`44403af`) | async, CPU (as the gate runs) | async, `--device cuda` |
|---|---:|---:|---:|
| 10 evaluations | 291 s | 113 s | 79 s |
| goriya eval (240 episodes) | 72.8 s | 29.5 s | 18.6 s |
| stalfos-keese-gel eval (720 episodes) | 67.0 s | 20.8 s | 13.7 s |
| 5 chain segments | 368 s | 190 s | 100 s |
| `eval-chain` full-main (40 runs) | 74.0 s | 39.3 s | 18.7 s |
| `eval-chain` d2 (40 runs) | 77.2 s | 37.7 s | 22.7 s |
| whole gate | 660 s | 304 s | 179 s |

Outcomes stay in the same range (the sampling noise changed, so individual episodes differ from the lockstep
run); the CPU and CUDA async gates produced identical eval summaries and chain counts. On CPU the fixed-shape
forward of 16–20 rows costs 1.9–3.1 ms (1.2 ms for one row) against ~0.35 ms for the CUDA graph, which is why
`--device cuda` is faster still (gate evaluations 113 s → 79 s). Torch's intra-op threads made
little difference for CPU evaluation (goriya: 29.2 s default, 30.9 s with 1 thread, 26.7 s with 4). A chain
segment waits for its longest runs (40 runs on 20 envs; run lengths vary several-fold, e.g. 134–1,547 steps on
d2 seeds 100–105); how much of the chain time that tail is was not measured.

## What's left

In order of size, for ow-zora p16:

1. **Emulation.** Still most of env-thread time (95% of the env benchmark). After the Nov 7 work the zcore profile
   is flat (no single item above ~5%; see [zcore speed](#zcore-speed-nov-7)): further gains mean many small
   changes or a rewrite of the dispatch, each held to the accuracy gate. 16 env threads on 10 physical cores
   already share cores.
2. **The rollout barrier.** Every env takes exactly `T` steps and per-env rollout times differ, so workers idle
   13–32% of a rollout waiting for the slowest env (see "The Rust actor" above). More envs than threads with
   work-stealing is the candidate; rebalancing steps between envs would change what a rollout contains.
3. **The PPO update** (launch-bound: ~17k kernel launches per update, the GPU busy a third of it): a fused Adam
   and fewer host syncs per minibatch would shave part of it.
4. **Reset-bound specialists.** The round-10 r37 runs (10 envs each, two runs sharing the machine) trained
   ow-octorok/ow-zora/ow-leever-peahat/ow-moblin at 850–1,650 steps/s but ow-start at 430–470: its episodes are a
   handful of steps, so state loads, RNG/spawn randomization and the frames until Link is controllable dominate.
   Profile an ow-start run before optimizing (state decode + `restore`, `gain_control` frames, observation
   rebuild on reset).
