# Training and evaluation throughput (Nov 5)

How fast the Rust port trains and evaluates compared with the old stack, what limited it, what changed, and what
limits it now.
Everything here is real `triforce.train` runs on the workstation: i9-10900X (10 cores, 20 hardware threads),
RTX 4090, idle unless noted.

## Results

Settled steps/s for the same env counts (higher is better):

| config        | old stack (logs) | new stack, lockstep | new stack, now | now / old |
|---------------|-----------------:|--------------------:|---------------:|----------:|
| goriya p8     |          452–471 |                 222 |          1,246 |     2.7×  |
| d2-rope p8    |          492–536 |                 199 |          1,119 |     2.2×  |
| ow-moblin p12 |          678–752 |                 261 |    1,563–1,607 |     2.2×  |
| ow-zora p16   |          902–936 |                 304 |          1,631 |     1.8×  |

"Now" is commit `b73fc84` (`scripts/train_bench.sh route-cache 25`; ow-moblin was run three times). Run-to-run
noise on these benches is about ±4%: the curriculum's shared progress depends on env completion order, so even
seeded runs don't play the same episodes twice.

## How it's measured

- **Old stack (stable-retro/FCEUmm, one subprocess per env, each running its own CPU inference):**
  `scripts/old_stack_sps.py` reads every `logs/*/<specialist>.jsonl`, takes the `sps` of each run's `progress`
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
- **Rollout timeline:** `scripts/rollout_timeline.py SPECIALIST N` collects rollouts with the promoted model (no
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

## Changes

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

## Where the time goes now

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

## Tried and not kept

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

## Evaluation and chain runs

`evaluate.py` and `runner.py` (`ChainPool`, also behind `run.py`, `play_grid.py` and `eval-chain`) stepped their
pools in lockstep too: `recv(min_ready=pool.active)`, one batched forward (per specialist for chains), `send`.
They now use the same machinery as training:

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

`scripts/port_gate.sh` (10 evaluations with `--parallel 16 --threads 16 --device cpu`, then five `eval-chain --runs
40 --parallel 20 --device cpu` segments), wall time per step from the log files' modification times:

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

1. **Emulation.** 92% of env-thread time is zcore emulating real frames; nothing above it is worth chasing. Any
   further env-side gain has to come from zcore itself (behaviour-neutral, checked with `tf ladder-golden` /
   `ladder-diff`), and 16 env threads on 10 physical cores already share cores.
2. **The end-of-rollout tail** (~1–1.5 ms of each env's ~7–8 ms per step, 15–20%). Every env must take exactly
   `T` steps and envs finish up to a second apart (workload differences between envs plus hyper-thread pairing), so
   the early finishers idle. Rebalancing steps between envs or starting the next rollout early would change what a
   rollout contains, so it stays.
3. **Decision latency** (~0.65–1.0 ms per step). The GPU forward is ~0.33 ms of 168 small kernels; sending the
   integer observation keys as uint8/int16 would save ~0.06 ms of copy; fusing kernels needs numerics changes
   (above).
4. **The PPO update** (0.3 s per iteration, ~12% of it, envs idle): a fused Adam and fewer host syncs per
   minibatch would shave part of it.
5. **Evaluation on CPU:** `port_gate.sh` passes `--device cpu` and `scripts/train_round.py`'s evaluations use
   the default CPU; `--device cuda` takes the gate from 304 s to 179 s with the same results in this measurement.
