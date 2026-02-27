# Performance Analysis Spec: Training Iteration Throughput

## Overview

Profiled 10,000 training iterations of the `overworld` model on the `full-game` scenario,
loading from the 2.5M checkpoint (`overworld-full-game_2549760.pt`), using CUDA and cProfile.
The goal was to identify and fix per-iteration overhead without changing reward semantics.

## Baseline Profile

**Throughput**: ~141 steps/sec (70.8s wall time for 10K steps)
**CPU time in profiler**: 48.6s (17.9M function calls)

### Time Breakdown (top categories)

| Category | Time (s) | % of CPU | Optimizable? |
|----------|----------|----------|--------------|
| NES Emulator (compute_step + _apply_rotation) | 10.3 | 21% | No (external C library) |
| Neural Network (conv2d + linear + relu) | 7.6 | 16% | No (GPU compute) |
| Observation construction | 5.8 | 12% | **Yes** |
| Backpropagation | 1.0 | 2% | No (GPU compute) |
| Action sampling (Categorical + multinomial) | 2.1 | 4% | No (PyTorch internals) |
| action_to_array (retro library) | 1.0 | 2% | No (external library) |
| Equipment check (critics.py) | 1.2 | 2.5% | **Yes** |
| Room.is_loaded (torch.isin) | 1.0 | 2% | **Yes** |
| full_location (MapLocation creation) | 0.7 | 1.5% | **Yes** |
| cached_property overhead | 1.5 | 3% | Marginal |
| Other | ~17 | 35% | Mixed |

### Key Finding: ~40% of CPU time is in non-optimizable external code

The NES emulator (10.3s), neural network forward/backward (8.6s), and retro library
action conversion (1.0s) together account for 41% of CPU time. These are external C/CUDA
code and cannot be optimized at the Python level.

## Hotspots Identified & Fixes Applied

### 1. Frame Stacking — `_get_stacked_frames` (2.7s → ~1.5s)

**Problem**: Created 3 individual tensors via `torch.as_tensor(frame, dtype=float32).permute()`
then called `torch.stack()`. This was the single largest optimizable hotspot:
- `torch.as_tensor`: 1.31s (30,810 calls — 3 per step)
- `torch.stack`: 1.43s (10,270 calls)

**Fix**: Use `numpy.stack()` to batch all frames in one operation, then
`torch.from_numpy().permute().contiguous().float()` for a single conversion.
The numpy stack is much cheaper than per-frame tensor creation + torch stack.

### 2. Room.is_loaded — `torch.isin` (0.97s → 0s)

**Problem**: `Room.is_loaded` called `torch.isin(self.tiles, WALKABLE_TILES).any()` on every
access. This created a boolean tensor the size of the tile grid, checked against 40+ walkable
tile codes, then reduced with `.any()`. Called ~9,000 times per 10K steps.

**Fix**: Pre-compute `is_loaded` once at Room construction time using a Python `frozenset`
lookup. Tile data doesn't change after room creation, so the result is cached as `self._is_loaded`.

### 3. Equipment Check — `critique_equipment_pickup` (1.23s → ~0.97s)

**Problem**: 20 individual method calls to `__check_one_equipment`, each calling
`__get_equipment_change` with `getattr()` + two `isinstance()` checks. This generated
204,800 `getattr` calls and 409,600 `isinstance` calls per 10K steps.

**Fix**: Replaced 20 separate method calls with a single loop over a class-level tuple of
attribute names. The loop body is inlined, eliminating two levels of method dispatch overhead.
Both `__check_one_equipment` and `__get_equipment_change` private methods were removed.

### 4. `full_location` Property (0.74s → 0s)

**Problem**: `ZeldaGame.full_location` was a `@property` that created a new `MapLocation`
object on every access. Called 264,208 times per 10K steps (~26 times per step) across
critics, observation wrapper, objectives, and state change tracking.

**Fix**: Changed to `@cached_property`. The location values (level, location, in_cave)
don't change within a single game state, so caching is safe.

### 5. Image Observation — `_get_image_observation` (1.28s → ~0.72s)

**Problem**: `GRAYSCALE_WEIGHTS.view(1, -1, 1, 1)` created a new view tensor every call.
When normalizing, two separate operations were performed: `frames / 255.0` then
`frames * weights`.

**Fix**: Pre-computed `_GRAYSCALE_WEIGHTS_4D` and `_GRAYSCALE_NORM_WEIGHTS_4D` (weights
pre-divided by 255) as module-level constants. The normalize+grayscale path now does
a single multiply+sum instead of divide, multiply, sum.

### 6. `_get_information` Observation (0.35s → ~0.31s)

**Problem**: Created 3 separate tensors (`torch.zeros(6)`, `torch.zeros(4)`, `torch.zeros(4)`)
then concatenated them. Also called a separate `_get_objectives_vector` method.

**Fix**: Single `torch.zeros(14)` allocation with direct index writes. Inlined the
objectives vector logic and added `_assign_direction_offset()` for the source direction
at offset 6. Eliminated the concatenation and one method call per step.

### 7. MapLocation.__init__ (0.24s → reduced)

**Problem**: 294K MapLocation creations per 10K steps, each doing an assertion
`assert 0 <= level <= 9` and computing `hash()` on every `__hash__` call.

**Fix**: Removed hot-path assertion (levels are always 0-9 from NES RAM), pre-computed
hash in `__init__` stored as `self._hash`, and added `__slots__` to reduce attribute
lookup overhead.

## Remaining Performance Bottlenecks (not addressed)

These are the remaining top costs that could be addressed in future work:

1. **NES Emulator** (10.3s / 21%): External C library, not optimizable from Python.
   Parallelizing environments would amortize this.

2. **Neural Network** (8.6s / 18%): GPU computation, expected cost. Could benefit from
   mixed precision training (fp16) or model architecture changes.

3. **Categorical Distribution** (2.1s / 4%): PyTorch's `dist.Categorical` validates
   constraints on every construction (0.4s) and computes `logsumexp` (0.6s). Could
   potentially use a custom minimal distribution class.

4. **cached_property overhead** (1.5s / 3%): `functools.cached_property.__get__` checks
   the instance `__dict__` on every access. ZeldaGame's `__setattr__` override adds
   overhead. Could be replaced with explicit lazy init pattern.

5. **GAE computation** (0.36s): Sequential Python loop with data dependency. Could be
   rewritten using torch operations but the savings are small.

6. **action_to_array** (1.0s / 2%): Called 7x per step due to frame skipping. This is
   in the retro library and cannot be modified.

## Summary

| Metric | Baseline | After | Change |
|--------|----------|-------|--------|
| Specific hotspot CPU time saved | — | — | ~3.5s per 10K steps |
| Function calls eliminated | 17.9M | 16.1M | -1.8M (-10%) |
| Files changed | — | 5 | — |
| Tests passing | 205 | 205 | No regressions |
| Pylint | 10.00 | 10.00 | Clean |

**Note on wall-time measurement**: Run-to-run variance on this system is ±3-5% due to GPU
scheduling, NES emulator timing, and thread contention. The specific hotspot improvements
are validated by function-level profiling (call counts and per-call times), even when
wall-time totals show variance. The optimizations eliminate ~1.8M function calls and reduce
CPU time in targeted hotspots by 3-4 seconds per 10K steps.
