# Round 8: Rust port, data regeneration (plan step 17, paused)

Paused on Nov 5 to merge the `rust` branch into main; resumes from main. Raw logs and per-run outputs were under
`target/step17/` and `target/l6/` in the worktree and are copied to main's `target/` (gitignored).

## Done

- `start.state` from `power_on_to_start`; `entry_states build --chains 0,1,2` (chain 0/2 Dodongo entries
  `2_0es_*` needed a retry at 12,000 iterations / patience 500); D1 bow rooms (`path --source 1_43w_<c>
  --rooms 1:0x33,1:0x23,1:0x22`); 0x6d off-route (`path --source 2_7ds_<c> --rooms 2:0x6d --route
  milestone1`); `milestone1-fairy` from `0_37c_<c>` for chains 0 and 1.
- Builder fixes: the room-key guard now applies to dungeon rooms only (game.yaml lists a 'key' treasure for
  overworld 0x5a, which stopped every fairy build there); concurrent builders lock the manifest; new entries
  record the exact `entry_states` command.
- Trajectories (default budgets, then 12,000/500 for the four unsolved states): boss-aquamentus 3/3,
  wallmaster 3/3, goriya 12/12, d2-rope 21/21, ow-zora 27/27.
- `triforce/chain_hp.json` rebuilt from the five chain segments below.

## Promoted models on the new stack (20 episodes per start state, sampled, seed 0)

| Specialist | Old success | New success | Δ pts | Old health change | New health change |
|---|---:|---:|---:|---:|---:|
| boss-aquamentus | 100.0% (60) | 100.0% (60) | +0.0 | +0.692 | +0.583 |
| wallmaster | 87.5% (120) | 100.0% (60) | +12.5 | −0.417 | +0.200 |
| goriya | 76.7% (240) | 76.7% (240) | +0.0 | −0.769 | −0.767 |
| d2-rope | 95.2% (420) | 92.6% (420) | −2.6 | −0.205 | −0.238 |
| ow-leever-peahat | 100.0% (80) | 96.7% (240) | −3.3 | −0.163 | −0.265 |
| ow-start | 98.5% (200) | 99.5% (200) | +1.0 | +0.278 | +0.278 |
| ow-zora | 97.4% (460) | 95.2% (540) | −2.2 | −0.147 | −0.226 |
| ow-octorok | 99.0% (620) | 97.7% (660) | −1.3 | −0.064 | −0.086 |
| stalfos-keese-gel | no eval.json | 96.4% (720) | | | −0.057 |

All within the 15-point gate. ow-moblin is not evaluated yet (its fairy-route states are incomplete).

## Chains (40 runs, seeds 0–39, promoted team) vs round 7 §19 (r35 requested-key arrivals)

| Segment | Round 7 | New stack |
|---|---:|---:|
| Full milestone1 | 1/40 | 1/40 |
| Full milestone1-fairy | 3/40 | 4/40 |
| Post-D1 main (`--from 0_37c_2`) | 4/40 | 3/40 |
| Post-D1 fairy (`--from 0_37c_2`) | 7/40 | 8/40 |
| D2 (`--from 2_7ds_2`) | 19/40 | 18/40 |

## L6 training parity (ow-start, 120 iterations × 4,096 steps, 8 envs, seed 1)

Old stack final `balanced_success_rate` 1.000 (210 steps/s, idle machine); new stack 1.000 (154 steps/s while
chain evaluations and state builds shared the 20 cores, so not a speed comparison). Gate (≥ old − 0.05) passes.
Curves: `target/l6/old-metrics.jsonl`, `target/l6/new-metrics.jsonl`.

## Open (resume here)

1. **Chain 2 fairy route stops at 0x5c.** Every chain-2 build enters 0x5c through the bottom-left lane
   (`0_5cw_2`: Link at (0, 181)), from which no exploration (up to 12,000 iterations / patience 500, three
   seeds) reaches the east exit; chains 0 and 1 enter higher and cross in under a minute. `0_5cw_2` is a
   dead-end start state for ow-moblin; `0_5dw_2`, `0_4ds_2` are missing and `0_4ce_2` is still the milestone1
   one. Fix the entry (e.g. require an entry cell that reaches the next exit) before ow-moblin uses it.
2. Explore and evaluate ow-moblin.
3. `entry_states audit` (0 problems required); backfill `command` on the manifest entries written before
   command recording. `tests/test_entry_states.py::test_configured_start_states_are_clean` fails on
   `2_6ds_<c>` (4 hearts, expected 3): `path` writes the off-route 0x6d entries with `route_step: -1`, so the
   test's route-hearts lookup treats them as pre-D1; the old manifest carried `from_step: 35`. Record the
   origin step for off-route `path` entries.
4. Restore main's watch-mode features on the native pool (see `todo.md`).
