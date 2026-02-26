# Investigation Plan: Reward System Review

## Goal
Deep investigation of the entire Triforce reward system. Produce detailed specs with findings
and recommendations for each area. Output goes to `docs/specs/reward-system/`.

## Status
STATUS: IN_PROGRESS

## Tasks
- [x] 01-reward-philosophy: Investigate reward philosophy — dense vs sparse vs PBRS vs hybrid. Research RL literature for game-playing agents. Analyze current reward distribution and signal quality. Write `docs/specs/reward-system/reward-philosophy.md`.
- [x] 02-movement-actions: Investigate movement action abstraction — tile-by-tile vs multi-tile vs destination. Analyze frame_skip_wrapper.py movement mechanics, measure movement characteristics, research how other RL game agents handle movement. Write repro scripts and `docs/specs/reward-system/movement-actions.md`.
- [x] 03-movement-rewards: Investigate movement reward signals — wavefront-based PBRS, time penalties, wall collision, danger zones. Compute real reward examples, trace through critics.py movement logic, write repro scripts showing reward math. Write `docs/specs/reward-system/movement-rewards.md`.
- [x] 04-wavefront-alternatives: Investigate wavefront pathfinding — current BFS implementation, enemy-aware alternatives, curiosity-based approaches, directional rewards. Profile wavefront computation, analyze failure scenarios, research navigation reward literature. Write `docs/specs/reward-system/wavefront-alternatives.md`.
- [x] 05-combat-rewards: Investigate combat reward system — hit/kill/miss/beam rewards, bomb economics, damage trades, attack direction checking. Trace through critique_attack logic, analyze combat reward distribution, write repro scripts. Write `docs/specs/reward-system/combat-rewards.md`.
- [ ] 06-exploration-rewards: Investigate exploration and location rewards — room discovery, wrong location penalties, cave transitions, objective system. Trace through critique_location_change, analyze penalty frequency, research exploration bonus approaches. Write `docs/specs/reward-system/exploration-rewards.md`.
- [ ] 07-equipment-rewards: Investigate equipment/item pickup rewards — flat vs tiered values, key usage, contextual value. Analyze equipment reward map, measure pickup frequency assumptions, review item importance. Write `docs/specs/reward-system/equipment-rewards.md`.
- [ ] 08-health-rewards: Investigate health and damage rewards — health loss/gain scaling, damage trade handling, danger double-penalty, reward removal on damage. Trace through critique_health_change and remove_rewards logic, compute examples. Write `docs/specs/reward-system/health-rewards.md`.
- [ ] 09-special-cases: Investigate special case rewards — wallmaster handling, blocking, stuck detection, cave attack prevention. Analyze each special case for necessity and correctness. Write `docs/specs/reward-system/special-cases.md`.
- [ ] 10-reward-scaling: Investigate reward scaling and composition — clamping behavior, magnitude scale gaps, reward normalization, multi-head possibilities. Analyze real reward distributions, measure clamping frequency, research PPO reward practices. Write `docs/specs/reward-system/reward-scaling.md`.
- [ ] 11-observation-space: Investigate observation space for movement decisions — missing features, wavefront visibility, local tile map, enemy type encoding. Analyze what the model sees vs what it needs, research observation design in game RL. Write `docs/specs/reward-system/observation-space.md`.
- [ ] Final review: Read all 11 specs end-to-end. Check for cross-topic consistency, verify code references, identify gaps. Add new tasks if issues found.

## Notes
- Input topic files are in docs/todo/ (00-overview.md through 11-observation-space.md)
- Each task corresponds to one input file, producing one output spec
- Repro scripts go in scripts/repros/
- Go deeper than the todo files — they are starting points, not conclusions
- The ROM is not available, so emulator-dependent repros should be written but marked
