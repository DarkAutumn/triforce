# Round 13: reset onto d2first, baseline retrain, the Lynel room

Lee, Nov 8: a complete reset onto the d2-first route (`docs/routes/d2first.md`), then a full retrain of every
family from scratch **with the current training setup** to set a new baseline (team + chains); Lee's new training
ideas come after that baseline. The Lynel room 0:0x0a is its own experimentation area, tackled like Dodongo.
Lee is away for the night: no questions until the retrain is done.

## Summary (in progress)

- Reset done (`c5659a2`, `4e24d95`, `eea1f1a`, `9307e86`, `b6d60b8`): d2first is the only route, 190 start states
  (chains 0–2), curricula re-explored. Observation fix for Lynel-style play (`37295b5`, §2).
- Baseline retrain `r47-base-ow` / `r47-base-dg` running (§3).

## 1. Reset (subagent D2FirstReset; decisions.md 2026-11-08)

Objectives follow the route step by step (D1 visits both key rooms: 0x73 W, 0x72, 0x73 E, 0x74, 0x73 N), 0x2c and
its heart cave in `ow-tektite`, `ow-leever-peahat` and the milestone1 routes removed, retreat ending at 0x0a
(ow-lynel: soft failure for the baseline; chains allow 3 retreats per run). States: `start` + 3 × (59 route steps,
0x78, the D1 bow rooms). Curricula: ow-zora 18/18, ow-moblin 18/18, d2-rope 18/18, wallmaster 3/3,
boss-aquamentus 3/3, goriya 12/12 (2 at 12000 iterations), moldorm 3/3 (12000 iterations), boss-dodongo 26 tracks
(seeds 0–9, 4 swallow kills dropped). Smoke chain with the old (milestone1) team: no errors; both runs ended at
0x58 going N (the old ow-octorok's habit), as expected.

## 2. The Lynel room 0:0x0a: mechanics (probes in `target/lynel/`)

- Geometry: cave top-left, a lake in the middle with a river south from it. Link enters bottom right (from 0x1a) and
  can't cross the river: up the right corridor, then west along the top strip between lake and rock wall to the
  cave. Every Lynel spawn is on that path.
- **Three spawns** (300 RNG seeds, entry S): (160, 93) middle of the top strip 55%, (192, 109) top of the right
  corridor 23%, (80, 93) on the strip near the cave 22%.
- Blue Lynel: 6 HP units, the wood sword takes 1 per hit (6 hits; white sword 3); **contact costs 2 hearts**; sword
  shots (type $57, not small-shield blockable); moves like a Goriya (UpdateLynel → UpdateGoriya).
- **Walk-through (Lee's trick, confirmed):** after a sword hit his invincibility timer runs ~31 frames, and while
  it runs `CheckMonsterCollisions` returns before `CheckLinkCollision` (Z_01), so Link overlaps him unharmed (probe:
  8–30 frames after the swing no damage; at 2 and 36 frames −2 hearts). Stunned monsters are skipped too.
  `hurts_on_touch` said 1 then; fixed for every monster (`37295b5`: 0 while hit-flashing, unless always
  invincible, or while stunned).
- Go-Explore solves every spawn without retreating (12 seeds of `0_0as_0`: mid-strip 7, corridor 2, near-cave 3;
  50–133 steps; Lynel left with 2–6 of 6 HP, so no kill); the cave (sword pickup) 6/6 in 9–20 steps; the way back
  out (`0_0ac_0`, white sword) 6/6 in 35–112 steps. Tracks in `target/lynel/explore/ow-lynel/` (not used by the
  baseline).

## 3. Baseline retrain (r47)

Two rounds side by side at 8 envs each (`docs/performance.md`), 8M steps per family, early stop at 0.99:
`train_round.py --round r47-base-ow --specialists ow-start,ow-octorok,ow-zora,ow-moblin,ow-tektite,ow-lynel,ow-boulder`
and `--round r47-base-dg --specialists d2-rope,goriya,moldorm,boss-dodongo,stalfos-keese-gel,wallmaster,boss-aquamentus`
(`--steps 8000000 --parallel 8 --eval-parallel 3 --no-promote --chain-runs 0 --train-args "--stop-success 0.99"`).
