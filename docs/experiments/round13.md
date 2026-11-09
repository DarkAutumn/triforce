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

## 4. Lynel room experiment r48-lynel-curriculum

Ahead of the round's own ow-lynel (which trains late in the overworld stream), a separate run at 4 envs: the
ow-lynel yaml plus a curriculum over the 24 explored tracks (12 entry tracks covering all three spawns, 6 cave, 6
way back; `specialists/arms/r48-lynel-curriculum/`), 4M steps, retreat still a soft failure. The curriculum
completed by iteration ~120; training success 0.95–1.0 after that, retreats ~0–1%.

Eval (100 per state, current yaml: hp [0.15, max], full_hp 0.5): **98.0%**, health change −0.29.

| state | success | health change | failures |
|---|---|---|---|
| 0_0as_0/1/2 (entry from 0x1a, wood sword) | 99 / 100 / 97% | −0.37 / −0.45 / −0.54 | 4 deaths |
| 0_0acave_0/1/2 (take the white sword) | 100% | 0 | — |
| 0_0ac_0/1/2 (cave → S, white sword) | 94 / 95 / 97% | −0.53 / −0.28 / −0.41 | 14 deaths |

By Lynel spawn (100 episodes each from a track's room-entry state with that spawn baked in,
`target/lynel/spawnstates/`): mid-strip **93%** (−0.58), corridor **100%** (−0.15), near-cave **96%** (−0.53).
No retreats. Failures are deaths, mostly at low starting health (1–2 hearts: 8 of the 18 in the main eval) but 4
at full health. So the room is passable for every spawn without the pirouette; the mid-strip spawn (55% of entries)
is the weakest, and the way back out with the white sword loses more runs than the way in.

## 5. Lynel room: the baseline ow-lynel (r47-base-ow, no curriculum, 8M)

Eval **98.4%**, health change −0.28 (r48 with the curriculum, 4M steps: 98.0%, −0.29). Entry 98/98/99%, the cave
100%, the way back 97/96/98%; 3 retreats in 300 entries (soft failures). Per spawn: mid-strip **98%**, corridor
**100%**, near-cave **95%** (2 retreats, 3 deaths). So the plain setup learns the room as well as the curriculum
does; the curriculum only got there in half the steps. Neither needed the pirouette, and with retreat a soft
failure the model almost never leaves. The baseline run stays the team's ow-lynel (it is the baseline).

Conclusion for now (Lee's goal: get through the room, however): reached, ~98% one-room for every spawn. Left
open: deaths at full health (a few per 100 on the way back with the white sword), whether a neutral retreat for
the near-cave spawn only would help (95% there), and how the room does in chains, where Link arrives with whatever
health 0x1a left him.

## 6. Baseline team (r47, all promoted)

| family | eval success | health change | steps (early stop at 0.99) |
|---|---|---|---|
| ow-start | 99.6% | +0.00 | 0.25M (stop) |
| ow-octorok | 98.3% | −0.07 | 8M |
| ow-zora | 98.9% | −0.16 | 8M |
| ow-moblin | 96.3% | −0.16 | 8M |
| ow-tektite | 98.7% | +0.05 | 8M |
| ow-lynel | 98.4% | −0.28 | 8M |
| ow-boulder | 99.9% | −0.20 | 1.4M (stop) |
| d2-rope | 98.3% | +0.29 | 6.4M (stop) |
| goriya | 91.2% | −0.41 | 8M |
| moldorm | 85.0% | −0.29 | 8M |
| boss-dodongo | 98.7% | +0.91 | 0.22M (stop) |
| stalfos-keese-gel | 99.4% | −0.01 | 0.66M (stop) |
| wallmaster | 99.7% | +0.10 | 0.87M (stop) |
| boss-aquamentus | 100% | +0.63 | 0.83M (stop) |

(Early-stop/step pairs from the train logs; the two streams ran ~13 h in all, 18:50–08:00.)

### Chains (120 runs, seeds 100–219, `triforce eval-chain`, default route d2first)

- **Full route: 0/120.** 66 runs reached Dodongo, killed him (1 bomb, ~10 steps) and took the heart, then stood
  in the room until the 2000-step budget: boss-dodongo's success was still the heart pickup (milestone1 ended
  there), so it never learned the W exit to the triforce room. Fixed (`c65b5f9`: success `exit-correct`, tracks
  re-explored through the exit: 26 after dropping 4 swallow kills) and retrained as r49 (§7). The other 54 died
  in D2: 0x5e Goriya 16, 0x1e Goriya 12, Moldorm 8, 0x4e 4, 0x7e 3, 0x2e 2, 0x6e 1; and on the way: 0x4c 4,
  0x5b 3, 0x4d 1.
- **From D2's exit (`--from 0_3cc_0`, 4 hearts) to the D1 triforce: 35/120 (29%).** No retreats at 0x0a. Deaths:
  D1 0x44 16, 0x63 11, 0x35 7, 0x52 5, 0x42 5, 0x72 2, 0x74 2, 0x53 2, 0x43 2; overworld 0x0a 9, 0x27 7, 0x38 6,
  0x1a 4, 0x18 3, 0x19 2, 0x17 2. Health is the story: Link has ~4 of 5 hearts at 0x0a, ~2.6 at 0x17/0x27 and
  ~2.0 entering D1 (no healing on the way), where the deaths pile up (per-room table: `target/r47/rooms.py`).
- **From D1's entrance (`--from 1_73s_0`, full 5 hearts): 115/120 (96%).** Aquamentus 3 deaths, 0x63 1, one
  wallmaster stall. So D1 itself is solved at full health; the post-D2 overworld's attrition is what loses runs.
