# Round 14: the bow (Dungeon 1 trap room 0x22 and the bow cellar 0x7f)

Lee, Nov 9: add the bow to the route. In Dungeon 1 go up to 0x22 (the game's first blade-trap room), avoid the
traps, push the west block N or S, take the stairs to the item cellar, take the bow, come back. New families:
`trap-room` (blade-trap rooms) and `equipment-room` (item cellars, "all pretty much the same"). Lee had found the
trap room very hard for plain PPO (the negative scores discourage it) and expected a curriculum to work.

## Summary

- Route detour mapped and built (`18e28b8`, subagent BowRoute): 0x43 N → 0x33 → 0x23 → W 0x22 → stairs → cellar
  0x7f (bow) → 0x22 → E 0x23 → S 0x33 → S 0x43 → E on as before. Cellars are sub-rooms (`in_cave`), 0x22 has
  `push_block: [N, S]`, the objective leads to the push spots, then the stairs, then the bow. States rebuilt from
  the detour on (205 files, audit 0 problems).
- **trap-room: 100% (600 eval episodes), 93% of visits without a hit, in 0.4M steps with the curriculum.** The same
  yaml without the curriculum: best training success 12.5%, 0% after 2M steps (episodes end `no-damage`/`stuck`:
  it learned to stand still). Lee's diagnosis holds and the curriculum fixes it.
- **equipment-room: 99.3%** (0.7M steps, curriculum), 67% of visits without a hit (4 Blue Keese).
- D1 families continued on the bow-carrying states (r51): stalfos-keese-gel 99.0%, goriya 93.4%, wallmaster 100%.
- Chains: D1 from its entrance **84/120 (70%)** with the detour (96% without it, round 13), the bow taken in 117/120;
  27 runs died to Aquamentus, who was trained only from ≥ 0.75 of max health and now meets Link at 0.5–2 hearts
  (deaths by arrival hearts: 0.5 → 6/6, 1.0 → 8/8, 1.5 → 4/8, 2.0 → 4/10, ≥ 2.5 → 5/79). Retraining him from 0.15
  (r53, §4). Full route 9/120 (round 13: 25/120 without the detour).

## 1. Mechanics (probes `target/traps/`, `asm:Trap_Update_Full`, `asm:PushBlock_Update`)

- Four blade traps (`$49`) in the corners (32/208, 93/189). They sense Link within 14 px of their row (slide
  horizontally) or column (vertically), slide out at 1.75 px/frame to the centre line, back at 0.5. Unkillable;
  `hurts_on_touch` 1, `killable` 0.
- A diamond of blocks walls in the stairs at the centre (128, 144); the west block (96, 144) is pushable after 16
  frames of pushing. N or S opens the way; E walls the stairs in until Link leaves.
- Crossing to the west half needs a trap row (y 101/181); Link outruns the pair if he keeps moving. A scripted path
  (`target/traps/path.py`) along the bottom is unhurt in 20/20 seeds.
- Cellar 0x7f: side view (no gravity; ladders and walkways are walkable tiles), Link arrives at the top of the left
  ladder, the bow lies on a walkway reached by the right ladder, 4 Blue Keese; the exit is up the left ladder.

## 2. Training (r50, 8 envs each, early stop 0.99)

| family | steps | eval | health change | no-hit visits | steps per visit |
|---|---|---|---|---|---|
| trap-room (curriculum, 6 tracks) | 0.40M | 100% | −0.08 | 93% | 26 |
| trap-room, no curriculum (r52) | 2.06M, stopped | 0% (best 12.5% in training) | — | — | 530 |
| equipment-room (curriculum, 3 tracks) | 0.71M | 99.3% | −0.20 | 67% | 44 |

## 3. D1 families on the new states (r51, continued from the r47 models, 4M budget, stop 0.99)

The detour changed 0x43's exit from W (N instead of E), added 0x23 from W and 0x33 from N, and every later state
carries the bow. Re-evaluated before continuing: stalfos-keese-gel 98.3% (0x43 from W 88–90%), goriya 88.3%
(0x23 from W 78–81%), wallmaster 95.0%. After: 99.0%, 93.4%, 100%. Promoted with trap-room and equipment-room.

## 4. Chains (120 runs, seeds 100–219)

- D1 from its entrance (`--from 1_73s_0`): 84/120. Deaths: Aquamentus 27, 0x44 5, 0x22 1, 0x43 1, 0x33 1; one
  stalfos-keese-gel stall. No cheated bombs.
- Full route: 9/120. Deaths: D2 0x5e 16, Moldorm 12, 0x1e 8; 0x0a 5; D1 0x35 6, 0x33 5, 0x72 5, 0x43 4, 0x63 4, ...
  The detour adds four D1 rooms of attrition on the way to Aquamentus.
- boss-aquamentus retrained from 0.15 of max health: r53 (results below).

## 5. Aquamentus from low health (r53) and the chains with the bow

`boss-aquamentus` continued from r47 with `hp: [0.15, max]` (was `[0.75, max]`): eval 99%, health change +0.89;
starts at ≤ 2 hearts won 58/61. Promoted. Chains (120 runs, seeds 100–219), with round 13 (no detour) for comparison:

| segment | round 13 | round 14 |
|---|---|---|
| D1 from its entrance, full health (`--from 1_73s_0`) | 115 (96%) | **108 (90%)**, bow 117+ |
| from D2's exit (`--from 0_3cc_0`, 4 hearts) | 35 (29%) | 19 (16%) |
| full route | 25 (21%) | 13 (11%) |

The detour costs D1 about 6 points at full health (0x44 5, 0x35 3, single deaths at 0x22/0x43/0x33). From the
low-health arrivals of a real run it costs much more: post-D2 runs now die in 0x33 (12), 0x44 (12), 0x23 (10), 0x43
(9) on top of the overworld (0x0a 9, 0x27 7). Link reaches D1 with ~2 of 5 hearts and the detour adds four fights
before Aquamentus; there is no healing anywhere between 0x2c and the D1 triforce. Health attrition is now the
route's main problem (with D2's Goriya/Moldorm rooms): a candidate for Lee's training ideas (and for the kill-streak
fairy, backlog).
