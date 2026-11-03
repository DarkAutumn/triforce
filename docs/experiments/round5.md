# Round 5: toward the Dungeon 2 boss (overnight, Nov 2–3)

Goal: specialists that carry the milestone-1 chain to the Dodongo room (2, 0x0e); the milestone ends on entering
it. Starting point: round 4's r4c chain follows the whole Dungeon 1 route but dies of health drain; Aquamentus is
at 0%; there was no route state for 0x1e.

Setup as in rounds 3–4 unless a row says otherwise (2M-step budget, balanced 95% early stop, 20 sampled episodes
per seed state for eval, 20-run chain). "Reach" columns count chain runs (of 20) that entered a room.

## Promotions before the experiments

On the round 3 states: ow-start, ow-leever-peahat, ow-moblin, d2-rope, d2-goriya → r3-baseline (better than the
round 1 models, `round3.md` section 3). ow-octorok stays round 1. ow-zora, d1-wallmaster → r3c. d1-melee → r4c.

## Where the chain loses health (r4c)

Mean hearts lost per room on the route: 0x48 0.23, 0x38 0.35, 0x37 0.29 (Link enters Dungeon 1 with 1–3 of 3
hearts), then 0x72 0.29, 0x74 0.68, 0x63 0.62, 0x53 0.50, 0x44 1.5. Training started every room at 1.5–3 hearts,
so no specialist practised the 0.5–1.5 hearts the chain arrives with.

## Experiments

### r5: d1-melee hp floor, Aquamentus boss hit reward

- **d1-melee `hp: [0.15, max]`** (was `[0.5, max]`): eval 95% (r4c 94%) with less damage per room (−0.16 vs
  −0.31 hearts) and shorter episodes (34 vs 47 steps); full 2M steps (no early stop). 0x53 is the weakest room
  (77%). Promoted.
- **boss-aquamentus: `reward-boss-hit`** (0.75 per hit on a boss, no combat decay; was 0.25 `reward-hit`) +
  curriculum: still 0%. The curriculum got one stage further (points 66/110, 37/77, 36/76; r3c 77, 45, 44) and boss
  hits show up late in training, but 1–8 episodes finish per 4,096-step iteration: Link flees and dodges for up to
  2,000 steps, so the stalled stage gets almost no samples.

| chain | 0x73 | 0x63 | 0x52 | 0x44 | 0x45 | 0x35 | where runs end |
|---|---|---|---|---|---|---|---|
| r4c | 19 | 13 | 6 | 4 | 1 | 1 | 0x74 ×6, 0x53 ×5, 0x44 ×3, 0x63 ×2, 0x43 ×2 |
| r5 | 19 | 18 | 9 | 6 | 1 | 1 | 0x53 ×6, 0x44 ×5, 0x63 ×3, 0x42 ×2, 0x74 ×1 |

### D2 route state for 0x1e

`entry_states build --chains 2 --start-step 41 --source 2_2es_2`: 0x2e → 0x1e found in 7.7k env steps
(`2_1es_2`); 0x1e → 0x0e failed 3 times (175k env steps). 0x1e's N shutter opens only when all 5 Red Goriyas are
dead. The milestone ends on entering 0x0e, so no Dodongo state is needed. `triforce.explore --specialist d2-goriya`
then solved 0x1e in 77k steps (158-step path), so d2-goriya gets the curriculum (r6).

d1-wallmaster `1_44w_2` is still unexplored with the enemy-health cell (8,000 iterations, 166k env steps, 6,340
cells).
