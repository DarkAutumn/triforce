# Round 4: d1-melee one-room episodes and route fixes

Goal (`todo.md`, round 4): make the curriculum a per-specialist config field, promote the best ow-zora and
d1-wallmaster, fix 0x43 → 0x33, end every d1-melee episode at its first room exit, and compare d1-melee per room
against round 3. Setup as in round 3 (16 workers, 2M-step budget, balanced 95% early stop, 20 sampled episodes per
seed state for eval, 20-run chain).

## 1. Changes

- **`curriculum: {enabled, trajectories, fraction, threshold}`** in the specialist config. `train` uses it unless
  `--curriculum DIR` overrides or `--no-curriculum` disables it. On for ow-zora and d1-wallmaster (and, after
  round 4, boss-aquamentus); trajectories default to `trajectories/<name>`.
- **`one_room: on`** (d1-melee): every correct room exit ends the episode as `exit-correct`, also inside the
  specialist's rooms. Before, an episode ran until Link left the 9-room family.
- **Health change metric:** `health_change` = hearts at exit − hearts at the start (after hp randomization), in
  metrics, progress events, eval (`mean_health_change`) and the training GUI.
- **Route fixes found by the chain** (each one moved the chain's failure point further into Dungeon 1):
  1. *0x43 → 0x33.* The objective now only offers a route room's route exits. That stopped the runner error, but
     r4's chain then died in 0x43 in every run that reached it, with no key for the locked E door.
  2. *Untagged key room.* 0x33 (untagged) holds a key and tied on path length with 0x72's key, so the objective let
     Link skip 0x72 and planned to fetch 0x33's key later. `GameMap._dijkstra` now only routes through rooms with
     a specialist tag (the runner can't play the others).
  3. *Route order on revisited rooms.* With no keys, 0x73's W (0x72) and E (0x74) exits tie. The policy took E
     first; back in 0x73 from the east with one key it had to go W, a case no start state covers, and r4b's chain
     looped 0x73 ↔ 0x74 for up to 226 rooms. `GameMap.route_first` now keeps the main-route exit the route takes
     first (W).
  4. *Runner:* a wrong exit into an untagged room is a gameplay failure (`left the route into untagged …`), not a
     game.yaml error.
- **Training GUI:** colorblind-safe palette (good = blue, bad = red, purple between).

## 2. Promotions

ow-zora and d1-wallmaster, all evaluated on the round 3 states:

| specialist | r2b | r2c | r3 baseline | r3c (curriculum) | promoted |
|---|---|---|---|---|---|
| ow-zora | 74% | 68% | 22% | **98%** | r3c |
| d1-wallmaster | 69% | 62% | 0% | **77%** | r3c |

The other specialists were promoted where round 3 beat the round 1 model on the new states (section 3 of
`round3.md`): ow-start, ow-leever-peahat, ow-moblin, d2-rope, d2-goriya → r3-baseline. ow-octorok stays round 1
(98% vs 94%); boss-aquamentus has no working model. d1-melee → r4c.

## 3. d1-melee per room

Round 3's model was also evaluated with one-room episodes (`eval-one-room.json`), so its per-room skill can be
compared directly; its family-episode numbers count every later room against the start room.

| room | r3 (family episodes) | r3 (one-room eval) | r4 | r4b | r4c |
|---|---|---|---|---|---|
| 0x73 | 31% | 99% | 100% | 100% | 100% |
| 0x72 | 15% | 82% | 98% | 100% | 100% |
| 0x74 | 25% | 90% | 85% | 83% | 73% |
| 0x63 | 47% | 97% | 92% | 93% | 90% |
| 0x53 | 63% | 90% | 87% | 83% | 87% |
| 0x52 | 80% | 92% | 95% | 98% | 97% |
| 0x42 | 90% | 97% | 98% | 100% | 100% |
| 0x43 | 93% | 93% | 90% | 98% | 92% |
| 0x36 | 100% | 100% | 100% | 80% | 100% |
| **all** | 52% | 94% | 95% | 95% | 94% |
| steps | 2.0M | | 512k | 778k | 799k |

- **One-room episodes don't make a better room player; they make the number honest and training 2.5–4× cheaper.**
  Round 3's model already played single rooms at 94%. The 52% was 90%-per-room compounding over 4–9 rooms.
  Round 4 reaches the same 94–95% with early stop at 0.5–0.8M steps.
- The Stalfos rooms (0x74, 0x53, 0x63) are the weakest at 73–93%, all deaths.
- r4b's 0x36 at 80% and r4c's 0x74 at 73% are single-seed swings (20–60 episodes per room).

One-room evals of other multi-room specialists (round 3 models): d2-rope 91% (0x3e 65%, the rest ≥ 90%; 83% with
family episodes), ow-octorok 97% (0x4c 70%).

## 4. Chain (milestone1, 20 runs)

| run | furthest D1 room | where runs end |
|---|---|---|
| r3 baseline | — | ow-zora 0x38/0x2d ×20 |
| r4 | 0x43 | D1 deaths ×18 (0x43 ×4: no key for the E door), 0x37 ×2 |
| r4b | 0x43 | D1 deaths, 0x73 ↔ 0x74 loops of up to 226 rooms, one runner error in 0x33 |
| r4c | **0x35 (Aquamentus)** | 0x74 ×6, 0x53 ×5, 0x44 ×3, 0x63 ×2, 0x43 ×2, 0x35 ×1, 0x38 ×1 |

r4c is the first chain where Link follows the whole Dungeon 1 route (0x73 → 0x72 → 0x73 → 0x74 → 0x73 → 0x63 → …
→ 0x45 → 0x35). Every D1 death in r4c is a **health drain**: Link enters Dungeon 1 with 1–3 hearts (the overworld
costs some), loses 0.5–1.5 per Stalfos room, and dies with 0.5–1.5 hearts in 0x74/0x53/0x63/0x44. Training
starts at 50–100% hp (1.5–3 hearts), so the policies never practise the 0.5–1 heart situations the chain
produces.

## Conclusion

Every round 4 item is done. One-room episodes cut d1-melee's training cost by 2.5–4× at the same per-room skill;
the three route fixes take the chain from "dies in 0x43" to "reaches Aquamentus". The next blockers are health
drain across rooms (train at the hp the chain arrives with) and Aquamentus itself (0%).

Raw data: `models/r4-one-room-*`, `models/r4b-route-keys-*`, `models/r4c-route-order-*` (summary, chain
summary, chain jsonl), `models/d1-melee/{r3-baseline,r4-one-room,r4b-route-keys,r4c-route-order}/eval*.json`,
`models/{d2-rope,ow-octorok}/r3-baseline/eval-one-room.json`, logs `logs/261102_00{5,6,7}_*`.
