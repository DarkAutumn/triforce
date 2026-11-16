# Round 25: the honest planner as a teacher (analysis, no training)

The honest planner (`temperature:0.5+relax:16+plan:16:1`, `registry.PLAN_ACT`; docs/rust-actor.md "Foresight",
docs/decisions.md 2026-11-16) tries each decision's drawn action against one future with the game's random numbers
redrawn and, when Link would lose health, takes the first of the rule's 16 best that wouldn't. Where its choice
differs from the model's is a label a training could learn from. This is what such a training would have to learn,
measured on the planner's own runs (clips on, the rule's seeds; 1000 runs of D5, D3 and the leg from 0_45c_0, 300 of
D4 and D2). Every chain room record with foresight carries the labels: `decisions.foresight.replaced`, per replaced
decision `[index in the visit, refused joint action, its ln p under the policy, its rank among the policy's joint
actions (0: the model's first choice), taken joint action, its ln p, Link's hearts]` (joint index = kind × 32 +
direction × 4 + duration, kinds MOVE WAIT SWORD BEAMS BOMB BOOMERANG ARROW CLIP RECORDER FOOD CANDLE, directions N S W
E NW NE SW SE, durations 4 8 16 32).

## What the planner changes, by family

| family | decisions | replaced | nothing safe | refused was the model's first choice | mean p(refused) | median p(taken) | most frequent refused → taken kinds |
|---|---:|---:|---:|---:|---:|---:|---|
| darknut | 413,704 | 5.4% | 0.7% | 78% | 0.63 | <0.001 | MOVE>MOVE, MOVE>WAIT, SWORD>MOVE |
| ow-peahat | 61,625 | 6.6% | 0.2% | 75% | 0.53 | 0.001 | MOVE>MOVE, MOVE>SWORD, MOVE>BEAMS |
| ow-zora | 68,144 | 5.4% | 1.5% | 83% | 0.60 | 0.020 | MOVE>MOVE, MOVE>BOOMERANG, SWORD>MOVE |
| boss-manhandla | 74,152 | 4.5% | 1.9% | 55% | 0.36 | 0.044 | MOVE>MOVE, BOOMERANG>MOVE, WAIT>MOVE, BOMB>MOVE |
| ow-moblin | 49,469 | 4.0% | 0.5% | 77% | 0.56 | 0.016 | MOVE>MOVE, MOVE>BEAMS, MOVE>BOOMERANG |
| boss-gleeok | 18,448 | 3.4% | 1.0% | 87% | 0.72 | 0.018 | SWORD>MOVE, MOVE>MOVE |
| ow-tektite | 19,769 | 3.3% | 0.4% | 85% | 0.60 | 0.052 | MOVE>SWORD, MOVE>MOVE, MOVE>BOOMERANG |
| moldorm | 16,340 | 2.9% | 1.1% | 87% | 0.73 | 0.001 | SWORD>MOVE, SWORD>WAIT |
| stalfos-keese-gel | 18,010 | 2.8% | 0.3% | 86% | 0.64 | 0.039 | MOVE>MOVE, MOVE>WAIT, CLIP>BOOMERANG |
| boss-digdogger | 39,980 | 2.6% | 1.0% | 85% | 0.67 | 0.011 | MOVE>MOVE, SWORD>MOVE |
| like-like | 37,517 | 2.1% | 0.1% | 69% | 0.49 | 0.057 | MOVE>MOVE, MOVE>BOOMERANG |
| gibdo | 169,747 | 2.1% | 0.2% | 75% | 0.54 | 0.023 | MOVE>MOVE, MOVE>WAIT, SWORD>MOVE |
| goriya | 40,245 | 1.6% | 0.3% | 85% | 0.69 | 0.017 | SWORD>MOVE, SWORD>WAIT, BEAMS>MOVE |
| trap-room | 62,814 | 1.4% | 0.3% | 76% | 0.59 | 0.006 | MOVE>MOVE, MOVE>BOOMERANG, MOVE>BEAMS |
| vire | 76,460 | 1.3% | 0.2% | 75% | 0.57 | 0.027 | MOVE>MOVE, MOVE>BOOMERANG |
| equipment-room | 125,824 | 1.3% | 0.2% | 87% | 0.72 | 0.003 | MOVE>MOVE, MOVE>WAIT |
| zol | 102,567 | 1.3% | <0.1% | 72% | 0.48 | 0.110 | MOVE>MOVE, MOVE>BEAMS, MOVE>SWORD |
| ow-leever | 22,036 | 1.2% | 0.1% | 86% | 0.74 | <0.001 | MOVE>MOVE, MOVE>WAIT |
| d2-rope | 52,072 | 0.5% | 0.2% | 86% | 0.71 | 0.006 | SWORD>MOVE, MOVE>MOVE |

So a training would have to change the model's choice in 1-7% of its decisions, mostly where the refused action is
its own first choice (55-87%) and the safe one an action it rates at a few percent or less (median p well under 0.1).
The typical correction is small: a shorter move (MOVE N 32 → MOVE N 4), a wait (MOVE N 8 → WAIT 4), stepping back or
sideways instead of walking or swinging into contact (SWORD S → MOVE W 8), or a stun first (CLIP N → BOOMERANG N).

## Where, by room (the rooms with the most replaced decisions)

| room | family | decisions | replaced | nothing safe | refused was first choice | most frequent refused → taken |
|---|---|---:|---:|---:|---:|---|
| 5:0x37 | darknut | 67,496 | 15.4% | 1.1% | 67% | MOVE N 8/16 → WAIT 4 or MOVE N 4 (the column: timing the Darknut) |
| 0:0x1f | ow-peahat | 37,771 | 10.3% | 0.3% | 74% | MOVE E 4 / N 4 → MOVE W 4, SWORD E, a sidestep (the Peahat's path) |
| 3:0x4d (Manhandla) | boss-manhandla | 74,152 | 4.5% | 1.9% | 55% | MOVE → MOVE, BOOMERANG/WAIT/BOMB → MOVE; many hits are unavoidable here |
| 5:0x05 (Olive Garden) | darknut | 137,814 | 2.3% | 0.7% | 87% | SWORD S/E into a touching Darknut → MOVE W 8/16, MOVE N 4/8 |
| 5:0x64 | darknut | 23,324 | 11.5% | 1.6% | 92% | MOVE E 32 / N 4 → MOVE E 4, MOVE N 32, WAIT |
| 3:0x59 | darknut | 72,435 | 3.5% | 0.3% | 89% | SWORD → MOVE |
| 3:0x69 | darknut | 22,339 | 11.0% | 2.3% | 90% | MOVE → MOVE, MOVE → WAIT |
| 0:0x2d | ow-zora | 12,466 | 14.2% | 4.4% | 80% | MOVE N 32 → MOVE N 4, BOOMERANG SE/SW |
| 5:0x66 / 5:0x65 / 5:0x47 | gibdo | 150,043 | 1.5-5.6% | 0.2-0.8% | 72-77% | MOVE W/N 32 → a short move, WAIT or BOOMERANG |
| 0:0x52 / 0:0x53 | ow-moblin | 32,852 | 5-6% | 0.5% | 72-82% | MOVE W 32 → MOVE W 4, BEAMS W, BOOMERANG W |
| 4:0x21 | stalfos-keese-gel | ~7,600 | 3.6% | 0.3% | 88% | CLIP N → BOOMERANG N/S or WAIT 4 (the Gels on the landing strip) |

Hearts lost a visit, the rule alone → the planner (0:0x1f: `plan:16:1`, 1000 paired runs; the dungeon rooms:
`plan:8:1` on the 300-run screens, within noise of `plan:16:1`): 0:0x1f 1.02 → 0.08, the Olive Garden 4.80 →
2.41, 5:0x65 westbound 0.74 → 0.29, 5:0x47 0.84 → 0.47, 5:0x37 1.20 → 1.08 (still a hit on most visits), 4:0x21 0.22 →
0.06, Manhandla 1.26 → about 1.4 (the planner does not fix Manhandla).

## How the planner could label training data (simplest first)

1. **A penalty on the refused action.** During ordinary rollouts, try each sampled action (`plan:1:1`, the planner
   with one candidate: it never changes play, it only labels, at one extra emulation of the action) and, where it
   would lose health in the redrawn future, add an auxiliary loss on that state pushing the refused action's
   probability down (e.g. −log(1 − p(refused))). Env, rewards and PPO stay as they are; the labels are honest (no real
   future). The cost is about 2x the rollout emulation.
2. **A target for the policy.** Where the planner replaces an action, a cross-entropy term toward the action it took
   (distillation, DAgger-like: the planner as the expert on the states the model reaches). Stronger than 1 (it says
   what to do, not only what not to), but the target is one of 16 rule-ranked candidates, not the best one.
3. **A mask during rollouts.** Masking the refused action while collecting removes the hit from the experience but
   gives no gradient against the refused action (the model never sees what it would have cost) and teaches reliance
   on the mask; useful only as a curriculum aid (Link reaches later rooms with more hearts).

Start with 1 on darknut and ow-peahat (the most replaced decisions, mostly first-choice errors). A later round
measures by chains with the models alone (the primary number), the planner beside them.
