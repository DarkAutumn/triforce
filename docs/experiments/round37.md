# Round 37

## The bomb rules (2.0.0.6, branch `r37-rules`)

The owner's three rules and his word on the 10th kill (AGENTS.md "The owner's three bomb rules";
docs/routes/d3first.md "Bombs" with the table of every dungeon step) are routes.yaml's top-level `bomb_rules:` under
`d3first`: `fights_only: true`, `wall_only: [3:0x5b]`, `treasure_below: 8`. `d2first` names none (its step list:
354 steps, the same as main's). With them, `d3first` crosses Dungeon 2's 0x5e (the owner: the goriya fight is
`d2first`'s), by splitting the shared `dungeon-2` block at that room.

### Where the golden replay bombed rooms it only crosses (the scout, before the rules)

In `corpus/replays/golden-d3first-single-network.tfreplay` 3 of 52 bombs were weapon bombs in rooms the route only
crosses: 3:0x69 twice (inbound, frame 12217: one Darknut killed; outbound, 13306: two) and 4:0x50 (20505, a miss).
Every other dungeon bomb was in a room the objective was fighting, or the one bomb at a route wall. In the record's
11,000 runs (`r36base-1k-*`, the 51 networks) four rooms held every such bomb: 4:0x50 (1,419 per 1000 runs that
reach it), 3:0x69 inbound (644), 4:0x61 (506), 3:0x69 outbound (409); 7:0x0c once. 3:0x69 inbound is the one room
where the bombs keep Link alive: visits with a weapon bomb died 6.0% (55 of 911), without 15.2% (109 of 715), with
the same 2.2 hearts in; the runs that bombed it then cheated 622 Dungeon 3 walls (0x5b E, 0x4c E), every one
payable by the bombs thrown there. The engine's reading of every dungeon step (clear, cross, wall) is the scout's
by construction: both read `zelda::objectives`' objective kind (FIGHT), so they agree on all 182 dungeon steps;
3:0x59 from N is a clear room and 3:0x69 a crossing, as the owner says.

### The 10th kill's bomb in the record

The 51 networks made 5,881 tenth damage-free kills in the record's 11,000 runs (`bomb_tenth_kills`); 898 of them by
bomb (15%, `bomb_armed`: each armed the forced drop of four), and 2,936 bombs were placed while `$50` was 9
(`bomb_tenth_tries`). 3,024 tenth kills fell in visits Link entered with bombs, so about 30% of the tenth kills made
with a bomb in hand were made by bomb. What pays for it is already in every family: `reward-armed-bomb-drop`
(0.75) on the arming kill, `reward-gained-bombs` (0.5) on the pickup; leaving it unused costs nothing. No term was
added.

### The rooms now fought by the sword alone, in the record

- **3:0x5b's last pass** (`r36base-1k-d3`): 648 visits entered with 0.53 bombs and 1.60 hearts, left with 1.52 and
  4.39 bombs; 76 died; 1.40 bombs placed a visit (the rule takes them, but the 10th kill's); 554 wall bombs cheated
  there in the 1000 runs (0.55 a run), most because Link threw his own at the Darknuts first.
- **7:0x1b** (`r36base-1k-d7`, round 34's floor of 2): 737 visits came in with 0.18 bombs and 3.59 hearts, so nearly
  every one was already a fight under the floor of 2; 506 died (687 per 1000 entering), the rest left with 1.28
  hearts and 1.30 bombs. The blanket floor of 8 makes a fight also of the few visits that came in with 2 to 7.

### The measurement

Today's 51 networks were not trained under the rules: what they lose is the cost the retraining starts from. One
batch through the queue, 1000 runs a segment, paired with `r36base-1k-*` by seed:

```
TRIFORCE_MODELS_DIR=<worktree>/models triforce eval-batch r37rules-1k --chains r37rules-1k --runs 1000 \
  --act model --parallel 10 --models-dir <worktree>/models \
  --segments 'full:start d3:3_7cs_0 d4:4_71s_0 post-d1:0_37c_v2_0 d2:2_7ds_0 d5:5_76s_0 d7:7_79s_0 d6:6_79s_0 d8:8_7es_0 d9:9_76s_0' \
  --chain 'd2first-100: --route d2first --runs 100'
```

`d2first-100` is compared run for run with main's `d2first-100-land` (2.0.0.5). The second arm of 7:0x1b (its
crossing as the room's override, `treasure_below_rooms: [{level: 7, room: '0x1b', below: 2}]`) needs its own
module build (routes.yaml is compiled in) and so its own `d7` batch.
