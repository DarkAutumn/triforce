# Round 26 walks: the leg from Dungeon 5 and Dungeon 7 in the running game (Nov 16)

`rest-of-game.md` "Gap list" (round 26, items 1-12) and `d7.md` §7, answered by walking the running game. Every
number here was observed (marked **walked**); how each item was staged, what was written to RAM, the trials and the
scripts are in its section. Scripts (`scripts/experiments/`): `r26_walks.py` (helpers, staging), `r26_leg_walks.py`
(items 1-4), `r26_d7_walks.py` (items 5, 6, 8, 9, 10, the blind crossings), `r26_wallmaster_bot.py` (item 7),
`r26_walk_scaffold.py` (a throwaway data root with a scouting route `d7walk` so today's models can be played in
Dungeon 7, item 12), `r26_replay_damage.py` (who hurt Link, from replays).

## The answers

| # | Item | Answer (one line) | Corrects |
|---|---|---|---|
| 1 | Rupees, bombs at 0:0x0b | **168-180 rupees (median 171), 28-40 after both shops**; bombs mean 3.3, 0 in 14-17%; keys 5 | `rest-of-game.md` ledger (142-154 / 157-168), `late-mechanics.md` §1 (bombs "0 in most chains"), `d7.md` §4 |
| 2 | 0x0b's whirlwind | Two plays facing down or left → 0x45 (60/60, 72/72 harvested arrivals); second B from +152 to +216 frames after the first; the chain sees one jump 0x0b → 0x45 landing at X 0, carried 87 more frames | `d7.md` §1 (frames, entry spot) |
| 3 | Shops 0x44, 0x34; armos | A purchase sets OW flag $10 but the shop **sells again on re-entry**; the walk-in row buys **0x44's bombs**; 0x34's key (80) is affordable on every arrival; armos wakes from W/E/N at once, stairs 63 frames later; exit (0x40,0x6D), Down wakes it again | `d7.md` §1, `late-mechanics.md` §2.1 ("0x34 cannot sell the wrong affordable ware") |
| 4 | 0x42's pond | B works after 5 frames of Up (Y $D5); stairs at B+232; **the pond refills on every visit** (flag $80 set but not honoured): one play per entry | `level-7.md` "At a glance", `d7.md` §1, `late-mechanics.md` §2.8 |
| 5 | 7:0x1c leave-and-return | Works: one small Digdogger dead (death state or a sword kill, 12/12), out W, back in: empty, N open | confirms `d7.md` §2 (inferred → walked) |
| 6 | 7:0x0c wall bomb | Never eaten from (0xD0,0x8D) facing E (0 of 330 bombs); failures are knockbacks off the spot while waiting | `d7.md` §2, §5 #8 (the eating risk is not the problem) |
| 7 | 7:0x0d clear | Scripted lure-and-strike: 40/40 clears, **0 hearts**, median 1739 frames; pushed E the stairs are still reached round the top; a chain capture = an `exit-wrong` record and a replay from 0x79 | game.yaml 0x0d note ("pushed E Link is left inside the ring"), `late-monsters.md` §3.3/§11.6 |
| 8 | Ladder refused | 7:0x49: in 15/40 immediate crossings, 9-86 frames (11 slots: 3 Goriyas, 3 boomerangs, 3 Keese, 2 Bubbles); all crossed. 0x18: ≤ 3/40, ≤ 11 frames | `d7.md` §2 (inferred → walked: a delay, never a block) |
| 9 | 7:0x1a CLIP | `CLIP N` offered at (0x60,0x9D) and 12 other points; CLIP N + MOVE E → stairs in 33/40 and 16/20 (225-257 frames) | confirms `d7.md` §2 |
| 10 | 7:0x28 FOOD | Never offered in the doorway; offered from frame 127 holding Up; `FOOD N` feeds at once (66 + 124 menu frames) | confirms `d7.md` §2 |
| 11 | 7:0x2a, boss-aquamentus | 6/20 (30%), 11 timeouts, 3 deaths; −6.5 hearts mean; planner 3/10 | — |
| 12 | Forced fights and crossings | Zero-shot from 11 hearts: goriya **0%** in 0x69, 0x19, 0x1b, 5% 0x49; boss-digdogger 10% in 0x1c and splits 0x39 (10%); wallmaster 0% in 0x0d; dodongo-crossing 90% in 0x0c. Every model fires arrows (up to 15 a run) | `d7.md` §5 ranking; `level-7.md` 0x39 trace ("starts at X $50, Y $9D") |

**The three findings that most change the plan**

1. **Today's models cannot play Dungeon 7's Goriya rooms at all, not even the crossings.** From full 11 hearts the
   goriya model dies in every run of 0x69 and 0x19 (bomb-wall crossings) and 0x1b, and in 19 of 20 of 0x49; the
   honest planner does no better in 0x69/0x19 (0/10). Blue Goriyas and their boomerangs did 75-80% of the damage.
   Blue Goriya rooms need training, not a widened room list. 0x0d is 0% with the wallmaster model (and 50% with the
   planner), while a 60-line script clears it 40/40 with no damage: the skill is learnable and cheap in hearts.
2. **The rupee ledger is wrong on both readings and ARROW is a leak.** 168-180 at 0x0b leaves 28-40 after both shops,
   the 0x34 key is affordable on every arrival, and a shop sells again on re-entry (91 → 11 for a second arrow).
   Every family with the bow and arrows fired them in D7 (0x1c: 14.7 a run with 31 rupees): ARROW must be masked,
   and shop steps need a wrong-ware and a re-entry rule.
3. **0x42's pond refills on every visit** (the documents say flag $80 keeps the stairs): any rule keyed on the flag
   would never offer the second play. With the whirlwind's arrival as a non-neighbour jump landing at X 0 still
   carried, the leg's two new step kinds are now pinned to frames.

## How the walks were staged

- **Dungeon rooms**: `triforce/probe.py` (`r26_walks.d7_state`/`stage_d7`) from `start.state`, with written before
  the walk-in: the white sword, wooden boomerang, bow and wooden arrows, raft, ladder, recorder (`whistle`), food (0
  after 7:0x28), pieces 1-5 (`InvTriforce` $1F), 11 full hearts (`HeartValues` $AA, `HeartPartial` $FF), the route's
  keys and bombs at that step (5 keys and 3 bombs into the level; `d7_ledger`), rupees 31 for the model runs, and the
  WorldFlags ($77F + room) of every door the 3-key route opened before (0x69 N, 0x38 N, 0x18 E, 0x19 E, 0x1a E, 0x1b
  E, 0x0c E, 0x29 E, both sides) plus 7:0x28's $10 after the Grumble. The pre-entry snapshot is saved as a state and
  every trial loads it with the game's own randomness for its seed (`Game.load(..., randomize_rng=True, seed=s)`).
- **Overworld**: from the route's states (5_14s_c walked onto the triforce; 0_54e_0, 0_52e_0 walked to the next
  room's edge with the monster slots emptied every frame until the scroll starts, `Walk.staging`), then the
  arrival's inventory written (printed by the scripts).
- **Models**: `r26_walk_scaffold.py` copies the data with the route `d7walk` (D7 with three keys, extending d2first),
  D7 rooms tagged goriya / boss-digdogger / dodongo-crossing (`bombs: required` for 0x0c's wall) / wallmaster /
  boss-aquamentus, `requires_bombs` on 0x69, 0x19, 0x1a, 0x0c, 0x29, `ram_randomization: {}` (11 hearts, the
  ledger's bombs). Promoted runs of round 25 (`FROM=triforce-r25`), act mode `temperature:0.5+relax:16` (each
  model's), 20 episodes per state; the planner `temperature:0.5+relax:16+plan:16:1`, 10 per state.

## 1. Rupees and bombs at 0:0x0b on chain arrivals (walked)

- Chains: `triforce eval-chain --from 5_76s_<c> --runs 100 --parallel 10 --harvest DIR` for c = 0, 1, 2 (77, 67, 70
  successes), with the round 25 promoted team. The 5:0x14 handoff states (214) hold what Link carries out (nothing
  between 5:0x14's entry and 0x0b changes rupees, bombs or keys): `r26_leg_walks.py ledger`.
- **Rupees 168-180, median 171** (the 5_76s states hold 170, 170, 168; Dungeon 5 adds 0-10). After 80 (arrows) and 60
  (food): **28-40, median 31**. Neither reading of `rest-of-game.md` holds: "142-154" (draft) and "157-168 at 5:0x76"
  (`d7.md` §4) are both low. 88-100 after the arrows: **the 0x34 key (80) is affordable on every arrival**.
- **Bombs**: harvested arrivals mean 3.26 (0: 34 of 214, 3: 108, 7: 39); round 25's `new-d5` (603 arrivals) 3.34 (0 in
  14%), `new-full` (186) 3.06 (0 in 17%). `d7.md` §4's "~3" stands; `late-mechanics.md` §1 "0 in most chains" is wrong.
- **Keys** 5 (598 of 603; 4 in 5).

## 2. The whirlwind from 0x0b (walked)

- Staged: 5_14s_0..2 walked Up onto the triforce (42 frames), the level ends, Link plays at 0x0b 538-600 frames later
  at **(0x70,0x7D) facing down, $523 = 0** (also on 72 harvested arrivals). B on the first frame after a state load
  reads nothing (the scripts wait one frame).
- Play 1 steps $523 0 → $FC (&7 = 4, L5). B during its 152-frame freeze is ignored (no play, no step). **The second
  play counts from +152 to +216 frames after the first**; from +217 Link is already being picked up: no play, no step,
  he lands in 0x0b. Two plays facing down or left → **0x45**; one → 0x0b; facing up or right: one → 0x3c, two → 0x74.
- Ride (two plays, second at +160): pickup at +364 (Link state $40, invincible 24), mode 6 at +432, mode 7 with
  RoomId flickering 0x44/0x45 (the room west of the landing), mode 5 in 0x45 at +536 with Link at (0x00,0x7D) →
  (0x00,0x8D) carried by a new whirlwind (slot 9), **released at (0x80,0x8D) facing E at +622/+627**. One play: 0x0b at
  (0x80,0xAD), +475. `IsWhirlwindSummoned` ($508) is 0 again on arrival.
- 60 seeds × 3 states, plays at +0/+152: 60/60 in 0x45, 7 lost half a heart; 72 harvested arrivals: 72/72, 10 lost
  half a heart (8 to the Peahat at the release spot, 2 to the Leever).
- **Executor and chain view**: WAIT acts take 17 frames until the pickup; the act covering it takes 177 frames and
  reports `changed_location` 0x0b → 0x45 directly (no 0x44 record), interrupt `enemy-near`, Link halted at (0x00,0x8D);
  the next act takes 87 frames (the carry). Type $2E is `ObjectClass::Other` (not an enemy). RECORDER is masked at
  0x0b today (legal: MOVE[S] WAIT).

## 3. The shops 0x44 and 0x34; 0x34's armos (walked)

Staged: 0_54e_0 walked N into 0x44, then written rupees 171, bombs 3, keys 5, hearts 11, `InvTriforce` $1F,
recorder; 0x34 entered live through 0x44's N gap (X 0x80) after the arrows. `r26_leg_walks.py armos` (runs `shops`).

- Cave: in at (0x70,0xD5) 98 frames after Up on the mouth (0x44: tile (8,2), Link (0x40,0x4D)); the text halts Link
  ~150 frames (0x44) / ~215 (0x34); he cannot go above Y $8D; the bottom row ends at X $80.
- **Right ware**: Up to Y $A5, Right to X $98, Up to Y $9D → bought; 0x44 arrows 171 → 91 (160 frames of count), 0x34
  food 91 → 31. **A purchase sets OW WorldFlags bit $10** ($6C3 0x44, $6B3 0x34) and the other two wares vanish.
- **Wrong wares**: Up to $9D at X $70 then Right crosses X $78: at 0x44 it **buys the bombs** (171 → 151, bombs +4);
  at 0x34 the blue ring (250) is unaffordable, nothing happens. Left at $9D: 0x44 shield (171 → 62), 0x34 **key (91 →
  11, keys 6)**. `late-mechanics.md` §2.1's "0x34 cannot sell the wrong affordable ware except food" is wrong at
  every measured rupee count.
- **Re-entry**: back in the 0x44 cave after buying, the shopkeeper and wares are back and the arrow sells again (91 →
  11, arrows already held). Flag $10 does not close a shop.
- Out of 0x44's cave Link stands at the mouth (0x40,0x4D).
- **Armos** (secret square (8,8), tiles C0 C2 / C1 C3): wakes on the first frame of RIGHT from (0x30,0x7D), LEFT from
  (0x50,0x7D), DOWN from (0x40,0x6D) (type $1E at (0x40,0x80), slot 11); from below Link walks up to Y $85 and is
  touched as it wakes. 63 frames later the tiles are 70 72 / 71 73 (stairs). Into the stairs from W, Leevers live, 40
  seeds: holding RIGHT 24/40 and 25/40 (76 frames; the rest knocked off by the Armos $1E and the Leevers $0F); tap,
  step back W 12 frames, wait 60, walk in: 40/40 and 23/40 on two staged entries; 0-6 hearts lost.
- Out of the food cave: (0x40,0x6D) facing down, statue back (C0..C3), flag $10. **DOWN from there wakes the Armos
  again** on the next frame; LEFT/RIGHT/UP don't.

## 4. 0x42's pond (walked)

Staged: 0_52e_0 walked to 0x52's N gap (monsters emptied), written as after 0x34 (31 rupees, food, arrows), 0x42
entered live from S. `r26_leg_walks.py pond`.

- In at (0x70,0xDD) facing up. B does nothing for the first 4 frames of Up; **from the 5th (Y $D5) it plays**.
- $523 unchanged (0 → 0). `SecretColorCycle` 1 → $C one step per 8 frames after the freeze; at **B+232** tiles (12,10)
  (13,10) (12,11) (13,11) go 90 90 95 95 → **70 72 71 73** and flag $80 is set. Stairs: Down to $BD, Left to X $60, Up:
  mode $10 at (0x60,0x8D) after 36 frames; 7:0x79 plays 122 frames later at (0x78,0xCD).
- **The pond refills on every visit**: drained (flag $80, stairs drawn), out S, back in: tiles 90 90 95 95, cycle 0, Up
  from X $60 stops at Y $A5; a second play drains it again (same frames; index still 0). Left mid-cycle the cycle runs
  back during the scroll and the flag stays 0. Code agrees: the overworld layout turns only tree, $EA and rock-wall
  squares into stairs on flag $80 (asm:RoomLayout_LayOutRoomOrCaveOW), not the pond.
  **Corrects** `level-7.md` "Once revealed, room flag `$80` keeps the stairs open on every later visit, so the recorder
  is needed there once", `d7.md` §1 "room flag `$80` then keeps them open (code)" and `late-mechanics.md` §2.8 "(kept)".

## 5. 7:0x1c: kill one, leave W, re-enter (walked)

`r26_d7_walks.py digdogger`. In at (0x10,0x8D); after 6 frames of Right (X $18) the Game mask offers RECORDER (not at
X $10); three small Digdoggers out of their clouds 240 frames after B. One or two put in the death state, Left out
into 0x1b: 0x1c's flag becomes $E2 (all dead banked); back in: **no Digdogger, N into 0x0c open**. None killed: the big
one is back, N shut. With a **sword kill** (stand inside the door, swing at one in reach): 12 of 20 seeds killed one,
and all 12 found the room empty with N open (the other 8 fell to 2 hearts first; 2.5-9 hearts lost, statue fire and
2-heart touches).

## 6. 7:0x0c: the E wall bomb among the Dodongos (walked)

`r26_d7_walks.py dodongos`. Up to $BD, Right to $D0, Up to $8D, bomb facing E: **opened 100/100 (two runs of 40 and
60), never eaten**. Waiting 0-239 frames at the spot first (190 seeds): never eaten; not opened in 65 — every one
knocked off the spot (mostly to (0xD0,0x6D)) by a Dodongo touch or statue fire before the bomb. Hearts lost by
Dodongos ($31) and statue fire ($55). The dodongo-crossing model (bombs `required`) crossed 18/20, −3.4 hearts.

## 7. 7:0x0d: a played clear, the E push, a capture in a chain (walked)

- Standing still at the entry (one probe walk): hands out from frame 13, Link knocked along the wall by a Bubble,
  captured at frame 256 (½ heart), in 7:0x79 after the drag (mode 3 at 395). (`level-7.md`'s 161 frames is one
  sample; Obs6 saw no capture in 300 frames with hearts held.)
- **Script** (`r26_wallmaster_bot.py`): lure at exactly (0x20,0x8D); once a hand is out, step to X $30 (a square
  further with the sword blocked), and swing facing W when the hand passes within reach. **40/40 clears, 0 hearts
  lost**, 1006-4353 frames (median 1739), median 4 Bubble touches. Then Left to $20, Up to $5D, Right to $D0, Down to
  $8D, Left: the block moves after 17 frames; stairs in 35/40 by the script (5 misses were the script's own path).
- **Pushed E** from inside the C of blocks (0xB0,0x8D): the block goes to X $D0, the stairs still appear at
  (0xD0,0x60); Link walks out the C's open W side, up X $20, along Y $5D: stairs 6/6, ~307 frames. game.yaml's note
  "pushed E Link is left inside the ring" is wrong: a detour, not a softlock.
- **Chain** (`eval-chain --route d7walk --from 7_0dw_0`, 20 runs, wallmaster model): 14 captured; the runner records
  0x0d as **`exit-wrong`** (½ heart lost), then 7:0x79 off-route (`step: None`), and the route re-plans from 0x69
  (step of `d7walk`), where the goriya model died in all 14. 6 runs ended `wallmaster exceeded 2000 steps`.

## 8. The ladder refused in 7:0x49 and 0x18 (walked)

`r26_d7_walks.py ladder`: Up from the entry after 0/120/300 frames, 40 seeds each. 0x49: Link stood at the water edge
(Y $85) with all 11 monster slots full and no ladder in 15/40, 2/40, 5/40 runs, for 1-86 frames; the slots then held
3 Blue Goriyas, 3 boomerangs ($5C), 3 Keese, 2 Bubbles. **All 120 crossed.** 0x18 (row 5): 0/40, 1/40, 3/40, ≤ 11 frames.
A delay under boomerang fire, never a block.

## 9. 7:0x1a's block clip by the agent's CLIP action (walked)

`r26_d7_walks.py clip` with `Game.set_clips(True)`: `skip_available` offers CLIP N at (0x60,0x9D), (0xA0,0x9D),
(0x80,0x7D), (0x70,0xAD), (0x90,0xAD) and CLIP S at 8 points above. From the W entry, Goriyas live: Down to $9D,
Right to $60, `CLIP N` → (0x60,0x8D) every time, `MOVE E` → stairs: **33/40 and 16/20** (225-257 frames); misses: CLIP
not offered on an arrival frame when a Goriya had just hit Link (offered 1 frame later), knocked off on the way. 0-1
heart lost.

## 10. 7:0x28: FOOD by the agent's action (walked)

`r26_d7_walks.py food`: in the doorway (0x78,0xCD) FOOD is never offered (200 frames). Holding Up, Link stops at Y $C5
under the text and FOOD[N,W,E] is offered from frame 127; `FOOD N` takes 66 frames + 124 menu frames (the sub-screen
switch) and the Grumble takes it within the act: `InvFood` 0, Grumble gone, flag $34 (bit $10). Up through N: 78 + 54
frames into 0x18. With the boomerang thrown first FOOD leaves the mask, offered again 42 frames later, then works.

## 11. 7:0x2a from W with the boss-aquamentus model (walked)

20 episodes from 11 hearts: 6 wins (30%, 3.5-8 hearts lost each, median 4258 frames), 11 timeouts (400 steps), 3
deaths to Aquamentus; damage 99 hearts from its head, 45.5 from fireballs over the 20; 5.9 arrows a run. Planner:
3/10, −1.85 hearts, 7 timeouts. The model reaches the boss but does not finish it from the W.

## 12. Every forced fight and crossing of level 7 for real

Models: round 25's promoted runs (goriya, boss-digdogger, dodongo-crossing, wallmaster, boss-aquamentus) from 11 full
hearts with the ledger's bombs; killers from `r26_replay_damage.py` (nearest object at the death's last damage).
Scripts: `r26_d7_walks.py crossings` (40 seeds, blind walk of the trace's path) and the bots above.

| Room | Who | Success | Hearts lost (mean) | Frames (median) | Killed by / hurt by |
|---|---|---|---|---|---|
| 0x69 S→N (bomb N) | goriya / planner / script | 0/20 / 0/10 / 31/40 | 11.0 / 11.0 / 2.4 | 1654 (deaths) / – / 263 | Blue Goriya 16, boomerang 4 / Blue Goriya / Goriya 71.5, boomerang 24 |
| 0x59 S→N | goriya / planner / script | 20/20 / 10/10 / 40/40 | 1.25 / 0.95 / 1.3 | 313 / 386 / 160 | Goriyas, boomerangs |
| 0x49 S→N (forced) | goriya / planner | 1/20 / 9/10 | 10.7 / 1.45 | 1396 (deaths) / 2046 | Blue Goriya 14, Keese 3 |
| 0x39 S→W | boss-digdogger (RECORDER on) / planner / script row 1 / script door row | 2/20 / 2/10 / 40/40 / 39/40 | 10.0 / 8.2 / 1.2 / 0.8 | 3816 / 6606 / 189 / 143 | the model splits it: small Digdoggers 149.5 hearts; script: the big one ($38) |
| 0x19 W→E (bomb E) | goriya / planner / script | 0/20 / 0/10 / 30/40 | 11.0 / 6.0 / 3.3 | 994 (deaths) | Blue Goriya 15, boomerang 5 |
| 0x1a W, plain fight + push | goriya / planner | 7/20 / 7/10 | 9.0 / 2.85 | 1847 (wins) | Blue Goriya, boomerang |
| 0x1b W→E (6 Blue Goriyas, 0 bombs) | goriya / planner | 0/20 / 5/10 | 10.75 / 3.7 | 948 (deaths) | Blue Goriya 12, boomerang 7 |
| 0x1c W→N (3 small Digdoggers) | boss-digdogger / planner / sword-kill script | 2/20 / 8/10 / 12/20 | 10.4 / 7.1 / 2.5-9 | 6071 (deaths) | statue fire 9, small Digdogger 9; the big one 48 hearts before the play |
| 0x0c S→E (bomb E) | dodongo-crossing / planner / script | 18/20 / 7/10 / 40/40 | 3.4 / 1.0 / ≤ 1 | 1874 (wins) | Dodongo 54.5, fire 13.5 |
| 0x0d W→stairs | wallmaster / planner / bot | 0/20 / 5/10 / 40/40 | 0.35 / 0.1 / 0 | – / 5088 / 1739 | 14 captures, 6 timeouts |
| 0x2a W→E | boss-aquamentus / planner | 6/20 / 3/10 | 6.5 / 1.85 | 4258 (wins) | Aquamentus 3 deaths |

- **Arrows**: every model fired the bought arrows (a rupee each, 31 rupees): 14.7 a run in 0x1c, 7.8 in 0x39, 5.9 in
  0x2a, 5.5 in 0x0c, 4.3 in 0x1a — the ARROW mask is needed before any D7 training (`d7.md` §5 #10, now walked).
- **0x39**: the big Digdogger's first position varies with the spawn cycle over nine spots ((0x80,0x8D) 13 of 80,
  (0x90,0x7D) 11, (0xA0,0x5D) 10, (0x60,0x7D) 10, (0xA0,0x9D) 9, (0x50,0x5D) 9, (0xC0,0x6D) 8, (0x30,0x6D) 7,
  (0x50,0x9D) 3). Over 40 seeds the door row was touched less (10/40) than row 1 (18/40). **Corrects** `level-7.md`
  "the Digdogger starts at X `$50`, Y `$9D`... door row 17 of 30, row 1 1 of 30" (one spawn, not the room). With
  `recorder: true` boss-digdogger splits it (as `d7.md` warned): RECORDER must be masked in 0x39.

## Not staged, and why

- **No chain arrival at 0x44, 0x34, 0x42 or in Dungeon 7**: the route ends at the D5 triforce; those rooms were
  reached from the route's own states with the monsters emptied on the way and the arrival's inventory written.
- **A bomb eaten at 0x0c's wall**: not provoked in 330 bombs from the wall spot; other bomb spots were not tried.
- **0x18 entered dark with its real fights**: probe lit; only the ladder was measured, not its Goriya fight.
- **Real chain capture handling past 0x69**: the goriya model died there in all 14 replays from 0x79, so the 13-room
  way back was not measured end to end.
