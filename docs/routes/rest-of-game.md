# The rest of the game: Dungeon 7 to Zelda (synthesis, Nov 16)

## What this is

- **The stretch:** from the D5 triforce (overworld 0x0b) to the end of the game along `docs/routes/first-quest.md`'s
  draft: the leg to Dungeon 7, Dungeon 7, the magic sword (0x21's gravestone), Dungeon 6, the leg and Dungeon 8, the
  leg and Dungeon 9, Ganon and Zelda.
- **The owner's decisions, Nov 16:** (1) the honest planner (`plan:K:S`) runs beside the models as a second, labelled act
  mode (golden path, play, teacher); the models alone stay the primary number; (2) breadth first to Ganon: each dungeon
  and leg gets rooms, states, families and a first trained team at a working rate (about 30% or better), one
  optimisation round after the first complete run; (3) run through, with a report, a tag and a push at the end of each
  dungeon, stopping only for the owner's decisions (route, cheats, rules); (4) clips on is primary, every new stretch also
  gets its plain way and a clips-off number, untuned; (5) bombs, rupees (at the two shops) and keys that are missing
  are cheated and counted per room, aim zero, no route change; (6) breaking models is allowed when a family cannot be
  trained without it (`convert-model` where the models stay identical, otherwise every family retrained, with the
  COMPAT bump).
- **Standing:** MINOR +1 per dungeon (1.6 = Dungeon 7 ... 1.9 = Dungeon 9); no potions, no blue ring, no wand, no
  book, no magic key; the red ring in level 9 is taken.
- **The six documents** (all in `docs/routes/`, mined from the disassembly, the level documents and the project's code
  before anything was built): `d7.md` (the leg from Dungeon 5 and level 7; proposed steps 209-247, inventory, work
  list), `d6.md` (the magic sword and level 6), `d8.md` (the leg and level 8; the whirlwind index table), `d9.md` (the
  leg, level 9, Ganon, the end), `late-monsters.md` (what each monster demands of observation, actions, masks,
  rewards), `late-mechanics.md` (the game's rules against what the route engine knows).
- **How to read it:** every row names the section it comes from (`d6.md` §4.5 is that document's section 4.5); the
  basis marks (trace / code / data / draft / inferred) carry over; a judgement made only here is marked (synthesis).

## The survival budget

Hearts refill only at a triforce piece (code, `late-monsters.md` §0) and by drops (heart +1, fairy +3, the 16th
damage-free kill's fairy; `late-mechanics.md` §3.15). A heart container adds one container and one heart, no refill
(`late-mechanics.md` §3.14). Dungeon 9 has no piece and the Triforce of Power refills nothing
(asm:Item_UpdatePowerTriforceFanfare@FillHearts is unreachable; `d9.md` §3.8, `late-mechanics.md` §0.5): **the D8
triforce is the last refill of the game.** No magic shield, no blue ring: every `$55`/`$56`/`$57`/`$58`/`$59` shot is
unblockable (`late-monsters.md` §0). The beam needs full hearts, so it is gone after the first hit of every fight
(asm:SwordShot_Make). A hit of D hearts kills at D hearts or fewer shown (inferred, `late-monsters.md` §1). Bubbles do
no damage but block the sword 151-160 frames and zero the 10/16-kill counters (`late-monsters.md` §3.2).

| Stretch | Max hearts; last refill | Sword | Forced fights and clears, in order: monsters, hits to kill each, damage per touch / shot | Hearts the fights-skipped walk lost | Touches that kill | Sources |
|---|---|---|---|---|---|---|
| D5 → D7 leg (0x0b ... 0x42) | 11; the D5 piece | white `$20` | none forced; crossed: 0x34's 4 Blue Leevers (1 a touch) and the secret Armos (`$30`, 2 hits; 1 a touch; from below 1, from the side 0) | ~1-2 (inferred from d2first visit costs) | 11 one-heart touches | `d7.md` §1, §4; `late-mechanics.md` §2.5 |
| Dungeon 7 | 11, 12 after Aquamentus; the D5 piece | white | 0x49: 3 Blue Goriyas (3 hits; touch 1, boomerang 1, parryable), 3 Keese (1; ½), 2 Bubbles; 0x1a plain way only: 3 Blue + 3 Red Goriyas (15 hits; red touch ½); 0x1b for its bombs: 6 Blue Goriyas (18 hits, six boomerangs); 0x1c: 3 small Digdoggers (4 hits each, 12; touch 2; 1.25-1.5 px/frame), 4 statues (½); 0x0d: 5 Wallmasters (1 hit, only on a trip; ½ + capture), 3 Bubbles; 0x2a Aquamentus (3 hits or 2 bombs; touch 1, three fireballs ½ each) | ~11 over the dungeon (trace), 2-3 each in 0x69 and 0x19 | 6 two-heart Digdogger touches; 11 one-heart; 22 half-heart | `d7.md` §2, §4; `late-monsters.md` §3.1-§3.6, §4 |
| D7 → sword → D6 leg (0x42 ... 0x22) | 12; the D7 piece | white until 0x21's cave, then magic `$40` | none forced; crossed: 0x50 5 red Lynels (2 white; touch 1, shot 2) on a slow corridor; 0x40/0x41/0x31/0x21 Ghinis (5 white, 3 magic; touch 1) and immortal flying Ghinis (1); 0x32 2 blue Lynels (2 magic; touch 2, shot 2), 2 red; 0x22 red Lynel | **2.2 by a straight scripted walk, clip way, the sword stop included (walked, `d6-walks.md` §1-§2)** | 6 blue-Lynel shots; 12 one-heart | `d6.md` §1.1-§1.4; `late-monsters.md` §3.22; `d6-walks.md` |
| Dungeon 6 | 12 minus the leg, 13 after Gohma; the D7 piece | magic | 0x58: 8 Keese (any hit; ½), 4 statues (½); 0x38: 2 Red Wizzrobes (1 hit; touch 1, **shot 4**), 2 Blue (3 hits; touch 2 even while flashing, shot 2), 3 Like-Likes (3 hits; 1 + hold), 1 Bubble, then the push; 0x39: 5 Vires (1 hit; 1), dark; 0x3a: as 0x38; 0x2d (for level 9's key): 3 Vires, 2 Bubbles; 0x1c Gohma: one arrow into the open eye (touch 2, fireball 1 every 65 frames) | 12-22 (draft); 0x48 traps 2 hearts in 30 frames on the door row (trace) | **3 Red Wizzrobe shots from full**; 2 Red + 1 Blue at 9-10 | `d6.md` §2, §4.5, §5, §6; `late-monsters.md` §3.7-§3.10, §4 |
| D6 → D8 leg (0x22 ... 0x6d) | 13; the D6 piece | magic | none forced; 0x22 red Lynel (touch 1, shot 2: a hit costs the whirlwind); Octoroks, Moblins ½; own candle fire ½ | **1.9 by straight scripted walks, 0.6 with ow-octorok in 0x4c (walked, `d8-walks.md` §1)**; the whirlwind and the tree 0 | 13 one-heart | `d8.md` §2.3, §7.2 |
| Dungeon 8 | 13, 14 after Gleeok; the D6 piece | magic | 0x6e crossed with Manhandla (hands 1 hit or a bomb; touch 1, fireball 1) while bombing its wall; 0x5e: 5 Blue Darknuts (2 hits, not from the front; touch 2), 4 statues (½); 0x4e, 0x3f crossed (Blue Darknut 2, Red 1, Gibdos 2, Bubbles); 0x4c crossed among 8 Pols Voices (3 hits; touch 2); 0x3e: 6 Blue Darknuts, statue fire; 0x3c Gleeok, four heads: 9 hits (3+2+2+2), up to 3 immortal flying heads (1), neck base 1, attached head 3/8, fireball 1 | **10.0 lost by the best plays seen, 2.5 taken back from the Darknuts' drops (walked, `d8-walks.md` §7)**: 0x6e 0.5, 0x5e 2.7, 0x4e 0.3, 0x3e 3.3, 0x3f 0.9, 0x4c 0.4 with arrows, Gleeok 1.9 from 13 hearts (3.0 from 6) | 7 two-heart touches | `d8.md` §4-§6, §7.1-§7.2; `late-monsters.md` §3.11-§3.14 |
| D8 → D9 leg (0x6d ... 0x05) | 14 full; 15 after 0x47 (+1, no refill); **the D8 piece is the last refill** | magic | none forced; 0x06 and 0x05 two blue Lynels each (touch 2, shot 2, unblockable), 0x05's bomb wait among six monsters; 0x17 ladder among Boulders; 0x38/0x48 Zora fireballs ½ unblockable | **walked Nov 17: 8.6 by a straight scripted walk, 7.2 by the best play per room (0x06 2.2, 0x05 2.9)** (`d9-walks.md` §1.2) | the 7th two-heart touch at 14 (the 8th at 15) | `d9.md` §1.2; `late-mechanics.md` §2.2; `d9-walks.md` |
| Dungeon 9 before the red ring (0x76 ... 0x00) | ≤ 15 minus the leg | magic | 0x55: 2 red Lanmolas (10 segment hits, 1 each; touch 2); 0x14's 5 Like-Likes in a spiral; 0x07 on the ring detour: 3 Blue (3 hits) + 2 Red Wizzrobes (1 hit), 3 Bubbles (11 hits; Red shot 4, Blue 2); crossings 0x65, 0x15, 0x17 with Wizzrobes and bomb waits | **walked Nov 17, crossings only: 12.6 (0x65's bomb wait 4.5, 0x27's 4.1, 0x17's 2.8; 0x14 0.05 fought with the sword, 4.4 walked)**; the fights are the other walk's (`d9-walks.md` §5.1) | from 15: the 4th Red shot, the 8th two-heart touch | `d9.md` §3.1, §4.1, §4.4; `late-monsters.md` §4; `d9-walks.md` |
| Dungeon 9 after the ring (0x00 ... 0x32) | ≤ 15, no refill; ring divides all damage by 4 | magic + silver arrow | 0x16 Patra (8 children 2 hits + body 3 = 19 hits; touch 2 → ½) for its bombs; 0x05, 0x20, 0x10 (Wizzrobes, 11 hits each; 0x10 + 3 Bubbles); 0x61 Patra (19); 0x30 (2 Blue + 2 Red among 4 Traps, 8 hits; Trap 1 → ¼); 0x03 (2 Zols, 2 Like-Likes, 2 Bubbles, 8 hits); 0x52 Patra (19); 0x42 Ganon: 4 sword hits while invisible, then one silver arrow in ~510 frames (touch 4 → 1, fireball 1 → ¼); 0x32's guard fires (1 hit; ⅛) | **walked Nov 17, crossings only to Ganon's door: 3.3 with the ring (the same rooms without it 8.9 from 0x06 on)**; the route part in all 23 of 15 with the ring, 21 without (`d9-walks.md` §6) | ~60 one-heart-equivalent hits; 15 Ganon touches | `d9.md` §3.1, §4.1-§4.5, §5, §6; `late-monsters.md` §3.15-§3.18; `d9-walks.md` |

Clips change the forced list: the block clip skips 7:0x1a's fight (`d7.md` §2), and five level 9 ring rooms (0x55, 0x07,
0x61, 0x20, 0x03) share the clippable layout `$1A`: 14 forced Wizzrobes and 2 Patras with clips, 24 and 3 without
(inferred, not walked; `d9.md` §3.4). Every fight cost in this table is from the code: no fight of Dungeons 7 or 9
except Ganon, and in Dungeon 6 only Gohma, one Vire and one Like-Like, was played (`late-monsters.md` §8, `d6.md` §12).

**Walked in round 27 (`d6-walks.md` §7), the leg to Dungeon 6 and Dungeon 6 with live monsters:** the leg costs a
straight walk 2.2 hearts; Dungeon 6 without 0x38 and 0x3a costs 8.2 by the best plays seen (blind walks in the
crossings, today's models in 0x58, 0x39, 0x2d, a script for Gohma): 10.3 of 12 before the two block rooms, which
cost 4.75 and 6.1 hearts in their best wins (like-like zero-shot: 4 of 10 with the planner, 10 of 20) and are lost
outright from 8 hearts (2 of 20, 0 of 20). The table's "12-22 (draft)" for Dungeon 6 is of that order. A doorway is
out of reach of Wizzrobe shots and Gohma's fireballs (0 of 320; 0 in 3000 frames).

**Walked in round 28 (`d8-walks.md` §7), the leg to Dungeon 8 and Dungeon 8 with live monsters:** the leg costs 0.6
hearts. In Dungeon 8 the best plays seen are scripts that read positions only: both Darknut rooms cleared 30 of 30
(2.7 and 3.3 hearts lost, 1.0 and 1.5 taken back from drops), 0x4c and its corner key with arrows 30 of 30 at 0.4
hearts and 3 rupees, Gleeok killed 30 of 30 from 13 hearts and 26 of 30 from 6. Link reaches Gleeok with about 6.8
hearts and leaves with about 4; about 0.7 of such runs would get through. **Today's models alone win next to nothing
in 0x5e and 0x3e (1 of 80), 0x4c (0 of 60) and Gleeok's room (0 of 40)**, nor the whirlwind step out of 0x22 or
0x6d's burn (0 of 40 each).

**Walked in round 29 (`d9-fights-walks.md`), the fights of Dungeon 9 with live monsters:** no model family clears
any of them (0 of 1,760 episodes, the red ring on); a rule-based fighter reading RAM positions clears all of them, and
its hearts are the best play seen: Lanmolas 1.0, the two `$47` Patras 0.7 and 2.5, 0x16's `$48` Patra 8.65 (2.33
with the ring), the Wizzrobe rooms 5.8-13 each without the ring and 1.5-6 with it, 0x07 before the ring 9.35, Ganon
1.25 with RAM eyes (screen only: 10-14 of 20 without the ring). **The fights alone sum to 41 hearts without the ring
and 26 with it, of 15 with no refill**: the table's "~60 one-heart-equivalent hits" after the ring is not a margin.
The Lanmolas do not return once a head has died; a Patra's pattern stops flipping when its slot-3 child dies; Ganon
stands on the room's spawn spot, not at one place.

**Rooms a 11-15 heart Link cannot be expected to cross the way today's models cross rooms**, ranked (synthesis; each
dossier's own ranking is in its section):

1. **9:0x07 and the pre-ring crossings 0x65, 0x15, 0x17**: Red Wizzrobe shots at 4 of ≤ 15 hearts with no refill since
   the D8 piece; "the single worst spot of the game" (`late-monsters.md` §4, §9; `d9.md` §4.1, §9).
2. **6:0x38 and 6:0x3a**: forced, shut in (0x38 has no retreat), three Red shots kill from full, the Bubble takes the
   sword, Red phase unobserved and flashing Wizzrobes marked harmless (`d6.md` §4.5, §10; `late-monsters.md` §3.7).
3. **9:0x42 Ganon**: invisible body, four timed stabs, a silver arrow inside ~510 frames, a sealed room, a rupee per
   try (`d9.md` §6, §9; `late-monsters.md` §3.17).
4. **8:0x3c Gleeok with four heads**, reached after 0x5e and 0x3e with no refill; the D4 two-head Gleeok already kills
   27% of arrivals at 2.9 hearts (`d8.md` §6, §10).
5. **7:0x0d**: five Wallmasters lured at the wall the critic punishes, Bubble knockback along it, a capture costs 13-15
   rooms (`d7.md` §2, §5; `late-monsters.md` §3.3).
6. **8:0x5e and 8:0x3e**: Blue Darknuts at 2 hearts a touch with statue fire the darknut family never saw, 0 bombs
   (`d8.md` §4.2, §4.4).
7. **The three forced Patras (9:0x16, 0x61, 0x52)**: 19 hits on nine orbiting targets, body untouchable until the last
   child dies (`d9.md` §4.5; `late-monsters.md` §3.16).
8. **9:0x05, 0x20, 0x10, 0x30** after the ring: forced Wizzrobes, Bubbles, Traps on 0x30's stairs (`d9.md` §4.1-§4.2).
9. **7:0x1b then 7:0x1c**: 18 hits under six Blue boomerangs, then three fast 2-heart Digdoggers with statues, at 11
   hearts after the blind walk's ~11 of contact (`d7.md` §2, §5).
10. **6:0x2c, 0x28, 0x78, 0x29**: crossings where the Reds appear on Link's path (`d6.md` §2.3).
11. **8:0x3f**: one way, a one-wide corridor where a W-facing Darknut cannot be hit, Bubbles (`d8.md` §5.2).
12. **9:0x55** Lanmolas before the ring; **6:0x1c** Gohma (one arrow, eye unobserved); the overworld Lynel rooms 0x50,
    0x32, 0x06, 0x05 (`d9.md` §4.4; `d6.md` §6; `late-monsters.md` §9).

Chains compound: 95% a room is a few percent from Dungeon 9's entrance to Zelda (inferred, `d9.md` §9); 70% in 0x38 ×
60% in 0x3a is under 40% before D6's crossings (inferred, `d6.md` §10). One-room families trained at `hp: [0.15, max]`
will meet arrivals at 2-5 hearts in 7:0x1b/0x1c/0x0d/0x2a (inferred, the Dungeon 5 pattern; `d7.md` §5).

## Ledgers on a run with no drops

Clips on, the draft's way, from the D5 triforce to Zelda. Two rupee readings are carried (see the note after the
table).

| Where | Max hearts | Keys | Bombs (cheated, counted from the D5 triforce) | Rupees: draft | Rupees: leg states | Sources |
|---|---|---|---|---|---|---|
| D5 triforce, 0:0x0b | 11 full | 5 (6-7 in some D5 states) | ~3 (see the note) | 142-154 | 157-168 + Dungeon 5's drops | `d7.md` §4; `late-mechanics.md` §1 |
| 0x44's shop: wooden arrow | 11 | 5 | 3 | 62-74 | 77-88 | `d7.md` §1; `late-mechanics.md` §2.1 |
| 0x34's armos shop: food | 11 | 5 | 3 | 2-14 | 17-28 | same |
| 7:0x69 N, 0x19 E, 0x1a E walls; key doors 0x38 N, 0x18 E | 11 | 3 | 3 → 0 (with 0 in: 3 cheated) | | | `d7.md` §4 |
| 7:0x1b: 6 Blue Goriyas for +4; key door E | 11 | 2 | 4 | | | `d7.md` §2 |
| 7:0x0c E, 0x29 E walls | 11 → 12 | 2 | 2 | | | `d7.md` §4 |
| D7 out, 0:0x42 | 12 full | 2 | 2 (0 cheated with 3 in) | 2-14 | 17-28 | |
| 0x21's cave: the magic sword (exactly 12 containers) | 12 | 2 | 2 | | | `d6.md` §1.5 |
| 6:0x79 W key door −1, 0x58 +1, 0x28 E wall, 0x2d +1, 0x2c/0x1c key door −1 | 12 → 13 | 2 | 1 | Gohma: −1 an arrow | | `d6.md` §0, §3 |
| D6 out, 0:0x22 | 13 full | 2 | 1 | 1-13 | 16-27 | `d8.md` §1 |
| 8:0x6e N wall | 13 | 2 | 0 | | | `d8.md` §5.1 |
| 8:0x5e +1, 0x4e N −1, 0x4c corner +1, 0x4c N wall | 13 | 3 | 0, **1 cheated (8:0x4c)** | | | `d8.md` §3.1, §7.3 |
| D8 out, 0:0x6d | 14 full | 3 | 0 | 1-13 | 16-27 | `d8.md` §7.3 |
| 0x47's take-any: the right ware | 15 (+1 heart, no refill) | 3 | 0 | | | `d9.md` §1.2 |
| 0x05's rock wall | 15 | 3 | 0, **1 cheated (0:0x05)** | | | `d9.md` §3.6 |
| 9:0x65 N, 0x26 E, 0x27 N, 0x17 N walls; key doors 0x14 E, 0x16 N | ≤ 15 | 1 | 0, **4 cheated** | | | `d9.md` §3.5, §3.6 |
| 9:0x16 Patra +4; 0x06 W, 0x20 N walls; 0x63 W key door | ≤ 15 | 0 | 2 | | | same |
| 9:0x61 key +1; leave N holding **2 bombs** (one way from here) | ≤ 15 | 1 | 2 | 9:0x62's Keese: +5 if cleared (optional) | | `d9.md` §3.3, §3.7 |
| 9:0x31 W, 0x04 W walls | ≤ 15 | 1 | 0 | | | `d9.md` §3.6 |
| 9:0x42 Ganon: the silver arrow | ≤ 15, red ring | 1 | 0 | 0-12 | 15-26 | `d9.md` §3.7, §6.5 |

**Bombs.** From the D5 triforce on, a no-drop run cheats 6 bombs with 3 carried into Dungeon 7 (8:0x4c, 0:0x05, four in
level 9) and 9 with none (`d7.md` §4, `d8.md` §7.3, `d9.md` §3.6); `first-quest.md`'s 7 counts the same walls from
Dungeon 5's entrance, 5:0x66 included. Every bomb spent as a weapon, eaten by a 7:0x0c Dodongo (code, `d7.md` §2) or
missed comes off a wall. Room bombs on the path not in the owner's Nov 14 rule: 7:0x69 (+4 for 5 Blue Goriyas, `d7.md` §3
#2) and 8:0x3f (+4 for 1 Blue + 2 Red Darknuts and 3 Gibdos in a one-way room, `d8.md` §13 #1). 9:0x23's +3 net is a
detour and is out by the owner's rule (`d9.md` §3.6).

**Bombs into Dungeon 7 (disagreement).** `d7.md` §4 reads ~3 from round 24's Digdogger arrivals (0, 4 or 8 bombs; 0.94
spent a fight); `late-mechanics.md` §1 says Dungeon 5 ends at 0 in most chains. Settle by reading the bomb count at
5:0x14 in the D5 chain jsonl. The difference is three cheated bombs.

**Rupees (disagreement, reconciled toward the leg states).** `first-quest.md`, `d6.md` §0, `d8.md` §1, `d9.md` §3.7 and
`late-mechanics.md` §1 carry the draft's 42-54 after `d2first`'s 0x67 cave + 100 = 142-154, leaving 2-14 after both
shops. `d5-leg.md`'s ledger measured 68/57/63 out of Dungeon 4 (4:0x11's rupee room and drops), so 157-168 at 5:0x76
and 17-28 after both shops (`d7.md` §3 #9, §4). The leg states are the measured number; the draft's is out of date.
Dungeon 5's own drops are unmeasured. Read the count at 0:0x0b on chain arrivals before trusting either. Consequences:

- At 0x44 the magic shield (130) and the bombs (20) are affordable on either reading (`d7.md` §1, `late-mechanics.md`
  §6). At 0x34 the key (80) is unaffordable on the draft's reading (`late-mechanics.md` §6) and **affordable in 2 of 3
  chains** on the leg states' (77-88 after the arrows, `d7.md` §1): buying it leaves no food and the Grumble blocks
  Dungeon 7.
- Each arrow costs a rupee and is refused at 0 (asm:Arrow_Wield). Gohma needs one, Ganon's silver arrow one. On the
  draft's low end (2) no arrow may miss in the whole game; on the leg states' (17) fifteen may.
- Overworld 0x07's door-repair cave (rock wall at (`$A0`,`$80`)) posts 20 with no test (code,
  asm:Cave_State2TalkOrShopOrDoorCharge): at the low end of either reading it leaves Link with too few for Ganon
  (`d9.md` §1.2, §3.7). With 0 rupees Ganon's room is a dead end (`d9.md` §6.5).
- Sources on the path: 6:0x28's 5 rupees if all 5 Wizzrobes die (`d6.md` §2.2), 8:0x6e +5 if Manhandla dies, 8:0x4e +5
  if all six die (`d8.md` §3.1), 9:0x62 +5 (`d9.md` §3.7), and the tenth damage-free kill's forced 5 (`late-mechanics.md`
  §3.15).

**Keys.** 5 in, 2 out of Dungeon 7; 2 out of Dungeon 6 (0x58's and 0x2d's taken); 3 out of Dungeon 8 (0x5e's and the
0x4c corner key); 1 left after level 9. No cheated key (`late-monsters.md` §0, `d6.md` §0, `d8.md` §3.1, `d9.md` §3.5).
Zero margin in level 9 until 0x61. The ledger holds only if no off-route key door is touched: 7:0x1b N, 6:0x29 N,
8:0x5e E and 0x4e W, 9:0x63 N and S each spend a key on touch (asm:Door_TouchKey). An accidental keyless pass leaves one
key more and the near side locked (inferred, harmless; `late-mechanics.md` §3.3). Some D5 states carry 6-7 keys
(`d7.md` §3 #6).

**Hearts.** Refills at the D7, D6 and D8 pieces only; containers 11 → 12 (Aquamentus) → 12 at 0x21 → 13 (Gohma) → 14
(Gleeok) → 15 (0x47). Any container missed before 0x21 (0x5f, 0x2f, the four bosses) makes the magic sword unobtainable
on this route (inferred, `late-mechanics.md` §2.3).

## Everything the engine lacks

"Inputs" says whether the item changes what a loaded model reads (observation layout or meaning, action space). The
mask is no network input (`late-monsters.md` §2.3), so mask items change no inputs. `triforce convert-model` widens the
vector keys `link`, `room_state`, `history`, `objective`, `target` by name and new action kinds; it does not touch
`entity_types`, `entity_features`, `grid` or `action_mask`, and the Rust actor's all-zero-column drop covers the
towers' first layers, not the entity projection (`late-monsters.md` §6). **Every (a) item that is an entity feature, and
every meaning change of an existing column, is therefore one compatibility break; plan them together** (the owner's decision
6). No new action kind is needed for the rest of the game (`late-monsters.md` §6.4).

### (a) What the models see

| Item | Needed by | Nearest code | Inputs |
|---|---|---|---|
| Red Wizzrobe phase (hidden / fading / solid) and frames to its shot (`ObjState` − `$B0`); Blue teleporting (`ObjRemDistance`), walk timer | `d6.md` §4.4; `d9.md` §8; `late-monsters.md` §2.2 G2, §6.3 `phase`, `phase_time` | `crates/env/src/observation.rs` entity features | yes, entity (break) |
| `targetable_now` (Patra body, Red Wizzrobe parity, Blue teleport, Ganon by phase, Manhandla base, big Digdogger, Peahat) | `late-monsters.md` §6.3, §5.7 | same | yes, entity (break) |
| `hurts_on_touch` corrected: flashing Wizzrobes hurt; hidden Reds, returning monster boomerangs, unemerged Lanmola segments, early Patra children, dying Ganon don't | `d6.md` §4.4, §9 #6; `late-monsters.md` §2.2 G5 | `observation.rs` (`harmless = monster && (stun > 0 \|\| flashing)`) | meaning change (break) |
| `killable` corrected: Patra body, Manhandla base, flying Ghini `$22`, Ganon by phase, guard fires killable | `late-monsters.md` §2.2 G6 | `observation.rs:829-830` | meaning change (break) |
| Gohma eye state, open timer, next-open counter | `d6.md` §6, §11; `late-monsters.md` §3.10 | entity `phase` | yes, entity (break) |
| Ganon phase, brown time left, sword count, visibility timer | `d9.md` §6.1, §8; `late-monsters.md` §3.17 | entity `phase`; Gleeok heads precedent | yes, entity (break); and the owner's question on hidden positions |
| Patra children live; Lanmola segment role; Manhandla base role; slot identity (6:0x78's ringleader) | `d9.md` §8; `late-monsters.md` §2.2 G1, §6.3 `slot_one`, `segment_role` | entity encoder has no slot input (`triforce/model.py`) | yes, entity (break). Manhandla's base as a synthesized hazard (as Gleeok's base, `game.rs`) needs no layout change (inferred, preferred) |
| Trap state (home, charging, returning) | `d9.md` §8, §9 #7 | Traps are hazards today | yes, entity (break) |
| Size of big monsters (Ganon 32×32, Gohma 48 px, big Digdogger, Dodongo) | `late-monsters.md` §2.2 G8 | `Enemy::dims` (`game.rs:150-160`), not observed | yes, entity (break) |
| Damage to Link per entity after the ring | `late-monsters.md` §2.2 G7, §6.3 | none | yes, entity (break) |
| Link on a shooter's line (`on_line`) | `late-monsters.md` §6.3 | none | yes, entity (break) |
| `link`: `sword_blocked`, `sword_blocked_time` (`$4C`) | `d6.md` §5.2; `d9.md` §8; `late-monsters.md` §2.2 G4, §6.1 | the mask's `is_sword_blocked` (`game.rs:1048-1067`) | yes, `link` (convertible) |
| `link`: `arrow_usable`, `rupees_spare` (over a route reserve), `rupees_low` (count/16; /255 cannot tell 1 from 0) | `d6.md` §6, §11; `late-monsters.md` §2.2 G9, §6.1 | `bomb_usable`; `rupees` | yes, `link` (convertible) |
| `link`: `damage_scale` by ring grade | `late-monsters.md` §6.1 | `ring_red` exists | yes, `link` (convertible) |
| `room_state`: `frame_phase_32`, `frame_phase_64`, `person_fireballs`, `ringleader_alive`, `wallmaster_ready` | `late-monsters.md` §2.2 G3, §6.2 | `room_state` | yes, `room_state` (convertible) |
| Old man's text halt (Link `ObjState` `$40`) | `d9.md` §8 | `halted` exists in `link` (synthesis: check it covers the text) | maybe none |
| Whirlwind: index `$523`, summoned `$508`, carrying `$522` | `d7.md` §1; `d8.md` §10; `late-mechanics.md` §2.9, §8 | none; needed only if the whirlwind is learned, not scripted | yes, `link`/`room_state` (convertible) |
| Overworld tile objects (gravestone, armos statue, tree) and revealed stairs; flying Ghinis | `late-mechanics.md` §2.4, §8 | tile objects are in no `GameView` list | yes (break if entities) |
| Shop wares | `late-mechanics.md` §2.1 | drawn through slot `$13`, not objects | yes; a mask avoids it |
| Triforce of Power `$672` | `late-mechanics.md` §3.2 | `triforce_1`..`triforce_8` | yes, `link` (convertible) |

Already observed (synthesis, read in `crates/env/src/observation.rs`): `paralyzed` (`$512`, held by a Like-Like; `d9.md`
§8 lists it as missing, `d6.md` §4.4 as present), `ring_red`, `arrow_wood`/`arrow_silver`, `bombs`, `keys`, `rupees`,
`triforce_1`-`8`, and the sword grade `sword_0`-`sword_3` (`d8.md` §4.4 inferred there was none). Dark rooms need
nothing: the tile grid is RAM (`d7.md` §2, `d9.md` §2).

### (b) What the models can do (masks; no inputs change)

| Item | Needed by | Nearest code |
|---|---|---|
| ARROW: `items: {arrows: false}` for every family but Gohma's and Ganon's (default today: true, no yaml sets it); a route `rupee_reserve` like `bomb_reserve`; ARROW masked at 0 rupees (`weapon_ready` checks the slot only); arrow target masks (Gohma's eye open, Ganon brown) | `d7.md` §5 #10; `d6.md` §6, §11; `late-mechanics.md` §0.3; `late-monsters.md` §6.4; `d9.md` §10 | `ItemsConfig` (`config.rs:352-383`), `bomb_reserve`, `bomb_targets` |
| RECORDER per route step: the whirlwind steps and 0x42 only on the overworld, 7:0x1c yes, **7:0x39 never** | `d7.md` §1-§2; `d8.md` §11.1; `late-mechanics.md` §2.8-§2.9; `late-monsters.md` §6.4 | `actions::recorder_wanted` (`actions.rs:381-395`) |
| FOOD only at 7:0x28 once the text has typed | `d7.md` §2; `late-mechanics.md` §3.11 | `ItemsConfig.food` |
| CANDLE only at 0x6d's and 0x47's trees; never at 0x4d's shop tree | `d8.md` §2.4, §11.2; `late-mechanics.md` §2.6 | `ItemsConfig.candle` |
| BOMB per step: only at the step's wall in 8:0x6e, 8:0x4c and north of 9:0x61; never at Ganon, Patra, Pols Voice, Gohma; `required` at 7:0x0c and 0:0x05; a mode for Wizzrobe rooms | `d8.md` §11.3; `d9.md` §10; `d6.md` §11; `d7.md` §2; `late-monsters.md` §6.4 | `env::config::BombUse`; round 22's wall mask (`required` families only) |
| Off-route key doors: 7:0x1b N, 6:0x29 N, 8:0x5e E, 8:0x4e W, 9:0x63 N/S | `d7.md` §4; `d6.md` §11; `d8.md` §5.4; `d9.md` §3.5 | wrong-exit penalties; asm:Door_TouchKey |
| Doorways into shut-in rooms: 6:0x28 N (Gleeok), 6:0x2c S (0x3c); overworld 0x07's door-repair cave | `d6.md` §5.7; `d9.md` §1.2 | wrong-exit objective |
| Shop wrong wares (MOVE onto X `$58`/`$78`, Y `$95`/`$9D`); 0x47's left ware | `d7.md` §1; `late-mechanics.md` §2.1, §6 | `missed` of the take-any (`objectives.rs:840-843`) |
| Gravestones but the secret one | `late-mechanics.md` §2.4 | none |
| Push only in the step's direction (a ring's W block pushed E closes the stairs until re-entry) | `d9.md` §3.4; `late-mechanics.md` §3.5 | `is_push_block_direction` (`actions.rs:368`) |
| Never A on a dungeon person or the Grumble (fireballs until Link leaves) | `d9.md` §2; `late-monsters.md` §6.4 | `actions.rs:348-356` (safe today only because those rooms hold no enemy) |
| Whirlwind as an executor macro ("play N times facing D, wait for pickup"), if the owner picks it | `d7.md` §10 #6; `d8.md` §11.1; `late-mechanics.md` §7 | the CLIP macro, `crates/zelda/src/executor.rs` |

Built: the RECORDER, FOOD and CANDLE kinds (round 24, 1.4.2.0), ARROW, the sub-screen B switch in the executor, SWORD
and BEAMS masked while a Bubble blocks the sword (`late-monsters.md` §2.3, `d6.md` §11, `d8.md` §5.2).

### (c) Route step types and objective rules (no inputs change)

| Item | Needed by | Nearest code |
|---|---|---|
| **The magic sword is never an objective, and the threshold is wrong**: `SWORD_UPGRADES` asks `max_health >= 13`, the game gives at 12 (`$B0`); no `(0, 0x21, cave)` room, no `treasure: magic-sword`; the ware at X `$78` exactly | `late-mechanics.md` §0.1, §2.3; `d6.md` §1.5 | `objectives.rs:60-64`; the white-sword cave step |
| Overworld stairs as targets: `cave_tiles` sees only a `$F3` mouth or a `bomb_secret`; every new entrance (0x34 armos, 0x21 gravestone, 0x42 pond, 0x6d and 0x47 trees) is `$70` stairs | `late-mechanics.md` §0.2; `d7.md` §1 | `Room::stairs_tiles`; the dungeon branch of `cave_tiles` |
| Secrets as steps: `secret: push` (gravestone, X `$90`, from above or below, flying Ghini expected), `secret: armos` (from the side), `secret: burn` (stand tile + facing, ~94 frames clear of the fire), the recorder pond; `entrance.how` is data read by nothing | `d6.md` §1.4; `d8.md` §2.4; `late-mechanics.md` §2.4-§2.8 | `game_map.rs:226-256` |
| Shops: `cave` exits and in-cave rooms for 0x44 and 0x34; `waypoint: arrows` / `food` done by inventory (`waypoint_done` reads flag `$10`, which any ware sets); wrong ware → `missed`; `requires_rupees` (exists, unused) | `d7.md` §1, §9.3; `late-mechanics.md` §2.1 | `waypoint: heart` (`WAYPOINTS`, `game_map.rs:266`); `cheats.rs` rupees |
| Whirlwind: `exit: recorder` / `entry: recorder`; `route_entry` for a jump (returns None today); plays computed from `$523` and `InvTriforce`; a missed pickup detected and re-planned | `d7.md` §1; `d8.md` §2.1-§2.2; `d9.md` §1.1; `late-mechanics.md` §2.9 | `STEP_EXITS`/`STEP_ENTRIES` (`game_map.rs:263-265`); `objectives.rs:242-260`; the draft's `exit/entry: recorder` |
| Feed the Grumble: FOOD target anywhere after the text; done by 7:0x28's flag `$10` or `InvFood` 0 | `d7.md` §2; `late-mechanics.md` §3.11 | the Digdogger split rule |
| `requires_bombs` on every wall room of the new levels (7:0x69, 0x19, 0x1a, 0x0c, 0x29 have none) and on 0:0x05 (`bomb_secret`) | `d7.md` §2, §3 #10; `late-mechanics.md` §2.7 | 0x2c, 0x67 |
| One-way shutters in the room graph (level 7's seven, 8:0x3f, 9:0x41/0x31/0x21/0x01 S); game.yaml exits are two-way by test | `late-mechanics.md` §3.1; `d9.md` §3.3; `d8.md` §5.2 | `exits_are_two_way_with_one_convention`; `shutter: true` is display only |
| Waypoints: keys of 6:0x58, 6:0x2d, 8:0x4c's corner, 9:0x61; hearts of 6:0x1c; the silver arrow 9:0x4f; the ring 9:0x00; the candle 7:0x4a (optional); 12 containers asserted before 0x21 | `d6.md` §1.5, §8; `d9.md` §10; `d7.md` §2; `late-mechanics.md` §6 | `WAYPOINTS`, cellar waypoints |
| Clear-then-push steps with trigger-5 stairs as the target (9:0x05, 0x10, 0x30; 6:0x3a); 6:0x38 opens both shutters (game.yaml names one) | `d9.md` §3.4; `d6.md` §3 | `push_block`, `push_block_opens`, `push_requires_combat` |
| Wrong-push recovery: leave and re-enter | `late-mechanics.md` §3.5 | `retreat`, the bounce limit |
| Retreat exits for fighting 8:0x5e and 0x3e in parts | `d8.md` §4.3, §11.4 | `retreat`; `MAX_RETREATS` 3, `MAX_BOUNCES` 5 (`chain.rs`) |
| FIGHT targets filtered by `targetable_now` | `late-monsters.md` §5.7 | `objectives.rs:1233-1248` |
| The end: `exit: end` on `GameMode` `$13` (`Route::load` demands a last step that exits `triforce`); the Triforce of Power as 9:0x42's objective after Ganon (the shutters open on the take); Zelda's trigger X `$70`-`$80`, Y exactly `$95`; 9:0x66's man opens with 8 pieces | `late-mechanics.md` §0.5, §3.18; `d9.md` §2, §7; `late-monsters.md` §10.6 | `chain.rs:161-165`; `terminal`; `final_triforce` |
| 0x22's entrance accepts X `$70`/`$78`/`$80` (game.yaml has one tile) | `d6.md` §1.6 | asm:Warp_CheckUnderLink |
| 0x61's Lost Woods `maze:` block checked under the engine (the plain way of the D7 → sword leg) | `late-mechanics.md` §2.13 | `MAZES`, `exit: maze` |

### (d) Rewards and the critic (no inputs change)

| Item | Needed by | Nearest code |
|---|---|---|
| Ganon's kill is invisible: the killing arrow changes no HP, the 4th sword hit is reset within the step, `room_state` keeps him living; success from the Triforce of Power and the N shutter; reward on `$486` decrements and the arrow hit | `late-monsters.md` §5.2; `d9.md` §10 | `is_living_monster` (`room_state.rs:256-259`) |
| Manhandla's base credited as a boss hit and `killable`: make it a hazard | `late-monsters.md` §5.3 | Gleeok's base (`game.rs:372-376`) |
| Boss list: `PATRA2` (`$48`) is not a boss (0x16's body pays 0.25, 0x61's 0.75); small Digdoggers are | `late-monsters.md` §5.4 | `enums.rs:333-352` |
| Dead dummies `$5D` (Patra children, Lanmola tails, guard fires) and the stall penalty keyed on room kills ramping mid-fight | `d9.md` §10; `late-monsters.md` §5.6 | Vire's red Keese precedent |
| `penalty-lost-health` caps at 1.0: a 4-heart Red shot costs as much as a 2-heart hit; scale with hearts at stake | `late-monsters.md` §5.9 | `critic.rs:385-391` |
| Wallmaster wall shaping punishes the lure 7:0x0d needs: off there | `late-monsters.md` §3.3, §5.11 | `critic.rs:396-430` |
| Rooms where waiting is right (Gohma ~400-500 frames, Patra's 8 children, Ganon's ~270-frame entry, traps, the Grumble's text, lures, a Red's 64 hidden frames, the tree's ~94 frames) against room-stalling (150), `no_damage_steps` (600), `no_progress_steps` | `late-monsters.md` §5.10; `d9.md` §9; `d8.md` §10 | `end_conditions.rs` |
| PBRS approach walks into Wizzrobe lines, blue Lynels, Ganon's middle; rewards for leaving a Red's line, killing Reds first | `d6.md` §10; `late-monsters.md` §5.8 | architecture.md "PBRS" |
| Hearts at exit; heart drops picked up in fight rooms; damage in crossing rooms (tried and dropped in round 6 for D1-D3; here every heart counts) | `d8.md` §10; `d6.md` §10 | `HeartTracker`; backlog "leave rooms with more hearts" |
| Giving food pays nothing (`reward-gained-food` exists) | `late-monsters.md` §10 | `critic.rs` |

### (e) Start states, chains and tooling (no inputs change)

| Item | Needed by | Nearest code |
|---|---|---|
| Fresh start states resurrect history-dependent rooms (`make_room_fresh` clears kill bits and `LevelKillCounts`: even a non-recurring Patra comes back): 7:0x1a back from its cellar, 8:0x5e in parts, 9:0x20, 0x61, 0x07, 0x10, 0x55, 0x05 | `late-mechanics.md` §0.8, §3.9 | harvested chain states (round 24 §12); `ram_randomization.room_kills` |
| Start states at the chain's real health (D6, D9) and staged kills (7:0x0d, 7:0x1b, Wizzrobe rooms); not for Digdogger rooms (one kill saves the room) | `d6.md` §10; `d7.md` §9.5; `late-monsters.md` §10 | `config.rs:453-516` |
| State names: a whirlwind arrival needs an entry letter (`STATE_NAME` accepts `n|s|e|w|c|cave`); `_v2` for later visits | `d7.md` §9.5; `d8.md` §11.6 | `triforce/entry_states.py` |
| Audit: HEARTS (12/13/14/15 along the way), SHORT (7:0x28's `Unknown_54` is not in `NOT_ROOM_MONSTERS`), BOSS-ENTRY | `d7.md` §9.5 | `entry_states.py:694-708` |
| Chain: the whirlwind arrival as the route's entry; a Wallmaster drag to 7:0x79 mid-dungeon; death (CONTINUE or failure); counted shop rupee cheats; new segments (post-D5 `0_0bc_0`, `7_79s_0`, ... level 9 split at 0x76, the ring, 0x61's second visit, 0x42) | `d7.md` §9.5; `d9.md` §9; `late-mechanics.md` §3.17, §8 | `chain.rs`; `eval_batch.py`, `r56_chains.sh` |
| Executor waits: the whirlwind ride, the Triforce of Power fanfare, Ganon's entry, the shop's 120-160-frame count, mode `$13` | `late-mechanics.md` §8 | `menu_frames` |
| Hearts-aware chain reports (hearts at Gleeok's door, as round 24 for D4) | `d8.md` §11.7 | chain `RoomRecord` |
| The planner as teacher in 0x38/0x3a, D9's Wizzrobe rooms, Gleeok (the owner's decision 1) | `d6.md` §10; `late-monsters.md` §9; `d8.md` §10 | `plan:K:S` |
| Tests that pin today's end (last step 5:0x14; triforce list; warp rooms) | `d7.md` §9.5 | `game_map.rs:1528-1545` |

### Built, though `first-quest.md` "What is not built" still lists it

| `first-quest.md` lists | Built | Where it says so |
|---|---|---|
| "Actions: the recorder, food, the candle ... each new action kind makes every trained model unusable" | RECORDER, FOOD, CANDLE kinds (round 24, 1.4.2.0; models widened by `convert-model`); only their masks and steps are missing | `late-monsters.md` §2.3; `late-mechanics.md` §8; `d8.md` §8 #5; `d9.md` §3.8 |
| "The overworld ladder clip ... `zelda::clips` offers ladder clips in dungeons only" | 0x5f's overworld ladder clip (round 24) | `d7.md` §9.3; `late-mechanics.md` §0.9 |
| "The objective has no way out of a maze by its sequence yet" | `exit: maze`, `MAZES` (0x61 included), `MazeLoop`, the Lost Hills stretch | `late-mechanics.md` §2.13; `d7.md` §9.3 |
| "the take-any caves' right ware" | `waypoint: heart` with `missed` on the potion (0x2f) | `late-mechanics.md` §2.2; `d9.md` §10 |
| "a tunnel (a cellar with two exits)" | 5:0x07, left by the step's own ladder | `d7.md` §9.3; `d8.md` §5.3; `d9.md` §3.2 |
| "a room left by the door it was entered by" | 1:0x72 E → E; caves S → S | `d7.md` §9.3; `late-mechanics.md` §0.9 (`d8.md` §11.4 still counts it as missing for 8:0x5e's retreat) |
| Families "Gibdo, Pols Voice, Digdogger, Blue Darknut" | gibdo, pols-voice, boss-digdogger, darknut (Blue Darknuts joined in round 24) | `late-monsters.md` §7; `d8.md` §4.4 |

## Softlocks and irreversible losses

| Where | Wrong action | Consequence | Guard or detection | Sources |
|---|---|---|---|---|
| Any overworld room after 0x0b | A RECORDER play outside the plan, or facing up/right | `$523` steps: every later whirlwind lands elsewhere | RECORDER masked on the overworld but at planned steps; plays computed from `$523` | `late-mechanics.md` §2.9, §6; `d8.md` §2.2 |
| 0x0b, 0x22, 0x6d | A hit before the pickup (Link invincible), Link off the row, the clock held, the second play too late | Whirlwind missed, none again until re-entry, the index already stepped | Detect the whirlwind gone with Link in the room; re-plan from `$523` (0x22: one down, one up nets 0) | `d7.md` §1; `d8.md` §2.2; `d9.md` §1.1 |
| 0x44's shop | Touch the shield (130) or the bombs (20) | One ware a visit; the shield leaves no arrows → Gohma impossible → Dungeon 6 impossible | Mask the wrong wares' squares, or `missed`; detect `InvMagicShield`, a bomb count up | `late-mechanics.md` §2.1, §6; `d7.md` §1 |
| 0x34's shop | Buy the key (80; affordable on the leg-state rupees) | No food: the Grumble blocks Dungeon 7 | Mask the wrong wares; keep 0x44 before 0x34 | `d7.md` §1 |
| 0x34, out of the cave | Down from the exit spot | Wakes the Armos again (1 a touch; inferred) | Walk sideways first | `late-mechanics.md` §2.5 |
| 7:0x39 | RECORDER | The big Digdogger splits into three fast 2-heart monsters in a crossing room | `recorder_wanted` per step | `d7.md` §2; `late-monsters.md` §3.4 |
| 7:0x38, 7:0x18 | Go on | Shutters shut behind: no way back | none needed on the route | `d7.md` §2 |
| 7:0x28 | Boomerang while the food lies down; hit the Grumble | Food removed (not consumed; place again); fireballs until Link leaves | Mask A on persons | `d7.md` §2; `late-mechanics.md` §3.11 |
| 7:0x0c | Bomb on a Dodongo's row in its mouth box | The bomb is eaten: a wall bomb lost | Bomb off the Dodongos' rows | `d7.md` §2 |
| 7:0x29 | Arrive without a bomb | Dead end (the E wall and the tunnel back are its only exits); cheat needed | `bomb_reserve` 2 from 0x1b | `d7.md` §2 |
| 7:0x0d | Stand at the wall; Bubble knockback along it | Capture: 7:0x79, history cleared, 13 room entries back through rooms full again; `wallmastered` in training | Boomerang stun; a chain rule for the drag | `d7.md` §2; `late-monsters.md` §3.3 |
| 7:0x1a, ring rooms | Push the W block E | Stairs closed until re-entry | Push direction mask; re-entry rule | `d7.md` §2; `late-mechanics.md` §3.5 |
| Trigger-5 rooms (7:0x0d, 9:0x05, 0x10, 0x30, 6:0x3a; 6:0x38 trigger 4) | Block pushed into a lane | Stairs walled off | Per-room push directions | `late-mechanics.md` §6 |
| Graveyard 0x40, 0x41, 0x31, 0x21 | Press into a gravestone | An immortal flying Ghini (1 a touch); unkillable once the room's Ghini is dead | Walk the free columns; mask or penalty but the secret stone | `d6.md` §1.3; `late-mechanics.md` §2.4 |
| 0x21's cave | Fewer than 12 containers; not at X `$78` | Nothing taken (retry possible) | Assert 12 before the step | `d6.md` §1.5 |
| 6:0x58, 6:0x2d, 8:0x4c, 9:0x61 | Leave without the key | Level 9 one key short | Key waypoints | `d6.md` §8; `d8.md` §5.5 |
| 6:0x29 N, 7:0x1b N, 8:0x5e E, 8:0x4e W, 9:0x63 N/S | Touch an off-route key door | A ledger key spent (8:0x5e E's keyless pass takes a key too when Link has one; with none it opens the door for good and he is not shut into 0x5f: walked, `d8-walks.md` §2.2); 9:0x63 W no longer opens | Key-door masks | `d6.md` §8; `d8.md` §5.4; `d9.md` §3.5 |
| 6:0x28 N, 6:0x2c S | Walk through the open doorway | Shut in with Gleeok (3 heads) or 5 Wizzrobes; in a chain the run ends | Doorway masks | `d6.md` §5.7, §8 |
| 6:0x48 | Stop on the door row or the top row | Caught between traps: 10-12 hearts in 600 frames | trap-room family | `d6.md` §5.3 |
| Before 6:0x28 | A bomb spent fighting | No bomb for the wall: Gleeok detour or a cheat; Dungeon 8 cheats one more | `bomb_reserve` 2 to 0x28, 1 after | `d6.md` §8, §11 |
| 6:0x1c | Fight with 0 rupees | No arrow: Gohma unkillable until rupees are found | ARROW mask at 0 rupees | `d6.md` §6 |
| 6:0x1c, 8:0x3c, 7:0x2a | Leave without the container | The container lost unless the level is walked again | Heart waypoints | `d6.md` §8 |
| 0x4d | CANDLE at the shop tree | A wrong cave exit | CANDLE mask | `d8.md` §2.3 |
| 8:0x7e | Leave S | Level reload: every partial kill lost | none | `d8.md` §3.2 |
| 8:0x6e | Bomb at the Manhandla, or miss the wall | The only bomb gone; each dead hand speeds the rest. Walked (`d8-walks.md` §4): a `required` family lays exactly the wall's bomb (60 of 60); boss-manhandla (`free`) uses 4.3 a run, 3.15 of them cheated in by the room's `requires_bombs` | Per-step bomb rule, or the room in a `required` family | `d8.md` §5.1 |
| 8:0x3f | Enter | One way; a W-facing Darknut in the one-wide corridor cannot be hit (not isolated in the walk) | none | `d8.md` §5.2 |
| 8:0x3c | Retreat S | Gleeok back with 4 heads and full HP (walked, `d8-walks.md` §6.3) | none | `d8.md` §6 |
| 0x47's take-any | The potion | The last container gone (14 hearts) | `missed` exists | `late-mechanics.md` §6; `d9.md` §1.2 |
| Overworld 0x07 | Bomb its rock wall and walk in | Door repair: −20 rupees with no test → too few for Ganon | Mask the cave entry | `d9.md` §1.2 |
| 9:0x66 | Arrive with 7 pieces | The man stays, the shutters stay shut | 8 pieces precondition | `d9.md` §3.1 |
| 9:0x65 | Bomb without the ladder over the water row | The bomb lands `$18` px off the hotspot | Step recipe | `d9.md` §3.1 |
| North of 9:0x61 | Leave N with fewer than 2 bombs, or spend one as a weapon | Stuck behind one-way shutters (0x31 W, 0x04 W); a recovery loop of 18 entries and 2 fights exists only with the spare key | `bomb_reserve` 2 from 0x16; BOMB only at the step's wall | `d9.md` §3.3, §3.6 |
| 9:0x41/0x31/0x21/0x01 S | Go north without the silver arrow, 2 bombs, 1 rupee | No way back south | Preconditions at 0x61 N | `d9.md` §3.3 |
| 9:0x20 | Leave it full on the first visit | Full again on the second: the fight again to move the block | Clear it on visit 1 | `d9.md` §3.1 |
| 9:0x42 | Arrive without the bow, the silver arrow or a rupee; miss with one rupee | Sealed in; only death | Preconditions; a rupee source or a counted cheat | `d9.md` §6.5; `late-monsters.md` §3.17 |
| 9:0x42 | Arrow in the blue phase; the brown window runs out | A rupee wasted; the sword count restarts | Arrow target mask; phase observation | `d9.md` §6.4 |
| After 0x44 | ARROW at ordinary monsters | Each costs a rupee; at the low end Ganon's room becomes a dead end long after the mistake | `arrows: false` but Gohma/Ganon; `rupee_reserve` | `late-mechanics.md` §6; `late-monsters.md` §5.1 |
| The ending (mode `$13`) | Start at submode 4 | Saves and starts the second quest | Never press Start in replays | `d9.md` §7 |
| Anywhere | Death | Chain run over; the game's CONTINUE (3 hearts, level restart, history cleared, items kept) is not modelled | The owner's question | `late-mechanics.md` §3.17; `d9.md` §2 |
| Any harm (Bubble, whirlwind pickup included) | — | Zeroes the 10th/16th damage-free kill counters: a pending forced bomb or fairy lost | none | `late-mechanics.md` §2.9, §3.15 |

## Families

Families are by enemy (the owner). Existing (`specialists/`): ow-start, ow-octorok, ow-zora, ow-moblin, ow-leever, ow-tektite,
ow-lynel, ow-boulder, ow-peahat, stalfos-keese-gel, goriya, trap-room, equipment-room, wallmaster, boss-aquamentus,
d2-rope, moldorm, boss-dodongo, zol, darknut, boss-manhandla, vire, like-like, boss-gleeok, gibdo, pols-voice,
dodongo-crossing, boss-digdogger.

| Stretch | Room → family | Sources |
|---|---|---|
| D5 → D7 leg | 0x0b (whirlwind) ow-leever; 0x45 ow-peahat; 0x55, 0x54 (three visits), 0x65, 0x44 ow-zora; 0x64 ow-octorok; 0x34 ow-leever (+ the armos step); shop caves 0x44, 0x34 ow-start (as 0x2f/0x0f) or each room's family; 0x53, 0x52, 0x42 (pond) ow-moblin | `d7.md` §0, §9.4 |
| Dungeon 7 | 0x69, 0x59, 0x49, 0x38, 0x18, 0x19, 0x1a (×2), 0x1b, 0x29 goriya, extended to Blue Goriyas (or a new blue-goriya); 0x79, 0x2b stalfos-keese-gel (layouts of 1:0x73 and 1:0x36) or the level's main family; 0x39 a new digdogger-crossing, or another family with RECORDER masked; 0x28 (Grumble, no monster) any family with `food: true`; 0x1c boss-digdogger (three small ones); 0x0c dodongo-crossing (with BOMB at the wall); 0x0d wallmaster, or its own family (the opposite skill to 1:0x45's); 0x4a, 0x7b equipment-room; 0x2a boss-aquamentus (entered from W, heart at X `$80`, against 1:0x35's S and `$C0`) | `d7.md` §0, §9.4; `late-monsters.md` §7 |
| D7 → sword → D6 leg | 0x42 (cave → S), 0x52 (N → W, new pair), 0x51 (wrap E → W) ow-moblin; 0x50 (E → N, new pair), 0x32, 0x22 ow-lynel; **0x40, 0x41, 0x31 (×2), 0x21 ow-ghini (new)**; 0x21's cave a cave step; plain way 0x61's Lost Woods ow-moblin (maze) | `d6.md` §1.1; `late-mechanics.md` §4 |
| Dungeon 6 | **0x78, 0x28, 0x29, 0x2c (crossings), 0x38, 0x3a (fights) wizzrobe (new)**; 0x68 zol; 0x58 stalfos-keese-gel (with statues, new to it); 0x48 trap-room; 0x39, 0x2d vire (magic sword: no split); 0x1d like-like; 0x08 tunnel unassigned (equipment-room by the tunnel precedent, synthesis); **0x1c boss-gohma (new)**; 0x79, 0x0c no monsters | `d6.md` §11; `late-monsters.md` §7 |
| D6 → D8 leg | 0x22 (whirlwind) ow-lynel by its Lynel (`d8.md` calls it a recorder step); 0x3c (whirlwind entry), 0x4c ow-octorok; 0x4d (W → S new), 0x5d (N → S at X `$C0` new), 0x6d (burn) ow-moblin | `d8.md` §2.3 |
| Dungeon 8 | 0x7e, 0x2c trivial; 0x6e a Manhandla crossing with a wall bomb (a new task for boss-manhandla, or a new family); 0x5e, 0x3e darknut (statues, magic sword, 0 bombs); 0x4e, 0x3f a new crossing family (`d8.md`) or darknut (`late-monsters.md`); 0x2f equipment-room; 0x4c pols-voice (with the corner key and the wall); 0x3c boss-gleeok (four heads, from S) | `d8.md` §11.6; `late-monsters.md` §7 |
| D8 → D9 leg | 0x6d ow-moblin (whirlwind); 0x37, 0x38 ow-octorok/ow-zora; 0x48 ow-leever; 0x47 (burn, take-any) steps; 0x28, 0x27 ow-peahat; 0x17 ow-boulder (ladder); 0x07, 0x06, 0x05 ow-lynel (0x05 with BOMB at the rock wall) | `d9.md` §1.2; `late-mechanics.md` §4 |
| Dungeon 9 | **0x65, 0x15, 0x17 (×2), 0x07 (×2), 0x05, 0x20 (×2), 0x10 (×2), 0x31, 0x30, 0x04 wizzrobe**; **0x55 lanmola (new)**; **0x16 (×2), 0x27 (×2), 0x61 (×2), 0x52 patra (new, both kinds)**; 0x14, 0x51 like-like; 0x41 like-like or trap-room; 0x03 zol or like-like; 0x26 (×2), 0x62 stalfos-keese-gel; 0x63 zol; cellars 0x00, 0x4f equipment-room; tunnels 0x60, 0x70, 0x75 (×2), 0x67, 0x77 equipment-room by precedent (`d9.md` maps their Keese to stalfos-keese-gel); 0x66, 0x06 person steps; **0x42 ganon (new)**; 0x32 the end step (a final family with Ganon, `late-mechanics.md` §7) | `d9.md` §10; `late-monsters.md` §7 |

New families: **ow-ghini** (0:0x40, 0x41, 0x31, 0x21), **wizzrobe** (6:0x78, 0x28, 0x29, 0x2c, 0x38, 0x3a; 9:0x65, 0x15,
0x17, 0x07, 0x05, 0x20, 0x10, 0x31, 0x30, 0x04: the biggest family of the game, crossing and fighting, `d6.md` §10 #3),
**boss-gohma** (6:0x1c), **lanmola** (9:0x55), **patra** (9:0x16, 0x27, 0x61, 0x52), **ganon** (9:0x42, perhaps with
0x32). Possible, by the owner's answer: blue-goriya (D7's nine Goriya rooms), digdogger-crossing (7:0x39), a D7 Wallmaster
family (7:0x0d), a Darknut-Gibdo crossing family (8:0x4e, 0x3f), a Manhandla crossing (8:0x6e). Existing families that
meet a new skill: goriya (blue: throw at once, 1 heart), wallmaster (lure and kill instead of rush), boss-digdogger
(three), darknut (statues, magic sword, no bombs), boss-gleeok (four heads, three flying), vire and like-like (magic
sword), stalfos-keese-gel (statues), ow-lynel (blue Lynels, magic sword), boss-aquamentus (another entry and heart
spot) (`late-monsters.md` §7). ARROW must be off in all of them but boss-gohma and ganon.

## Gap list: what must be walked in the running game before it is trusted

Ordered by the round that needs it first. A walk uses the running game (throwaway scripts on `triforce/probe.py`, or
chain states), with live monsters and real hearts unless said.

**Round 26 (Dungeon 7 and the leg from Dungeon 5)**

1. Rupees and bombs at 0:0x0b on real chain arrivals: settles the 2-14 / 17-28 rupee readings and the 0 / ~3 bombs
   (`d7.md` §4; `late-mechanics.md` §1; `late-monsters.md` §11.8).
2. 0x0b's whirlwind with the leg's real state: two plays facing down → 0x45, one → 0x0b; `$523` before and after; the
   window of the second play; whether the executor or the classifier treats the whirlwind `$2E` as an enemy; the
   chain's view of the arrival (`d7.md` §7 #1; `late-mechanics.md` §9 #4).
3. 0x44 and 0x34: whether a purchase sets the room's item flag; wrong-ware touches from the walk-in path with the leg's
   rupees; the exit spot after the armos shop; whether the Armos comes back (`d7.md` §7 #2; `late-mechanics.md` §9
   #2-#3).
4. 0x42: the recorder from one step in; the stairs' position for the objective; leaving mid-cycle (`d7.md` §7 #3).
5. 7:0x1c: kill one small Digdogger, leave W, re-enter: empty, shutter open? (`d7.md` §7 #4)
6. 7:0x0c: a wall bomb with the Dodongos walking: how often eaten (`d7.md` §7 #5).
7. 7:0x0d: a full played clear with the white sword (none exists); pushed E, can Link still reach the stairs; a capture
   inside a chain run (what the runner records, where the run goes) (`d7.md` §7 #6, #11; `late-monsters.md` §11.6;
   `late-mechanics.md` §9 #10).
8. 7:0x49 and 0x18: the ladder refused with 8 monsters and boomerangs out (`d7.md` §7 #7).
9. 7:0x1a's block clip by the agent's CLIP action (5:0x64's 27 points, same layout) (`d7.md` §7 #8).
10. 7:0x28: FOOD by the agent's action from the doorway; the boomerang interplay (`d7.md` §7 #9).
11. 7:0x2a from W with the boss-aquamentus model (`d7.md` §7 #10).
12. Every forced fight of level 7 for real (0x49, 0x1b, 0x1c, 0x0d, 0x2a; 0x1a plain): hearts lost, to set start-health
    ranges (`late-monsters.md` §11.7).

**Round 27 (the magic sword and Dungeon 6)**

*Walked Nov 16: `d6-walks.md` answers items 1 and 3-7 (the stairs are not kept and no flying Ghini appears when Link
pushes from below and stops on the stairs; killing the Ghini first was not tried; slow tiles 0.77 px a frame; 0x22 at
X `$70`/`$78`/`$80`; every fight and crossing; the touch during immunity hurts; the safe column holds; shots pass a
held Link; 0x2d first at (`$78`,`$4D`); 0x18 and 0x3c shut). Item 2 only by buttons and one model state.*

1. 0x21: push from above and from below; does a flying Ghini fade in on the stairs; does killing the room's Ghini first
   make it safe; does the stairs square come back after the cave (asm:Secret_RestoreStairsOW); the cave's exit spot
   (`$50`,`$6D`) against the stairs (`$90`,`$90`); the ware at X `$78` (`d6.md` §12 #6; `late-mechanics.md` §9 #1;
   `late-monsters.md` §11.3).
2. 0x61's Lost Woods under the engine's maze (the plain way; walked by buttons Nov 14 only) (`late-mechanics.md` §9 #6).
3. Slow tiles: speed in 0x50's N corridor, 0x32's N steps, 0x22 and 0x40 (`d6.md` §12 #7); 0x22's entrance at X `$78`
   and `$80` (`d6.md` §12 #8).
4. Every D6 fight live with real hearts: 0x58, 0x38, 0x39, 0x3a, 0x2d, Gohma with misses and fireballs (`d6.md` §12 #1).
5. The Wizzrobe crossings from the real entry frame: 0x78, 0x28 (with the bomb wait), 0x29, 0x2c (traced only with its
   Wizzrobes removed) (`d6.md` §12 #2); whether monsters run during a shutter walk-in (`d6.md` §12 #4).
6. A Wizzrobe's touch during its immunity, against the observation's `hurts_on_touch` (`d6.md` §12 #5;
   `late-monsters.md` §11.4); the vertical-shot safe column (`d6.md` §12 #3); whether shots pass a held Link
   (`d6.md` §12 #11).
7. 0x2d's first movable spot and its Vires' side of the water; 0x18 and 0x3c's shutters (`d6.md` §12 #9-#10).

**Round 28 (Dungeon 8 and the leg from Dungeon 6)**

*Walked Nov 17: `d8-walks.md` answers items 1-6 (two plays from the chain-made counter land in 0x3c 30 of 30, the
second play counts on frames 152-216, a miss is recovered by one play down and one up after a re-entry; the tree
burns from X `$B8`-`$CF` facing W and from 0x5d's left opening too; the wall bomb kills a hand in 3 of 60 and never
armed the bomb drop; both Darknut rooms are won 30 of 30 by a script and 1 of 80 by today's model, banked kills
outlast the history; the keyless pass takes a key or opens the door, it never strands; with arrows 0x4c and its key
cost 0.4 hearts; Gleeok resets on a retreat, rows `$9D` and lower are out of the heads' reach, a script kills it 30
of 30 from 13 hearts and 19 of 30 from 4). Not isolated: 0x3f's corridor Darknut.*

1. 0x22 → 0x3c from a chain-made state (index from 0x0b's plays), the Lynel live; the second play's window; recovery
   from a missed pickup (`d8.md` §12 #1; `late-mechanics.md` §9 #4).
2. 0x6d's burn with the CANDLE action from the draft's stand tile and from 0x5d's left opening (`d8.md` §12 #2;
   `late-mechanics.md` §9 #5).
3. 8:0x6e with a live Manhandla: the wall bomb near and far from the base; the forced 4-bomb drop on 0x5e's first kill;
   the base's credited hits (`d8.md` §12 #3; `late-monsters.md` §11.5).
4. 8:0x5e and 0x3e fought with statues at 13 hearts, magic sword, 0 bombs; fighting in parts past the room history;
   spawn positions on re-entry (`d8.md` §12 #4).
5. 0x5e's keyless pass at the E door; 0x3f's corridor Darknut; 0x4c's ring exit and corner key with live Pols Voices
   (`d8.md` §12 #5-#7).
6. Gleeok with four heads at 3-6 hearts; a retreat S mid-fight (does it reset?); the rows its heads cannot reach
   (`d8.md` §12 #8); first movable spots per door (`d8.md` §12 #9).

**Round 29 (Dungeon 9, Ganon, the end)**

*Walked Nov 17, the route part: `d9-walks.md` answers item 3 (the Like-Like corridors, the bomb waits, 0x30's lure
with the room emptied, 0x04's crossing), item 6 (the whirlwind with the real index and a miss, 0x47, 0x17, 0x05),
the arrival spots of item 5 (open or key door `$10` px in, bombed wall or shutter `$20`) and item 7 against the
scaffold's episodes (the env refills 0x07, 0x10, 0x20 and puts 0x61's Patra back; the game keeps them empty). Items
1, 2, 4 and the rest of 5 are the fights' walk. 0x03's fight (Zols, Like-Likes, the push) was not played here.*

1. Every level 9 fight live: 0x55, 0x07, 0x05, 0x20, 0x10, 0x30, 0x03, the three Patras (`d9.md` §11 #1).
2. 0x55's re-entry after the worms die, in the history and after CONTINUE (`late-monsters.md` §11.2).
3. Like-Like corridors 0x14, 0x51, 0x41 (`d9.md` §11 #2); bomb waits with live monsters 0x65, 0x27, 0x17, 0x31, 0x04 and
   overworld 0x05 (`d9.md` §11 #3); 0x30's Trap lure onto the stairs, 0x04's crossing (`d9.md` §11 #4).
4. Ganon under the engine, with the critic and observation recorded: first movement, jump side by hit parity, the
   strike spots for four facings, the brown window, a silver arrow off his row; what the Triforce of Power pays; mode
   `$13` (`d9.md` §11 #5; `late-monsters.md` §11.1; `late-mechanics.md` §9 #7).
5. The Blue Wizzrobe's column-shot immunity (`d9.md` §11 #6); the block clips of 0x55, 0x07, 0x61, 0x20, 0x03 with
   monsters alive (`d9.md` §11 #7); arrival spots per door kind and Zelda's Y `$95` after a fire's knockback (`d9.md`
   §11 #8).
6. The leg: 0x6d's whirlwind with the real index history and the missed-pickup recovery; 0x47's tree and its right
   ware; 0x17's ladder among Boulders; 0x05's bomb spot (`d9.md` §11 #9).
7. Return-visit rooms from built states against chain arrivals (9:0x20, 0x61, 0x07, 0x10) (`late-mechanics.md` §9 #8).

## Where the documents disagree

| Point | One reading | The other | What settles it |
|---|---|---|---|
| Rupees after both shops | 2-14 (`first-quest.md`; `d6.md` §0; `d8.md` §1; `d9.md` §3.7; `late-mechanics.md` §1: the draft's 42-54 + 100) | 17-28 (`d7.md` §3 #9, §4: `d5-leg.md`'s measured 157-168) | The count at 0:0x0b on chain arrivals; the leg states are the measured number |
| 0x34's key (80) | Unaffordable after 0x44 (`late-mechanics.md` §2.1, §6) | Affordable in 2 of 3 chains (`d7.md` §1) | Follows from the rupee reading |
| Bombs out of Dungeon 5 | ~3 (`d7.md` §4, round 24's Digdogger arrivals) | 0 in most chains (`late-mechanics.md` §1) | The bomb count at 5:0x14 in chain jsonl |
| Banked kills | "Kept while the room is among the last six added to the history" (`first-quest.md`) | Partial kills kept until the level reloads, even outside the history; only an all-dead room refills after six additions (code, asm:RoomHistory_SaveKillCountUW, asm:RoomHistory_ModifyObjCountUW; level-8.md trace) (`d8.md` §4.3, §8 #1) | Code and trace agree; fix `first-quest.md`; `d8.md` §12 #4 confirms past the history |
| Lanmola re-entry and death | level-9.md: the worms come back on every later entry and after a death; `d9.md` §2: CONTINUE clears the history, Lanmolas back | Code reading: banked as 2 in `LevelKillCounts`, none while in the history; CONTINUE keeps `LevelKillCounts` (asm:Mode8Continue_Init; inferred) (`late-monsters.md` §3.15, §8; `d8.md` §4.3) | Walk 0x55's re-entry and after CONTINUE |
| The Triforce of Power's refill | enemies.md "Bubble": refills started by its fanfare | level-9.md and code: none (asm:Item_UpdatePowerTriforceFanfare@FillHearts unreachable) (`d9.md` §3.8; `late-mechanics.md` §0.5) | Code; fix enemies.md |
| The magic sword's threshold | `objectives.rs:60-64` asks 13 | The game gives at 12 (asm:Cave_State2TalkOrShopOrDoorCharge) (`late-mechanics.md` §0.1) | Code: fix the engine |
| Flashing Wizzrobes | The observation marks them harmless | They hurt during their immunity (enemies.md, code) (`d6.md` §9 #6; `late-monsters.md` §2.2 G5) | `d6.md` §12 #5; fix the flag |
| 0x21's flying Ghinis | `first-quest.md` (Nov 13) walked the push with no mention | overworld.md (Nov 9): Ghinis kept knocking Link off; code reading: the push itself releases one on the stairs (inferred) (`d6.md` §9 #8; `late-monsters.md` §8) | Walk 0x21 |
| 0x21 after the cave | game.yaml and the draft: stairs where the stone stood | asm:Secret_RestoreStairsOW shows the gravestone again on re-layout (inferred) (`late-mechanics.md` §2.4) | Walk 0x21 |
| 0x22's entrance | game.yaml: one tile `[14, 8]` | Code: X `$70`/`$78`/`$80` (asm:Warp_CheckUnderLink) (`d6.md` §1.6, §9 #3) | Walk |
| Rupees for Gohma | level-6.md: "a few dozen rupees for misses" | The draft: 2-14, one shot each for Gohma and Ganon (`d6.md` §9 #1) | The owner (rupee question) and the rupee reading |
| D6's forced fights | `first-quest.md`: two Wizzrobe rooms to clear, the four Wizzrobe crossings "without a fight" | 0x58, 0x39 and 0x2d are forced too; the crossings' Reds appear on the path (0x2c was traced with its Wizzrobes removed) (`d6.md` §2.3, §9 #4, #7) | `d6.md` §12 #2 |
| 6:0x38's push | level-6.md: both shutters open (trace) | game.yaml `push_block_opens: N` | Incomplete, not wrong: add the other side |
| 7:0x69's bombs | level-7.md: a route source (+4) | The owner (Nov 14): 7:0x1b only; the draft: "need not die" (`d7.md` §3 #2) | The owner |
| 7:0x2a's heart | 1:0x35 (same layout): heart at X `$C0`, entered from S | 7:0x2a: heart at X `$80`, entered from W (`d7.md` §2, §3 #8) | Both right; boss-aquamentus has seen only one |
| 0x5d's openings | game.yaml: only the right one counts | Layout and level-8.md: the tree burns from the W part too (`d8.md` §8 #4) | Walked: both (`d8-walks.md` §1.3); game.yaml's note is wrong, its `span` a fair choice |
| 0x6d's tree | level-8.md: "Not walked: the entrance tree" | game.yaml: walked Nov 14 (`d8.md` §8 #2) | Walked again Nov 17; level-8.md corrected |
| 8:0x4e/0x3f's family | A new crossing family (`d8.md` §11.6) | darknut (`late-monsters.md` §7) | The owner (families) |
| Sword grade in the observation | None found (`d8.md` §4.4, inferred) | `sword_0`-`sword_3` in `link` (synthesis, `observation.rs`) | The code |
| Held by a Like-Like | Missing (`d9.md` §8) | `paralyzed` observed (`d6.md` §4.4; `observation.rs`) | The code |
| 9:0x32's fires | level-9.md step 32: two | `Zelda_Init` makes four (`d9.md` §3.8) | Both right; two are off Link's path (inferred) |
| Level 9's tunnels | level-9.md: four tunnel cellars | The draft: six passages (`d9.md` §3.2) | Both right: five tunnels, 0x75 twice |
| Level 9's walls | level-9.md: 6 (9 with the ring) | The draft: eight walls inside (`d9.md` §3.6) | Same count, the draft leaves out 0x05 |
| Level 9's forced fights | The draft's list | Five are layout `$1A` rooms a block clip may skip (`d9.md` §3.4) | The draft's list is the clips-off list; walk the clips |
| Ganon after death | level-9.md: ashes, the item, the shutters on the take | The project reads no hit, no kill, a living monster (inferred) (`late-monsters.md` §5.2, §8) | Walk with the critic attached |
| What is built | `first-quest.md` "What is not built" | Built in round 24 (table above) | The code; update `first-quest.md` |
| The chase target | AGENTS.md: shadow Link copied from the current position | enemies.md: the mirrored point for up to 70 frames; both agree it is not always Link (`late-monsters.md` §8) | Read asm:Mode5Play_UpdateWorld against enemies.md |
| Room left by its entry door | Built (`d7.md` §9.3; `late-mechanics.md` §0.9) | Missing for 8:0x5e's retreat (`d8.md` §11.4) | `d8.md` needs a planned retreat step with chain caps, which is more than the exit kind |

## Open questions for the owner

Answered by the Nov 16 decisions and removed: where the D7 round ends (decisions 2-3 and the version rule: at the D7
triforce, the sword in round 27; `d7.md` §8 #9); whether shop rupees may be cheated (decision 5; `d7.md` §8 #6); level
9's key from level 8, 0x4c's corner or 0x5d's (decision 5, no route change: the draft's corner key; `d8.md` §13 #2);
block clips in level 9's ring rooms (decision 4 and the standing rule: the clip way is primary, the fights are the plain
way; `d9.md` §12 #2); the planner as a teacher (decision 1; `d6.md` §10, `d8.md` §10); breaking models for new features
(decision 6; `d7.md` §10 #5).

1. **Arrows outside Gohma and Ganon.** (a) ARROW off in every family but boss-gohma and ganon, with a route
   `rupee_reserve` (recommended in `late-monsters.md` §12 #1; costs a yaml line a family and a route field). (b) Allowed
   in Pols Voice rooms (8:0x4c: one arrow kills a line) at a rupee each against the budget. (c) Left on: every
   Wizzrobe arrow is spent from Ganon's (`d7.md` §8 #7; `late-mechanics.md` §10 #2).
2. **Rupees short at Gohma or Ganon** (decision 5 covers the two shops only). (a) A counted cheated rupee at a boss
   when Link has 0: one rule, ends no run. (b) A rupee source on the path made required: 9:0x62's 8 Keese (+5) or
   6:0x28's 5 Wizzrobes (+5, a fight instead of a crossing). (c) A miss may end the run (`d6.md` §13 #1; `d9.md` §12
   #3; `late-mechanics.md` §10 #2).
3. **The whirlwind, the overworld secrets and the shops: scripted or learned.** (a) Executor macros computed from RAM
   (plays and facing from `$523`), the model waits: no input change, exact. (b) Learned under tight masks: nothing in a
   one-room episode rewards a count that pays off three legs later, one stray play breaks three legs, and the index
   would have to enter the observation (`d7.md` §8 #5; `d8.md` §13 #5; `late-mechanics.md` §10 #3, §7).
4. **Shops' wrong wares.** Mask them (cleanest) or let a wrong buy end the run as `missed` (`late-mechanics.md` §10
   #4).
5. **Death.** The game's CONTINUE (3 hearts, the level restarts, items kept) as part of a run, at least to measure
   whether the game can be finished, or a death stays a failed run (`late-mechanics.md` §10 #1; `d9.md` §12 #5).
6. **A Wallmaster capture in a chain** (7:0x0d → 7:0x79): end the run, or replay the 13-15 rooms back
   (`late-monsters.md` §12 #4).
7. **Families.** Blue Goriyas in goriya (as Blue Darknuts joined darknut, the owner's own precedent) or a new blue-goriya
   (nine rooms); 7:0x0d in wallmaster or its own; 7:0x1c in boss-digdogger; 7:0x39's crossing as its own family or
   with RECORDER masked; 8:0x4e/0x3f in darknut or a crossing family; one wizzrobe family that crosses and fights, or
   two (`d7.md` §8 #1-#3; `d8.md` §11.6; `d6.md` §10 #3).
8. **Room bombs on the path that the Nov 14 rule did not name**: 7:0x69 (+4 for 5 Blue Goriyas; removes the three
   cheats of a run that enters with none) and 8:0x3f (+4 for six monsters in a one-way room at 2-heart touches;
   removes 8:0x4c's cheat). Decision 5 aims at zero cheats; each costs a fight (`d7.md` §8 #4; `d8.md` §13 #1).
9. **Bombs as Wizzrobe weapons.** `never` outside the walls (both D6 bombs are spoken for) or `free` with the extra
   bombs counted as cheats (a bomb kills a Red in 1, a Blue in 3) (`d6.md` §13 #2).
10. **Wrong exits that trap**: mask the doorways into shut-in rooms (6:0x28 N, 6:0x2c S), the off-route key doors and
    0x07's door-repair cave as a rule ("never enter a shut-in room the route does not use"), or leave them to
    wrong-exit penalties (in a chain one step N in 6:0x28 ends the run) (`d6.md` §13 #3).
11. **7:0x39's big Digdogger**: walk round it (row 1: 1 of 30 touched) or split it (three fast 2-heart monsters in a
    crossing room) (`late-monsters.md` §12 #3).
12. **7:0x1c's leave-and-return shortcut** (if the walk confirms it): 4 hits instead of 12 (`d7.md` §8 #8).
13. **Fighting 8:0x5e and 0x3e in parts** through the entry door: allowed, and may the chain's caps (3 retreats a run,
    5 bounces) rise for these rooms (`d8.md` §13 #3)?
14. **8:0x6e's Manhandla**: cross it as the draft does, or kill it (4 hits; it stays dead; 0x6e becomes a safe retreat
    for 0x5e; +5 rupees) (`d8.md` §13 #4).
15. **Hidden positions in the observation**: Ganon's and the Red Wizzrobes' are in RAM every frame and invisible on the
    screen; keep them (today's default, an advantage a player lacks) or make the model infer them (`d9.md` §12 #1;
    `late-monsters.md` §12 #2).
16. **The end step**: end on the first frame of mode `$13`, or play the ending to the Start prompt in the golden path
    and replays (never pressing Start) (`d9.md` §12 #4).

## A proposed order of work

Dependency order (synthesis). **[P]** items of one tier go in parallel worktrees; the rest wait for what they name.

**Before round 26, once (decisions and the input break).** The owner's answers to questions 1, 3, 5, 6, 7, 15 (they shape
masks, steps and the observation). Design the whole observation change of "Everything the engine lacks" (a) for the four
rounds at once, so that the models break once (decision 6): the `link` and `room_state` widenings convert; the entity
features and the corrected `hurts_on_touch`/`killable` do not, unless `convert-model` and the Rust actor's zero-column
drop are extended to the entity projection (`late-monsters.md` §6). Two ways: extend the converter and keep every model
(the meaning changes still need a retrain of the families they touch), or retrain every family once at the start of
round 26. Features whose RAM was never walked (Wizzrobe phases, Ganon) carry a risk of a second break if a walk proves
them wrong.

**Round 26: Dungeon 7 (1.6)**
1. Walks of round 26's gap list, first (no code).
2. [P] game.yaml: the stretch's `specialist:` tags, `requires_bombs` on D7's five wall rooms, the shops' `cave` exits
   and in-cave rooms, armos and pond targets, level 7's one-way shutters (`d7.md` §10 #3; `late-mechanics.md` §3.1).
3. [P] Route vocabulary: `exit/entry: recorder` with `route_entry` for a jump, `waypoint: arrows|food` by inventory
   with the wrong-ware `missed`, overworld stairs targets, the armos and pond steps, the Grumble step; the magic sword
   threshold fix and the 0x21 cave room can land here (one line and one room) (`d7.md` §10 #4; `late-mechanics.md` §5).
   Split from item 2 or sequence them: both touch `game_map.rs`.
4. [P] Masks: RECORDER per step (never 7:0x39), FOOD at 7:0x28, `arrows: false` and `rupee_reserve` before the first
   arrow is bought at 0x44, the shops' wrong wares, off-route key doors (`d7.md` §10 #5; `late-monsters.md` §6.4).
5. [P] The observation break and the converter (if decided), and the whirlwind macro (if the owner picks it) (`d7.md` §10
   #6).
6. [P] Chain and state tooling that every later round reuses: whirlwind arrivals as entries, state names, the
   Wallmaster drag, harvested states for history-dependent rooms (7:0x1a), new segments.
7. After 2-5: the scouting route `d7` (steps 209-247), yamls, states, curricula per family [P], training per stream [P],
   played states for skipped fights, chains, the merge into `d2first`, the tag (`d7.md` §10 #7-#15).

**Round 27: the magic sword and Dungeon 6 (1.7)** (needs round 26's stairs targets, secret steps, arrow masks, the
observation)
1. Walks of round 27's gap list.
2. [P] The gravestone push step (flying Ghinis expected), the 0x21 cave step, the 12-container assertion, ow-ghini; the
   0x51 wrap and 0x61's Lost Woods plain way.
3. [P] The wizzrobe family on the observation break: staged starts, chain-health starts, the corrected damage penalty,
   the planner as teacher in 0x38/0x3a.
4. [P] boss-gohma: the arrow target mask, eye features, arrows held in its states.
5. [P] D6's data and route: key waypoints (0x58, 0x2d), heart waypoint, push rooms, doorway and key-door masks,
   `bomb_reserve` 2 to 0x28; vire with the magic sword.

**Round 28: Dungeon 8 (1.8)** (needs the whirlwind step from round 26, the secret-step pattern from round 27)
1. Walks of round 28's gap list.
2. [P] The tree burn step and the CANDLE masks (0x6d; 0x47 reuses it in round 29).
3. [P] Per-step bomb rules (0x6e, 0x4c), retreat exits and chain caps (if the owner allows fighting in parts).
4. [P] darknut with statues and the magic sword at 0 bombs; the 0x4e/0x3f crossing; pols-voice with the corner key and
   the wall; the Manhandla crossing and the base as a hazard (critic).
5. [P] boss-gleeok with four heads, trained from low-heart harvested arrivals; hearts-aware chain reports.

**Round 29: Dungeon 9, Ganon, the end (1.9)** (needs everything before; the end step changes `Route::load`)
1. Walks of round 29's gap list.
2. [P] The end: `exit: end`, the Triforce of Power objective, Zelda's trigger, the executor through mode `$13`; 9:0x66's
   8-piece precondition.
3. [P] ganon: phase features (from the break), the kill detector, the arrow target mask, states at the scene's end and
   in the brown phase.
4. [P] patra (both kinds) and lanmola: dead-dummy rewards, `PATRA2` in the boss list, stall rules.
5. [P] Level 9's data: one-way shutters, the ring and silver arrow waypoints, `bomb_reserve` 2 from 0x16, 0:0x05's
   `bomb_secret`, 0x47's in-cave take-any and burned-tree entrance, 0x07's cave masked, history-dependent states,
   chain splits at 0x76, the ring, 0x61's second visit and 0x42.
6. The first complete run, then the one optimisation round over the whole route (decision 2).
