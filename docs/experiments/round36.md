# Round 36

## The corrected route

The owner corrected `d3first` on Nov 21 (decisions 2026-11-21 "The program"; backlog "Track R"): Dungeons 3, 4, 1,
0x2c's heart, 0x2d's 30 rupees, Dungeon 2, 0x0f's 100 rupees, Dungeon 5, the whirlwind to Level 4's screen, the raft
to 0x55, **a wrong warp there** to the meat under the Armos (0x34), the arrows (0x44), Dungeon 7 with the wood sword
at 9 containers, the world wrap at 0x60, the hearts at 0x5f and 0x2f, **a wrong warp at 0x3f** into the graveyard
(0x31), the magic sword at 12, Dungeons 6, 8, 9. The route replaces `d3first` under its name (292 steps; branch
`r36-route`; `docs/routes/d3first.md`, its walks `docs/routes/d3first-walks-c.md`). `d2first` is untouched: its 354
steps compared field by field with `main`'s, and a 100-run chain identical run for run.

### The recorder chain

Every overworld play of the recorder steps the whirlwind's index (`TeleportingLevelIndex`, `$523`;
asm:Whirlwind_StepLevelIndex): +1 facing up or right, -1 facing down or left, then on past every level whose piece
Link lacks. So the plays of the whole route are one chain: Level 5's exit sets what 0x55 starts from, 0x55 sets
0x3f's, 0x3f sets Level 6's exit. `scripts/experiments/r36_recorder_search.py` played every sequence of one to
three plays at each recorder step (each facing; a turn is a one-frame press the frame before the B; the later plays
on the first free frame, the last one's delay swept for its window) from the route's states with the corrected
order's pieces and the index carried, choosing the fewest turns between plays, then the fewest plays, then the
widest window:

| Where | Index in | The plays | Last play's window | Lands |
|---|---|---|---|---|
| 0:0x0b out of Level 5 | L1, pieces 1-5 | 2 facing down, as Link comes out | 63 frames | 0x45, index on L4 |
| 0:0x55, in the water | L4 | N, a one-frame turn W, again | 62 | 0x34 (dock warp) |
| 0:0x3f, in the water | L4, pieces 1-5, 7 | 2 facing N | 51 | 0x31 (dock warp), index on L7 |
| 0:0x22 out of Level 6 | L7, pieces 1-7 | Link turns up, then 2 facing up | 64 | 0x3c, index on L2 |
| 0:0x6d out of Level 8 | L2, all | 1 facing down | - | 0x37 |

**Why the owner's remembered counts ("twice north" at 0x55, "once" at 0x3f) cannot be right for this order.** Played
from every arriving index (`--index`): two N plays at 0x55 land in 0x34 only from Level 2's index (1 → 2 → 3, Level
4's stand-in room one row up), and one N play at 0x3f lands in 0x31 only from Level 5's or Level 6's (on past Level 6,
whose piece Link lacks, to Level 7). But a whirlwind into 0x45 always leaves the index on Level 4 (every way out of
Level 5 to Level 4's screen was played), and 0x55's warp lands in 0x34 only with the index on Level 4 at the pickup,
so 0x3f always starts from Level 4: no index history of this order makes either count right. From Level 4, N N at
0x55 runs on past Levels 6-8 (no pieces) to Level 1's 0x26, and one N play at 0x3f stops on Level 5's. Swapped
("once" at 0x55, "twice" at 0x3f) would be right from Level 3's index at 0x55, which no warp into 0x45 gives. In the
water only N and W turn Link in place (S and E move him), so 0x55 needs the one-frame turn.

### How the warps are played

The engine plays a dock warp as the current recorder warps are played, with the block clip in front (the owner's
"warp whistling again, with a block clip in front of it"; the engine performing the whole sequence once Link clips
at the right spot is accepted for now). On the step (0:0x55 N → recorder, step 149; 0:0x3f N → recorder, step 191)
the session's mask leaves one action where it applies (`zelda::session::dock_warp_mask`): MOVE S on the dock below
its trigger after the raft's ride down (any other step boards the raft back up), CLIP N where the block clip into
the water is legal (`zelda::clips::dock_clip_at`), RECORDER on the water spot (0x55 X `$78`, 0x3f X `$58`), which
plays every play the index needs (`Executor::run_dock_warp`: N for a step up, W for a step down, the turn one frame
before the B) and waits for the pickup; a MOVE along the land stops on the clip's column. Between the dock and the
column the family's own mask: the network walks from the dock to the column. A whirlwind out of a dock room into
any room but the landing ends a chain run `stuck` (`dock-warp-missed`). The landing is the level's
`Whirlwind_PrevRoomIds` entry one row up (asm:Whirlwind_PrevRoomIds, asm:Dock_Update).
