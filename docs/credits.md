# Credits: where the ideas came from

The owner plans a video about the whole project. This ledger keeps the credit straight: which ideas were the
owner's, which the AI agents came up with, which were joint, and what came of each. It is meant to stop the agents
from taking credit for the owner's ideas, and to keep the owner's ideas that didn't work from being quietly
forgotten.

**Who is who.** *The owner* is the person who started the project, wrote the first two versions by hand and directs
this one. *The agents* are the LLM coding agents (leads, scouts, reviewers, experiment workers) that wrote this
version's code and ran its experiments. *Outside doc* means a design document the owner brought into the project
from outside this repository. The records say at least two of them (the savestates note and the better-play plan)
were written by other AI agents outside the project; none was written by the agents working here.

**Sources.** The owner's own typed messages (the session history from June to November 2026; only his messages count
as his ideas, never an agent's summary of them), `docs/decisions.md` (headings tagged with the owner are his calls),
`docs/experiments/round*.md`, `backlog.md`, `backlog.old.md`, `README.md` "History" and the git log.

**How to read the tables.**
- **From:** Owner, Agents, Joint (says who did which part) or Outside doc.
- **Result:** *Worked* (as proposed), *Modified* (worked after a change, says what), *Failed* (says why), *Open*
  (untested or still running), *Answered* (a question or a check, with the answer).
- **Record:** `dec MM-DD` is the `docs/decisions.md` heading of that date; `rN §k` is
  `docs/experiments/roundN.md`; `bl` is `backlog.md`, `bl-old` is `backlog.old.md`. All dates are 2026 unless
  given.

## The short version

- **The owner** made the project itself and the first two versions by hand (2023 to early 2026). In this version he
  set almost every route through the game (from his own speedrunning: the clips, wraps, wrong warps, forced bomb
  drops and boss tricks), made the big direction calls (Rust, inference in Rust, families by enemy, the target
  pointer, one network, one unbroken run in 10,000), specified most of the tools (replays, golden paths, the
  players, the GUIs, the job queue, versioning, the annotated disassembly), and steered by watching replays. Most
  of his game knowledge was right. Several of his training ideas carried over from v1 did not help here (the
  heart-versus-exit shaping, the 100-step cutoff, the pass-through penalty, the trap-room curriculum hypothesis).
- **The agents** wrote nearly all of this version's code (about 1,500 commits in November 2026), built the
  emulator and the Rust stack, designed the mechanism behind almost every one of the owner's ideas, and found most
  of what made training work here: episode length as the curriculum lever, chains as the measure, the
  hearts-aware early stop, heart values, the honest planner and the planner as a teacher, the hit heads as a
  shift with their own loss, contact features, harvested handoff states, the census of one team, and the
  statistics that decide what is kept. Many of the agents' own ideas failed too and are listed below.
- **Outside docs** supplied Go-Explore with a backward curriculum (which solved Zora and Wallmaster) and the
  better-play plan (of which self-imitation and per-model act modes were kept; the other seven ideas were not).

## Milestones

| When | Milestone | Record |
|---|---|---|
| Dec 2023 to 2024 | v1: the owner's hand-written project (stable-retro, stable-baselines3); beat Dungeon 1 | README History |
| 2025 to Mar 2026 | v2: the owner's own network architectures and a PPO written from the paper | README History |
| Jun to Aug 2026 | v2 with LLM help: an agent-run experiment loop the owner designed; still stuck in Dungeon 1 | session history |
| Nov 1 | v3 starts from a design doc the owner brought; first working version the same day | git log |
| Nov 2 | Go-Explore solves the Zora and Wallmaster rooms | r2 |
| Nov 5 | The Rust emulator and environment replace Python and FCEUmm | dec 11-05 |
| Nov 8 | Dodongo beaten the owner's way; the `d2first` route | r12, r13 |
| Nov 10 | First full run to the Dungeon 3 triforce (4/120) | r16 |
| Nov 12 | Four dungeons (1.4.0.0); the annotated disassembly lands | r20 |
| Nov 13 | Clips (wraps, block and ladder clips) in the agent | r21 |
| Nov 15 | Five dungeons (1.5.0.0) | r24 |
| Nov 19 | First complete game by the models: a golden path from power on, no cheated items | r30 |
| Nov 20 | 2.0.0.0: the speedrunners' `d3first` route becomes the main one | r32 |
| Nov 22 | The corrected `d3first` route; one network wins a golden path | r36, r37 |
| Nov 23 | First unbroken win on the corrected route: one of 100 single runs from power on (2.0.0.11 team, 6 cheated wall bombs) | git log |

## Before this repository (the owner's v1 and v2)

| When | Idea | From | Result | Record |
|---|---|---|---|---|
| 2023-24 | Learn RL by hand on Zelda: v1 with a few models (overworld, dungeon, with and without beams) | Owner | Worked: beat Dungeon 1 | README History |
| 2025-26 | v2: own architectures, PPO from the paper, training circuits | Owner | Worked partly: the overworld played well, never beat Dungeon 1 | README History |
| Jun 16 | An agent-controlled training loop: the LLM sleeps, wakes at milestones or anomalies, decides to continue, fix or stop, keeps a journal and ends with a list of harness fixes | Owner (design); agents built the extension | Worked as a harness; experiments 1 to 5 still didn't finish Dungeon 1 | session history |
| Aug 3 | Have a stronger agent write a research plan for Dungeon 1 that weaker agents work through item by item | Owner | Done; two items landed before v3 started | session history |

Several of v1's habits came back as suggestions in v3 (heart-versus-exit shaping, the 100-step cutoff, the shield
block reward, enemy positions as distance and direction). They are listed where they were tried.

## Route design

| When | Idea | From | Result | Record |
|---|---|---|---|---|
| Nov 1 | Families revised from the design doc's guess (merges, `ow-moblin` added) | Agents | Worked | dec 11-01 |
| Nov 2 | Boss rooms keep only the entry that matters; clean the states up rather than keep bad ones | Owner | Worked | session history |
| Nov 2 | Route fixes: objectives offer only route exits; a wrong exit is a failure, not a data error | Agents | Worked: the chain reached Aquamentus | r4 §1 |
| Nov 3 | **Families by enemy, not by dungeon** ("a room full of red goriyas will be exactly the same skills") | Owner | Worked after a continuation (goriya r19 beat the split models); the rule holds for every family since | dec 11-03, r6 §1 |
| Nov 3 | Off-route D1 rooms toward the bow added to training | Owner | Worked: 0x33 0% to 75% | r6 §2 |
| Nov 3 | A second route past the fairy pond | Joint (owner the route; agents the waypoints) | Worked (37/40 reached D2); removed with the Nov 8 reset | r6 §9, dec 11-03 |
| Nov 4 | D2: go east first for the 0x7e key | Owner | Modified: a plain continuation failed (0/60); with a backward curriculum 60/60 | r7 §17-19 |
| Nov 8 | **`d2first`**: D2 first, the 0x2c heart, the white sword, then D1; reset everything onto it | Owner (route); agents (route steps, `retreat` at 0x0a) | Worked: first full-route baseline 25/120 | dec 11-08, r13 |
| Nov 8 | Split Moldorm into its own family; Lynel, Tektite and boulder families | Owner | Worked (Moldorm stayed the weakest family) | r11 §2, r13 |
| Nov 9 | The bow detour (trap room 0x22, push block, cellar); "all equipment rooms are pretty much the same" | Joint (owner the detour and families; agents cellars as sub-rooms) | Worked: bow taken 117/120 | r14, dec 11-09 |
| Nov 9 | Map the overworld with its two mazes' rules; route D3 by directions ("W N N W W N S S...") | Owner (directions, maze rules); agents walked them | Worked: his D3 directions matched exactly | r15 |
| Nov 9 | The overworld path to D3, `ow-leever` and `zol` families | Agents (owner away) | Worked | r15, dec 11-09 |
| Nov 11 | Fight the Goriyas before Moldorm for bombs, judged by the owner's rubric (hearts and bombs out, not only success) | Owner (idea and rubric); agents A/B | Worked: 2 points of success for more hearts and bombs in 40% of runs; adopted | r19 §6 |
| Nov 11-12 | Dungeon 4 ("the Breakfast Room"), merged into `d2first` with its keys | Owner | Worked: 1.4.0.0; the scouting had missed the ladder and Gleeok's heads (agents fixed the engine) | r20, dec 11-12 |
| Nov 12 | Route 4:0x21 the long way until the ladder clip exists | Owner (caught the failure from the numbers) | Worked: 0x21 97.3% | r20 §3 |
| Nov 13 | Round 22 route: 0x67's secret, the wrap 0x75 to 0x74, D3 0x5b to 0x5c to 0x4c | Owner | Modified: the wrap needed a new "wrap exit" (agents); D3 change saves about 900 frames | r22 §3-4 |
| Nov 13 | D4 "0x11 to 0x10 to 0x20 to 0x12" for a key | Owner | Modified: those rooms don't border; agents read it as 0x01 to 0x02 to 0x12 | r22 §4 |
| Nov 14 | The rest of the first quest drawn on the map before building; the owner's overworld legs | Joint (owner the order and legs; agents routed the dungeons) | Worked after the owner's review (no shops for bombs, take the keys) | dec 11-13, 11-14 |
| Nov 14 | Dungeon 5 notes: bombs only in the Olive Garden and at the boss, skip 0x66 on entry, 0x65's bombs eastbound only, clip 0x64 | Owner; agents turned them into rules | Mostly worked: westbound bombs still taken 16% of the time | r24 §6, r25 §5 |
| Nov 14 | The Olive Garden (5:0x05) is "one of the hardest rooms in the game" | Owner | Confirmed: the route's biggest loss | r24, r25 §13 |
| Nov 16 | The rest of the game breadth-first (D7, magic sword, D6, D8, D9), about 30% per first team, and "harder than you think": mine the disassembly first | Owner | Worked: 1.9.0.0 ends the game; the walks found errors in the docs | r26-29 |
| Nov 16 | Prepare D7, D6, D8 and D9 in parallel | Owner | Worked | dec 11-17 |
| Nov 17 | Build each stretch as a scouting route, then merge it | Agents | Worked | dec 11-17 |
| Nov 17 | The agents' "working answers" to open route questions, each the owner's to overrule | Agents | Worked: every stretch proved by a scripted walk | r27-29 |
| Nov 19 | After watching the first win: D1 out and back in for a free door, skip key rooms, bomb 0x53, take 6:0x29's key, D9 by 0x56 | Owner | Modified: cost D6 and D9 until like-like was retrained | r31, r33 |
| Nov 19 | **`d3first`**, the speedrunners' route, main once a golden path wins | Owner | Worked: 2.0.0.0 on Nov 20 | r32 |
| Nov 20 | Where the owner's list disagreed with the map, walk the nearest working way and log it | Agents | Worked | r32 |
| Nov 20 | `own_models_until` (route-only runs stop at the magic sword) | Agents | Worked: fixed a D8 loss | dec 11-20 |
| Nov 21 | The corrected `d3first` order with wrong warps ("it's what I told you, but it's wrong") | Owner | Worked: 2.0.0.5, golden 3/3 | r36 |
| Nov 22 | Cross 2:0x5e on `d3first`; drop the 0:0x2d rupee cave | Owner (from replays) | Worked: 2.0.0.6, 2.0.0.7 | r37, r38 |
| Nov 23 | Moldorm passed through; take 9:0x56's key instead | Joint (agents' key ledger showed the key only matters in D9; owner's call) | Open (round 39) | bl |
| Nov 23-24 | The fresh generation's `d3first`: both screens wrapped before Level 3 and no kill before 3:0x5b; 3:0x6b crossed (its key left), one Darknut of 0x5b's first pass for its bombs, 0x59 at 4 or 8 bombs with a reserve of 4, the Moldorm's key back; 9:0x56 kept | Owner (the plan, the wrap in 0x76, the 50-50 cycle starts, the 4-or-8 starts); Route40 (the route data, the ledger showing 9:0x56's key is needed, the side-aware start knobs, 0x1f's row-3 cost, the walks) | Built, not trained (r40-route) | r40 |

## Speedrun tricks: clips, wraps, warps

All of the glitch knowledge is the owner's; the agents made the agent able to use it.

| When | Idea | From | Result | Record |
|---|---|---|---|---|
| Nov 8 | Lynel room: walk through Lynel during the sword-hit frames; a "pirouette" to reroll the spawn | Owner | Modified: the walk-through was confirmed in the ASM; the room reached about 98% without the pirouette | r13 §2, §5 |
| Nov 12 | Use the grid-point skip: overworld wraps, block clips, ladder clips | Owner (the tricks, exact pixels and subpixel walk-in) | Worked; reproduced on the emulator by the agents | r21 |
| Nov 12 | "Smaller moves don't clip, longer moves do" | Owner | Failed: the `long` design had a bad seed and was removed | r21 §1-2 |
| Nov 12 | Test every design on a gauntlet (0x59, 0x5a, 0x5b) and in the trap room, with the owner's experiment plan | Joint (owner the plan; agents four designs and the runs) | Worked: every design learned; the CLIP action (the owner's pick) had no bad seed | r21-gauntlet, dec 11-12 |
| Nov 12 | A south-route curriculum in 1:0x22 lets the agent find the clip by accident | Owner | Failed: the clip was found as surely without a curriculum | r21-trap-room |
| Nov 12 | D4 shortcut 0x31 to 0x21 by the ladder clip | Owner | Worked: 4:0x30 deaths 119 to 0; learned with no curriculum | r21 §5 |
| Nov 12 | Block clip in 4:0x50 | Owner (room); agents measured | Worked: 2,397/2,400 clips, fewer hearts lost | r21-vire-50 |
| Nov 13 | A clips switch at run time ("ok rest of the run no clips"); train with clips on and masked | Owner (idea); agents (stretches, re-planning, the training mix) | Worked: clips off cuts the full route 13.6% to 4.8% | r22 §2 |
| Nov 15 | "You are optimizing clipping, not no-clip" | Owner | Rule | r25 |
| Nov 15 | Point the path through 0x1a's wrap; hold the ladder clip while its landing is occupied | Agents | Failed: wrong exits; too small a gain | r25 §6, §10 |
| Nov 18 | Drop no-clip runs; bypass the Lanmola room by a block clip | Owner | Worked: half the clip training saved; 0x07 costs 1.3 hearts against 9.35 | dec 11-18 |
| Nov 17 | Block clips in D9's ring rooms | Agents (found by search) | Worked: fight budget 25.7 to 14.8 hearts | r29 §5 |
| Nov 21 | Wrong warps at 0x55 and 0x3f by the recorder | Owner (the glitches) | Modified: real, but his remembered recorder counts fit no history; agents found them by search | r36 |
| Nov 22 | Dock warp as an engine macro that waits for shots and strikes monsters below | Agents | Worked: leg 31.7% to 78.9% with no retraining | r36 |
| Nov 24 | 3:0x69 crossed by a line (E along the top, S down the wall column, E to the stairs), swording the Darknut in the way to walk through it while it flashes | Owner (the line, the sword-through, "never takes damage"); agents (the line in the emulator, `path_costs` (d3first's only), the scripted rule and its look-ahead, the task's `demos:` key; Line40's beam search found unhit ways in, Line40b made a best-first search the demonstration builder) | Built for the fresh generation; the rule's and the builder's numbers in r40; trains with the generation | r40 "3:0x69 crossed by a line" |

## Combat, the hit head and actions

| When | Idea | From | Result | Record |
|---|---|---|---|---|
| Nov 2 | Boss hit reward for Aquamentus | Agents | Failed alone; `max_steps: 400` (agents) was the lever: 0% to 93% | r5 |
| Nov 3 | Aquamentus wanders after the kill | Owner (watching); agents (through the heart and out) | Worked: 92 to 11 actions | r7 §2 |
| Nov 4 | Reward blocking by the shield clink, as v1 did | Owner | Modified: the detector missed 6 of 7 blocks (agents fixed it); the plain weight kept, stronger ones not | r7 §6 |
| Nov 8 | **Dodongo the good way**: a bomb beside him, the smoke stuns, then the sword; count bombs used | Owner | Worked: 300/300, 86% by stun | r12 |
| Nov 9 | Blade traps need a curriculum (PPO gave up on them in v1) | Owner | Confirmed: 100% with, 0% without | r14 §2 |
| Nov 9 | Aquamentus trained from low hearts | Agents | Worked: D1 84 to 108 of 120 | r14 §5 |
| Nov 11 | Darknuts can't be bombed from the front | Owner | Confirmed and broadened by the agents (facing alone decides) | r19 §1 |
| Nov 11 | Bombs as a weapon in rooms with tough enemies, a reserve, the 10th kill | Owner ("suggestions, not rules"); agents (bomb modes, schema 3) | Worked for the bomb families (goriya 93% to 96.4%); random bombs at every reset broke bombless D2 until a 50% share | r19 |
| Nov 12 | No sideways attack in doorways | Agents | Failed: one pixel too strict, D2 82% to 69%; fixed | r20 §2, r21 §3 |
| Nov 12 | **Fewer memory writes**: attack by buttons, choose items through the menu, paused frames free | Owner | Worked | r21 §4 |
| Nov 14 | Digdogger: bomb, then the recorder, so the bomb kills him small | Owner | Modified: one bomb only halves him; agents added a sword hit | r24 |
| Nov 14 | Recorder, food and candle actions; widen the models instead of retraining | Joint (owner chose convert; agents built it) | Worked: bit-identical play | r24 §2 |
| Nov 15 | Low-health act rules | Agents | Failed: no variant helped | r25 §4 |
| Nov 16 | **The honest planner** (one step of look-ahead with redrawn RNG) | Agents; owner set its role (a labelled second number, never the result) | Worked as a measure: full route 19.7% to 82.6% with it | r25 §4, §12 |
| Nov 17 | Walk scripts turned into rules and used as demonstrations | Agents | Worked: 8:0x5e 8% to 80%, Gleeok 0% to 100% | r28 |
| Nov 21 | A learned guard network that predicts hits | Agents; owner liked it if look-ahead stays in training | Paused by the owner before its first chain | r35 |
| Nov 21 | The guard "should be a piece of the network", its logits an input to the action | Owner ("an idea, not a directive") | Modified: as an input it took no fewer hits; the agents' hit head that shifts the action probabilities and keeps its own loss worked | r38 §1-2, §11 |
| Nov 22 | **Hit heads** across families | Agents (design); owner (the idea of one network) | Worked: D7 15% to 77%, D4 21% to 68%, D8 39% to 81% | r38, dec 11-22 |
| Nov 22 | A danger zone about 3 tiles in front of shooters while their shot is ready | Owner | Failed so far: reach measured within 2 tiles, under 1% of hits | r38 §6 |
| Nov 22 | Bubbles cost time and hearts; penalise their touch | Owner (question); agents measured | Built; no trained result yet | r38 |
| Nov 23 | Beams at range: reward ranged kills, cost an untaken lined-up shot (v1 did a version) | Owner; agents built the terms | Failed at these weights; stronger ones next | dec 11-23, bl |
| Nov 23 | Red candle and boomerang for Bubble rooms; snipe 6:0x78's Wizzrobes | Owner | Open (a 0x78 swing earned about 0.1 bombs a run) | bl, r33 |
| Nov 23 | Drop the executor's closing no-button frame for the fresh generation, no switch kept | Agents found the frame (Peahat40, measuring the 0x1f line); owner (the cut-over, "don't look back"); agents built the rule (kept after every action but a MOVE in play, measured on today's team) and fixed the seven engine faults it hid | Open (r40-noclose, for review): post-d7 +0.5 ± 0.9 points and 2.4% fewer frames, post-d5 -2.0 ± 3.0 and 4.6% fewer | dec 11-23, r40 |
| Nov 24 | An item's target tiles are the tiles Link takes it from, nothing wider (2.0.1.0) | Agents (Peahat40 found Ganon's Triforce of Power out of reach on its target tile; the lead ruled; Peahat40 built and tested it per item kind) | Open (r40-gen-targets, for review) | dec 11-24 |
| Nov 24 | Fight Darknuts as the owner does: never on the line behind one, shadow one at 90 degrees 20 off and 8 ahead, sword its side and walk on the way it went, expect the wall turn; bombs only for two or more, laid near (128, 128); as a scripted fighter, its demonstrations, and two switched-off proposals (a chase cost, bombs paid only from two) | Owner (the rules, his three takes); agents (Darknut40: the fighter over the env's actions with the game's sword box and turn rule, the take's six swings in a test, 3x5b's one-kill first pass, the measurements, the demonstration builder, `zelda::darknut_rules` and the three keys) | Built; 3x5b's first pass at 1.5 hearts 299/300 with one kill; the rest in r40 | r40 "The owner's Darknut fighter" |
| Nov 24 | Three switches of the fresh generation by measurement: the red candle and the boomerang in Bubble rooms, `penalty-aimless`, 7:0x1b fight or crossing | Owner (the candle strat, the aimless score, the 0x1b override); agents (Switch40) measured, and added a learned candle arm when today's networks never chose it | 7:0x0d: candle learned +9.8 ± 1.6 points (one seed; more self-burns), boomerang masked -61.5; 6:0x38 neither helps; aimless and 0x1b: queued, provisional | r40 |
| Nov 26 | Goriyas their own teacher; a stronger block reward; a cost for attacking with a shot near and no monster near ("near" from Link's whole box: the shot's speed against the attack's lockout, a monster walking at Link into a swing) | Owner (the split, the penalty and his two proximities, "not a directive"); the lead (the block reward, the split as the fallback); Goriya48 (the block decay on its own count, the rule `zelda::exposed` with the aimed exemption, the measurement on the overworld recordings) | Built, the teacher not yet trained | r40 "The Goriya teacher" |

## Bombs, keys and rupees

| When | Idea | From | Result | Record |
|---|---|---|---|---|
| Nov 2 | Bombs off until D2 | Joint (agents proposed; owner agreed) | Worked | r3 §2 |
| Nov 8 | Natural bomb counts; a cheated bomb only in a room that needs one, counted on screen | Owner | Worked | r12 §2 |
| Nov 9 | **The forced bomb drop**: 10 kills without damage, the 10th by bomb; light it up in the observation | Owner (speedrun knowledge); agents corrected the folk version in the ASM | Worked: in every family since round 19 | r15, r19 |
| Nov 9 | One ledger for cheated bombs, keys and rupees ("well designed, not ad hoc") | Owner; agents chose the rules | Worked | dec 11-09 |
| Nov 9 | Manhandla's body hits count toward the bomb drop | Owner ("I believe") | Partly right (agents) | bl-old |
| Nov 9/11 | 0x5b's treasure only below 4 bombs; a D3 reserve of 5 | Owner | Failed: darknut stalled at 0x59 with bombs masked; agents replaced the 0x5b rule with "fight first" (deaths 145 to 59) | r19 §7, r23 §9 |
| Nov 13 | Fish for bombs in 4:0x12 ("not a directive") | Owner | Measured, not built: Link arrives with 0 bombs | r22 §9 |
| Nov 13-14 | BOMB only where it opens the wall; one bomb per wall | Joint (owner "fix the 0x21 bombs"); agents both masks | Worked: 3.8 to 1.2 bombs a visit | r22 §5, r23 |
| Nov 15 | Force Manhandla's drop to bombs in training and make picking them up the first goal | Owner | Worked: out of D3 62.9% to 75.8% | r25 §3 |
| Nov 16 | The arrow leak (every arrow costs a rupee) | Agents found | Fixed | r26 |
| Nov 17 | A granted bomb only opens the wall; the reserve also holds in wall rooms | Agents | Modified: cheated bombs in D7 7.77 to 0 a run after one form was rejected | r26 §13 |
| Nov 17 | Fight 7:0x1b only at 4 bombs or fewer | Owner | Failed: goriya won it 4%; later changed | r26 §12 |
| Nov 20 | A bag of 8 bombs at D5 on `d3first` | Agents (from "we can cheat them") | Reversed by the owner ("Don't blindly add 8 bombs") | r32, r35 |
| Nov 21 | Never buy bombs; drops only; a win without a cheated bomb | Owner | Open: golden paths still cheat 1 to 11 | dec 11-21 |
| Nov 22 | **The bomb rules**: bombs only where a room must be cleared, 0x5b wall only, fight for bombs below 8, the 10th kill always | Owner (from replays); agents the mechanism | Worked: weapon bombs in crossings 1.94 to 0 | r37, dec 11-22 |
| Nov 22 | Bombs left on the floor in 8:0x3f | Owner (watching) | Fixed | r37 |
| Nov 23 | Pay rupees once a pickup, sized by amount | Owner | Built; the opening network not yet retrained | r38 |
| Nov 23 | Flag "the next kill may drop bombs" and point the target at it | Owner | Open | bl |
| Nov 25 | **Force the carrier kill where the observation lights it** (3x5b's first pass: `carriers: fight`, the exit no target while a carrier is named, leaving then a training failure, a chain goes on); a task's room values hold for harvested starts too; why the teacher crossed (the kill paid nothing, a hit cost 6-10, half the room's starts dark) | Owner (the ruling: "force kill that darknut if the part of its observation is lit up"); the lead (the harvested-starts rule, the chain cautions); Carrier43 (the reward table, the stale harvests found, the flag, the ending, the room list) | Built (r43-carrier); arms not run (training stopped) | r40 "Why the Darknut teacher walks past 3x5b's bomb carriers" |
| Nov 25 | **The drop counters at a reset as real play brings them** (a real (`$627`, `$50`, `$51`) triple of the door's kind, two thirds by real play's kill counts and a third uniform 0-15, the drop cycle 0-9; one rule for every task; 3x5b from S as before); **Manhandla and Dodongo never leave bombs behind** (`drops-left` fails the training episode; the armed bomb drop kept frequent) | Owner (every room randomized but 3x5b, "closer to real values than uniform", the bosses' bombs); the lead (the scan read, the triples, the shares); Seeds43 (the power-on scan); Carrier43 (the table and its script, the draw, the ending, the bosses' settings, the proofs) | Built (r43-carrier); not yet trained | r40 "Why the Darknut teacher walks past 3x5b's bomb carriers" 7, dec 11-25 |

## Observation

| When | Idea | From | Result | Record |
|---|---|---|---|---|
| Nov 3 | Audit the observation ("does Link know an enemy is stunned?") | Owner | Worked: found a hex bug in the boomerang states | r7 §3 |
| Nov 4 | A bundle: chase target, boomerang phases, hazard states (from the "shadow Link" report) | Agents | Failed: Zora collapsed, chains stalled; not cut over | r7 §3, §21 |
| Nov 4 | Past snapshots (1 to 4 frames back) | Owner | Open for long; later past observations and a GRU gave nothing | bl-old, r36 |
| Nov 4 | Enemy facing; positions as distance and direction from Link (as in v1) | Owner | Later in another form (schema 7's Link-relative features) | bl-old, r36 |
| Nov 7 | **Direction and distance to the nearest target** ("a little bit cheating, but almost certainly a big difference") | Owner | Worked far beyond expectation: damage about 10 times lower; D2 arrivals 44 to 74-87 | r10 §8, dec 11-07 |
| Nov 7 | Every dropped item as a target (hearts only below full) | Owner | Modified: dungeons only; the overworld got the pointer alone (agents) | r10 §8-10 |
| Nov 7 | Widen old checkpoints to the new schema instead of retraining | Agents | Worked, and became the habit (schema 6: 38,360/38,360 episodes identical) | dec 11-07, 11-16 |
| Nov 12 | Gleeok's heads as enemies; clip availability as inputs | Agents | Worked | r20, r21 |
| Nov 12 | Don't give the "leader" enemy: "the player never knows this" | Owner | Rule | session history |
| Nov 16 | All input changes in one break | Owner (rule); agents (schema 6) | Worked | dec 11-16 |
| Nov 21 | Keep the room identity input but test it | Owner | Answered: an embedding changed nothing | r36 |
| Nov 22 | **Contact features**: monster paths a few frames ahead by arithmetic | Agents; owner approved and asked for projectiles too | Worked: schema 7 | r36, dec 11-22 |
| Nov 22 | Raw RAM bytes, wider towers, a tile patch, the action as input | Agents | Failed: raw RAM hurt | r36 |
| Nov 23 | **Range-aware contact**: frames to contact with a horizon, "not soon" beyond it | Owner | Built (schema 8, off by default): level with or above schema 7 on one family's bench; horizon 16 | r40 |
| Nov 23 | **The shooter's threat lane**, cut where Link can still leave | Owner (idea); agents (columns, the game's shot delay) | Built (schema 8): catches more lane hits on the bench; not yet trained | r38 §6, r40 |
| Nov 23 | **The bomb-carrier flag** as an input, one definition with Metric B | Owner (flag); agents (BombStats' predicate) | Wired, reads 0 until `r40-bombstats` lands | r40 |
| Nov 24 | **Where the monsters are sent, and who may turn at Link**: the chase target, its flip and Link's distance from the mirror's fixed point (the "hotspot"); per grid-point walker, the frames to its next grid point, the turn toward the target there, its chance and whether it reverses onto Link; and which monsters the Darknut rules cover | Owner (the lesson, "don't chase", shadowing at 90°, the clumping zone, the audit question); the lead (the game's rule behind it); Chase40 (the audit of every monster's movement, the columns, the proof on replays) | Built (schema 9, `obs.chase` off by default); not yet trained | r40 "The chase target", `docs/owner/darknuts.md` |
| Nov 26 | **When a Goriya throws next**: loaded, its throw timer, and its throw within the next decision (a blue's by the game's rule, a red's as its 2-in-256 roll a frame) | Owner (knowing when a Goriya is loaded); the lead (the three columns from the game's rule); Goriya48 (the rule walked frame by frame, the proof on the golden replay) | Built (schema 10, `obs.shooter` on for the Goriya teacher only); not yet trained | r40 "The Goriya teacher" |
| Nov 27 | **When every shooter fires next**: the same three columns for Lynels, Moblins, Octoroks (their 32-frame stand), the Zora, Aquamentus, Gohma and the red Wizzrobe; the blue Wizzrobe left out; a trained `hitnet` widened to the new input with zero columns and continued | Owner ("apply that (b) not only to goriya but also all enemies that shoot"); the lead (the scope, and the order as the owner's rule: widen a trained teacher with zero weights and continue, no architecture change first); Shooters49 (each type's rule from the code, the exactness on five replays, the register-read gates, the widening at `--load` and its proof) | Built; the overworld teacher's widened continuation dry-run only, not trained | r40 "Every shooter", docs/lab.md "Continuing with a new input" |

## Training method

| When | Idea | From | Result | Record |
|---|---|---|---|---|
| Nov 1 | One PPO specialist per room family | Outside doc (v2 design) | Worked as the frame until the one-network program | r1 |
| Nov 2 | **Go-Explore with a backward curriculum** | Outside doc ("Training with Savestates and Branching") | Worked: Zora 37% to 97%, Wallmaster 18% to 87%; regressed one family | r2 §3 |
| Nov 2 | Enemy health and a progress term in the Go-Explore cell | Agents | Worked | r2, r3 |
| Nov 2 | Branched rollouts (VinePPO, GRPO); search distilled into the net | Outside doc | Never run; planner imitation later | bl-old |
| Nov 2 | Train with full and partial hearts (beams on and off); route-true max hearts | Owner | Worked: part of every start since | r3 §2 |
| Nov 2 | Episode length sets the curriculum rate (`max_steps`); one-room episodes | Agents | Worked | r4, r5 |
| Nov 3 | A damage penalty for pass-through rooms ("only keep it if it helps") | Owner | Failed: no chain gain | r6 §5 |
| Nov 3 | Chain-matched health, hearts as objectives when hurt | Agents | Failed (and a CLI bug voided the first arms) | r6, r10 |
| Nov 4 | Cut a run after about 100 steps without progress (v1) | Owner | Failed as proposed (cut real successes); became the no-damage counter | r7 §10 |
| Nov 4 | A finite timeout counting real damage, in training and the player | Owner | Worked | r10 §6 |
| Nov 4 | Track lingering in rooms | Owner | Worked: a standing metric | r7 §1 |
| Nov 4 | Link leaves the start without the sword: fix it | Owner (the problem); agents (targeted training) | Worked: 0/120 | r10 §3 |
| Nov 4/6 | The heart closer than the exit wins the wavefront (v1) | Owner | Failed: no pickup gain | r7 §9 |
| Nov 4/6 | Boomerang on or off after D1 | Owner (the experiment) | Answered: no difference; kept | r10 §2 |
| Nov 7 | A fixed reward for a heart; health gained as the metric ("current health dominates everything") | Owner | The fixed reward added nothing on top of the pointer; the metric drove the heart series | r10 §8 |
| Nov 8 | Treat the Lynel room like a boss with a curriculum | Owner | Answered: both arms about 98%; the curriculum only halves the steps | r13 §4 |
| Nov 9 | Act from the policy only; search only in training ("feels like cheating") | Owner | Rule | dec 11-09 |
| Nov 9 | Try the better-play ideas one by one, not as one bundle | Owner | Worked: each measured on three seeds | r16 |
| Nov 10 | Self-imitation learning; argmax or sampling per model | Outside doc (better-play), agents' variants | Worked: D3 11 to 34 of 120, first full routes | r16, better-play-reply |
| Nov 10 | Entropy annealing, outcome fine-tuning, best-of-N, Gumbel expert iteration, chain-aware training, PLR, PPG | Outside doc | Failed: none beat a control | better-play-reply |
| Nov 11 | Heart loss matters even on success | Owner | Partly: the agents' hearts-aware early stop fixed two families; goriya didn't move | r18 |
| Nov 11 | Early stop also waits for hearts | Agents | Worked: full route 2.6% to 10.6% | r18 |
| Nov 13 | Variety in play without the rare bad move; drop actions under about 10% | Owner | Modified: the threshold looped (113 stuck); agents' temperature 0.5 plus `relax:16` chosen by the owner | r23 |
| Nov 14 | Leave rooms with more hearts ("beat the room with 1 heart and died in the next") | Joint (framing from an agents' research list; owner chose it) | Worked through the agents' `heart_value`: D5 +13.5, D4 +11 | r25 |
| Nov 15 | Clone a scripted rule (5:0x37 at the owner's column `$70`) | Joint (owner the column; agents the cloning) | Worked: 1.19 to 0.20 hearts a visit; hand-made curricula and path costs failed there | r25 §5 |
| Nov 15 | `path_costs` off spawn spots; a general `spawn_cost` | Agents | Modified: hand-placed costs worked in two rooms; the general rule failed | r25 §6, §10 |
| Nov 16 | Use the planner as a teacher | Joint (owner asked it be tried; agents built it) | Worked: +2.4 to +4.0 on every seed | r25, r27 |
| Nov 19 | Harvested handoff states; one team picked by a census | Agents | Worked: no segment worse, D2 +10 | dec 11-19, route-team |
| Nov 21 | **Few networks, ideally one**, as rounds of experiments | Owner | Open: one network won golden paths but chains were weaker; "doesn't count" | r36, r37 |
| Nov 21 | One network by distilling the 51 | Agents | Modified: golden paths yes, chains no | r36, r37 |
| Nov 22 | **Target: one unbroken run in 10,000** (the record holder: 252 wins in 13,135 tries) | Owner | First unbroken win on Nov 23 (1 of 100 runs from power on, 6 cheated wall bombs) | dec 11-22, git log |
| Nov 22 | An "aimless walk" score | Owner; agents built score and penalty | Modified: act-mode fixes failed, fine-tunes fixed it | r38 |
| Nov 23 | Ow-lynel trained on `d3first`'s leg | Agents | Worked: 24% to 99% | dec 11-23 |
| Nov 24 | The fresh generation's curricula audited against today's start states by the explorer's own reset (a track is tied to its state by name only), the stale ones dropped, the missing re-explored, the folder proven the way a training loads it | The lead (the task, the store's rule, the proof by the training's own checks); Tracks40 (the reset as the reference, the audit, compose and proof script, the re-explore, the declarations) | Built (r40-tracks, for review): 419 of the store's 479 `d3first` tracks stale and 49 stateless, 269 of 294 states re-explored, 25 declared untracked; the folder `trajectories-gen1` loads for all 57 task files | r40 "The generation's curricula" |
| Nov 24 | The lab trainer made fit for the generation's teachers: the teachers as data (`--group`), the recipe on record (manifest, `--same-recipe-as`), refusals for silent faults (a route's states, a task's warm start, obs switches against each other and the network's token groups, demonstrations recorded without a key), `--resume` of a stopped run; of the single-task trainer's extras only what a from-scratch family needed (none) | The lead (the pins, the groups, the ask; the `untracked` fault under `--state-route` found); GroupScout (the map of what was missing); Trainer41 (the build, the refusals' rules, the resume, the point-5 reading) | Built (r40-trainer, for review): two seeds of `--group rooms` with one stopped and resumed to the same 32,768 steps; first timing about 480 decisions a second at 8 envs | r40 "The lab trainer for the generation's teachers" |
| Nov 24 | Tracks for the 27 states the generation's curricula left untracked (Tracks40's 25 and 3x59's two at 4 bombs), by its own means in order: the scripted rules brought to today's route and recorded from the explorer's reset, Darknut40's fighter's unhit visits (3x59 from N, and 3x5b's last pass as a new room of the fighter), the explorer aimed by what its failed searches reached | The lead (the task, the means and their order, the rule against an old network's play); Tracks41 (the rules' env and `--hearts reset`, the fighter's last pass and its tracks, the explorer's "reached" report and its reading: the budget, not the step limit; 3x5b's first-pass tracks found stale and replaced) | Built (r40-tracks2, for review): all 27 tracked, 20 by seven rules (no heart lost), 4 by the fighter (unhit), 3 by the explorer at 20,000 iterations; none from a network; the folder `trajectories-gen2` loads for all 72 task files | r40 "The untracked states' tracks" |
| Nov 24 | Stage 2's tools made fit for the generation: recording with teachers named per family or per group by run, on the route's task files, carrying the hit labels beside the teachers' labels; a teacher or student that would read a switched-off block refused; imitation with the lab's own hit loss in the same pass, a fresh `hitnet` from the lab's builder, round 38's two-pass recipe as defaults; a correction round through the same named-teacher path | The owner (one network; the hit head inside it); the lead (the pins, the ask); GroupScout (the map of what was missing); Stage41 (the build, the held-out rule, the proof) | Built (r40-stage, for review): proof with stand-in teachers, 16,384 rows recorded, a fresh 1.6M `hitnet` imitated (held-out KL 1.71 → 1.10), a correction round and a second pass (0.89 → 0.55 on its larger held-out set), PPO from it with a fading teacher | r40 "Stage 2's tools for the generation" |
| Nov 24 | The generation's tasks say what it trains on (the two input switches, the aimless penalty, from scratch) in one common block every d3first task extends; a chain plays a network with the inputs it trained with, and refuses one that would read a switched-off block as zeros; darknut's demonstrations recorded again with the inputs on | The lead (the pins, `extends:` lists, the guard); the owner (the inputs, the aimless default's experiment, the candle strat); Tasks41 (the common block, the chain config rule `CHAIN_FROM_ROUTE_TASK`, the guard, the rebuilds) | Built (r40-tasks, for review); bare tasks and `d2first` unchanged by hash; today's team identical on Dungeon 1 both routes | r40 "The generation's tasks" |

## Emulator, Rust and infrastructure

| When | Idea | From | Result | Record |
|---|---|---|---|---|
| Nov 1 | A tool to play Zelda by hand as an LLM, built first, for debugging | Owner | Worked: `python -m triforce.play` | session history |
| Nov 1 | Drive the emulator core directly | Agents | Worked: 582 to 3,744 frames/s | dec 11-01 |
| Nov 5 | **Port to Rust** with the TetaNES emulator | Owner (direction, an outside architecture doc); agents (plan and port) | Worked: zero parity differences | dec 11-05 |
| Nov 5 | Label every FCEUmm quirk, then restore hardware behaviour and retire the oracle | Owner | Worked: all 13 restored, no model lost | r9, dec 11-05 |
| Nov 5 | Save once after the name screen, never replay it | Owner | Worked: the base of every movie | dec 11-05 |
| Nov 5 | Raise test coverage; then profile | Owner | Worked | r10 |
| Nov 7 | Make zcore faster with nothing about the game changed; benchmark the whole game; find the right `--parallel` | Owner | Worked: 1.60 times; `--parallel 16` | r10 §7 |
| Nov 7/9 | Fork the beam look-ahead onto other threads; a continuation queue | Owner ("I'm not sure it's a good idea") | Not built: under his own gain threshold | dec 11-09 |
| Nov 8 | Power-on RAM $00 | Joint (agents raised it; owner chose) | Worked | dec 11-08 |
| Nov 9 | **Inference in Rust**, one call per rollout, replays that reproduce probabilities | Owner; agents designed it | Worked | r16, dec 11-09 |
| Nov 9 | Overlap the PPO update with the next rollout | Agents; owner approved by his speedup rule | Worked: +20% to 45% | dec 11-09 |
| Nov 9 | GPU or CPU inference, measured | Owner | Answered: CPU | dec 11-09 |
| Nov 10 | A whole-repo review by a stronger agent, a ledger of findings | Owner (process); agents (192 findings, 186 fixed) | Worked: a fix exposed Zora counted as killed | r17 |
| Nov 11 | Versioning (major the owner's, then dungeons, compatibility, revision) | Owner | Worked; minor redefined at 2.0 | dec 11-11, 11-19 |
| Nov 11 | Tests under about 10 minutes; a timeout on every test | Owner | Worked | bl-old |
| Nov 12 | **The annotated disassembly** for agents, with an index and enemy and item notes | Owner; agents annotated it | Worked: it corrected many repo claims | bl-old |
| Nov 12 | Dungeon documents for agents | Owner | Worked | session history |
| Nov 13 | Training and evaluation niced; evaluations batched | Owner | Worked | dec 11-13 |
| Nov 15 | **A job queue** (two at a time, no central controller); shortest first with age | Owner (rules); agents (flock design, estimates) | Worked | dec 11-15 |
| Nov 17 | Every finished stretch lands on main, pushed and playable | Owner | Rule | dec 11-17 |
| Nov 21 | The lab: any network in chains | Agents | Worked | r36 |
| Nov 23 | A golden replay per model version, backed up; 250 runs per new model with best, worst and stuck runs | Owner | Open | session history |
| Nov 23 | Versioning's third number is the retrain generation (90% of the promoted models retrained from scratch); models load by their checkpoint's schemas, not the version | Owner; agents found the version gates and built the schema check (`OLDEST_OBS_SCHEMA` for a change of meaning) | Worked: every promoted run of today's families loads | dec 11-23 |
| Nov 23 | A version's first three numbers each mean a fresh generation: every network retrained from scratch, breaking the ship of Theseus; the next generation's teachers trained fresh too | Owner; the lead traced the lineage (73 of 76 promoted networks are continuations, 69 with lost origins) | Rule set; first fresh generation is 2.0.1.0 | dec 11-23 |
| Nov 23 | The repeatability check: one family (zol on `d3first`) trained from scratch on today's recipe, three seeds, timed, against today's network | Owner's question ("is our approach repeatable or are we building the ship of Theseus"); the lead's recipe (one family first, then waves); Fresh40 ran it | Modified: on `d3` one seed of two reaches the bare promoted policy within noise (-0.7 ± 2.1), the other is 5.7 short; neither has the hit head (+6.0); about 1.5 h a 4M-step seed | r40 |
| Nov 23 | **One job queue for the machine**: everything heavier than a build or a test runs through `triforce job`, replacing the Nov 15 per-tool line | Owner (the job API: one command, lanes, a build and test lane, the golden verb, timeouts, progress, enforcement); the lead (the join-first finding: chains loaded their networks before joining the line; the single scheduler on SQLite and kernel locks, confirmed by the owner); built by Queue40 | Built | r40-queue |
| Nov 23 | **Test timing logs and a parallel Python suite**: every test run writes a JSONL of its tests' times into the logs folder, read together for history; the Rust suite too; the Python suite's long poles and parallelism tackled | Owner (the logs, their purpose, the Rust lines, the parallel cleanup); built by Tests40 (`triforce.testlog`, `triforce test-times`, pytest-xdist in 8 processes) | Worked: the Python suite 410-433 s → about 1 minute | r40-tests |
| Nov 23 | **The shared store**: one place outside every checkout for everything shared (ROM, named data snapshots, network runs, teams, what jobs file), written only through one program, with a separate entry point for the lead, models moved there too, and a staged cutover; the aggregate statistics kept for good as SQL | Owner (the location outside the checkouts, named snapshots, writes only through an API, the separate lead entry point, the staged cutover, the aggregate in SQL); the lead (the store's shape: layout, sealed runs, immutable teams with slots, copies not hard links, summaries kept for good); built by Store40 (`triforce.paths` and its Rust twin, `triforce.store`, `triforce store`, `triforce-lead`, the import, team views, the statistics tables with Stats40's fields) | Built, not yet cut over | r40-store |
| Nov 23 | **Recipe integrity**: a written inventory of every file a run reads or writes outside git (`docs/data-layout.md`), then every input a run declares that resolves to nothing stops it before it starts, naming where it looked (start-state globs, harvested states also in the lab and under a running pool, curriculum tracks unless the task declares them untracked, warm starts, a team's missing network, a dropped hit head without `--headless`, a missing `--demos` file, a module not built from its sources); then each training, chain and evaluation records its manifest (code, ROM, task and its `extends:` chain, data by count and content, networks by SHA-1, models dir, arguments, seeds, job), `triforce manifest diff` between any two runs (also records from before manifests, saying what they cannot tell), `train --same-recipe-as RUN --except` and `eval-chain --same-team-as OUTPUT --except`, and red flags (`jobs.flag`, every listing and the GUI's badge; every job start's preflight: main's tracked files edited, a worktree behind main, a missing data link, a stale module; a training drawing none of a data set its family's promoted run drew) with `triforce job doctor` | Owner (robust per-run statistics for every agent's runs, red flags for runs that saw none of an input, worktree setup that cannot repeat the two void results of Nov 23; the inventory first, then shared state in one place); the lead (the analysis of the failure class and the design: refusals, manifest, diff, flags, doctor, the main-checkout-is-dirty flag); built by Recipe40 (the inventory and its silent defaults, `triforce.inputs`, `triforce.build`'s stamp, the `curriculum.untracked` declaration, `triforce.manifest` and its readers of old records, the preflight's checks, the training's own flags) | Refusals landed; manifest, diff and flags built | r40-recipe |
| Nov 24 | The generation's base made green: tests that read a state file's contents or the wall clock made to say what they test (a seed chosen again for 3x69's rule, the emu takes walking to the door's column and dying with half a heart, the GUI's clock held), main's heart encoding's moved traces compared step by step and recorded, and a SIGTERM held while a training logs its metrics (TensorBoard's queue turned the interrupt into a `RuntimeError`, exit 1) | The lead (the task and the suspects); Gen40fix (the causes, the fixes) | Built (r40-genfix, for review) | r40 "The generation's base made green" |
| Nov 24 | The stored states audited for the pre-2.0.0.16 hearts bytes (a whole heart below full as (H, $00)) and a corrected `harvested` snapshot for the fresh generation: the game's code read for every write of `$66F` / `$670` (play leaves a full heart beside an empty partial only mid-refill, or after 31 to 127 borrows since the last refill to full), 69,522 states read, the 70 ow-lynel `0_0ac_h*` the only ones, each rewritten at the same hearts and proved byte for byte | The owner (the find behind 2.0.0.16); the lead (the scope, the question of play's own (H, $00)); Hearts41 (the reading, the audit, the copy and its proof) | Built (r40-hearts; `target/harvested-gen1` for the lead to install) | r40 "Hearts in the stored states" |
| Nov 24 | **The queues are one file the owner edits and saves**: each queue a name, `max-parallel`, optionally `max-job-length` and `load-average-max` (above the line nothing new starts, running work finishes, read once a minute); the default the queues of Nov 23 plus one spare place while the load average is at or under 15; replaces the morning's `places` stop-gap | Owner (the file, the fields, the load rule and its cadence, the default); built by Queue41 (`queue_dir()/queues.yaml` read live by every waiting job, the file's order as the order a job tries, the limits, the last good file kept when a save is refused, the shared load reading, older code reckoned by its constants, `triforce job queues`) | Built | r40-queues |
| Nov 24 | **Main reads the store**: the main checkout's own data folders compared with the store by content (all identical, nothing only in main, every file in the archive too), the suites given the store's current team instead of main's `models/`, the tests and scripts that leaned on main's folders made explicit, the documents' old paths mapped, and a move aside (never a delete) the lead runs in one step after the owner's yes, with its way back | Owner (one store outside every checkout, the staged cutover); the lead (the reconciliation: main like every worktree, prepared before anything moves); built by Store41 (`triforce.reconcile` compare / aside / back, `job.test_steps` and the suites' team view, `job doctor`'s ROM and networks lines, `docs/store-paths.md`) | Built; the move waits for the owner | round40.md "Main reads the store" |
| Nov 25 | **Watch the steps a second, and fix what made them half**: the stage-1 teachers' 300-400 steps a second traced to the generation's network being served eagerly (CUDA graphs off in trainings since Nov 23; the games asleep half their time), not to a regression; the graphs back in the lab trainer, captured once at the first rollout; every training's steps a second on record with its normal, SLOW and an IN/OUT OF BOUNDS last line, `triforce job speed` | Owner (the question, "1000 sps+", the record in the db with what ran beside, the alarm and the bounds line, the hearts line); Perf43 (the diagnosis from /proc and the queue's db, the capture-once design, the alternating and audit measurements, the record's design) | Worked: x2.2 steps a second for an 8-game teacher, x1.1-1.4 for a 20-game one, 0 of 81,937 picks changed | round40.md "Where a teacher's time goes" |

## Tools for watching

All specified by the owner and built by the agents unless noted.

| When | Idea | Result | Record |
|---|---|---|---|
| Nov 2 | A JSONL event stream per run and a training GUI with bars; colorblind-safe colours (blue for good) | Worked | r4 |
| Nov 3 | `play-zelda.py`: a grid of live games at 60 fps | Worked; later replaced by replay queues | session history |
| Nov 4 | Play many games, show the best; swap only on a room change or game over | Worked | session history |
| Nov 11 | **Replays** (TFREPLAY): inputs, memory edits, the cheat ledger, optional rewards and action probabilities, from power on or a state | Worked | dec 11-11 |
| Nov 11 | **The golden path**: race many workers to each milestone, stitch one replay | Worked: the first complete games | dec 11-11, r30 |
| Nov 11 | play-zelda from queues (success, stuck, died, random, best); headless runs; sound, video, a controller overlay | Worked | dec 11-11 |
| Nov 13 | Statistics by segment, full run and room | Worked | r21 §8 |
| Nov 14 | A web view of the training GUI for the phone | Worked | session history |
| Nov 23 | **Replays that say on their own what the agent was steered to, could do and chose**: the objective and role-labelled targets (`reward-only` ones too), the action mask with its rules, every decision with its distribution and danger, named by the replay's own META (no paths, no route data); a chain's report from its replays alone; a thoughts overlay for the video. The owner's spec and answers; the agents the format (replay version 2), the reader and the overlay | Worked: the Dungeon 8 chain played the same 200 runs frame for frame with it | session history, `docs/architecture.md` "Replays" |
| Nov 23 | **The GUI draws every job's progress with its own widgets**, never a text bar; the 250-run test reports what he watches live (runs done, wins, each triforce's runs, deaths and stucks, the furthest step, runs a second and time left), as a chain, a batch, a golden race and a training show theirs. The owner's ask ("the gui+you launching is 99% of my workflow"); built by Gui40 | Built | r40-gui |
| Nov 23 | **Music and sound effects as two parts that switch off apart**, in play-zelda (keys and flags) and in the videos (two audio tracks), with the low-hearts beep on its own switch; the item fanfare and the end-of-level tune go with the effects though the game plays them through its song driver. The owner's idea and division; the lead's design (the game's own sound driver run again beside the emulator, a music copy and an effects copy each fed only its own requests, so the music keeps every voice an effect would take; the default sound unchanged); Sound40 built it (`zcore::tracks`: its own small 6502, a mirror driver that finds the game's own writes to the driver's state, a no-beep copy for the beep alone off, the DMC's resting level shared with the music) and the proofs (`tf replay sound --check`) | Built; READY FOR REVIEW | r40-sound, `docs/architecture.md` "Sound" |
| Nov 23 | **A right click in the GUI moves a waiting job to the top of its line** (and back; `triforce job top`), without touching what older code queued. The owner's ask ("a right click in the gui that goosed something to the top of the fast or slow queue"); built by Gui40 | Built | r40-top |
| Nov 24 | **`emu-zelda.py`: the owner plays a room himself and keeps it as a replay** the agents read (the Olive Garden by the sword alone first), on the project's own emulator with a PS5 controller: the route's start state, paused until a game button, Create a hard pause, square saves a snippet, triangle / R1 start over with a new / the same seed, DONE on leaving the room or dying (a cellar under it is the same take), - and + by 5 frames a second with the sound only near the NES's rate, the centermost monitor. The owner's spec; Emu40 built it (`_native.LiveGame` recording as a chain run records, `triforce.emu` with the DONE rule a setting, `triforce.emu_view`) | Built; the controller, monitors and sound device are the owner's to try | `docs/architecture.md` "Playing a room by hand" |

## Watching and steering

The owner watches replays and reports what he sees. Much of the progress came from this loop.

| When | Report | Result | Record |
|---|---|---|---|
| Nov 3 | The Moblin room with two openings fails | Agents found exit spans and a wavefront lane: 18% to 97% | r6 §9, dec 11-03 |
| Nov 4 | Monsters chase "shadow Link"; Link lingers | Lingering became a metric; the shadow-Link explanation was not supported by replays | r7 |
| Nov 4 | Goriya rooms: attacking standing still forever | The no-damage counter | r10 §6 |
| Nov 11 | The endless 0x4c/0x4d bounce | Fixed (agents found the cause) | dec 11-11 |
| Nov 12 | Gleeok isn't learning | Agents found the shaping swamped it: 22% to 95% | r20 §3 |
| Nov 19 | Eight route changes after the first complete game | See route design | r31 |
| Nov 22 | Bombs wasted in 3:0x69 and on 0x5b's Darknuts | The bomb rules | r37 |
| Nov 23 | The opening walks back and forth because its network learned another route's opening | Confirmed: 661 to 4 aimless frames | dec 11-23 |
| Nov 23 | 8:0x4c's loop: trained while its key was still taken | Confirmed: 92% to 100% | dec 11-23 |
| Nov 23 | 3:0x5b walks between the wall and the bombs ("two targets?") | Targets are right; the network predates the rules; retrain open | r38 |
| Nov 23 | Two Zols reappear in the Breakfast Room | Investigated; no finding in the docs yet | session history |

## Evaluation and process

| When | Idea | From | Result | Record |
|---|---|---|---|---|
| Nov 2 | Per-start-state (balanced) early stop | Agents | Worked | r2 |
| Nov 3 | The chain is the measure, not single rooms | Agents | Worked: a standing rule | r6, r7 |
| Nov 4 | The alternate-YAML CLI bug voided five arms | Agents found | Fixed | r7 §4 |
| Nov 10 | Three training seeds per arm (noise is about 2 points) | Agents | Worked: the main lesson of better-play | r16 |
| Nov 11 | Chains with 1000 runs | Owner ("that'll get a better read") | Worked: the standard | r18 |
| Nov 11 | Get back within 5% of the old team before new work | Owner | Worked | r18 |
| Nov 15 | About 5 points is a result at 1000 runs; confirm on fresh seeds | Agents | Worked | r25 §7 |
| Nov 22 | Bench shortlists, chains decide | Agents | Worked | r37 |
| Nov 23 | Two bomb metrics in every 250 test, judged only by what the observation tells the agent: the 10th-kill bomb (taken with a bomb or not, full bag apart) and the bomb carrier (the next kill one that may drop bombs); computable from a replay alone | Owner (metrics, edge cases, replay scoring); agents built the counter | Built; baseline pending | architecture.md "Bomb windows" |
| Nov 23 | One standard statistics file for every training, chain and evaluation, kept in one place and read by one reader (per room and family: hearts lost, hits, shots fired and shots that hit, kills from range, beams held, time to each step, deaths, stuck runs); red flags for a run that saw none of something its task or route says it must | Owner (the request); the lead (the design: records to one file, paired comparison in the owner's order, the canaries); agents built it (`run_stats`, the shot-hit and beams-held counters, the flags' thresholds) | Open (r40-stats, for review) | session history |
| Nov 24 | Bomb carriers as targets in crossed dungeon rooms (the exit and the carriers whose death may drop bombs now, one list with the observation's column); a cheated wall bomb only while the wall is the objective; the route's goals (wrap, no kill where crossed, the drop cycle, no hit on a planned path, carriers) tracked per visit; aimless frames in the stats | Owner (the Dungeon 3 plan, the wall-bomb rule, "track goals"); the lead (carrier targets, the goal list, aimless frames in stats); Carrier40 built them (one carrier list, the ledger's objective verdict, goals from route data) | Built; for review (r40-carrier) | round40.md "Carriers, the wall's grant and the goals" |
| Nov 24 | A teacher of the fresh generation judged in chains with one command (one run in every slot of its group's families, `+@GROUP=FAMILY/RUN`; chained only where its families play, each dungeon from the previous triforce and from its entrance) and set beside the 2.0.0.15 baseline per dungeon, paired by seed | The lead (the ask, from GroupScout's map; the baseline's two starts); Judge41 built it (the group slots, `eval-batch --teacher`, `triforce judge compare`, the yardstick's shared-run sentence) | Built; for review (r40-judge) | round40.md "A teacher judged in chains, beside the baseline" |
| Nov 24 | A training's numbered checkpoint judged while it trains: made a sealed run of its own (copied from an open run, which stays untouched; linked from a sealed one), `@itN` / `@latest` in `eval-batch --teacher`, `judge checkpoints` (one express job a checkpoint and scope, one table by iteration beside today's team on its own route and on the generation's route); found on the way: today's team dies in 3x5b's first pass on the generation's route, and the walkers' 0 of 300 at it114 is a room not yet learned (equal rates in the chain and the training env under equal conditions), with the cycle-0 fight drawn on half the resets | The lead (the ask, the second base, `--with`, the diagnosis's three questions); Check41 built it, found the confound and ran the diagnosis | Built; for review (r41-check) | round40.md "A teacher's checkpoints judged (Check41)" |
| Nov 24 | What a teacher really trains on: the walkers' "Darknut collapse" traced to harvested states of the route's first order (no Dungeon 2 piece: no target where a step leaves by stairs), a fifth of the snapshot stale, the round-robin warm-up walking one room's block of files for most of a run; the fix: the seed draw picks the room entry and harvested states stand in for it, a random warm-up, stale harvests refused at reset, never written, audited per snapshot (`harvest_audit`), seed and harvested starts logged apart; the seed-state skill curve showing no collapse | The owner (the question: what exactly happened, Darknuts their own teacher); the lead (the first findings: blocks in the per-room counts, it912 winning the seed states, the failing `h6xxx` states); Harvest42 (the cause in RAM and the objective code, the audit, the reconstruction, the curve, the fix); the lead made the cleaned snapshot | Built; for review (r42-harvest) | round40.md "What the teachers train on" |
| Nov 25 | One start state per door, real: every `d3first` state walked, the machine carried door to door (no redraw, no fresh room, no refill), from console power on through Dungeon 1 and then in segments side by side from each dungeon exit's state checked against the route, the explorer started exactly from it and held to the plan's kills (routes.yaml `clear: true|false|N`, derived from the ROM's room data), a fight it cannot end ended by the game's own death, Ganon's door by the round-29 rule; proven door by door against 445 real runs and by identical placements in real replays | Owner (the rule: "1 save state per 'door/entry'", "real, not fake", "never train on an un-real state"; "add it if you need it" for the route key); Seeds43 (the finding: `$524` decides placement and is redrawn; the scans of real runs); the lead (the task, the plan's decided steps confirmed, the segments, the hearts fallback, Ganon by the rule); States44 built it (the key and its checks, exact starts, the kill limit, the walk and its segments, the record and the audit, the proof) | 285 states, one per door; `$524` the runs' commonest at 238 of 242 compared doors, the plan's composed value at 131 of 133; placements identical at 27 of 30 (the other 3 the plan's first-pass kill) | round40.md "The real walk" |
| Nov 25 | Curriculum tracks and the Darknut demonstrations again from the walked states, in the route's layout: the explorer loads the route's planned `$524` (or one named reachable value), every track says the layout and the counters it started from; a track start keeps its layout and monsters (proven: RAM equal to the track's state but the random phase, 822 of 822); fighter tracks in every other reachable layout of the hard Darknut doors and from both bags and both drop cycles the route brings; the old runs read for which rooms lag in layouts they had no track in; the finding that a track start redrew the room's counters | Owner (the rule: "never train on an un-real state", the route's layout weighed); the lead (the task, the second-track question, the track-start ruling); Carrier43 (the track-start rule built); Tracks45 (the layout load, the records, the proof, the tracks, the demonstrations, the reading of the old runs, the counter finding) | Built (r45-tracks): every family's `d3first` tracks on the whole walked route and the Darknut demonstrations staged | round40.md "Tracks in the route's layout" |

## How credit is kept from now on

- Every round doc says where its ideas came from: the owner's, the agents', or joint (who did which part), and what
  came of each (worked, worked after a change, failed and why, open). A line or a short section is enough; no
  verbatims needed. `AGENTS.md` asks for this in its standing rules.
- Only the owner's own typed messages make an idea his. A lead's or agent's summary of what he wants is not the
  source, and his questions are not proposals unless he says so. When the agents build the mechanism for his idea,
  say so; when his idea fails, say so plainly.
- Decisions headed with the owner keep marking his calls in `docs/decisions.md`.
- At the end of a round, or every few rounds, add the round's main ideas to the tables above, in the theme they
  belong to.
