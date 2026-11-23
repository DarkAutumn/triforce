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
| Nov 23 | **One job queue for the machine**: everything heavier than a build or a test runs through `triforce job`, replacing the Nov 15 per-tool line | Owner (the job API: one command, lanes, a build and test lane, the golden verb, timeouts, progress, enforcement); the lead (the join-first finding: chains loaded their networks before joining the line; the single scheduler on SQLite and kernel locks, confirmed by the owner); built by Queue40 | Built | r40-queue |
| Nov 23 | **Test timing logs and a parallel Python suite**: every test run writes a JSONL of its tests' times into the logs folder, read together for history; the Rust suite too; the Python suite's long poles and parallelism tackled | Owner (the logs, their purpose, the Rust lines, the parallel cleanup); built by Tests40 (`triforce.testlog`, `triforce test-times`, pytest-xdist in 8 processes) | Worked: the Python suite 410-433 s → about 1 minute | r40-tests |

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
