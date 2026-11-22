# Experiment backlog

The long-term plan: what the project is trying to do now, the experiments queued for it, and what was found but not
acted on. The short-term list is the lead's todo. Everything before Nov 21, 2026 (rounds 1-35: the whole game on
`d2first`, the first `d3first`, 2.0.0.3) is in `backlog.old.md`, kept as it stood; an item carried over from it is
named below with its place there.

## The program (the owner, Nov 21): the corrected d3 route, played by one network

**The end goal:** beat the game on the d3 route with a single neural network. A handful is acceptable and is likely
where the project starts and may end; one is still what to try for. "You should consider this a reset of
expectations": the merged models are not gated on matching today's specialists segment by segment (they are still
compared with them).

**The target, and the only stopping point (the owner, Nov 22):** **one run in 10,000 tries that completes the game
without restarting**, one network, the corrected route. A golden path does not count ("No, it doesn't count"); it
is wanted first all the same, to watch (one network won one on the corrected route on Nov 22:
`corpus/replays/golden-d3first-single-network.tfreplay`). The lead does not stop short of that target unless the
owner says so.

**Stronger combat is the goal, and it must look it:** "your goal should be stronger combat. it's ok if combat is
weaker compared to the old model, but the gameplay should be 'fun' to watch, and look like skilled combat, not just
completing a run because it accidentally got lucky." Hits taken a room visit is the number that says it.

**`d2first` is out of this program:** "Once d3 is completely, completely done, I MAY want a single (separate) model
that does d2 first. The d3 first winning model shouldn't be trained on d2 routes." It stays as it is, 34 networks.

**Contact features are allowed** (monsters' and shots' paths extended a few frames by arithmetic on the current
state, no emulator step): "This is fine, hopefully you do this with projectiles too."

**It is an experiment, in many rounds, not one attempt.** The owner: "You may change the observation drastically,
you may try multiple neural network architectures, you could try many different techniques (lstm, rnn, graph
networks, etc), or not try those things if you don't think they would help. You have full access to the game's ram,
you could normalize some of the data with embeddings or small calculations so the model doesn't have to. [...] This
is going to be multiple rounds of experiments, this is NOT a single round. Plan accordingly."

**What counts as one model:** one shared body with several heads, the head chosen by a deterministic rule (the
route step), is one model. No learned head selector ("overkill").

**The scale to think in:** the world-record holder has 252 completions in 13,135 attempts (1.9%, resets included).
That is the order of magnitude for a golden path, and for the later goal of beating Ganon from the start in one
unsegmented run.

**The route** is corrected (below) and replaces `d3first` under the same name. Both changes need retraining, so
they are done together, in whatever order works.

**Standing rules that still hold** (AGENTS.md): clips on only; no potions, no CONTINUE; the planner (the emulator's
look-ahead at play) is reported beside, never the result; look-ahead in training is fine, never at inference; no
route hands items over; a required wall with no bomb in hand is still given one and counted ("cheated bombs are
fine (for now) to make up the shortfall... but only as a way to progress going to rooms"); bombs are never bought:
drops only; things answerable by playing the game are found in the emulator and its RAM, not asked.

## Track R: the corrected d3 route

The owner's order: Dungeons 3, 4, 1, the heart at 0x2c, the bomb-wall rupees at 0x2d, Dungeon 2, the 100 rupees at
0x0f, Dungeon 5, the recorder to Level 4's screen, the raft south to 0x55, **the wrong warp out of 0x55** to the
meat under the Armos (0x34), the arrows (0x44), Dungeon 7 (wood sword, 9 containers), the world wrap at 0x60, the
hearts at 0x5f and 0x2f, **the wrong warp out of 0x3f** into the graveyard, the magic sword (12 containers),
Dungeons 6, 8, 9. Kept although speedruns skip them: 0x47's heart and the red ring. No blue candle (the red one
from Dungeon 7).

What the Nov 21 investigation found (reports kept by the lead; the facts go into the route's documents as it is
built):

- Both wrong warps are one mechanism, read from the disassembly and reproduced in the emulator with controller
  input: a whirlwind carries Link over the dock's trigger spot, the dock starts a raft ride north
  (`asm:Dock_Update` checks only his position), and the game scrolls up instead of east from the whirlwind's
  stand-in room (`Whirlwind_PrevRoomIds`): Link lands one row above the stand-in room of the level the whirlwind was
  heading for. 0x34 is Level 4's; the graveyard's 0x31 is Level 7's.
- The destination depends on the whirlwind index each play steps. From the route as reconstructed (two plays facing
  down out of Level 5 to reach 0x45): at 0x55 "north, a one-frame turn west, play again" lands at 0x34 and "north
  twice" lands at 0x26; at 0x3f two plays north land in 0x31 and one lands on the start screen. The owner's memory
  of the counts may be of another index: settle it by search in the emulator, not by asking.
- Tolerances: Link's Y in the water must be exact; any X west of the dock works (the owner clips right beside the
  dock at 0x3f); the second play has a 50-64 frame window; every miss leaves him alive at a recoverable screen.
- Hearts 7 / 8 / 9 at Levels 2 / 5 / 7 (today 10 / 7 / 11); keys never cheated; rupees 10 + 30 + 100 against the
  meat (60) and the arrows (80), with 0x3d's 30 as the only cushion; bombs 26 walls plus Manhandla and Dodongo
  against 28 from fixed pickups.
- Of 891 start states 483 are unchanged, 72 differ in an exit or the index, 249 hold other hearts or items, 9
  depend on the 0x3f landing, 78 go; about 300 are rebuilt, all between 0x2d's cave and the magic sword.

- [x] **Recorder plays for the whole route, found by search** (2.0.0.5; round36.md "The recorder chain"): out of
  Level 5 two facing S; 0x55 N, a one-frame turn W, play; 0x3f N twice; out of Level 6 turn up, then two; out of
  Level 8 one S. The owner's counts fit no index history of this order.
- [x] **Engine** (2.0.0.5; "How the warps are played"): the dock warp is a macro the engine plays from the dock
  step (MOVE S, CLIP N, RECORDER), with a sword strike S from the dock at a monster below and a wait while a shot
  flies; a wrong landing ends the run `dock-warp-missed`.
- [x] **Route data, states, audit, tests, documents** (2.0.0.5): 292 steps, steps 90-216 rebuilt on three chains,
  861 states audited clean, `docs/routes/d3first.md` and `d3first-walks-c.md`.
- [x] **Baseline** (2.0.0.5; round36.md "The baseline", "The golden race"): every segment 1000 runs (300 where
  marked), the route from the start, a golden race from power on that won three of three (411 runs).
- [x] Decided by the lead: 0x3d's 30 rupees taken; 0x2d to Level 2 by 0x3d, 0x4d, 0x4c; the turn at Level 6's
  exit stands (the search found none without it); 7:0x1b stays a crossing.

What the baseline leaves open, in the order it costs the route (runs lost per 1000 entering the room; nothing has
been retrained on the corrected order):

- [ ] **The leg from Level 7 to Level 6, 22.4%:** ow-lynel at 0x50 (644: leaves N into 0x40, its `d2first` way,
  and bounces between 0x40 and 0x50) and 0x60 (320: back E instead of the wrap W). Training on the corrected order
  (explorer tracks made in round 36).
- [ ] **Out of Level 6 to Level 8, 32.0%:** ow-lynel at 0x22 (680) walks back into Level 6 instead of turning up
  and playing (180 then die to 6:0x78's Wizzrobes).
- [ ] **Dungeon 7 at 9 containers and 2 bombs, 13.1%:** goriya at 7:0x1b (689, against 149 at 11 with the bag),
  7:0x1a 116, 7:0x19 70; boss-digdogger 7:0x1c 161.
- [ ] **The leg from Level 5 to Level 7, 78.9%** after the dock change (0x55 still loses 85, 58 of them rafted
  back to the bounce limit).
- [ ] **ow-leever at 0x0b's whirlwind out of Level 5** (86: two plays facing S now; 62 runs bounce between 0x0b and
  0x1b).
- [ ] **The whirlwinds at 0x0b and 0x22 could be played by the engine as the docks are** (a macro from the step
  that plays the search's sequence): it would take the network out of ow-leever's 86 and ow-lynel's 680 (not
  measured: what the networks do before the step stays theirs).
- [ ] **Cheated wall bombs, 0.76 a run in Dungeon 7** (7:0x1a 0.75), the largest count on a dungeon the reorder
  changed (`d1`'s 1.00 at 1:0x53 is in a segment it did not change).
- [ ] **Bombs as a weapon only in dungeon rooms the route must clear** (the owner, Nov 22: "The agent wastes bombs
  in rooms I wish it didn't. For example d3 0x69. In dungeons, link should only use bombs in rooms it somehow needs
  to clear. [...] it's fine to use bombs in 0x59 because we have to clear the room to open the south door, in 0x69
  it needs to slide past the darknuts to get the raft, then out of the room (I use sword-hit an enemy to make them
  pass-through in this room, not kills with bombs)."). Today the rule is by family (`items: {bombs: free | required
  | tenth | never}`) and the route's `bomb_reserve`; the owner's rule is by room: BOMB as a weapon is offered in a
  dungeon room only while the step's objective is to clear it (a shutter the monsters hold, a treasure or block
  that appears on the clear, a boss), and at a wall the route leaves through; nowhere else in a dungeon. To build
  on branch `r37-rules` (no training is interrupted for it); it also saves bombs for the walls that are
  cheated today. The rooms where the golden replay's network threw bombs that this rule forbids: scouted Nov 22,
  the list is in round37.md "Bombs thrown where the route needs no clear".
- [ ] **Rooms where the owner forbids bombs but for the wall** (Nov 22: "some rooms I want to explicitly say 'don't
  use bombs in this room just because the owner says so' [...] 0x5b is a great example, the point of fighting the
  darknuts in that room is to get the bomb drop, wasting the bombs to kill them is not the right play. BUT link
  still needs to bomb the wall to get through there. So no bombs except the 1 to blow up the wall."): a step mode
  in routes.yaml; 3:0x5b is the first, more will come from the owner's watching.
- [ ] **The floor for fighting a room for its bombs is the bag** (the owner, Nov 22, a blanket rule): the route
  fights a room for its bomb treasure while Link holds 7 or fewer (it was 5 at 8:0x3f, 2 at 7:0x1b on `d3first`,
  the last pass of 3:0x5b by its own rule). **7:0x1b's six Blue Goriyas** are then a wood-sword fight again whenever
  the bag is not full (round 34 had made it a crossing because the fight killed half the runs). The owner: "that's
  tough. I want you to measure the impact, and you have the ability to skip that fight for bombs if it kills too
  many runs [...] Speedrunners skip that fight, but speedrunners are better at 'forcing' bombs than our model [...]
  Feel free to decide this room either way, but I want you to try to improve the model's combat before calling it
  'skip'." So: measured both ways with the rules; combat first (the contact inputs with Goriyas and their
  boomerangs, the threat lane below, the hit head); the lead then decides, tells the owner with the numbers, and a
  skip is recorded as an option to revisit if too many bombs are cheated later.
- **Over all three: the 10th kill's bomb is always allowed and encouraged** (the owner, Nov 22: "we must ALWAYS
  allow bombs and deeply encourage them if we are on the 10th kill force drop, no matter the room or enemies, that
  remains true"): the forced drop of four is the route's bomb supply.
- **`d2first` gets the same bomb rules later** (the owner: "d2-first is also in-scope for bomb rules for when we go
  back and work on that single-model. Don't waste time on d2 first until we've completed this work on d3-first,
  entirely."): the rules are route-level switches, off for `d2first` until then.
- [ ] **Two route changes from the owner's watching (Nov 22), on branch `r37-rules`:** "Drop the fight with the
  goriyas in d2 0x5e, that's only for d2 first. (It's high bomb chances on start of a d2 run, but needless
  heart/time wasting in the d3 first route). Overworld: drop the bomb for rupees in 0xd2, keep the armos rupees in
  3d (losing too many hearts in that d2 fight, and it's redundant with 0x3d)... we may have to cheat rupees more,
  but that 1 30 secret + the 100 secret + drops is the intended route." So on `d3first` 2:0x5e is crossed, and the
  bomb-wall cave of 0:0x2d (the lead's reading of "0xd2": to confirm with the owner) is left out; the purse is then
  10 + 30 + 100 and drops against the meat and the arrows, with the shortfall cheated and counted.
- **An input to try, the owner's idea (Nov 22): a shooter's threat lane.** "maybe a projectile-using enemy with no
  current projectile always has a 'danger zone' in front of them unless their projectile is in air... at which
  point we use that projectile." And: "only maybe 3 tiles (ish, you measure) in front of the enemies with
  projectiles. IE there's no reason to be scared of the tile in front of an octorok all the way across the screen
  [...] Perhaps based on the speed of the projectile." Per shooter that can fire, along its facing: the frames a
  shot fired now would take to reach Link at that shot's speed, cut off where he could still leave the lane, and
  for his five motions whether he walks into or out of it; off while its shot flies (the shot's own contact inputs
  carry the danger then). Schema 6's `in_line` is the bare form; to be ranked on the hit bench (hits by shots fired
  after the decision as their own slice) and carried into an arm if it helps.
- [x] **A Bubble's touch is not penalised** (done Nov 22, round38.md "A Bubble's touch": 1.4 touches a visit
  to its rooms, about 0.075 hearts and 90 frames a touch; `penalty-sword-blocked` 0.05; counted beside hits) (the owner, Nov 22: "link does get hit by the blue bubble often, does
  he get penalized for that directly?"): it does no damage, so neither the health penalty, the yardstick's hits nor
  the hit head's labels see it. Branch `r38-bubble`: what a touch costs, a penalty a touch, touches counted where
  hits are.
- [ ] **A room's bombs left on the floor** (the owner, Nov 22, 8:0x3f: a drop lifted Link from 2 to 6 and he
  ignored the room's bombs): the objective and the shaping must want bombs on the floor whenever he holds fewer
  than 8, and the decision to fight a room for its bombs holds for the visit. On branch `r37-rules`.
- [ ] **Aimless walking** (the owner, Nov 22: "the opening room link walks back and forth a while before grabbing
  the sword. In 0x4c [Dungeon 8] in the golden replay, he walks back and forth after the bombed wall is opened
  before going north. While these aren't run killers or anything, they are unaesthetic to watch [...] you could
  create an 'aimless walk' score, where there are no enemies and all link has to do is go to the cave, door,
  heart, triforce, etc (and isn't being actively pushed to an item... walking towards a heart that disappears
  shouldn't count). Some amount of missed steps is totally fine"). **The score is built and measured** (round38.md
  "Aimless walking": `aimless_frames` and its companions in every chain room record, visit, rollout, bench and
  room-stats; `penalty-aimless`, default off). Both rooms are confident loops that only sampling breaks (golden
  replay: 457 and 1,282 aimless frames; today's medians 493 and 3,723 a visit, 0x4c 7% stuck). Act mode alone:
  `relax:4` trims the opening about 14%, nothing helps 0x4c. Next: the first training experiment written there
  (`ow-start` and `pols-voice` fine-tuned with `penalty-aimless` 0.002 and 0.005 a frame against 0); below combat
  (the owner).
- [ ] **Later, the owner's stated goal:** a win without a cheated bomb. Drops only; the walls' shortfall is counted
  until the bomb economy closes it (old backlog "A win without a cheated bomb").

## Track M: fewer networks, toward one

**Where it starts (verified Nov 21):** 51 networks play the route (34 families, 17 of them with a second,
wood-sword run). All are the same architecture (about 866,000 parameters: a room embedding, a 1-layer entity
transformer, a 3-layer tile conv, two 256-wide towers, factored kind/direction/duration heads) on the same
observation (schema 6, every family with `objective` and `target` on) with nearly the same reward settings. The
observation already holds what tells situations apart: the room, each monster's type, the objective, the sword.
What differs between "specialists" is mostly the training task (rooms, start states, demonstrations, which items
the mask allows), not the network.

**Target shape, a hypothesis to test:** one larger shared body; one action head, with dedicated heads only where a
boss needs them, chosen by the route step; a hit-prediction head inside the network whose output feeds the action
choice (the owner's idea: the guard as a piece of the network), trained on labels made in training (what happened
to the action taken; look-ahead for the others), never looking ahead at play.

### Infrastructure (blocks the experiments)

- [x] **A lab that can try any network** (on `main` since Nov 22, `docs/lab.md`; its extensions for memory, the
  hit head's labels and sampling by need are on `r36-lab` until verified): a training run over several families at once; teachers (today's models)
  as an imitation target that fades; experimental architectures playable by the same evaluations and chains
  without a hand-written Rust port each (the Rust actor stays the production path; winners are ported); old and
  new architectures loadable side by side.
- [x] **A yardstick for experiments** (`triforce/lab/bench.py`; today's 51 networks: 91.9% success, 0.241 hearts
  lost an episode, 0.697 hits a room visit, 20 minutes): a fixed, cheap suite (representative rooms at the hearts runs bring, with
  success, hearts lost and hits a visit) and a full tier (segment chains, the route from the start, the golden
  path), so two networks are compared on the same thing in an hour, not a day.

### Experiment queue (in order; each names the decision it feeds)

1. [x] **Cross-play: how specialised are the specialists?** (round36.md "Cross-play", Nov 22.) No merge is free:
   the best stand-in network costs 7-8 points on a segment (ow-zora for ow-tektite on `post-d1`, goriya for
   dodongo-crossing on `d7`), and the matrix predicts the chain room by room. How general a network is follows how
   many rooms its family trained in (ow-zora and ow-moblin play the other overworld tasks at 0.87-0.88 of their own
   networks; goriya, darknut and stalfos-keese-gel 0.70-0.74 of the dungeon rooms and 0.80-0.84 of the overworld;
   the narrow families 0.09-0.21). What does not transfer is a room's own objective (wrong exits, special actions
   at one exit) and boss fights. **Decided from it:** merging means training with every family's rooms in the mix,
   never picking a best network; the first groups are the overworld (10 families and the item cellars), the
   dungeon rooms (15), the bosses as one body with a head each, then overworld and dungeon rooms in one body.
2. [x] **What the observation lacks for seeing a hit coming** (round36.md "What the observation lacks",
   `r36-hit`, Nov 22). Not information: trees on the flat observation read the look-ahead's verdict better (0.76 /
   0.68 precision = recall) than a policy-shaped network does (0.66 / 0.54 at 10 epochs, 0.73 / 0.66 at 60); raw
   RAM bytes hurt. It is representation, then data and training time: contact geometry per entity (the touch rule
   extrapolated along straight lines) adds 0.04-0.10, more data helps steeply, hits land in an action's first
   frames. **Feeds:** observation v7's contact block and Link-relative pixel scale (its §4), a per-entity hit head
   (§5), and the supervised hit bench as the cheap screen for representations and shapes. **The screen's answer
   (§7-§8, Nov 22):** on that bench (precision = recall on the planner's / the model's own draws; today's shape
   0.66 / 0.56, trees 0.76 / 0.68) each monster's position relative to Link in pixels on the touch test's scale
   (`rel`, ten columns a monster) with **Link as a token attending to the monster tokens** (no mean pool) reads the
   hit at 0.86 / 0.81, 0.87 / 0.81 with the contact columns, 0.89 / 0.85 on 2 million decisions; the Link token
   alone on today's observation 0.85 / 0.78; it holds on Gleeok (0.77 / 0.68) and now moves Darknuts (0.85 / 0.78).
   No gain from wider or deeper towers, a tile patch round Link, past observations, the action as an input. A
   learned veto with it would catch 26 of 32 hits per 1000 decisions for 6 wrong refusals (round 35's guard: 15
   for 17). Labels: the look-ahead's verdicts on untaken actions are needed (the taken action's outcome alone:
   0.51). Nothing from it has played yet.
3. [ ] **Capacity by imitation alone** (round36.md "One network by imitation" on `r36-merge`; in progress): ONE
   network of today's size, copying the 51 on 1.49 million of their decisions, keeps 98.1% of their success on the
   yardstick (−1.6 ± 0.3 points; the worst ten slots 88%); what it does not copy are fights (Gleeok with the wood
   sword −32, Dungeon 8's Darknuts −23, the wallmaster −12.5). Open: the larger sizes, the room identity, the data
   curve, and that network on the route's chains and in a golden race.
4. [ ] **The wood-sword pairs first** (51 to 34; round36.md "The wood-sword pairs" on `r36-merge`): on the
   yardstick darknut, goriya, zol and ow-octorok hold, wallmaster is not shown to, boss-gleeok fails (−24); but
   Dungeon 3's chain loses 4.7 ± 1.8 with the merged darknut and zol (the rooms that already kill most): **the
   yardstick shortlists, only chains decide.** Open: teachers kept on until both parts match, rows toward the
   hard part, judged on the chain.
5. [ ] **Groups** (34 to a handful: overworld, dungeon rooms, bosses, utility rooms, or what cross-play says), then
   the groups into one body.
6. [ ] **The hit head in the network:** as an auxiliary target only; as an input to the action choice (its own
   labels only, so it stays an honest predictor); against the planner on the same rooms. The lab can train it
   (`--aux`, `--try-others`: four other actions tried on a quarter of the decisions cost no measurable time); a
   first smoke on today's shape was too short to read (0.10 at the veto rate). Next with the shape and inputs
   experiment 2 chose (below).
7. [ ] **Observation v7** (specified by experiment 2: round36.md §4, §8): per monster ten `rel` columns and 23
   contact columns from the RAM (the game's touch rule extrapolated straight and stopped at walls: arithmetic on
   the current state, no emulator step; for the owner to confirm it counts as a "small calculation" and not
   look-ahead), as a NEW observation key so today's networks play unchanged; the room identity kept and also
   tried without.
8. [ ] **Architectures:** a wider and deeper body; Link as a token attending to monsters and to tile patches; a
   graph over entities; memory (GRU/LSTM) where something is hidden or timed (Wizzrobes, Ganon, a Like-Like's
   hold); a head a boss against one head for all.
9. [ ] **Training method:** imitation of teachers on the student's own states then PPO with the imitation fading,
   against from scratch; value normalisation across tasks; prioritised sampling across tasks (`plr.py`); the
   look-ahead as a teacher at full weight (old backlog "Idea 3: search distilled into the network"; round 34's
   0.05-0.15 imitation of the planner's rows did not take).
10. [ ] **One network on the corrected route:** the merged model(s) trained on the new states; a golden path with
    them; then single runs at the world-record scale.

Each experiment is logged in a round document (`round36.md` on) with its setup, numbers and what it decided; an
idea tried and dropped is marked here with the round, not deleted.

## Carried over from the old backlog

- The learned guard as built (branch `r35-guard`): superseded as a design by the hit head (queue 6); its label
  logging and data are what queue 2 starts from.
- Round 35's stopped Dungeon 3 trainings and the ledger of where Dungeon 3 is lost (`backlog.old.md` "Where the
  work stopped"; round35.md): the ledger still describes the rooms; the runs are specialists and are not continued.
- What loses each route today, per room (`backlog.old.md` "`d3first`, the main route", "The whole game on
  `d2first`"): the yardstick's rooms are chosen from these lists.
- The unused `grant:` vocabulary; one like-like run for both routes; the per-route tables never refitted;
  `select-checkpoint --segment` on `d3first`: tool and data debts, picked up when the code they touch is touched.

## Questions for the owner (next check-in)

- None blocking. The first results that will want steering: the cross-play matrix (how far merging goes for free),
  the oracle test (what the observation should become), and how the wrong warps are offered to a network (a macro
  the engine plays, or the plays themselves).
