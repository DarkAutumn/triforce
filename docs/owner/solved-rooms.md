# Solved rooms and your takes (Nov 25)

You asked for your takes to be kept "somewhere durable (along with the other replays I make for you, this may be an
ongoing thing)" and for the solved rooms to be written down so that "we are going to compare at the end of this
full retraining if the agents learned the 'Solved' route or not". This page says where things are, how a take you
play gets in, what the file says, and how the comparison will read.

## Where your takes are

- In git: `replays/owner/<room>/` under the names you saved them with (`replays/owner/5x47/`,
  `replays/owner/5x37/`, `replays/owner/5x64/`; a whole game from power on goes in `replays/owner/game/`; the
  Darknut lesson's three stay in `replays/owner/darknut/`). Plain `.tfreplay` files: `play-zelda.py
  replays/owner/5x47/` plays them.
- `replays/owner/INDEX.md`: one row a take (room, side, date, seed, frames, how it ended, the layout `$524` it was
  entered with and whether that is the one the route brings to that door, the chase flag, the hearts and items you
  came in with, cheats, replay version); for a whole game, the rooms it passes and the frame you were first in play
  in each.
- In the archive: `/backup/triforce/replays/<version>/owner/`, under the same names.
- Nothing in `~/triforce-review/` is moved or renamed: it is read and copied.

**How a take gets in:** only the ones you name ("not all save states I play need saving, only the ones I tell you
about"). Play on `emu-zelda.py` as you do (it saves into `~/triforce-review/played/`), tell the lead which takes to
keep by their file names, and `scripts/owner_takes.py add NAME...` (a queue job) files those: it checks each is
yours and replays exactly, copies it, archives it and rewrites the index; the lead commits it. Takes you saved by
accident or replaced with better ones are left where they are and never filed.

## The solved rooms: `triforce/solved_rooms.yaml`

One entry a room and the side it is entered by: your words (quoted whole, with the date), the lead's description
of the way through from your words and your takes read frame by frame, the layout it is claimed for, what a perfect
pass costs, the takes that show it, what is still open, and the criteria a network's pass must meet to count as
your way. **No cellar is ever solved** (your rule: the Keese move): the file refuses one. **Nothing trains on it**:
it is for the comparison only, unless you say otherwise.

Each entry was also put to the game: scripted inputs over 3000 random phases in the route's own layout, both chase
states (the "phases" below; `scripts/solved_phases.py`).

| Room | Your way, short | What the game says (3000 phases, the route's layout) |
|---|---|---|
| 5x47 from S | Hold N and never let go; a swing or not | Holding N alone: 40% unhurt. With one swing as Link reaches the second row: 96% (every phase in chase state 0, 93% in chase state 1). With two swings, one on the ladder over the middle water and one at the second row: **3000 of 3000**, 123 frames. All your takes were in chase state 0; in chase state 1 the Gibdo spawned on the third row walks E along it into your column as you step off the ladder, which the first swing meets. |
| 5x37 from S | N out of the doorway, W half a tile, N to the wall, E/N out; a swing on the ladder if the Darknut steps in | Your way with one swing as Link steps onto the ladder: **3000 of 3000**, 121 frames. Without the swing: 25%. |
| 3x69 from N (the raft room, in) | E along the top until you can't, S to the stairs' row, E to the stairs; sword the Darknut in the way to pass through | No script reaches your "never": the line alone is hit every time; with a swing whenever a Darknut is in reach, 20%; round 40's rule, which predicts each Darknut's walk, 51%. Your word stands; a take of yours would show the timing. |
| 3x69 out of the cellar | "Leaving is always safe if you head right out" | **3000 of 3000** either way (N to the top row then E then N, or E then N), 77-100 frames. |

**Your question on 5x37** ("I'm not sure if holding pure north is possible with every possible rng"): no. Pure
north at X $78 is never unhurt without the sword (0 of 3000: the Darknut spawned in the middle at X $80 walks
into that column), and unhurt in 1373 of 3000 (46%) with a swing on the ladder; it would save 10 frames (111
against 121). Your take `5x37_S_seed886279` is that: pure north, the swing hits the Darknut's shield, two hits.

The two 5x37 takes you did not name: `5x37_S_seed761531` is your way with the swing on the ladder, then a step E
around the same Darknut near the top (144 frames, unhurt); `5x37_S_seed886279` is the pure-north try above. None of
your 5x37 takes is in the route's layout (it brings `$524` = 2; yours are 0, 3, 4, 4), so the claim for the route's
layout rests on the phases. Three of your seven 5x47 takes are in the route's layout (6); the odd one you had never
seen is 7.

## How the comparison will be read

At the end of the retrain the lead runs `triforce solved compare NEW_TEAM_REPLAYS OLD_TEAM_REPLAYS` on the
new team's chain replays beside today's. A table a team, one row a solved room:

- **Plain facts**: how many passes, how many left by the solved exit, hits a pass, seconds and frames a pass (beside
  yours).
- **The solved way**: the share of passes meeting every criterion, in all layouts and in the route's own (the claim
  is made for that one), and each criterion's own share, so a miss can be read:
  - 5x47: N held on every frame but a swing's (one empty frame between two decisions allowed), no other direction,
    at most two swings, no hit, out N within 136 frames.
  - 5x37: X $70 (within 2 pixels) from the ladder to the second row, never S, at most two swings and each at a
    monster in reach ahead, no hit, out N within 150 frames.
  - 3x69 in: within half a tile of the line (S onto the top row, E to X $90, S to Y $8D, E to the stairs), only S
    and E pressed, no hit, onto the stairs.
  - 3x69 out: only N and E pressed, no hit, out N within 110 frames.
- Your own takes are scored the same way: every take the file names passes (a test checks it); the 5x47 take in
  the layout you had never seen (one hit) and your pure-north 5x37 take do not.

## Today's team, for the record (the baseline)

Today's networks (2.0.0.15, the store's current team) measured the same way before the retrain: 300 chain runs
of Dungeon 5 from its entrance and 300 of Dungeon 3 from its entrance (seeds 0-299, replays kept, on `main`'s code),
the 2.0.0.15 golden race's and best single run's passes, and beside them the 1000-run baseline's records
(`r40-2015-1k`, plain facts only):

| Room | Passes | Out by the solved exit | Hits a pass | Median frames (yours) | **Your way** | in the route's layout | What it misses most | Records: visits, out, hits a visit, died |
|---|---:|---:|---:|---|---:|---|---|---|
| 5x47 from S | 224 | 94.6% | 0.12 | 177 (121-124) | **6.7%** | 7.3% of 150 | N held 6.7% (it steps W and E), within 136 frames 12.5% | 918, 96.4%, 0.16, 3.6% |
| 5x37 from S | 212 | 95.8% | 0.42 | 157 (125-126) | **31.1%** | 28.8% of 139 | unhurt 67.9%, X $70 over the middle 72.6%, within 150 frames 40.6% | 885, 94.5%, 0.47, 4.5% |
| 3x69 in | 292 | 85.6% | 0.66 | 157 (none) | **0.0%** | 0.0% of 265 | on the line 4.5%, unhurt 53.8% | 968, 84.7%, 0.82, 15.3% |
| 3x69 out | 240 | 100% | 0.00 | 98 (77-100 scripted) | **98.8%** | 98.6% of 218 | within 110 frames 98.8% | 767, 100%, 0.00, 0.0% |

Read it this way: today's team gets through 5x47 almost always but not your way (it wanders off X $78 and takes
half again as long); 5x37 your way about a third of the time; 3x69 in never on your line and hurt in half its
passes; out of the cellar already your way. Your own takes score 100% on 5x37 and on the six clean 5x47 ones (the
seventh, the layout you had never seen, took a hit).
