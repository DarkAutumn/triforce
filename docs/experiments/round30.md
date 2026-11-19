# Round 30: the optimisation round over the whole route

## The first quarter: moldorm, ow-lynel, ow-tektite, ow-start (Dungeon 2, after it, Dungeon 1)

Every one of the full route's runs plays the first quarter: from the game start through Dungeon 2, the heart cave
and the white sword, to Dungeon 1's triforce. `r29land-1k-full` (1000 runs, seeds 0-999, clips on, models alone)
leaves Dungeon 1 in 730 of them (step 68: 0:0x37 out of Dungeon 1's cave).

### The cause

Where the 270 runs end before step 68, by family (the room records of `r29land-1k-full`; the planner beside,
`r29land-plan-full`, loses 5 of 300 there):

| family | runs ended | rooms |
|---|---:|---|
| goriya (another lane of this round) | 110 | 2:0x1e 41, 2:0x5e 29, 1:0x44 25, 1:0x23 15 |
| stalfos-keese-gel | 25 | 1:0x33 12 (2 of them the step limit), 0x43 4, 0x72, 0x74, 0x63 2 each, 0x53 1 |
| **moldorm** | **22** | 2:0x3e |
| **ow-lynel** | **21** | 0:0x0a: the first visit 3, out of the cave with the white sword 18 |
| ow-zora | 20 | 0:0x38 13, 0x27 6, 0x28 1 |
| d2-rope | 18 | Dungeon 2's rope rooms |
| boss-aquamentus | 15 | 1:0x35 |
| **ow-tektite** | **13** | 0:0x1a back S, after 0x0a |
| equipment-room | 12 | 1:0x7f (the bow) |
| boss-dodongo | 11 | 2:0x0e |
| ow-octorok | 3 | |
| **ow-start** | 0 | no heart lost in 0x77, 0x1c, 0x1b |

**Dungeon 1's 92 are hearts carried in.** It is entered with 3.31 of 5 hearts (the planner 3.94); every one of its
deaths (goriya 40, stalfos-keese-gel 23, Aquamentus 15, the bow cellar 12) is an entry below 2 hearts. The rooms
that take those hearts are the overworld after Dungeon 2: 0:0x0a (ow-lynel) and, after it, 0x1a (ow-tektite) and
0x27 (ow-zora).

Per visit (models | the planner):

- **2:0x3e (moldorm, `r75-f-b` since round 25)**: 959 visits, 2.63 of 3 hearts in, 0.32 | 0.19 hearts lost, 1.35 |
  0.74 hits, 22 | 0 deaths. Entered at full hearts it loses 0.41 and nobody dies; every death is an entry below 3
  hearts (below 1: 5 of 12; 1-1.9: 12 of 88; 2-2.9: 5 of 247), the hearts 2:0x5e and the rope rooms take. The same
  in `r77b-1k-full` and `r76-1k-full` (the same model). Single rooms (`eval.json`, drawn hearts) 90.7%, −0.17.
- **0:0x0a, the second visit (ow-lynel `r27l-lynel-s81`)**: 874 visits, 4.28 of 5 in, 0.24 | 0.10 lost, 18 | 0
  deaths; entered with 4 or more hearts 0.27 lost and 11 deaths (`main`'s `r74-lynel` in `r76-1k-full`: 0.11, 4).
  The first visit is back (0.34 | 0.19, 3 deaths; `r74-lynel` 0.40). Round 27 §15.3's open lever.
- **0:0x1a back S (ow-tektite `r74-leg2`)**: 4.12 in, 0.30 | 0.14 lost; its 13 deaths are entries below 2 hearts
  (8 below 1): the hearts 0x0a lost.
