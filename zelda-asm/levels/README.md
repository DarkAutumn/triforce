# Levels (first quest)

One document per dungeon, for an agent that has to get Link through it: what to bring, what Link gets and what each
item changes from then on, the monsters and the boss, the route room by room, the optional detours and the rules that
bite there. Level numbers are the internal ones (`CurLevel`); in the first quest the game shows the same number.

| Level | Item(s) | Boss | Must bring (beyond a sword) | Document |
|---|---|---|---|---|
| 1 | bow (cellar), wooden boomerang | Aquamentus | nothing | [level-1.md](level-1.md) |
| 2 | magic boomerang | Dodongo (bombs) | nothing (bombs lie in the level) | [level-2.md](level-2.md) |
| 3 | raft (cellar) | Manhandla | nothing (one bomb lies in the level) | [level-3.md](level-3.md) |
| 4 | ladder (cellar), used at once in room `$30` | Gleeok, two heads (sword only) | raft (island entrance); two bombs save a detour | [level-4.md](level-4.md) |
| 5 | recorder (cellar) | Digdogger (recorder) | ladder, bombs | [level-5.md](level-5.md) |
| 6 | magic rod (cellar) | Gohma (arrow) | bow, arrows, ladder | [level-6.md](level-6.md) |
| 7 | red candle (cellar) | Aquamentus | recorder (entrance, Digdogger), food (hungry Goriya), ladder, bombs | [level-7.md](level-7.md) |
| 8 | book, magic key (cellars) | Gleeok | candle (entrance), bombs, bow and arrows (Gohma) | [level-8.md](level-8.md) |
| 9 | silver arrow, red ring (cellars) | Ganon (silver arrow) | all eight Triforce pieces, bombs, bow, keys or the magic key | [level-9.md](level-9.md) |

Each document has the same headings, so an agent can search them: `# At a glance`, `# What Link must bring`,
`# What Link gets here`, `# Map`, `# Rooms`, `# Floor plans`, `# Monsters and boss`, `# Route`, `# Optional detours`,
`# Hazards and rules that matter here`, `# Second quest`.

The blocks between `<!-- generated:... -->` markers (facts table, door map, room table, floor plans) are written from
the ROM by `tools/romdata.py`; edit the text around them, never inside. The rest was written from the code and the
data, and walked in the running game where a section says so (`(trace)`). Levels 1-4 were walked; levels 5-9 come
from the data and the code only.

The second quest uses other level blocks (`data/levelblock-uw-*-q2.md`); these documents do not cover it.
