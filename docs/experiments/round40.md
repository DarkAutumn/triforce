# Round 40

## Firing lines: a fight's targets at full hearts (branch `r40-fireline`)

The owner's note (backlog "Beams at range, not melee"): Link walks up to monsters and hits them in melee even at full
hearts with beams available (3:0x6b: entered full 93% of visits, 83% of sword kills in melee; round39.md "Full hearts,
still melee"). The lead's reading: in a fight the objective's targets are the monsters' own tiles
(`crates/zelda/src/objectives.rs`, the FIGHT branch's `enemy_tiles`), so the movement reward (PBRS) pays every step
closer and the observation's `target` input points at the monster; the rewards teach melee even from scratch.

### The switch (built; off by default)

`reward.firing_lines: true` in a family's yaml (`env::config::RewardConfig`, default false, so every family's inputs
and rewards are as before unless its yaml says so). While Link holds a sword and full hearts (within half a heart:
`zelda::beams::shoots_at_full`, the game's own test; a flying shot does not matter, so the targets don't flicker
between shots), a fight's PBRS and `target` destinations are the **firing-line tiles** of the monsters the objective
fights (`zelda::beams::firing_lines`, labelled `Role::FiringLine`, a replay's TARGETS `firing-line` with the monster's
slot and type); dropped items join them as before. Short of hearts, outside a fight, with no line, or when Link can
walk to none of them, the targets are the monsters' own tiles as today (`env::objective::firing_line_aims`, used by
the env, the chain's observation and reward, and the replay's TARGETS alike, so training and play see the same).

A firing-line tile is a tile of Link's corner (tile aligned) from which a shot pressed toward the monster meets it
where it stands now, chosen from the game's numbers:

- **In line:** the shot's hit point, (6, 8) from its corner fired up or down, (8, 6) fired sideways, passes within
  12 pixels of the monster's middle across the line (asm:Collision_CheckThinWeaponVsMonster_Thresholds; the same test
  `beams::shot_meets` projects): three tile columns or rows a monster.
- **At range:** the monster at least 48 pixels ahead along the line (`FIRING_RANGE` = the sword's reach to a
  monster's corner, 32, `MELEE_REACH`, plus the 16 pixels ahead of Link the shot is made, `SHOT_AHEAD`): the shot is
  made between Link and the monster and meets it out of melee range even as it closes during the 12-frame swing; and
  no monster of the room within melee reach (32 pixels, corner to corner) of the tile. The touch test (middles within
  9 pixels, asm:Link_CheckMonsterTouch_NoReset) is well inside that.
- **The line is always clear:** a sword shot flies over every tile and stops only at the room's bounds
  (zelda-asm/items.md "Shared rules", asm:Room_BoundDirHorizontally/Vertically), so blocks between Link and the
  monster are no obstacle. This is the point in 3:0x6b: the Gels and Keese behind its blocks are shot across them,
  not walked round to. (The assignment asked for "a clear line (no wall or block between)"; the game says there is
  none to check.)
- **Where Link can fire:** his corner inside the room's floor (no doorway; inside asm:Link_FilterInput's bands where
  the sword's button is read), his feet (the row below his corner's) on walkable tiles; a Darknut only from the sides
  its shield doesn't face (`beams::shootable`). Not Ganon (his phase's tiles stay), not Gohma (a stand tile, no
  monster label).

Tests: `zelda::beams::tests` on hand-built rooms (the lines of a Gel in an open room, exactly; blocks between kept,
a block underfoot removing the two tiles standing on it; a Darknut's shield; another monster's melee reach) and
`env::objective::tests::a_fight_at_full_hearts_steers_to_the_firing_lines` (a hand-made 3:0x6b RAM: the PBRS targets
are the lines, Link a step from one, the TARGETS record `monster` off the wavefront and `firing-line` on it; the
switch off or one heart short leaves the objective untouched). The experiment's yaml:
`experiments/r40/zol-fireline.yaml` (`specialists/d3first/zol.yaml` plus the switch).

On the real room (`d3first_3_6bs_0`, the zol@d3first yaml with the switch on and off, the same seeds, Link walked
down the `target` input's shortest way; no job): resets drawn at full hearts (seeds 1 and 2) steer him to a firing
line one tile from the door (distance 1, then 0: he stands on it) where the switch off sends him 6 tiles up to a Gel;
resets drawn short of hearts (seeds 0 and 3: 1.5 and 2 of 3) walk the same tiles with the switch on and off.

### The switch off changes nothing (Nov 23)

The `d3` segment (`--from 3_7cs_0 --stop-at step:24`), 200 runs, seeds 0-199, the 2.0.0.13 team, `--act model`, once on
main's module and once on this branch's (both on main's `models/`): 77 of 200 arrive on both; every run's result,
reason, steps and frames equal (1,870,702 frames each), and every room record equal but for the decisions'
`surprisal` and `act_entropy` sums, which differ by at most 4e-5 (the GPU's float noise; the actions themselves are
equal). `target/r40f/id-main-d3.jsonl`, `id-fireline-d3.jsonl`.

### The experiment

zol on `d3first` (it plays 3:0x6b), every arm three seeds (401-403). From scratch WITHOUT the switch is Fresh40's
`r40-fresh-zol-s401..403` (branch `r40-fresh`; not trained again here). This branch: from scratch WITH the switch, the
same recipe and budget (`triforce train --specialist experiments/r40/zol-fireline.yaml --parallel 8 --iterations 977
--checkpoint-every 61 --seed 40N --stop-success 1.01`, 4M steps); continued from today's zol@d3first policy
(`--load models/zol/r34d3-zol-s341 --iterations 244 --sil-coeff 0.1`, 1M steps) WITH the switch, and the same
continuation WITHOUT it (`specialists/d3first/zol.yaml`), the fair base for a continuation. Measured in zol's rooms
(`scripts/experiments/r40_fireline.py`: far kills per sword kill, unfired lined-up decisions, hits and hearts lost
a visit, frames) and on the `d3` chain (1000 runs paired with the promoted team, only `zol@d3first` swapped by
`--checkpoint zol@d3first=RUN`, the switch arms with `--specialist-config experiments/r40/zol-fireline.yaml` so the
target input they were trained on is the one they play with).

### First arm: the continuation with firing lines, seed 401 (Nov 23)

**Withdrawn (a recipe fault):** this arm (and a scratch run stopped at 3.0M steps) trained in a worktree without
`triforce/states/harvested`, so its 25% harvested resets had no states (run.json `harvested.states: 0`), unlike the
lineage and Fresh40's arm. Kept as `models/zol/r40-fireline-*-s401-noharvest`; retrained with them. The numbers
below are that arm's, for the record only.

`d3` 1000 runs, seeds 0-999, paired with the promoted 2.0.0.13 team (`target/r40f/base-d3-1k.jsonl`,
`cont-with-s401-d3-1k.jsonl`; `scripts/experiments/r40_fireline.py records`). The rate decides (AGENTS.md "What
decides, in order"): **39.2% -> 32.1%, -7.1 ± 2.1 points, a real loss.** The rest, beside it: it shoots (shots a
visit in zol's rooms 2.15 -> 3.41, far share of sword kills 18.3% -> 46.8%, 3:0x6b 18% -> 57%; lined-up decisions not
fired 64% -> 39%), and it is slower and hit more (hits a visit 0.139 -> 0.260, hearts lost 0.077 -> 0.169; frames to
leave 3:0x6b +336 ± 22, to the D3 triforce +523 ± 215, 8.7 s; deaths in 3:0x49 5 -> 57), and loses the beams sooner
(rooms cleared with them 5.62 -> 4.73, -0.89 ± 0.20; 3:0x6b entered with them 89.3% -> 81.0%, first lost in 3:0x7b
107 -> 191 runs, where this arm shoots less). Not a verdict on the switch: one seed, a continuation, and the base
carries zol's hit head (`r38head-fam-w4`) where the continuation is the bare policy; the fair bases (the
continuation without the switch, the bare policy `r34d3-zol-s341` alone) are queued, and the replays of both chains
to see where the time and the hits go.

### A second variant (READY FOR REVIEW, not trained)

The lines replace the monsters' own tiles, so the path search leads to the nearest line tile even when a Gel stands
next to Link (a line is at least 48 pixels from its monster: walking back from melee to shoot). `reward.firing_lines_melee:
true` (beside `firing_lines`; off by default) keeps the monsters' own tiles as targets beside their lines: the
wavefront then leads to whichever is nearer, melee at hand, a line when the monster is far. Test: Link two rows below
a Gel is 3 steps from the line below it and 2 from the Gel; with the variant he is led to the Gel
(`a_fight_at_full_hearts_steers_to_the_firing_lines`). Yaml: `experiments/r40/zol-fireline-melee.yaml`.
