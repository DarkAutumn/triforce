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

### The experiment (not run yet: the queue was stopped)

zol on `d3first` (it plays 3:0x6b). Baseline arm: Fresh40's zol@d3first from scratch without the switch (branch
`r40-fresh`: `triforce train --specialist specialists/d3first/zol.yaml --parallel 8 --iterations 1954
--checkpoint-every 122 --seed 40N --stop-success 1.01`, seeds 401-403, 8.0M steps). Arms of this branch, the same
recipe adding only the switch (`--specialist experiments/r40/zol-fireline.yaml`): from scratch, seeds 401-403; a
continuation of today's zol@d3first policy (`--load models/zol/r34d3-zol-s341`) with the switch. Measured in zol's
rooms (`scripts/experiments/r40_fireline.py`: far kills per sword kill, unfired lined-up decisions, hits and hearts
lost a visit, frames) and on the `d3` chain (1000 runs paired, the frozen 2.0.0.12 team with only `zol@d3first`
swapped by `--checkpoint zol@d3first=RUN`, the switch arms with `--specialist-config
experiments/r40/zol-fireline.yaml`). The 200-run identity proof (today's teams, switch off, before/after) and the
arms wait for the new queue.
