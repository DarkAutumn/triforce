# How the owner fights Darknuts (Nov 24)

The owner played the Olive Garden (5:0x05, six Blue Darknuts) in `emu-zelda.py` to show it. His three takes are in
`replays/owner/darknut/` (play them with `play-zelda.py replays/owner/darknut/`; read them frame by frame as
docs/architecture.md "Reading a take" says). His words are quoted whole; below them, what the game's code says
about each rule and what the takes measure. "All darknuts are fought the same [...] the same logic applies every
time you fight darknuts."

The takes were made with `--hearts 2` before 2.0.0.16's fix: Link holds 2 hearts and an empty one, so the hit in
`dont_chase` leaves him standing at none.

## The owner's rules, in his words

### Don't chase darknuts (`blue_darknut-dont_chase.tfreplay`)

> It doesn't matter if it's in their code or just a quirk of all enemies, but darknuts you chase directly will often
> turn around straight at you. Since they have a shield you can't sword them to gain invulnerability, that will
> click off their shield and they will turn straight into you. The dont_chase replay shows this. You never try to
> get to a darknut moving away from you on your row by following them directly.

### Melee: play with anticipation (`blue_darknut-play_with_anticipation.tfreplay`)

> To fight darknuts in melee you always work one tile from them shadowing them in parallel. If they are walking E,
> you are one tile north or south of them (closer to a 1.25 tiles N or S) also walking E. For whatever reason,
> darknuts won't do a 180 if you are at a 90 degree angle to them, and if they did that's a safe move for you. They
> will most of the time either continue straight or turn in the direction you are in (occasionally the other 90
> degrees but that's also safe for you...just don't chase on their tile line moving away).
>
> As you can see from the play_with_anticipation. I'm shadowing that darknut south of him. When I hit I IMMEDIATELY
> start moving again in the same direction he is, and I try to be 1.5 tile away. If he turns towards me I can either
> keep moving the direction he was going (hopefully passing by him or through him since my sword hit).
>
> Let's talk through this walk for the first few steps:
>
> He goes E, I go E trying to stay 1.5 tiles S of him, hitting him. He gets close to the east wall, I know he will
> stop and turn and MOST of the time he'll turn S. So I anticipate he will and sword towards the wall south of where
> I think he'll be (I miss, my positioning was a little off, but it was the right idea...if he turned north I was
> safe, and could reposition to shadow him moving west as he hit the north wall and turned...not following from
> behind.)
>
> 1 step south and he's immediately moving W, then he makes an immediate S turn but again I'm safe (1 tile away). I
> continue to sword him moving south, moving in parallel till he hits the wall and goes W again. At the end of the
> replay when he's as far south as possible moving W, he makes one more turn N, and as you can see I hit him and
> kept moving W in the direction he was going avoiding him touching me because I was 1.5 tiles away and already
> moving W as he turned north.
>
> Anticipate his movements and stay 90 degrees.

### With bombs (`blue_darknut-bombs_with_melee.tfreplay`)

> If you have bombs, you always want to hit 2+ darknuts with one bomb. Just hitting a single darknut with a bomb is
> mostly a waste. The trick here is "shadow link". If you are on the edges of the screen or south-ish the darknuts
> scatter, but for whatever reason if you stay north of the block, and in the 2x5 tile area north of it (not the top
> row, no further than 2 squares from the block in EW range) the darknuts will start to clump up right on top of
> you. This is dangerous, but SUPER important for bombs. Take a look at the times I'm in that range in
> bombs_with_melee.
>
> My goal here is: do not chase, get them to clump up and anticipate where they are moving based on their patterns.
> Most of the time they continue walking forward (if I'm parallel or ahead of them) OR turning 90 degrees in my
> direction.
>
> bomb 1: I'm in the right zone, and two are pathing so I place a bomb parallel then walk the same direction they
> are to try to get them to not turn. Then I head back to the middle to get the darknuts to clump again.
>
> bomb 2: same deal-ish. I take melee swings where they are offered, but with a lot hovering around I'm mostly
> trying to position them for a bomb.
>
> Bomb 3: I get hit here for the only time in the room, and the bomb might have been better placed, but I still hit
> two of them.
>
> bomb 4: I whiff this one. They were clumped up, but took an unlucky turn. Now I have some opportunity for melee on
> one, so I do the same as the melee section above since I can't hit multiple.
>
> bomb 5: I've got two moving in the same direction, I drop the bomb. I then stop moving because a bomb behind a
> darknut hitting their back will shoot them forward. They both happen to die, but I'm anticipating them rocketing
> past me so I can't keep walking west.
>
> Now neither line up for me, so I'm back to my melee strategy, and never follow behind them.
>
> bomb 6: I catch both of them after having meleed for a bit. I don't need the last two, I can just use the melee
> strategy to finish him off.

## What the game's code says (the lead, from `zelda-asm/`)

It is in their code, and it is one rule for all three.

- **How a Darknut turns** (`asm:Wanderer_MoveAndTurn`, zelda-asm/enemies.md "Darknut"): at each 16-pixel grid
  point, with a chance of 129 in 256, it turns TOWARD its chase target if the target is within 8 pixels of it on
  one axis; otherwise it turns to a perpendicular direction when its own random turn timer has run out.
  - **Don't chase**: directly behind it on its row, Link is within 8 pixels on that axis at every grid point it
    reaches, so every 16 pixels it has an even chance of turning straight back at him.
  - **Stay at 90 degrees, 1.25-1.5 tiles off**: 20-24 pixels off its line Link is outside the 8, so it cannot turn
    back along its row; the only "toward" turn left is when it comes level with him (within 8 pixels along its
    travel), and that is a 90-degree turn in his direction, which he is already walking away from.
- **The shield** (`asm:Damage_HandleWeaponHit@CheckDarknut`): a hit is refused only when the weapon points straight
  at its front; from the side or from behind it lands, knocks it back and makes it harmless for about 32 frames
  ("passing by him or through him since my sword hit").
- **Clumping** (`asm:Mode5Play_UpdateWorld@CheckChaseTarget`, RAM `$60`-`$62`, docs/ram-catalog.md): the monsters'
  chase target flips every 0-70 frames between Link's position and its MIRROR, (255 - X, 255 - Y). The mirror of a
  point near (128, 128) is itself. So where Link stands close to (128, 128) both targets are on him and every
  Darknut is sent at him all the time; anywhere else half of their turns are aimed at the far side of the room and
  they scatter. In the Olive Garden (128, 128) is a tile and a half north of the block: the owner's zone. It is the
  same point in every room, whatever stands in it.
- **A bomb behind a Darknut** (`asm:Bomb_CheckBombOrFireVsMonster`, `asm:Shove_Begin`): the blast knocks it back in
  the direction Link faced when he laid the bomb, parried or not: laid behind them while both walk the same way, it
  shoots them forward, which is why he stops walking.

## What the takes measure (read frame by frame)

- **`dont_chase`** (163 frames): Link steps onto the Darknut's row 22 pixels behind it at frame 96 while it walks W;
  it reaches the grid point x = 128, turns E, and touches him at frame 112.
- **`play_with_anticipation`** (547 frames): 13 swings, 8 land, no hit taken. Link's distance from the nearest
  Darknut's line of travel: median 20 pixels (1.25 tiles), a tenth of the frames under 8, a tenth over 32; the
  swings that land are made 15-24 pixels off its line and 5-13 pixels ahead of it.
- **`bombs_with_melee`** (2,472 frames, six Darknuts killed, 6 bombs, 23 swings, one hit taken at frame 992):

  | Bomb | Frame | Link from (128, 128) | Darknuts within 40 pixels | Hurt | Dead |
  |---:|---:|---|---:|---:|---:|
  | 1 | 299 | 24 W, 19 N | 3 | 2 | 0 |
  | 2 | 493 | 5 E, 19 N | 2 | 2 | 1 |
  | 3 | 981 | 22 E, 3 N | 2 | 2 | 1 |
  | 4 | 1184 | 57 W, 3 N | 2 | 0 | 0 |
  | 5 | 1428 | 32 E, 19 N | 3 | 2 | 2 |
  | 6 | 2194 | 8 E, 4 N | 2 | 2 | 1 |

  Five of six bombs hurt two; the one that hurt none was laid 57 pixels from the point. With Link within 48 pixels
  of (128, 128) the Darknuts are 50-53 pixels from him on average and 1.5 of them within 40; farther out, 80 pixels
  and 0.8.

## What the networks cannot see today

Nothing in the observation says where the monsters are being sent: neither the flip (`$60`) nor the target (`$61`,
`$62`) is an input, and no column says "this one may turn at you at its next grid point". The entity inputs do carry
"on Link's row" and "on Link's column". Proposed for the fresh generation, the owner to decide: see the round's
document.

## The owner's answers to the lead's questions (Nov 24)

**Is the clump zone the same in every room?** "I think so but you'd have to check enemy movement code, a lot of it
is different than darknuts. Things like Vire and Blue Wizzrobes, and stalfos definitely follow it. What is true is
that area top center of the screen is where both link and shadow link are, so if the monsters are programmed to go
after link/shadow link, that's the hotspot. Moving outwards out of that area generally lets you isolate monsters, in
the top center clumps them. For rooms where you want to bomb them (usually darknuts), you want them clumped. For
rooms you have to kill all enemies, you don't want them scattered about (even if not using bombs) because then you
have to chase them down (and this is a speedrun). Look through the movement code for exceptions to movement that
might make it not gravitate towards the two (red wizzrobes right?), and find the 'range' where link/shadowlink is
the closest in terms of tiles. That will tell you the bounds. It would be good to indirectly teach this to the
model."

**Gibdos and Stalfos the same way?** "Generally it's darknuts because you can't hit them from the front. Stalfos
don't hit that hard and have medium health, so they aren't too hard to deal with. Gibdos...somewhat but not.
Fighting gibdos from the side is preferable, but I chase them from behind ANTICIPATING they will turn around, so I
do chase but I stay that 1.25 tiles (ish) behind and I'm ready with a quick sword because that will actually hit
(unlike darknuts). However, chasing gibdos from behind is also where I lose a ton of health. So it's not good
advice. I'd check their movement code, but it's likely fighting them exactly like a darknut is what I should be
doing anyway."

**A little ahead of it on purpose?** "Yes, a little ahead is the goal (but it's hard to do this in real time and
stay consistent). The idea is if they make a 90 degree turn towards me, my plan was always 'sword to them, then
IMMEDIATELY move in the direction they were'. So if they turn towards me and I'm a little ahead, I will move past
them even though they tried to hit me, and I'm already parallel again (though now I need to get repositioned ~1.25
tiles away and a little in front)."

**The two inputs (where the monsters are sent; a monster may turn at Link at its next grid point)?** "Yes that's
fine. Do that in this upcoming 2.0.1 though."

**A scripted Darknut fighter and demonstrations from it?** "I think so? If you can somehow show the teachers how to
do this and discourage direct following that would help the agent learn this. [...] this is how every real player
kills darknuts, I personally don't know how you would teach the model this, but if you can figure out how it's a
huge win. Similarly, teaching the model to use bombs when it hits 2+ things. A darknut by itself isn't really much
of a threat except in confined spaces."

## Which monsters these rules cover (Chase40, from `zelda-asm/`)

The owner asked: "what enemies have unique movements that wouldn't be affected by these strategies". Every moving
monster's update routine was read (zelda-asm/mechanics.md "Who chases what" has the routine for each row). Three
questions: **don't chase** (followed on its own line from behind, it can turn straight back at Link); **shadow at
90 degrees is safe** (off its line by more than its alignment window it cannot reverse onto Link, and the only turn
toward him is the 90-degree one); **gathers at the hotspot** (it steers by the chase target, Link or his mirror, so
it clumps where the two coincide, about (128, 128), and scatters elsewhere).

| Monster | Don't chase applies | Shadow at 90° safe | Gathers at the hotspot | Its rule |
|---|---|---|---|---|
| Darknut red / blue, **Gibdo**, Stalfos, Like-Like, Vire | yes (129 in 256 a decision) | yes (window 8 px) | yes | `asm:Wanderer_MoveAndTurn`, turn rate $80; a Gibdo walks exactly as a red Darknut (`asm:Gibdo_Update`) |
| Moblin, Octorok, Blue Leever, Ghini, Bubble | yes (Ghini always; Moblin 161, Octorok 113 or 161, Bubble 65 in 256) | yes | yes | the same wanderer rule, other turn rates |
| Zol, Gel, Dodongo | yes, rarely (33 in 256) | yes | yes, weakly | the same, rate $20, Zols and Gels pause at each square |
| Goriya, Lynel, Armos | **yes, always**, within $51 (81) px | **no** | yes, within $51 px | `asm:Goriya_Update`: at every grid point it faces the target along the longer axis if under $51 px; off its line it still reverses whenever Link is farther behind it than beside it |
| Rope | **yes, always**, and it rushes at 1.5 px a frame | **no**: level with Link it charges him | no | `asm:Rope_Update`: Link's real position, less than 8 px off its column or row at a grid point |
| **Blue Wizzrobe** | yes, on its turn beat | no | **no** (the owner expected yes) | `asm:Wizzrobe_Blue_TurnTowardLink`: every 128 frames it turns toward Link's real side on one axis, at any distance; teleports 32 px diagonally from itself; shoots only when already facing Link on his 16-px row or exact column |
| **Red Wizzrobe** | - | - | **no** (as the owner guessed) | `asm:Wizzrobe_Red_State3Appear`: appears 2-5 squares from Link's real position facing him, shoots once on its 16th solid frame; never walks |
| Keese, Peahat, flying Ghini | no (one 45° step per 16 frames) | - | partly (chase in 80-96 of 256 decisions) | `asm:Flyer_State2Chase` |
| Moldorm, Patra, Gleeok's flying head | no (45° steps) | - | yes (192-208 in 256) | flyers that mostly chase |
| Manhandla, Digdogger, Tektite, Boulders | no | - | partly | one 45° step toward the target half the time (Tektite at each jump; Boulders fall in its half of the screen) |
| Lanmola | on a column yes, on a row no | no | no | `asm:Lanmola_UpdateHead`: toward Link's real position 128 in 256 at its points |
| Ganon | yes | no | no | the Blue Wizzrobe's turn, toward Link's real position |
| Red Leever, Zora, Trap, Wallmaster, Pols Voice, Gohma, Aquamentus, Gleeok's attached heads, statues | - | - | no | own rules, below |

**The ones none of the three rules covers, or that move by a rule of their own** (what the code says to do):
- **Red Wizzrobe**: it is placed by Link's real position, 2-5 squares off in a row or column, facing him, and fires
  once along that line. Step off its row or column before its shot; it can be hit only while solid.
- **Blue Wizzrobe**: it ignores the mirror; it shoots only when already facing Link on his row or exact column, on a
  32-frame beat. Stay off its rows and columns; clumping it at the hotspot does not work.
- **Goriyas, Lynels, Armos**: "don't chase" holds more strongly than for Darknuts (certain, from up to 81 px behind),
  and shadowing beside it is safe only level with it or ahead of it, never behind.
- **Rope**: charges Link's real position at triple speed the moment he is within 7 px of its column or row at one of
  its grid points; neither following nor shadowing is safe; meet it head on with the sword or stay off its lines.
- **Red Leever**: surfaces 40 px from Link along his facing (sometimes behind him) and runs straight back along that
  line: turn or step aside after it surfaces.
- **Trap**: charges when Link is within 13 px of its row or column: stay off its lines.
- **Wallmaster**: comes out only while Link stands against a wall: stay off the walls.
- **Zora, Aquamentus, Gohma, Gleeok, statues**: fireballs aimed once at Link's real position, then straight: a
  sideways step after the shot; their bodies move at random or in fixed patterns.
- **Pols Voice**: random walk and jumps; reads no target at all.
- **The fliers** (Keese, Peahat, flying Ghini, Moldorm, Patra, Gleeok's flying head, Manhandla, Digdogger) and
  Tektites: they turn at most 45 degrees every 16 frames (a Tektite per jump), so none turns straight back; the
  chasing ones drift toward the hotspot when Link is there.
