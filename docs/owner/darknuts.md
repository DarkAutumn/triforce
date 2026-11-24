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
