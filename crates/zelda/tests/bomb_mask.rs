//! Round 22 (4:0x21's bomb waste): BOMB kept to the bombs that open the route's wall (`Session::set_bomb_targets`,
//! `zelda::bombs::BombTarget`, `zelda::bombs::landing`).  Played in the game for every room the rule touches on the
//! route (the `requires_bombs` rooms of the `required` families; round 26: every bomb-wall room, where a granted
//! bomb only opens the wall whatever the family's mode): around each wall, from every grid point and in
//! every direction the game lets Link place a bomb, a BOMB the mask offers opens the wall with that one bomb, and one
//! it doesn't offer leaves the wall shut.  Round 23: one bomb per wall (`BombTarget::awaits_bomb`, `one_bomb`).
mod common;

use common::{load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zcore::ZCore;
use zelda::bombs::{BombTarget, landing, live_bombs};
use zelda::game::GameView;
use zelda::{Action, ActionKind, Direction, GameMap, LoadOptions, Position, Session, Var};

/// Object type of the overworld rock wall tile object (asm:RockWall_Update).
const ROCK_WALL: u8 = 0x63;

const CARDINALS: [Direction; 4] = [Direction::N, Direction::S, Direction::W, Direction::E];

fn act(s: &mut Session<ZCore>, text: &str) {
    s.act(Action::parse(text).unwrap()).unwrap();
}

/// The target is still shut.
fn shut(state: &GameView, target: BombTarget) -> bool {
    match target {
        BombTarget::Wall(d) => state.bomb_walls().iter().any(|w| w.wall_direction == d),
        BombTarget::Rock(_) => state.ow_bomb_secret_closed(),
    }
}

/// Link placed at (x, y) facing N with bombs, the room's monsters and traps gone, a rock wall's tile object kept
/// (test setup).
fn place(s: &mut Session<ZCore>, x: u8, y: u8) {
    for slot in 1..=11u16 {
        if s.state().peek(0x34F + slot) != ROCK_WALL {
            zelda::Nes::poke(s.nes_mut(), 0x34F + slot, 0);
        }
    }
    s.apply_ram(&[
        (Var::LinkX, x),
        (Var::LinkY, y),
        (Var::LinkDirection, 8),
        (Var::Bombs, 4),
        (Var::HeartsAndContainers, 0x55),
        (Var::PartialHearts, 0xFF),
    ]);
}

/// The state loaded and played to its first frame where Link can move, the target still shut.
fn start(state: &str, target: BombTarget) -> Session<ZCore> {
    let nes = open_core().expect("ROM");
    let mut s = Session::new(nes, Arc::new(GameMap::builtin()));
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    s.load(&load_named_state(state), &LoadOptions::default(), &mut rng).unwrap();
    for _ in 0..40 {
        if s.state().peek(0x12) == 5 && s.mask().kind[ActionKind::Move.index()] {
            break;
        }
        act(&mut s, "WAIT 8");
    }
    assert!(shut(s.state(), target), "{state}: {target:?} is shut at the start");
    s
}

/// The grid points within 48 px of `centre` Link can be placed on.
fn grid(centre: (i32, i32)) -> impl Iterator<Item = (u8, u8)> {
    (-48..=48).step_by(8).flat_map(move |dy| {
        (-48..=48).step_by(8).filter_map(move |dx| {
            let (x, y) = (centre.0 + dx, centre.1 + dy);
            ((0x10..=0xE0).contains(&x) && (0x45..=0xD5).contains(&y)).then_some((x as u8, y as u8))
        })
    })
}

fn bomb(d: Direction) -> Action {
    Action::new(ActionKind::Bomb, d, 0).unwrap()
}

/// Sweeps the grid points within 48 px of `centre`: (offered, opened) pairs, all equal.  Returns the number of
/// BOMBs offered and the number refused that the game would have placed.
fn sweep(state: &str, target: BombTarget, centre: (i32, i32)) -> (usize, usize) {
    let mut s = start(state, target);
    let begin = s.snapshot();
    let (mut offered, mut refused) = (0, 0);
    for (x, y) in grid(centre) {
        for d in CARDINALS {
            s.restore(&begin);
            s.set_bomb_targets(None);
            place(&mut s, x, y);
            if !s.mask().is_allowed(&bomb(d)) {
                continue;
            }
            s.set_bomb_targets(Some(&[target]));
            let allowed = s.mask().is_allowed(&bomb(d));
            s.act(bomb(d)).unwrap();
            for _ in 0..8 {
                act(&mut s, "WAIT 16");
            }
            let opened = !shut(s.state(), target);
            assert_eq!(
                allowed, opened,
                "{state}: Link at (${x:02X}, ${y:02X}), BOMB {d:?}: offered {allowed}, the bomb opened {target:?}: {opened} (room flags ${:02X}, cave tile {:?}, link {:?})",
                s.state().peek(0x67F + u16::from(s.state().location())),
                s.state().room().cave_tile,
                s.state().link.position
            );
            if allowed { offered += 1 } else { refused += 1 }
        }
    }
    (offered, refused)
}

/// What the env offers once a bomb has done its work: the target while it is shut, nothing once it is open
/// (`route_bomb_targets` drops an opened wall).
fn retarget(s: &mut Session<ZCore>, target: BombTarget) {
    let targets: &[BombTarget] = if shut(s.state(), target) { &[target] } else { &[] };
    s.set_bomb_targets(Some(targets));
}

fn no_bomb_offered(s: &Session<ZCore>) -> bool {
    CARDINALS.into_iter().all(|d| !s.mask().is_allowed(&bomb(d)))
}

/// Round 23 (a second BOMB in the first one's fuse or blast at 4:0x21): one bomb per target.
/// - After a bomb the mask offers in reach of the target, no BOMB is offered while that bomb lives (fuse, blast,
///   fading), though for a wall the game would place one from the blast on; then the target is open.
/// - A bomb out of reach blocks nothing: with it still live a BOMB in reach is offered, and it opens the target.
/// - A bomb in reach that is gone without its blast (its slot emptied) frees the target again.
fn one_bomb(state: &str, target: BombTarget, centre: (i32, i32)) {
    let mut s = start(state, target);
    let begin = s.snapshot();
    let spots: Vec<(u8, u8, Direction)> =
        grid(centre).flat_map(|(x, y)| CARDINALS.map(move |d| (x, y, d))).collect();
    let offered_at = |s: &mut Session<ZCore>, targets: Option<&[BombTarget]>, keep: &dyn Fn(Position) -> bool| {
        spots.iter().copied().find(|&(x, y, d)| {
            s.restore(&begin);
            place(s, x, y);
            s.set_bomb_targets(targets);
            s.mask().is_allowed(&bomb(d)) && landing(s.state(), d).is_some_and(keep)
        })
    };
    let (x, y, d) = offered_at(&mut s, Some(&[target]), &|_| true).expect("a BOMB the mask offers");
    let (fx, fy, fd) = offered_at(&mut s, None, &|at| !target.in_reach(at)).expect("a BOMB out of reach");

    // The good bomb, then nothing until it is gone.
    s.restore(&begin);
    place(&mut s, x, y);
    s.set_bomb_targets(Some(&[target]));
    s.act(bomb(d)).unwrap();
    let (mut steps, mut game_would) = (0, false);
    while live_bombs(s.state()).next().is_some() {
        assert!(target.awaits_bomb(s.state()) || !shut(s.state(), target), "{state}: the bomb lies in reach");
        retarget(&mut s, target);
        assert!(no_bomb_offered(&s), "{state}: step {steps}: a second BOMB offered while the first lives (slots ${:02X} ${:02X})", s.state().peek(0xBC), s.state().peek(0xBD));
        s.set_bomb_targets(None);
        game_would |= s.mask().is_allowed(&bomb(d));
        act(&mut s, "WAIT 4");
        steps += 1;
        assert!(steps < 40, "{state}: the bomb never went");
    }
    assert!(!shut(s.state(), target), "{state}: the one bomb opened {target:?}");
    if matches!(target, BombTarget::Wall(_)) {
        // A rock wall opens as the blast starts, so the game itself has no bomb target left by then.
        assert!(game_would, "{state}: the game would have placed a second bomb in the first one's blast");
    }

    // A bomb out of reach, then the good one while it still lives.
    s.restore(&begin);
    place(&mut s, fx, fy);
    s.set_bomb_targets(None);
    s.act(bomb(fd)).unwrap();
    while live_bombs(s.state()).next().is_some() && !s.mask().is_allowed(&bomb(fd)) {
        act(&mut s, "WAIT 4");
    }
    let far = live_bombs(s.state()).next().expect("the far bomb is still live");
    assert!(!target.in_reach(far), "{state}: the far bomb at {far:?}");
    place(&mut s, x, y);
    retarget(&mut s, target);
    assert!(s.mask().is_allowed(&bomb(d)), "{state}: a bomb out of reach blocked the good one");
    s.act(bomb(d)).unwrap();
    for _ in 0..8 {
        act(&mut s, "WAIT 16");
    }
    assert!(!shut(s.state(), target), "{state}: the good bomb opened {target:?} beside the far one");

    // The good bomb vanishes in its fuse: the target is free again.
    s.restore(&begin);
    place(&mut s, x, y);
    s.set_bomb_targets(Some(&[target]));
    s.act(bomb(d)).unwrap();
    retarget(&mut s, target);
    assert!(no_bomb_offered(&s), "{state}: a second BOMB offered in the first one's fuse");
    for slot in [0xBC, 0xBD] {
        zelda::Nes::poke(s.nes_mut(), slot, 0);
    }
    s.apply_ram(&[]);
    retarget(&mut s, target);
    assert!(s.mask().is_allowed(&bomb(d)), "{state}: the target stays blocked after its bomb is gone");
}

/// One room on the route where a `required` family bombs: it offers some BOMBs, refuses some the game would place,
/// and the two agree with the game (`sweep`); one bomb per wall (`one_bomb`).
fn room(state: &str, target: BombTarget, centre: (i32, i32)) {
    if open_core().is_none() {
        return;
    }
    let (offered, refused) = sweep(state, target, centre);
    println!("{state}: {offered} BOMBs offered, {refused} refused");
    assert!(offered > 0 && refused > 0, "{state}: {offered} offered, {refused} refused");
    one_bomb(state, target, centre);
}

/// ow-tektite: 0x2c's rock wall (the take-any cave's heart).
#[test]
fn bombs_at_0x2c_open_its_rock_wall() {
    room("0_2ce_0", BombTarget::rock((18, 12)), (0x90, 0xAD));
}

/// ow-octorok: 0x67's rock wall (the secret rupees, round 22).
#[test]
fn bombs_at_0x67_open_its_rock_wall() {
    room("0_67s_0", BombTarget::rock((14, 2)), (0x70, 0x6D));
}

/// trap-room: Dungeon 3's 0x4c E wall into Manhandla's room (entered from S since round 22).
#[test]
fn bombs_at_3_0x4c_open_its_east_wall() {
    room("3_4cs_0", BombTarget::Wall(Direction::E), (0xC0, 0x8D));
}

/// trap-room: Dungeon 4's rupee room 0x11, its N wall into the Keese room 0x01 (round 22).
#[test]
fn bombs_at_4_0x11_open_its_north_wall() {
    room("4_11s_0", BombTarget::Wall(Direction::N), (0x78, 0x6D));
}

/// stalfos-keese-gel: 4:0x21's N wall, from its W door (the plain way) and from its island (the clip way, where
/// round 21's chains threw 3.4-3.8 bombs a visit).
#[test]
fn bombs_at_4_0x21_open_its_north_wall_from_both_entries() {
    room("4_21w_0", BombTarget::Wall(Direction::N), (0x78, 0x6D));
    room("4_21s_0", BombTarget::Wall(Direction::N), (0x78, 0x6D));
}

/// darknut (`free`): Dungeon 3's central Darknut room, its E wall into 0x5c on the last pass (round 26: a granted
/// bomb goes to the wall only).
#[test]
fn bombs_at_3_0x5b_open_its_east_wall() {
    room("3_5bw_0", BombTarget::Wall(Direction::E), (0xC0, 0x8D));
}

/// gibdo: Dungeon 5's 0x66 and 0x65, their W walls on the way in (round 24).
#[test]
fn bombs_at_5_0x66_and_0x65_open_their_west_walls() {
    room("5_66s_0", BombTarget::Wall(Direction::W), (0x30, 0x8D));
    room("5_65e_0", BombTarget::Wall(Direction::W), (0x30, 0x8D));
}

/// goriya (`free`): Dungeon 7's Blue Goriya wall rooms on the way up, 0x69 N and 0x19 E (round 26: a granted bomb
/// goes to the wall only).
#[test]
fn bombs_at_7_0x69_and_0x19_open_their_walls() {
    room("7_69s_0", BombTarget::Wall(Direction::N), (0x78, 0x6D));
    room("7_19w_0", BombTarget::Wall(Direction::E), (0xC0, 0x8D));
}

/// goriya (`free`): 7:0x1a's E wall back from the candle cellar, and 7:0x29's (a dead end without a bomb).
#[test]
fn bombs_at_7_0x1a_and_0x29_open_their_east_walls() {
    room("7_1ac_0", BombTarget::Wall(Direction::E), (0xC0, 0x8D));
    room("7_29c_0", BombTarget::Wall(Direction::E), (0xC0, 0x8D));
}

/// dodongo-crossing: 7:0x0c's E wall into the Wallmaster room (round 26).
#[test]
fn bombs_at_7_0x0c_open_its_east_wall() {
    room("7_0cs_0", BombTarget::Wall(Direction::E), (0xC0, 0x8D));
}
