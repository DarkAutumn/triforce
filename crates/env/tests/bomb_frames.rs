//! The bomb metrics' death attribution on real frames (`zelda::bomb_windows::cycle_deaths`): the corpus's 2.0.0.13
//! single run, where two deaths drop a fairy and the fairy's init zeroes Item_ObjMonsterType (`$412`, the flyer's
//! position fraction) on the very frame the slot turns into the item. The dead type is the one the slot held while
//! dying.

use std::path::Path;
use std::sync::Arc;

use replay::{Player, Replay};
use zcore::Rom;
use zelda::GameMap;
use zelda::bomb_windows::{Step, cycle_deaths};
use zelda::game::GameView;
use zelda::tables::RAM_SIZE;

const REPLAY: &str = "corpus/replays/single-run-d3first-2.0.0.13-55-networks.tfreplay";

fn view(player: &Player, map: &Arc<GameMap>) -> Arc<GameView> {
    let mut image = Box::new([0u8; RAM_SIZE]);
    image[..0x800].copy_from_slice(player.nes().ram());
    image[0x800..].copy_from_slice(player.nes().prg_ram());
    Arc::new(GameView::new(&image, u64::from(player.frame()), false, map.clone()))
}

#[test]
fn a_fairy_drop_does_not_hide_the_dead_type() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(REPLAY);
    let (Some(bytes), Ok(data)) = (test_rom::bytes(), std::fs::read(&path)) else {
        eprintln!("skipped: no ROM or no {REPLAY}");
        return;
    };
    let rom = Rom::from_ines(&bytes).expect("the Zelda ROM");
    let no_files = |name: &str| Err(std::io::Error::new(std::io::ErrorKind::NotFound, name.to_owned()));
    let mut player = Player::open(rom, Replay::parse(data).expect("replay"), &no_files).expect("player");
    let map = Arc::new(GameMap::builtin());
    // (input frames run before the death's frame, slot, the type that died): 3:0x5b's Darknut (a bomb carrier,
    // cycle 9 → 0) and 9:0x16's Patra (cycle 4 → 5).
    for (frame, slot, kind) in [(25_126u32, 2u8, 0x0Bu8), (144_232, 1, 0x48)] {
        player.seek(frame).expect("seek");
        let prev = view(&player, &map);
        assert!(player.step(false).expect("step"));
        let curr = view(&player, &map);
        assert_eq!((prev.peek(0x34F + u16::from(slot)), curr.peek(0x34F + u16::from(slot))), (kind, 0x60));
        assert_eq!(curr.peek(0x412 + u16::from(slot)), 0, "the fairy zeroed $412");
        let step = Step { prev: &prev, curr: &curr, left: false, frames: 1, own_prev: prev.bombs() };
        let (moves, deaths) = cycle_deaths(&step);
        assert_eq!(moves, 1);
        assert!(deaths.iter().any(|d| d.slot == slot && d.kind == kind), "frame {frame}: {deaths:?}");
    }
}
