//! `tf replay clip`: each visit of recorded runs to one room, as a replay of its own (`tfreplay::rewrite` with one
//! window) that starts from the machine state at the visit's first frame.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ::zcore::Rom;
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value, json};
use tfreplay::{ENDING_DEATH, ENDING_OTHER, Ending, Player, Replay, Rewrite, Window};
use zelda::executor::MODE_GAMEPLAY;
use zelda::game::{MODE_CAVE, MODE_CELLAR, MODE_DYING, MODE_GAME_OVER};
use zelda::vars::Var;

/// Leaving a cellar; the shortcut cave's mode.
const MODE_LEAVE_CELLAR: u8 = 0x0A;
const MODE_CAVE_SHORTCUT: u8 = 0x0C;

/// A room: level, room id, and whether it is the cave or cellar under it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Room {
    pub level: u8,
    pub room: u8,
    pub cave: bool,
}

impl Room {
    /// `L:RR` with the room in hex (`7:1b`, `7:0x1b`; `0:xx` the overworld), `:cave` for its cave or cellar.
    pub fn parse(text: &str) -> Result<Room> {
        let bad = || format!("--room {text}: expected L:RR, the room in hex (7:1b, 0:0x77, 0:77:cave)");
        let mut parts = text.split(':');
        let (Some(level), Some(room)) = (parts.next(), parts.next()) else { bail!(bad()) };
        let cave = match parts.next() {
            None => false,
            Some("cave") => true,
            Some(_) => bail!(bad()),
        };
        ensure!(parts.next().is_none(), bad());
        let level: u8 = level.parse().with_context(bad)?;
        let room = u8::from_str_radix(room.trim_start_matches("0x"), 16).with_context(bad)?;
        ensure!(level <= 9 && room < 0x80, bad());
        Ok(Room { level, room, cave })
    }

    /// `7:0x1b`, `0:0x77:cave` (as `tf replay targets` writes rooms).
    pub fn text(self) -> String {
        format!("{}:0x{:02x}{}", self.level, self.room, if self.cave { ":cave" } else { "" })
    }

    /// The file-name form: `7x1b`, `0x77cave`.
    pub fn tag(self) -> String {
        format!("{}x{:02x}{}", self.level, self.room, if self.cave { "cave" } else { "" })
    }
}

/// Where the machine is before an input frame (after the frame's RESET, before its edits): `$12` GameMode,
/// `$10` CurLevel, `$EB` RoomId.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Look {
    pub mode: u8,
    pub level: u8,
    pub room: u8,
}

impl Look {
    pub fn of(ram: &[u8; 0x800]) -> Look {
        let byte = |var: Var| ram[usize::from(var.addr())];
        Look { mode: byte(Var::Mode), level: byte(Var::Level), room: byte(Var::Location) }
    }

    /// In a cave or cellar, or on the way in or out of one.
    fn cave(self) -> bool {
        matches!(self.mode, MODE_CELLAR | MODE_LEAVE_CELLAR | MODE_CAVE | MODE_CAVE_SHORTCUT)
    }

    fn room(self) -> Room {
        Room { level: self.level, room: self.room, cave: self.cave() }
    }

    /// Link can act: mode 5, in a cave or cellar its mode (`$0B`, `$09`).
    fn playable(self) -> bool {
        matches!(self.mode, MODE_GAMEPLAY | MODE_CAVE | MODE_CELLAR)
    }
}

/// How a visit ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    /// Link died in the room: the visit ends at the first frame of the dying mode (`$11`).
    Death,
    /// Into this room (the first frame outside it).
    Left(Room),
    /// The replay ended in the room.
    Replay,
}

/// One visit: the input frames `first..=last` in the room, `play` the first and last of them in a playable mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Visit {
    pub number: usize,
    pub first: u32,
    pub last: u32,
    pub play: Option<(u32, u32)>,
    pub end: End,
}

/// The visits to `room` in `looks` (one per input frame, then the machine after the last).  A visit runs while the
/// room holds and ends at a death there; the dying and game over frames count as the room Link died in (a cave's
/// mode is gone by then), and belong to no visit up to the next playable frame, so a respawn in the same room is
/// a visit of its own.  A scroll shows the next room's id for a frame or two and comes back to this one: frames
/// out of the room that Link comes back from without having played anywhere belong to the visit.
pub fn visits(looks: &[Look], room: Room) -> Vec<Visit> {
    let frames = looks.len().saturating_sub(1) as u32;
    let mut rooms: Vec<Room> = Vec::with_capacity(looks.len());
    for look in looks {
        let here = match (look.mode, rooms.last()) {
            (MODE_DYING | MODE_GAME_OVER, Some(&before)) => before,
            _ => look.room(),
        };
        rooms.push(here);
    }
    let at = |k: u32| looks[k as usize];
    let room_at = |k: u32| rooms[k as usize];
    let mut out = Vec::new();
    let mut k = 0;
    while k < frames {
        if room_at(k) != room {
            k += 1;
            continue;
        }
        let first = k;
        loop {
            while k < frames && room_at(k) == room && at(k).mode != MODE_DYING {
                k += 1;
            }
            if k == frames || room_at(k) == room {
                break;
            }
            match (k..frames).find(|&j| room_at(j) == room || at(j).playable()) {
                Some(back) if room_at(back) == room => k = back,
                _ => break,
            }
        }
        // The machine after the visit's last input frame says how it ended.
        let next = at(k);
        let end = if room_at(k) != room {
            End::Left(room_at(k))
        } else if next.mode == MODE_DYING {
            End::Death
        } else {
            End::Replay
        };
        let last = k.checked_sub(1).filter(|&last| last >= first);
        while k < frames && room_at(k) == room && !at(k).playable() {
            k += 1;   // after a death: the game over screens, up to a respawn here
        }
        let Some(last) = last else { continue };   // dying as it came in
        let playable = |&f: &u32| room_at(f) == room && at(f).playable();
        let play = (first..=last).find(playable).zip((first..=last).rev().find(playable));
        out.push(Visit { number: out.len() + 1, first, last, play, end });
    }
    out
}

/// What `tf replay clip` does with each file.
pub struct Options {
    pub room: Room,
    /// One visit (1-based) or all.
    pub visit: Option<usize>,
    /// Only the playable frames of each visit.
    pub playable: bool,
    pub out: PathBuf,
}

type Resolve<'a> = &'a (dyn Fn(&str) -> std::io::Result<Vec<u8>> + Sync);

/// Where the machine is before each input frame, and after the last.
fn scan(rom: &Arc<Rom>, replay: &Replay, resolve: Resolve<'_>) -> Result<Vec<Look>> {
    let mut player = Player::open(rom.clone(), replay.clone(), resolve)?;
    let mut looks = Vec::with_capacity(player.frames() as usize + 1);
    loop {
        player.press_resets();
        looks.push(Look::of(player.nes().ram()));
        if !player.step(false)? {
            return Ok(looks);
        }
    }
}

/// The ending the source replay records (META `ending` and the header's kind).
fn source_ending(replay: &Replay, meta: &Map<String, Value>) -> Ending {
    let text = |key: &str| {
        meta.get("ending").and_then(|e| e.get(key)).and_then(Value::as_str).unwrap_or_default().to_owned()
    };
    Ending { kind: replay.ending(), result: text("result"), outcome: text("outcome"), reason: text("reason") }
}

/// What [`clip_file`] did: a line per visit (written or skipped), and the clips written.
pub struct Clipped {
    pub lines: Vec<String>,
    pub written: usize,
}

/// Writes `file`'s clips into `options.out`.
pub fn clip_file(rom: &Arc<Rom>, file: &Path, options: &Options, resolve: Resolve<'_>) -> Result<Clipped> {
    let bytes = std::fs::read(file).with_context(|| format!("reading {}", file.display()))?;
    let replay = Replay::parse(bytes).with_context(|| file.display().to_string())?;
    let meta = replay.meta()?;
    let room = options.room;
    let all = visits(&scan(rom, &replay, resolve)?, room);
    if all.is_empty() {
        return Ok(Clipped { lines: vec![format!("never in {}", room.text())], written: 0 });
    }
    let chosen: Vec<Visit> = match options.visit {
        None => all.clone(),
        Some(n) => match all.get(n.wrapping_sub(1)) {
            Some(&visit) => vec![visit],
            None => bail!("no visit {n} to {} ({} visits)", room.text(), all.len()),
        },
    };

    let span = |v: &Visit| if options.playable { v.play.map(|(a, b)| (a, b + 1)) } else { Some((v.first, v.last + 1)) };
    let mut lines = BTreeMap::new();
    let mut cuts = Vec::new();
    for visit in &chosen {
        match (visit.play, span(visit)) {
            (Some(_), Some(cut)) => cuts.push((*visit, cut)),
            _ => {
                let line = format!("visit {}: frames {}-{} never playable, skipped", visit.number, visit.first, visit.last);
                lines.insert(visit.number, line);
            }
        }
    }
    let wanted: Vec<u32> = cuts.iter().flat_map(|&(_, (from, to))| [from, to]).collect();
    let states = tfreplay::cut_states(rom.clone(), replay.clone(), resolve, &wanted)?;

    let name = file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let stem = file.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    std::fs::create_dir_all(&options.out).with_context(|| format!("creating {}", options.out.display()))?;
    let written = cuts.len();
    for (visit, (from, to)) in cuts {
        let (how, ending) = match visit.end {
            End::Death => ("death".to_owned(), Ending {
                kind: ENDING_DEATH,
                result: "death".into(),
                outcome: String::new(),
                reason: format!("died in {}", room.text()),
            }),
            End::Left(next) => (format!("left for {}", next.text()), Ending {
                kind: ENDING_OTHER,
                result: "left".into(),
                outcome: String::new(),
                reason: format!("left {} for {}", room.text(), next.text()),
            }),
            End::Replay => ("replay ended".to_owned(), source_ending(&replay, &meta)),
        };
        let mut about = Map::new();
        about.insert("source".into(), json!(name));
        about.insert("room".into(), json!(room.text()));
        about.insert("visit".into(), json!(visit.number));
        about.insert("visits".into(), json!(all.len()));
        about.insert("visit_frames".into(), json!([visit.first, visit.last + 1]));
        about.insert("playable".into(), json!(options.playable));
        about.insert("ending".into(), json!(how));
        let window = Window { from, to, start: &states[&from].0, end: Some(&states[&to].1), about };
        let spec = Rewrite { windows: vec![window], ending: Some(ending), ..Rewrite::default() };
        let bytes = tfreplay::rewrite(&replay, &spec)?;
        let path = options.out.join(format!("{stem}_{}_v{}.tfreplay", room.tag(), visit.number));
        std::fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
        let line = format!(
            "visit {}: frames {from}-{} ({} frames), {how} -> {}",
            visit.number,
            to - 1,
            to - from,
            path.display()
        );
        lines.insert(visit.number, line);
    }
    Ok(Clipped { lines: lines.into_values().collect(), written })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn look(mode: u8, level: u8, room: u8) -> Look {
        Look { mode, level, room }
    }

    #[test]
    fn rooms_parse_in_hex_with_an_optional_cave() {
        assert_eq!(Room::parse("7:1b").unwrap(), Room { level: 7, room: 0x1b, cave: false });
        assert_eq!(Room::parse("7:0x1b").unwrap(), Room { level: 7, room: 0x1b, cave: false });
        assert_eq!(Room::parse("0:77:cave").unwrap(), Room { level: 0, room: 0x77, cave: true });
        assert_eq!(Room::parse("0:77:cave").unwrap().tag(), "0x77cave");
        assert_eq!(Room::parse("7:1b").unwrap().text(), "7:0x1b");
        for bad in ["7", "7:zz", "7:1b:x", "10:1b", "7:80", "7:1b:cave:x"] {
            assert!(Room::parse(bad).is_err(), "{bad}");
        }
    }

    /// Two visits (scroll in, play, scroll out; then back, play, die, game over), the frames around them elsewhere.
    #[test]
    fn visits_end_at_leaving_and_at_a_death() {
        let room = Room::parse("7:1b").unwrap();
        let looks = vec![
            look(5, 7, 0x1c),
            look(7, 7, 0x1b), // 1: scrolling in
            look(5, 7, 0x1b), // 2
            look(5, 7, 0x1b), // 3
            look(7, 7, 0x1b), // 4: scrolling out
            look(5, 7, 0x1a), // 5: played in the next room
            look(5, 7, 0x1b), // 6
            look(MODE_DYING, 7, 0x1b), // 7
            look(8, 7, 0x1b), // 8: game over, in no visit
            look(5, 7, 0x0b), // 9
            look(MODE_CELLAR, 7, 0x1b), // 10: the cellar under it is another room
            look(5, 7, 0x1b), // 11: the machine after the last input, out of the cellar
        ];
        let v = visits(&looks, room);
        assert_eq!(v, vec![
            Visit { number: 1, first: 1, last: 4, play: Some((2, 3)), end: End::Left(Room { level: 7, room: 0x1a, cave: false }) },
            Visit { number: 2, first: 6, last: 6, play: Some((6, 6)), end: End::Death },
        ]);
        let cellar = visits(&looks, Room::parse("7:1b:cave").unwrap());
        let up = End::Left(Room { level: 7, room: 0x1b, cave: false });
        assert_eq!(cellar, vec![Visit { number: 1, first: 10, last: 10, play: Some((10, 10)), end: up }]);
        let mut ended = looks.clone();
        ended[11] = look(MODE_CELLAR, 7, 0x1b);
        let cellar = visits(&ended, Room::parse("7:1b:cave").unwrap());
        assert_eq!(cellar, vec![Visit { number: 1, first: 10, last: 10, play: Some((10, 10)), end: End::Replay }]);
    }

    #[test]
    fn a_visit_never_playable_has_no_play_span() {
        let looks = vec![look(7, 7, 0x1b), look(7, 7, 0x1b), look(5, 7, 0x1c)];
        let v = visits(&looks, Room::parse("7:1b").unwrap());
        assert_eq!(v, vec![Visit { number: 1, first: 0, last: 1, play: None, end: End::Left(Room::parse("7:1c").unwrap()) }]);
    }

    /// A scroll out of the room loads the next room's id for two frames, then shows this one's until it is done.
    #[test]
    fn a_scroll_showing_the_next_room_for_a_moment_stays_one_visit() {
        let looks = vec![
            look(5, 7, 0x1c),
            look(7, 7, 0x1b), // 1: scrolling in
            look(5, 7, 0x1b), // 2
            look(7, 7, 0x1b), // 3: scrolling out
            look(7, 7, 0x1a), // 4: the next room's id
            look(7, 7, 0x1b), // 5: this one's again
            look(7, 7, 0x1b), // 6
            look(5, 7, 0x1a), // 7: in the next room
            look(5, 7, 0x1a),
        ];
        let v = visits(&looks, Room::parse("7:1b").unwrap());
        let left = End::Left(Room::parse("7:1a").unwrap());
        assert_eq!(v, vec![Visit { number: 1, first: 1, last: 6, play: Some((2, 2)), end: left }]);
    }

    /// Dying in a room and respawning in it (a dungeon's entrance): the game over screens belong to no visit, the
    /// respawn is a visit of its own.
    #[test]
    fn a_respawn_in_the_room_is_a_new_visit() {
        let looks = vec![
            look(5, 7, 0x1b),
            look(MODE_DYING, 7, 0x1b), // 1
            look(MODE_GAME_OVER, 7, 0x1b), // 2
            look(3, 7, 0x1b), // 3: the dungeon drawn again
            look(5, 7, 0x1b), // 4: playing again
            look(5, 7, 0x1b), // 5
            look(5, 7, 0x1c), // 6
        ];
        let v = visits(&looks, Room::parse("7:1b").unwrap());
        let left = End::Left(Room::parse("7:1c").unwrap());
        assert_eq!(v, vec![
            Visit { number: 1, first: 0, last: 0, play: Some((0, 0)), end: End::Death },
            Visit { number: 2, first: 4, last: 5, play: Some((4, 5)), end: left },
        ]);
    }

    /// The dying and game over modes are no cave modes: those frames count as the room Link died in, so a death in
    /// a cave ends its visit as a death and makes no visit to the overworld room above it.
    #[test]
    fn a_death_in_a_cave_is_a_death_there() {
        let looks = vec![
            look(5, 0, 0x77),
            look(MODE_CAVE, 0, 0x77), // 1
            look(MODE_CAVE, 0, 0x77), // 2
            look(MODE_DYING, 0, 0x77), // 3
            look(MODE_GAME_OVER, 0, 0x77), // 4
            look(5, 0, 0x77), // 5: respawned outside
            look(5, 0, 0x77),
        ];
        let cave = visits(&looks, Room::parse("0:77:cave").unwrap());
        assert_eq!(cave, vec![Visit { number: 1, first: 1, last: 2, play: Some((1, 2)), end: End::Death }]);
        let outside = visits(&looks, Room::parse("0:77").unwrap());
        let into = End::Left(Room::parse("0:77:cave").unwrap());
        assert_eq!(outside, vec![
            Visit { number: 1, first: 0, last: 0, play: Some((0, 0)), end: into },
            Visit { number: 2, first: 5, last: 5, play: Some((5, 5)), end: End::Replay },
        ]);
    }
}
