//! Human/LLM-readable text dumps of the game state (`triforce/describe.py`), used by the `play` tool.

use zelda::actions::ActionMask;
use zelda::enums::{ArrowKind, BoomerangKind, Direction, SwordKind};
use zelda::game::GameView;
use zelda::room::{
    COLS, EAST_DOOR_TILE, NORTH_DOOR_TILE, ROWS, SOUTH_DOOR_TILE, TOP_CAVE_TILE, WEST_DOOR_TILE,
};
use zelda::{Change, Tile};

const CARDINALS: [Direction; 4] = [Direction::N, Direction::S, Direction::W, Direction::E];

/// `ZeldaEnemyKind` names.
pub fn enemy_name(id: u8) -> String {
    let name = match id {
        0x01 => "BlueLynel",
        0x02 => "RedLynel",
        0x03 => "BlueMoblin",
        0x04 => "RedMoblin",
        0x05 => "BlueGoriya",
        0x06 => "RedGoriya",
        0x07 => "RedSlowOctorok",
        0x08 => "RedOctorok",
        0x09 => "BlueSlowOctorok",
        0x0A => "BlueOctorok",
        0x0B => "RedDarknut",
        0x0C => "BlueDarknut",
        0x0D => "BlueTektite",
        0x0E => "RedTektite",
        0x0F => "BlueLeever",
        0x10 => "RedLeever",
        0x11 => "Zora",
        0x12 => "Vire",
        0x13 => "Zol",
        0x14 => "GreenGel",
        0x15 => "BlueGel",
        0x16 => "PolsVoice",
        0x17 => "LikeLike",
        0x18 => "DigdoggerChild",
        0x1A => "Peahat",
        0x1B => "BlueKeese",
        0x1C => "RedKeese",
        0x1D => "BlackKeese",
        0x1E => "Armos",
        0x21 => "Ghini",
        0x22 => "FlyingGhini",
        0x23 => "BlueWizzrobe",
        0x24 => "RedWizzrobe",
        0x25 => "PatraChild1",
        0x26 => "PatraChild2",
        0x27 => "Wallmaster",
        0x28 => "Rope",
        0x2A => "Stalfos",
        0x2B => "Bubble",
        0x2C => "BlueBubble",
        0x2D => "RedBubble",
        0x30 => "Gibdo",
        0x31 => "Dodongo",
        0x32 => "Dodongo2",
        0x33 => "BlueGohma",
        0x34 => "RedGohma",
        0x38 => "Digdogger2",
        0x39 => "Digdogger3",
        0x3A => "RedLamnola",
        0x3B => "BlueLamnola",
        0x3C => "Manhandla",
        0x3D => "Aquamentus",
        0x3E => "Ganon",
        0x41 => "Moldorm",
        0x42 => "Gleeok",
        0x43 => "Gleeok2",
        0x44 => "Gleeok3",
        0x45 => "Gleeok4",
        0x46 => "GleeokHead",
        0x47 => "Patra",
        0x48 => "Patra2",
        _ => return id.to_string(),
    };
    name.to_owned()
}

/// `ZeldaItemKind` names.
pub fn item_name(id: i16) -> String {
    let name = match id {
        -109 => "TriforceOfPower",
        -108..=-101 => return format!("Triforce{}", -100 - id),
        -4 => "Map",
        -3 => "Compass",
        -2 => "HeartContainer",
        -1 => "Key",
        0x00 => "Bombs",
        0x0F => "BlueRupee",
        0x18 => "Rupee",
        0x21 => "Clock",
        0x22 => "Heart",
        0x23 => "Fairy",
        _ => return id.to_string(),
    };
    name.to_owned()
}

/// `ZeldaProjectileId` names.
pub fn projectile_name(id: u8) -> String {
    match id {
        83 => "OtorokRock".to_owned(),
        85 => "ZoraFireball".to_owned(),
        91 => "Arrow".to_owned(),
        92 => "Boomerang".to_owned(),
        _ => id.to_string(),
    }
}

fn sword_name(s: SwordKind) -> &'static str {
    match s {
        SwordKind::None => "NONE",
        SwordKind::Wood => "WOOD",
        SwordKind::White => "WHITE",
        SwordKind::Magic => "MAGIC",
    }
}

fn arrow_name(a: ArrowKind) -> &'static str {
    match a {
        ArrowKind::None => "NONE",
        ArrowKind::Wood => "WOOD",
        ArrowKind::Silver => "SILVER",
    }
}

fn boomerang_name(b: BoomerangKind) -> &'static str {
    match b {
        BoomerangKind::None => "NONE",
        BoomerangKind::Wood => "WOOD",
        BoomerangKind::Magic => "MAGIC",
    }
}

/// Python's `f"{x:g}"` for the health values (halves and quarters).
fn g(x: f64) -> String {
    let s = format!("{x}");
    s.strip_suffix(".0").map_or(s.clone(), str::to_owned)
}

/// Multi-line summary: location, Link, inventory, room, entities, legal actions.
pub fn describe_state(state: &GameView, mask: Option<&ActionMask>) -> String {
    let link = &state.link;
    let tile = link.tile();
    let mut lines = vec![
        format!(
            "Location: level {} room 0x{:02x}{}  mode={}",
            state.level(),
            state.location(),
            if state.in_cave() { " (cave)" } else { "" },
            state.mode()
        ),
        format!(
            "Link: pos=({},{}) tile=({},{}) facing={} hp={}/{} status=0x{:02x}",
            link.position.x,
            link.position.y,
            tile.0,
            tile.1,
            link.direction.name(),
            g(state.health()),
            state.max_health(),
            link.status
        ),
        format!(
            "Inventory: sword={} bombs={} keys={} rupees={} boomerang={} bow={} arrows={} beams={} triforce={}",
            sword_name(state.sword()),
            state.bombs(),
            state.keys(),
            state.rupees(),
            boomerang_name(state.boomerang()),
            state.bow() as u8,
            arrow_name(state.arrows()),
            if state.has_beams() { "yes" } else { "no" },
            state.triforce_pieces()
        ),
    ];
    if state.level() != 0 {
        let doors: Vec<String> = CARDINALS
            .iter()
            .map(|&d| {
                let kind = if state.is_door_locked(d) {
                    "locked"
                } else if state.is_door_barred(d) {
                    "barred"
                } else if state.room().is_door_tile_walkable(d) {
                    "open"
                } else {
                    "wall"
                };
                format!("{}={kind}", d.name())
            })
            .collect();
        let walls: Vec<String> = state
            .bomb_walls()
            .iter()
            .map(|w| format!("'{}'", w.wall_direction.name()))
            .collect();
        lines.push(format!(
            "Doors: {}  push_block_pending={}  bomb_walls=[{}]",
            doors.join(" "),
            if state.push_block_pending() {
                "True"
            } else {
                "False"
            },
            walls.join(", ")
        ));
    }
    if let Some(t) = state.treasure() {
        let tile = t.tile();
        lines.push(format!(
            "Treasure: at ({},{}) tile=({},{})",
            t.position.x, t.position.y, tile.0, tile.1
        ));
    }
    lines.push(format!(
        "Entities ({} enemies, {} items, {} projectiles, {} hazards):",
        state.enemies.len(),
        state.items.len(),
        state.projectiles.len(),
        state.hazards.len()
    ));
    let mut enemies: Vec<_> = state.enemies.iter().collect();
    enemies.sort_by_key(|e| e.index);
    for e in enemies {
        let mut flags = Vec::new();
        if !e.is_active() {
            flags.push("inactive".to_owned());
        }
        if e.is_dying() {
            flags.push("dying".to_owned());
        }
        if e.is_stunned() {
            flags.push(format!("stun={}", e.stun_timer));
        }
        let tile = e.tile();
        lines.push(format!(
            "  [{:2}] enemy {} pos=({},{}) tile=({},{}) hp={} dir={} dist={:.0} {}",
            e.index,
            enemy_name(e.id),
            e.position.x,
            e.position.y,
            tile.0,
            tile.1,
            e.health,
            e.direction.name(),
            e.distance,
            flags.join(" ")
        ));
    }
    let mut items: Vec<_> = state.items.iter().collect();
    items.sort_by_key(|i| i.index);
    for i in items {
        lines.push(format!(
            "  [{:2}] item {} pos=({},{}) timer={} dist={:.0}",
            i.index,
            item_name(i.id),
            i.position.x,
            i.position.y,
            i.timer,
            i.distance
        ));
    }
    let mut projectiles: Vec<_> = state.projectiles.iter().collect();
    projectiles.sort_by_key(|p| p.index);
    for p in projectiles {
        lines.push(format!(
            "  [{:2}] projectile {} pos=({},{}) dir={} dist={:.0}",
            p.index,
            projectile_name(p.id),
            p.position.x,
            p.position.y,
            p.direction.name(),
            p.distance
        ));
    }
    let mut hazards: Vec<_> = state.hazards.iter().collect();
    hazards.sort_by_key(|h| h.index);
    for h in hazards {
        let name = match h.id {
            0x1F => "BoulderSet".to_owned(),
            0x20 => "Boulder".to_owned(),
            0x2B..=0x2D => enemy_name(h.id),
            0x40 => "StandingFire".to_owned(),
            0x49 | 0x4A => "Trap".to_owned(),
            id => id.to_string(),
        };
        lines.push(format!(
            "  [{:2}] hazard {} pos=({},{}) dist={:.0}",
            h.index, name, h.position.x, h.position.y, h.distance
        ));
    }
    if let Some(mask) = mask {
        lines.push(format!("Legal: {mask}"));
    }
    lines.join("\n")
}

/// ASCII map of the 32x22 tile grid. Legend: '.' walkable, '#' blocked, 'L' Link, 'E'/'e' active/inactive
/// enemy, 'i' item, '*' projectile, 'H' hazard, 'T' treasure, 'K' locked door, '=' barred door, 'P' push block,
/// 'C' cave.
pub fn render_grid(state: &GameView) -> String {
    let room = state.room();
    let (cols, rows) = (COLS, ROWS);
    let mut grid: Vec<Vec<char>> = (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    if room.is_tile_walkable(c, r) {
                        '.'
                    } else {
                        '#'
                    }
                })
                .collect()
        })
        .collect();
    let set = |grid: &mut Vec<Vec<char>>, (c, r): Tile, ch: char| {
        if (0..cols).contains(&c) && (0..rows).contains(&r) {
            grid[r as usize][c as usize] = ch;
        }
    };
    if state.level() == 0 {
        for c in 0..cols {
            for r in 0..rows {
                if room.tile(c, r) == TOP_CAVE_TILE {
                    set(&mut grid, (c, r), 'C');
                }
            }
        }
    }
    for t in state.push_block_tiles() {
        set(&mut grid, t, 'P');
    }
    if state.level() != 0 {
        for (d, t) in [
            (Direction::N, NORTH_DOOR_TILE),
            (Direction::S, SOUTH_DOOR_TILE),
            (Direction::E, EAST_DOOR_TILE),
            (Direction::W, WEST_DOOR_TILE),
        ] {
            if state.is_door_locked(d) {
                set(&mut grid, t, 'K');
            } else if state.is_door_barred(d) {
                set(&mut grid, t, '=');
            }
        }
    }
    let mark = |grid: &mut Vec<Vec<char>>, tile: Tile, ch: char, w: i32, h: i32| {
        for dx in 0..w {
            for dy in 0..h {
                set(grid, (tile.0 + dx, tile.1 + dy), ch);
            }
        }
    };
    if let Some(t) = state.treasure() {
        mark(&mut grid, t.tile(), 'T', 1, 1);
    }
    for i in &state.items {
        mark(&mut grid, i.tile(), 'i', 1, 1);
    }
    for p in &state.projectiles {
        mark(&mut grid, p.tile(), '*', 1, 1);
    }
    for h in &state.hazards {
        mark(&mut grid, h.tile(), 'H', 2, 2);
    }
    for e in &state.enemies {
        mark(
            &mut grid,
            e.tile(),
            if e.is_active() { 'E' } else { 'e' },
            2,
            2,
        );
    }
    mark(&mut grid, state.link.tile(), 'L', 2, 2);
    let header: String = "    "
        .chars()
        .chain((0..cols).map(|c| char::from_digit((c % 16) as u32, 16).unwrap()))
        .collect();
    let mut out = vec![header];
    for (r, row) in grid.iter().enumerate() {
        out.push(format!("{r:3x} {}", row.iter().collect::<String>()));
    }
    out.join("\n")
}

/// One step's outcome: frames, interrupt, hits, items, health.
pub fn describe_change(change: &Change) -> String {
    let mut parts = vec![format!("{} -> {} frames", change.action, change.frames)];
    if let Some(i) = change.interrupt {
        parts.push(format!("interrupt={}", i.name()));
    }
    if change.changed_location() {
        parts.push(format!(
            "moved to level {} room 0x{:02x}",
            change.state.level(),
            change.state.location()
        ));
    }
    if !change.enemies_hit.is_empty() {
        let hits: Vec<String> = change
            .enemies_hit
            .iter()
            .map(|(s, d)| format!("{s}: {d}"))
            .collect();
        parts.push(format!("hits={{{}}}", hits.join(", ")));
    }
    if !change.enemies_stunned.is_empty() {
        let s: Vec<String> = change.enemies_stunned.iter().map(u8::to_string).collect();
        parts.push(format!("stunned=[{}]", s.join(", ")));
    }
    if !change.items_gained.is_empty() {
        let s: Vec<String> = change
            .items_gained
            .iter()
            .map(|i| format!("'{}'", item_name(*i)))
            .collect();
        parts.push(format!("items=[{}]", s.join(", ")));
    }
    if change.health_lost != 0.0 {
        parts.push(format!("health_lost={}", g(change.health_lost)));
    }
    if change.health_gained != 0.0 {
        parts.push(format!("health_gained={}", g(change.health_gained)));
    }
    parts.join("  ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_the_python_enums() {
        assert_eq!(enemy_name(0x06), "RedGoriya");
        assert_eq!(enemy_name(0x99), "153");
        assert_eq!(item_name(-1), "Key");
        assert_eq!(item_name(-103), "Triforce3");
        assert_eq!(item_name(-109), "TriforceOfPower");
        assert_eq!(projectile_name(83), "OtorokRock");
        assert_eq!(g(3.0), "3");
        assert_eq!(g(2.5), "2.5");
    }
}
