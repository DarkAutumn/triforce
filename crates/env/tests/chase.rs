//! Schema 9's `chase` and `entity_chase` (`env::observation::chase_blocks`) on hand-built RAM: the RAM bytes the
//! blocks read, column by column.

use env::observation::{CHASE_SIZE, ENTITY_CHASE_SIZE, chase_blocks, feature_names};

fn col(names: &[String], name: &str) -> usize {
    names.iter().position(|n| n == name).unwrap_or_else(|| panic!("no column {name}"))
}

/// 2 KB of RAM with Link's corner at (x, y), the chase target on him.
fn ram(x: u8, y: u8) -> Vec<u8> {
    let mut r = vec![0u8; 0x800];
    (r[0x70], r[0x84]) = (x, y);
    (r[0x61], r[0x62]) = (x, y);
    r
}

#[test]
fn the_global_block_reads_the_target_the_flip_and_the_hotspot() {
    let names = &feature_names().chase;
    assert_eq!(names.len(), CHASE_SIZE);
    // Link on the hotspot's square: his mirror 1 px off on X, 5 on Y; the target on him.
    let mut r = ram(128, 125);
    r[0x4A] = 3; // ChaseLongTimer
    r[0x26] = 4; // LongTimerCycle: ticks in 5 frames, then 10 apart
    let (g, _) = chase_blocks(&r);
    assert_eq!(g[col(names, "other")], 0.0);
    assert_eq!((g[col(names, "target_dx")], g[col(names, "target_dy")]), (0.0, 0.0));
    assert_eq!(g[col(names, "flip_time")], 25.0 / 70.0);
    assert_eq!(g[col(names, "hotspot_dx")], 0.5 / 128.0);
    assert_eq!(g[col(names, "hotspot_dy")], -2.5 / 128.0);
    assert_eq!(g[col(names, "mirror_gap")], 5.0 / 255.0);
    // The mirror spell, Link in the room's lower left: the target on the far side of the point.
    let mut r = ram(48, 173);
    (r[0x60], r[0x61], r[0x62]) = (1, 48 ^ 0xFF, 173 ^ 0xFF);
    let (g, _) = chase_blocks(&r);
    assert_eq!(g[col(names, "other")], 1.0);
    assert_eq!(g[col(names, "target_dx")], (207.0 - 48.0) / 255.0);
    assert_eq!(g[col(names, "target_dy")], (82.0 - 173.0) / 255.0);
    assert_eq!(g[col(names, "flip_time")], 1.0 / 70.0);
    assert_eq!(g[col(names, "mirror_gap")], 159.0 / 255.0);
}

#[test]
fn a_darknut_walking_away_on_links_row_may_turn_straight_back() {
    let names = &feature_names().entity_chase;
    assert_eq!(names.len(), ENTITY_CHASE_SIZE);
    // A Red Darknut in RAM slot 2 walking left on Link's row, 8 px from its next grid point; Link 24 px behind it.
    let mut r = ram(0x78, 0x8D);
    let s = 2;
    r[0x34F + s] = 0x0B;
    (r[0x70 + s], r[0x84 + s]) = (0x60, 0x8D);
    (r[0x98 + s], r[0x3F8 + s]) = (2, 2);
    r[0x3BC + s] = 0x20;
    r[0x394 + s] = 0xF8;
    r[0x478 + s] = 0xFF;
    let (_, per) = chase_blocks(&r);
    let row = &per[s - 1];
    assert!(per.iter().enumerate().all(|(i, p)| i == s - 1 || p.iter().all(|&v| v == 0.0)));
    assert_eq!(row[col(names, "turns")], 1.0);
    // 8 px at half a pixel a frame; walking left from a fraction of 0 the first sub-step borrows at once: pixels
    // on frames 1, 3, ..., 15.
    assert_eq!(row[col(names, "grid_time")], 15.0 / 64.0);
    assert_eq!(row[col(names, "toward")], 1.0);
    assert_eq!(row[col(names, "toward_E")], 1.0);
    assert_eq!(row[col(names, "reverse")], 1.0);
    assert_eq!(row[col(names, "reverse_link")], 1.0);
    assert_eq!(row[col(names, "timer_turn")], 0.0);
    assert!(row[col(names, "chance")] >= 129.0 / 256.0);
    // Shadowing 20 px below its row: no turn toward Link, none back along its row.
    let mut r2 = r.clone();
    (r2[0x84], r2[0x62]) = (0x8D + 20, 0x8D + 20);
    let (_, per) = chase_blocks(&r2);
    let row = &per[s - 1];
    assert_eq!((row[col(names, "toward")], row[col(names, "reverse_link")], row[col(names, "chance")]), (0.0, 0.0, 0.0));
    // Off with the clock: it makes no decision.
    r[0x66C] = 1;
    assert!(chase_blocks(&r).1.iter().all(|p| p.iter().all(|&v| v == 0.0)));
}
