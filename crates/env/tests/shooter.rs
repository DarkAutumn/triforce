//! Schema 10's `entity_shooter` (`env::observation::shooter_block`) on hand-built RAM: a blue Goriya about to throw,
//! a red one with its timer running, one waiting for its boomerang, a Darknut beside them (no row); a Moblin in its
//! stand, a surfacing Zora, a red Wizzrobe, Aquamentus and Gohma on their timers, a blue Wizzrobe (no row).

use env::observation::{ENTITY_SHOOTER_SIZE, TREASURE_SLOT, feature_names, shooter_block};

fn col(name: &str) -> usize {
    feature_names().entity_shooter.iter().position(|n| n == name).unwrap_or_else(|| panic!("no column {name}"))
}

/// A Goriya of `obj_type` in RAM slot `slot`, standing (q-speed 0: no grid point ahead), wanting to shoot.
fn goriya(r: &mut [u8], slot: usize, obj_type: u8, timer: u8) {
    r[0x34F + slot] = obj_type;
    (r[0x70 + slot], r[0x84 + slot]) = (0x40, 0x8D);
    r[0x412 + slot] = 1;
    r[0x28 + slot] = timer;
}

#[test]
fn the_columns_read_the_throw_rule_per_goriya() {
    assert_eq!(feature_names().entity_shooter.len(), ENTITY_SHOOTER_SIZE);
    let mut r = vec![0u8; 0x800];
    (r[0x70], r[0x84]) = (0x60, 0x8D);
    goriya(&mut r, 1, 0x05, 1);
    goriya(&mut r, 2, 0x06, 0x38);
    goriya(&mut r, 3, 0x05, 0);
    r[0xAC + 3] = 0x80;                                 // its boomerang is out
    r[0x34F + 4] = 0x0B;                                // a Red Darknut
    let block = shooter_block(&r);
    let (loaded, time, will) = (col("loaded"), col("throw_time"), col("will_throw"));
    // Observation slot i is RAM slot i + 1.
    assert_eq!((block[0][loaded], block[0][will]), (1.0, 1.0));
    assert_eq!(block[0][time], 1.0 - 1.0 / 112.0);
    // The red one's timer reads 0 only past the 16 frames: loaded, no chance yet.
    assert_eq!((block[1][loaded], block[1][will]), (1.0, 0.0));
    assert_eq!(block[1][time], 0.5);
    assert_eq!(block[2], [0.0; ENTITY_SHOOTER_SIZE]);
    assert_eq!(block[3], [0.0; ENTITY_SHOOTER_SIZE]);
    assert_eq!(block[TREASURE_SLOT], [0.0; ENTITY_SHOOTER_SIZE]);
    // Its timer at 4: the red one may throw on frames 4-16, 2 in 256 a frame.
    r[0x28 + 2] = 4;
    let p = 1.0 - (1.0f32 - 2.0 / 256.0).powi(13);
    assert!((shooter_block(&r)[1][will] - p).abs() < 1e-6);
}

#[test]
fn the_other_shooters_fill_their_rows_by_their_timers() {
    let mut r = vec![0u8; 0x800];
    (r[0x34F + 1], r[0x451 + 1], r[0x412 + 1]) = (0x03, 0x14, 1);       // a Moblin 4 frames from its arrow
    (r[0x34F + 2], r[0xAC + 2], r[0x28 + 2]) = (0x11, 3, 0xFF);         // a Zora 2 frames from its fireball
    (r[0x34F + 3], r[0xAC + 3]) = (0x24, 0xB8);                         // a red Wizzrobe 8 frames from its shot
    r[0x34F + 4] = 0x23;                                                // a blue Wizzrobe: not covered
    (r[0x34F + 5], r[0x28 + 5]) = (0x3D, 0x70);                         // Aquamentus, its volley 112 frames off
    (r[0x34F + 6], r[0x380 + 6]) = (0x33, 0x10);                        // Gohma, its fireball 16 frames off
    (r[0x34F + 7], r[0x451 + 7]) = (0x09, 0);                           // a blue Octorok walking: 33 frames at least
    let block = shooter_block(&r);
    let (loaded, time, will) = (col("loaded"), col("throw_time"), col("will_throw"));
    let row = |i: usize| (block[i][loaded], block[i][time], block[i][will]);
    assert_eq!(row(0), (1.0, 1.0 - 4.0 / 112.0, 1.0));
    assert_eq!(row(1), (1.0, 1.0 - 2.0 / 112.0, 1.0));
    assert_eq!(row(2), (1.0, 1.0 - 8.0 / 112.0, 1.0));
    assert_eq!(block[3], [0.0; ENTITY_SHOOTER_SIZE]);
    assert_eq!(row(4), (1.0, 0.0, 0.0));
    assert_eq!(row(5), (1.0, 1.0 - 16.0 / 112.0, 1.0));
    assert_eq!(row(6), (1.0, 1.0 - 33.0 / 112.0, 0.0));
}
