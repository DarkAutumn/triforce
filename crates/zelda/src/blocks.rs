//! Confirmed shield collisions from consecutive executed frames.
//!
//! Z_01 CheckLinkCollision, Z_04 CheckShotLinkCollision / UpdateFireball / UpdateMonsterArrow and Z_07
//! DrawBoomerangAndCheckCollision: require the projectile's collision signature, the shield eligibility, facing
//! and proximity checks and this frame's parry request ($604 bit 0).

use crate::enums::MapLocation;

const BASIC_SHOTS: [u8; 4] = [0x53, 0x54, 0x5B, 0x5C];
const MAGIC_SHOTS: [u8; 4] = [0x55, 0x57, 0x58, 0x59];

fn blockable(kind: u8) -> bool {
    BASIC_SHOTS.contains(&kind) || MAGIC_SHOTS.contains(&kind)
}

/// One actual collision; `frame` is the session frame counter after the colliding frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectileBlock {
    pub frame: u64,
    pub slot: u8,
    pub projectile_id: u8,
    pub full_location: MapLocation,
}

#[derive(Clone, Copy, Debug)]
struct Shot {
    slot: usize,
    kind: u8,
    direction: u8,
    attributes: u8,
    x: u8,
    y: u8,
    can_collide: bool,
}

/// Outgoing blockable shots before a frame (`ShieldCollisionWatch`).
#[derive(Clone, Debug, Default)]
pub struct ShieldWatch {
    shots: Vec<Shot>,
    world: (u8, u8, u8),
    health: (u8, u8),
}

impl ShieldWatch {
    pub fn new(ram: &[u8; 0x800]) -> ShieldWatch {
        let read = |a: usize| ram[a];
        let mut watch = ShieldWatch::default();
        for slot in 1..13 {
            let kind = read(0x34F + slot);
            if blockable(kind) && read(0xAC + slot) & 0xF0 == 0x10 {
                let direction = read(0x98 + slot);
                let (x, y) = (read(0x70 + slot), read(0x84 + slot));
                let mut can_collide = true;
                if kind == 0x55 && read(0x28 + slot) == 0 {
                    let horizontal = if x < read(0x346) {
                        2
                    } else if x >= read(0x347) {
                        1
                    } else {
                        0
                    };
                    let vertical = if y < read(0x348) {
                        8
                    } else if y >= read(0x349) {
                        4
                    } else {
                        0
                    };
                    can_collide = direction & (horizontal | vertical) == 0;
                }
                watch.shots.push(Shot {
                    slot,
                    kind,
                    direction,
                    attributes: read(0x4BF + slot),
                    x,
                    y,
                    can_collide,
                });
            }
        }
        if !watch.shots.is_empty() {
            watch.world = (read(0x10), read(0xEB), read(0x12));
            watch.health = (read(0x66F), read(0x670));
        }
        watch
    }

    /// The collisions the frame after `new` produced.
    pub fn collisions(&self, ram: &[u8; 0x800], frame: u64) -> Vec<ProjectileBlock> {
        let read = |a: usize| ram[a];
        if self.shots.is_empty() || read(0x604) & 1 == 0 {
            return Vec::new();
        }
        let (level, room, mode) = self.world;
        if !matches!(mode, 5 | 11) || self.world != (read(0x10), read(0xEB), read(0x12)) {
            return Vec::new();
        }
        if read(0x66C) != 0
            || read(0x4F0) != 0
            || read(0x3D) != 0
            || read(0x512) != 0
            || read(0xAC) & 0xF0 != 0
            || (read(0x66F), read(0x670)) < self.health
        {
            return Vec::new();
        }
        let facing = read(0x98);
        let (lx, ly) = (read(0x70) as i32, read(0x84) as i32);
        let mut events = Vec::new();
        for shot in &self.shots {
            let slot = shot.slot;
            if !shot.can_collide || read(0x3D + slot) != 0 {
                continue;
            }
            let combined = facing | shot.direction;
            if combined & 0x0C != 0x0C && combined & 0x03 != 0x03 {
                continue;
            }
            if MAGIC_SHOTS.contains(&shot.kind) && read(0x676) == 0 {
                continue;
            }
            let (current_kind, phase) = (read(0x34F + slot), read(0xAC + slot));
            let signature = if shot.kind == 0x5C {
                current_kind == shot.kind
                    && phase == 0x20
                    && read(0x3D0 + slot) == 3
                    && (read(0x70 + slot), read(0x84 + slot)) != (shot.x, shot.y)
            } else if shot.kind == 0x55 {
                current_kind == 0
                    && phase == 0
                    && read(0x380 + slot) == facing
                    && read(0x394 + slot) == 0
            } else {
                current_kind == shot.kind
                    && phase == 0x30
                    && read(0x380 + slot) == facing
                    && read(0x394 + slot) == 0
            };
            if !signature {
                continue;
            }
            let half = if shot.attributes & 0x40 != 0 { 4 } else { 8 };
            let dx = (read(0x70 + slot) as i32 + half - lx - 8) & 0xFF;
            let dy = (read(0x84 + slot) as i32 - ly) & 0xFF;
            if dx.min(256 - dx) >= 9 || dy.min(256 - dy) >= 9 {
                continue;
            }
            events.push(ProjectileBlock {
                frame,
                slot: slot as u8,
                projectile_id: shot.kind,
                full_location: MapLocation::new(level, room, mode == 11),
            });
        }
        events
    }
}

#[cfg(test)]
pub(crate) mod tests {
    //! Port of the shield-collision unit tests in tests/test_core.py.
    use super::*;

    /// Shield-eligible Link with one outgoing hostile shot, already within the collision box.
    pub(crate) fn shield_ram(kind: u8) -> [u8; 0x800] {
        let mut ram = [0u8; 0x800];
        for (a, v) in [
            (0x10, 1),
            (0x12, 5),
            (0xEB, 0x44),
            (0x70, 128),
            (0x84, 144),
            (0x98, 1),
            (0x66F, 0x22),
            (0x670, 255),
            (0x34F + 11, kind),
            (0xAC + 11, 0x17),
            (0x70 + 11, 140),
            (0x84 + 11, 144),
            (0x98 + 11, 2),
            (0x4BF + 11, 0xE1),
            (0x3D0 + 11, 2),
        ] {
            ram[a] = v;
        }
        ram[0x346..0x34A].copy_from_slice(&[24, 224, 80, 200]);
        ram
    }

    pub(crate) fn parry_frame(ram: &mut [u8; 0x800], kind: u8) {
        ram[0x604] = 1;
        ram[0x70 + 11] = 137;
        ram[0xAC + 11] = if kind == 0x5C { 0x20 } else { 0x30 };
        ram[0x3D0 + 11] = 3;
        ram[0x380 + 11] = ram[0x98];
        ram[0x394 + 11] = 0;
        if kind == 0x55 {
            ram[0x34F + 11] = 0;
            ram[0xAC + 11] = 0;
        }
    }

    fn block(frame: u64, kind: u8) -> ProjectileBlock {
        ProjectileBlock {
            frame,
            slot: 11,
            projectile_id: kind,
            full_location: MapLocation::new(1, 0x44, false),
        }
    }

    #[test]
    fn shield_collision_types_use_request_and_signature() {
        for kind in [0x53, 0x55, 0x57, 0x5C] {
            let mut ram = shield_ram(kind);
            ram[0x676] = 1;
            let before = ShieldWatch::new(&ram);
            parry_frame(&mut ram, kind);
            assert_eq!(ram[0x605], 0);
            assert_eq!(
                before.collisions(&ram, 100),
                vec![block(100, kind)],
                "kind {kind:#x}"
            );
            let repeated = ShieldWatch::new(&ram);
            ram[0x605] = 1;
            assert!(repeated.collisions(&ram, 101).is_empty());
        }
    }

    #[test]
    fn shield_collision_rejects_clock_stun_wrong_facing_damage_and_missing_request() {
        for (a, v) in [
            (0x66C, 1),
            (0x4F0, 1),
            (0x3D, 1),
            (0x3D + 11, 1),
            (0x512, 1),
            (0xAC, 0x10),
            (0xAC, 0x40),
            (0x98, 2),
            (0x66F, 0x21),
            (0x670, 127),
            (0x604, 0),
            (0x12, 6),
        ] {
            let mut ram = shield_ram(0x5C);
            let before = ShieldWatch::new(&ram);
            parry_frame(&mut ram, 0x5C);
            ram[a] = v;
            assert!(before.collisions(&ram, 100).is_empty(), "{a:#x}={v:#x}");
        }
    }

    #[test]
    fn boomerang_disappearance_and_natural_return_are_not_blocks() {
        for phase in [0, 0x10, 0x28, 0x30, 0x40, 0x50] {
            let mut ram = shield_ram(0x5C);
            let before = ShieldWatch::new(&ram);
            parry_frame(&mut ram, 0x5C);
            ram[0xAC + 11] = phase;
            assert!(before.collisions(&ram, 100).is_empty());
        }
    }

    #[test]
    fn magic_shield_does_not_block_unshieldable_fireballs() {
        let mut ram = shield_ram(0x56);
        ram[0x676] = 1;
        let before = ShieldWatch::new(&ram);
        parry_frame(&mut ram, 0x56);
        assert!(before.collisions(&ram, 100).is_empty());
    }

    #[test]
    fn immune_clink_with_nearby_unchanged_projectile_is_not_a_block() {
        let mut ram = shield_ram(0x5C);
        let before = ShieldWatch::new(&ram);
        ram[0x604] = 1;
        ram[0x605] = 1;
        assert!(before.collisions(&ram, 100).is_empty());
    }

    #[test]
    fn magic_shots_require_magic_shield() {
        for kind in [0x55, 0x57] {
            let mut ram = shield_ram(kind);
            let before = ShieldWatch::new(&ram);
            parry_frame(&mut ram, kind);
            assert!(before.collisions(&ram, 100).is_empty());
        }
    }

    #[test]
    fn fireball_disappearance_needs_collision_bounce_evidence() {
        for (a, v) in [(0x380 + 11, 2), (0x394 + 11, 2)] {
            let mut ram = shield_ram(0x55);
            ram[0x676] = 1;
            let before = ShieldWatch::new(&ram);
            parry_frame(&mut ram, 0x55);
            ram[a] = v;
            assert!(before.collisions(&ram, 100).is_empty());
        }
    }

    #[test]
    fn shield_collision_uses_half_width_midpoint_and_strict_axis_thresholds() {
        let mut ram = shield_ram(0x5C);
        let before = ShieldWatch::new(&ram);
        parry_frame(&mut ram, 0x5C);
        ram[0x70 + 11] = 140;
        ram[0x84 + 11] = 152;
        assert_eq!(before.collisions(&ram, 100).len(), 1);
        ram[0x70 + 11] = 141;
        assert!(before.collisions(&ram, 100).is_empty());
    }

    #[test]
    fn unrelated_clink_does_not_credit_a_stationary_wall_spark_or_boundary_fireball() {
        let mut ram = shield_ram(0x5C);
        let before = ShieldWatch::new(&ram);
        parry_frame(&mut ram, 0x5C);
        ram[0x70 + 11] = 140;
        assert!(before.collisions(&ram, 100).is_empty());

        let mut ram = shield_ram(0x55);
        ram[0x676] = 1;
        ram[0x70] = 16;
        ram[0x70 + 11] = 16;
        let before = ShieldWatch::new(&ram);
        parry_frame(&mut ram, 0x55);
        ram[0x70 + 11] = 16;
        assert!(before.collisions(&ram, 100).is_empty());
    }

    #[test]
    fn real_before_trace_credits_collision_even_when_sound_endpoints_are_silent() {
        let mut ram = shield_ram(0x5C);
        (ram[0x70], ram[0x84], ram[0x98], ram[0x66F], ram[0x670]) = (80, 181, 2, 33, 254);
        (ram[0x70 + 11], ram[0x84 + 11], ram[0x98 + 11]) = (77, 173, 1);
        let before = ShieldWatch::new(&ram);
        (ram[0x70 + 11], ram[0xAC + 11], ram[0x3D0 + 11], ram[0x604]) = (80, 0x20, 3, 1);
        assert_eq!(before.collisions(&ram, 2974), vec![block(2974, 0x5C)]);
    }
}
