//! `GameView` decoding on hand-built RAM images (no emulator): object slots, health, inventory, weapons and
//! movement bounds.

use std::sync::{Arc, LazyLock};
use zelda::enums::{
    ActionKind, AnimationKind, AnimationState, ArrowKind, BoomerangKind, Direction, ObjectClass,
    Position, SwordKind, enemy, item, projectile,
};
use zelda::game::{GameView, MODE_CAVE, MODE_DYING, MODE_GAME_OVER};
use zelda::tables::{self, RAM_SIZE};
use zelda::{GameMap, Var};

static MAP: LazyLock<Arc<GameMap>> = LazyLock::new(|| Arc::new(GameMap::builtin()));

/// A RAM image of a playable dungeon room with Link at (120, 141), facing north.
#[derive(Clone)]
struct Ram([u8; RAM_SIZE]);

impl Ram {
    fn new() -> Ram {
        let mut r = Ram([0; RAM_SIZE]);
        r.set(Var::Level, 1)
            .set(Var::Location, 0x44)
            .set(Var::Mode, 5)
            .set(Var::LinkX, 120)
            .set(Var::LinkY, 141)
            .set(Var::LinkDirection, 8)
            .set(Var::TreasureFlag, 0xFF);
        r
    }

    fn set(&mut self, var: Var, value: u8) -> &mut Ram {
        self.0[var.addr() as usize] = value;
        self
    }

    fn table(&mut self, t: (usize, usize), slot: usize, value: u8) -> &mut Ram {
        self.0[t.0 + slot] = value;
        self
    }

    /// Object `id` in `slot` at (x, y).
    fn object(&mut self, slot: usize, id: u8, x: u8, y: u8) -> &mut Ram {
        self.table(tables::OBJ_ID, slot, id)
            .table(tables::OBJ_POS_X, slot, x)
            .table(tables::OBJ_POS_Y, slot, y)
    }

    fn view(&self) -> GameView {
        GameView::new(&self.0, 0, false, MAP.clone())
    }
}

#[test]
fn object_slots_decode_into_enemies_items_and_projectiles_by_distance() {
    let mut r = Ram::new();
    r.object(1, enemy::STALFOS, 120, 181) // 40 below Link
        .table(tables::OBJ_HEALTH, 1, 0x3F)
        .table(tables::OBJ_DIRECTION, 1, 2)
        .object(2, enemy::RED_GORIYA, 130, 141) // 10 right
        .object(3, 0x40, 120, 150) // not a monster
        .object(4, 0x60, 110, 141) // dropped item, 10 left
        .table(tables::OBJ_STATUS, 4, item::HEART as u8)
        .table(tables::ITEM_TIMER, 4, 0x30)
        .object(5, projectile::ARROW, 100, 141)
        .object(6, 0x63, 100, 141) // decorations and the push block are not projectiles
        .object(7, 0x68, 100, 141)
        .object(8, 0x6A, 100, 141)
        .object(9, 0x64, 100, 141)
        .object(10, enemy::BLUE_GORIYA, 110, 141); // ties the Goriya in slot 2
    let v = r.view();
    let enemies: Vec<(u8, u8)> = v.enemies.iter().map(|e| (e.index, e.id)).collect();
    assert_eq!(
        enemies,
        [
            (2, enemy::RED_GORIYA),
            (10, enemy::BLUE_GORIYA),
            (1, enemy::STALFOS)
        ],
        "by distance, ties in slot order"
    );
    let stalfos = v.enemy_by_index(1).unwrap();
    assert_eq!(stalfos.health, 3, "health is the high nibble");
    assert_eq!(stalfos.direction, Direction::W);
    assert_eq!((stalfos.distance, stalfos.vector), (40.0, (0.0, 1.0)));
    assert_eq!(v.items.len(), 1);
    let heart = v.item_by_index(4).unwrap();
    assert_eq!(
        (heart.id, heart.timer, heart.vector),
        (item::HEART, 0x30, (-1.0, 0.0))
    );
    let shots: Vec<u8> = v.projectiles.iter().map(|p| p.index).collect();
    assert_eq!(shots, [5]);
    assert!(v.projectiles[0].blockable());
    assert_eq!(v.link.position, Position::new(120, 141));
    assert_eq!(v.link.direction, Direction::N);
}

/// F067: every object type's class follows asm:Obj_Update_JumpTable (types >= $6A are cave people); Gleeok's flying
/// head ($46), which never dies (asm:Gleeok_UpdateFlyingHead), is a hazard.
#[test]
fn object_classes_follow_the_jump_table() {
    use ObjectClass::*;
    let table: [(&str, ObjectClass); 0x6A] = [
        ("DoNothing", Empty),
        ("Lynel", Monster), ("Lynel", Monster), ("Moblin", Monster), ("Moblin", Monster),
        ("Goriya", Monster), ("Goriya", Monster), ("Octorock", Monster), ("Octorock", Monster),
        ("Octorock", Monster), ("Octorock", Monster), ("Darknut", Monster), ("Darknut", Monster),
        ("TektiteOrBoulder", Monster), ("TektiteOrBoulder", Monster), ("BlueLeever", Monster),
        ("RedLeever", Monster), ("Zora", Monster), ("Vire", Monster), ("Zol", Monster), ("Gel", Monster),
        ("Gel", Monster), ("PolsVoice", Monster), ("LikeLike", Monster), ("Digdogger", Monster),
        ("DoNothing", Other), ("Peahat", Monster), ("Keese", Monster), ("Keese", Monster), ("Keese", Monster),
        ("Armos", Monster), ("BoulderSet", Hazard), ("TektiteOrBoulder", Hazard), ("Ghini", Monster),
        ("FlyingGhini", Monster), ("BlueWizzrobe", Monster), ("RedWizzrobe", Monster), ("PatraChild", Monster),
        ("PatraChild", Monster), ("Wallmaster", Monster), ("Rope", Monster), ("DoNothing", Other),
        ("Stalfos", Monster), ("Bubble", Hazard), ("Bubble", Hazard), ("Bubble", Hazard), ("Whirlwind", Other),
        ("PondFairy", Other), ("Gibdo", Monster), ("Dodongo", Monster), ("Dodongo", Monster), ("Gohma", Monster),
        ("Gohma", Monster), ("RupeeStash", Other), ("Grumble", Other), ("Zelda", Other), ("Digdogger", Monster),
        ("Digdogger", Monster), ("Lamnola", Monster), ("Lamnola", Monster), ("Manhandla", Monster),
        ("Aquamentus", Monster), ("Ganon", Monster), ("GuardFire", Monster), ("StandingFire", Hazard),
        ("Moldorm", Monster), ("Gleeok", Monster), ("Gleeok", Monster), ("Gleeok", Monster), ("Gleeok", Monster),
        ("GleeokHead", Hazard), ("Patra", Monster), ("Patra", Monster), ("Trap", Hazard), ("Trap", Hazard),
        ("UnderworldPerson", Person), ("UnderworldPerson", Person), ("UnderworldPerson", Person),
        ("UnderworldPerson", Person), ("UnderworldPerson", Person), ("UnderworldPerson", Person),
        ("UnderworldPersonLifeOrMoney", Person), ("UnderworldPerson", Person), ("MonsterShot", Shot),
        ("MonsterShot", Shot), ("Fireball", Shot), ("Fireball", Shot), ("MonsterShot", Shot),
        ("MonsterShot", Shot), ("MonsterShot", Shot), ("MonsterShot", Shot), ("MonsterArrow", Shot),
        ("ArrowOrBoomerang", Shot), ("DeadDummy", Other), ("FluteSecret", Other), ("DoNothing", Other),
        ("Item", Item), ("Dock", TileObject), ("RockOrGravestone", TileObject), ("RockWall", TileObject),
        ("Tree", TileObject), ("RockOrGravestone", TileObject), ("RockOrGravestone", TileObject),
        ("RockWall", TileObject), ("Block", TileObject), ("DoNothing", Other),
    ];
    for (id, (name, class)) in table.iter().enumerate() {
        assert_eq!(ObjectClass::of(id as u8), *class, "${id:02X} {name}");
    }
    for id in 0x6A..=0xFF {
        assert_eq!(ObjectClass::of(id), Person, "${id:02X} CavePerson");
    }
    // Hazards hurt but the invisible boulder generator; people, items and tile objects never do.
    assert!(ObjectClass::can_hurt(0x20) && ObjectClass::can_hurt(0x49) && ObjectClass::can_hurt(0x2B));
    assert!(!ObjectClass::can_hurt(0x1F) && !ObjectClass::can_hurt(0x6B) && !ObjectClass::can_hurt(0x61));
}

/// F067/F038: boulders, traps and bubbles are hazards; old men, docks and the boulder generator are neither enemies
/// nor projectiles.
#[test]
fn hazards_and_non_entities_leave_the_enemy_and_projectile_lists() {
    let mut r = Ram::new();
    r.object(1, 0x1F, 112, 157)
        .object(2, 0x20, 100, 100)
        .object(3, 0x49, 60, 100)
        .object(4, enemy::BUBBLE, 80, 100)
        .object(5, 0x6B, 120, 120)
        .object(6, 0x61, 140, 120);
    let v = r.view();
    assert!(v.enemies.is_empty(), "{:?}", v.enemies);
    assert!(!v.has_active_enemies());
    let hazards: std::collections::BTreeSet<u8> = v.hazards.iter().map(|h| h.index).collect();
    assert_eq!(hazards, [1, 2, 3, 4].into());
    assert!(v.projectiles.is_empty());
}

#[test]
fn an_object_on_link_has_no_direction() {
    let mut r = Ram::new();
    r.object(1, enemy::STALFOS, 120, 141);
    let e = &r.view().enemies[0];
    assert_eq!((e.distance, e.vector), (0.0, (0.0, 0.0)));
}

#[test]
fn activity_depends_on_enemy_kind() {
    let state = |id: u8, status: u8, spawn: u8| {
        let mut r = Ram::new();
        r.object(1, id, 20, 100)
            .table(tables::OBJ_STATUS, 1, status)
            .table(tables::OBJ_SPAWN_STATE, 1, spawn)
            .table(tables::OBJ_HEALTH, 1, 0x10);
        r.view().enemies[0].clone()
    };
    for (id, status, active) in [
        (enemy::RED_LEEVER, 3, true),
        (enemy::BLUE_LEEVER, 2, false), // still burrowed
        (enemy::ZORA, 1, false),
        (enemy::ZORA, 2, true),
        (enemy::ZORA, 4, true),
        (enemy::ZORA, 5, false),
        (enemy::WALLMASTER, 1, true),
        (enemy::WALLMASTER, 0, false),
    ] {
        assert_eq!(
            state(id, status, 0).is_active(),
            active,
            "{id:#x} status {status}"
        );
    }
    assert!(state(enemy::STALFOS, 0, 0).is_active());
    assert!(!state(enemy::STALFOS, 0, 1).is_active(), "still spawning");
    for spawn in 0x10..=0x13 {
        assert!(state(enemy::STALFOS, 0, spawn).is_dying());
    }
    assert!(!state(enemy::STALFOS, 0, 0x14).is_dying());

    // Dying enemies are not active enemies.
    let mut r = Ram::new();
    r.object(1, enemy::STALFOS, 20, 100)
        .table(tables::OBJ_SPAWN_STATE, 1, 0x10);
    assert!(!r.view().has_active_enemies());
    r.table(tables::OBJ_SPAWN_STATE, 1, 0);
    assert!(r.view().has_active_enemies());
}

#[test]
fn aquamentus_is_three_tiles_wide() {
    let mut r = Ram::new();
    r.object(1, enemy::AQUAMENTUS, 160, 112)
        .object(2, enemy::STALFOS, 40, 112);
    let v = r.view();
    let aqua = v
        .enemies
        .iter()
        .find(|e| e.id == enemy::AQUAMENTUS)
        .unwrap();
    let stalfos = v.enemies.iter().find(|e| e.id == enemy::STALFOS).unwrap();
    assert_eq!(aqua.self_tiles().len(), 9);
    assert_eq!(aqua.link_overlap_tiles().len(), 16);
    assert_eq!(stalfos.self_tiles(), [(5, 7), (5, 8), (6, 7), (6, 8)]);
    // Overlap tiles: every top-left tile from which Link's 2x2 box touches it.
    assert_eq!(stalfos.link_overlap_tiles().len(), 9);
    assert!(stalfos.link_overlap_tiles().contains(&(4, 6)));
}

#[test]
fn health_decodes_filled_partial_and_containers() {
    let health = |hc: u8, partial: u8| {
        let mut r = Ram::new();
        r.set(Var::HeartsAndContainers, hc)
            .set(Var::PartialHearts, partial);
        let v = r.view();
        (v.health(), v.max_health(), v.is_health_full())
    };
    // Low nibble: filled hearts - 1 counted with the partial; high nibble: containers - 1.
    assert_eq!(health(0x22, 0xFF), (3.0, 3, true));
    assert_eq!(health(0x22, 0x80), (3.0, 3, true));
    assert_eq!(health(0x22, 0x7F), (2.5, 3, false));
    assert_eq!(health(0x22, 0x01), (2.5, 3, false));
    assert_eq!(health(0x22, 0x00), (2.0, 3, false));
    assert_eq!(health(0x20, 0x00), (0.0, 3, false));
    // Filled == containers drops the partial, more than containers is capped.
    assert_eq!(health(0x23, 0xFF), (3.0, 3, true));
    assert_eq!(health(0x2F, 0xFF), (3.0, 3, true));
    assert_eq!(health(0xFF, 0xFF), (16.0, 16, true));
}

#[test]
fn beams_need_a_sword_exactly_full_health_and_an_idle_beam() {
    let beams = |hc: u8, partial: u8, sword: u8, beam: u8| {
        let mut r = Ram::new();
        r.set(Var::HeartsAndContainers, hc)
            .set(Var::PartialHearts, partial)
            .set(Var::Sword, sword)
            .set(Var::BeamAnimation, beam);
        let v = r.view();
        (v.has_beams(), v.available_actions())
    };
    let (yes, actions) = beams(0x22, 0x80, 1, 0);
    assert!(yes);
    assert!(actions[ActionKind::Beams.index()] && !actions[ActionKind::Sword.index()]);
    assert!(!beams(0x22, 0x7F, 1, 0).0, "half a heart short");
    assert!(!beams(0x21, 0xFF, 1, 0).0, "one heart short");
    assert!(!beams(0x22, 0x80, 0, 0).0, "no sword");
    let (no, actions) = beams(0x22, 0x80, 1, 16);
    assert!(!no, "a beam is already flying");
    assert!(actions[ActionKind::Sword.index()] && !actions[ActionKind::Beams.index()]);
    let (_, actions) = beams(0x22, 0x80, 0, 0);
    assert!(!actions[ActionKind::Sword.index()] && !actions[ActionKind::Beams.index()]);
}

#[test]
fn b_item_actions_follow_the_inventory() {
    let actions = |vars: &[(Var, u8)]| {
        let mut r = Ram::new();
        for &(v, x) in vars {
            r.set(v, x);
        }
        r.view().available_actions()
    };
    let none = actions(&[]);
    assert_eq!(
        none,
        [true, true, false, false, false, false, false, false],
        "MOVE and WAIT only (CLIP is the session's)"
    );
    assert!(actions(&[(Var::Bombs, 1)])[ActionKind::Bomb.index()]);
    let bow_and_arrows = [(Var::Bow, 1), (Var::Arrows, 1), (Var::Rupees, 1)];
    assert!(actions(&bow_and_arrows)[ActionKind::Arrow.index()]);
    for missing in 0..3 {
        let mut vars = bow_and_arrows.to_vec();
        vars[missing].1 = 0;
        assert!(
            !actions(&vars)[ActionKind::Arrow.index()],
            "{:?}",
            vars[missing].0
        );
    }
    assert!(actions(&[(Var::RegularBoomerang, 1)])[ActionKind::Boomerang.index()]);
    assert!(actions(&[(Var::MagicBoomerang, 1)])[ActionKind::Boomerang.index()]);
}

#[test]
fn inventory_kinds_and_dungeon_items() {
    let mut r = Ram::new();
    assert_eq!(
        (r.view().arrows(), r.view().boomerang(), r.view().sword()),
        (ArrowKind::None, BoomerangKind::None, SwordKind::None)
    );
    r.set(Var::Arrows, 2)
        .set(Var::RegularBoomerang, 1)
        .set(Var::Sword, 2);
    assert_eq!(
        (r.view().arrows(), r.view().boomerang(), r.view().sword()),
        (ArrowKind::Silver, BoomerangKind::Wood, SwordKind::White)
    );
    r.set(Var::MagicBoomerang, 1);
    assert_eq!(r.view().boomerang(), BoomerangKind::Magic, "magic wins");

    // Compass/map: the level's bit, level 9 its own byte, never on the overworld.
    r.set(Var::Compass, 0b1).set(Var::Map, 0b100);
    let v = r.view();
    assert!(v.has_compass() && !v.has_map());
    r.set(Var::Level, 3);
    let v = r.view();
    assert!(!v.has_compass() && v.has_map());
    r.set(Var::Level, 9)
        .set(Var::Compass, 0xFF)
        .set(Var::Map, 0xFF);
    assert!(!r.view().has_compass() && !r.view().has_map());
    r.set(Var::Compass9, 1).set(Var::Map9, 1);
    assert!(r.view().has_compass() && r.view().has_map());
    r.set(Var::Level, 0);
    assert!(!r.view().has_compass() && !r.view().has_map());

    r.set(Var::Triforce, 0b101);
    let v = r.view();
    assert_eq!(v.triforce_pieces(), 2);
    assert!(v.has_triforce(1) && !v.has_triforce(2) && v.has_triforce(3));
    assert!(!v.has_triforce(0));
}

#[test]
fn treasure_is_an_item_only_while_its_flag_is_clear() {
    let mut r = Ram::new();
    r.set(Var::TreasureX, 120).set(Var::TreasureY, 101);
    assert_eq!(r.view().treasure(), None);
    r.set(Var::TreasureFlag, 0);
    let t = r.view().treasure().unwrap();
    assert_eq!((t.index, t.id, t.timer), (-1, item::TREASURE, 255));
    assert_eq!(
        (t.position, t.distance, t.vector),
        (Position::new(120, 101), 40.0, (0.0, -1.0))
    );
}

#[test]
fn modes_decide_cave_and_game_over() {
    let mut r = Ram::new();
    assert!(!r.view().in_cave() && !r.view().game_over());
    r.set(Var::Mode, MODE_CAVE);
    assert!(r.view().in_cave() && r.view().full_location().in_cave);
    for mode in [MODE_DYING, MODE_GAME_OVER] {
        r.set(Var::Mode, mode);
        assert!(r.view().game_over());
    }
}

#[test]
fn weapon_directions_respect_screen_bounds() {
    let dirs = |level: u8, x: u8, y: u8| {
        let mut r = Ram::new();
        r.set(Var::Level, level)
            .set(Var::LinkX, x)
            .set(Var::LinkY, y);
        let v = r.view();
        (v.sword_directions(), v.item_directions())
    };
    use Direction::*;
    assert_eq!(dirs(1, 120, 141), (vec![E, W, N, S], vec![E, W, N, S]));
    // Dungeon (measured, N006, with the facing poked): A on E/W x 23..=216, N y 84..=197, S y 85..=197; B also
    // E x <= 208, W x >= 31, S y <= 189. Outside the room's bounds (x < $21 or > $D0, y < $5E or > $BD) Link is
    // in a doorway, where the game never turns him sideways (measured: a perpendicular press does nothing, or
    // walks him back into the room first), so only the doorway's axis is left there.
    assert_eq!(dirs(1, 23, 141).0, vec![E, W]);
    assert_eq!(dirs(1, 22, 141), (vec![], vec![]));
    assert_eq!(dirs(1, 30, 141).1, vec![E], "too close to the west wall to throw west");
    assert_eq!(dirs(1, 31, 141).1, vec![E, W]);
    assert_eq!(dirs(1, 33, 141).1, vec![E, W, N, S]);
    assert_eq!(dirs(1, 208, 141).1, vec![E, W, N, S]);
    assert_eq!(dirs(1, 209, 141).1, vec![W]);
    assert_eq!(dirs(1, 216, 141).0, vec![E, W]);
    assert_eq!(dirs(1, 217, 141).0, vec![]);
    assert_eq!(dirs(1, 120, 83), (vec![], vec![]));
    assert_eq!(dirs(1, 120, 84), (vec![N], vec![N]));
    assert_eq!(dirs(1, 120, 85).1, vec![N, S]);
    assert_eq!(dirs(1, 120, 94).1, vec![E, W, N, S]);
    assert_eq!(dirs(1, 120, 189).1, vec![E, W, N, S]);
    assert_eq!(dirs(1, 120, 190).1, vec![N]);
    assert_eq!(dirs(1, 120, 197).0, vec![N, S]);
    assert_eq!(dirs(1, 120, 198), (vec![], vec![]));
    // Overworld: A on x 7..=232, y 69..=213; B also N y >= 84.
    assert_eq!(dirs(0, 7, 69), (vec![E, W, N, S], vec![E, S]));
    assert_eq!(dirs(0, 6, 68), (vec![], vec![]));
    assert_eq!(dirs(0, 233, 214), (vec![], vec![]));
    // Off every axis the sword is screen-locked: no beams.
    let mut r = Ram::new();
    r.set(Var::LinkX, 10)
        .set(Var::LinkY, 60)
        .set(Var::Sword, 1)
        .set(Var::HeartsAndContainers, 0x22)
        .set(Var::PartialHearts, 0xFF);
    let v = r.view();
    assert!(v.is_sword_screen_locked() && !v.has_beams());
}

/// F182: the mask offers only attacks the game would start: a busy boomerang slot, a blocked sword, Link in an
/// item state, a full bomb pair or a facing outside Link_BorderBounds mask the option.
#[test]
fn the_mask_offers_only_attacks_the_game_starts() {
    use zelda::actions::compute_action_mask;
    use ActionKind::*;
    let armed = |x: u8, y: u8| {
        let mut r = Ram::new();
        r.set(Var::LinkX, x)
            .set(Var::LinkY, y)
            .set(Var::Sword, 1)
            .set(Var::HeartsAndContainers, 0x21)
            .set(Var::Bombs, 4)
            .set(Var::RegularBoomerang, 1)
            .object(5, enemy::STALFOS, 60, 120);
        r
    };
    let allowed = |r: &Ram, kind: ActionKind, d: Direction| {
        compute_action_mask(&r.view()).is_allowed(&zelda::Action::new(kind, d, 0).unwrap())
    };
    let r = armed(120, 141);
    for kind in [Sword, Boomerang, Bomb] {
        assert!(allowed(&r, kind, Direction::N), "{kind:?}");
    }
    let mut busy = r.clone();
    busy.table(tables::OBJ_STATUS, 0x0F, 0x10);
    assert!(!allowed(&busy, Boomerang, Direction::N));
    busy.table(tables::OBJ_STATUS, 0x0F, 0x90); // returning: the game lets Link throw again
    assert!(allowed(&busy, Boomerang, Direction::N));
    let mut blocked = r.clone();
    blocked.0[0x4C] = 5;
    assert!(!allowed(&blocked, Sword, Direction::N) && allowed(&blocked, Bomb, Direction::N));
    let mut holding = r.clone();
    holding.table(tables::OBJ_STATUS, 0, 0x20);
    assert!(!allowed(&holding, Sword, Direction::N) && !allowed(&holding, Bomb, Direction::N));
    let mut bombs = r.clone();
    bombs.table(tables::OBJ_STATUS, 0x10, 0x12).table(tables::OBJ_STATUS, 0x11, 0x10);
    assert!(!allowed(&bombs, Bomb, Direction::N), "both bomb slots hold unexploded bombs");
    // Underworld bounds: y 83 / 199 no attack N/S; x 213 no B item E (the sword still swings E).
    for y in [83, 199] {
        assert!(!allowed(&armed(120, y), Sword, Direction::N) && !allowed(&armed(120, y), Bomb, Direction::S));
    }
    assert!(!allowed(&armed(213, 141), Bomb, Direction::E) && allowed(&armed(213, 141), Sword, Direction::E));
}

#[test]
fn animation_slots_decode_their_states() {
    let state = |var: Var, value: u8, kind: AnimationKind| {
        let mut r = Ram::new();
        r.set(var, value);
        r.view().animation_state(kind)
    };
    use AnimationKind as K;
    use AnimationState::*;
    for (var, value, kind, want) in [
        (Var::BeamAnimation, 16, K::Beams, Active),
        (Var::BeamAnimation, 17, K::Beams, Hit),
        (Var::BeamAnimation, 15, K::Beams, Inactive),
        (Var::BeamAnimation, 0x80, K::Magic, Active),
        (Var::BombOrFlameAnimation, 0x22, K::Magic, Active),
        (Var::BombOrFlameAnimation, 0x22, K::Flame1, Active),
        (Var::BombOrFlameAnimation2, 0x22, K::Flame2, Active),
        (Var::BombOrFlameAnimation, 0x22, K::Flame2, Inactive),
        (Var::BombOrFlameAnimation, 18, K::Bomb1, Active),
        (Var::BombOrFlameAnimation, 19, K::Bomb1, Hit),
        (Var::BombOrFlameAnimation2, 20, K::Bomb2, Hit),
        (Var::BombOrFlameAnimation2, 21, K::Bomb2, Inactive),
        (Var::ArrowMagicAnimation, 10, K::Arrow, Active),
        (Var::ArrowMagicAnimation, 21, K::Arrow, Active),
        (Var::ArrowMagicAnimation, 22, K::Arrow, Inactive),
        (Var::ArrowMagicAnimation, 9, K::Arrow, Inactive),
        (Var::BaitOrBoomerangAnimation, 0x10, K::Boomerang, Active),
        (Var::BaitOrBoomerangAnimation, 0x57, K::Boomerang, Active),
        (Var::BaitOrBoomerangAnimation, 0x58, K::Boomerang, Inactive),
        (Var::BaitOrBoomerangAnimation, 0x0F, K::Boomerang, Inactive),
    ] {
        assert_eq!(state(var, value, kind), want, "{var:?}={value:#x} {kind:?}");
    }
}

#[test]
fn push_block_is_pending_until_pushed() {
    let mut r = Ram::new();
    r.object(11, 0x68, 0x40, 0x60);
    let v = r.view();
    assert!(v.push_block_pending());
    assert_eq!(v.push_block_tiles(), [(8, 4), (8, 5), (9, 4), (9, 5)]);
    r.table(tables::OBJ_STATUS, 11, 1);
    assert!(!r.view().push_block_pending() && r.view().push_block_tiles().is_empty());
    // Only slot 11 holds the push block.
    let mut r = Ram::new();
    r.object(10, 0x68, 0x40, 0x60);
    assert!(!r.view().push_block_pending());
}

#[test]
fn overworld_screen_edges_scroll() {
    // F059: at the top edge (y = $3D) or one step before it, MOVE N scrolls to the room above
    // (asm:Link_CheckScreenEdge), as W does at the west edge.
    let can = |x: u8, y: u8, d: Direction| {
        let mut r = Ram::new();
        r.set(Var::Level, 0)
            .set(Var::Location, 0x77)
            .set(Var::LinkX, x)
            .set(Var::LinkY, y);
        r.view().can_link_move(d)
    };
    assert!(can(120, 0x3D, Direction::N));
    assert!(can(120, 0x45, Direction::N));
    assert!(can(8, 120, Direction::W));
    // In a cave Link can't walk north past the old man's line.
    let mut r = Ram::new();
    r.set(Var::Mode, MODE_CAVE).set(Var::LinkY, 0x8D);
    assert!(!r.view().can_link_move(Direction::N));
}

#[test]
fn doorway_lock_allows_only_the_doorway_axis() {
    let can = |doorway: u8, d: Direction| {
        let mut r = Ram::new();
        r.set(Var::DoorwayDir, doorway).set(Var::LinkGridOffset, 0);
        r.view().can_link_move(d)
    };
    // Standing in the north doorway: only N/S, so E/W are refused without consulting the room.
    assert!(!can(8, Direction::E));
    assert!(!can(8, Direction::W));
    // Mid-tile (grid offset != 0) Link keeps going whatever the doorway.
    let mut r = Ram::new();
    r.set(Var::DoorwayDir, 8).set(Var::LinkGridOffset, 3);
    assert!(r.view().can_link_move(Direction::E));
}
