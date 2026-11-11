//! What changed over one option, with look-ahead credit for delayed weapons.

use crate::actions::{Action, ActionMask};
use crate::blocks::ProjectileBlock;
use crate::enums::{AnimationKind, AnimationState, enemy};
use crate::executor::{Interrupt, Machine, OptionResult};
use crate::game::GameView;
use crate::ledger::{FutureCreditLedger, Prediction, add_hit};
use crate::nes::Nes;
use crate::vars::Var;
use std::sync::Arc;

/// Safety cap on one look-ahead.
pub const LOOK_AHEAD_MAX_FRAMES: u32 = 400;

/// Weapons whose activation runs a look-ahead, in order.
pub const LOOK_AHEAD_WEAPONS: [AnimationKind; 8] = [
    AnimationKind::Beams,
    AnimationKind::Magic,
    AnimationKind::Bomb1,
    AnimationKind::Bomb2,
    AnimationKind::Flame1,
    AnimationKind::Flame2,
    AnimationKind::Arrow,
    AnimationKind::Boomerang,
];

/// Effects seen in the step itself (no look-ahead, nothing discounted): what endings, linger and visit metrics
/// count. A hit is seen on the step it lands.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Observed {
    /// (slot, damage).
    pub enemies_hit: Vec<(u8, i32)>,
    pub enemies_stunned: Vec<u8>,
    pub items_gained: Vec<i16>,
}

impl Observed {
    pub fn hits(&self) -> usize {
        self.enemies_hit.len()
    }
}

/// The changes between two consecutive controllable states.
#[derive(Clone, Debug)]
pub struct Change {
    pub action: Action,
    pub frames: u32,
    pub interrupt: Option<Interrupt>,
    pub control_frames: u32,
    /// Confirmed shield blocks during the option (never changed by look-ahead).
    pub projectile_blocks: Vec<ProjectileBlock>,
    pub previous: Arc<GameView>,
    pub state: Arc<GameView>,
    /// The mask of `state` (as `GameSession.act` sets it).
    pub action_mask: ActionMask,
    pub health_lost: f64,
    pub health_gained: f64,
    /// Seen this step (endings and metrics read these).
    pub observed: Observed,
    /// Credited (rewards read these): (slot, damage), observed plus predicted, minus earlier predictions.
    pub enemies_hit: Vec<(u8, i32)>,
    pub enemies_stunned: Vec<u8>,
    pub items_gained: Vec<i16>,
}

impl Change {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new<N: Nes>(
        m: &mut Machine<N>,
        prev: Arc<GameView>,
        curr: Arc<GameView>,
        action: Action,
        option: OptionResult,
        action_mask: ActionMask,
        ledger: &mut FutureCreditLedger,
        look_ahead: bool,
    ) -> Change {
        // A container pickup adds one filled heart along with the container, so take the container delta out
        // of the filled-heart delta: what is left is a hit or a refill in the same step. A container loss (a
        // cheat or a reload) keeps counting the container as lost.
        let delta = curr.health() - prev.health();
        let grown = (curr.max_health() - prev.max_health()) as f64;
        let (health_gained, health_lost) = if grown >= 0.0 {
            (grown + (delta - grown).max(0.0), (grown - delta).max(0.0))
        } else {
            (0.0, -grown)
        };
        let mut change = Change {
            action,
            frames: option.frames,
            interrupt: option.interrupt,
            control_frames: option.control_frames,
            projectile_blocks: option.projectile_blocks,
            previous: prev,
            state: curr,
            action_mask,
            health_lost,
            health_gained,
            observed: Observed::default(),
            enemies_hit: Vec::new(),
            enemies_stunned: Vec::new(),
            items_gained: Vec::new(),
        };
        if change.changed_location() {
            ledger.clear();
        } else {
            let (prev, curr) = (change.previous.clone(), change.state.clone());
            let seen = &mut change.observed;
            observe_damage(&prev, &curr, &mut seen.enemies_hit, &mut seen.enemies_stunned);
            observe_items(&prev, &curr, &mut seen.items_gained);
            change.enemies_hit = seen.enemies_hit.clone();
            change.enemies_stunned = seen.enemies_stunned.clone();
            change.items_gained = seen.items_gained.clone();
            ledger.discount(
                curr.frame,
                &mut change.enemies_hit,
                &mut change.enemies_stunned,
                &mut change.items_gained,
            );
            // A weapon that finished without its predicted hit landing predicted wrong: its credit was paid at the
            // firing step already, so the leftover entries go rather than discount an unrelated later hit.
            ledger.drop_finished(|w| curr.animation_state(w) == AnimationState::Inactive);
            if look_ahead {
                for weapon in LOOK_AHEAD_WEAPONS {
                    if curr.animation_state(weapon) == AnimationState::Active
                        && prev.animation_state(weapon) != AnimationState::Active
                    {
                        change.predict(m, &curr, weapon, ledger);
                    }
                }
            }
        }
        change
    }

    pub fn changed_location(&self) -> bool {
        self.previous.full_location() != self.state.full_location()
    }

    pub fn gained_triforce(&self) -> bool {
        self.previous.triforce_pieces() < self.state.triforce_pieces()
            || (!self.previous.triforce_of_power() && self.state.triforce_of_power())
    }

    pub fn blocks(&self) -> usize {
        self.projectile_blocks.len()
    }

    pub fn damage_dealt(&self) -> i32 {
        self.enemies_hit.iter().map(|(_, d)| d).sum()
    }

    pub fn hits(&self) -> usize {
        self.enemies_hit.len()
    }

    /// Simulates until the weapon deactivates on a cloned machine state (no input, invincible hearts, other
    /// weapon slots zeroed, no render, no frame sink, no shield watch), credits and records the prediction, then
    /// restores the machine. Only the marginal effect is credited: what other weapons' pending predictions already
    /// promise (damage up to the enemy's hit value, a stun, an item) is not paid again.
    ///
    /// A look-ahead that can't credit anything isn't run: the prediction only follows `start`'s live enemies (and,
    /// for a boomerang or arrow, its items), so without them it is empty and changes nothing.
    fn predict<N: Nes>(
        &mut self,
        m: &mut Machine<N>,
        start: &Arc<GameView>,
        weapon: AnimationKind,
        ledger: &mut FutureCreditLedger,
    ) {
        let fetches = matches!(weapon, AnimationKind::Boomerang | AnimationKind::Arrow);
        if start.enemies.iter().all(|e| e.is_dying()) && (!fetches || start.items.is_empty()) {
            return;
        }
        let saved = m.nes.snapshot();
        let rendered = m.nes.rendered();
        for var in disabled_slots(weapon) {
            m.nes.poke(var.addr(), 0);
        }
        let map = start.game_map().clone();
        let mut curr = start.clone();
        let mut frame = start.frame;
        for _ in 0..LOOK_AHEAD_MAX_FRAMES {
            if curr.game_over()
                || start.location() != curr.location()
                || curr.animation_state(weapon) == AnimationState::Inactive
            {
                break;
            }
            m.nes.poke(Var::HeartsAndContainers.addr(), 0xFF);
            m.nes.run_frame(0, false);
            frame += 1;
            curr = Arc::new(GameView::new(
                &m.ram_image(),
                frame,
                start.just_exited_cave,
                map.clone(),
            ));
        }
        let mut prediction = Prediction {
            frame: start.frame,
            weapon: Some(weapon),
            ..Default::default()
        };
        let (mut hits, mut stuns, mut items) = (Vec::new(), Vec::new(), Vec::new());
        observe_damage(start, &curr, &mut hits, &mut stuns);
        if fetches {
            observe_items(start, &curr, &mut items);
        }
        let pending = ledger.predictions();
        for (slot, dmg) in hits {
            let promised: i32 = pending
                .iter()
                .flat_map(|p| &p.enemy_hits)
                .filter(|(s, _)| *s == slot)
                .map(|(_, d)| d)
                .sum();
            let value = start
                .enemy_by_index(slot)
                .map_or(dmg, |e| if e.health > 0 { e.health } else { 1 });
            let dmg = dmg.min(value - promised);
            if dmg > 0 {
                add_hit(&mut prediction.enemy_hits, slot, dmg);
            }
        }
        let mut pending_stuns: Vec<u8> = pending.iter().flat_map(|p| p.enemy_stuns.iter().copied()).collect();
        for slot in stuns {
            if let Some(i) = pending_stuns.iter().position(|&s| s == slot) {
                pending_stuns.swap_remove(i);
            } else {
                prediction.enemy_stuns.push(slot);
            }
        }
        let mut pending_items: Vec<i16> = pending.iter().flat_map(|p| p.items.iter().copied()).collect();
        for item in items {
            if let Some(i) = pending_items.iter().position(|&x| x == item) {
                pending_items.swap_remove(i);
            } else {
                prediction.items.push(item);
            }
        }
        for &(slot, dmg) in &prediction.enemy_hits {
            add_hit(&mut self.enemies_hit, slot, dmg);
        }
        self.enemies_stunned
            .extend_from_slice(&prediction.enemy_stuns);
        self.items_gained.extend_from_slice(&prediction.items);
        ledger.add_prediction(prediction);
        // The look-ahead rendered nothing: the action's last frame stays the screen (`Game.screen()`).
        m.nes.restore_keeping_frame(&saved, rendered);
    }
}

/// The look-ahead zeroes every weapon slot but the one it follows.
fn disabled_slots(weapon: AnimationKind) -> Vec<Var> {
    let keep = match weapon {
        AnimationKind::Beams | AnimationKind::Magic => Var::BeamAnimation,
        AnimationKind::Bomb1 | AnimationKind::Flame1 => Var::BombOrFlameAnimation,
        AnimationKind::Bomb2 | AnimationKind::Flame2 => Var::BombOrFlameAnimation2,
        AnimationKind::Arrow => Var::ArrowMagicAnimation,
        AnimationKind::Boomerang => Var::BaitOrBoomerangAnimation,
        AnimationKind::Bait => panic!("no look-ahead for {weapon:?}"),
    };
    [
        Var::BeamAnimation,
        Var::BombOrFlameAnimation,
        Var::BombOrFlameAnimation2,
        Var::BaitOrBoomerangAnimation,
        Var::ArrowMagicAnimation,
    ]
    .into_iter()
    .filter(|&v| v != keep)
    .collect()
}

/// Enemy health drops, kills and new stuns from `prev` to `curr`.
fn observe_damage(
    prev: &GameView,
    curr: &GameView,
    hits: &mut Vec<(u8, i32)>,
    stuns: &mut Vec<u8>,
) {
    for enemy in &prev.enemies {
        if enemy.is_dying() {
            continue;
        }
        match curr.enemy_by_index(enemy.index) {
            Some(now) => {
                let dmg = enemy.health - now.health;
                if dmg > 0 {
                    add_hit(hits, enemy.index, dmg);
                } else if now.is_dying() {
                    add_hit(
                        hits,
                        enemy.index,
                        if enemy.health > 0 { enemy.health } else { 1 },
                    );
                }
                if now.is_stunned() && !enemy.is_stunned() {
                    stuns.push(enemy.index);
                }
            }
            // A Zora destroys itself after a dive (Z_04 UpdateZora @DestroyWhenDone), and a step can span a whole
            // surface-and-dive cycle. A killed one turns into a dropped item ($60, Z_07 metastate $14) in its slot
            // (or is still dying, handled above): only that is a kill. Boulders, which destroy themselves off
            // screen, are hazards, not enemies, so they never get here.
            None if enemy.id == enemy::ZORA && curr.item_by_index(enemy.index as i8).is_none() => {}
            None => add_hit(
                hits,
                enemy.index,
                if enemy.health > 0 { enemy.health } else { 1 },
            ),
        }
    }
}

/// Items gone from `prev` to `curr` before their timers ran out (picked up). The lifetime byte counts half-frames
/// (UpdateItem decrements it every other frame, Z_04.asm:11209-11218): an item expires 2·timer−1 or 2·timer
/// frames after `prev`, by frame parity.
fn observe_items(prev: &GameView, curr: &GameView, items: &mut Vec<i16>) {
    let elapsed = curr.frame.saturating_sub(prev.frame);
    for item in &prev.items {
        if curr.item_by_index(item.index).is_none() && elapsed + 1 < 2 * item.timer as u64 {
            items.push(item.id);
        }
    }
}

#[cfg(test)]
mod tests {
    //! Damage/item/health observation and look-ahead credit on RAM images (no emulator).
    use super::*;
    use crate::enums::{enemy, item};
    use crate::executor::tests::{Script, playable, scripted};
    use crate::game_map::GameMap;
    use crate::tables::{self, RAM_SIZE};

    const SLOT: usize = 3;

    fn image(ram: &[u8; 0x800]) -> [u8; RAM_SIZE] {
        let mut out = [0u8; RAM_SIZE];
        out[..0x800].copy_from_slice(ram);
        out
    }

    fn view(ram: &[u8; 0x800], frame: u64) -> Arc<GameView> {
        static MAP: std::sync::LazyLock<Arc<GameMap>> =
            std::sync::LazyLock::new(|| Arc::new(GameMap::builtin()));
        Arc::new(GameView::new(&image(ram), frame, false, MAP.clone()))
    }

    /// A Stalfos with `health` hearts in slot 3, far from Link.
    fn with_stalfos(health: u8) -> [u8; 0x800] {
        let mut ram = playable();
        ram[tables::OBJ_ID.0 + SLOT] = enemy::STALFOS;
        ram[tables::OBJ_HEALTH.0 + SLOT] = health << 4;
        (
            ram[tables::OBJ_POS_X.0 + SLOT],
            ram[tables::OBJ_POS_Y.0 + SLOT],
        ) = (40, 100);
        ram
    }

    fn change(
        m: &mut Machine<Script>,
        prev: Arc<GameView>,
        curr: Arc<GameView>,
        ledger: &mut FutureCreditLedger,
        look_ahead: bool,
    ) -> Change {
        let option = OptionResult {
            frames: 4,
            interrupt: None,
            control_frames: 4,
            projectile_blocks: Vec::new(),
        };
        Change::new(
            m,
            prev,
            curr,
            Action::parse("WAIT 4").unwrap(),
            option,
            ActionMask::default(),
            ledger,
            look_ahead,
        )
    }

    fn observe(prev: &[u8; 0x800], curr: &[u8; 0x800], elapsed: u64) -> Change {
        let mut m = scripted(*curr, |_, _| {});
        let mut ledger = FutureCreditLedger::default();
        change(
            &mut m,
            view(prev, 100),
            view(curr, 100 + elapsed),
            &mut ledger,
            false,
        )
    }

    #[test]
    fn damage_kills_dying_enemies_and_new_stuns() {
        let prev = with_stalfos(3);
        let mut hurt = prev;
        hurt[tables::OBJ_HEALTH.0 + SLOT] = 1 << 4;
        assert_eq!(observe(&prev, &hurt, 4).enemies_hit, vec![(SLOT as u8, 2)]);

        // Gone from its slot or in the death sparkle: credited with the health it had left.
        let mut gone = prev;
        gone[tables::OBJ_ID.0 + SLOT] = 0;
        assert_eq!(observe(&prev, &gone, 4).enemies_hit, vec![(SLOT as u8, 3)]);
        let mut dying = prev;
        dying[tables::OBJ_SPAWN_STATE.0 + SLOT] = 0x10;
        assert_eq!(observe(&prev, &dying, 4).enemies_hit, vec![(SLOT as u8, 3)]);
        // A zero-health enemy still counts as one hit when it dies.
        let weak = with_stalfos(0);
        let mut killed = weak;
        killed[tables::OBJ_ID.0 + SLOT] = 0;
        assert_eq!(
            observe(&weak, &killed, 4).enemies_hit,
            vec![(SLOT as u8, 1)]
        );
        // Already dying before the step: nothing more to credit.
        assert_eq!(observe(&dying, &gone, 4).enemies_hit, vec![]);

        // F038: objects that destroy themselves are no kills: a boulder that left the screen (a hazard) and a
        // Zora whose dive ended (gone, whatever ObjState it had at the step's start: a step can span a whole
        // surface-and-dive cycle).
        let mut boulder = playable();
        boulder[tables::OBJ_ID.0 + SLOT] = 0x20;
        boulder[tables::OBJ_HEALTH.0 + SLOT] = 0xF0;
        boulder[tables::OBJ_POS_Y.0 + SLOT] = 0xEE;
        let mut empty = boulder;
        empty[tables::OBJ_ID.0 + SLOT] = 0;
        assert_eq!(observe(&boulder, &empty, 4).enemies_hit, vec![]);
        let mut zora = playable();
        zora[tables::OBJ_ID.0 + SLOT] = enemy::ZORA;
        zora[tables::OBJ_HEALTH.0 + SLOT] = 0x20;
        zora[tables::OBJ_STATUS.0 + SLOT] = 5;
        let mut dived = zora;
        dived[tables::OBJ_ID.0 + SLOT] = 0;
        assert_eq!(observe(&zora, &dived, 4).enemies_hit, vec![]);
        zora[tables::OBJ_STATUS.0 + SLOT] = 3;
        assert_eq!(observe(&zora, &dived, 4).enemies_hit, vec![]);
        // A killed Zora becomes a dropped item ($60) in its slot.
        let mut dropped = dived;
        dropped[tables::OBJ_ID.0 + SLOT] = 0x60;
        assert_eq!(observe(&zora, &dropped, 4).enemies_hit, vec![(SLOT as u8, 2)]);

        let mut stunned = prev;
        stunned[tables::OBJ_STUN_TIMER.0 + SLOT] = 0x10;
        assert_eq!(
            observe(&prev, &stunned, 4).enemies_stunned,
            vec![SLOT as u8]
        );
        let mut still = stunned;
        still[tables::OBJ_STUN_TIMER.0 + SLOT] = 0x0C;
        assert_eq!(observe(&stunned, &still, 4).enemies_stunned, vec![]);
    }

    #[test]
    fn vanished_items_count_as_pickups_only_before_their_timer_runs_out() {
        let mut prev = playable();
        prev[tables::OBJ_ID.0 + 2] = 0x60;
        prev[tables::OBJ_STATUS.0 + 2] = item::HEART as u8;
        prev[tables::ITEM_TIMER.0 + 2] = 8;
        let mut curr = prev;
        curr[tables::OBJ_ID.0 + 2] = 0;
        // The timer counts half-frames: timer 8 lives 15 or 16 more frames.
        assert_eq!(observe(&prev, &curr, 14).items_gained, vec![item::HEART]);
        assert_eq!(
            observe(&prev, &curr, 15).items_gained,
            vec![],
            "expired, not taken"
        );
        assert_eq!(observe(&prev, &prev, 2).items_gained, vec![]);
    }

    #[test]
    fn container_changes_count_whole_hearts() {
        let prev = playable(); // 3 of 3 hearts
        let mut up = prev;
        up[Var::HeartsAndContainers.addr() as usize] = 0x33;
        let c = observe(&prev, &up, 4);
        assert_eq!((c.health_gained, c.health_lost), (1.0, 0.0));
        let c = observe(&up, &prev, 4);
        assert_eq!((c.health_gained, c.health_lost), (0.0, 1.0));
        let mut half = prev;
        half[Var::HeartsAndContainers.addr() as usize] = 0x21;
        half[Var::PartialHearts.addr() as usize] = 0x7F;
        let c = observe(&prev, &half, 4);
        assert_eq!((c.health_gained, c.health_lost), (0.0, 1.5));
        // a container pickup (both nibbles +1) in the step that also costs half a heart: 5 of 6 -> 5.5 of 7
        let mut prev = playable();
        prev[Var::HeartsAndContainers.addr() as usize] = 0x54;
        prev[Var::PartialHearts.addr() as usize] = 0xFF;
        let mut curr = prev;
        curr[Var::HeartsAndContainers.addr() as usize] = 0x65;
        curr[Var::PartialHearts.addr() as usize] = 0x7F;
        let c = observe(&prev, &curr, 4);
        assert_eq!((c.health_gained, c.health_lost), (1.0, 0.5));
    }

    #[test]
    fn changing_rooms_observes_nothing_and_drops_predictions() {
        let prev = with_stalfos(3);
        let mut next = prev;
        next[Var::Location.addr() as usize] = 0x45;
        next[tables::OBJ_ID.0 + SLOT] = 0;
        let mut m = scripted(next, |_, _| {});
        let mut ledger = FutureCreditLedger::default();
        ledger.add_prediction(Prediction {
            frame: 100,
            enemy_hits: vec![(1, 1)],
            ..Default::default()
        });
        let c = change(
            &mut m,
            view(&prev, 100),
            view(&next, 104),
            &mut ledger,
            true,
        );
        assert!(c.changed_location() && c.enemies_hit.is_empty());
        assert!(ledger.predictions().is_empty());
    }

    /// Beam fired at frame 0 of the script: it hits the Stalfos on frame 5 and ends on frame 8.
    fn beam_script(frame: u64, ram: &mut [u8; 0x800]) {
        if frame >= 5 {
            ram[tables::OBJ_HEALTH.0 + SLOT] = 2 << 4;
        }
        if frame >= 8 {
            ram[Var::BeamAnimation.addr() as usize] = 0;
        }
    }

    #[test]
    fn look_ahead_credits_the_firing_step_once() {
        let prev = with_stalfos(3);
        let mut fired = prev;
        fired[Var::BeamAnimation.addr() as usize] = 16;
        fired[Var::BombOrFlameAnimation.addr() as usize] = 0x05; // another slot, zeroed for the look-ahead
        let mut m = scripted(fired, beam_script);
        let mut ledger = FutureCreditLedger::default();
        let c = change(
            &mut m,
            view(&prev, 100),
            view(&fired, 104),
            &mut ledger,
            true,
        );
        assert_eq!(
            c.enemies_hit,
            vec![(SLOT as u8, 1)],
            "credited when the beam leaves"
        );
        assert_eq!(c.observed, Observed::default(), "nothing landed yet");
        assert_eq!(ledger.predictions().len(), 1);
        assert_eq!(ledger.predictions()[0].frame, 104);
        // The look-ahead ran on a copy: the machine, its frame counter and the other slot are untouched.
        assert_eq!(m.nes.ram, fired);
        assert_eq!(m.frame, 0);
        assert!(m.nes.buttons.iter().all(|&b| b == 0));

        // The real hit lands on a later step: not rewarded again.
        let mut landed = fired;
        landed[tables::OBJ_HEALTH.0 + SLOT] = 2 << 4;
        landed[Var::BeamAnimation.addr() as usize] = 0;
        let mut m = scripted(landed, |_, _| {});
        let c = change(
            &mut m,
            view(&fired, 104),
            view(&landed, 112),
            &mut ledger,
            true,
        );
        assert_eq!(c.enemies_hit, vec![]);
        assert_eq!(c.observed.enemies_hit, vec![(SLOT as u8, 1)], "seen when it lands");
        assert!(ledger.predictions().is_empty());

        // Without look-ahead the hit is credited when observed.
        let mut ledger = FutureCreditLedger::default();
        let mut m = scripted(fired, beam_script);
        let c = change(
            &mut m,
            view(&prev, 100),
            view(&fired, 104),
            &mut ledger,
            false,
        );
        assert!(c.enemies_hit.is_empty() && ledger.predictions().is_empty());
        let c = change(
            &mut m,
            view(&fired, 104),
            view(&landed, 112),
            &mut ledger,
            false,
        );
        assert_eq!(c.enemies_hit, vec![(SLOT as u8, 1)]);
    }

    #[test]
    fn a_weapon_already_active_does_not_look_ahead_again() {
        let mut fired = with_stalfos(3);
        fired[Var::BeamAnimation.addr() as usize] = 16;
        let mut m = scripted(fired, beam_script);
        let mut ledger = FutureCreditLedger::default();
        let c = change(
            &mut m,
            view(&fired, 100),
            view(&fired, 104),
            &mut ledger,
            true,
        );
        assert!(c.enemies_hit.is_empty() && ledger.predictions().is_empty());
    }

    /// A look-ahead only runs when it can credit something: a beam fired in a room without a live enemy emulates no
    /// frame (and predicts nothing); with a Stalfos there, it runs.
    #[test]
    fn no_look_ahead_without_a_live_enemy() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static FRAMES: AtomicU64 = AtomicU64::new(0);
        fn counted(frame: u64, ram: &mut [u8; 0x800]) {
            FRAMES.fetch_add(1, Ordering::Relaxed);
            beam_script(frame, ram);
        }
        for (prev, runs) in [(playable(), false), (with_stalfos(3), true)] {
            let mut fired = prev;
            fired[Var::BeamAnimation.addr() as usize] = 16;
            let mut m = scripted(fired, counted);
            let mut ledger = FutureCreditLedger::default();
            FRAMES.store(0, Ordering::Relaxed);
            let c = change(&mut m, view(&prev, 100), view(&fired, 104), &mut ledger, true);
            assert_eq!(FRAMES.load(Ordering::Relaxed) > 0, runs);
            assert_eq!(c.enemies_hit.is_empty(), !runs);
            assert_eq!(ledger.predictions().is_empty(), !runs);
            assert_eq!(m.nes.ram, fired);
        }
    }

    /// F054: a second weapon fired at an enemy another weapon's pending prediction already kills is not credited
    /// again; over the firing steps and the landing step the 1-hp Stalfos pays one hit.
    #[test]
    fn a_second_weapon_is_credited_only_its_marginal_effect() {
        let kill = |frame: u64, ram: &mut [u8; 0x800]| {
            if frame >= 5 {
                ram[tables::OBJ_ID.0 + SLOT] = 0;
            }
            if frame >= 8 {
                ram[Var::BeamAnimation.addr() as usize] = 0;
                ram[Var::ArrowMagicAnimation.addr() as usize] = 0;
            }
        };
        let prev = with_stalfos(1);
        let mut beam = prev;
        beam[Var::BeamAnimation.addr() as usize] = 16;
        let mut ledger = FutureCreditLedger::default();
        let mut m = scripted(beam, kill);
        let a = change(&mut m, view(&prev, 100), view(&beam, 104), &mut ledger, true);
        assert_eq!(a.enemies_hit, vec![(SLOT as u8, 1)]);
        let mut arrow = beam;
        arrow[Var::ArrowMagicAnimation.addr() as usize] = 10;
        let mut m = scripted(arrow, kill);
        let b = change(&mut m, view(&beam, 104), view(&arrow, 108), &mut ledger, true);
        assert_eq!(b.enemies_hit, vec![], "already promised by the beam");
        let mut landed = arrow;
        landed[tables::OBJ_ID.0 + SLOT] = 0;
        let mut m = scripted(landed, |_, _| {});
        let c = change(&mut m, view(&arrow, 108), view(&landed, 112), &mut ledger, true);
        assert_eq!(c.observed.enemies_hit, vec![(SLOT as u8, 1)]);
        assert_eq!(c.enemies_hit, vec![], "the landing pays nothing more");
    }

    /// F054 (wrong slot): a beam predicted to hit the Stalfos finishes without hitting him; when the sword hits him
    /// later, that hit is credited (the beam's leftover prediction went when the beam ended).
    #[test]
    fn a_finished_weapons_unused_prediction_does_not_swallow_a_later_hit() {
        let prev = with_stalfos(3);
        let mut fired = prev;
        fired[Var::BeamAnimation.addr() as usize] = 16;
        let mut ledger = FutureCreditLedger::default();
        let mut m = scripted(fired, beam_script);
        let c = change(&mut m, view(&prev, 100), view(&fired, 104), &mut ledger, true);
        assert_eq!(c.enemies_hit, vec![(SLOT as u8, 1)]);
        // The beam ends; the Stalfos is untouched (it hit something else).
        let mut ended = fired;
        ended[Var::BeamAnimation.addr() as usize] = 0;
        let mut m = scripted(ended, |_, _| {});
        let c = change(&mut m, view(&fired, 104), view(&ended, 112), &mut ledger, true);
        assert!(c.enemies_hit.is_empty() && ledger.predictions().is_empty());
        // A later sword hit on him is new credit.
        let mut stabbed = ended;
        stabbed[tables::OBJ_HEALTH.0 + SLOT] = 2 << 4;
        let mut m = scripted(stabbed, |_, _| {});
        let c = change(&mut m, view(&ended, 112), view(&stabbed, 120), &mut ledger, true);
        assert_eq!(c.enemies_hit, vec![(SLOT as u8, 1)]);
    }
}
