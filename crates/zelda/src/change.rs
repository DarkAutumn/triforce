//! What changed over one option, with look-ahead credit for delayed weapons.

use crate::actions::{Action, ActionMask};
use crate::clips::ClipOutcome;
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

/// Which fired weapons the session simulates forward to credit their hits to the firing step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookAhead {
    Off,
    /// Bombs only: what [`crate::bombs::BombStats`] needs to tell a bomb's hits (evaluation and chains, which score
    /// no rewards; a bomb's prediction is rare and cheap).
    Bombs,
    /// Every weapon (training: the rewards credit the firing step).
    All,
}

impl LookAhead {
    fn covers(self, weapon: AnimationKind) -> bool {
        match self {
            LookAhead::Off => false,
            LookAhead::Bombs => matches!(weapon, AnimationKind::Bomb1 | AnimationKind::Bomb2),
            LookAhead::All => true,
        }
    }
}

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
    /// Slots whose monster died this step (it started dying, or its slot emptied): kills the game's `RoomKills`
    /// may not count (a Vire's Red Keese, Gleeok's heads).
    pub enemies_killed: Vec<u8>,
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
    /// Frames that count against the action: every frame it played but the item sub-screen's (the game is paused
    /// there). Discounting, the frame penalty, the observation's history, heart timers and linger use these.
    pub frames: u32,
    /// Frames on the item sub-screen choosing the B item (`crate::submenu`), not in `frames`. Counts of frames
    /// played (session and episode totals, per-visit and chain frames, replays) add them back.
    pub menu_frames: u32,
    pub interrupt: Option<Interrupt>,
    pub control_frames: u32,
    /// Confirmed shield blocks during the option (never changed by look-ahead).
    pub projectile_blocks: Vec<ProjectileBlock>,
    /// The skip the option performed, if any (`crate::clips`).
    pub clip: Option<ClipOutcome>,
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
        look_ahead: LookAhead,
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
            frames: option.frames - option.menu_frames,
            menu_frames: option.menu_frames,
            interrupt: option.interrupt,
            control_frames: option.control_frames,
            projectile_blocks: option.projectile_blocks,
            clip: option.clip,
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
            observe_damage(
                &prev,
                &curr,
                &mut seen.enemies_hit,
                &mut seen.enemies_stunned,
                &mut seen.enemies_killed,
            );
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
            for weapon in LOOK_AHEAD_WEAPONS {
                if look_ahead.covers(weapon)
                    && curr.animation_state(weapon) == AnimationState::Active
                    && prev.animation_state(weapon) != AnimationState::Active
                {
                    change.predict(m, &curr, weapon, ledger);
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

    /// The step's BOMB action placed a bomb (a bomb slot went from inactive to active).
    pub fn placed_bomb(&self) -> bool {
        use crate::enums::{AnimationKind, AnimationState};
        self.action.kind == crate::enums::ActionKind::Bomb
            && [AnimationKind::Bomb1, AnimationKind::Bomb2].into_iter().any(|k| {
                self.state.animation_state(k) != AnimationState::Inactive
                    && self.previous.animation_state(k) == AnimationState::Inactive
            })
    }

    /// The step made the 10th damage-free kill (`$50` 9 → 10, or 9 → 0 when the forced drop already fell within
    /// the step and the kill count went up; a hit to Link that clears both is no kill).
    pub fn tenth_kill(&self) -> bool {
        let (prev, curr) = (&self.previous, &self.state);
        prev.next_kill_arms_drop()
            && match curr.help_drop_count() {
                10 => true,
                0 => curr.kill_streak() > prev.kill_streak(),
                _ => false,
            }
    }

    /// The step's 10th kill armed a forced bomb drop (a bomb kill: `$51` set), or spent one: the 10th kill's drop
    /// fell within the step as bombs.
    pub fn armed_bomb_drop(&self) -> bool {
        self.tenth_kill()
            && (self.state.help_drop_is_bombs()
                || (self.state.help_drop_count() == 0 && self.dropped(crate::enums::item::BOMBS)))
    }

    /// A dropped item of kind `id` appeared in the room this step (a slot holding it that didn't before).
    pub fn dropped(&self, id: i16) -> bool {
        let had = |index: i8| self.previous.items.iter().any(|i| i.index == index && i.id == id);
        self.previous.full_location() == self.state.full_location()
            && self.state.items.iter().any(|i| i.id == id && !had(i.index))
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
        // Predicted kills aren't kills yet: only the observed step records them.
        observe_damage(start, &curr, &mut hits, &mut stuns, &mut Vec::new());
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

/// Enemy health drops, kills and new stuns from `prev` to `curr`.  Gleeok's attached heads share one HP byte, so a
/// hit on them is credited once (the largest credit among them), to the head that took it or fell.
fn observe_damage(
    prev: &GameView,
    curr: &GameView,
    hits: &mut Vec<(u8, i32)>,
    stuns: &mut Vec<u8>,
    kills: &mut Vec<u8>,
) {
    let mut gleeok_hit: Option<(u8, i32)> = None;
    for enemy in &prev.enemies {
        if enemy.is_dying() {
            continue;
        }
        let gleeok_head = enemy.id == enemy::GLEEOK_NECK_HEAD;
        let mut credit = |hits: &mut Vec<(u8, i32)>, dmg: i32| {
            if !gleeok_head {
                add_hit(hits, enemy.index, dmg);
            } else if gleeok_hit.is_none_or(|(_, best)| dmg > best) {
                gleeok_hit = Some((enemy.index, dmg));
            }
        };
        let killed = if enemy.health > 0 { enemy.health } else { 1 };
        match curr.enemy_by_index(enemy.index) {
            Some(now) => {
                let dmg = enemy.health - now.health;
                if dmg > 0 {
                    credit(hits, dmg);
                } else if now.is_dying() {
                    credit(hits, killed);
                }
                if now.is_dying() {
                    kills.push(enemy.index);
                }
                if now.is_stunned() && !enemy.is_stunned() {
                    stuns.push(enemy.index);
                }
            }
            // A Zora destroys itself after a dive (asm:Zora_Update@DestroyIfDived), and a step can span a whole
            // surface-and-dive cycle. A killed one turns into a dropped item ($60, metastate $14,
            // asm:Obj_EndMetastate) in its slot (or is still dying, handled above): only that is a kill. Boulders,
            // which destroy themselves off screen, are hazards, not enemies, so they never get here.
            None if enemy.id == enemy::ZORA && curr.item_by_index(enemy.index as i8).is_none() => {}
            None => {
                credit(hits, killed);
                kills.push(enemy.index);
            }
        }
    }
    if let Some((index, dmg)) = gleeok_hit {
        add_hit(hits, index, dmg);
    }
}

/// Items gone from `prev` to `curr` before their timers ran out (picked up). The lifetime byte counts half-frames
/// (asm:Drop_UpdateItem decrements it every other frame): an item expires 2·timer−1 or 2·timer
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
            clip: None,
            menu_frames: 0,
        };
        Change::new(
            m,
            prev,
            curr,
            Action::parse("WAIT 4").unwrap(),
            option,
            ActionMask::default(),
            ledger,
            if look_ahead { LookAhead::All } else { LookAhead::Off },
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

    /// A kill is observed whether or not the game counts it in RoomKills (a Vire's Red Keese): the slot empties or
    /// starts dying.  A dived Zora and a mere hit are no kills.
    #[test]
    fn kills_are_observed_without_room_kills() {
        let mut keese = playable();
        keese[tables::OBJ_ID.0 + SLOT] = enemy::RED_KEESE;
        keese[tables::OBJ_HEALTH.0 + SLOT] = 0x10;
        let mut gone = keese;
        gone[tables::OBJ_ID.0 + SLOT] = 0;
        let kill = observe(&keese, &gone, 4);
        assert_eq!(kill.observed.enemies_killed, vec![SLOT as u8]);
        assert_eq!(kill.state.room_kills(), kill.previous.room_kills());
        let mut dying = keese;
        dying[tables::OBJ_SPAWN_STATE.0 + SLOT] = 0x10;
        assert_eq!(observe(&keese, &dying, 4).observed.enemies_killed, vec![SLOT as u8]);
        let stalfos = with_stalfos(3);
        let mut hurt = stalfos;
        hurt[tables::OBJ_HEALTH.0 + SLOT] = 1 << 4;
        assert!(observe(&stalfos, &hurt, 4).observed.enemies_killed.is_empty());
        let mut zora = playable();
        zora[tables::OBJ_ID.0 + SLOT] = enemy::ZORA;
        zora[tables::OBJ_STATUS.0 + SLOT] = 5;
        let mut dived = zora;
        dived[tables::OBJ_ID.0 + SLOT] = 0;
        assert!(observe(&zora, &dived, 4).observed.enemies_killed.is_empty());
    }

    /// Gleeok with two necks (type $43 in slot 1): heads at (0x70, 0x80) and (0x88, 0x80), sharing slot 5's HP $A0.
    fn with_gleeok() -> [u8; 0x800] {
        let mut ram = playable();
        ram[tables::OBJ_ID.0 + 1] = enemy::GLEEOK2;
        ram[tables::OBJ_HEALTH.0 + 1] = 0xA0;
        ram[tables::OBJ_HEALTH.0 + crate::game::GLEEOK_HEAD_SLOT] = 0xA0;
        for (neck, x) in [(0, 0x70), (1, 0x88)] {
            ram[crate::game::GLEEOK_NECK_X[neck] + 4] = x;
            ram[crate::game::GLEEOK_NECK_Y[neck] + 4] = 0x80;
        }
        ram
    }

    /// The env sees one enemy per attached head and the neck base as a hazard; a hit on the shared HP pays once,
    /// the base's and a flying head's pay nothing, and a cut head is one hit and one kill.
    #[test]
    fn gleeok_heads_are_the_enemies_and_share_one_hit() {
        let prev = with_gleeok();
        let seen = view(&prev, 0);
        let heads: Vec<_> = seen.enemies.iter().map(|e| (e.index, e.id, e.position.x, e.health)).collect();
        assert_eq!(
            heads.iter().copied().collect::<std::collections::BTreeSet<_>>(),
            [(2, enemy::GLEEOK_NECK_HEAD, 0x70, 10), (3, enemy::GLEEOK_NECK_HEAD, 0x88, 10)].into()
        );
        assert!(seen.hazards.iter().any(|h| h.index == 1));

        let mut hit = prev;
        hit[tables::OBJ_HEALTH.0 + crate::game::GLEEOK_HEAD_SLOT] = 0x80;
        let hits = observe(&prev, &hit, 4).enemies_hit;
        assert!(hits.len() == 1 && hits[0].1 == 2, "{hits:?}");

        let mut base = prev;
        base[tables::OBJ_HEALTH.0 + 1] = 0x80;
        assert_eq!(observe(&prev, &base, 4).enemies_hit, vec![]);

        let mut flying = prev;
        flying[tables::OBJ_ID.0 + 7] = enemy::GLEEOK_HEAD;
        flying[tables::OBJ_HEALTH.0 + 7] = 0xF0;
        let mut landed = flying;
        landed[tables::OBJ_ID.0 + 7] = 0;
        let gone = observe(&flying, &landed, 4);
        assert!(gone.enemies_hit.is_empty() && gone.observed.enemies_killed.is_empty());

        // Neck 0's head falls: its bit in $511, the shared HP reset to $60 for the next head.
        let mut low = prev;
        low[tables::OBJ_HEALTH.0 + crate::game::GLEEOK_HEAD_SLOT] = 0x20;
        let mut cut = low;
        cut[crate::game::GLEEOK_DEAD_NECK_MASK] = 1;
        cut[tables::OBJ_HEALTH.0 + crate::game::GLEEOK_HEAD_SLOT] = 0x60;
        let fell = observe(&low, &cut, 4);
        assert_eq!((fell.enemies_hit, fell.observed.enemies_killed), (vec![(2, 2)], vec![2]));
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
