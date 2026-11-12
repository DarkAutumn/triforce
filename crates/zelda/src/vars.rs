//! Named RAM variables (`name -> (address, signed)`; the names are those of the old stable-retro `data.json`).
//!
//! Every variable is one byte of CPU RAM; signed ones (`=i1`) decode as `i8`.

macro_rules! vars {
    ($($variant:ident = ($name:literal, $addr:literal, $signed:literal),)*) => {
        /// A named RAM variable.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum Var { $($variant,)* }

        impl Var {
            /// Every variable, in the old data.json order.
            pub const ALL: &'static [Var] = &[$(Var::$variant,)*];

            /// The variable's name (the old data.json key; Python's `GameState.var` and `Game.set_value` take it).
            pub const fn name(self) -> &'static str {
                match self { $(Var::$variant => $name,)* }
            }

            /// CPU RAM address.
            pub const fn addr(self) -> u16 {
                match self { $(Var::$variant => $addr,)* }
            }

            /// Whether the byte decodes as `i8`.
            pub const fn signed(self) -> bool {
                match self { $(Var::$variant => $signed,)* }
            }

            /// Looks a variable up by its name.
            pub fn from_name(name: &str) -> Option<Var> {
                match name { $($name => Some(Var::$variant),)* _ => None }
            }

            /// Decodes the variable from a RAM byte.
            pub const fn decode(self, byte: u8) -> i32 {
                if self.signed() { byte as i8 as i32 } else { byte as i32 }
            }
        }
    };
}

vars! {
    LinkObjTimer = ("link_obj_timer", 0x028, false),
    LinkShoveDir = ("link_shove_dir", 0x0C0, false),
    UndergroundExitType = ("underground_exit_type", 0x05A, false),
    DoorwayDir = ("doorway_dir", 0x053, false),
    TriggeredDoorCmd = ("triggered_door_cmd", 0x054, false),
    CurOpenedDoors = ("cur_opened_doors", 0x0EE, false),
    Level = ("level", 0x010, false),
    Location = ("location", 0x0EB, false),
    Mode = ("mode", 0x012, false),
    LinkX = ("link_x", 0x070, false),
    LinkY = ("link_y", 0x084, false),
    LinkStatus = ("link_status", 0x0AC, false),
    LinkDirection = ("link_direction", 0x098, false),
    LinkGridOffset = ("link_grid_offset", 0x394, true),
    SwordBlockedLongTimer = ("sword_blocked_timer", 0x04C, false),
    SwordBlocked = ("sword_blocked", 0x52E, false),
    Rng0 = ("rng_0", 0x018, false),
    Rng1 = ("rng_1", 0x019, false),
    Rng2 = ("rng_2", 0x01A, false),
    Rng3 = ("rng_3", 0x01B, false),
    Rng4 = ("rng_4", 0x01C, false),
    Rng5 = ("rng_5", 0x01D, false),
    Rng6 = ("rng_6", 0x01E, false),
    Rng7 = ("rng_7", 0x01F, false),
    Rng8 = ("rng_8", 0x020, false),
    Rng9 = ("rng_9", 0x021, false),
    Rng10 = ("rng_10", 0x022, false),
    Rng11 = ("rng_11", 0x023, false),
    Rng12 = ("rng_12", 0x024, false),
    SwordAnimation = ("sword_animation", 0x0B9, false),
    BeamAnimation = ("beam_animation", 0x0BA, false),
    BaitOrBoomerangAnimation = ("bait_or_boomerang_animation", 0x0BB, false),
    BombOrFlameAnimation = ("bomb_or_flame_animation", 0x0BC, false),
    BombOrFlameAnimation2 = ("bomb_or_flame_animation2", 0x0BD, false),
    ArrowMagicAnimation = ("arrow_magic_animation", 0x0BE, false),
    SoundPulse1 = ("sound_pulse_1", 0x605, false),
    Triforce = ("triforce", 0x671, false),
    TriforceOfPower = ("triforce_of_power", 0x672, false),
    HeartsAndContainers = ("hearts_and_containers", 0x66F, false),
    PartialHearts = ("partial_hearts", 0x670, false),
    KillStreak = ("kill_streak", 0x627, false),
    RoomKills = ("room_kills", 0x34F, false),
    Rupees = ("rupees", 0x66D, false),
    RupeesToAdd = ("rupees_to_add", 0x67D, false),
    RupeesToSubtract = ("rupees_to_subtract", 0x67E, false),
    Bombs = ("bombs", 0x658, false),
    BombMax = ("bomb_max", 0x67C, false),
    Keys = ("keys", 0x66E, false),
    TreasureX = ("treasure_x", 0x083, false),
    TreasureY = ("treasure_y", 0x097, false),
    TreasureFlag = ("treasure_flag", 0x0BF, false),
    Sword = ("sword", 0x657, false),
    Arrows = ("arrows", 0x659, false),
    Bow = ("bow", 0x65A, false),
    Candle = ("candle", 0x65B, false),
    Whistle = ("whistle", 0x65C, false),
    Food = ("food", 0x65D, false),
    Potion = ("potion", 0x65E, false),
    MagicRod = ("magic_rod", 0x65F, false),
    Raft = ("raft", 0x660, false),
    Book = ("book", 0x661, false),
    Ring = ("ring", 0x662, false),
    StepLadder = ("step_ladder", 0x663, false),
    MagicKey = ("magic_key", 0x664, false),
    PowerBracelet = ("power_bracelet", 0x665, false),
    Letter = ("letter", 0x666, false),
    RegularBoomerang = ("regular_boomerang", 0x674, false),
    MagicBoomerang = ("magic_boomerang", 0x675, false),
    MagicShield = ("magic_shield", 0x676, false),
    Compass = ("compass", 0x667, false),
    Map = ("map", 0x668, false),
    Compass9 = ("compass9", 0x669, false),
    Map9 = ("map9", 0x66A, false),
    Clock = ("clock", 0x66C, false),
    SelectedItem = ("selected_item", 0x656, false),
    ObjHealth1 = ("obj_health_1", 0x486, false),
    ObjHealth2 = ("obj_health_2", 0x487, false),
    ObjHealth3 = ("obj_health_3", 0x488, false),
    ObjHealth4 = ("obj_health_4", 0x489, false),
    ObjHealth5 = ("obj_health_5", 0x48A, false),
    ObjHealth6 = ("obj_health_6", 0x48B, false),
    ObjHealth7 = ("obj_health_7", 0x48C, false),
    ObjHealth8 = ("obj_health_8", 0x48D, false),
    ObjHealth9 = ("obj_health_9", 0x48E, false),
    ObjHealthA = ("obj_health_a", 0x48F, false),
    ObjHealthB = ("obj_health_b", 0x490, false),
    ObjHealthC = ("obj_health_c", 0x491, false),
}

/// The 13 LFSR bytes $18-$24 (asm:Nmi_Isr@ScrambleRandom rotates 13; monsters read `Random+1, X` up to $24).
pub const RNG_VARS: [Var; 13] = [
    Var::Rng0,
    Var::Rng1,
    Var::Rng2,
    Var::Rng3,
    Var::Rng4,
    Var::Rng5,
    Var::Rng6,
    Var::Rng7,
    Var::Rng8,
    Var::Rng9,
    Var::Rng10,
    Var::Rng11,
    Var::Rng12,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_vars_cover_all_13_lfsr_bytes() {
        let addrs: Vec<u16> = RNG_VARS.iter().map(|v| v.addr()).collect();
        assert_eq!(addrs, (0x18..=0x24).collect::<Vec<u16>>());
    }

    #[test]
    fn names_round_trip_and_signed_vars_decode_as_i8() {
        for var in Var::ALL {
            assert_eq!(Var::from_name(var.name()), Some(*var), "{}", var.name());
        }
        assert_eq!(Var::from_name("no-such-var"), None);
        assert_eq!(Var::LinkGridOffset.decode(0xF8), -8);
    }
}
