//! NES PPU (Picture Processing Unit) implementation.
//!
//! [`Ppu`] is the state a 2C02 keeps - registers, OAM and palette RAM. The frame it draws is
//! not part of it: the [`Bus`] owns that buffer and lends it to the pixel path.
//! Anything that reaches the cartridge is an `impl Bus` block further down this file, since a CHR
//! or nametable fetch resolves through the board: [`Bus::ppu_clock`](crate::bus::Bus::ppu_clock)
//! drives a dot and [`Bus::chr_peek`](crate::bus::Bus::chr_peek) resolves an address, while the
//! register reads and the pixel pipeline are here.
//!
//! # Stability
//!
//! [`Ppu`]'s fields are the emulation's internal wiring. They are public so that tools can digest
//! them, but they track the implementation, not a stable interface; the stable entry point is
//! [`ZCore`](crate::ZCore). Fields documented as *derived* are caches of state that lives
//! elsewhere; writing one from outside desynchronizes the emulator rather than changing what it
//! does.

use crate::{bus::Bus, common::ResetKind, memory::ConstArray};
use scroll::Scroll;
use serde::{Deserialize, Serialize};
use sprite::Sprite;
use std::cmp::Ordering;

pub mod ctrl;
pub mod mask;
pub mod scroll;
pub mod sprite;
mod span;
pub mod status;

/// Nametable Mirroring Mode
///
/// <https://wiki.nesdev.org/w/index.php/Mirroring#Nametable_Mirroring>
#[derive(Default, Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[must_use]
pub enum Mirroring {
    /// Nametables mirrored left-to-right, giving two vertically-scrollable screens.
    Vertical = 0,
    /// Nametables mirrored top-to-bottom, giving two horizontally-scrollable screens.
    #[default]
    Horizontal = 1,
    /// All four nametables show CIRAM's first 1K bank.
    SingleScreenA = 2,
    /// All four nametables show CIRAM's second 1K bank.
    SingleScreenB = 3,
}

/// Palette RAM which enforces mirroring.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[must_use]
#[repr(transparent)]
pub struct PaletteRam(ConstArray<u8, 32>);

impl PaletteRam {
    /// Return palette address, mirrored.
    //
    // Mirroring on read rather than storing both halves of each backdrop mirror at write time:
    // this pays a second, dependent load per pixel and is still faster, by 4.1% at one code layout
    // and 0.9% at another - measured twice because a single layout cannot tell a real result from
    // an alignment draw. The pair is 5.4% of frame time and neither load is bounds checked, so
    // there was little to win, and the branch write-mirroring adds sits in the CPU write path.
    #[inline(always)]
    const fn mirror(addr: u16) -> usize {
        const PALETTE_MIRROR: [u8; 32] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 17, 18, 19, 4, 21, 22, 23, 8,
            25, 26, 27, 12, 29, 30, 31,
        ];
        PALETTE_MIRROR[(addr & 0x1F) as usize] as usize
    }
}

impl PaletteRam {
    /// Read a colour, with the $3F10/$3F14/$3F18/$3F1C backdrop mirrors applied.
    #[inline(always)]
    fn peek(&self, addr: u16) -> u8 {
        self.0[Self::mirror(addr)]
    }

    /// Write a colour, with the backdrop mirrors applied.
    #[inline(always)]
    fn write(&mut self, addr: u16, val: u8) {
        self.0[Self::mirror(addr)] = val;
    }
}

/// Whether a PPU address fetches a nametable attribute byte.
///
/// A free function rather than an extension trait on `u16`: with one implementor and no generic
/// use, the trait only bought an import at each of the four call sites.
#[inline(always)]
#[must_use]
pub const fn is_attr(addr: u16) -> bool {
    (addr & (size::NAMETABLE - 1)) >= addr::ATTR_OFFSET
}

/// Whether a PPU address lands in palette RAM.
#[inline(always)]
#[must_use]
pub const fn is_palette(addr: u16) -> bool {
    addr >= addr::PALETTE_START
}

/// NES PPU.
///
/// See: <https://wiki.nesdev.org/w/index.php/PPU>
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[must_use]
#[repr(C)]
pub struct Ppu {
    /// Master clock synced to Cpu master clock.
    pub master_clock: u32,
    /// (0, 340) cycles per scanline.
    pub cycle: u16,
    /// (0, 261) NTSC or (0, 311) PAL/Dendy scanlines per frame.
    pub scanline: u16,
    /// First dot on which the background is drawn, or [`Ppu::NEVER_DRAWN`] when it is not.
    ///
    /// Collapses "is the background on, and is this dot past the left-column clip" into one
    /// comparison for the pixel path, which asks it 61,440 times a frame.
    // Derived from `mask_bits`, kept in step by every $2001 write.
    pub mask_min_draw_bg_cycle: u16,
    /// First dot on which sprites are drawn. See [`Ppu::mask_min_draw_bg_cycle`].
    pub mask_min_draw_spr_cycle: u16,
    /// $2001 PPUMASK rendering enable, i.e. background or sprites shown (write-only).
    pub mask_rendering_enabled: bool,
    /// $2001 PPUMASK rendering enable as of the previous dot; it takes effect a cycle late.
    pub mask_prev_rendering_enabled: bool,
    /// Whether [`Ppu::mask_rendering_enabled`] still has that delayed update to apply.
    pub mask_pending_rendering_update: bool,
    // === 20 ===
    /// $2000 PPUCTRL background pattern table select, `$0000` or `$1000` (write-only).
    pub ctrl_bg_select: u16,
    /// $2000 PPUCTRL sprite pattern table select for 8x8, `$0000` or `$1000` (write-only).
    pub ctrl_spr_select: u16,
    /// $2000 PPUCTRL sprite height, 8 or 16 (write-only).
    pub ctrl_spr_height: u16,
    // === 26 ===
    /// $2002 PPUSTATUS sprite 0 hit (read-only).
    pub status_spr_zero_hit: bool,
    /// $2002 PPUSTATUS sprite overflow (read-only).
    pub status_spr_overflow: bool,
    /// $2002 PPUSTATUS vblank started (read-only).
    pub status_in_vblank: bool,
    // === 30 ===
    /// $2000 PPUCTRL VRAM address increment: 32 when set, 1 when clear (write-only).
    pub ctrl_vram_increment: bool,
    /// $2000 PPUCTRL master/slave select (write-only).
    pub ctrl_master_slave: u8,
    /// $2000 PPUCTRL NMI-on-vblank enable (write-only).
    pub ctrl_nmi_enabled: bool,
    /// $2005 PPUSCROLL and $2006 PPUADDR (write-only).
    pub scroll: Scroll,
    /// Scanline that Vertical Blank (VBlank) starts on.
    pub vblank_scanline: u16,
    /// Scanline that Prerender starts on.
    pub prerender_scanline: u16,
    /// Tile shift low byte.
    pub tile_shift_lo: u16,
    /// Tile shift high byte.
    pub tile_shift_hi: u16,
    /// Tile address.
    pub tile_addr: u16,
    /// Tile fetch buffer low byte.
    pub tile_lo: u8,
    /// Tile fetch buffer high byte.
    pub tile_hi: u8,
    /// Master clock divider.
    pub clock_divider: u8,
    /// Whatever was last read or written to to the Ppu.
    pub open_bus: u8,

    /// Current tile palette.
    pub curr_palette: u8,
    /// Previous tile palette.
    pub prev_palette: u8,
    /// Next tile palette.
    pub next_palette: u8,

    /// Scanline is visible. *Derived* from `scanline`; recomputed once per scanline so the
    /// per-dot paths test a bool instead of a range. Writing it does not move the PPU.
    pub is_visible_scanline: bool,
    /// Scanline is a pre-render scanline. *Derived* from `scanline`; see
    /// [`Ppu::is_visible_scanline`].
    pub is_prerender_scanline: bool,
    /// Scanline is a render scanline. *Derived* from `scanline`; see
    /// [`Ppu::is_visible_scanline`].
    pub is_render_scanline: bool,

    // === 64 : end of cache line ===
    // The rest of $2000. Nothing reads these while fetching - a $2000/$2007 access or a vblank NMI
    // does - so they stay out of the per-dot working set above.
    // The rest of $2001. Colour emphasis and greyscale are folded in over runs of pixels rather
    // than per pixel, so nothing here is read while fetching either.
    /// $2001 PPUMASK colour emphasis bits, positioned for the frame buffer (write-only).
    pub mask_emphasis: u16,
    /// $2001 PPUMASK greyscale mask applied to a colour: `0x30` when set, `0x3F` when clear.
    pub mask_grayscale: u8,
    /// $2001 PPUMASK raw register bits (write-only).
    pub mask_bits: mask::Bits,

    // Sprite/OAM evaluation.
    /// Sprite is in scanline range.
    pub spr_in_range: bool,
    /// Sprite 0 is in scanline range.
    pub spr_zero_in_range: bool,
    /// Secondary OAM address.
    pub secondary_oamaddr: u8,
    /// OAM evaluation is complete for scanline.
    pub oam_eval_done: bool,
    /// OAM address low byte.
    pub oamaddr_lo: u8,
    /// OAM address high byte.
    pub oamaddr_hi: u8,
    /// OAM data fetch buffer.
    pub oam_fetch: u8,
    /// $2003 OAM addr (write-only).
    pub oamaddr: u8,
    /// Sprite 0 is visible.
    pub spr_zero_visible: bool,
    /// Number of sprites on the current scanline.
    pub spr_count: u8,
    /// Sprite overflow count (> 8 on a scanline).
    pub overflow_count: u8,

    /// Frames run since power-on (FQ-08: counted at dot 1 of line 240).
    pub frame_count: u32,
    /// How much of the frame being drawn has had greyscale and colour emphasis applied.
    //
    // Rendering stores raw palette colours and the $2001 bits are folded in over whole runs of
    // pixels, so a frame that never touches them - which is nearly every frame - pays nothing per
    // pixel. Kept up to date whether or not pixels are being stored.
    pub color_bits_applied: usize,
    // === 104 ===
    /// Palette RAM: the 32 colours currently loaded, at $3F00-$3F1F.
    pub palette: PaletteRam,
    // === 136 ===
    /// Secondary OAM data on a given scanline.
    pub secondary_oamdata: ConstArray<u8, 32>,

    // === 160 ===
    /// Each scanline can hold 8 sprites at a time before the `spr_overflow` flag is set.
    pub sprites: [Sprite; 8],
    /// Which of the scanline's sprites cover each pixel (indexed by x, i.e. dot - 1), one bit per
    /// index into [`Ppu::sprites`].
    ///
    /// A set bit means that sprite's 8-pixel span contains the pixel, so the pixel path visits only
    /// the sprites that can contribute instead of scanning all [`Ppu::spr_count`] of them and
    /// range-testing each.
    // Rebuilt every scanline.
    pub spr_cover: ConstArray<u8, 256>,
    // === 520 ===
    /// $2004 Object Attribute Memory (OAM) data (read/write).
    pub oamdata: ConstArray<u8, 256>,

    // === 776 ===
    /// NMI pending.
    pub nmi_pending: bool,

    /// $2007 PPUDATA buffer.
    pub vram_buffer: u8,
    /// Prevents VBL from being triggered this frame.
    pub prevent_vbl: bool,
    /// FQ-01: frames left in which VBlank never sets - FCEUmm's `ppudead`, two frames after
    /// every reset in which its CPU runs a whole frame without VBlank.
    pub dead_frames: u8,
    /// FQ-03: PPUSTATUS bits 5-7 as they stood when the current CPU instruction began. FCEUmm
    /// raises and clears VBlank, and clears sprite 0 hit and overflow at the end of VBlank,
    /// between instructions, so a $2002 read sees those changes only from the next instruction
    /// on. (A sprite 0 hit or overflow raised during rendering is seen at once.)
    pub status_at_instr: u8,
    /// FQ-04/FQ-05: FCEUmm's NMI request. 1 = take the NMI before the next instruction (FQ-05),
    /// 2 = after one more instruction (FQ-04: NMI enabled through $2000 while VBlank is up).
    pub fceu_nmi: u8,
}

impl Default for Ppu {
    fn default() -> Self {
        Self::new()
    }
}

pub mod addr {
    //! Address constants.

    /// First nametable, at the start of the PPU's VRAM window.
    pub const NAMETABLE_START: u16 = 0x2000;
    /// Offset of a nametable's attribute table from its own start.
    pub const ATTR_OFFSET: u16 = 0x03C0;

    /// First palette entry, i.e. the universal background colour.
    pub const PALETTE_START: u16 = 0x3F00;
    /// One past the last palette entry.
    pub const PALETTE_END: u16 = 0x3F20;
}

pub mod size {
    //! Memory size constants.

    /// Visible pixels per scanline.
    pub const WIDTH: u16 = 256;
    /// Visible scanlines per frame.
    pub const HEIGHT: u16 = 240;
    /// Pixels in one frame.
    pub const FRAME: usize = (WIDTH * HEIGHT) as usize;

    /// Bytes in one nametable, attribute table included.
    pub const NAMETABLE: u16 = 0x0400;
    /// Bytes of primary OAM: 64 four-byte sprites, the most a frame can hold.
    pub const OAM: usize = 256;
    /// Bytes of secondary OAM: 8 four-byte sprites, the most a scanline can hold.
    pub const SECONDARY_OAM: usize = 32;

    /// Bytes of CIRAM, i.e. the two 1K nametables on the console itself.
    pub const VRAM: usize = 0x0800;
    /// Bytes of palette RAM: the 32 colours loadable at a time, out of the 64 the PPU can make.
    pub const PALETTE: usize = 32;
}

pub mod cycle {
    //! Cycle constants.
    //! <https://www.nesdev.org/wiki/PPU_rendering>

    use std::ops::RangeInclusive;

    /// First cycle of a scanline, an idle dot.
    pub const START: u16 = 0;
    /// The cycle odd frames skip when rendering is enabled.
    pub const ODD_SKIP: u16 = 339;
    /// Last cycle of a scanline.
    pub const END: u16 = 340;

    /// Tile data fetching starts.
    pub const VISIBLE_START: u16 = 1;
    /// Tile data fetching ends: 2 cycles each for 4 fetches = 32 tiles.
    pub const VISIBLE_END: u16 = 256;

    /// Cycle on which the VBlank flag is set and cleared.
    pub const VBLANK: u16 = VISIBLE_START;

    /// Secondary OAM clear starts.
    pub const OAM_CLEAR_START: u16 = 1;
    /// Secondary OAM clear ends.
    pub const OAM_CLEAR_END: u16 = 64;

    /// Sprite evaluation for the next scanline starts.
    pub const SPR_EVAL_START: u16 = 65;
    /// One past [`SPR_EVAL_START`], to split up match arms.
    pub const SPR_EVAL_START1: u16 = 66;
    /// One before [`SPR_EVAL_END`], to split up match arms.
    pub const SPR_EVAL_END0: u16 = 255;
    /// Sprite evaluation ends.
    pub const SPR_EVAL_END: u16 = 256;
    /// Fetching the next scanline's sprites starts.
    pub const SPR_FETCH_START: u16 = 257;
    /// Sprite fetching ends: 2 cycles each for 4 fetches = 8 sprites.
    pub const SPR_FETCH_END: u16 = 320;
    /// [`SPR_FETCH_START`]..=[`SPR_FETCH_END`].
    pub const SPR_FETCH_RANGE: RangeInclusive<u16> = SPR_FETCH_START..=SPR_FETCH_END;

    /// Prefetching the next scanline's tile data starts.
    pub const BG_PREFETCH_START: u16 = 321;
    /// Background prefetch ends: 2 cycles each for 4 fetches = 2 tiles.
    pub const BG_PREFETCH_END: u16 = 336;
    /// [`BG_PREFETCH_START`]..=[`BG_PREFETCH_END`].
    pub const BG_PREFETCH_RANGE: RangeInclusive<u16> = BG_PREFETCH_START..=BG_PREFETCH_END;

    /// Two dummy nametable fetches start; what the hardware does with them is unknown.
    pub const BG_DUMMY_START: u16 = 337;
    /// Dummy fetches end, at the end of the scanline.
    pub const BG_DUMMY_END: u16 = END;

    /// Increment the Y scroll, the screen's last visible pixel having been reached.
    pub const INC_Y: u16 = 256;
    /// Copying the Y scroll from `t` to `v` starts (pre-render scanline only).
    pub const COPY_Y_START: u16 = 280;
    /// Copying the Y scroll ends.
    pub const COPY_Y_END: u16 = 304;
    /// [`COPY_Y_START`]..=[`COPY_Y_END`].
    pub const COPY_Y_RANGE: RangeInclusive<u16> = COPY_Y_START..=COPY_Y_END;

    /// Master clock cycles per PPU cycle on NTSC.
    pub const DIVIDER_NTSC: u8 = 4;
}

pub mod scanline {
    //! Scanline constants.
    //! <https://www.nesdev.org/wiki/PPU_rendering>

    /// First scanline of a frame.
    pub const START: u16 = 0;

    /// First visible scanline.
    pub const VISIBLE_START: u16 = START;
    /// Last visible scanline.
    pub const VISIBLE_END: u16 = 239;

    /// The idle scanline between the last visible one and VBlank.
    pub const POSTRENDER: u16 = 240;
    /// NTSC pre-render scanline, where the next frame's state is set up.
    pub const PRERENDER_NTSC: u16 = 261;

    /// NTSC scanline on which VBlank starts.
    pub const VBLANK_NTSC: u16 = 241;
}

impl Ppu {
    /// The NTSC palette the `Ntsc` video filter samples, as raw RGB triples.
    pub const NTSC_PALETTE: &'static [u8] = include_bytes!("../ntscpalette.pal");

    /// NES PPU System Palette
    /// 64 total possible colors, though only 32 can be loaded at a time
    #[rustfmt::skip]
    pub const SYSTEM_PALETTE: [(u8,u8,u8); 64] = [
        // 0x00
        (0x54, 0x54, 0x54), (0x00, 0x1E, 0x74), (0x08, 0x10, 0x90), (0x30, 0x00, 0x88), // $00-$03
        (0x44, 0x00, 0x64), (0x5C, 0x00, 0x30), (0x54, 0x04, 0x00), (0x3C, 0x18, 0x00), // $04-$07
        (0x20, 0x2A, 0x00), (0x08, 0x3A, 0x00), (0x00, 0x40, 0x00), (0x00, 0x3C, 0x00), // $08-$0B
        (0x00, 0x32, 0x3C), (0x00, 0x00, 0x00), (0x00, 0x00, 0x00), (0x00, 0x00, 0x00), // $0C-$0F
        // 0x10
        (0x98, 0x96, 0x98), (0x08, 0x4C, 0xC4), (0x30, 0x32, 0xEC), (0x5C, 0x1E, 0xE4), // $10-$13
        (0x88, 0x14, 0xB0), (0xA0, 0x14, 0x64), (0x98, 0x22, 0x20), (0x78, 0x3C, 0x00), // $14-$17
        (0x54, 0x5A, 0x00), (0x28, 0x72, 0x00), (0x08, 0x7C, 0x00), (0x00, 0x76, 0x28), // $18-$1B
        (0x00, 0x66, 0x78), (0x00, 0x00, 0x00), (0x00, 0x00, 0x00), (0x00, 0x00, 0x00), // $1C-$1F
        // 0x20
        (0xEC, 0xEE, 0xEC), (0x4C, 0x9A, 0xEC), (0x78, 0x7C, 0xEC), (0xB0, 0x62, 0xEC), // $20-$23
        (0xE4, 0x54, 0xEC), (0xEC, 0x58, 0xB4), (0xEC, 0x6A, 0x64), (0xD4, 0x88, 0x20), // $24-$27
        (0xA0, 0xAA, 0x00), (0x74, 0xC4, 0x00), (0x4C, 0xD0, 0x20), (0x38, 0xCC, 0x6C), // $28-$2B
        (0x38, 0xB4, 0xCC), (0x3C, 0x3C, 0x3C), (0x00, 0x00, 0x00), (0x00, 0x00, 0x00), // $2C-$2F
        // 0x30
        (0xEC, 0xEE, 0xEC), (0xA8, 0xCC, 0xEC), (0xBC, 0xBC, 0xEC), (0xD4, 0xB2, 0xEC), // $30-$33
        (0xEC, 0xAE, 0xEC), (0xEC, 0xAE, 0xD4), (0xEC, 0xB4, 0xB0), (0xE4, 0xC4, 0x90), // $34-$37
        (0xCC, 0xD2, 0x78), (0xB4, 0xDE, 0x78), (0xA8, 0xE2, 0x90), (0x98, 0xE2, 0xB4), // $38-$3B
        (0xA0, 0xD6, 0xE4), (0xA0, 0xA2, 0xA0), (0x00, 0x00, 0x00), (0x00, 0x00, 0x00), // $3C-$3F
    ];

    /// Create a new PPU instance.
    pub fn new() -> Self {
        let mut ppu = Self {
            master_clock: 0,
            clock_divider: cycle::DIVIDER_NTSC,
            cycle: 0,
            scanline: 0,
            vblank_scanline: scanline::VBLANK_NTSC,
            prerender_scanline: scanline::PRERENDER_NTSC,
            is_visible_scanline: true,
            is_prerender_scanline: false,
            is_render_scanline: true,
            open_bus: 0x00,

            ctrl_bg_select: 0,
            ctrl_spr_select: 0,
            ctrl_spr_height: 8,
            ctrl_vram_increment: false,
            ctrl_master_slave: 0,
            ctrl_nmi_enabled: false,

            status_spr_zero_hit: false,
            status_spr_overflow: false,
            status_in_vblank: false,

            mask_min_draw_bg_cycle: 0,
            mask_min_draw_spr_cycle: 0,
            mask_rendering_enabled: false,
            mask_prev_rendering_enabled: false,
            mask_pending_rendering_update: false,
            mask_emphasis: 0,
            mask_grayscale: 0x3F,
            mask_bits: mask::Bits::empty(),

            scroll: Scroll::new(),

            // NOTE: PPU RAM is a bit more predictable at power on - games like Huge Insect don't
            // properly initialize both nametables, which can result in garbage sprites when
            // randomizing CIRAM.
            palette: PaletteRam(ConstArray::new()),

            prev_palette: 0x00,
            curr_palette: 0x00,
            next_palette: 0x00,
            tile_shift_lo: 0x0000,
            tile_shift_hi: 0x0000,
            tile_lo: 0x00,
            tile_hi: 0x00,
            tile_addr: 0x0000,

            nmi_pending: false,

            oam_fetch: 0x00,
            oamaddr: 0x0000,
            oamaddr_lo: 0x00,
            oamaddr_hi: 0x00,
            oam_eval_done: false,
            secondary_oamaddr: 0x0000,
            overflow_count: 0,
            spr_in_range: false,
            spr_zero_in_range: false,
            spr_zero_visible: false,
            spr_count: 0,
            vram_buffer: 0x00,

            oamdata: ConstArray::new(),
            secondary_oamdata: ConstArray::new(),
            sprites: [Sprite::new(); 8],
            spr_cover: ConstArray::new(),

            prevent_vbl: false,
            frame_count: 0,
            color_bits_applied: 0,

            dead_frames: 0,
            status_at_instr: 0,
            fceu_nmi: 0,
        };

        // Decode both registers from zero rather than trusting the literals above: everything
        // derived from them - the draw thresholds especially, whose "never" value is not zero -
        // then has exactly one definition.
        ppu.write_ctrl(0);
        ppu.write_mask(0);
        ppu.update_emphasis();

        ppu
    }

    /// Return the current frame number.
    #[inline(always)]
    #[must_use]
    pub const fn frame_number(&self) -> u32 {
        self.frame_count
    }

    /// Whether the current frame number is odd.
    #[inline(always)]
    #[must_use]
    pub const fn frame_is_odd(&self) -> bool {
        self.frame_count & 1 == 1
    }

    /// The master clock by which the PPU must be caught up for the CPU to see its next
    /// frame-boundary (FQ-08, line 240 dot 1) or NMI-request (FQ-05, line 241 dot 13) dot.
    ///
    /// Counts dots from the current one assuming the odd-frame skip (FQ-09) happens on the way,
    /// so it is never late; when it does not, the catch-up comes a dot early and this is asked
    /// again.
    #[must_use]
    pub(crate) fn next_event_clock(&self) -> u32 {
        const LINE: u32 = cycle::END as u32 + 1;
        let frame = u32::from(self.prerender_scanline + 1) * LINE;
        let pos = u32::from(self.scanline) * LINE + u32::from(self.cycle);
        let dots_to = |line: u16, dot: u16| {
            let target = u32::from(line) * LINE + u32::from(dot);
            match (target + frame - pos) % frame {
                0 => frame,
                d => d,
            }
        };
        let frame_end = dots_to(self.vblank_scanline - 1, 1);
        let nmi = dots_to(self.vblank_scanline, 13);
        let dots = frame_end.min(nmi) - 1;
        self.master_clock + dots * u32::from(self.clock_divider)
    }

    /// Load the passed given buffer with RGBA pixels from the current nametables.
    ///
    /// `chr` is `$0000-$2FFF` as the PPU currently sees it - pattern tables and nametables, banked
    /// and mirrored - which [`Bus::copy_ppu_bus`](crate::bus::Bus::copy_ppu_bus) fills. Taking it
    /// as an argument is what lets a debugger render from a snapshot on another thread.
    pub fn load_nametables(&self, chr: &[u8], nametables: &mut [u8]) {
        for i in 0..4 {
            let base_addr = addr::NAMETABLE_START + i * size::NAMETABLE;
            let x_offset = (i % 2) * size::WIDTH;
            let y_offset = (i / 2) * size::HEIGHT;

            for addr in base_addr..(base_addr + size::NAMETABLE - 64) {
                let x_scroll = addr & Scroll::COARSE_X_MASK;
                let y_scroll = (addr & Scroll::COARSE_Y_MASK) >> 5;

                let base_nametable_addr =
                    addr::NAMETABLE_START | (addr & (Scroll::NT_X_MASK | Scroll::NT_Y_MASK));
                let base_attr_addr = base_nametable_addr + addr::ATTR_OFFSET;

                let tile_index = u16::from(chr[usize::from(addr)]);
                let tile_addr = self.ctrl_bg_select | (tile_index << 4);

                let supertile = ((y_scroll & 0xFC) << 1) + (x_scroll >> 2);
                let attr = u16::from(chr[usize::from(base_attr_addr + supertile)]);
                let attr_shift = (x_scroll & 0x02) | ((y_scroll & 0x02) << 1);
                let palette_addr = ((attr >> attr_shift) & 0x03) << 2;

                let tile_num = x_scroll + (y_scroll << 5);
                let tile_x = (tile_num % 32) << 3;
                let tile_y = (tile_num / 32) << 3;

                for y in 0..8 {
                    let tile_addr = tile_addr + y;
                    let tile_lo = chr[usize::from(tile_addr)];
                    let tile_hi = chr[usize::from(tile_addr + 8)];
                    for x in 0..8 {
                        let tile_palette = (((tile_hi >> x) & 1) << 1) | (tile_lo >> x) & 1;
                        let palette = palette_addr | u16::from(tile_palette);
                        let color = self
                            .palette
                            .peek(addr::PALETTE_START | ((palette & 0x03 > 0) as u16 * palette));
                        let x = tile_x + (7 - x);
                        let y = tile_y + y;
                        Self::set_pixel(
                            u16::from(color & self.mask_grayscale) | self.mask_emphasis,
                            x + x_offset,
                            y + y_offset,
                            2 * size::WIDTH,
                            nametables,
                        );
                    }
                }
            }
        }
    }

    /// Load the given buffer with RGBA pixels from the current pattern tables.
    ///
    /// See [`Ppu::load_nametables`] for what `chr` is.
    pub fn load_pattern_tables(&self, chr: &[u8], pattern_tables: &mut [u8]) {
        for i in 0..2 {
            let start = i * 0x1000;
            let end = start + 0x1000;
            let x_offset = (i % 2) * size::WIDTH / 2;
            for tile_addr in (start..end).step_by(16) {
                let tile_x = ((tile_addr % 0x1000) % 256) / 2;
                let tile_y = ((tile_addr % 0x1000) / 256) * 8;
                for y in 0..8 {
                    let tile_lo = u16::from(chr[usize::from(tile_addr + y)]);
                    let tile_hi = u16::from(chr[usize::from(tile_addr + y + 8)]);
                    for x in 0..8 {
                        let palette = (((tile_hi >> x) & 0x01) << 1) | ((tile_lo >> x) & 0x01);
                        let color = u16::from(self.palette.peek(addr::PALETTE_START | palette));
                        let x = tile_x + (7 - x);
                        let y = tile_y + y;
                        Self::set_pixel(color, x + x_offset, y, size::WIDTH, pattern_tables);
                    }
                }
            }
        }
    }

    /// Load the OAM entries as a grid of tiles, the screen they place, and their decoded sprites.
    ///
    /// `oam_table` is a 64x64 grid of one 8x8 cell per entry, so an 8x16 entry shows its top tile.
    /// `sprite_nametable` is a full screen showing sprites alone. Background priority needs the
    /// background pixel under each sprite, which exists only while the PPU renders that dot, so a
    /// snapshot taken at one dot cannot mix the two. [`Sprite::bg_priority`] reports the flag.
    pub fn load_oam(
        &self,
        chr: &[u8],
        oam_table: &mut [u8],
        sprite_nametable: &mut [u8],
        sprites: &mut [Sprite],
    ) {
        // TODO: de-duplicate this with load_sprites
        for (i, oamdata) in self.oamdata.chunks(4).enumerate().rev() {
            if let [y, tile_index, attr, x] = oamdata {
                let sprite_x = u16::from(*x);
                let sprite_y = u16::from(*y);
                let tile_index = u16::from(*tile_index);
                let palette = ((attr & 0x03) << 2) | 0x10;
                let bg_priority = (attr & 0x20) == 0x20;
                let flip_horizontal = (attr & 0x40) == 0x40;
                let flip_vertical = (attr & 0x80) == 0x80;

                let height = self.ctrl_spr_height;
                let tile_addr = if height == 16 {
                    // Use bit 0 of tile index to determine pattern table
                    ((tile_index & 0x01) * 0x1000) | ((tile_index & 0xFE) << 4)
                } else {
                    self.ctrl_spr_select | (tile_index << 4)
                };

                sprites[i] = Sprite {
                    x: sprite_x,
                    y: sprite_y,
                    tile_addr,
                    palette,
                    bg_priority,
                    flip_horizontal,
                    ..Sprite::default()
                };

                let tile_x = (i % 8) as u16 * 8;
                let tile_y = (i / 8) as u16 * 8;
                // Every row the sprite has, so an 8x16 reaches the screen whole. The `+= 8`
                // below steps over the gap to an 8x16's second tile, a tile past the first.
                for y in 0..height {
                    let mut line_offset = if flip_vertical { (height) - 1 - y } else { y };
                    if height == 16 && line_offset >= 8 {
                        line_offset += 8;
                    }
                    let tile_lo = chr[usize::from(tile_addr + line_offset)];
                    let tile_hi = chr[usize::from(tile_addr + line_offset + 8)];
                    for x in 0..8 {
                        let spr_color = if flip_horizontal {
                            (((tile_hi >> x) & 0x01) << 1) | ((tile_lo >> x) & 0x01)
                        } else {
                            (((tile_hi << x) & 0x80) >> 6) | (((tile_lo << x) & 0x80) >> 7)
                        };
                        let palette = palette + spr_color;
                        let color = self.palette.peek(
                            addr::PALETTE_START
                                | ((palette & 0x03 > 0) as u16 * u16::from(palette)),
                        );

                        // One 8x8 cell per sprite, so the grid takes the top tile of an 8x16.
                        if y < 8 {
                            Self::set_pixel(
                                u16::from(color),
                                tile_x + x,
                                tile_y + y,
                                64,
                                oam_table,
                            );
                        }

                        let x = sprite_x + x;
                        // Sprites are fetched on the scanline before the one they appear on, so
                        // an entry's first row lands a scanline below the Y it names.
                        let y = sprite_y + y + 1;
                        let left_clip_spr = x < 8 && !self.show_left_spr();
                        // A transparent pixel is the absence of a sprite, so it leaves what is
                        // already on the screen. Painting the backdrop there lets the empty corner
                        // of one sprite rub out the sprite beside it.
                        if spr_color != 0
                            && self.show_spr()
                            && !left_clip_spr
                            && x < size::WIDTH
                            && y < size::HEIGHT
                        {
                            Self::set_pixel(u16::from(color), x, y, size::WIDTH, sprite_nametable);
                        }
                    }
                }
            }
        }
    }

    /// Load the given buffer with RGBA pixels from the current palettes.
    pub fn load_palettes(&self, palettes: &mut [u8], colors: &mut [u8]) {
        for addr in addr::PALETTE_START..addr::PALETTE_END {
            let offset = addr - addr::PALETTE_START;
            let x = offset % 16;
            let y = offset / 16;
            let color = self.palette.peek(addr);
            colors[usize::from(offset)] = color;
            Self::set_pixel(u16::from(color), x, y, 16, palettes);
        }
    }

    fn set_pixel(color: u16, x: u16, y: u16, width: u16, pixels: &mut [u8]) {
        let index = (color as usize) * 3;
        let idx = 4 * (usize::from(x) + usize::from(y) * usize::from(width));
        assert!(Ppu::NTSC_PALETTE.len() > index + 2);
        assert!(pixels.len() > 2);
        assert!(idx + 2 < pixels.len());
        pixels[idx] = Ppu::NTSC_PALETTE[index];
        pixels[idx + 1] = Ppu::NTSC_PALETTE[index + 1];
        pixels[idx + 2] = Ppu::NTSC_PALETTE[index + 2];
        pixels[idx + 3] = 0xFF;
    }

    #[inline(always)]
    const fn increment_vram_addr(&mut self) {
        // During rendering, v increments coarse X and coarse Y simultaneously
        if self.scanline > scanline::VISIBLE_END || !self.mask_rendering_enabled {
            self.scroll
                .increment(self.ctrl_vram_increment as u16 * 31 + 1);
        } else {
            self.scroll.increment_x();
            self.scroll.increment_y();
        }
    }

    fn start_vblank(&mut self) {
        // FQ-01: no VBlank in FCEUmm's dead frames after a reset.
        if !self.prevent_vbl && self.dead_frames == 0 {
            self.set_in_vblank(true);
            // FQ-05: FCEUmm requests the NMI 12 dots later (see `Bus::fceumm_events`).
        }
        self.prevent_vbl = false;
    }

    fn stop_vblank(&mut self) {
        self.set_spr_zero_hit(false);
        self.set_spr_overflow(false);
        self.reset_in_vblank();
        self.nmi_pending = false;
        // FQ-13: the I/O latch (`open_bus`) never decays; FCEUmm's `PPUGenLatch` keeps its value.
    }

    /// One dot of sprite evaluation (dots 65-256): odd dots read OAM, even dots write secondary
    /// OAM and step the OAM address.
    #[inline(always)]
    fn oam_eval_cycle(&mut self, cycle: u16) {
        self.oam_eval_dots(cycle, cycle);
    }

    /// Sprite evaluation over dots `from..=to` (within 65-256), the state held in locals for the
    /// run and stored once at the end.
    #[inline(always)]
    fn oam_eval_dots(&mut self, from: u16, to: u16) {
        let scanline = self.scanline;
        let height = self.ctrl_spr_height;
        let mut oamaddr = self.oamaddr;
        let mut oam_eval_done = self.oam_eval_done;
        let mut secondary_oamaddr = self.secondary_oamaddr;
        let mut oam_fetch = self.oam_fetch;
        let mut spr_in_range = self.spr_in_range;
        let mut spr_zero_in_range = self.spr_zero_in_range;
        let mut oamaddr_hi = self.oamaddr_hi;
        let mut oamaddr_lo = self.oamaddr_lo;
        let mut overflow_count = self.overflow_count;
        let mut overflow = self.status_spr_overflow;

        // One even dot: a write to secondary OAM and a step of the OAM address.
        macro_rules! even_dot {
            () => {
                let secondary_oamindex = usize::from(secondary_oamaddr) & 0x1F;

                // oamaddr rolled over, so we're done reading
                if oam_eval_done {
                    oamaddr_hi = (oamaddr_hi + 1) & 0x3F;
                    if secondary_oamaddr >= 0x20 {
                        oam_fetch = self.secondary_oamdata[secondary_oamindex];
                    }
                } else {
                    // If previously not in range, interpret this byte as y
                    let y = u16::from(oam_fetch);
                    spr_in_range |= !spr_in_range && (y..y + height).contains(&scanline);

                    // Even cycles are writes to Secondary OAM
                    if secondary_oamaddr < 0x20 {
                        self.secondary_oamdata[secondary_oamindex] = oam_fetch;

                        if spr_in_range {
                            oamaddr_lo += 1;
                            secondary_oamaddr += 1;

                            spr_zero_in_range |= oamaddr_hi == 0x00;
                            if oamaddr_lo == 0x04 {
                                spr_in_range = false;
                                oamaddr_lo = 0x00;
                                oamaddr_hi = (oamaddr_hi + 1) & 0x3F;
                                oam_eval_done |= oamaddr_hi == 0x00;
                            }
                        } else {
                            oamaddr_hi = (oamaddr_hi + 1) & 0x3F;
                            oam_eval_done |= oamaddr_hi == 0x00;
                        }
                    } else {
                        oam_fetch = self.secondary_oamdata[secondary_oamindex];
                        if spr_in_range {
                            overflow = true;
                            oamaddr_lo += 1;
                            if oamaddr_lo == 0x04 {
                                oamaddr_lo = 0x00;
                                oamaddr_hi = (oamaddr_hi + 1) & 0x3F;
                            }

                            match overflow_count.cmp(&0) {
                                Ordering::Equal => overflow_count = 3,
                                Ordering::Greater => {
                                    overflow_count -= 1;
                                    let no_overflow = overflow_count == 0;
                                    oam_eval_done |= no_overflow;
                                    if no_overflow {
                                        oamaddr_lo = 0;
                                    }
                                }
                                Ordering::Less => (),
                            }
                        } else {
                            oamaddr_hi = (oamaddr_hi + 1) & 0x3F;
                            oamaddr_lo = (oamaddr_lo + 1) & 0x03;
                            oam_eval_done |= oamaddr_hi == 0x00;
                        }
                    }
                }
                oamaddr = (oamaddr_hi << 2) | (oamaddr_lo & 0x03);
            };
        }

        // Odd dots read OAM, even dots write: run them as (odd, even) pairs.
        let mut dot = from;
        if dot & 0x01 == 0 && dot <= to {
            even_dot!();
            dot += 1;
        }
        while dot < to {
            let pairs = (to - dot + 1) / 2;
            if oam_eval_done {
                // Done: each pair reads OAM, steps the address and, past eight sprites, reloads
                // the fetch from secondary OAM. A run of them leaves the last read and the address.
                let last_read = if pairs == 1 {
                    oamaddr
                } else {
                    (((oamaddr_hi + pairs as u8 - 1) & 0x3F) << 2) | (oamaddr_lo & 0x03)
                };
                oam_fetch = if secondary_oamaddr >= 0x20 {
                    self.secondary_oamdata[usize::from(secondary_oamaddr) & 0x1F]
                } else {
                    self.oamdata[usize::from(last_read)]
                };
                oamaddr_hi = (oamaddr_hi + pairs as u8) & 0x3F;
                oamaddr = (oamaddr_hi << 2) | (oamaddr_lo & 0x03);
                dot += 2 * pairs;
                break;
            }
            if !spr_in_range
                && secondary_oamaddr < 0x20
                && oamaddr_lo == 0
                && oamaddr == oamaddr_hi << 2
            {
                // Scanning: a sprite out of range costs a pair that reads its Y, writes it to
                // the same secondary OAM slot and steps to the next sprite. A run of them leaves
                // only the last Y and the address.
                let max = pairs.min(64 - u16::from(oamaddr_hi)) as u8;
                let mut run = 0;
                while run < max {
                    let y = u16::from(self.oamdata[usize::from(oamaddr_hi + run) << 2]);
                    if scanline.wrapping_sub(y) < height {
                        break;
                    }
                    run += 1;
                }
                if run > 0 {
                    oam_fetch = self.oamdata[usize::from(oamaddr_hi + run - 1) << 2];
                    self.secondary_oamdata[usize::from(secondary_oamaddr) & 0x1F] = oam_fetch;
                    oamaddr_hi = (oamaddr_hi + run) & 0x3F;
                    oam_eval_done |= oamaddr_hi == 0x00;
                    oamaddr = oamaddr_hi << 2;
                    dot += 2 * u16::from(run);
                    continue;
                }
            }
            oam_fetch = self.oamdata[usize::from(oamaddr)];
            even_dot!();
            dot += 2;
        }
        if dot == to {
            oam_fetch = self.oamdata[usize::from(oamaddr)];
        }

        self.oamaddr = oamaddr;
        self.oamaddr_hi = oamaddr_hi;
        self.oamaddr_lo = oamaddr_lo;
        self.oam_eval_done = oam_eval_done;
        self.secondary_oamaddr = secondary_oamaddr;
        self.oam_fetch = oam_fetch;
        self.spr_in_range = spr_in_range;
        self.spr_zero_in_range = spr_zero_in_range;
        self.overflow_count = overflow_count;
        self.status_spr_overflow = overflow;
    }

    fn spr_eval_cycle(&mut self) {
        // Local variables improve cache locality
        match self.cycle {
            // 1. Clear Secondary OAM
            // 1..=64
            cycle::OAM_CLEAR_START..=cycle::OAM_CLEAR_END => {
                self.oam_fetch = 0xFF;
                // Hardware clears secondary OAM one byte at a time across cycles 1-64. Nothing
                // reads it until sprite evaluation starts at cycle 65, so filling once on the
                // last cycle of the range leaves identical state - including when rendering is
                // enabled part way through - while avoiding rewriting all 32 bytes on all 64
                // cycles of every scanline.
                if self.cycle == cycle::OAM_CLEAR_END {
                    self.secondary_oamdata = ConstArray::filled(0xFF);
                }
            }
            // 2. Read OAM to find first eight sprites on this scanline
            // 3. With > 8 sprites, check (wrongly) for more sprites to set overflow flag
            // 64..=256
            cycle::SPR_EVAL_START => {
                self.spr_in_range = false;
                self.spr_zero_in_range = false;
                self.secondary_oamaddr = 0x00;
                self.oam_eval_done = false;
                self.oamaddr_hi = (self.oamaddr >> 2) & 0x3F;
                self.oamaddr_lo = self.oamaddr & 0x03;
                self.oam_eval_cycle(self.cycle);
            }
            cycle::SPR_EVAL_END => {
                self.spr_zero_visible = self.spr_zero_in_range;
                self.spr_count = self.secondary_oamaddr >> 2;
                self.oam_eval_cycle(self.cycle);
            }
            cycle::SPR_EVAL_START1..=cycle::SPR_EVAL_END0 => self.oam_eval_cycle(self.cycle),
            _ => (),
        }
    }

    /// Reload the BG tile shift registers from the last fetched tile bytes and rotate the
    /// attribute palette latches, at the start of each 8-dot tile fetch.
    //
    // Not run on the garbage sprite-period NT fetches or the dummy NT fetches at 337-340:
    // hardware has no reload event at either, and rotating on the dummy fetches would collapse
    // the palette latch pipeline before dot 1, rendering the first tile's left pixels with the
    // second prefetched tile's attributes.
    #[inline(always)]
    fn reload_bg_shifters(&mut self) {
        self.prev_palette = self.curr_palette;
        self.curr_palette = self.next_palette;

        self.tile_shift_lo |= u16::from(self.tile_lo);
        self.tile_shift_hi |= u16::from(self.tile_hi);
    }

    #[inline]
    fn pixel_palette(&mut self, cycle: u16) -> u8 {
        let x = cycle - 1;
        let fine_x = self.scroll.fine_x;
        let bg_shift = 15 - fine_x;

        let bg_mask = u8::from(cycle > self.mask_min_draw_bg_cycle);
        let bg_color = bg_mask
            * ((((self.tile_shift_hi >> bg_shift) & 0x01) << 1)
                | ((self.tile_shift_lo >> bg_shift) & 0x01)) as u8;

        let mut covering = self.spr_cover[usize::from(x)];
        if (covering != 0) & (cycle > self.mask_min_draw_spr_cycle) {
            while covering != 0 {
                // Lowest set bit first, which is sprite priority order.
                let i = covering.trailing_zeros() as usize;
                covering &= covering - 1;
                let sprite = &self.sprites[i];

                // The cover bits are rebuilt from `sprites` each scanline, but only when rendering
                // was on at dot 257 - toggling it mid-frame can leave a bit set against a sprite
                // that has since moved, so the span is still checked here.
                let spr_shift = x.wrapping_sub(sprite.x);
                if spr_shift <= 7 {
                    let spr_shift = if sprite.flip_horizontal {
                        spr_shift
                    } else {
                        7 - spr_shift
                    };
                    let spr_color = (((sprite.tile_hi >> spr_shift) & 0x01) << 1)
                        | ((sprite.tile_lo >> spr_shift) & 0x01);

                    if spr_color != 0 {
                        if self.mask_rendering_enabled
                            & !self.status_spr_zero_hit
                            & self.spr_zero_visible
                            & (cycle != 256)
                            & (i == 0)
                            & (bg_color != 0)
                        {
                            // Assigned directly rather than through `set_spr_zero_hit`: that takes
                            // `&mut self`, which would conflict with the borrow of `self.sprites`
                            // above and force a copy of the sprite per pixel.
                            self.status_spr_zero_hit = true;
                        }

                        if !sprite.bg_priority | (bg_color == 0) {
                            return sprite.palette + spr_color;
                        }
                        break;
                    }
                }
            }
        }

        let palette_mask = u8::from((fine_x + (x & 0x07)) < 8);
        let palette = palette_mask * self.prev_palette + (1 - palette_mask) * self.curr_palette;
        palette + bg_color
    }

    /// Sprite 0 hit detection at one visible dot when no pixels are being drawn.
    #[inline]
    fn headless_sprite_zero_hit(&mut self, cycle: u16) {
        if !self.spr_zero_visible || self.status_spr_zero_hit {
            return;
        }

        let bg_mask = u8::from(cycle > self.mask_min_draw_bg_cycle);
        if (bg_mask == 0)
            | (cycle <= self.mask_min_draw_spr_cycle)
            | (cycle == 256)
            | (self.spr_cover[usize::from(cycle - 1)] == 0)
        {
            return;
        }

        let bg_shift = 15 - self.scroll.fine_x;
        let bg_color = bg_mask
            * ((((self.tile_shift_hi >> bg_shift) & 0x01) << 1)
                | ((self.tile_shift_lo >> bg_shift) & 0x01)) as u8;
        if bg_color == 0 {
            return;
        }

        let sprite = &self.sprites[0];
        let spr_shift = cycle.wrapping_sub(sprite.x).wrapping_sub(1);
        if spr_shift <= 7 {
            let spr_shift = if sprite.flip_horizontal {
                spr_shift
            } else {
                7 - spr_shift
            };
            let spr_color = (((sprite.tile_hi >> spr_shift) & 0x01) << 1)
                | ((sprite.tile_lo >> spr_shift) & 0x01);
            if spr_color != 0 {
                self.set_spr_zero_hit(true);
            }
        }
    }

    #[inline(always)]
    fn render_pixel(&mut self, frame: &mut crate::bus::FrameBuffer, cycle: u16) {
        let addr = self.scroll.addr();
        let color = if self.mask_rendering_enabled || !is_palette(addr) {
            let palette = u16::from(self.pixel_palette(cycle));
            self.palette
                .peek(addr::PALETTE_START | ((palette & 0x03 > 0) as u16 * palette))
        } else {
            self.palette.peek(addr)
        };

        frame[usize::from(cycle - 1) + (usize::from(self.scanline) << 8)] = u16::from(color);
    }

    /// One past the frame-buffer index of the last pixel rendered.
    #[inline]
    fn rendered_through(&self) -> usize {
        if self.scanline > scanline::VISIBLE_END {
            return size::FRAME;
        }
        (usize::from(self.scanline) << 8) + usize::from(self.cycle).min(256)
    }

    /// Fold greyscale and colour emphasis into every pixel rendered since this last ran.
    ///
    /// Called where the $2001 bits are about to change and once at the end of a frame, so each run
    /// is covered by the settings it was drawn under. Neither bit is set in the overwhelming
    /// majority of frames, which is the case this exists to make free. Without a `frame` (this
    /// frame is not being drawn) only the bookkeeping moves.
    fn apply_color_bits(&mut self, through: usize, frame: Option<&mut crate::bus::FrameBuffer>) {
        let through = through.min(size::FRAME);
        if through <= self.color_bits_applied {
            return;
        }
        if let Some(frame) = frame
            && (self.mask_grayscale != 0x3F || self.mask_emphasis != 0)
        {
            let grayscale = u16::from(self.mask_grayscale);
            let emphasis = self.mask_emphasis;
            for pixel in &mut frame[self.color_bits_applied..through] {
                *pixel = (*pixel & grayscale) | emphasis;
            }
        }
        self.color_bits_applied = through;
    }

    // $2002 | R   | PPUSTATUS
    //       | 0-5 | Unknown (???)
    //       |   6 | Sprite0 Hit Flag, 1 = PPU rendering has hit sprite #0
    //       |     | This flag resets to 0 when VBlank starts, or CPU reads $2002
    //       |   7 | VBlank Flag, 1 = PPU is generating a Vertical Blanking Impulse
    //       |     | This flag resets to 0 when VBlank ends, or CPU reads $2002
    /// Reads $2002 PPUSTATUS, which clears the VBlank flag and resets the address latch as a side
    /// effect. Use [`Ppu::peek_status`] to look without disturbing it.
    ///
    /// As FCEUmm does it:
    /// - FQ-03: VBlank is raised and cleared between instructions, so the read sees the flag as of
    ///   the start of the reading instruction and clears only that; a VBlank raised during the
    ///   instruction survives it.
    /// - FQ-04: the read never cancels an NMI.
    /// - FQ-02: there is no read-before-VBlank suppression race.
    /// - FQ-13: the returned byte (status bits over the latch's low five) becomes the I/O latch.
    pub fn read_status(&mut self) -> u8 {
        // Bits 5/6: a set during the instruction shows at once, a clear only from the next one.
        let status = (self.status_at_instr & 0x80)
            | ((self.read_status_bits() | self.status_at_instr) & 0x60)
            | (self.open_bus & 0x1F);
        self.open_bus = status; // FQ-13
        if self.status_at_instr & 0x80 != 0 {
            self.reset_in_vblank();
            self.status_at_instr &= !0x80;
        }
        self.scroll.reset_latch();
        status
    }

    // $2002 | R   | PPUSTATUS
    //       | 0-5 | Unknown (???)
    //       |   6 | Sprite0 Hit Flag, 1 = PPU rendering has hit sprite #0
    //       |     | This flag resets to 0 when VBlank starts, or CPU reads $2002
    //       |   7 | VBlank Flag, 1 = PPU is generating a Vertical Blanking Impulse
    //       |     | This flag resets to 0 when VBlank ends, or CPU reads $2002
    //
    // Non-mutating version of `read_status`.
    /// Reads $2002 PPUSTATUS without its side effects.
    #[inline(always)]
    pub const fn peek_status(&self) -> u8 {
        // Only upper 3 bits are connected for this register
        (self.read_status_bits() & 0xE0) | (self.open_bus & 0x1F)
    }

    // $2003 | W   | OAMADDR
    //       |     | Used to set the address in the 256-byte Sprite Memory to be
    //       |     | accessed via $2004. This address will increment by 1 after
    //       |     | each access to $2004. The Sprite Memory contains coordinates,
    //       |     | colors, and other attributes of the sprites.
    /// Writes $2003 OAMADDR, the index the next OAM access starts from.
    #[inline(always)]
    pub const fn write_oamaddr(&mut self, val: u8) {
        self.open_bus = val;
        self.oamaddr = val;
    }

    // $2004 | RW  | OAMDATA
    //       |     | Used to read the Sprite Memory. The address is set via
    //       |     | $2003 and increments after each access. The Sprite Memory
    //       |     | contains coordinates, colors, and other attributes of the
    //       |     | sprites.
    /// Reads $2004 OAMDATA at the current OAM address.
    #[inline(always)]
    pub fn read_oamdata(&mut self) -> u8 {
        self.open_bus = self.peek_oamdata();
        self.open_bus
    }

    // $2004 | RW  | OAMDATA
    //       |     | Used to read the Sprite Memory. The address is set via
    //       |     | $2003 and increments after each access. The Sprite Memory
    //       |     | contains coordinates, colors, and other attributes of the
    //       |     | sprites.
    // Non-mutating version of `read_oamdata`.
    /// Reads $2004 OAMDATA without side effects.
    #[inline(always)]
    pub fn peek_oamdata(&self) -> u8 {
        // Reading OAMDATA during rendering will expose OAM accesses during sprite evaluation and loading
        if self.scanline <= scanline::VISIBLE_END
            && self.mask_rendering_enabled
            && cycle::SPR_FETCH_RANGE.contains(&self.cycle)
        {
            self.secondary_oamdata[self.secondary_oamaddr as usize]
        } else {
            self.oamdata[self.oamaddr as usize]
        }
    }

    // $2004 | RW  | OAMDATA
    //       |     | Used to write the Sprite Memory. The address is set via
    //       |     | $2003 and increments after each access. The Sprite Memory
    //       |     | contains coordinates, colors, and other attributes of the
    //       |     | sprites.
    /// Writes $2004 OAMDATA at the current OAM address, incrementing it.
    pub fn write_oamdata(&mut self, mut val: u8) {
        self.open_bus = val;

        if self.mask_rendering_enabled && (self.is_visible_scanline || self.is_prerender_scanline) {
            // https://www.nesdev.org/wiki/PPU_registers#OAMDATA
            // Writes to OAMDATA during rendering do not modify values, but do perform a glitch
            // increment of OAMADDR, bumping only the high 6 bits
            self.oamaddr = self.oamaddr.wrapping_add(4);
        } else {
            if self.oamaddr & 0x03 == 0x02 {
                // Bits 2-4 of sprite attr (byte 2) are unimplemented and always read back as 0
                val &= 0xE3;
            }
            self.oamdata[self.oamaddr as usize] = val;
            self.oamaddr = self.oamaddr.wrapping_add(1);
        }
    }

    // $2005 | W   | PPUSCROLL
    //       |     | There are two scroll registers, vertical and horizontal,
    //       |     | which are both written via this port. The first value written
    //       |     | will go into the Vertical Scroll Register (unless it is >239,
    //       |     | then it will be ignored). The second value will appear in the
    //       |     | Horizontal Scroll Register. The Name Tables are assumed to be
    //       |     | arranged in the following way:
    //       |     |
    //       |     |           +-----------+-----------+
    //       |     |           | 2 ($2800) | 3 ($2C00) |
    //       |     |           +-----------+-----------+
    //       |     |           | 0 ($2000) | 1 ($2400) |
    //       |     |           +-----------+-----------+
    //       |     |
    //       |     | When scrolled, the picture may span over several Name Tables.
    //       |     | Remember, though, that because of the mirroring, there are
    //       |     | only 2 real Name Tables, not 4.
    /// Writes $2005 PPUSCROLL: X then Y, selected by the shared address latch.
    #[inline(always)]
    pub fn write_scroll(&mut self, val: u8) {
        self.open_bus = val;
        self.scroll.write(val);
    }

    // $2006 | W   | PPUADDR
    /// Writes $2006 PPUADDR: high byte then low, selected by the shared address latch.
    #[inline(always)]
    pub fn write_addr(&mut self, val: u8) {
        self.open_bus = val;
        self.scroll.write_addr(val);
    }

    /// Resets the PPU. A soft reset leaves VRAM and palette RAM alone, as the console does.
    pub fn reset(&mut self, kind: ResetKind) {
        self.master_clock = 0;
        self.cycle = 0;
        // FQ-07: FCEUmm's frame starts at its frame boundary on the post-render line (dot 1
        // here, FQ-08), with the PPU a dot ahead of where the CPU's cycle count puts it
        // (measured: starting at dot 2 reproduces FCEUmm's instruction timing frame for
        // frame). After a reset its CPU runs from scanline 240, a whole frame before the PPU
        // completes its first.
        self.cycle = 2;
        self.scanline = self.vblank_scanline - 1;
        self.is_visible_scanline = false;
        self.is_prerender_scanline = false;
        self.is_render_scanline = false;
        self.open_bus = 0x00;

        self.reset_mask(kind);
        self.scroll.reset(kind);
        self.reset_ctrl(kind);

        self.reset_status(kind);
        self.nmi_pending = false;

        self.oam_fetch = 0x00;
        self.oam_eval_done = false;
        self.secondary_oamaddr = 0x0000;
        self.overflow_count = 0;
        self.spr_in_range = false;
        self.spr_zero_in_range = false;
        self.spr_zero_visible = false;
        self.spr_count = 0;
        self.vram_buffer = 0x00;
        self.color_bits_applied = 0;

        if kind == ResetKind::Hard {
            self.oamaddr = 0x0000;
            self.oamdata = ConstArray::new();
        }
        self.sprites = [Sprite::new(); 8];
        self.spr_cover = ConstArray::new();
        self.prevent_vbl = false;
        // FQ-01: FCEUmm's two dead frames after every reset.
        self.dead_frames = 2;
        self.frame_count = 0;
    }
}

/// The PPU's view of the console: everything here reaches the cartridge, so it takes the [`Bus`]
/// rather than the [`Ppu`].
impl Bus {
    /// Read a byte from CHR-ROM/RAM/CIRAM at a given address.
    #[inline(always)]
    pub(crate) fn chr_read(&mut self, addr: u16) -> u8 {
        self.memory.chr_peek(addr)
    }

    /// Peek a byte from CHR-ROM/RAM/CIRAM at a given address.
    #[inline(always)]
    pub fn chr_peek(&self, addr: u16) -> u8 {
        self.memory.chr_peek(addr)
    }

    /// Write a byte to CHR-RAM/CIRAM at a given address.
    #[inline(always)]
    pub(crate) fn chr_write(&mut self, addr: u16, val: u8) {
        self.memory.chr_write(addr, val);
    }

    /// Route a read on the PPU bus: pattern tables and nametables through the board, palette RAM
    /// on the PPU itself.
    ///
    /// Has a side effect - it latches the PPU's open bus. [`Bus::ppu_bus_peek`] observes without
    /// it.
    #[inline]
    pub(crate) fn ppu_bus_read(&mut self, addr: u16) -> u8 {
        self.ppu.open_bus = match addr {
            0x0000..=0x3EFF => self.chr_read(addr),
            0x3F00..=0x3FFF => self.ppu.palette.peek(addr),
            // The PPU address bus is 14 bits wide; nothing above $3FFF reaches here.
            _ => 0x00,
        };
        self.ppu.open_bus
    }

    /// Route a read on the PPU bus with no side effects at all.
    #[inline]
    #[must_use]
    pub fn ppu_bus_peek(&self, addr: u16) -> u8 {
        match addr {
            0x0000..=0x3EFF => self.chr_peek(addr),
            0x3F00..=0x3FFF => self.ppu.palette.peek(addr),
            _ => 0x00,
        }
    }

    /// Route a write on the PPU bus.
    #[inline]
    pub(crate) fn ppu_bus_write(&mut self, addr: u16, val: u8) {
        self.ppu.open_bus = val;
        match addr {
            0x0000..=0x3EFF => self.chr_write(addr, val),
            0x3F00..=0x3FFF => self.ppu.palette.write(addr, val),
            _ => {}
        }
    }

    /// Return the current Nametable mirroring mode.
    #[inline]
    pub fn mirroring(&self) -> Mirroring {
        self.mapper.mirroring()
    }

    /// Fetch BG nametable byte.
    ///
    /// See: <https://wiki.nesdev.org/w/index.php/PPU_scrolling#Tile_and_attribute_fetching>
    #[inline]
    fn fetch_bg_nt_byte(&mut self) {
        let nametable_addr_mask = 0x0FFF; // Only need lower 12 bits
        let addr = addr::NAMETABLE_START | (self.ppu.scroll.addr() & nametable_addr_mask);
        let tile_index = u16::from(self.chr_read(addr));
        self.ppu.tile_addr =
            self.ppu.ctrl_bg_select | (tile_index << 4) | (self.ppu.scroll.v >> 12);
    }

    /// Fetch BG attribute byte.
    ///
    /// See: <https://wiki.nesdev.org/w/index.php/PPU_scrolling#Tile_and_attribute_fetching>
    #[inline(always)]
    fn fetch_bg_attr_byte(&mut self) {
        let addr = self.ppu.scroll.attr_addr();
        let shift = self.ppu.scroll.attr_shift();
        self.ppu.next_palette = ((self.chr_read(addr) >> shift) & 0x03) << 2;
    }

    /// Fetch 4 tiles and write out shift registers every 8th cycle.
    /// Each tile fetch takes 2 cycles.
    ///
    /// See: <https://wiki.nesdev.org/w/index.php/PPU_scrolling#Tile_and_attribute_fetching>
    #[inline(always)]
    fn bg_fetch_cycle(&mut self, cycle: u16) {
        let phase = cycle & 0x07;

        // Uses conditional branches instead of a match, because it compiles to jump table and one
        // indirect jump per rendered dot accounts for half of branch mispredicts. Each test is
        // periodic in the dot counter and is easier to predict.
        if phase & 1 == 0 {
            // Even phases don't fetch. Phase 0 is the start of a new block: increment Coarse X
            // every 8 cycles (e.g. 8 pixels) since sprites are 8x wide.
            if phase == 0 && self.ppu.mask_prev_rendering_enabled {
                self.ppu.scroll.increment_x();
                // Increment Fine Y when we reach the end of the screen
                if cycle == cycle::INC_Y {
                    self.ppu.scroll.increment_y();
                }
            }
        } else if phase < 4 {
            if phase == 1 {
                self.ppu.reload_bg_shifters();
                self.fetch_bg_nt_byte();
            } else {
                self.fetch_bg_attr_byte();
            }
        } else if phase == 5 {
            self.ppu.tile_lo = self.chr_read(self.ppu.tile_addr);
        } else {
            self.ppu.tile_hi = self.chr_read(self.ppu.tile_addr + 8);
        }
    }

    /// Loads one sprite slot of the next scanline (dots 260, 268, ..., 316).
    #[inline(never)]
    fn load_sprites(&mut self, cycle: u16) {
        // Local variables improve cache locality
        let scanline = self.ppu.scanline;
        let spr_count = usize::from(self.ppu.spr_count);

        let idx = (cycle - cycle::SPR_FETCH_START) as usize / 8;
        let oam_idx = idx << 2;

        if let [y, tile_index, attr, x] = self.ppu.secondary_oamdata[oam_idx..=oam_idx + 3] {
            let x = u16::from(x);
            let y = u16::from(y);
            let mut tile_index = u16::from(tile_index);
            let flip_vertical = (attr & 0x80) == 0x80;

            let height = self.ppu.ctrl_spr_height;
            // Should be in the range 0..=7 or 0..=15 depending on sprite height
            let mut line_offset = if (y..y + height).contains(&scanline) {
                scanline - y
            } else {
                0
            };
            if flip_vertical {
                line_offset = height - 1 - line_offset;
            }

            if idx >= spr_count {
                line_offset = 0;
                tile_index = 0xFF;
            }

            let tile_addr = if height == 16 {
                // Use bit 0 of tile index to determine pattern table
                let sprite_select = (tile_index & 0x01) * 0x1000;
                if line_offset >= 8 {
                    line_offset += 8;
                }
                sprite_select | ((tile_index & 0xFE) << 4) | line_offset
            } else {
                self.ppu.ctrl_spr_select | (tile_index << 4) | line_offset
            };

            if idx < spr_count {
                self.ppu.sprites[idx] = Sprite {
                    x,
                    y,
                    tile_addr,
                    tile_lo: self.chr_read(tile_addr),
                    tile_hi: self.chr_read(tile_addr + 8),
                    palette: ((attr & 0x03) << 2) | 0x10,
                    bg_priority: (attr & 0x20) == 0x20,
                    flip_horizontal: (attr & 0x40) == 0x40,
                };
                // Indexed by pixel x (dot - 1): the last pixel, x = 255, is dot 256.
                let x = usize::from(x);
                let bit = 1 << idx;
                for dot in &mut self.ppu.spr_cover[x..(x + 8).min(256)] {
                    *dot |= bit;
                }
            } else {
                // Fetches for remaining sprites/hidden fetch tile $FF
                // Required for accurate MMC3 IRQ
                let _ = self.chr_read(tile_addr);
                let _ = self.chr_read(tile_addr + 8);
            }
        }
    }

    // https://wiki.nesdev.org/w/index.php/PPU_OAM
    #[inline]
    fn spr_fetch_cycle(&mut self) {
        // OAMADDR is held at $00 (an internal reset: the CPU-facing I/O latch is untouched).
        self.ppu.oamaddr = 0x00;

        match self.ppu.cycle & 0x07 {
            // Garbage NT sprite fetch (257, 265, 273, etc.)
            // Required for proper MC-ACC IRQs (MMC3 clone)
            1 => self.fetch_bg_nt_byte(),   // Garbage NT fetch
            3 => self.fetch_bg_attr_byte(), // Garbage attr fetch
            // Cycle 260, 268, etc. This is an approximation (each tile is actually loaded in 8
            // steps (e.g from 257 to 264))
            4 => self.load_sprites(self.ppu.cycle),
            _ => (),
        }
    }

    /// The visible and pre-render scanlines: background/sprite fetches for every dot. This is
    /// the hot path - ~92% of scanlines land here.
    #[inline]
    fn clock_render_scanline(&mut self) {
        if self.ppu.cycle <= cycle::VISIBLE_END {
            if self.ppu.is_visible_scanline {
                self.ppu.spr_eval_cycle();
            }

            self.bg_fetch_cycle(self.ppu.cycle);

            if self.ppu.is_prerender_scanline && self.ppu.cycle <= 8 && self.ppu.oamaddr >= 0x08 {
                // If OAMADDR is not less than eight when rendering starts, the eight bytes
                // starting at OAMADDR & 0xF8 are copied to the first eight bytes of OAM
                let addr = (self.ppu.cycle as usize) - 1;
                let oamindex = (self.ppu.oamaddr as usize & 0xF8) + addr;
                self.ppu.oamdata[addr] = self.ppu.oamdata[oamindex];
            }
        } else if self.ppu.cycle <= cycle::SPR_FETCH_END {
            if self.ppu.mask_prev_rendering_enabled && self.ppu.cycle == cycle::SPR_FETCH_START {
                // Copy X bits at the start of a new line since we're going to start writing
                // new x values to t
                self.ppu.scroll.copy_x();
                self.ppu.spr_cover = ConstArray::new();
            }
            // 280..=304
            if self.ppu.is_prerender_scanline && cycle::COPY_Y_RANGE.contains(&self.ppu.cycle) {
                // Y scroll bits are supposed to be reloaded during this pixel range of PRERENDER
                // if rendering is enabled
                // https://wiki.nesdev.org/w/index.php/PPU_rendering#Pre-render_scanline_.28-1.2C_261.29
                self.ppu.scroll.copy_y();
            }
            self.spr_fetch_cycle();
        } else {
            // 336
            if self.ppu.cycle <= cycle::BG_PREFETCH_END {
                self.bg_fetch_cycle(self.ppu.cycle);
            } else {
                // 337..=340: two dummy NT fetches of two dots each; the reads land on 337 and
                // 339. MMC5 counts consecutive same-address NT reads to detect scanlines, so the
                // read count here is part of the bus contract, not just filler.
                if self.ppu.cycle & 0x01 == 1 {
                    self.fetch_bg_nt_byte();
                }
            }

            self.ppu.oam_fetch = self.ppu.secondary_oamdata[0];

            if self.ppu.is_prerender_scanline
                && self.ppu.cycle == cycle::ODD_SKIP
                && self.ppu.frame_is_odd()
            {
                // NTSC behavior while rendering - each odd PPU frame is one clock shorter
                // (skipping from 339 over 340 to 0)
                self.ppu.cycle = cycle::END;
            }
        }
    }

    /// Advance to the next scanline. Taken once every 341 dots, so kept out of line to keep it
    /// out of the hot path's instruction footprint.
    #[cold]
    #[inline(never)]
    fn end_scanline(&mut self) {
        self.ppu.cycle = 0;
        self.ppu.scanline += 1;
        // === POST-RENDER (240/261) ===
        match self.ppu.scanline {
            s if s == self.ppu.vblank_scanline - 1 => {
                // Every visible scanline is done, and this is where the frame counter advances -
                // so it is the last point before a consumer can ask for the buffer. The mark stays
                // at the end of the buffer through vblank, so a $2001 write there finds nothing
                // outstanding rather than folding the bits into the finished frame a second time.
                let (ppu, frame) = self.ppu_and_frame();
                ppu.apply_color_bits(size::FRAME, frame);
                // FQ-08: FCEUmm's frame ends one dot later (`Bus::fceumm_events`).
            }
            s if s == self.ppu.prerender_scanline => {
                // No sprite evaluation happens on the pre-render line, so its sprite fetches
                // load the dummy $FF tiles and no sprites are drawn on line 0 - not line 239's
                // evaluation for line 240. Also fixes shaking in Ninja Gaiden 3 stage 1 after
                // beating the boss. <https://www.nesdev.org/wiki/PPU_sprite_evaluation>
                self.ppu.spr_count = 0;
            }
            s if s > self.ppu.prerender_scanline => {
                // Wrap scanline back to 0
                self.ppu.scanline = 0;
                self.ppu.color_bits_applied = 0;
            }
            _ => (),
        }

        self.ppu.is_visible_scanline = self.ppu.scanline <= scanline::VISIBLE_END;
        self.ppu.is_prerender_scanline = self.ppu.scanline == self.ppu.prerender_scanline;
        self.ppu.is_render_scanline = self.ppu.is_visible_scanline | self.ppu.is_prerender_scanline;
    }

    /// Clocks the PPU a single dot, whatever state it is in.
    ///
    /// The general form of the dot loop; [`Bus::ppu_clock_to`] runs whole spans of dots through
    /// specialised paths instead and comes here only for dots those do not cover (a $2001 or
    /// $2006 update still settling).
    #[inline(never)]
    pub fn ppu_clock(&mut self) {
        // === SCANLINE TRANSITION (cycle 340) ===
        if self.ppu.cycle >= cycle::END {
            self.end_scanline();
            return;
        }

        self.ppu.cycle += 1;

        // FQ-08, FQ-05: FCEUmm's frame boundary and NMI request.
        if self.ppu.cycle == 1 || self.ppu.cycle == 13 {
            self.fceumm_events();
        }

        // === RENDER LINE (scanlins 0-239, 261) ===
        if self.ppu.mask_rendering_enabled {
            if self.ppu.is_render_scanline {
                self.clock_render_scanline();
            }
        } else if self.ppu.is_prerender_scanline
            && self.ppu.cycle == cycle::ODD_SKIP
            && self.ppu.frame_is_odd()
            && self.ppu.dead_frames == 0
        {
            // FQ-09: FCEUmm shortens every other frame by one dot (its `kook`) whether or not
            // rendering is enabled; hardware only does it while rendering.
            self.ppu.cycle = cycle::END;
        }

        self.ppu.clock_mask();
        self.ppu.scroll.delayed_update();

        // The pixel and the shift registers both want the visible dots, so the range is tested
        // once for the two of them rather than once each.
        if self.ppu.cycle <= cycle::VISIBLE_END {
            // Pixels should be put even if rendering is disabled, as this is what blanks out the
            // screen. Rendering disabled just means we don't evaluate/read bg/sprite info
            if self.ppu.is_visible_scanline {
                let cycle = self.ppu.cycle;
                let (ppu, frame) = self.ppu_and_frame();
                match frame {
                    Some(frame) => ppu.render_pixel(frame, cycle),
                    None => ppu.headless_sprite_zero_hit(cycle),
                }
            }
            self.ppu.tile_shift_lo <<= 1;
            self.ppu.tile_shift_hi <<= 1;
        } else if cycle::BG_PREFETCH_RANGE.contains(&self.ppu.cycle) {
            self.ppu.tile_shift_lo <<= 1;
            self.ppu.tile_shift_hi <<= 1;
        }

        // === VBLANK / IDLE ===
        // Both edges land on dot 1, so that compare goes first and rejects 340 of every 341 dots.
        if self.ppu.cycle == cycle::VBLANK {
            if self.ppu.scanline == self.ppu.vblank_scanline {
                self.ppu.start_vblank();
            } else if self.ppu.is_prerender_scanline {
                self.ppu.stop_vblank();
            }
        }
    }

    /// Events FCEUmm (`fceumm_compat`) places at fixed points of its frame:
    ///
    /// - FQ-08: the frame ends one dot into the post-render line, 341 dots before VBlank rises
    ///   (FQ-01: the dead frames after a reset count down with it).
    /// - FQ-05: 12 dots after VBlank rises the NMI is requested if enabled then, to be taken at
    ///   the next instruction boundary.
    ///
    /// Measured against FCEUmm on the whole corpus with FQ-07's start position: every frame
    /// boundary, NMI entry and VBlank poll lands on the same instruction.
    #[cold]
    #[inline(never)]
    fn fceumm_events(&mut self) {
        let (scanline, cycle) = (self.ppu.scanline, self.ppu.cycle);
        if cycle == 1 && scanline == self.ppu.vblank_scanline - 1 {
            self.ppu.frame_count = self.ppu.frame_count.wrapping_add(1);
            self.ppu.dead_frames = self.ppu.dead_frames.saturating_sub(1);
        } else if cycle == 13
            && scanline == self.ppu.vblank_scanline
            && self.ppu.dead_frames == 0
            && self.ppu.ctrl_nmi_enabled
        {
            self.ppu.fceu_nmi = 1;
        }
    }

    // $2007 | RW  | PPUDATA
    /// Reads $2007 PPUDATA. Everything below the palettes returns the *previous* read's buffered
    /// value, and the address is incremented either way.
    pub fn read_data(&mut self) -> u8 {
        let addr = self.ppu.scroll.addr();
        self.ppu.increment_vram_addr();

        // Buffering quirk resulting in a dummy read for the CPU
        // for reading pre-palette data in $0000 - $3EFF
        let prev_open_bus = self.ppu.open_bus;
        let val = self.ppu_bus_read(addr);
        self.ppu.open_bus = if addr < addr::PALETTE_START {
            let buffer = self.ppu.vram_buffer;
            self.ppu.vram_buffer = val;
            buffer
        } else {
            // Set internal buffer with mirrors of nametable when reading palettes
            // Since we're reading from > $3EFF subtract $1000 to fill
            // buffer with nametable mirror data
            self.ppu.vram_buffer = self.ppu_bus_read(addr - 0x1000);
            // Hi 2 bits of palette should be open bus
            val | (prev_open_bus & 0xC0)
        };


        self.ppu.open_bus
    }

    // $2007 | RW  | PPUDATA
    //
    // Non-mutating version of `read_data`.
    /// Reads $2007 PPUDATA without side effects.
    pub fn peek_data(&self) -> u8 {
        let addr = self.ppu.scroll.addr();
        if addr < addr::PALETTE_START {
            self.ppu.vram_buffer
        } else {
            // Since we're reading from > $3EFF subtract $1000
            // Hi 2 bits of palette should be open bus
            self.ppu_bus_peek(addr - 0x1000) | (self.ppu.open_bus & 0xC0)
        }
    }

    // $2007 | RW  | PPUDATA
    /// Writes $2007 PPUDATA at the current VRAM address, incrementing it.
    pub fn write_data(&mut self, val: u8) {
        let addr = self.ppu.scroll.addr();
        self.ppu.increment_vram_addr();
        self.ppu_bus_write(addr, val);
    }

    // $2000 | RW  | PPUCTRL
    //       | 0-1 | Name Table to show:
    //       |     |
    //       |     |           +-----------+-----------+
    //       |     |           | 2 ($2800) | 3 ($2C00) |
    //       |     |           +-----------+-----------+
    //       |     |           | 0 ($2000) | 1 ($2400) |
    //       |     |           +-----------+-----------+
    //       |     |
    //       |     | Remember, though, that because of the mirroring, there are
    //       |     | only 2 real Name Tables, not 4.
    //       |   2 | Vertical Write, 1 = PPU memory address increments by 32:
    //       |     |
    //       |     |    Name Table, VW=0          Name Table, VW=1
    //       |     |   +----------------+        +----------------+
    //       |     |   |----> write     |        | | write        |
    //       |     |   |                |        | V              |
    //       |     |
    //       |   3 | Sprite Pattern Table address, 1 = $1000, 0 = $0000
    //       |   4 | Screen Pattern Table address, 1 = $1000, 0 = $0000
    //       |   5 | Sprite Size, 1 = 8x16, 0 = 8x8
    //       |   6 | Hit Switch, 1 = generate interrupts on Hit (incorrect ???)
    //       |   7 | VBlank Switch, 1 = generate interrupts on VBlank
    /// Writes $2000 PPUCTRL: nametable select, VRAM increment, pattern table selects, sprite size
    /// and NMI enable.
    pub fn write_ctrl(&mut self, val: u8) {
        self.ppu.open_bus = val;
        let was_enabled = self.ppu.ctrl_nmi_enabled;
        self.ppu.write_ctrl(val);
        self.ppu.scroll.write_nametable_select(val);
        // FQ-04: enabling NMI while VBlank is up requests a delayed NMI; disabling never cancels
        // one already requested.
        if !was_enabled && self.ppu.ctrl_nmi_enabled && self.ppu.status_at_instr & 0x80 != 0 {
            self.ppu.fceu_nmi = 2;
        }
    }

    // $2001 | RW  | PPUMASK
    //       |   0 | Unknown (???)
    //       |   1 | BG Mask, 0 = don't show background in left 8 columns
    //       |   2 | Sprite Mask, 0 = don't show sprites in left 8 columns
    //       |   3 | BG Switch, 1 = show background, 0 = hide background
    //       |   4 | Sprites Switch, 1 = show sprites, 0 = hide sprites
    //       | 5-7 | Unknown (???)
    /// Writes $2001 PPUMASK: greyscale, the two left-column clips, the two render enables and
    /// colour emphasis.
    #[inline(always)]
    pub fn write_mask(&mut self, val: u8) {
        self.ppu.open_bus = val;
        // Settle the pixels drawn under the old greyscale/emphasis before adopting the new ones.
        let (ppu, frame) = self.ppu_and_frame();
        let through = ppu.rendered_through();
        ppu.apply_color_bits(through, frame);
        self.ppu.write_mask(val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cart::Cart, mapper::Sxrom};

    /// A sprite at X=248 covers pixels 248-255; the last one is dot 256 and must be drawn.
    #[test]
    fn sprite_pixels_reach_x_255() {
        let mut bus = Bus::default();
        let mut cart = Cart::empty_sized(0x4000, 0);
        cart.mapper = crate::mapper::Nrom::load(&mut cart).unwrap();
        bus.load_cart(cart);
        bus.reset(ResetKind::Hard);
        for row in 0..16 {
            bus.ppu_bus_write(0x0010 + row, 0xFF);
        }
        bus.ppu_bus_write(0x3F13, 0x30);
        for (i, v) in [100u8, 1, 0, 248].into_iter().enumerate() {
            bus.ppu.oamdata[i] = v;
        }
        for i in 4..256 {
            bus.ppu.oamdata[i] = 0xFF;
        }
        bus.write_mask(0x14);
        for _ in 0..(3 * 341 * 262) {
            bus.ppu_clock();
        }
        let frame = bus.frame();
        let row = &frame[103 * 256..104 * 256];
        let sprite = row[248];
        assert_ne!(sprite, frame[100], "the sprite shows at x=248");
        assert!(
            row[248..256].iter().all(|&p| p == sprite),
            "{:?}",
            &row[244..256]
        );
    }

    /// Sprite evaluation does not run on the pre-render line, so nothing line 239 found for
    /// line 240 may be drawn on line 0. <https://www.nesdev.org/wiki/PPU_sprite_evaluation>
    #[test]
    fn line_239_sprites_are_not_drawn_on_line_0() {
        let mut bus = Bus::default();
        let mut cart = Cart::empty_sized(0x4000, 0);
        cart.mapper = crate::mapper::Nrom::load(&mut cart).unwrap();
        bus.load_cart(cart);
        bus.reset(ResetKind::Hard);
        // Tile 1 is solid colour 3; sprite colour 3 is white.
        for row in 0..16 {
            bus.ppu_bus_write(0x0010 + row, 0xFF);
        }
        bus.ppu_bus_write(0x3F13, 0x30);
        // One sprite at Y=235: rows 236-243, still in range when line 239 evaluates line 240.
        for (i, v) in [235u8, 1, 0, 40].into_iter().enumerate() {
            bus.ppu.oamdata[i] = v;
        }
        for i in 4..256 {
            bus.ppu.oamdata[i] = 0xFF;
        }
        bus.write_mask(0x14);
        for _ in 0..(3 * 341 * 262) {
            bus.ppu_clock();
        }
        let frame = bus.frame();
        let backdrop = frame[100];
        assert_ne!(
            frame[239 * 256 + 44],
            backdrop,
            "the sprite shows on line 239"
        );
        assert!(
            (40..48).all(|x| frame[x] == backdrop),
            "line 0 must show no sprite: {:?}",
            &frame[36..52]
        );
    }

    #[test]
    fn vram_writes() {
        let mut bus = Bus::default();
        bus.ppu.write_addr(0x23);
        bus.ppu.write_addr(0x05);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.write_data(0x66); // write to $2305

        assert_eq!(bus.chr_read(0x2305), 0x66);
    }

    #[test]
    fn vram_reads() {
        let mut bus = Bus::default();
        bus.write_ctrl(0x00);
        bus.ppu_bus_write(0x2305, 0x66);

        bus.ppu.write_addr(0x23);
        bus.ppu.write_addr(0x05);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.read_data(); // buffer read
        assert_eq!(bus.ppu.scroll.addr(), 0x2306);
        assert_eq!(bus.read_data(), 0x66);
        assert_eq!(bus.ppu.scroll.addr(), 0x2307);
    }

    #[test]
    fn vram_read_pagecross() {
        let mut bus = Bus::default();
        bus.write_ctrl(0x00);
        bus.ppu_bus_write(0x21FF, 0x66);
        bus.ppu_bus_write(0x2200, 0x77);

        bus.ppu.write_addr(0x21);
        bus.ppu.write_addr(0xFF);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.read_data(); // buffer read
        assert_eq!(bus.read_data(), 0x66);
        assert_eq!(bus.read_data(), 0x77);
    }

    #[test]
    fn vram_read_vertical_increment() {
        let mut bus = Bus::default();
        bus.write_ctrl(0b100);
        bus.ppu_bus_write(0x21FF, 0x66);
        bus.ppu_bus_write(0x21FF + 32, 0x77);
        bus.ppu_bus_write(0x21FF + 64, 0x88);

        bus.ppu.write_addr(0x21);
        bus.ppu.write_addr(0xFF);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.read_data(); // buffer read
        assert_eq!(bus.read_data(), 0x66);
        assert_eq!(bus.read_data(), 0x77);
        assert_eq!(bus.read_data(), 0x88);
    }

    // Horizontal: https://wiki.nesdev.org/w/index.php/Mirroring
    //   [0x2000 A ] [0x2400 a ]
    //   [0x2800 B ] [0x2C00 b ]
    #[test]
    fn vram_horizontal_mirror() {
        let mut bus = Bus::default();
        bus.ppu.write_addr(0x24);
        bus.ppu.write_addr(0x05);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.write_data(0x66); // write to a at $2405

        bus.ppu.write_addr(0x28);
        bus.ppu.write_addr(0x05);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.write_data(0x77); // write to B at $2805

        bus.ppu.write_addr(0x20);
        bus.ppu.write_addr(0x05);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.read_data(); // buffer read
        assert_eq!(bus.read_data(), 0x66); // read A from $2005

        bus.ppu.write_addr(0x2C);
        bus.ppu.write_addr(0x05);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.read_data(); // buffer read
        assert_eq!(bus.read_data(), 0x77); // read b from $2C05
    }

    // Vertical: https://wiki.nesdev.org/w/index.php/Mirroring
    //   [0x2000 A ] [0x2400 B ]
    //   [0x2800 a ] [0x2C00 b ]
    #[test]
    fn vram_vertical_mirror() {
        let mut bus = Bus::default();
        let mut cart = Cart::default();
        cart.mapper = Sxrom::load(&mut cart).unwrap();
        // Set vertical mirroring mode via 5 writes
        let mut val = 0b00_00_00_01_00;
        for _ in 0..5 {
            cart.mapper
                .write_register(&mut cart.memory, 0x8000, val & 0b11);
            cart.mapper.clock();
            cart.mapper.clock();
            val >>= 2;
        }
        bus.load_cart(cart);

        bus.ppu.write_addr(0x20);
        bus.ppu.write_addr(0x05);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.write_data(0x66); // write to A at $2005

        bus.ppu.write_addr(0x2C);
        bus.ppu.write_addr(0x05);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.write_data(0x77); // write to b at $2C05

        bus.ppu.write_addr(0x28);
        bus.ppu.write_addr(0x05);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.read_data(); // buffer read
        assert_eq!(bus.read_data(), 0x66); // read a from $2805

        bus.ppu.write_addr(0x24);
        bus.ppu.write_addr(0x05);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.read_data(); // buffer read
        assert_eq!(bus.read_data(), 0x77); // read B from $2405
    }

    #[test]
    fn read_status_resets_latch() {
        let mut bus = Bus::default();
        bus.ppu_bus_write(0x2305, 0x66);

        bus.ppu.write_addr(0x21);
        bus.ppu.write_addr(0x23);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.ppu.write_addr(0x05);
        bus.read_data(); // buffer read
        assert_ne!(bus.read_data(), 0x66);

        bus.ppu.read_status();

        bus.ppu.write_addr(0x23);
        bus.ppu.write_addr(0x05);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.read_data(); // buffer read
        assert_eq!(bus.read_data(), 0x66);
    }

    #[test]
    fn vram_mirroring() {
        let mut bus = Bus::default();
        bus.write_ctrl(0);
        bus.ppu_bus_write(0x2305, 0x66);

        bus.ppu.write_addr(0x63); // 0x6305 mirrors to 0x2305
        bus.ppu.write_addr(0x05);
        // PPU writes to $2006 are delayed by 2 PPU clocks
        bus.ppu_clock();
        bus.ppu_clock();
        bus.read_data(); // buffer read
        assert_eq!(bus.ppu.scroll.addr(), 0x2306);
        assert_eq!(bus.read_data(), 0x66);
        assert_eq!(bus.ppu.scroll.addr(), 0x2307);
    }

    #[test]
    #[ignore = "FQ-03: $2002 reads see VBlank as of the instruction start"]
    fn read_status_resets_vblank() {
        let mut bus = Bus::default();
        bus.ppu.set_in_vblank(true);

        let status = bus.ppu.read_status();
        assert_eq!(status >> 7, 1);
        assert_eq!(bus.ppu.read_status_bits() >> 7, 0);
    }

    #[test]
    fn sprite_zero_hit_headless_visible_cycle() {
        let mut bus = Bus::default();
        bus.write_mask(0x18);
        bus.render = false;
        bus.ppu.scanline = 0;
        bus.ppu.cycle = 10;
        bus.ppu.scroll.fine_x = 0;

        bus.ppu.tile_shift_lo = 0x8000;
        bus.ppu.tile_shift_hi = 0x0000;

        bus.ppu.spr_zero_visible = true;
        bus.ppu.spr_cover[8..16].fill(1 << 0);

        bus.ppu.sprites[0].x = 8;
        bus.ppu.sprites[0].tile_lo = 0b0100;
        bus.ppu.sprites[0].tile_hi = 0b0000;
        bus.ppu.sprites[0].flip_horizontal = true;
        bus.ppu.sprites[0].bg_priority = false;

        bus.ppu_clock();

        assert!(bus.ppu.status_spr_zero_hit);
    }

    #[test]
    fn oam_read_write() {
        let mut bus = Bus::default();
        bus.ppu.write_oamaddr(0x10);
        bus.ppu.write_oamdata(0x66);
        bus.ppu.write_oamdata(0x77);

        bus.ppu.write_oamaddr(0x10);
        assert_eq!(bus.ppu.read_oamdata(), 0x66);

        bus.ppu.write_oamaddr(0x11);
        assert_eq!(bus.ppu.read_oamdata(), 0x77);
    }
}
