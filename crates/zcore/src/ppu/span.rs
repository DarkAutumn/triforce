//! The PPU's dot loop, run a span of dots at a time.
//!
//! [`Bus::ppu_clock`] clocks one dot in any state. The CPU no longer clocks the PPU every cycle
//! (see [`Bus::ppu_sync`](crate::bus::Bus::ppu_sync)), so a catch-up covers anything from a few
//! dots to a whole frame, and nearly all of them sit in one of two steady states that this runs a
//! scanline segment at a time, taking the per-dot decisions once per span:
//!
//! - rendering a visible or pre-render line ([`Bus::render_span`]);
//! - idling: rendering off, or a post-render or vblank line ([`Bus::idle_span`]).
//!
//! Each leaves exactly the state the same dots through `ppu_clock` would. A dot in any other state
//! (a $2001 enable change or a $2006 address still settling) goes through `ppu_clock`.

use crate::{
    bus::{Bus, FrameBuffer},
    memory::ConstArray,
    ppu::{Ppu, addr, cycle, is_palette, scroll::Scroll},
};

/// Last dot whose background fetches a headless line may skip: from dot 233 on, the last three
/// tiles' fetches set every piece of fetch state the rest of the line leaves behind.
const SKIP_FETCH_END: u16 = 232;

/// How many of the dots `from..=to` fall in `lo..=hi`.
#[inline(always)]
const fn overlap(from: u16, to: u16, lo: u16, hi: u16) -> u16 {
    let a = if from > lo { from } else { lo };
    let b = if to < hi { to } else { hi };
    b.saturating_sub(a) + (a <= b) as u16
}

/// Byte `i` of `SPREAD_BITS[b]` is bit `7 - i` of `b`: a tile row's pattern bits, one per pixel lane.
static SPREAD_BITS: [u64; 256] = {
    let mut table = [0u64; 256];
    let mut b = 0;
    while b < 256 {
        let mut i = 0;
        while i < 8 {
            table[b] |= (((b >> (7 - i)) & 1) as u64) << (8 * i);
            i += 1;
        }
        b += 1;
    }
    table
};

impl Ppu {
    /// Shifts the background tile registers by `n` dots.
    #[inline(always)]
    const fn shift_bg(&mut self, n: u16) {
        if n >= 16 {
            self.tile_shift_lo = 0;
            self.tile_shift_hi = 0;
        } else {
            self.tile_shift_lo <<= n;
            self.tile_shift_hi <<= n;
        }
    }

    /// The colour [`Ppu::render_pixel`] gives each pixel palette index with rendering on: the
    /// backdrop for every transparent index, the palette entry otherwise. Constant over a span:
    /// palette RAM changes only through a $2007 write, which ends it.
    #[inline(always)]
    fn resolved_palette(&self) -> [u16; 32] {
        std::array::from_fn(|p| {
            let p = p as u16;
            u16::from(self.palette.peek(addr::PALETTE_START | ((p & 0x03 > 0) as u16 * p)))
        })
    }

    /// The 8 pixels of a whole tile (dots `dot..dot + 8`, `dot` ≡ 1 mod 8) on which no sprite
    /// lies, as [`Ppu::render_pixel`] draws them with rendering on, then the 8 shifts. The
    /// palette indexes are built in the 8 byte lanes of a `u64`, pixel `i` in lane `i`; lanes
    /// never carry (a lane is at most `0x1F + 3`).
    #[inline(always)]
    fn render_bg_tile(&mut self, frame: &mut FrameBuffer, dot: u16, colors: &[u16; 32]) {
        const LANES: u64 = 0x0101_0101_0101_0101;
        let fine_x = self.scroll.fine_x;
        // Pixel i takes bit 15 - fine_x - i of the shift registers: bit 7 - i of these.
        let lo = usize::from(((self.tile_shift_lo << fine_x) >> 8) as u8);
        let hi = usize::from(((self.tile_shift_hi << fine_x) >> 8) as u8);
        let mut bg = SPREAD_BITS[lo] | (SPREAD_BITS[hi] << 1);
        // Pixels at or left of `mask_min_draw_bg_cycle` draw no background.
        let clipped = (self.mask_min_draw_bg_cycle + 1).saturating_sub(dot);
        if clipped >= 8 {
            bg = 0;
        } else {
            bg &= u64::MAX << (8 * clipped);
        }
        // The first 8 - fine_x pixels still come from the previous tile's palette.
        let prev_lanes = 8 - fine_x;
        let prev_mask = if prev_lanes >= 8 { u64::MAX } else { (1u64 << (8 * prev_lanes)) - 1 };
        let prev = u64::from(self.prev_palette & 0x1F) * LANES;
        let curr = u64::from(self.curr_palette & 0x1F) * LANES;
        let index = ((prev & prev_mask) | (curr & !prev_mask)) + bg;

        let row = (usize::from(self.scanline) << 8) + usize::from(dot - 1);
        let out: &mut [u16; 8] = (&mut frame[row..row + 8]).try_into().expect("8 dots");
        for (i, pixel) in out.iter_mut().enumerate() {
            *pixel = colors[(index >> (8 * i)) as usize & 0x1F];
        }
        self.shift_bg(8);
    }

    /// Sprite evaluation over the visible-line dots `from..=to` (within 1-256), as
    /// [`Ppu::spr_eval_cycle`] runs it a dot at a time.
    #[inline]
    fn spr_eval_dots(&mut self, from: u16, to: u16) {
        let mut dot = from;
        if dot <= cycle::OAM_CLEAR_END {
            self.oam_fetch = 0xFF;
            if to >= cycle::OAM_CLEAR_END {
                self.secondary_oamdata = ConstArray::filled(0xFF);
            }
            dot = cycle::SPR_EVAL_START;
        }
        if dot > to {
            return;
        }
        if dot == cycle::SPR_EVAL_START {
            self.spr_in_range = false;
            self.spr_zero_in_range = false;
            self.secondary_oamaddr = 0x00;
            self.oam_eval_done = false;
            self.oamaddr_hi = (self.oamaddr >> 2) & 0x3F;
            self.oamaddr_lo = self.oamaddr & 0x03;
        }
        let end = to.min(cycle::SPR_EVAL_END0);
        if dot <= end {
            self.oam_eval_dots(dot, end);
        }
        if to == cycle::SPR_EVAL_END {
            self.spr_zero_visible = self.spr_zero_in_range;
            self.spr_count = self.secondary_oamaddr >> 2;
            self.oam_eval_dots(cycle::SPR_EVAL_END, cycle::SPR_EVAL_END);
        }
    }
}

/// Steps coarse X `n` times (`n` below 32), as `n` calls of [`Scroll::increment_x`] would.
#[inline(always)]
const fn increment_x_by(scroll: &mut Scroll, n: u16) {
    let x = (scroll.v & Scroll::COARSE_X_MASK) + n;
    let mut v = (scroll.v & !Scroll::COARSE_X_MASK) | (x & Scroll::COARSE_X_MASK);
    if x > Scroll::COARSE_X_MASK {
        v ^= Scroll::NT_X_MASK;
    }
    scroll.v = v;
}

impl Bus {
    /// A whole tile's four fetches (dots ≡ 1, 3, 5, 7 mod 8), after the reload on its first dot.
    #[inline(always)]
    fn fetch_tile(&mut self) {
        self.ppu.reload_bg_shifters();
        self.fetch_bg_nt_byte();
        self.fetch_bg_attr_byte();
        self.ppu.tile_lo = self.chr_read(self.ppu.tile_addr);
        self.ppu.tile_hi = self.chr_read(self.ppu.tile_addr + 8);
    }

    /// Clocks the PPU forward until it catches up to `clock` master cycles.
    #[inline]
    pub fn ppu_clock_to(&mut self, clock: u32) {
        let divider = u32::from(self.ppu.clock_divider);
        let start = self.ppu.master_clock;
        if start + divider > clock {
            return;
        }
        let dots = (clock - start) / divider;
        self.ppu.master_clock = start + dots * divider;
        self.ppu_run(dots);
    }

    /// Runs `dots` PPU clocks.
    #[inline(never)]
    fn ppu_run(&mut self, mut dots: u32) {
        while dots > 0 {
            let ppu = &self.ppu;
            if ppu.cycle >= cycle::END {
                self.end_scanline();
                dots -= 1;
                continue;
            }
            if ppu.mask_pending_rendering_update
                || ppu.mask_prev_rendering_enabled != ppu.mask_rendering_enabled
                || ppu.scroll.delay_v_cycles != 0
            {
                self.ppu_clock();
                dots -= 1;
                continue;
            }
            let span = dots.min(u32::from(cycle::END - ppu.cycle)) as u16;
            let ran = if ppu.mask_rendering_enabled && ppu.is_render_scanline {
                self.render_span(span)
            } else {
                self.idle_span(span)
            };
            dots -= u32::from(ran);
        }
    }

    /// Up to `span` dots of a line on which nothing is fetched (rendering off, or a line outside
    /// 0-239 and 261), stopping at dot 340. Returns how many clocks that took.
    #[inline(never)]
    fn idle_span(&mut self, span: u16) -> u16 {
        let from = self.ppu.cycle + 1;
        let to = self.ppu.cycle + span;

        // The vblank edges. Nothing else here reads what they change.
        if from == cycle::VBLANK {
            self.ppu.cycle = cycle::VBLANK;
            if self.ppu.scanline == self.ppu.vblank_scanline {
                self.ppu.start_vblank();
            } else if self.ppu.is_prerender_scanline {
                self.ppu.stop_vblank();
            }
        }

        // Pixels: with rendering off a visible line shows the backdrop, or the palette entry `v`
        // points at (`Ppu::render_pixel`). Headless, no sprite 0 hit is possible with the
        // background off.
        if self.ppu.is_visible_scanline && from <= cycle::VISIBLE_END {
            let (ppu, frame) = self.ppu_and_frame();
            if let Some(frame) = frame {
                let v = ppu.scroll.addr();
                let color = if is_palette(v) {
                    ppu.palette.peek(v)
                } else {
                    ppu.palette.peek(addr::PALETTE_START)
                };
                let row = usize::from(ppu.scanline) << 8;
                let end = to.min(cycle::VISIBLE_END);
                frame[row + usize::from(from - 1)..row + usize::from(end)].fill(u16::from(color));
            }
        }

        let shifts = overlap(from, to, cycle::VISIBLE_START, cycle::VISIBLE_END)
            + overlap(from, to, cycle::BG_PREFETCH_START, cycle::BG_PREFETCH_END);
        self.ppu.shift_bg(shifts);

        self.ppu.cycle = to;
        span
    }

    /// Up to `span` dots of a visible or pre-render line with rendering on and settled,
    /// stopping at dot 340. Returns how many clocks that took.
    #[inline(never)]
    fn render_span(&mut self, span: u16) -> u16 {
        let from = self.ppu.cycle + 1;
        let to = self.ppu.cycle + span;
        let mut dot = from;

        if dot <= cycle::VISIBLE_END {
            let end = to.min(cycle::VISIBLE_END);
            self.render_visible_dots(dot, end);
            dot = end + 1;
        }
        if dot <= to && dot <= cycle::SPR_FETCH_END {
            let end = to.min(cycle::SPR_FETCH_END);
            self.sprite_fetch_dots(dot, end);
            dot = end + 1;
        }
        while dot <= to && dot <= cycle::BG_PREFETCH_END {
            if dot & 0x07 == 1 && dot + 7 <= to {
                self.fetch_tile();
                self.ppu.shift_bg(8);
                self.ppu.scroll.increment_x();
                dot += 8;
            } else {
                self.bg_fetch_cycle(dot);
                self.ppu.shift_bg(1);
                dot += 1;
            }
            self.ppu.oam_fetch = self.ppu.secondary_oamdata[0];
        }
        while dot <= to {
            // 337..=340: the dummy nametable fetches land on 337 and 339.
            if dot & 0x01 == 1 {
                self.fetch_bg_nt_byte();
            }
            self.ppu.oam_fetch = self.ppu.secondary_oamdata[0];
            if dot == cycle::ODD_SKIP && self.ppu.is_prerender_scanline && self.ppu.frame_is_odd()
            {
                // Odd frames are a dot shorter while rendering.
                self.ppu.cycle = cycle::END;
                return dot - from + 1;
            }
            dot += 1;
        }

        self.ppu.cycle = to;
        span
    }

    /// Dots `from..=to` (within 1-256) of a rendering line: background fetches and pixels, then
    /// sprite evaluation (the two share no state but `spr_zero_visible`, which evaluation sets on
    /// dot 256, where no sprite 0 hit can land).
    #[inline(always)]
    fn render_visible_dots(&mut self, from: u16, to: u16) {
        let visible = self.ppu.is_visible_scanline;
        let mut dot = from;

        // Nothing drawn and no sprite 0 hit possible: up to the last three tiles, whose fetches
        // overwrite every bit of fetch state the line leaves (tile address and bytes, attribute
        // latches, shift registers), only `v`'s coarse X steps matter.
        let draws = visible && (self.render || self.ppu.spr_zero_visible);
        if !draws && to == cycle::VISIBLE_END && dot <= SKIP_FETCH_END {
            let steps = SKIP_FETCH_END / 8 - (dot - 1) / 8;
            increment_x_by(&mut self.ppu.scroll, steps);
            dot = SKIP_FETCH_END + 1;
        }

        let mut colors = None;
        while dot <= to {
            if dot & 0x07 == 1 && dot + 7 <= to {
                // A whole tile: its fetches only feed the next tile, so they go first.
                self.fetch_tile();
                if visible {
                    let (ppu, frame) = self.ppu_and_frame();
                    match frame {
                        Some(frame) => {
                            let x = usize::from(dot - 1);
                            let cover: [u8; 8] =
                                ppu.spr_cover[x..x + 8].try_into().expect("8 dots");
                            if u64::from_ne_bytes(cover) == 0 {
                                let colors = colors.get_or_insert_with(|| ppu.resolved_palette());
                                ppu.render_bg_tile(frame, dot, colors);
                            } else {
                                for d in dot..dot + 8 {
                                    ppu.render_pixel(frame, d);
                                    ppu.shift_bg(1);
                                }
                            }
                        }
                        None if ppu.spr_zero_visible && !ppu.status_spr_zero_hit => {
                            for d in dot..dot + 8 {
                                ppu.headless_sprite_zero_hit(d);
                                ppu.shift_bg(1);
                            }
                        }
                        None => ppu.shift_bg(8),
                    }
                } else {
                    self.ppu.shift_bg(8);
                }
                self.ppu.scroll.increment_x();
                if dot + 7 == cycle::INC_Y {
                    self.ppu.scroll.increment_y();
                }
                dot += 8;
            } else {
                self.bg_fetch_cycle(dot);
                if visible {
                    let (ppu, frame) = self.ppu_and_frame();
                    match frame {
                        Some(frame) => ppu.render_pixel(frame, dot),
                        None => ppu.headless_sprite_zero_hit(dot),
                    }
                }
                self.ppu.shift_bg(1);
                dot += 1;
            }
        }

        let ppu = &mut self.ppu;
        if visible {
            ppu.spr_eval_dots(from, to);
        } else {
            // Pre-render line.
            if from <= 8 && ppu.oamaddr >= 0x08 {
                // OAMADDR at 8 or more when rendering starts copies the eight bytes at
                // OAMADDR & 0xF8 over the first eight.
                for dot in from..=to.min(8) {
                    let addr = usize::from(dot) - 1;
                    let oamindex = (usize::from(ppu.oamaddr) & 0xF8) + addr;
                    ppu.oamdata[addr] = ppu.oamdata[oamindex];
                    ppu.spr_range.valid = false;
                }
            }
            if from == cycle::VBLANK {
                ppu.stop_vblank();
            }
        }
    }

    /// Dots `from..=to` (within 257-320) of a rendering line: sprite fetches.
    #[inline(always)]
    fn sprite_fetch_dots(&mut self, from: u16, to: u16) {
        if from == cycle::SPR_FETCH_START {
            self.ppu.scroll.copy_x();
            self.ppu.spr_cover = ConstArray::new();
        }
        // OAMADDR is held at $00 throughout (an internal reset: the I/O latch is untouched).
        self.ppu.oamaddr = 0x00;
        if self.ppu.is_prerender_scanline {
            // The pre-render line reloads Y from `t` part way through (280-304).
            for dot in from..=to {
                if cycle::COPY_Y_RANGE.contains(&dot) {
                    self.ppu.scroll.copy_y();
                }
                match dot & 0x07 {
                    1 => self.fetch_bg_nt_byte(),
                    3 => self.fetch_bg_attr_byte(),
                    4 => self.load_sprites(dot),
                    _ => (),
                }
            }
            return;
        }
        // `v` holds still here, so of the garbage nametable and attribute fetches only the last
        // of each leaves anything behind; and a slot past `spr_count` loads nothing (its fetches
        // are the dummy tile $FF).
        let first = |phase: u16| from + (phase.wrapping_sub(from) & 0x07);
        let live = cycle::SPR_FETCH_START + 8 * u16::from(self.ppu.spr_count);
        let mut dot = first(4);
        while dot <= to && dot < live {
            self.load_sprites(dot);
            dot += 8;
        }
        if first(1) <= to {
            self.fetch_bg_nt_byte();
        }
        if first(3) <= to {
            self.fetch_bg_attr_byte();
        }
    }
}
