//! Tests for the FCEUmm quirks zcore still reproduces (plan step 18a restores them to hardware
//! one at a time). Each asserts the FCEUmm behaviour; the hardware behaviour each replaces is in
//! `FCEUMM_QUIRKS.md`.

use crate::{bus::Bus, cart::Cart, common::ResetKind, mapper::Nrom};

/// An NROM console spinning on `JMP $0000` in work RAM.
fn console() -> Bus {
    let mut bus = Bus::default();
    let mut cart = Cart::empty();
    cart.mapper = Nrom::load(&mut cart).unwrap();
    bus.load_cart(cart);
    bus.reset(ResetKind::Hard);
    for (addr, val) in [(0x0000u16, 0x4Cu8), (0x0001, 0x00), (0x0002, 0x00)] {
        bus.cpu_bus_write(addr, val);
    }
    bus.cpu.pc = 0x0000;
    bus
}

/// Clock the PPU alone until the frame counter changes; returns the dots it took.
fn ppu_dots_to_next_frame(bus: &mut Bus) -> u32 {
    let start = bus.ppu.frame_number();
    let mut dots = 0;
    while bus.ppu.frame_number() == start {
        bus.ppu_clock();
        dots += 1;
    }
    dots
}

#[test]
fn fq06_reset_spends_no_cycles() {
    assert_eq!(console().cpu.cycle, 0);
}

#[test]
fn fq07_ppu_starts_on_the_post_render_line() {
    let bus = console();
    assert_eq!((bus.ppu.scanline, bus.ppu.cycle), (240, 2));
}

#[test]
fn fq08_frame_ends_one_dot_into_line_240() {
    let mut bus = console();
    ppu_dots_to_next_frame(&mut bus);
    assert_eq!((bus.ppu.scanline, bus.ppu.cycle), (240, 1));
}

#[test]
fn fq01_no_vblank_in_the_two_frames_after_reset() {
    let mut bus = console();
    let mut seen = [false; 3];
    for seen in &mut seen {
        let start = bus.ppu.frame_number();
        while bus.ppu.frame_number() == start {
            bus.ppu_clock();
            *seen |= bus.ppu.status_in_vblank;
        }
    }
    assert_eq!(seen, [false, false, true]);
}
