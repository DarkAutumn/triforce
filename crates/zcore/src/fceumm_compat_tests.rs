//! Tests for `Config::fceumm_compat` (quirks FQ-01..FQ-11). Each asserts the compat-on behaviour and,
//! where it differs, that the default (hardware) behaviour is unchanged.

use crate::{bus::Bus, cart::Cart, common::ResetKind, cpu::IrqFlags, mapper::Nrom};

/// An NROM console spinning on `JMP $0000` in work RAM.
fn console(compat: bool) -> Bus {
    let mut bus = Bus::default();
    let mut cart = Cart::empty();
    cart.mapper = Nrom::load(&mut cart).unwrap();
    bus.load_cart(cart);
    bus.ppu.fceumm_compat = compat;
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
    assert_eq!(console(true).cpu.cycle, 0);
    assert_eq!(console(false).cpu.cycle, 7);
}

#[test]
fn fq07_ppu_starts_on_the_post_render_line() {
    let bus = console(true);
    assert_eq!((bus.ppu.scanline, bus.ppu.cycle), (240, 2));
    let bus = console(false);
    // Hardware: line 0, a few dots in after the 7 reset cycles.
    assert_eq!(bus.ppu.scanline, 0);
    assert!(bus.ppu.cycle < 30, "{}", bus.ppu.cycle);
}

#[test]
fn fq08_frame_ends_one_dot_into_line_240() {
    let mut bus = console(true);
    ppu_dots_to_next_frame(&mut bus);
    assert_eq!((bus.ppu.scanline, bus.ppu.cycle), (240, 1));
    let mut bus = console(false);
    ppu_dots_to_next_frame(&mut bus);
    assert_eq!((bus.ppu.scanline, bus.ppu.cycle), (240, 0));
}

#[test]
fn fq01_no_vblank_in_the_two_frames_after_reset() {
    let mut bus = console(true);
    let mut seen = [false; 3];
    for seen in &mut seen {
        let start = bus.ppu.frame_number();
        while bus.ppu.frame_number() == start {
            bus.ppu_clock();
            *seen |= bus.ppu.status_in_vblank;
        }
    }
    assert_eq!(seen, [false, false, true]);

    let mut bus = console(false);
    let start = bus.ppu.frame_number();
    let mut seen = false;
    while bus.ppu.frame_number() == start {
        bus.ppu_clock();
        seen |= bus.ppu.status_in_vblank;
    }
    // Hardware: line 0 to 240 has no VBlank, but the next frame does.
    assert!(!seen);
    let start = bus.ppu.frame_number();
    while bus.ppu.frame_number() == start {
        bus.ppu_clock();
        seen |= bus.ppu.status_in_vblank;
    }
    assert!(seen);
}

#[test]
fn fq09_odd_frames_drop_a_dot_with_rendering_off() {
    let mut bus = console(true);
    for _ in 0..3 {
        ppu_dots_to_next_frame(&mut bus);
    }
    let lens: Vec<u32> = (0..4).map(|_| ppu_dots_to_next_frame(&mut bus)).collect();
    assert_eq!(lens.iter().sum::<u32>(), 4 * 89_342 - 2, "{lens:?}");
    assert!(lens.contains(&89_341) && lens.contains(&89_342), "{lens:?}");

    let mut bus = console(false);
    ppu_dots_to_next_frame(&mut bus);
    let lens: Vec<u32> = (0..4).map(|_| ppu_dots_to_next_frame(&mut bus)).collect();
    assert_eq!(
        lens,
        vec![89_342; 4],
        "rendering is off: hardware never skips"
    );
}

#[test]
fn fq05_nmi_requested_12_dots_after_vblank() {
    let mut bus = console(true);
    for _ in 0..2 {
        ppu_dots_to_next_frame(&mut bus);
    }
    bus.write_ctrl(0x80);
    while bus.ppu.fceu_nmi == 0 {
        bus.ppu_clock();
    }
    assert_eq!((bus.ppu.scanline, bus.ppu.cycle), (241, 13));
    assert!(!bus.ppu.nmi_pending, "the hardware NMI line is not used");

    // Taken at the end of the instruction during which it was requested.
    let mut bus = console(true);
    for _ in 0..2 {
        let start = bus.ppu.frame_number();
        while bus.ppu.frame_number() == start {
            bus.clock_instr();
        }
    }
    bus.write_ctrl(0x80);
    let sp = bus.cpu.sp;
    let mut steps = 0;
    while bus.cpu.sp == sp {
        let requested_before = bus.ppu.fceu_nmi != 0;
        bus.clock_instr();
        steps += 1;
        assert!(steps < 50_000, "NMI never taken");
        if bus.cpu.sp != sp {
            assert!(!requested_before, "taken in the same step it was requested");
        }
    }
    assert_eq!(bus.cpu.sp, sp.wrapping_sub(3));
}

#[test]
fn fq03_status_read_sees_flags_as_of_the_instruction_start() {
    let mut bus = console(true);
    // Raised during the reading instruction: the read misses it and leaves it set.
    bus.ppu.status_in_vblank = true;
    bus.ppu.status_at_instr = 0;
    assert_eq!(bus.ppu.read_status() & 0x80, 0);
    assert!(bus.ppu.status_in_vblank);
    // Up when the instruction started: seen and cleared.
    bus.ppu.status_at_instr = 0x80;
    assert_eq!(bus.ppu.read_status() & 0x80, 0x80);
    assert!(!bus.ppu.status_in_vblank);
    // Sprite 0 hit cleared at the end of VBlank during the reading instruction: still seen.
    bus.ppu.status_spr_zero_hit = false;
    bus.ppu.status_at_instr = 0x40;
    assert_eq!(bus.ppu.read_status() & 0x40, 0x40);
    // Raised during the reading instruction: seen at once.
    bus.ppu.status_spr_zero_hit = true;
    bus.ppu.status_at_instr = 0;
    assert_eq!(bus.ppu.read_status() & 0x40, 0x40);
}

#[test]
fn fq02_fq04_status_read_neither_suppresses_vblank_nor_cancels_nmi() {
    let mut bus = console(true);
    bus.ppu.scanline = bus.ppu.vblank_scanline;
    bus.ppu.cycle = 0;
    bus.ppu.fceu_nmi = 1;
    bus.ppu.read_status();
    assert!(!bus.ppu.prevent_vbl, "FQ-02: no suppression race");
    assert_eq!(bus.ppu.fceu_nmi, 1, "FQ-04: a read never cancels an NMI");

    let mut bus = console(false);
    bus.ppu.scanline = bus.ppu.vblank_scanline;
    bus.ppu.cycle = 0;
    bus.ppu.read_status();
    assert!(
        bus.ppu.prevent_vbl,
        "hardware: reading on the edge suppresses VBlank"
    );
}

#[test]
fn fq04_enabling_nmi_during_vblank_delays_one_instruction() {
    let mut bus = console(true);
    bus.ppu.status_at_instr = 0x80;
    bus.write_ctrl(0x80);
    assert_eq!(bus.ppu.fceu_nmi, 2);
    let sp = bus.cpu.sp;
    bus.clock_instr();
    assert_eq!(bus.cpu.sp, sp, "first instruction after the write runs");
    bus.clock_instr();
    assert_eq!(bus.cpu.sp, sp.wrapping_sub(3), "then the NMI is taken");

    // Disabling never cancels a requested NMI.
    let mut bus = console(true);
    bus.ppu.fceu_nmi = 1;
    bus.write_ctrl(0x00);
    assert_eq!(bus.ppu.fceu_nmi, 1);
}

#[test]
fn fq10_oam_dma_takes_512_cycles() {
    let mut bus = console(true);
    // STA $4014; NOP at $0010.
    for (addr, val) in [
        (0x0010u16, 0x8Du8),
        (0x0011, 0x14),
        (0x0012, 0x40),
        (0x0013, 0xEA),
    ] {
        bus.cpu_bus_write(addr, val);
    }
    bus.cpu.pc = 0x0010;
    bus.cpu.acc = 0x02;
    let before = bus.cpu.cycle;
    bus.clock_instr();
    assert_eq!(
        bus.cpu.cycle - before,
        4 + 512,
        "the DMA runs inside the STA"
    );
    let before = bus.cpu.cycle;
    bus.clock_instr();
    assert_eq!(bus.cpu.cycle - before, 2);
}

/// NOPs from $0010 on; returns the CPU cycles each instruction step took.
fn nop_steps(bus: &mut Bus, n: usize) -> Vec<u32> {
    for addr in 0x0010u16..0x0800 {
        bus.cpu_bus_write(addr, 0xEA);
    }
    bus.cpu.pc = 0x0010;
    (0..n)
        .map(|_| {
            let before = bus.cpu.cycle;
            bus.clock_instr();
            bus.cpu.cycle - before
        })
        .collect()
}

/// Start a 17-byte DMC sample at the fastest rate.
fn start_dmc(bus: &mut Bus) {
    for (addr, val) in [
        (0x4010u16, 0x0Fu8),
        (0x4012, 0x00),
        (0x4013, 0x01),
        (0x4015, 0x10),
    ] {
        bus.cpu_bus_write(addr, val);
    }
}

#[test]
fn fq11_dmc_fetch_takes_four_cycles() {
    let mut bus = console(true);
    start_dmc(&mut bus);
    let steps = nop_steps(&mut bus, 1);
    assert_eq!(
        steps,
        vec![2 + 4],
        "the first hook after enabling fetches at once"
    );
}

#[test]
fn fq12_dmc_fetches_once_per_eight_timer_periods() {
    let mut bus = console(true);
    start_dmc(&mut bus);
    let steps = nop_steps(&mut bus, 2000);
    let mut at = Vec::new();
    let mut cycle = 0;
    for s in steps {
        if s == 6 {
            at.push(cycle);
        }
        cycle += s;
    }
    // The first fetch is immediate, the second waits for the first byte's eight bits to start
    // shifting (the buffer is only emptied into the shifter), then one byte per 8 * 54 cycles.
    assert!(at.len() >= 4, "{at:?}");
    for w in at.windows(2).skip(1) {
        let gap = w[1] - w[0];
        assert!(
            (8 * 54 - 2..=8 * 54 + 6).contains(&gap),
            "gap {gap} in {at:?}"
        );
    }
    // The hardware fetcher is idle while compat timing is in charge.
    assert!(!bus.cpu.irq_flags.intersects(IrqFlags::DMA_DMC));
}
