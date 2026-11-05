//! 6502 Central Processing Unit (CPU) implementation.
//!
//! <https://wiki.nesdev.org/w/index.php/CPU>
//!
//! [`Cpu`] is the register file and cycle counters - the state a 6502 keeps. Everything it *does*
//! is an inherent method on [`Bus`], since a memory access moves the whole console: reading a byte
//! clocks the PPU, the APU and the board on the way past. Those `impl Bus` blocks are here and in
//! [`instr`].
//!
//! # Stability
//!
//! [`Cpu`]'s fields are the emulation's internal wiring - registers and cycle counters. They are
//! public so that tools can digest them, but they track the implementation, not a stable
//! interface; the stable entry point is [`ZCore`](crate::ZCore).

use crate::{
    bus::Bus,
    common::ResetKind,
    cpu::instr::{AddrMode, InstrRef},
};
use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::fmt::{self};

pub mod instr;

/// The CPU at an instruction boundary, before the opcode fetch: what a trace hook is handed.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TraceRow {
    /// Address of the instruction.
    pub pc: u16,
    /// Its opcode.
    pub opcode: u8,
    /// Accumulator.
    pub a: u8,
    /// X index.
    pub x: u8,
    /// Y index.
    pub y: u8,
    /// Status flags.
    pub p: u8,
    /// Stack pointer.
    pub sp: u8,
    /// CPU cycle counter (32-bit, wrapping).
    pub cycle: u32,
    /// PPU scanline.
    pub scanline: u16,
    /// PPU dot within the scanline.
    pub dot: u16,
}

/// A per-instruction callback (see [`ZCore::set_trace`](crate::ZCore::set_trace)).
pub type TraceHook = Box<dyn FnMut(&TraceRow) + Send + Sync>;

bitflags! {
    /// Interrupt and DMA state, kept as one byte because it is tested on every cycle.
    ///
    /// The `PREV_` flags are the previous cycle's values, which is what makes an interrupt take
    /// effect one instruction late the way the hardware does.
    #[derive(Default, Serialize, Deserialize, Debug, Copy, Clone)]
    #[must_use]
    pub struct IrqFlags: u8 {
        /// NMI line asserted.
        const NMI = 1 << 0;
        /// NMI line as of the previous cycle.
        const PREV_NMI = 1 << 1;
        /// NMI was pending as of the previous cycle.
        const PREV_NMI_PENDING = 1 << 2;
        /// An IRQ will be serviced after the current instruction.
        const RUN_IRQ = 1 << 3;
        /// `RUN_IRQ` as of the previous cycle.
        const PREV_RUN_IRQ = 1 << 4;
        /// A DMC sample fetch is stealing cycles.
        const DMA_DMC = 1 << 5;
        /// The CPU is halted for a DMA.
        const DMA_HALT = 1 << 6;
        /// The DMA's alignment dummy read.
        const DMA_DUMMY_READ = 1 << 7;
    }
}

// Status Registers
// https://wiki.nesdev.org/w/index.php/Status_flags
// 7654 3210
// NVUB DIZC
// |||| ||||
// |||| |||+- Carry
// |||| ||+-- Zero
// |||| |+--- Interrupt Disable
// |||| +---- Decimal Mode - Not used in the NES but still has to function
// |||+------ Break - 1 when pushed to stack from PHP/BRK, 0 from IRQ/NMI
// ||+------- Unused - always set to 1 when pushed to stack
// |+-------- Overflow
// +--------- Negative
bitflags! {
    /// CPU Status Registers.
    #[derive(Default, Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq)]
    #[must_use]
    pub struct Status: u8 {
        /// Carry.
        const C = 1;
        /// Zero.
        const Z = 1 << 1;
        /// Disable interrupt.
        const I = 1 << 2;
        /// Decimal mode; unused by the NES, but the flag itself still works.
        const D = 1 << 3;
        /// Break: set when pushed by PHP/BRK, clear when pushed by IRQ/NMI.
        const B = 1 << 4;
        /// Unused; always set when pushed to the stack.
        const U = 1 << 5;
        /// Overflow.
        const V = 1 << 6;
        /// Negative.
        const N = 1 << 7;
    }
}
/// The Central Processing Unit status and registers
#[derive(Default, Copy, Clone, Serialize, Deserialize)]
#[must_use]
#[repr(C)]
pub struct Cpu {
    /// Total CPU cycles run.
    pub cycle: u32,
    /// Master clock cycles run, which is what the PPU and APU are caught up against.
    pub master_clock: u32,
    /// Program counter.
    pub pc: u16,
    /// The current instruction's operand.
    pub operand: u16,
    /// Master cycles before a read or write within the current instruction.
    pub start_cycles: u8,
    /// Master cycles after a read or write within the current instruction.
    pub end_cycles: u8,
    /// The current instruction's addressing mode.
    pub addr_mode: AddrMode,
    /// Stack pointer; the stack itself is at $0100-$01FF.
    pub sp: u8,
    /// Accumulator.
    pub acc: u8,
    /// X index register.
    pub x: u8,
    /// Y index register.
    pub y: u8,
    /// Status register.
    pub status: Status,
    /// Interrupt and DMA state.
    pub irq_flags: IrqFlags,
    /// Source page of an OAM DMA in progress.
    pub dma_oam_addr: Option<u16>,
    /// Set when an invalid opcode has jammed the CPU; it keeps cycling but stops fetching.
    pub corrupted: bool,
}

impl Cpu {
    /// NTSC Master Clock Rate in Hz.
    pub const NTSC_MASTER_CLOCK_RATE: f32 = 21_477_272.0;
    /// NTSC CPU Clock Rate in Hz.
    pub const NTSC_CPU_CLOCK_RATE: f32 = Self::NTSC_MASTER_CLOCK_RATE / 12.0;

    // Represents CPU/PPU alignment and would range from 1..=ppu.clock_divider-1
    // if random PPU alignment was emulated
    // See: https://www.nesdev.org/wiki/PPU_frame_timing#CPU-PPU_Clock_Alignment
    pub(crate) const PPU_OFFSET: u32 = 1;

    /// NMI vector Address.
    pub const NMI_VECTOR: u16 = 0xFFFA;
    /// IRQ vector address
    pub const IRQ_VECTOR: u16 = 0xFFFE;
    /// Reset vector address.
    pub const RESET_VECTOR: u16 = 0xFFFC;
    /// Power on status register values.
    pub const POWER_ON_STATUS: Status = Status::U.union(Status::I);
    /// Power on stack pointer value.
    pub const POWER_ON_SP: u8 = 0xFD;
    /// Stack Pointer base address.
    pub const SP_BASE: u16 = 0x0100;

    /// `BRK` opcode, takes the IRQ vector.
    pub const BRK: u8 = 0x00;
    /// `JSR` opcode, jumps to location, save return address.
    pub const JSR: u8 = 0x20;
    /// `RTI` opcode, return from interrupt.
    pub const RTI: u8 = 0x40;
    /// `RTS` opcode, return from subroutine.
    pub const RTS: u8 = 0x60;

    /// Create a new CPU.
    pub fn new() -> Self {
        Self {
            cycle: 0,
            master_clock: 0,
            // NTSC_MASTER_CLOCK_DIVIDER / 2
            start_cycles: 6,
            end_cycles: 6,
            pc: 0x0000,
            operand: 0,
            addr_mode: AddrMode::default(),
            sp: 0x00,
            acc: 0x00,
            x: 0x00,
            y: 0x00,
            status: Self::POWER_ON_STATUS,
            irq_flags: IrqFlags::default(),
            dma_oam_addr: None,
            corrupted: false,
        }
    }

    /// Resets the registers.
    ///
    /// Updates the SP and Status values to defined constants. [`Bus::reset`] is what resets the
    /// rest of the console, fetches the reset vector into PC, and runs the seven cycles the reset
    /// itself takes.
    pub fn reset(&mut self, kind: ResetKind) {
        match kind {
            ResetKind::Soft => {
                self.status.set(Status::I, true);
                // Reset pushes to the stack similar to IRQ, but since the read bit is set, nothing is
                // written except the SP being decremented
                self.sp = self.sp.wrapping_sub(0x03);
            }
            ResetKind::Hard => {
                self.acc = 0x00;
                self.x = 0x00;
                self.y = 0x00;
                self.status = Self::POWER_ON_STATUS;
                self.sp = Self::POWER_ON_SP;
            }
        }

        self.cycle = 0;
        self.master_clock = 0;
        self.irq_flags = IrqFlags::default();
        self.corrupted = false;
    }

    /// Start OAM DMA.
    #[inline]
    pub const fn start_oam_dma(&mut self, addr: u16) {
        self.irq_flags = self.irq_flags.union(IrqFlags::DMA_HALT);
        self.dma_oam_addr = Some(addr);
    }

    // Interrupt flag functions

    /// Clear [`IrqFlags`] flags for the given bits.
    #[inline(always)]
    pub(crate) const fn clear_irq_flags(&mut self, flags: IrqFlags) {
        self.irq_flags = self.irq_flags.difference(flags);
    }

    /// Returns `true` if the [`IrqFlags`] register is set.
    #[inline(always)]
    pub(crate) const fn irq_flags(&self, flags: IrqFlags) -> bool {
        self.irq_flags.intersection(flags).bits() == flags.bits()
    }

    // Status Register functions

    /// Set [`Status`] flags for the given bits.
    #[inline(always)]
    pub(crate) const fn set_status(&mut self, status: Status) {
        self.status = status.difference(Status::U).difference(Status::B);
    }

    /// Returns the [`Status`] register as a byte.
    #[inline(always)]
    pub(crate) const fn status_bit(&self, reg: Status) -> u8 {
        self.status.intersection(reg).bits()
    }

    /// Set accumulator and update [`Status`] flags based on value.
    #[inline(always)]
    pub(crate) fn set_acc(&mut self, val: u8) {
        self.set_zn_status(val);
        self.acc = val;
    }

    /// Set x and update [`Status`] flags based on value.
    #[inline(always)]
    pub(crate) fn set_x(&mut self, val: u8) {
        self.set_zn_status(val);
        self.x = val;
    }

    /// Set y and update [`Status`] flags based on value.
    #[inline(always)]
    pub(crate) fn set_y(&mut self, val: u8) {
        self.set_zn_status(val);
        self.y = val;
    }

    /// Set stack pointer.
    #[inline(always)]
    pub(crate) const fn set_sp(&mut self, val: u8) {
        self.sp = val;
    }

    /// Set both [`Status::Z`] and [`Status::N`] flags based on value.
    #[inline(always)]
    pub(crate) fn set_zn_status(&mut self, val: u8) {
        self.status.set(Status::Z, val == 0x00);
        self.status.set(Status::N, val & 0x80 > 0);
    }

    // Utilities

    /// Returns whether two addresses are on different memory pages.
    #[inline(always)]
    #[must_use]
    pub(crate) const fn pages_differ(addr1: u16, addr2: u16) -> bool {
        (addr1 & 0xFF00) != (addr2 & 0xFF00)
    }

    /// Returns whether a memory page is crossed using relative address.
    #[inline(always)]
    #[must_use]
    pub(crate) const fn page_crossed(addr: u16, offset: i16) -> bool {
        ((addr as i16 + offset) as u16 & 0xFF00) != (addr & 0xFF00)
    }
}

/// The CPU's view of the console: every access below moves the whole machine, so it takes the
/// [`Bus`] rather than the [`Cpu`].
impl Bus {
    /// NTSC CPU clock rate.
    #[inline]
    #[must_use]
    pub const fn clock_rate(&self) -> f32 {
        Cpu::NTSC_CPU_CLOCK_RATE
    }

    /// Peek at the next instruction.
    #[inline]
    pub fn next_instr(&self) -> InstrRef {
        let opcode = self.peek(self.cpu.pc);
        Cpu::INSTR_REF[usize::from(opcode)]
    }

    /// Process an interrupted request.
    ///
    /// <https://wiki.nesdev.org/w/index.php/IRQ>
    ///  #  address R/W description
    /// --- ------- --- -----------------------------------------------
    ///  1    PC     R  fetch PCH
    ///  2    PC     R  fetch PCL
    ///  3  $0100,S  W  push PCH to stack, decrement S
    ///  4  $0100,S  W  push PCL to stack, decrement S
    ///  5  $0100,S  W  push P to stack, decrement S
    ///  6    PC     R  fetch low byte of interrupt vector
    ///  7    PC     R  fetch high byte of interrupt vector
    #[cold]
    #[inline(never)]
    pub fn irq(&mut self) {
        self.read(self.cpu.pc); // Dummy read
        self.read(self.cpu.pc); // Dummy read
        self.push_word(self.cpu.pc);

        // Pushing status to the stack has to happen after checking NMI since it can hijack the BRK
        // IRQ when it occurs between cycles 4 and 5.
        // https://www.nesdev.org/wiki/CPU_interrupts#Interrupt_hijacking
        //
        // Set U and !B during push
        let status = ((self.cpu.status | Status::U) & !Status::B).bits();
        let nmi = self.cpu.irq_flags(IrqFlags::NMI);
        self.push_byte(status);
        self.cpu.status.set(Status::I, true);

        if nmi {
            self.cpu.clear_irq_flags(IrqFlags::NMI);
            self.cpu.pc = self.read_word(Cpu::NMI_VECTOR);
            self.clock_sync();
        } else {
            self.cpu.pc = self.read_word(Cpu::IRQ_VECTOR);
        }
    }

    /// Handle CPU interrupt requests, if any are pending.
    ///
    /// <https://www.nesdev.org/wiki/CPU_interrupts>
    ///
    /// - `PREV_NMI` takes `NMI`: the internal signal goes high during φ1 of the cycle that follows
    ///   the one where the edge is detected, and stays high until the NMI has been handled.
    /// - The edge detector polls the NMI line during φ2 of each cycle (hence here in `end_cycle`)
    ///   and raises `NMI` when it goes from low on the previous cycle to high on this one.
    /// - The IRQ status at the end of the second-to-last cycle is what matters, so `PREV_RUN_IRQ`
    ///   keeps the previous `RUN_IRQ`.
    ///
    /// Done on the flag byte's bits directly: it runs on every CPU cycle.
    #[inline(always)]
    fn handle_interrupts(&mut self) {
        let nmi_pending = u8::from(self.ppu.nmi_pending);
        let run_irq = u8::from(self.apu.irq_pending() & !self.cpu.status.intersects(Status::I));
        let f = self.cpu.irq_flags.bits();
        let nmi = f & IrqFlags::NMI.bits();
        let prev_nmi_pending = (f >> 2) & 1;
        let nmi_edge = !prev_nmi_pending & nmi_pending;
        let kept = IrqFlags::DMA_DMC | IrqFlags::DMA_HALT | IrqFlags::DMA_DUMMY_READ;
        let f = (f & kept.bits())
            | nmi
            | nmi_edge
            | (nmi << 1)
            | (nmi_pending << 2)
            | (run_irq << 3)
            | ((f & IrqFlags::RUN_IRQ.bits()) << 1);
        self.cpu.irq_flags = IrqFlags::from_bits_retain(f);
    }

    /// Start a CPU cycle.
    ///
    /// The PPU is not clocked here: it runs behind and is caught up by [`Bus::ppu_sync`] when
    /// something can observe it. That lands it on exactly the dots per-cycle clocking would have.
    #[inline(always)]
    pub(crate) fn start_cycle(&mut self, increment: u8) {
        self.cpu.master_clock = self.cpu.master_clock.wrapping_add(u32::from(increment));
        self.cpu.cycle = self.cpu.cycle.wrapping_add(1);
        self.cpu_clock();
    }

    /// End a CPU cycle.
    ///
    /// The NMI edge detector polls /NMI here, so the PPU is caught up first if a VBlank edge (or
    /// the frame boundary) may have come during the cycle.
    #[inline(always)]
    pub(crate) fn end_cycle(&mut self, increment: u8) {
        self.cpu.master_clock = self.cpu.master_clock.wrapping_add(u32::from(increment));
        if self.cpu.master_clock >= self.ppu_deadline {
            self.ppu_sync();
        }
        if !self.irq_quiet {
            self.handle_interrupts();
        }
    }

    /// Whether [`Bus::handle_interrupts`] would leave everything as it is: no interrupt input
    /// high and no interrupt flag up (the DMA bits are not its business).
    #[inline(always)]
    fn interrupts_quiet(&self) -> bool {
        !self.apu.irq_pending()
            && !self.ppu.nmi_pending
            && self.cpu.irq_flags.bits() & 0x1F == 0
    }

    /// Catches the PPU up to where per-CPU-cycle clocking would have it now.
    ///
    /// Called before anything that observes or changes what the PPU does - a `$2000-$3FFF`
    /// access, a board register write, a frame-boundary or VBlank-edge dot falling due, a trace
    /// row - so it sees and changes the same dots it would have clocked one CPU cycle at a time.
    #[inline(always)]
    pub(crate) fn ppu_sync(&mut self) {
        self.ppu_sync_to(self.cpu.master_clock.saturating_sub(Cpu::PPU_OFFSET));
    }

    /// [`Bus::ppu_sync`] to an explicit master clock.
    #[inline(never)]
    pub(crate) fn ppu_sync_to(&mut self, clock: u32) {
        self.ppu_clock_to(clock);
        if self.ppu.nmi_pending {
            self.irq_quiet = false;
        }
        self.set_ppu_deadline();
    }

    /// Sets [`Bus::ppu_deadline`] from the PPU's next frame-boundary or VBlank-edge dot.
    #[inline(always)]
    fn set_ppu_deadline(&mut self) {
        self.ppu_deadline = self.ppu.next_event_clock() + Cpu::PPU_OFFSET;
    }

    /// A DMA cycle: the first after the halt clears it, the next the DMC's dummy read; OAM DMA
    /// cycles count as those for a DMC fetch that comes in while it runs.
    #[inline(always)]
    fn start_dma_cycle(&mut self) {
        if self.cpu.irq_flags(IrqFlags::DMA_HALT) {
            self.cpu.clear_irq_flags(IrqFlags::DMA_HALT);
        } else {
            self.cpu.clear_irq_flags(IrqFlags::DMA_DUMMY_READ);
        }
        self.start_cycle(self.cpu.start_cycles - 1);
    }

    /// OAM and DMC DMA (nesdev "DMA"), run when the CPU halts at a read with one pending.
    ///
    /// The halted read is repeated, then DMA runs on get (even) and put (odd) cycles: OAM DMA
    /// copies 256 bytes as get/put pairs after aligning to a get cycle (513 or 514 cycles from a
    /// `$4014` write), a DMC sample fetch reads on a get cycle after its halt and a dummy cycle
    /// (3 or 4 cycles alone, fewer inside an OAM DMA, whose cycles count as its halt and dummy).
    /// The CPU's repeated reads do not repeat for `$4016`/`$4017` (the joypads would shift).
    #[cold]
    #[inline(never)]
    fn handle_dma(&mut self, addr: u16) {
        self.start_cycle(self.cpu.start_cycles - 1);
        self.cpu_bus_read(addr);
        self.end_cycle(self.cpu.start_cycles + 1);
        self.cpu.clear_irq_flags(IrqFlags::DMA_HALT);

        let skip_dummy_reads = addr == 0x4016 || addr == 0x4017;
        let mut oam_offset = 0;
        let mut oam_dma_count = 0;
        let mut read_val = 0;
        loop {
            let dma_dmc = self.cpu.irq_flags(IrqFlags::DMA_DMC);
            let dma_oam_addr = self.cpu.dma_oam_addr;
            if !dma_dmc && dma_oam_addr.is_none() {
                break;
            }
            if self.cpu.cycle & 0x01 == 0x00 {
                if dma_dmc
                    && !self.cpu.irq_flags(IrqFlags::DMA_HALT)
                    && !self.cpu.irq_flags(IrqFlags::DMA_DUMMY_READ)
                {
                    // The DMC's fetch, its halt and dummy cycles done.
                    self.start_dma_cycle();
                    self.cpu_bus_read(self.apu.dmc.dma_addr());
                    self.end_cycle(self.cpu.start_cycles + 1);
                    self.apu.dmc.load_buffer();
                    self.cpu.clear_irq_flags(IrqFlags::DMA_DMC);
                    // The buffer moved the DMC's next fetch, and the last byte may raise its IRQ.
                    self.tick_deadline = 0;
                    self.irq_quiet = false;
                } else if let Some(oam_addr) = dma_oam_addr {
                    self.start_dma_cycle();
                    read_val = self.cpu_bus_read(oam_addr + oam_offset);
                    self.end_cycle(self.cpu.start_cycles + 1);
                    oam_offset += 1;
                    oam_dma_count += 1;
                } else {
                    // A DMC fetch still halting or on its dummy cycle, no OAM DMA.
                    self.start_dma_cycle();
                    if !skip_dummy_reads {
                        self.cpu_bus_read(addr);
                    }
                    self.end_cycle(self.cpu.start_cycles + 1);
                }
            } else if dma_oam_addr.is_some() && oam_dma_count & 0x01 == 0x01 {
                // OAM DMA's put cycle after each get.
                self.start_dma_cycle();
                self.cpu_bus_write(0x2004, read_val);
                self.end_cycle(self.cpu.start_cycles + 1);
                oam_dma_count += 1;
                if oam_dma_count == 0x200 {
                    self.cpu.dma_oam_addr = None;
                }
            } else {
                // Align to a get cycle for OAM DMA or the DMC's fetch.
                self.start_dma_cycle();
                if !skip_dummy_reads {
                    self.cpu_bus_read(addr);
                }
                self.end_cycle(self.cpu.start_cycles + 1);
            }
        }
    }

    // Stack Functions

    /// Push a byte to the stack.
    #[inline(always)]
    pub(crate) fn push_byte(&mut self, val: u8) {
        self.write(Cpu::SP_BASE | u16::from(self.cpu.sp), val);
        self.cpu.sp = self.cpu.sp.wrapping_sub(1);
    }

    /// Pull a byte from the stack.
    #[inline(always)]
    #[must_use]
    pub(crate) fn pop_byte(&mut self) -> u8 {
        self.cpu.sp = self.cpu.sp.wrapping_add(1);
        self.read(Cpu::SP_BASE | u16::from(self.cpu.sp))
    }

    /// Peek byte at the top of the stack.
    #[inline]
    #[must_use]
    pub fn peek_stack(&self) -> u8 {
        self.peek(Cpu::SP_BASE | u16::from(self.cpu.sp.wrapping_add(1)))
    }

    /// Peek at the top of the stack.
    #[inline]
    #[must_use]
    pub fn peek_stack_u16(&self) -> u16 {
        let lo = self.peek(Cpu::SP_BASE | u16::from(self.cpu.sp));
        let hi = self.peek(Cpu::SP_BASE | u16::from(self.cpu.sp.wrapping_add(1)));
        u16::from_le_bytes([lo, hi])
    }

    /// Push a word (two bytes) to the stack
    #[inline(always)]
    pub(crate) fn push_word(&mut self, val: u16) {
        let [lo, hi] = val.to_le_bytes();
        self.push_byte(hi);
        self.push_byte(lo);
    }

    /// Pull a word (two bytes) from the stack
    #[inline(always)]
    pub(crate) fn pop_word(&mut self) -> u16 {
        let lo = self.pop_byte();
        let hi = self.pop_byte();
        u16::from_le_bytes([lo, hi])
    }

    // Memory accesses

    /// Read a byte, spending a full CPU cycle - which clocks the PPU, the APU and the board.
    #[inline(always)]
    pub fn read(&mut self, addr: u16) -> u8 {
        if self.cpu.irq_flags(IrqFlags::DMA_HALT) {
            self.handle_dma(addr);
        }

        self.start_cycle(self.cpu.start_cycles - 1);
        let val = self.cpu_bus_read(addr);
        self.end_cycle(self.cpu.end_cycles + 1);
        val
    }

    /// Read a byte without side effects, and without moving the console.
    #[inline(always)]
    #[must_use]
    pub fn peek(&self, addr: u16) -> u8 {
        self.cpu_bus_peek(addr)
    }

    /// Write a byte, spending a full CPU cycle - which clocks the PPU, the APU and the board.
    #[inline(always)]
    pub fn write(&mut self, addr: u16, val: u8) {
        self.start_cycle(self.cpu.start_cycles + 1);
        if addr == 0x4014 {
            self.cpu.start_oam_dma(u16::from(val) << 8);
        } else {
            self.cpu_bus_write(addr, val);
        }
        self.end_cycle(self.cpu.end_cycles - 1);
    }

    /// Fetch a byte and increments PC by 1.
    #[inline(always)]
    #[must_use]
    pub(crate) fn fetch_byte(&mut self) -> u8 {
        let val = self.read(self.cpu.pc);
        self.cpu.pc = self.cpu.pc.wrapping_add(1);
        val
    }

    /// Fetch opcode operand based on addressing mode.
    #[inline(always)]
    #[must_use]
    fn fetch_operand(&mut self) -> u16 {
        match self.cpu.addr_mode {
            AddrMode::ACC | AddrMode::IMP => self.acc_imp(),
            AddrMode::IMM | AddrMode::REL | AddrMode::ZP0 => self.imm_rel_zp(),
            AddrMode::ZPX => self.zpx(),
            AddrMode::ZPY => self.zpy(),
            AddrMode::IND => self.ind(),
            AddrMode::IDX => self.idx(),
            AddrMode::IDY => self.idy(false),
            AddrMode::IDYW => self.idy(true),
            AddrMode::ABS => self.abs(),
            AddrMode::ABX => self.abx(false),
            AddrMode::ABXW => self.abx(true),
            AddrMode::ABY => self.aby(false),
            AddrMode::ABYW => self.aby(true),
            AddrMode::OTH => 0,
        }
    }

    /// Fetch a 16-bit word and increments PC by 2.
    #[inline(always)]
    #[must_use]
    pub(crate) fn fetch_word(&mut self) -> u16 {
        let lo = self.fetch_byte();
        let hi = self.fetch_byte();
        u16::from_le_bytes([lo, hi])
    }

    /// Read operand value.
    #[inline(always)]
    #[must_use]
    pub(crate) fn read_operand(&mut self) -> u8 {
        if matches!(
            self.cpu.addr_mode,
            AddrMode::ACC | AddrMode::IMP | AddrMode::IMM | AddrMode::REL
        ) {
            self.cpu.operand as u8
        } else {
            self.read(self.cpu.operand)
        }
    }

    /// Read a 16-bit word.
    #[inline(always)]
    #[must_use]
    pub fn read_word(&mut self, addr: u16) -> u16 {
        let lo = self.read(addr);
        let hi = self.read(addr.wrapping_add(1));
        u16::from_le_bytes([lo, hi])
    }

    /// Peek a 16-bit word without side effects.
    #[inline]
    #[must_use]
    pub fn peek_word(&self, addr: u16) -> u16 {
        let lo = self.peek(addr);
        let hi = self.peek(addr.wrapping_add(1));
        u16::from_le_bytes([lo, hi])
    }

    /// Hands the instruction about to run to the trace hook.
    #[cold]
    #[inline(never)]
    fn trace_instr(&mut self) {
        let c = &self.cpu;
        let row = TraceRow {
            pc: c.pc,
            opcode: self.peek(c.pc),
            a: c.acc,
            x: c.x,
            y: c.y,
            p: c.status.bits(),
            sp: c.sp,
            cycle: c.cycle,
            scanline: self.ppu.scanline,
            dot: self.ppu.cycle,
        };
        if let Some(hook) = self.trace.as_mut() {
            hook(&row);
        }
    }

    /// Runs all components up to master clock, synchronizing them.
    #[inline(always)]
    pub fn clock_sync(&mut self) {
        self.ppu_sync_to(self.cpu.master_clock);
        self.cpu.master_clock = self.cpu.master_clock.saturating_sub(self.ppu.master_clock);
        self.ppu.master_clock = 0;
        self.apu.clock_sync();
        self.tick_deadline = 0;
        self.irq_quiet = false;
        self.set_ppu_deadline();
    }

    /// Runs instructions until the frame counter advances (FQ-08), then synchronizes.
    pub fn clock_frame(&mut self) {
        // The PPU is caught up between frames; anything may have changed it, the APU or the
        // interrupt state since.
        self.set_ppu_deadline();
        self.tick_deadline = 0;
        self.irq_quiet = false;
        let frame = self.ppu.frame_number();
        while frame == self.ppu.frame_number() {
            self.step_instr();
        }
        self.clock_sync();
    }

    /// Runs the CPU one instruction, leaving the PPU caught up with it.
    pub fn clock_instr(&mut self) {
        self.set_ppu_deadline();
        self.tick_deadline = 0;
        self.irq_quiet = false;
        self.step_instr();
        self.ppu_sync();
    }

    /// Runs the CPU one instruction, catching the PPU up only as far as the instruction needed.
    #[inline(always)]
    fn step_instr(&mut self) {
        if self.trace.is_some() {
            self.ppu_sync();
            self.trace_instr();
        }

        let opcode = self.fetch_byte(); // Cycle 1
        let op = Cpu::OPS[usize::from(opcode)];
        self.cpu.addr_mode = op.addr_mode();
        self.cpu.operand = self.fetch_operand();
        op.run(self);

        if self
            .cpu
            .irq_flags
            .intersects(IrqFlags::PREV_RUN_IRQ | IrqFlags::PREV_NMI)
        {
            self.irq();
        }
        if !self.irq_quiet {
            self.irq_quiet = self.interrupts_quiet();
        }
    }
}

impl fmt::Debug for Cpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::result::Result<(), fmt::Error> {
        f.debug_struct("Cpu")
            .field("cycle", &self.cycle)
            .field("pc", &format_args!("${:04X}", self.pc))
            .field("sp", &format_args!("${:02X}", self.sp))
            .field("acc", &format_args!("${:02X}", self.acc))
            .field("x", &format_args!("${:02X}", self.x))
            .field("y", &format_args!("${:02X}", self.y))
            .field("status", &self.status)
            .field("interrupt_flags", &self.irq_flags)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::{cart::Cart, cpu::instr::Instr::*, mapper::Nrom};

    /// An instruction is as long as its addressing mode's operand says.
    #[test]
    fn every_opcodes_length_matches_what_executing_it_consumes() {
        use super::*;
        let mut bus = Bus::default();
        let mut cart = Cart::empty();
        cart.mapper = Nrom::load(&mut cart).unwrap();
        bus.load_cart(cart);

        for instr_ref in Cpu::INSTR_REF.iter() {
            // A jam never leaves the instruction, and a branch or jump moves PC by its own rules.
            if matches!(
                instr_ref.instr,
                HLT | JMP | JSR | RTS | RTI | BRK | BCC | BCS | BEQ | BMI | BNE | BPL | BVC | BVS
            ) {
                continue;
            }
            bus.reset(ResetKind::Hard);
            bus.cpu_bus_write(0x0000, instr_ref.opcode);
            bus.cpu.pc = 0x0000;
            let len = 1 + u16::from(instr_ref.addr_mode.operand_len());
            bus.clock_instr();
            let consumed = bus.cpu.pc;
            assert_eq!(
                len, consumed,
                "${:02X} {:?} #{:?} runs {consumed} bytes",
                instr_ref.opcode, instr_ref.instr, instr_ref.addr_mode
            );
        }
    }

    #[test]
    #[ignore = "FQ-06: reset spends no cycles; expects 7"]
    fn cycle_timing() {
        use super::*;
        let mut bus = Bus::default();
        let mut cart = Cart::empty();
        cart.mapper = Nrom::load(&mut cart).unwrap();
        bus.load_cart(cart);
        bus.reset(ResetKind::Hard);
        bus.clock_instr();

        assert_eq!(bus.cpu.cycle, 14, "cpu after power + one clock");

        for instr_ref in Cpu::INSTR_REF.iter() {
            let extra_cycle = match instr_ref.instr {
                BCC | BNE | BPL | BVC => 1,
                _ => 0,
            };
            // Ignore invalid opcodes
            if instr_ref.instr == HLT {
                continue;
            }
            bus.reset(ResetKind::Hard);
            bus.cpu_bus_write(0x0000, instr_ref.opcode);
            bus.clock_instr();
            let cpu_cyc = u32::from(7 + instr_ref.cycles + extra_cycle);
            assert_eq!(
                bus.cpu.cycle, cpu_cyc,
                "cpu ${:02X} {:?} #{:?}",
                instr_ref.opcode, instr_ref.instr, instr_ref.addr_mode
            );
        }
    }

    /// OAM DMA halts the CPU at the read after `STA $4014` for 513 cycles, 514 when it halts on
    /// a put (odd) cycle (nesdev "DMA").
    #[test]
    fn oam_dma_takes_513_or_514_cycles_from_the_next_read() {
        use super::*;
        let mut lens = Vec::new();
        // With and without a 3-cycle `LDA $00` first, so the halt lands on both cycle parities.
        for prefix in [&[][..], &[0xA5, 0x00][..]] {
            let mut bus = Bus::default();
            let mut cart = Cart::empty();
            cart.mapper = Nrom::load(&mut cart).unwrap();
            bus.load_cart(cart);
            bus.reset(ResetKind::Hard);
            let code: Vec<u8> = [prefix, &[0x8D, 0x14, 0x40, 0xEA][..]].concat();
            for (addr, &val) in (0x0010u16..).zip(&code) {
                bus.cpu_bus_write(addr, val);
            }
            bus.cpu.pc = 0x0010;
            bus.cpu.acc = 0x02;
            if !prefix.is_empty() {
                bus.clock_instr();
            }
            let before = bus.cpu.cycle;
            bus.clock_instr();
            assert_eq!(bus.cpu.cycle - before, 4, "the STA itself");
            let before = bus.cpu.cycle;
            bus.clock_instr();
            lens.push(bus.cpu.cycle - before - 2);
        }
        lens.sort_unstable();
        assert_eq!(lens, [513, 514]);
    }

    /// A playing DMC sample halts the CPU for each byte, 3 or 4 cycles between NOPs, once per
    /// eight timer periods once the first byte is shifting, and its last byte raises the DMC IRQ
    /// (nesdev "APU DMC", "DMA").
    #[test]
    fn dmc_fetches_halt_the_cpu_once_per_byte() {
        use super::*;
        let mut bus = Bus::default();
        let mut cart = Cart::empty();
        cart.mapper = Nrom::load(&mut cart).unwrap();
        bus.load_cart(cart);
        bus.reset(ResetKind::Hard);
        // NOPs from $0010, looping back with `JMP $0010` at $07FD.
        for addr in 0x0010u16..0x07FD {
            bus.cpu_bus_write(addr, 0xEA);
        }
        for (addr, val) in [(0x07FDu16, 0x4Cu8), (0x07FE, 0x10), (0x07FF, 0x00)] {
            bus.cpu_bus_write(addr, val);
        }
        bus.cpu.pc = 0x0010;
        // IRQ on, fastest rate (54 cycles); 17 bytes from $C000.
        for (addr, val) in [(0x4010u16, 0x8Fu8), (0x4012, 0x00), (0x4013, 0x01), (0x4015, 0x10)] {
            bus.cpu_bus_write(addr, val);
        }
        let mut fetches = Vec::new();
        for _ in 0..4000 {
            let base = if bus.cpu.pc == 0x07FD { 3 } else { 2 };
            let before = bus.cpu.cycle;
            bus.clock_instr();
            let stolen = bus.cpu.cycle - before - base;
            if stolen > 0 {
                assert!((3..=4).contains(&stolen), "a fetch stole {stolen} cycles");
                fetches.push(before);
            }
        }
        assert_eq!(fetches.len(), 17, "{fetches:?}");
        // The second fetch waits for the first byte to reach the shifter.
        for gap in fetches.windows(2).skip(1).map(|w| w[1] - w[0]) {
            assert!((8 * 54 - 4..=8 * 54 + 4).contains(&gap), "gap {gap} in {fetches:?}");
        }
        assert!(bus.apu.dmc.irq_pending && bus.apu.peek_status() & 0x80 != 0);
        assert_eq!(bus.apu.dmc.bytes_remaining, 0);
    }
}
