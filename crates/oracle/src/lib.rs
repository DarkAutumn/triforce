//! Dev-only binding to the FCEUmm oracle (`target/oracle/libtforacle.so`, built by
//! `oracle/fceumm/build.sh`). FCEUmm is GPL: it is loaded at runtime and never linked into shipping
//! crates. FCEUmm keeps its state in globals, so one oracle exists per process; `Oracle::open` fails
//! while another is open (tests using it must serialize, e.g. with [`lock`]).

use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use std::ffi::{CString, c_void};
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

static OPEN: AtomicBool = AtomicBool::new(false);
static LOCK: Mutex<()> = Mutex::new(());

/// Serializes oracle use across test threads in one process.
pub fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// One executed instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceRow {
    pub pc: u16,
    pub opcode: u8,
    pub a: u8,
    pub x: u8,
    pub y: u8,
    pub p: u8,
    pub sp: u8,
    /// CPU cycles since the start of the frame (FCEUmm `timestamp`); informational only.
    pub cycle: u64,
    pub scanline: i16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawRow {
    pc: u16,
    opcode: u8,
    a: u8,
    x: u8,
    y: u8,
    p: u8,
    sp: u8,
    cycle: u64,
    scanline: i16,
}

type TraceCb = unsafe extern "C" fn(*const RawRow, *mut c_void);

struct Api {
    close: unsafe extern "C" fn(),
    run_frame: unsafe extern "C" fn(u8),
    ram: unsafe extern "C" fn() -> *mut u8,
    prg_ram: unsafe extern "C" fn() -> *mut u8,
    read_chr: unsafe extern "C" fn(*mut u8),
    read_ciram: unsafe extern "C" fn(*mut u8),
    read_palette: unsafe extern "C" fn(*mut u8),
    read_oam: unsafe extern "C" fn(*mut u8),
    read_pixels: unsafe extern "C" fn(*mut u8),
    state_size: unsafe extern "C" fn() -> usize,
    save_state: unsafe extern "C" fn(*mut u8, usize) -> c_int,
    load_state: unsafe extern "C" fn(*const u8, usize) -> c_int,
    set_trace: unsafe extern "C" fn(Option<TraceCb>, *mut c_void),
    count: unsafe extern "C" fn() -> i32,
}

pub struct Oracle {
    api: Api,
    ram: *mut u8,
    prg_ram: *mut u8,
    frame: u64,
    _lib: Library,
}

/// `$TF_ORACLE`, else `<workspace>/target/oracle/libtforacle.so`.
pub fn library_path() -> PathBuf {
    match std::env::var_os("TF_ORACLE") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/oracle/libtforacle.so"),
    }
}

pub fn available() -> bool {
    library_path().exists()
}

unsafe extern "C" fn collect(row: *const RawRow, user: *mut c_void) {
    let rows = unsafe { &mut *(user as *mut Vec<TraceRow>) };
    let r = unsafe { *row };
    rows.push(TraceRow {
        pc: r.pc,
        opcode: r.opcode,
        a: r.a,
        x: r.x,
        y: r.y,
        p: r.p,
        sp: r.sp,
        cycle: r.cycle,
        scanline: r.scanline,
    });
}

impl Oracle {
    /// Loads the library, loads the ROM and powers on (PRG-RAM zero-filled, no frame run yet).
    pub fn open(rom_path: &Path) -> Result<Oracle> {
        let path = library_path();
        ensure!(path.exists(), "oracle library not found at {}: run oracle/fceumm/build.sh", path.display());
        if OPEN.swap(true, Ordering::SeqCst) {
            bail!("only one FCEUmm oracle can be open per process");
        }
        let result = (|| -> Result<Oracle> {
            let lib = unsafe { Library::new(&path) }.with_context(|| format!("loading {}", path.display()))?;
            macro_rules! sym {
                ($n:literal) => {
                    *unsafe { lib.get(concat!($n, "\0").as_bytes()) }.context($n)?
                };
            }
            let open: unsafe extern "C" fn(*const c_char) -> c_int = sym!("tfo_open");
            let api = Api {
                close: sym!("tfo_close"),
                run_frame: sym!("tfo_run_frame"),
                ram: sym!("tfo_ram"),
                prg_ram: sym!("tfo_prg_ram"),
                read_chr: sym!("tfo_read_chr"),
                read_ciram: sym!("tfo_read_ciram"),
                read_palette: sym!("tfo_read_palette"),
                read_oam: sym!("tfo_read_oam"),
                read_pixels: sym!("tfo_read_pixels"),
                state_size: sym!("tfo_state_size"),
                save_state: sym!("tfo_save_state"),
                load_state: sym!("tfo_load_state"),
                set_trace: sym!("tfo_set_trace"),
                count: sym!("tfo_count"),
            };
            let rom = CString::new(rom_path.to_string_lossy().as_bytes())?;
            let rc = unsafe { open(rom.as_ptr()) };
            ensure!(rc == 0, "tfo_open({}) failed with {rc}", rom_path.display());
            let (ram, prg_ram) = unsafe { ((api.ram)(), (api.prg_ram)()) };
            Ok(Oracle { api, ram, prg_ram, frame: 0, _lib: lib })
        })();
        if result.is_err() {
            OPEN.store(false, Ordering::SeqCst);
        }
        result
    }

    /// Runs one frame (to scanline 240) with the given NES controller byte.
    pub fn run_frame(&mut self, buttons: u8) {
        unsafe { (self.api.run_frame)(buttons) };
        self.frame += 1;
    }

    /// Frames run since power-on (or set by `load_state`).
    pub fn frame(&self) -> u64 {
        self.frame
    }

    pub fn ram(&self) -> &[u8; 0x800] {
        unsafe { &*(self.ram as *const [u8; 0x800]) }
    }

    pub fn prg_ram(&self) -> &[u8; 0x2000] {
        unsafe { &*(self.prg_ram as *const [u8; 0x2000]) }
    }

    /// Writes RAM ($0000-$07FF) or PRG-RAM ($6000-$7FFF). Panics on other addresses.
    pub fn poke(&mut self, addr: u16, value: u8) {
        match addr {
            0x0000..0x0800 => unsafe { *self.ram.add(addr as usize) = value },
            0x6000..0x8000 => unsafe { *self.prg_ram.add(addr as usize - 0x6000) = value },
            _ => panic!("poke address ${addr:04X} is outside $0000-$07FF/$6000-$7FFF"),
        }
    }

    /// PPU $0000-$1FFF as currently mapped.
    pub fn chr(&self) -> [u8; 0x2000] {
        let mut b = [0; 0x2000];
        unsafe { (self.api.read_chr)(b.as_mut_ptr()) };
        b
    }

    pub fn ciram(&self) -> [u8; 0x800] {
        let mut b = [0; 0x800];
        unsafe { (self.api.read_ciram)(b.as_mut_ptr()) };
        b
    }

    /// $3F00-$3F1F as PPU reads see them (FCEUmm mirroring, no grayscale).
    pub fn palette(&self) -> [u8; 32] {
        let mut b = [0; 32];
        unsafe { (self.api.read_palette)(b.as_mut_ptr()) };
        b
    }

    pub fn oam(&self) -> [u8; 256] {
        let mut b = [0; 256];
        unsafe { (self.api.read_oam)(b.as_mut_ptr()) };
        b
    }

    /// Palette indices (`& 0x3F`) of the last frame, 256x240, row-major.
    pub fn pixels(&self, out: &mut [u8; 256 * 240]) {
        unsafe { (self.api.read_pixels)(out.as_mut_ptr()) };
    }

    /// FCEUmm savestate plus the frame counter (8 bytes, little-endian, appended).
    pub fn save_state(&self) -> Vec<u8> {
        let n = unsafe { (self.api.state_size)() };
        let mut buf = vec![0u8; n];
        assert_eq!(unsafe { (self.api.save_state)(buf.as_mut_ptr(), n) }, 0, "tfo_save_state failed");
        buf.extend_from_slice(&self.frame.to_le_bytes());
        buf
    }

    pub fn load_state(&mut self, buf: &[u8]) -> Result<()> {
        ensure!(buf.len() > 8, "oracle state too short");
        let (body, frame) = buf.split_at(buf.len() - 8);
        ensure!(unsafe { (self.api.load_state)(body.as_ptr(), body.len()) } == 0, "tfo_load_state failed");
        self.frame = u64::from_le_bytes(frame.try_into().unwrap());
        Ok(())
    }

    /// FCEUmm's CPU run budget after the last frame, in dots (`X.count / 16`; zero or negative:
    /// how far the last instruction ran past the frame's end). Debugging aid for frame timing.
    pub fn budget_dots(&self) -> f64 {
        f64::from(unsafe { (self.api.count)() }) / 16.0
    }

    /// Runs one frame while recording every instruction.
    pub fn trace_frame(&mut self, buttons: u8) -> Vec<TraceRow> {
        let mut rows: Vec<TraceRow> = Vec::with_capacity(32_768);
        unsafe { (self.api.set_trace)(Some(collect), &mut rows as *mut Vec<TraceRow> as *mut c_void) };
        self.run_frame(buttons);
        unsafe { (self.api.set_trace)(None, std::ptr::null_mut()) };
        rows
    }
}

impl Drop for Oracle {
    fn drop(&mut self) {
        unsafe { (self.api.close)() };
        OPEN.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rom() -> Option<PathBuf> {
        let p = std::env::var_os("TRIFORCE_ROM")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../roms/zelda.nes"));
        (p.exists() && available()).then_some(p)
    }

    #[test]
    fn reopen_is_deterministic_and_states_roundtrip() {
        let Some(rom) = rom() else {
            eprintln!("ROM not found: set TRIFORCE_ROM or create roms/zelda.nes");
            return;
        };
        let _g = lock();
        let run = |n: usize| {
            let mut o = Oracle::open(&rom).unwrap();
            assert!(Oracle::open(&rom).is_err(), "second oracle must be refused");
            for i in 0..n {
                o.run_frame(if i % 64 == 0 { 0x08 } else { 0 });
            }
            (*o.ram(), o.save_state())
        };
        let (a, sa) = run(300);
        let (b, _) = run(300);
        assert_eq!(a, b);

        let mut o = Oracle::open(&rom).unwrap();
        o.load_state(&sa).unwrap();
        assert_eq!(o.frame(), 300);
        assert_eq!(*o.ram(), a);
        let snap = o.save_state();
        for _ in 0..120 {
            o.run_frame(0);
        }
        let after = *o.ram();
        o.load_state(&snap).unwrap();
        for _ in 0..120 {
            o.run_frame(0);
        }
        assert_eq!(*o.ram(), after);
    }
}
