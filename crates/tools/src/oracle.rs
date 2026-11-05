//! FCEUmm oracle backend: `target/oracle/libtforacle.so` (built by `oracle/fceumm/build.sh`) loaded
//! at runtime. FCEUmm keeps its state in globals, so one oracle exists per process.

use crate::backend::{Backend, TraceRow};
use crate::formats::Domain;
use anyhow::{Context, Result, bail, ensure};
use libloading::{Library, Symbol};
use std::ffi::{CString, c_void};
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

static OPEN: AtomicBool = AtomicBool::new(false);

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

pub struct Oracle {
    lib: Library,
}

pub fn library_path() -> PathBuf {
    match std::env::var_os("TF_ORACLE") {
        Some(p) => PathBuf::from(p),
        None => crate::rom::workspace_root().join("target/oracle/libtforacle.so"),
    }
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
    pub fn open(rom_path: &std::path::Path) -> Result<Oracle> {
        let path = library_path();
        ensure!(path.exists(), "oracle library not found at {}: run oracle/fceumm/build.sh", path.display());
        if OPEN.swap(true, Ordering::SeqCst) {
            bail!("only one FCEUmm oracle can be open per process");
        }
        let lib = unsafe { Library::new(&path) }.with_context(|| format!("loading {}", path.display()))?;
        let rom = CString::new(rom_path.to_string_lossy().as_bytes())?;
        let rc = unsafe {
            let f: Symbol<unsafe extern "C" fn(*const c_char) -> c_int> = lib.get(b"tfo_open")?;
            f(rom.as_ptr())
        };
        if rc != 0 {
            OPEN.store(false, Ordering::SeqCst);
            bail!("tfo_open failed with {rc}");
        }
        Ok(Oracle { lib })
    }

    fn sym<T>(&self, name: &[u8]) -> Symbol<'_, T> {
        unsafe { self.lib.get(name).expect("oracle symbol") }
    }

    fn mem(&self, name: &[u8], len: usize) -> &mut [u8] {
        let f: Symbol<unsafe extern "C" fn() -> *mut u8> = self.sym(name);
        unsafe { std::slice::from_raw_parts_mut(f(), len) }
    }

    fn read_into(&self, name: &[u8], len: usize, out: &mut Vec<u8>) {
        out.resize(len, 0);
        let f: Symbol<unsafe extern "C" fn(*mut u8)> = self.sym(name);
        unsafe { f(out.as_mut_ptr()) };
    }
}

impl Drop for Oracle {
    fn drop(&mut self) {
        let f: Symbol<unsafe extern "C" fn()> = self.sym(b"tfo_close");
        unsafe { f() };
        OPEN.store(false, Ordering::SeqCst);
    }
}

impl Backend for Oracle {
    fn name(&self) -> &'static str {
        "fceumm"
    }

    fn run_frame(&mut self, buttons: u8) {
        let f: Symbol<unsafe extern "C" fn(u8)> = self.sym(b"tfo_run_frame");
        unsafe { f(buttons) };
    }

    fn poke(&mut self, addr: u16, value: u8) {
        match addr {
            0x0000..0x0800 => self.mem(b"tfo_ram", 0x800)[addr as usize] = value,
            0x6000..0x8000 => self.mem(b"tfo_prg_ram", 0x2000)[addr as usize - 0x6000] = value,
            _ => panic!("poke address ${addr:04X} not allowed"),
        }
    }

    fn read_domain(&mut self, domain: Domain, out: &mut Vec<u8>) -> bool {
        match domain {
            Domain::Ram => {
                out.clear();
                out.extend_from_slice(self.mem(b"tfo_ram", 0x800));
            }
            Domain::PrgRam => {
                out.clear();
                out.extend_from_slice(self.mem(b"tfo_prg_ram", 0x2000));
            }
            Domain::Chr => self.read_into(b"tfo_read_chr", 0x2000, out),
            Domain::Ciram => self.read_into(b"tfo_read_ciram", 0x800, out),
            Domain::Palette => self.read_into(b"tfo_read_palette", 32, out),
            Domain::Oam => self.read_into(b"tfo_read_oam", 256, out),
            Domain::Pixels => self.read_into(b"tfo_read_pixels", 256 * 240, out),
            _ => return false,
        }
        true
    }

    fn trace_frame(&mut self, buttons: u8) -> Vec<TraceRow> {
        let mut rows: Vec<TraceRow> = Vec::with_capacity(32_768);
        let set: unsafe extern "C" fn(Option<TraceCb>, *mut c_void) = *self.sym(b"tfo_set_trace");
        unsafe { set(Some(collect), &mut rows as *mut Vec<TraceRow> as *mut c_void) };
        self.run_frame(buttons);
        unsafe { set(None, std::ptr::null_mut()) };
        rows
    }
}
