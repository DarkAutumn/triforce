//! Emulator backends driven by movies, and the digest/replay machinery shared by `golden`, `diff`
//! and the ladder commands.

use crate::formats::{Domain, Golden, Movie, digest8};
use anyhow::Result;

pub use oracle::TraceRow;

/// Rows compare on `pc, opcode, a, x, y, sp` and the N/V/D/I/Z/C flags (B and the unused bit are not
/// real register state).
pub fn same_instruction(a: &TraceRow, b: &TraceRow) -> bool {
    a.pc == b.pc
        && a.opcode == b.opcode
        && a.a == b.a
        && a.x == b.x
        && a.y == b.y
        && a.sp == b.sp
        && (a.p & 0xCF) == (b.p & 0xCF)
}

pub trait Backend {
    fn name(&self) -> &'static str;
    fn run_frame(&mut self, buttons: u8);
    fn poke(&mut self, addr: u16, value: u8);
    /// Fill `out` with the bytes of a domain. Returns false when the backend can't produce it.
    fn read_domain(&mut self, domain: Domain, out: &mut Vec<u8>) -> bool;
    /// Run one frame while recording every instruction.
    fn trace_frame(&mut self, buttons: u8) -> Vec<TraceRow>;
    /// Debugging aid: how many dots the last frame's final instruction ran past the frame end.
    fn overshoot_dots(&self) -> Option<f64> {
        None
    }
    fn ram(&mut self) -> [u8; 0x800] {
        let mut v = Vec::new();
        self.read_domain(Domain::Ram, &mut v);
        v.try_into().expect("RAM domain is 2 KiB")
    }
}

/// Wraps a backend so every frame runs with instruction tracing on; counts instructions.
pub struct Tracing<B: Backend>(pub B, pub u64);

impl<B: Backend> Backend for Tracing<B> {
    fn name(&self) -> &'static str {
        self.0.name()
    }
    fn run_frame(&mut self, buttons: u8) {
        self.1 += self.0.trace_frame(buttons).len() as u64;
    }
    fn poke(&mut self, addr: u16, value: u8) {
        self.0.poke(addr, value)
    }
    fn read_domain(&mut self, domain: Domain, out: &mut Vec<u8>) -> bool {
        self.0.read_domain(domain, out)
    }
    fn trace_frame(&mut self, buttons: u8) -> Vec<TraceRow> {
        self.0.trace_frame(buttons)
    }
}

/// Pixel rows a comparison ignores: called with the record number and the 256x240 frame, it blanks
/// (zeroes) the rows to ignore before the frame is digested. See `corpus::PixelAllowlist`.
pub type PixelBlank<'a> = &'a dyn Fn(usize, &mut [u8]);

/// Digests of the enabled domains. Record 0 has no pixel digest (0).
pub fn digests(
    b: &mut dyn Backend,
    mask: u16,
    record: usize,
    blank: PixelBlank,
    buf: &mut Vec<u8>,
) -> Result<Vec<u64>> {
    let mut out = Vec::new();
    for d in Domain::in_mask(mask) {
        if record == 0 && d == Domain::Pixels {
            out.push(0);
            continue;
        }
        buf.clear();
        anyhow::ensure!(b.read_domain(d, buf), "backend {} can't read domain {}", b.name(), d.name());
        if d == Domain::Pixels {
            blank(record, buf);
        }
        out.push(digest8(buf));
    }
    Ok(out)
}

/// Bring a freshly powered-on `b` to `movie`'s starting state: play its base movie (recursively),
/// or nothing for a power-on movie.
pub fn play_base(b: &mut dyn Backend, movie: &Movie) -> Result<()> {
    if let Some(name) = movie.base() {
        let base = Movie::load(&crate::corpus::movie_path(name))?;
        anyhow::ensure!(base.rom_sha1 == movie.rom_sha1, "base movie {name} is for a different ROM");
        play_movie(b, &base)?;
    }
    Ok(())
}

/// Play `movie` (base included) on a freshly powered-on `b`, including pokes tagged after its last
/// frame.
pub fn play_movie(b: &mut dyn Backend, movie: &Movie) -> Result<()> {
    play_base(b, movie)?;
    let index = movie.poke_index();
    let mut next = 0;
    for (k, &x) in movie.inputs.iter().enumerate() {
        apply_pokes(b, movie, &index, &mut next, k);
        b.run_frame(x);
    }
    apply_pokes(b, movie, &index, &mut next, movie.inputs.len());
    Ok(())
}

/// Replay `movie` on a freshly powered-on `b`, calling `each(k, &digests)` for records `0..=frames`
/// (record 0 is the movie's starting state: power-on, or the end of its base). Stops early when
/// `each` returns false. `limit` caps the number of input frames run.
pub fn replay(
    b: &mut dyn Backend,
    movie: &Movie,
    mask: u16,
    limit: Option<usize>,
    blank: PixelBlank,
    mut each: impl FnMut(usize, &[u64]) -> bool,
) -> Result<()> {
    play_base(b, movie)?;
    let mut buf = Vec::with_capacity(256 * 240);
    let d0 = digests(b, mask, 0, blank, &mut buf)?;
    if !each(0, &d0) {
        return Ok(());
    }
    let index = movie.poke_index();
    let mut next = 0usize;
    let frames = limit.unwrap_or(movie.inputs.len()).min(movie.inputs.len());
    for k in 0..frames {
        apply_pokes(b, movie, &index, &mut next, k);
        b.run_frame(movie.inputs[k]);
        let d = digests(b, mask, k + 1, blank, &mut buf)?;
        if !each(k + 1, &d) {
            return Ok(());
        }
    }
    Ok(())
}

pub fn apply_pokes(
    b: &mut dyn Backend,
    movie: &Movie,
    index: &[(u32, std::ops::Range<usize>)],
    next: &mut usize,
    frame: usize,
) {
    while *next < index.len() && index[*next].0 as usize <= frame {
        let (f, r) = &index[*next];
        if *f as usize == frame {
            for p in &movie.pokes[r.clone()] {
                b.poke(p.addr, p.value);
            }
        }
        *next += 1;
    }
}

pub fn golden_of(
    b: &mut dyn Backend,
    movie: &Movie,
    key: [u8; 32],
    mask: u16,
    blank: PixelBlank,
) -> Result<Golden> {
    let mut records = Vec::with_capacity(movie.inputs.len() + 1);
    replay(b, movie, mask, None, blank, |_, d| {
        records.push(d.to_vec());
        true
    })?;
    Ok(Golden { domain_mask: mask, movie_hash: key, records })
}

/// First record where `b` diverges from `golden`, with the differing domains.
pub struct Divergence {
    pub record: usize,
    pub domains: Vec<Domain>,
}

pub fn first_divergence(
    b: &mut dyn Backend,
    movie: &Movie,
    golden: &Golden,
    mask: u16,
    blank: PixelBlank,
) -> Result<Option<Divergence>> {
    let gdomains = golden.domains();
    let mask = mask & golden.domain_mask;
    let ours: Vec<Domain> = Domain::in_mask(mask).collect();
    let cols: Vec<usize> = ours.iter().map(|d| gdomains.iter().position(|g| g == d).unwrap()).collect();
    anyhow::ensure!(
        golden.records.len() == movie.inputs.len() + 1,
        "golden has {} records for a {}-frame movie",
        golden.records.len(),
        movie.inputs.len()
    );
    let mut found = None;
    replay(b, movie, mask, None, blank, |k, d| {
        let mut bad = Vec::new();
        for (i, dom) in ours.iter().enumerate() {
            if d[i] != golden.records[k][cols[i]] {
                bad.push(*dom);
            }
        }
        if bad.is_empty() {
            true
        } else {
            found = Some(Divergence { record: k, domains: bad });
            false
        }
    })?;
    Ok(found)
}
