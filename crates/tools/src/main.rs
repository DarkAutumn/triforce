//! `tf`: movie, golden, diff, trace and corpus tooling for the zcore port.

mod backend;
mod corpus;
use movie as formats;
mod ladder;
mod oracle;
mod palette;
mod rom;
mod script;
mod zcore;

use anyhow::{Context, Result, bail, ensure};
use backend::{Backend, apply_pokes, first_divergence, golden_of};
use clap::{Parser, Subcommand};
use formats::{Domain, Golden, Movie, ORACLE_MASK, movie_file_hash};
use rand::{RngExt, SeedableRng};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "tf", about = "triforce emulator fidelity tools")]
struct Cli {
    /// ROM path (default: $TRIFORCE_ROM, else roms/zelda.nes).
    #[arg(long, global = true)]
    rom: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Create or inspect movies.
    Movie {
        #[command(subcommand)]
        cmd: MovieCmd,
    },
    /// Generate oracle goldens.
    Golden {
        movie: Option<PathBuf>,
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Every corpus movie (short goldens into corpus/golden, long into target/corpus-golden).
        #[arg(long)]
        all: bool,
        #[arg(long)]
        jobs: Option<usize>,
        /// Trace every instruction while generating (the golden must not change).
        #[arg(long)]
        trace: bool,
    },
    /// Compare a backend with the oracle golden; exit 0 or print the first divergence.
    Diff {
        movie: Option<PathBuf>,
        #[arg(long, default_value = "zcore")]
        backend: String,
        #[arg(long)]
        golden: Option<PathBuf>,
        #[arg(long)]
        all: bool,
        #[arg(long)]
        jobs: Option<usize>,
        /// Ignore the pixel domain (memory and VRAM only).
        #[arg(long)]
        no_pixels: bool,
    },
    /// Replay a movie to record K in the oracle and a backend and list the differing bytes.
    Show {
        movie: PathBuf,
        #[arg(long)]
        frame: usize,
        #[arg(long, default_value = "zcore")]
        backend: String,
        /// Print at most this many differing bytes per domain.
        #[arg(long, default_value_t = 32)]
        limit: usize,
        /// Also write both frames as PPM images into this directory.
        #[arg(long)]
        ppm: Option<PathBuf>,
    },
    /// Per-frame instruction counts in the oracle and a backend (debugging frame boundaries).
    #[command(hide = true)]
    FrameLens {
        movie: PathBuf,
        #[arg(long, default_value_t = 30)]
        frames: usize,
        #[arg(long, default_value = "zcore")]
        backend: String,
        /// Only print frames from this record on.
        #[arg(long, default_value_t = 1)]
        from: usize,
    },
    /// Trace one frame in the oracle and a backend and report the first differing instruction.
    TraceDiff {
        movie: PathBuf,
        #[arg(long)]
        frame: usize,
        #[arg(long, default_value = "zcore")]
        backend: String,
        /// Print this many rows of context around the divergence.
        #[arg(long, default_value_t = 8)]
        context: usize,
        /// Also compare CPU cycles relative to the aligned start (FCEUmm's per-frame timestamp is
        /// made cumulative) and report the first instruction whose timing differs.
        #[arg(long)]
        cycles: bool,
        /// Print aligned rows START..END of both traces (e.g. 30..60).
        #[arg(long)]
        rows: Option<String>,
    },
    /// Generate all-domain goldens for ladder self-diffs.
    LadderGolden {
        movies: Vec<PathBuf>,
        #[arg(short, long)]
        out: PathBuf,
        #[arg(long, default_value = "zcore")]
        backend: String,
        #[arg(long)]
        jobs: Option<usize>,
    },
    /// Compare the backend with ladder goldens on all 11 domains.
    LadderDiff {
        dir: PathBuf,
        #[arg(long, default_value = "zcore")]
        backend: String,
        #[arg(long)]
        jobs: Option<usize>,
    },
    /// Corpus manifest and coverage gate.
    Corpus {
        #[command(subcommand)]
        cmd: CorpusCmd,
    },
    /// Frames per second per core.
    Bench {
        #[arg(long, default_value = "zcore")]
        backend: String,
        #[arg(long, default_value_t = 20_000)]
        frames: usize,
    },
    /// Internal: diff one movie (used by --all fan-out).
    #[command(hide = true)]
    DiffOne { movie: PathBuf, backend: String, golden: PathBuf, mask: u16 },
    /// Internal: golden for one movie on any backend (ladder fan-out).
    #[command(hide = true)]
    GoldenOne { movie: PathBuf, out: PathBuf, backend: String, mask: u16 },
    /// Internal: per-row pixel differences between the oracle and a backend over a whole movie.
    #[command(hide = true)]
    PixelRows { movie: PathBuf, backend: String },
    /// Internal: replay a movie in the oracle and check it ends in Dungeon 2's boss room.
    #[command(hide = true)]
    EndRoom { movie: PathBuf },
}

#[derive(Subcommand)]
enum MovieCmd {
    /// Record a script in the oracle.
    Script {
        script: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Record random holds starting from the end of a base corpus movie.
    Random {
        #[arg(long, default_value = "power_on_to_start")]
        base: String,
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        frames: usize,
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Print movie information.
    Info { movie: PathBuf },
    /// Strip a base movie's frames from the front of a power-on movie and mark it `base`.
    Rebase {
        movie: PathBuf,
        #[arg(long, default_value = "power_on_to_start")]
        base: String,
        #[arg(short, long)]
        out: PathBuf,
    },
}

#[derive(Subcommand)]
enum CorpusCmd {
    /// Rebuild corpus/manifest.toml from corpus/movies/*.tfmovie.
    Sync,
    /// Coverage gate.
    Check,
}

pub fn make_backend(name: &str, rom: &rom::RomInfo) -> Result<Box<dyn Backend>> {
    match name {
        "fceumm" | "oracle" => Ok(Box::new(oracle::Oracle::open(&rom.path)?)),
        "zcore" => Ok(Box::new(zcore::Zcore::open(&rom.bytes)?)),
        _ => bail!("unknown backend {name:?} (available: fceumm, zcore)"),
    }
}

fn jobs(j: Option<usize>) -> usize {
    j.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))
}

fn main() {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::exit(2);
        }
    }
}

fn check_movie_rom(movie: &Movie, rom: &rom::RomInfo, path: &Path) -> Result<()> {
    ensure!(movie.rom_sha1 == rom.payload_sha1, "{} was recorded for a different ROM", path.display());
    Ok(())
}

fn run(cli: Cli) -> Result<i32> {
    let rom = rom::load(cli.rom.as_deref())?;
    match cli.cmd {
        Cmd::Movie { cmd } => movie_cmd(cmd, &rom),
        Cmd::Golden { movie, out, all, jobs: j, trace } => {
            if all {
                return corpus::golden_all(&rom, jobs(j));
            }
            let movie = movie.context("tf golden needs a movie or --all")?;
            let out = out.context("tf golden needs -o")?;
            if trace {
                let movie_data = Movie::load(&movie)?;
                let allow = corpus::PixelAllowlist::load()?;
                let name = movie_data.name().to_owned();
                let blank = |k: usize, px: &mut [u8]| allow.blank(&name, k, px);
                let mut b = backend::Tracing(oracle::Oracle::open(&rom.path)?, 0);
                let g = golden_of(&mut b, &movie_data, corpus::golden_key(&movie)?, ORACLE_MASK, &blank)?;
                g.save(&out)?;
                println!("traced {} instructions", b.1);
                return Ok(0);
            }
            golden_one(&rom, &movie, &out, "fceumm", ORACLE_MASK)?;
            Ok(0)
        }
        Cmd::Diff { movie, backend, golden, all, jobs: j, no_pixels } => {
            let mask = if no_pixels { ORACLE_MASK & !Domain::Pixels.bit() } else { ORACLE_MASK };
            if all {
                return corpus::diff_all(&rom, &backend, mask, jobs(j));
            }
            let movie = movie.context("tf diff needs a movie or --all")?;
            let golden = match golden {
                Some(g) => g,
                None => corpus::golden_path_for(&movie)?,
            };
            diff_one(&rom, &movie, &backend, &golden, mask)
        }
        Cmd::DiffOne { movie, backend, golden, mask } => diff_one(&rom, &movie, &backend, &golden, mask),
        Cmd::GoldenOne { movie, out, backend, mask } => {
            golden_one(&rom, &movie, &out, &backend, mask)?;
            Ok(0)
        }
        Cmd::EndRoom { movie } => corpus::end_room(&mut oracle::Oracle::open(&rom.path)?, &movie),
        Cmd::PixelRows { movie, backend } => pixel_rows(&rom, &movie, &backend),
        Cmd::TraceDiff { movie, frame, backend, context, cycles, rows } => {
            trace_diff(&rom, &movie, frame, &backend, context, cycles, rows.as_deref())
        }
        Cmd::Show { movie, frame, backend, limit, ppm } => show(&rom, &movie, frame, &backend, limit, ppm),
        Cmd::FrameLens { movie, frames, backend, from } => frame_lens(&rom, &movie, frames, from, &backend),
        Cmd::LadderGolden { movies, out, backend, jobs: j } => corpus::ladder_golden(&rom, &movies, &out, &backend, jobs(j)),
        Cmd::LadderDiff { dir, backend, jobs: j } => corpus::ladder_diff(&rom, &dir, &backend, jobs(j)),
        Cmd::Corpus { cmd } => match cmd {
            CorpusCmd::Sync => corpus::sync(),
            CorpusCmd::Check => corpus::check(&rom),
        },
        Cmd::Bench { backend, frames } => bench(&rom, &backend, frames),
    }
}

fn movie_cmd(cmd: MovieCmd, rom: &rom::RomInfo) -> Result<i32> {
    match cmd {
        MovieCmd::Script { script: path, out } => {
            let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
            let mut directives = script::parse(&text)?;
            let name = out.file_stem().unwrap().to_string_lossy().into_owned();
            let mut b = oracle::Oracle::open(&rom.path)?;
            let mut movie = Movie::new(
                rom.payload_sha1,
                serde_json::json!({"name": name, "tags": [], "source": {"script": path.file_name().unwrap().to_string_lossy()}}),
            );
            if let Some(script::Directive::Base(p)) = directives.first().cloned() {
                let base = Movie::load(&corpus::movie_path(&p))?;
                check_movie_rom(&base, rom, &path)?;
                backend::play_movie(&mut b, &base)?;
                movie.meta["base"] = serde_json::json!(p);
                directives.remove(0);
            }
            script::record(&mut b, &mut movie, &directives)?;
            movie.save(&out)?;
            println!("{}: {} frames, {} pokes", out.display(), movie.inputs.len(), movie.pokes.len());
            Ok(0)
        }
        MovieCmd::Random { base, seed, frames, out } => {
            let p = Movie::load(&corpus::movie_path(&base))?;
            check_movie_rom(&p, rom, &corpus::movie_path(&base))?;
            let mut b = oracle::Oracle::open(&rom.path)?;
            backend::play_movie(&mut b, &p)?;
            let name = out.file_stem().unwrap().to_string_lossy().into_owned();
            let mut movie = Movie::new(
                rom.payload_sha1,
                serde_json::json!({"name": name, "base": base, "tags": ["random"], "source": {"seed": seed}}),
            );
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            let end = movie.inputs.len() + frames;
            while movie.inputs.len() < end {
                let hold = rng.random_range(4..=32usize);
                let mut buttons: u8 = rng.random();
                if buttons & 0x08 != 0 && !rng.random_bool(0.02) {
                    buttons &= !0x08;
                }
                for _ in 0..hold.min(end - movie.inputs.len()) {
                    script::step(&mut b, &mut movie, buttons);
                }
            }
            movie.save(&out)?;
            println!("{}: {} frames", out.display(), movie.inputs.len());
            Ok(0)
        }
        MovieCmd::Info { movie: path } => {
            let m = Movie::load(&path)?;
            println!("name:   {}", m.name());
            println!("frames: {}", m.inputs.len());
            println!("pokes:  {}", m.pokes.len());
            println!("rom:    {}", rom::hex(&m.rom_sha1));
            println!("meta:   {}", m.meta);
            Ok(0)
        }
        MovieCmd::Rebase { movie: path, base, out } => {
            let m = Movie::load(&path)?;
            ensure!(m.base().is_none(), "{} already has base {:?}", path.display(), m.base());
            let b = Movie::load(&corpus::movie_path(&base))?;
            ensure!(b.base().is_none(), "base {base} must start from power-on");
            let n = b.inputs.len();
            ensure!(m.inputs.len() >= n && m.inputs[..n] == b.inputs[..], "{} does not start with {base}", path.display());
            let prefix_pokes: Vec<_> = m.pokes.iter().filter(|p| (p.frame as usize) < n).copied().collect();
            let base_pokes: Vec<_> = b.pokes.iter().filter(|p| (p.frame as usize) < n).copied().collect();
            ensure!(prefix_pokes == base_pokes, "{} pokes differ from {base} inside the base", path.display());
            let mut r = m.clone();
            r.inputs.drain(..n);
            r.pokes = m
                .pokes
                .iter()
                .filter(|p| p.frame as usize >= n)
                .map(|p| formats::Poke { frame: p.frame - n as u32, ..*p })
                .collect();
            r.meta["base"] = serde_json::json!(base);
            r.save(&out)?;
            println!("{}: {} frames after {base}", out.display(), r.inputs.len());
            Ok(0)
        }
    }
}

/// Oracle comparisons (no ladder-only domains) key goldens on the movie plus the pixel allowlist and
/// blank allowlisted rows; ladder self-diffs compare everything and key on the movie alone.
fn comparison_key(movie_path: &Path, mask: u16) -> Result<([u8; 32], Option<corpus::PixelAllowlist>)> {
    if mask & !ORACLE_MASK == 0 {
        Ok((corpus::golden_key(movie_path)?, Some(corpus::PixelAllowlist::load()?)))
    } else {
        Ok((movie_file_hash(movie_path)?, None))
    }
}

pub fn golden_one(rom: &rom::RomInfo, movie_path: &Path, out: &Path, backend: &str, mask: u16) -> Result<()> {
    let movie = Movie::load(movie_path)?;
    check_movie_rom(&movie, rom, movie_path)?;
    let (key, allow) = comparison_key(movie_path, mask)?;
    let name = movie.name().to_owned();
    let blank = |k: usize, px: &mut [u8]| {
        if let Some(a) = &allow {
            a.blank(&name, k, px)
        }
    };
    let mut b = make_backend(backend, rom)?;
    let g = golden_of(b.as_mut(), &movie, key, mask, &blank)?;
    g.save(out)?;
    Ok(())
}

/// Replay a movie in the oracle and a backend side by side and count, per pixel row, the frames in
/// which that row differs (debugging pixel divergences).
fn pixel_rows(rom: &rom::RomInfo, movie_path: &Path, backend: &str) -> Result<i32> {
    let movie = Movie::load(movie_path)?;
    let mut o = oracle::Oracle::open(&rom.path)?;
    let mut b = make_backend(backend, rom)?;
    backend::play_base(&mut o, &movie)?;
    backend::play_base(b.as_mut(), &movie)?;
    let index = movie.poke_index();
    let (mut no, mut nb) = (0, 0);
    let (mut po, mut pb) = (Vec::new(), Vec::new());
    let mut rows = [0usize; 240];
    let mut first = [usize::MAX; 240];
    let mut frames = 0;
    for k in 0..movie.inputs.len() {
        apply_pokes(&mut o, &movie, &index, &mut no, k);
        apply_pokes(b.as_mut(), &movie, &index, &mut nb, k);
        o.run_frame(movie.inputs[k]);
        b.run_frame(movie.inputs[k]);
        o.read_domain(Domain::Pixels, &mut po);
        b.read_domain(Domain::Pixels, &mut pb);
        let mut any = false;
        for (r, (ro, rb)) in po.chunks(256).zip(pb.chunks(256)).enumerate() {
            if ro != rb {
                rows[r] += 1;
                first[r] = first[r].min(k + 1);
                any = true;
            }
        }
        frames += usize::from(any);
    }
    println!("{}: {frames} of {} frames differ", movie_path.display(), movie.inputs.len());
    for (r, &n) in rows.iter().enumerate() {
        if n > 0 {
            println!("  row {r:3}: {n:6} frames (first {})", first[r]);
        }
    }
    Ok(if frames == 0 { 0 } else { 1 })
}

pub fn diff_one(rom: &rom::RomInfo, movie_path: &Path, backend: &str, golden_path: &Path, mask: u16) -> Result<i32> {
    let movie = Movie::load(movie_path)?;
    check_movie_rom(&movie, rom, movie_path)?;
    let golden = Golden::load(golden_path)?;
    let (key, allow) = comparison_key(movie_path, golden.domain_mask)?;
    ensure!(
        golden.movie_hash == key,
        "{} is stale (other movie or pixel allowlist) for {}: run `tf golden --all`",
        golden_path.display(),
        movie_path.display()
    );
    let name = movie.name().to_owned();
    let blank = |k: usize, px: &mut [u8]| {
        if let Some(a) = &allow {
            a.blank(&name, k, px)
        }
    };
    let mut b = make_backend(backend, rom)?;
    match first_divergence(b.as_mut(), &movie, &golden, mask, &blank)? {
        None => {
            println!("{}: {} frames identical ({})", movie_path.display(), movie.inputs.len(), backend);
            Ok(0)
        }
        Some(d) => {
            let names: Vec<&str> = d.domains.iter().map(|x| x.name()).collect();
            println!("{}: first divergence at frame {}: {}", movie_path.display(), d.record, names.join(","));
            Ok(1)
        }
    }
}

fn show(rom: &rom::RomInfo, movie_path: &Path, frame: usize, backend: &str, limit: usize, ppm: Option<PathBuf>) -> Result<i32> {
    let movie = Movie::load(movie_path)?;
    ensure!(frame <= movie.inputs.len(), "movie has only {} frames", movie.inputs.len());
    let mut o = oracle::Oracle::open(&rom.path)?;
    let mut b = make_backend(backend, rom)?;
    for x in [&mut o as &mut dyn Backend, b.as_mut()] {
        backend::play_base(x, &movie)?;
        let index = movie.poke_index();
        let mut next = 0;
        for k in 0..frame {
            apply_pokes(x, &movie, &index, &mut next, k);
            x.run_frame(movie.inputs[k]);
        }
    }
    if let Some(dir) = &ppm {
        std::fs::create_dir_all(dir)?;
        let (mut po, mut pb) = (Vec::new(), Vec::new());
        o.read_domain(Domain::Pixels, &mut po);
        b.read_domain(Domain::Pixels, &mut pb);
        palette::write_ppm(&dir.join(format!("fceumm-{frame}.ppm")), &po)?;
        palette::write_ppm(&dir.join(format!("{backend}-{frame}.ppm")), &pb)?;
    }
    let (mut bo, mut bb) = (Vec::new(), Vec::new());
    let mut same = true;
    for d in Domain::in_mask(ORACLE_MASK) {
        o.read_domain(d, &mut bo);
        b.read_domain(d, &mut bb);
        let diffs: Vec<usize> = (0..bo.len()).filter(|&i| bo[i] != bb[i]).collect();
        if diffs.is_empty() {
            continue;
        }
        same = false;
        if d == Domain::Pixels {
            let rows: std::collections::BTreeSet<usize> = diffs.iter().map(|i| i / 256).collect();
            println!("pixels: {} differ in {} rows: {:?}", diffs.len(), rows.len(), rows.iter().take(limit).collect::<Vec<_>>());
            for &i in diffs.iter().take(limit) {
                println!("  ({:3},{:3}): fceumm {:02X}  {backend} {:02X}", i % 256, i / 256, bo[i], bb[i]);
            }
            continue;
        }
        let base = match d {
            Domain::PrgRam => 0x6000,
            _ => 0,
        };
        println!("{}: {} bytes differ", d.name(), diffs.len());
        for &i in diffs.iter().take(limit) {
            println!("  {:04X}: fceumm {:02X}  {backend} {:02X}", base + i, bo[i], bb[i]);
        }
    }
    if same {
        println!("record {frame}: all oracle domains identical");
    }
    Ok(if same { 0 } else { 1 })
}

fn frame_lens(rom: &rom::RomInfo, movie_path: &Path, frames: usize, from: usize, backend: &str) -> Result<i32> {
    let movie = Movie::load(movie_path)?;
    let mut o = oracle::Oracle::open(&rom.path)?;
    let mut b = make_backend(backend, rom)?;
    backend::play_base(&mut o, &movie)?;
    backend::play_base(b.as_mut(), &movie)?;
    let index = movie.poke_index();
    let (mut no, mut nb) = (0, 0);
    for k in 0..frames.min(movie.inputs.len()) {
        apply_pokes(&mut o, &movie, &index, &mut no, k);
        apply_pokes(b.as_mut(), &movie, &index, &mut nb, k);
        let to = o.trace_frame(movie.inputs[k]);
        let tb = b.trace_frame(movie.inputs[k]);
        if k + 1 < from {
            continue;
        }
        let (fo, fb) = (to.first().map_or(0, |r| r.pc), tb.first().map_or(0, |r| r.pc));
        let (co, cb) = (to.first().map_or(0, |r| r.cycle as i64), tb.first().map_or(0, |r| r.cycle as i64));
        // NMI entry: the first row whose stack pointer is three below the previous row's.
        let nmi = |rows: &[backend::TraceRow]| {
            let start = rows.first().map_or(0, |r| r.cycle as i64);
            rows.windows(2).find(|w| w[1].sp == w[0].sp.wrapping_sub(3)).map_or(-1, |w| w[1].cycle as i64 - start)
        };
        let (ro, rb) = (o.ram(), b.ram());
        println!(
            "frame {:5}: fceumm {:6} instr start {fo:04X} | {backend} {:6} instr start {fb:04X} | ram {} | overshoot {} {} | start-cycle delta {} | nmi entry {} {}",
            k + 1,
            to.len(),
            tb.len(),
            if ro == rb { "same" } else { "DIFF" },
            o.overshoot_dots().unwrap_or(f64::NAN),
            b.overshoot_dots().unwrap_or(f64::NAN),
            cb - co,
            nmi(&to),
            nmi(&tb)
        );
    }
    Ok(0)
}

fn trace_diff(
    rom: &rom::RomInfo,
    movie_path: &Path,
    frame: usize,
    backend: &str,
    context: usize,
    cycles: bool,
    rows: Option<&str>,
) -> Result<i32> {
    ensure!(frame >= 1, "--frame is a 1-based record number");
    let movie = Movie::load(movie_path)?;
    ensure!(frame <= movie.inputs.len(), "movie has only {} frames", movie.inputs.len());
    let mut o = oracle::Oracle::open(&rom.path)?;
    let mut b = make_backend(backend, rom)?;
    backend::play_base(&mut o, &movie)?;
    backend::play_base(b.as_mut(), &movie)?;
    let index = movie.poke_index();
    let (mut no, mut nb) = (0, 0);
    for k in 0..frame - 1 {
        apply_pokes(&mut o, &movie, &index, &mut no, k);
        o.run_frame(movie.inputs[k]);
        apply_pokes(b.as_mut(), &movie, &index, &mut nb, k);
        b.run_frame(movie.inputs[k]);
    }
    apply_pokes(&mut o, &movie, &index, &mut no, frame - 1);
    apply_pokes(b.as_mut(), &movie, &index, &mut nb, frame - 1);
    let mut buf_o = Vec::new();
    let mut buf_b = Vec::new();
    for d in Domain::in_mask(ORACLE_MASK) {
        if d == Domain::Pixels {
            continue;
        }
        o.read_domain(d, &mut buf_o);
        b.read_domain(d, &mut buf_b);
        if buf_o != buf_b {
            println!("note: domain {} already differs before frame {frame}", d.name());
        }
    }
    // Frame boundaries fall a few instructions apart in different emulators, so trace two frames
    // (K-1 and K when K > 1) and align the starts before comparing.
    let mut to = o.trace_frame(movie.inputs[frame - 1]);
    let mut tb = b.trace_frame(movie.inputs[frame - 1]);
    if frame < movie.inputs.len() {
        apply_pokes(&mut o, &movie, &index, &mut no, frame);
        apply_pokes(b.as_mut(), &movie, &index, &mut nb, frame);
        to.extend(o.trace_frame(movie.inputs[frame]));
        tb.extend(b.trace_frame(movie.inputs[frame]));
    }
    println!("oracle {} instructions, {} {} instructions (frames {frame}..={})", to.len(), backend, tb.len(), frame + 1);
    let run = |i: usize, j: usize| (0..16).all(|k| match (to.get(i + k), tb.get(j + k)) {
        (Some(x), Some(y)) => backend::same_instruction(x, y),
        _ => false,
    });
    let mut align = None;
    'search: for s in 0..400 {
        for i in 0..=s {
            if run(i, s - i) {
                align = Some((i, s - i));
                break 'search;
            }
        }
    }
    let Some((oi, bi)) = align else {
        println!("could not align the traces within 400 instructions");
        return Ok(1);
    };
    println!("aligned: oracle row {oi} = {backend} row {bi}");
    let line = |r: &backend::TraceRow| {
        format!(
            "PC:{:04X} OP:{:02X} A:{:02X} X:{:02X} Y:{:02X} P:{:02X} SP:{:02X} SL:{:4} CYC:{}",
            r.pc, r.opcode, r.a, r.x, r.y, r.p, r.sp, r.scanline, r.cycle
        )
    };
    let n = (to.len() - oi).min(tb.len() - bi);
    if let Some(spec) = rows {
        let (a, b) = spec.split_once("..").context("--rows START..END")?;
        let (a, b): (usize, usize) = (a.parse()?, b.parse()?);
        for j in a..b.min(to.len().saturating_sub(oi)).max(a) {
            if let Some(r) = to.get(oi + j) {
                println!("{:7} fceumm  {}", oi + j, line(r));
            }
            if let Some(r) = tb.get(bi + j) {
                println!("{:7} {:7} {}", bi + j, backend, line(r));
            }
        }
    }
    if cycles {
        // Both traces carry CPU cycles since power-on; compare them relative to the aligned start.
        let cum: Vec<u64> = to.iter().map(|r| r.cycle).collect();
        let (o0, b0) = (cum[oi], tb[bi].cycle);
        // A single-row difference is the OAM DMA showing up after STA $4014 in FCEUmm but before
        // the next instruction in the backend; report differences that persist for two rows.
        let differs = |k: usize| cum[oi + k] - o0 != tb[bi + k].cycle - b0;
        if let Some(k) = (0..n.saturating_sub(1)).find(|&k| differs(k) && differs(k + 1)) {
            println!("first timing difference at aligned row {k}");
            for j in k.saturating_sub(context)..(k + 2).min(n) {
                println!("{:7} fceumm  +{:6} {}", oi + j, cum[oi + j] - o0, line(&to[oi + j]));
                println!("{:7} {:7} +{:6} {}", bi + j, backend, tb[bi + j].cycle - b0, line(&tb[bi + j]));
            }
        } else {
            println!("timing agrees on {n} aligned rows");
        }
    }
    match (0..n).find(|&k| !backend::same_instruction(&to[oi + k], &tb[bi + k])) {
        None => {
            println!("traces agree on {n} aligned rows");
            Ok(0)
        }
        Some(k) => {
            println!("first differing instruction at aligned row {k} (oracle {}, {backend} {})", oi + k, bi + k);
            for j in k.saturating_sub(context)..(k + 2).min(n) {
                println!("{:7} fceumm {}", oi + j, line(&to[oi + j]));
                println!("{:7} {:7} {}", bi + j, backend, line(&tb[bi + j]));
            }
            Ok(1)
        }
    }
}

/// The same seeded random input from the end of `power_on_to_start`, once drawing every frame and
/// once (for backends that can skip it) drawing none.
fn bench(rom: &rom::RomInfo, backend: &str, frames: usize) -> Result<i32> {
    let movie = Movie::load(&corpus::movie_path("power_on_to_start"))?;
    for render in [true, false] {
        let mut b = make_backend(backend, rom)?;
        backend::play_movie(b.as_mut(), &movie)?;
        if !b.set_render(render) && !render {
            println!("{backend}: always renders");
            break;
        }
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let t = std::time::Instant::now();
        for _ in 0..frames {
            b.run_frame(rng.random::<u8>() & !0x0C);
        }
        let dt = t.elapsed().as_secs_f64();
        let mode = if render { "on" } else { "off" };
        println!("{backend}: {:.0} frames/s (render {mode}, 1 core)", frames as f64 / dt);
    }
    Ok(0)
}

