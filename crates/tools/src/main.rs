//! `tf`: movie, golden, diff, trace and corpus tooling for the zcore port.

mod backend;
mod corpus;
mod formats;
mod oracle;
mod rom;
mod script;
mod tetanes;

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
        "tetanes" => Ok(Box::new(tetanes::Tetanes::open(&rom.bytes)?)),
        _ => bail!("unknown backend {name:?} (available: fceumm, tetanes)"),
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
                let mut b = backend::Tracing(oracle::Oracle::open(&rom.path)?, 0);
                let g = golden_of(&mut b, &movie_data, movie_file_hash(&movie)?, ORACLE_MASK)?;
                g.save(&out)?;
                println!("traced {} instructions", b.1);
                return Ok(0);
            }
            golden_one(&rom, &movie, &out, "fceumm", ORACLE_MASK)?;
            Ok(0)
        }
        Cmd::Diff { movie, backend, golden, all, jobs: j } => {
            if all {
                return corpus::diff_all(&rom, &backend, jobs(j));
            }
            let movie = movie.context("tf diff needs a movie or --all")?;
            let golden = match golden {
                Some(g) => g,
                None => corpus::golden_path_for(&movie)?,
            };
            diff_one(&rom, &movie, &backend, &golden, ORACLE_MASK)
        }
        Cmd::DiffOne { movie, backend, golden, mask } => diff_one(&rom, &movie, &backend, &golden, mask),
        Cmd::GoldenOne { movie, out, backend, mask } => {
            golden_one(&rom, &movie, &out, &backend, mask)?;
            Ok(0)
        }
        Cmd::EndRoom { movie } => corpus::end_room(&mut oracle::Oracle::open(&rom.path)?, &movie),
        Cmd::TraceDiff { movie, frame, backend, context } => trace_diff(&rom, &movie, frame, &backend, context),
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

pub fn golden_one(rom: &rom::RomInfo, movie_path: &Path, out: &Path, backend: &str, mask: u16) -> Result<()> {
    let movie = Movie::load(movie_path)?;
    check_movie_rom(&movie, rom, movie_path)?;
    let mut b = make_backend(backend, rom)?;
    let g = golden_of(b.as_mut(), &movie, movie_file_hash(movie_path)?, mask)?;
    g.save(out)?;
    Ok(())
}

pub fn diff_one(rom: &rom::RomInfo, movie_path: &Path, backend: &str, golden_path: &Path, mask: u16) -> Result<i32> {
    let movie = Movie::load(movie_path)?;
    check_movie_rom(&movie, rom, movie_path)?;
    let golden = Golden::load(golden_path)?;
    ensure!(
        golden.movie_hash == movie_file_hash(movie_path)?,
        "{} was generated from a different movie than {}",
        golden_path.display(),
        movie_path.display()
    );
    let allow = corpus::PixelAllowlist::load()?;
    let name = movie.name().to_owned();
    let skip = |k: usize| allow.covers(&name, k);
    let mut b = make_backend(backend, rom)?;
    match first_divergence(b.as_mut(), &movie, &golden, mask, &skip)? {
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

fn trace_diff(rom: &rom::RomInfo, movie_path: &Path, frame: usize, backend: &str, context: usize) -> Result<i32> {
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
    let to = o.trace_frame(movie.inputs[frame - 1]);
    let tb = b.trace_frame(movie.inputs[frame - 1]);
    println!("oracle {} instructions, {} {} instructions", to.len(), backend, tb.len());
    let first = to.iter().zip(&tb).position(|(x, y)| !backend::same_instruction(x, y));
    let line = |r: &backend::TraceRow| {
        format!(
            "PC:{:04X} OP:{:02X} A:{:02X} X:{:02X} Y:{:02X} P:{:02X} SP:{:02X} SL:{:4} CYC:{}",
            r.pc, r.opcode, r.a, r.x, r.y, r.p, r.sp, r.scanline, r.cycle
        )
    };
    let first_sl = to.iter().zip(&tb).position(|(x, y)| x.scanline != y.scanline);
    if let Some(i) = first_sl {
        println!("first scanline difference at row {i}: oracle {} vs {} {}", to[i].scanline, backend, tb[i].scanline);
    }
    match first {
        None if to.len() == tb.len() => {
            println!("traces identical");
            Ok(0)
        }
        None => {
            println!("traces agree on {} rows, then lengths differ", to.len().min(tb.len()));
            Ok(1)
        }
        Some(i) => {
            println!("first differing instruction at row {i}");
            let lo = i.saturating_sub(context);
            for j in lo..(i + 2).min(to.len().min(tb.len())) {
                println!("{:7} fceumm {}", j, line(&to[j]));
                println!("{:7} {:6} {}", j, backend, line(&tb[j]));
            }
            Ok(1)
        }
    }
}

fn bench(rom: &rom::RomInfo, backend: &str, frames: usize) -> Result<i32> {
    let movie = Movie::load(&corpus::movie_path("power_on_to_start"))?;
    let mut b = make_backend(backend, rom)?;
    backend::play_movie(b.as_mut(), &movie)?;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let t = std::time::Instant::now();
    for _ in 0..frames {
        b.run_frame(rng.random::<u8>() & !0x0C);
    }
    let dt = t.elapsed().as_secs_f64();
    println!("{backend}: {:.0} frames/s (render on, 1 core)", frames as f64 / dt);
    Ok(0)
}

