//! `tf`: movie, zcore self-diff, trace, corpus and benchmark tooling for zcore.

mod corpus;
use movie as formats;
mod replay;
mod rom;
mod script;
mod zcore;

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use formats::Movie;
use rand::{RngExt, SeedableRng};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "tf", about = "triforce zcore movie, regression and benchmark tools")]
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
    /// Write all-domain zcore goldens (the regression baseline for `ladder-diff`).
    LadderGolden {
        /// Movies to replay (default: every corpus movie).
        movies: Vec<PathBuf>,
        #[arg(short, long)]
        out: PathBuf,
        /// Only the corpus movies tagged `short` (the committed `corpus/golden/` set).
        #[arg(long)]
        short: bool,
        #[arg(long)]
        jobs: Option<usize>,
    },
    /// Replay the movie of every golden in DIR on zcore; exit 0 or print each movie's first divergence.
    LadderDiff {
        /// Golden directory (default: the committed corpus/golden).
        dir: Option<PathBuf>,
        #[arg(long)]
        jobs: Option<usize>,
    },
    /// Print every instruction zcore runs in records FRAME..FRAME+COUNT-1 of a movie (diff two builds' output).
    Trace {
        movie: PathBuf,
        /// 1-based record number of the first traced frame.
        #[arg(long)]
        frame: usize,
        #[arg(long, default_value_t = 1)]
        count: usize,
    },
    /// Corpus manifest and coverage gate.
    Corpus {
        #[command(subcommand)]
        cmd: CorpusCmd,
    },
    /// zcore frames per second on one core, render on and off.
    Bench {
        #[arg(long, default_value_t = 20_000)]
        frames: usize,
    },
}

#[derive(Subcommand)]
enum MovieCmd {
    /// Record a script on zcore.
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
        Cmd::LadderGolden { movies, out, short, jobs: j } => corpus::ladder_golden(&rom, &movies, short, &out, jobs(j)),
        Cmd::LadderDiff { dir, jobs: j } => corpus::ladder_diff(&rom, &dir.unwrap_or_else(corpus::golden_dir), jobs(j)),
        Cmd::Trace { movie, frame, count } => trace(&rom, &movie, frame, count),
        Cmd::Corpus { cmd } => match cmd {
            CorpusCmd::Sync => corpus::sync(),
            CorpusCmd::Check => corpus::check(&rom),
        },
        Cmd::Bench { frames } => bench(&rom, frames),
    }
}

fn movie_cmd(cmd: MovieCmd, rom: &rom::RomInfo) -> Result<i32> {
    match cmd {
        MovieCmd::Script { script: path, out } => {
            let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
            let mut directives = script::parse(&text)?;
            let name = out.file_stem().unwrap().to_string_lossy().into_owned();
            let mut b = zcore::Zcore::open(&rom.bytes)?;
            let mut movie = Movie::new(
                rom.payload_sha1,
                serde_json::json!({"name": name, "tags": [], "source": {"script": path.file_name().unwrap().to_string_lossy()}}),
            );
            if let Some(script::Directive::Base(p)) = directives.first().cloned() {
                let base = Movie::load(&corpus::movie_path(&p))?;
                check_movie_rom(&base, rom, &path)?;
                replay::play_movie(&mut b, &base)?;
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
            let mut b = zcore::Zcore::open(&rom.bytes)?;
            replay::play_movie(&mut b, &p)?;
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

/// One line per instruction of records `frame..frame + count`, prefixed with the record number, so
/// two zcore builds' output can be compared with `diff`.
fn trace(rom: &rom::RomInfo, movie_path: &Path, frame: usize, count: usize) -> Result<i32> {
    ensure!(frame >= 1, "--frame is a 1-based record number");
    let movie = Movie::load(movie_path)?;
    check_movie_rom(&movie, rom, movie_path)?;
    let last = (frame + count.max(1) - 1).min(movie.inputs.len());
    ensure!(frame <= last, "movie has only {} frames", movie.inputs.len());
    let mut b = zcore::Zcore::open(&rom.bytes)?;
    b.set_render(false);
    replay::play_base(&mut b, &movie)?;
    let index = movie.poke_index();
    let mut next = 0;
    for k in 0..last {
        replay::apply_pokes(&mut b, &movie, &index, &mut next, k);
        if k + 1 < frame {
            b.run_frame(movie.inputs[k]);
            continue;
        }
        for r in b.trace_frame(movie.inputs[k]) {
            println!(
                "{:6} PC:{:04X} OP:{:02X} A:{:02X} X:{:02X} Y:{:02X} P:{:02X} SP:{:02X} SL:{:3} DOT:{:3} CYC:{}",
                k + 1,
                r.pc,
                r.opcode,
                r.a,
                r.x,
                r.y,
                r.p,
                r.sp,
                r.scanline,
                r.dot,
                r.cycle
            );
        }
    }
    Ok(0)
}

/// The same seeded random input from the end of `power_on_to_start`, once drawing every frame and
/// once drawing none.
fn bench(rom: &rom::RomInfo, frames: usize) -> Result<i32> {
    let movie = Movie::load(&corpus::movie_path("power_on_to_start"))?;
    for render in [true, false] {
        let mut b = zcore::Zcore::open(&rom.bytes)?;
        replay::play_movie(&mut b, &movie)?;
        b.set_render(render);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let t = std::time::Instant::now();
        for _ in 0..frames {
            b.run_frame(rng.random::<u8>() & !0x0C);
        }
        let dt = t.elapsed().as_secs_f64();
        let mode = if render { "on" } else { "off" };
        println!("zcore: {:.0} frames/s (render {mode}, 1 core)", frames as f64 / dt);
    }
    Ok(0)
}
