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
    /// zcore frames per second on one core, render on and off: random input on the start screen, or with
    /// `--movie`/`--agent` timed windows spread evenly over each corpus movie's whole length.
    Bench {
        #[arg(long, default_value_t = 20_000)]
        frames: usize,
        /// Corpus movie names to replay instead (e.g. chain-milestone1-137); repeatable.
        #[arg(long = "movie")]
        movies: Vec<String>,
        /// Replay every corpus movie tagged `agent` (recorded chain runs of the trained models).
        #[arg(long)]
        agent: bool,
        /// Replay every `*.json` recording in this directory (`scripts/record_bench_runs.py`: chain runs of the
        /// current team recorded on the current core, which the corpus agent movies predate).
        #[arg(long)]
        recordings: Option<PathBuf>,
        /// With movies: timed windows per movie, evenly spaced from its first to its last frame.
        #[arg(long, default_value_t = 10)]
        windows: usize,
        /// With movies: frames per timed window.
        #[arg(long, default_value_t = 500)]
        window_frames: usize,
        /// Window-start states, made by an untimed replay the first time (keyed by the state format and a digest
        /// of the run's origin, inputs and pokes).
        #[arg(long, default_value = "target/bench-cache")]
        cache: PathBuf,
        /// Only this render mode ("on" or "off"); default both.
        #[arg(long)]
        render: Option<String>,
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
        Cmd::Bench { frames, mut movies, agent, recordings, windows, window_frames, cache, render } => {
            if agent {
                movies.extend(corpus::Manifest::load()?.movie.into_iter()
                    .filter(|e| e.tags.iter().any(|t| t == "agent")).map(|e| e.name));
            }
            let mut runs = movies.iter().map(|m| BenchRun::from_movie(m)).collect::<Result<Vec<_>>>()?;
            if let Some(dir) = recordings {
                let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)?.filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|e| e == "json")).collect();
                paths.sort();
                for p in paths {
                    runs.push(BenchRun::from_recording(&p)?);
                }
            }
            let w = Windows { count: windows.max(1), frames: window_frames.max(1), cache };
            bench(&rom, frames, &runs, &w, render.as_deref())
        }
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

/// Timed windows of a movie bench.
struct Windows {
    count: usize,
    frames: usize,
    cache: PathBuf,
}

impl Windows {
    /// First frames of the windows of a movie `len` frames long: evenly spaced, the last ending on its last frame.
    fn starts(&self, len: usize) -> Vec<usize> {
        if len <= self.count * self.frames {
            return vec![0];
        }
        let span = len - self.frames;
        (0..self.count).map(|i| if self.count == 1 { 0 } else { i * span / (self.count - 1) }).collect()
    }
}

/// A gameplay run to bench: where it starts, one controller byte per frame, and the pokes made before frame k.
struct BenchRun {
    name: String,
    origin: Origin,
    inputs: Vec<u8>,
    /// `(frame, addr, value)`, frames non-decreasing: written before input `frame` runs.
    pokes: Vec<(usize, u16, u8)>,
    /// Digest of everything the run's frames depend on (origin, inputs, pokes): part of the window cache key, so
    /// a re-recorded run with the same name misses the old run's cached states.
    digest: u64,
}

enum Origin {
    /// A corpus movie: power on and play its base movie.
    Base(Movie),
    /// A recording (`scripts/record_bench_runs.py`): a TFZSTATE start state.
    State(Vec<u8>),
}

impl BenchRun {
    fn from_movie(name: &str) -> Result<BenchRun> {
        let movie = Movie::load(&corpus::movie_path(name))?;
        let pokes = movie.pokes.iter().map(|p| (p.frame as usize, p.addr, p.value)).collect();
        let inputs = movie.inputs.clone();
        let chain = formats::movie_chain_hash(&corpus::movie_path(name), &corpus::corpus_dir().join("movies"))?;
        let digest = formats::digest8(&chain);
        Ok(BenchRun { name: name.to_owned(), origin: Origin::Base(movie), inputs, pokes, digest })
    }

    /// A recording: JSON with `origin` (state file path), `buttons` and `pokes` (`[frame, addr, value]`).
    fn from_recording(path: &Path) -> Result<BenchRun> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let origin = std::fs::read(v["origin"].as_str().context("recording origin")?)?;
        let inputs: Vec<u8> = v["buttons"].as_array().context("buttons")?.iter()
            .map(|b| b.as_u64().map(|b| b as u8).context("button byte")).collect::<Result<_>>()?;
        let pokes: Vec<(usize, u16, u8)> = v["pokes"].as_array().context("pokes")?.iter().map(|p| -> Result<_> {
            let p = p.as_array().context("poke")?;
            let n = |i: usize| p.get(i).and_then(serde_json::Value::as_u64).context("poke field");
            Ok((n(0)? as usize, n(1)? as u16, n(2)? as u8))
        }).collect::<Result<_>>()?;
        let name = path.file_stem().context("recording name")?.to_string_lossy().into_owned();
        let mut bytes = origin.clone();
        bytes.extend_from_slice(&inputs);
        for &(f, a, v) in &pokes {
            bytes.extend_from_slice(&(f as u64).to_le_bytes());
            bytes.extend_from_slice(&a.to_le_bytes());
            bytes.push(v);
        }
        let digest = formats::digest8(&bytes);
        Ok(BenchRun { name, origin: Origin::State(origin), inputs, pokes, digest })
    }

    /// A machine at the run's frame 0.
    fn start(&self, rom: &rom::RomInfo) -> Result<zcore::Zcore> {
        let mut b = zcore::Zcore::open(&rom.bytes)?;
        b.set_render(false);
        match &self.origin {
            Origin::Base(movie) => replay::play_base(&mut b, movie)?,
            Origin::State(bytes) => b.load_state(bytes)?,
        }
        Ok(b)
    }

    /// Runs frames `from..to`, making the pokes due before each.
    fn play(&self, b: &mut zcore::Zcore, from: usize, to: usize) {
        let mut p = self.pokes.partition_point(|&(f, _, _)| f < from);
        for k in from..to {
            while p < self.pokes.len() && self.pokes[p].0 == k {
                b.poke(self.pokes[p].1, self.pokes[p].2);
                p += 1;
            }
            b.run_frame(self.inputs[k]);
        }
    }
}

/// The state at `start` of `run`, from the cache or else by replaying every window start of the run once and
/// caching them.
fn window_state(rom: &rom::RomInfo, run: &BenchRun, starts: &[usize], start: usize, w: &Windows)
    -> Result<Vec<u8>> {
    // Keyed by the state format and the machine's layout, so a cached start never decodes into another build's,
    // and by the run's digest, so a changed run with the same name never starts from the old run's states.
    let key = format!("v{}-{}", ::zcore::state_file::VERSION, std::mem::size_of::<::zcore::MachineState>());
    let path = |s: usize| w.cache.join(format!("{key}-{}-{:016x}-{s}.tfzstate", run.name, run.digest));
    if let Ok(bytes) = std::fs::read(path(start)) {
        let mut probe = zcore::Zcore::open(&rom.bytes)?;
        if probe.load_state(&bytes).is_ok() {
            return Ok(bytes);
        }
    }
    std::fs::create_dir_all(&w.cache)?;
    let mut b = run.start(rom)?;
    let mut k = 0usize;
    for &s in starts {
        run.play(&mut b, k, s);
        k = s;
        std::fs::write(path(s), b.save_state())?;
    }
    Ok(std::fs::read(path(start))?)
}

/// The same seeded random input from the end of `power_on_to_start`, once drawing every frame and
/// once drawing none; or timed windows spread over the given runs, reported overall and by the level
/// (overworld 0, dungeons 1-9) at each window's start.
fn bench(rom: &rom::RomInfo, frames: usize, runs: &[BenchRun], w: &Windows, only: Option<&str>) -> Result<i32> {
    let modes: Vec<bool> = match only {
        Some("on") => vec![true],
        Some("off") => vec![false],
        None => vec![true, false],
        Some(other) => anyhow::bail!("--render must be on or off, got {other:?}"),
    };
    for render in modes {
        let mode = if render { "on" } else { "off" };
        if runs.is_empty() {
            let movie = Movie::load(&corpus::movie_path("power_on_to_start"))?;
            let mut b = zcore::Zcore::open(&rom.bytes)?;
            replay::play_movie(&mut b, &movie)?;
            b.set_render(render);
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
            let t = std::time::Instant::now();
            for _ in 0..frames {
                b.run_frame(rng.random::<u8>() & !0x0C);
            }
            let dt = t.elapsed().as_secs_f64();
            println!("zcore: {:.0} frames/s (render {mode}, 1 core)", frames as f64 / dt);
            continue;
        }
        // level -> (frames, seconds)
        let mut by_level: std::collections::BTreeMap<u8, (usize, f64)> = Default::default();
        for run in runs {
            let starts = w.starts(run.inputs.len());
            for &start in &starts {
                let state = window_state(rom, run, &starts, start, w)?;
                let mut b = zcore::Zcore::open(&rom.bytes)?;
                b.load_state(&state)?;
                b.set_render(render);
                let level = b.ram()[0x10];
                let end = (start + w.frames).min(run.inputs.len());
                let t = std::time::Instant::now();
                run.play(&mut b, start, end);
                let entry = by_level.entry(level).or_default();
                entry.0 += end - start;
                entry.1 += t.elapsed().as_secs_f64();
            }
        }
        let (total, seconds) = by_level.values().fold((0, 0.0), |(f, s), &(a, b)| (f + a, s + b));
        let levels: Vec<String> = by_level.iter()
            .map(|(l, (f, s))| format!("L{l} {:.0} ({f} frames)", *f as f64 / s)).collect();
        println!("zcore: {:.0} frames/s (render {mode}, {} runs, {total} frames in windows, 1 core) | {}",
                 total as f64 / seconds, runs.len(), levels.join(", "));
    }
    Ok(0)
}
