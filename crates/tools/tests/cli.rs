//! End-to-end runs of the `tf` binary on the real ROM and the committed corpus. Every output goes
//! to a temporary directory; the corpus is only read.

use movie::{Golden, Movie};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Arc;
use tfreplay::{Context, Ending, RecordOptions, Recorder, Replay};
use zcore::{Rom, ZCore, state_file};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Whether `tf` can find the ROM. Missing: panics (`test_rom`), or false with `TRIFORCE_ALLOW_NO_ROM=1`.
fn have_rom() -> bool {
    test_rom::path().is_some()
}

/// `tf` as a job of the machine's queue runs it (`$TRIFORCE_JOB_ID`, the job's id or 0 as in the Python tests): its
/// heavy subcommands refuse to run outside one without a terminal, and a test's output is no terminal.
fn tf_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_tf"));
    command.env("TRIFORCE_JOB_ID", std::env::var("TRIFORCE_JOB_ID").unwrap_or_else(|_| "0".to_owned()));
    command.current_dir(repo());
    command
}

fn tf(args: &[&str]) -> Output {
    tf_command().args(args).output().expect("tf runs")
}

/// Outside a job and without a terminal the heavy subcommands refuse (exit 2) with the `triforce job run` form of
/// the command, before they read anything; the light ones (one replay verified) run.
#[test]
fn heavy_subcommands_run_only_as_jobs() {
    let outside = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_tf")).args(args).env_remove("TRIFORCE_JOB_ID").current_dir(repo())
            .stdin(std::process::Stdio::null()).output().expect("tf runs")
    };
    for args in [&["ladder-diff"][..], &["ladder-golden", "--short", "-o", "x y"], &["corpus", "check"],
                 &["replay", "verify", "a.tfreplay", "b.tfreplay"], &["replay", "verify", "corpus/replays"],
                 &["bench", "--agent"]] {
        let out = outside(args);
        assert_eq!(out.status.code(), Some(2), "{args:?}: {}", stderr(&out));
        let text = stderr(&out);
        assert!(text.contains("runs under the machine's job queue"), "{args:?}: {text}");
        assert!(text.contains("triforce job run --agent AGENT --task TASK --subtask SUBTASK"), "{text}");
        assert!(text.contains(&format!(" -- {} {}", env!("CARGO_BIN_EXE_tf"), args[0])), "{text}");
    }
    assert!(stderr(&outside(&["ladder-golden", "--short", "-o", "x y"])).ends_with(" -o 'x y'\n"));
    let out = outside(&["replay", "verify", "missing.tfreplay"]);
    assert!(!stderr(&out).contains("job queue"), "{}", stderr(&out));
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// A fresh directory under the system temp dir, removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> TempDir {
        let dir = std::env::temp_dir().join(format!("tf-cli-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    fn arg(&self, name: &str) -> String {
        self.join(name).to_string_lossy().into_owned()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `movie script` records the frames and pokes of a script after its base, checking its asserts
/// as it goes, and `movie info` reads the result back.
#[test]
fn movie_script_records_what_info_reports() {
    if !have_rom() {
        return;
    }
    let tmp = TempDir::new("script");
    std::fs::write(
        tmp.join("walk.txt"),
        "base: power_on_to_start\ntags: e2e, tiny\nassert ram[0x12] == 0x05  # in play\nRIGHT 8\npoke 0x10 0x00\n- 2\n",
    )
    .unwrap();
    let out = tf(&["movie", "script", &tmp.arg("walk.txt"), "-o", &tmp.arg("walk.tfmovie")]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(stdout(&out).contains("10 frames, 1 pokes"), "{}", stdout(&out));

    let m = Movie::load(&tmp.join("walk.tfmovie")).unwrap();
    assert_eq!(m.inputs, [[0x80; 8].as_slice(), &[0; 2]].concat());
    assert_eq!((m.pokes[0].frame, m.pokes[0].addr), (8, 0x10));
    assert_eq!(m.base(), Some("power_on_to_start"));
    assert_eq!(m.tags(), ["e2e", "tiny"]);

    let info = tf(&["movie", "info", &tmp.arg("walk.tfmovie")]);
    assert!(info.status.success(), "{}", stderr(&info));
    let text = stdout(&info);
    for line in ["name:   walk", "frames: 10", "pokes:  1", "rom:    a12d74c73a0481599a5d832361d168f4737bbcf6"] {
        assert!(text.contains(line), "{line:?} missing from\n{text}");
    }
}

/// A failed assert fails the command with the address, expected value and frame, and writes no
/// movie.
#[test]
fn a_failing_script_assert_writes_no_movie() {
    if !have_rom() {
        return;
    }
    let tmp = TempDir::new("assert");
    std::fs::write(tmp.join("bad.txt"), "base: power_on_to_start\n- 3\nassert ram[0x12] == 0x06\n").unwrap();
    let out = tf(&["movie", "script", &tmp.arg("bad.txt"), "-o", &tmp.arg("bad.tfmovie")]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("assert ram[$0012] == 0x06 failed after frame 3"), "{}", stderr(&out));
    assert!(!tmp.join("bad.tfmovie").exists());

    let info = tf(&["movie", "info", &tmp.arg("bad.txt")]);
    assert_eq!(info.status.code(), Some(2));
    assert!(stderr(&info).contains("not a TFMOVIE"), "{}", stderr(&info));
}

/// `movie random` is a pure function of its seed, and `movie rebase` strips the base's frames
/// from a power-on movie, shifting its pokes.
#[test]
fn random_is_seeded_and_rebase_strips_the_base() {
    if !have_rom() {
        return;
    }
    let tmp = TempDir::new("random");
    for (name, seed) in [("a", "7"), ("b", "7"), ("c", "8")] {
        let out = tf(&["movie", "random", "--seed", seed, "--frames", "40", "-o", &tmp.arg(&format!("{name}.tfmovie"))]);
        assert!(out.status.success(), "{}", stderr(&out));
    }
    let load = |name: &str| Movie::load(&tmp.join(&format!("{name}.tfmovie"))).unwrap();
    let (a, b, c) = (load("a"), load("b"), load("c"));
    assert_eq!(a.inputs.len(), 40);
    assert_eq!(a.base(), Some("power_on_to_start"));
    assert_eq!(a.inputs, b.inputs, "same seed, same movie");
    assert_ne!(a.inputs, c.inputs, "another seed, another movie");

    // A power-on movie: power_on_to_start's frames, then three of its own and a poke.
    let base = Movie::load(&repo().join("corpus/movies/power_on_to_start.tfmovie")).unwrap();
    let mut full = Movie::new(base.rom_sha1, serde_json::json!({"name": "full"}));
    full.inputs = base.inputs.clone();
    full.inputs.extend([0x01, 0x02]);
    full.push_poke(0x6000, 9).unwrap();
    full.inputs.push(0x04);
    full.save(&tmp.join("full.tfmovie")).unwrap();
    let out = tf(&["movie", "rebase", &tmp.arg("full.tfmovie"), "-o", &tmp.arg("rebased.tfmovie")]);
    assert!(out.status.success(), "{}", stderr(&out));
    let r = load("rebased");
    assert_eq!(r.inputs, [0x01, 0x02, 0x04]);
    assert_eq!((r.pokes[0].frame, r.pokes[0].addr), (2, 0x6000));
    assert_eq!(r.base(), Some("power_on_to_start"));

    // Rebasing again, or a movie that doesn't start with the base, is refused.
    let out = tf(&["movie", "rebase", &tmp.arg("rebased.tfmovie"), "-o", &tmp.arg("x.tfmovie")]);
    assert!(stderr(&out).contains("already has base"), "{}", stderr(&out));
    full.inputs[0] ^= 1;
    full.save(&tmp.join("full.tfmovie")).unwrap();
    let out = tf(&["movie", "rebase", &tmp.arg("full.tfmovie"), "-o", &tmp.arg("x.tfmovie")]);
    assert!(stderr(&out).contains("does not start with power_on_to_start"), "{}", stderr(&out));
}

/// `--out` without a file name is an error (exit 2), not a panic.
#[test]
fn movie_out_needs_a_file_name() {
    if !have_rom() {
        return;
    }
    let tmp = TempDir::new("outname");
    let out = tf(&["movie", "random", "--seed", "1", "--frames", "1", "-o", &format!("{}/..", tmp.0.display())]);
    assert_eq!(out.status.code(), Some(2), "{}", stderr(&out));
    assert!(stderr(&out).contains("needs a file name"), "{}", stderr(&out));
}

/// Fresh goldens of the short movies replay without divergence; a golden whose digest was changed
/// reports the record and domain, one recorded for another version of its movie is stale, and
/// either makes `ladder-diff` exit 1.
#[test]
fn ladder_diff_passes_fresh_goldens_and_reports_tampered_ones() {
    if !have_rom() {
        return;
    }
    let tmp = TempDir::new("ladder");
    let dir = tmp.arg("golden");
    let out = tf(&["ladder-golden", "--short", "-o", &dir]);
    assert!(out.status.success(), "{}{}", stdout(&out), stderr(&out));
    let shorts = std::fs::read_dir(tmp.join("golden")).unwrap().count();
    assert!(stdout(&out).contains(&format!("wrote {shorts} of {shorts} goldens")), "{}", stdout(&out));

    let out = tf(&["ladder-diff", &dir]);
    assert_eq!(out.status.code(), Some(0), "{}", stdout(&out));
    assert!(stdout(&out).contains(&format!("0 of {shorts} movies diverge")), "{}", stdout(&out));

    let path = tmp.join("golden/pause_spam.tfgolden");
    let mut g = Golden::load(&path).unwrap();
    g.records[100][0] ^= 1; // RAM digest after input 99
    g.save(&path).unwrap();
    let path = tmp.join("golden/title_idle_3600.tfgolden");
    let mut g = Golden::load(&path).unwrap();
    g.movie_hash[0] ^= 1;
    g.save(&path).unwrap();

    let out = tf(&["ladder-diff", &dir]);
    let text = stdout(&out);
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(text.contains("pause_spam: first divergence at frame 100: ram\n"), "{text}");
    assert!(text.contains("title_idle_3600.tfgolden is stale"), "{text}");
    assert!(text.contains(&format!("2 of {shorts} movies diverge")), "{text}");
}

/// The build under test replays the committed goldens (`corpus/golden/`) without divergence, and
/// that set is exactly the manifest's `short` movies (a deleted golden would otherwise go unchecked).
#[test]
fn the_committed_goldens_match_this_build() {
    if !have_rom() {
        return;
    }
    let manifest = std::fs::read_to_string(repo().join("corpus/manifest.toml")).unwrap();
    let manifest: toml::Value = toml::from_str(&manifest).unwrap();
    let mut shorts: Vec<String> = manifest["movie"].as_array().unwrap().iter()
        .filter(|e| e["tags"].as_array().unwrap().iter().any(|t| t.as_str() == Some("short")))
        .map(|e| e["name"].as_str().unwrap().to_owned())
        .collect();
    shorts.sort();
    let mut goldens: Vec<String> = std::fs::read_dir(repo().join("corpus/golden")).unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "tfgolden"))
        .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    goldens.sort();
    assert_eq!(goldens, shorts);

    let out = tf(&["ladder-diff"]);
    assert_eq!(out.status.code(), Some(0), "{}{}", stdout(&out), stderr(&out));
    assert!(stdout(&out).contains(&format!("0 of {} movies diverge", shorts.len())), "{}", stdout(&out));
}

/// `corpus check` measures coverage by replaying the movies: a corpus of only the short movies and
/// one tagged `power-on-to-d2` that never gets there has no gameplay in Dungeons 2 and 3 and fails.
#[test]
fn corpus_check_fails_without_dungeon_2_gameplay() {
    if !have_rom() {
        return;
    }
    let tmp = TempDir::new("corpus-check");
    std::fs::create_dir_all(tmp.join("movies")).unwrap();
    let src = repo().join("corpus/movies");
    for e in std::fs::read_dir(&src).unwrap() {
        let p = e.unwrap().path();
        let m = Movie::load(&p).unwrap();
        if m.tags().iter().any(|t| t == "short") {
            std::fs::copy(&p, tmp.join("movies").join(p.file_name().unwrap())).unwrap();
        }
    }
    // A movie claiming Dungeon 2 by its tag, but playing only the base's last frames.
    let mut liar = Movie::new(Movie::load(&src.join("pause_spam.tfmovie")).unwrap().rom_sha1,
        serde_json::json!({"name": "liar", "base": "power_on_to_start", "tags": ["power-on-to-d2"]}));
    liar.inputs = vec![0; 10];
    liar.save(&tmp.join("movies/liar.tfmovie")).unwrap();
    let corpus = |args: &[&str]| {
        tf_command().args(args).env("TF_CORPUS", &tmp.0).output().unwrap()
    };
    let out = corpus(&["corpus", "sync"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let out = corpus(&["corpus", "check"]);
    let text = stdout(&out);
    assert_eq!(out.status.code(), Some(1), "{text}");
    assert!(text.contains("level 2: 0 gameplay frames (need 10000) FAIL"), "{text}");
    assert!(text.contains("corpus check FAILED"), "{text}");
}

/// `ladder-golden` takes explicit movies only from corpus/movies (where `ladder-diff` looks for a
/// golden's movie); a movie elsewhere is refused with a message naming the corpus, and a corpus
/// movie given by path works.
#[test]
fn ladder_golden_refuses_movies_outside_the_corpus() {
    if !have_rom() {
        return;
    }
    let tmp = TempDir::new("ladder-ext");
    let out = tf(&["movie", "random", "--seed", "3", "--frames", "50", "-o", &tmp.arg("rnd.tfmovie")]);
    assert!(out.status.success(), "{}", stderr(&out));
    let out = tf(&["ladder-golden", &tmp.arg("rnd.tfmovie"), "-o", &tmp.arg("g")]);
    assert_ne!(out.status.code(), Some(0));
    assert!(stderr(&out).contains("is not a corpus movie"), "{}", stderr(&out));
    assert!(stderr(&out).contains("corpus/movies"), "{}", stderr(&out));
    // Same stem as a corpus movie, but another file: refused too.
    std::fs::copy(tmp.join("rnd.tfmovie"), tmp.join("pause_spam.tfmovie")).unwrap();
    let out = tf(&["ladder-golden", &tmp.arg("pause_spam.tfmovie"), "-o", &tmp.arg("g")]);
    assert!(stderr(&out).contains("is not a corpus movie"), "{}", stderr(&out));

    let out = tf(&["ladder-golden", "corpus/movies/pause_spam.tfmovie", "-o", &tmp.arg("g")]);
    assert!(out.status.success(), "{}{}", stdout(&out), stderr(&out));
    let out = tf(&["ladder-diff", &tmp.arg("g")]);
    assert_eq!(out.status.code(), Some(0), "{}", stdout(&out));
}

const RECORDING_START: &str = "triforce/states/0_77c_0.state";

/// A recorded run as `scripts/record_bench_runs.py` writes it: a TFREPLAY that embeds its start state
/// (`0_77c_0`), here 60 frames of `button`.
fn recording(button: u8) -> Vec<u8> {
    let rom = Rom::from_ines(&test_rom::bytes().unwrap()).unwrap();
    let start = std::fs::read(repo().join(RECORDING_START)).unwrap();
    let mut nes = ZCore::power_on(rom.clone());
    nes.restore(&state_file::decode(&rom, &start).unwrap().machine);
    let meta = serde_json::json!({"triforce": "test", "segment": "full", "models": [
        {"name": "ow-start", "act": "temperature:0.75+relax:16"}, {"name": "ow-octorok", "act": "argmax"},
        {"name": "goriya", "act": "temperature:0.75+relax:16"}]});
    let options = RecordOptions { meta, ..Default::default() };
    let mut rec = Recorder::new(Arc::new(options));
    rec.embed_start(Some("0_77c_0"), Context::default(), &start);
    for _ in 0..60 {
        nes.run_frame(button, false);
        rec.frame(button);
    }
    rec.finish(500, Ending::default(), Context::default(), &state_file::encode(&rom, &nes.snapshot(), 0))
}

/// `bench --recordings` caches window-start states per run content: re-recording a run under the
/// same name with other inputs makes new cache files instead of reusing the old run's states.
#[test]
fn bench_cache_is_keyed_by_the_run_content() {
    if !have_rom() {
        return;
    }
    let tmp = TempDir::new("bench-cache");
    let record = |button: u8| {
        std::fs::create_dir_all(tmp.join("rec")).unwrap();
        std::fs::write(tmp.join("rec/full-500.tfreplay"), recording(button)).unwrap();
        let out = tf(&[
            "bench", "--recordings", &tmp.arg("rec"), "--windows", "2", "--window-frames", "10", "--cache",
            &tmp.arg("cache"), "--render", "off",
        ]);
        assert!(out.status.success(), "{}{}", stdout(&out), stderr(&out));
    };
    let files = || -> Vec<(String, Vec<u8>)> {
        let mut v: Vec<_> = std::fs::read_dir(tmp.join("cache")).unwrap()
            .map(|e| e.unwrap().path())
            .map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), std::fs::read(&p).unwrap()))
            .collect();
        v.sort();
        v
    };
    record(0x80); // RIGHT
    let first = files();
    assert_eq!(first.len(), 2, "one state per window start");
    record(0x80);
    assert_eq!(files(), first, "the same run reuses its cache");
    record(0x40); // LEFT
    let both = files();
    assert_eq!(both.len(), 4, "a changed run gets its own window states");
    let late = |v: &[(String, Vec<u8>)], suffix: &str| -> Vec<Vec<u8>> {
        v.iter().filter(|(n, _)| n.ends_with(suffix)).map(|(_, b)| b.clone()).collect()
    };
    let ends = late(&both, "-50.tfzstate");
    assert_eq!(ends.len(), 2);
    assert_ne!(ends[0], ends[1], "the late window starts from each run's own state");
}

/// `movie import` turns a recording into a corpus movie that starts from a copy of the recording's start
/// state and plays to the machine the recording ended on; a recording the core no longer reproduces, or one
/// without its start state, is refused.  `replay info` and `replay verify` read the recording itself.
#[test]
fn movie_import_writes_a_movie_that_plays_to_the_recorded_end() {
    if !have_rom() {
        return;
    }
    let tmp = TempDir::new("import");
    std::fs::create_dir_all(tmp.join("movies")).unwrap();
    let bytes = recording(0x80);
    std::fs::write(tmp.join("full-500.tfreplay"), &bytes).unwrap();
    let out = tf(&["replay", "verify", &tmp.arg("full-500.tfreplay")]);
    assert_eq!(out.status.code(), Some(0), "{}{}", stdout(&out), stderr(&out));
    assert!(stdout(&out).ends_with("full-500.tfreplay: ok, 60 frames, 0 edits, 2 states\n"), "{}", stdout(&out));
    let out = tf(&["replay", "info", &tmp.arg("full-500.tfreplay")]);
    let info = stdout(&out);
    for line in ["frames:   60", "source:   embedded state 0_77c_0", "triforce: test", "states:   1 START, 1 END",
                 "act:      argmax: ow-octorok; temperature:0.75+relax:16: ow-start, goriya"] {
        assert!(info.contains(line), "{line:?} missing from:\n{info}");
    }

    let out = tf(&["movie", "import", &tmp.arg("full-500.tfreplay"), "-o", &tmp.arg("movies/chain-full-500.tfmovie")]);
    assert!(out.status.success(), "{}{}", stdout(&out), stderr(&out));
    let movie = Movie::load(&tmp.join("movies/chain-full-500.tfmovie")).unwrap();
    assert_eq!((movie.inputs.len(), movie.pokes.len()), (60, 0));
    assert_eq!(movie.meta["tags"], serde_json::json!(["agent", "chain", "full"]));
    assert_eq!(movie.meta["source"]["seed"], 500);
    let origin = movie.meta["origin"].as_str().unwrap();
    assert!(origin.starts_with("0_77c_0-"), "{origin}");
    let start = std::fs::read(movie::origin_path(&tmp.join("movies"), origin)).unwrap();
    assert_eq!(start, std::fs::read(repo().join(RECORDING_START)).unwrap());
    let rom = Rom::from_ines(&test_rom::bytes().unwrap()).unwrap();
    let mut nes = ZCore::power_on(rom.clone());
    nes.restore(&state_file::decode(&rom, &start).unwrap().machine);
    for &buttons in &movie.inputs {
        nes.run_frame(buttons, false);
    }
    let replay = Replay::parse(bytes.clone()).unwrap();
    let end = state_file::decode(&rom, replay.end_state().blob).unwrap().machine;
    assert!(state_file::machines_equal(&end, &nes.snapshot()), "the movie ends on the recorded machine");

    // The same inputs recorded against another end state: the core does not reproduce the recording.
    let other = Replay::parse(recording(0x40)).unwrap();
    let stitched = {
        let mut rec = Recorder::new(Arc::new(RecordOptions::default()));
        rec.embed_start(Some("0_77c_0"), Context::default(), replay.start_state().unwrap().blob);
        for buttons in replay.inputs() {
            rec.frame(buttons);
        }
        rec.finish(500, Ending::default(), Context::default(), other.end_state().blob)
    };
    std::fs::write(tmp.join("desync.tfreplay"), stitched).unwrap();
    let out = tf(&["movie", "import", &tmp.arg("desync.tfreplay"), "-o", &tmp.arg("movies/chain-desync.tfmovie")]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("the recording desyncs"), "{}", stderr(&out));
    assert!(!tmp.join("movies/chain-desync.tfmovie").exists());
    let out = tf(&["replay", "verify", &tmp.arg("full-500.tfreplay"), &tmp.arg("desync.tfreplay")]);
    assert_eq!(out.status.code(), Some(1), "{}", stdout(&out));
    assert!(stdout(&out).contains("desync.tfreplay: diverged at frame 60"), "{}", stdout(&out));
}

/// The compatibility corpus (`corpus/replays/`; AGENTS.md "Versioning"): every replay recorded by an earlier
/// build still plays on this one to the states it recorded.  It holds a power-on start with every optional
/// stream (`v0-power-on`), a state-file start (`v0-state`), and the complete runs under names that say what they
/// are: `golden-d2first-to-dungeon3-triforce` (Nov 11's golden path to the Dungeon 3 triforce),
/// `golden-d2first-34-networks` (1.9.0.0), `golden-d3first-51-networks` (the owner's corrected order with its two
/// dock warps, his bomb rules and without 0:0x2d's cave: recorded on 2.0.0.12, 153,136 frames, 2 cheated bombs),
/// `golden-d3first-single-network` (one network in every slot, 2.0.0.11), `single-run-d3first-old-order-51-networks`
/// and `planner-d2first`.  A failure here means replays people kept no longer replay: read the AGENTS.md rule
/// before touching the files.
#[test]
fn the_replay_corpus_still_verifies() {
    if !have_rom() {
        return;
    }
    let mut files: Vec<String> = std::fs::read_dir(repo().join("corpus/replays")).unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".tfreplay"))
        .map(|name| format!("corpus/replays/{name}"))
        .collect();
    files.sort();
    for needed in ["golden-d2first-to-dungeon3-triforce", "v0-power-on", "v0-state", "golden-d2first-34-networks",
                   "golden-d3first-51-networks", "golden-d3first-single-network"] {
        assert!(files.contains(&format!("corpus/replays/{needed}.tfreplay")), "{needed} is missing: {files:?}");
    }
    let out = tf(&["replay", "verify", "corpus/replays"]);         // the folder: every *.tfreplay in it
    let text = stdout(&out);
    assert_eq!(out.status.code(), Some(0), "{text}{}", stderr(&out));
    for file in &files {
        assert!(text.contains(&format!("{file}: ok, ")), "{text}");
    }
    assert_eq!(text.lines().count(), files.len(), "{text}");

    let info = |name: &str| stdout(&tf(&["replay", "info", &format!("corpus/replays/{name}.tfreplay")]));
    let golden = info("golden-d2first-to-dungeon3-triforce");
    for line in ["version:  0", "ending:   win", "segments: 3", "states:   1 END, 2 CHECKPOINT", "took the level 3 triforce"] {
        assert!(golden.contains(line), "{line}: {golden}");
    }
    assert!(golden.contains("source:   power-on LINK ") && golden.contains("(0 edits before it)"), "{golden}");
    // Recorded before act modes: META says `deterministic`, read as argmax or sample.
    assert!(golden.lines().any(|l| l.starts_with("act:      ") && (l.contains("sample") || l.contains("argmax"))), "{golden}");
    let power_on = info("v0-power-on");
    for line in ["source:   power-on LINK 100", "KEYFRAME", " REWARDS ", " DECISIONS ", "(0 edits before it)"] {
        assert!(power_on.contains(line), "{line}: {power_on}");
    }
    let state = info("v0-state");
    for line in ["source:   embedded state 0_3cc_0", "1 START, 1 END", " REWARDS ", " DECISIONS "] {
        assert!(state.contains(line), "{line}: {state}");
    }
    let ending = info("golden-d3first-51-networks");
    for line in ["version:  1", "frames:   153136", "ending:   win", "segments: 9", "states:   1 END, 8 CHECKPOINT",
                 "triforce: 2.0.0.12", "the game ended in (level 9, room 0x32)"] {
        assert!(ending.contains(line), "{line}: {ending}");
    }
    // One network in every model slot (round 38's student after its correction round, with the shared hit head:
    // models/r38c/d1-all-w8; 2.0.0.11): a golden path of the corrected route from power on.
    let single = info("golden-d3first-single-network");
    for line in ["version:  1", "frames:   161250", "ending:   win", "segments: 9", "(every model, 54)", "triforce: 2.0.0.11",
                 "the game ended in (level 9, room 0x32)"] {
        assert!(single.contains(line), "{line}: {single}");
    }
}

#[test]
fn corpus_check_passes() {
    if !have_rom() {
        return;
    }
    let out = tf(&["corpus", "check"]);
    assert_eq!(out.status.code(), Some(0), "{}", stdout(&out));
    assert!(stdout(&out).contains("corpus check passed"), "{}", stdout(&out));
}

/// `trace` prints one row per instruction of exactly the requested records, numbered from 1.
#[test]
fn trace_prints_the_requested_records() {
    if !have_rom() {
        return;
    }
    let movie = "corpus/movies/power_on_to_start.tfmovie";
    let out = tf(&["trace", movie, "--frame", "3", "--count", "2"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let text = stdout(&out);
    let records: Vec<&str> = text.lines().map(|l| l.split_whitespace().next().unwrap()).collect();
    let rows_of = |r: &str| records.iter().filter(|&&x| x == r).count();
    // Roughly one instruction per 3 of a frame's 29,780 cycles.
    assert!(rows_of("3") > 5000 && rows_of("4") > 5000, "{} rows", records.len());
    assert_eq!(rows_of("3") + rows_of("4"), records.len(), "only records 3 and 4");
    assert!(records.windows(2).all(|w| w[0] <= w[1]), "records in order");
    let first = text.lines().next().unwrap();
    for field in ["PC:", "OP:", "A:", "X:", "Y:", "P:", "SP:", "SL:", "DOT:", "CYC:"] {
        assert!(first.contains(field), "{field} missing from {first:?}");
    }

    let out = tf(&["trace", movie, "--frame", "0"]);
    assert!(stderr(&out).contains("1-based"), "{}", stderr(&out));
    let out = tf(&["trace", movie, "--frame", "663"]);
    assert!(stderr(&out).contains("movie has only 662 frames"), "{}", stderr(&out));
    // Running into the end stops there.
    let out = tf(&["trace", movie, "--frame", "662", "--count", "5"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(stdout(&out).lines().all(|l| l.trim_start().starts_with("662 ")));
}

#[test]
fn a_missing_rom_is_refused_up_front() {
    let out = tf(&["--rom", "/nonexistent/zelda.nes", "corpus", "check"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("ROM not found"), "{}", stderr(&out));
}

/// `tf replay clip` (the owner, Nov 26: "show me just one room"): a run that is in its start room twice (before
/// and after the sword cave) gives two clips, each a replay that verifies, starts at the visit's first frame in
/// mode 5 there and ends at its last, and plays the source's frames machine for machine.
#[test]
fn replay_clip_writes_a_replay_for_each_visit() {
    if !have_rom() {
        return;
    }
    let tmp = TempDir::new("clip");
    let stem = "single-run-d3first-2.0.0.15-55-networks";
    let source = format!("corpus/replays/{stem}.tfreplay");
    let out = tf(&["replay", "clip", &source, "--room", "0:77", "--playable", "--out", &tmp.arg("clips")]);
    assert_eq!(out.status.code(), Some(0), "{}{}", stdout(&out), stderr(&out));
    assert!(stdout(&out).ends_with(&format!("2 clips of 0:0x77 written to {}\n", tmp.arg("clips"))), "{}", stdout(&out));
    let mut names: Vec<String> = std::fs::read_dir(tmp.join("clips")).unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    assert_eq!(names, [format!("{stem}_0x77_v1.tfreplay"), format!("{stem}_0x77_v2.tfreplay")]);

    let rom = Rom::from_ines(&test_rom::bytes().unwrap()).unwrap();
    let no_files = |name: &str| Err(std::io::Error::new(std::io::ErrorKind::NotFound, name.to_owned()));
    let replay = Replay::parse(std::fs::read(repo().join(&source)).unwrap()).unwrap();
    let in_play = |nes: &ZCore| {
        let ram = nes.ram();
        (ram[0x12], ram[0x10], ram[0xEB]) == (5, 0, 0x77)
    };
    for (visit, name) in names.iter().enumerate() {
        let path = tmp.join("clips").join(name);
        let out = tf(&["replay", "verify", &path.to_string_lossy()]);
        assert_eq!(out.status.code(), Some(0), "{}", stdout(&out));
        let clip = Replay::parse(std::fs::read(&path).unwrap()).unwrap();
        let meta = clip.meta().unwrap();
        let about = &meta["clip"];
        assert_eq!((&meta["kind"], &about["kind"]), (&serde_json::json!("clip"), &serde_json::json!("chain")));
        assert_eq!(about["source"], format!("{stem}.tfreplay"));
        assert_eq!((&about["room"], &about["visit"], &about["visits"]), (&serde_json::json!("0:0x77"), &serde_json::json!(visit + 1), &serde_json::json!(2)));
        assert_eq!((clip.ending_name(), &meta["ending"]["result"]), ("other", &serde_json::json!("left")));
        let (from, to) = (about["from"].as_u64().unwrap() as u32, about["to"].as_u64().unwrap() as u32);
        assert_eq!(clip.frames(), to - from);
        // The objective and the mask in force at `from` open the clip, at its frame 0 (the first visit starts before
        // the models take over, so before any).
        let targets = replay.targets().map(Result::unwrap).filter(|t| t.frame <= from).last();
        let first = clip.targets().next().unwrap().unwrap();
        match targets {
            Some(targets) => assert_eq!(first, tfreplay::Targets { frame: 0, ..targets }, "visit {}", visit + 1),
            None => assert!(visit == 0 && first.frame > 0, "visit {}: no objective before frame {from}", visit + 1),
        }
        let mask = replay.masks().map(Result::unwrap).filter(|m| m.frame <= from).last();
        let first = clip.masks().next().unwrap().unwrap();
        match mask {
            Some(mask) => assert_eq!(first, tfreplay::Mask { frame: 0, ..mask }, "visit {}", visit + 1),
            None => assert!(visit == 0 && first.frame > 0, "visit {}: no mask before frame {from}", visit + 1),
        }

        // The source just before the clip is not in play there; the clip's first and last frames are; its end is not.
        let mut source = tfreplay::Player::open(rom.clone(), replay.clone(), &no_files).unwrap();
        source.seek(from - 1).unwrap();
        assert!(!in_play(source.nes()), "visit {}: frame {} is in play", visit + 1, from - 1);
        assert!(source.step(false).unwrap());
        let mut cut = tfreplay::Player::open(rom.clone(), clip.clone(), &no_files).unwrap();
        assert!(in_play(cut.nes()), "visit {}: the clip starts out of play", visit + 1);
        assert_eq!(cut.nes().ram()[..], source.nes().ram()[..]);
        while cut.frame() < cut.frames() {
            if cut.frame() + 1 == cut.frames() {
                assert!(in_play(cut.nes()), "visit {}: the clip's last frame is out of play", visit + 1);
            }
            assert!(cut.step(false).unwrap() && source.step(false).unwrap());
            assert_eq!(cut.nes().ram()[..], source.nes().ram()[..], "visit {}: source frame {}", visit + 1, source.frame());
        }
        assert!(!in_play(cut.nes()), "visit {}: still in play after the clip", visit + 1);
        assert_eq!(source.frame(), to);
    }

    let out = tf(&["replay", "info", &tmp.join("clips").join(&names[0]).to_string_lossy()]);
    assert!(stdout(&out).contains("source:   embedded checkpoint"), "{}", stdout(&out));
    assert!(stdout(&out).contains("clip:     {"), "{}", stdout(&out));
    let out = tf(&["replay", "clip", &source, "--room", "0:77", "--visit", "3", "--out", &tmp.arg("none")]);
    assert_eq!(out.status.code(), Some(1));
    assert!(stdout(&out).contains("no visit 3 to 0:0x77 (2 visits)"), "{}", stdout(&out));
}
