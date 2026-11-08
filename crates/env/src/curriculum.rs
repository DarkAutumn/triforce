//! Backward curriculum over explored trajectories (`triforce/curriculum.py`; Salimans & Chen 2018).
//!
//! `Explorer::write` stores one successful path per seed state as `<dir>/<state>/NNNN.state` plus
//! `trajectory.json`; more paths from the same seed state (other explore seeds, so other RNG draws: Dodongo's
//! patterns) live beside it as `<dir>/<state>~s<seed>/` and are tracks of their own, keyed by directory name.
//! Each trajectory has a start *point*: episodes start uniformly in `[point, point + stride)`,
//! `stride = ceil(length / stages)`; the point starts one stride before the end.  Once `window` episodes of the
//! current stage finish, the point moves a stride earlier if at least `threshold` of them succeeded, else the
//! stage's counters restart.  A trajectory whose point reaches 0 is complete and no longer sampled.
//!
//! Progress lives in memory, shared by every env of a run through [`SharedCurriculum`], and is rewritten
//! atomically to `state_path` (`<run dir>/curriculum.json`, the Python schema) on every change.

use crate::config::glob;
use crate::error::EnvError;
use indexmap::IndexMap;
use parking_lot::Mutex;
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const TRAJECTORY_FILE: &str = "trajectory.json";

/// Curriculum settings.
#[derive(Clone, Debug, PartialEq)]
pub struct CurriculumSpec {
    /// Directory written by the explorer for one specialist.
    pub trajectories: PathBuf,
    /// Progress file (one per training run).
    pub state_path: PathBuf,
    /// Share of resets that start from a trajectory while any is incomplete.
    pub fraction: f64,
    pub stages: usize,
    pub window: u32,
    pub threshold: f64,
}

impl CurriculumSpec {
    pub fn new(trajectories: PathBuf, state_path: PathBuf, fraction: f64, threshold: f64) -> Self {
        CurriculumSpec {
            trajectories,
            state_path,
            fraction,
            stages: 10,
            window: 10,
            threshold,
        }
    }
}

/// One trajectory's progress (`curriculum.json` value).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Track {
    pub length: usize,
    pub point: usize,
    pub attempts: u32,
    pub successes: u32,
    pub stages_passed: u32,
}

/// A curriculum start: (track name, index into its states).
pub type CurriculumTag = (String, usize);

/// The curriculum part of a terminal info.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CurriculumInfo {
    pub track: String,
    pub index: usize,
    pub length: usize,
    pub point: usize,
}

/// Track name (its directory: the seed state, or `<state>~s<seed>` for a variant) -> savestate paths along its
/// successful path (index 0 = room entry).
pub fn load_trajectories(directory: &Path) -> Result<IndexMap<String, Vec<PathBuf>>, EnvError> {
    let mut tracks = IndexMap::new();
    let mut metas: Vec<PathBuf> = std::fs::read_dir(directory)
        .map(|it| {
            it.filter_map(Result::ok)
                .map(|e| e.path().join(TRAJECTORY_FILE))
                .filter(|p| p.is_file())
                .collect()
        })
        .unwrap_or_default();
    metas.sort();
    for meta_path in metas {
        let text = std::fs::read_to_string(&meta_path)?;
        let meta: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| EnvError::State(format!("{}: {e}", meta_path.display())))?;
        let paths = glob(&meta_path.with_file_name("[0-9]*.state"));
        if meta
            .get("success")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            && !paths.is_empty()
        {
            if meta.get("start_state").and_then(|v| v.as_str()).is_none() {
                return Err(EnvError::State(format!("{}: no start_state", meta_path.display())));
            }
            let name = meta_path
                .parent()
                .and_then(Path::file_name)
                .map(|n| n.to_string_lossy().into_owned())
                .expect("a trajectory file lives in a track directory");
            tracks.insert(name, paths);
        }
    }
    Ok(tracks)
}

/// Steps the start point moves per stage.
pub fn stride_for(length: usize, stages: usize) -> usize {
    length.div_ceil(stages).max(1)
}

/// The shared curriculum of a training run.
#[derive(Debug)]
pub struct Curriculum {
    pub spec: CurriculumSpec,
    pub tracks: IndexMap<String, Vec<PathBuf>>,
    progress: IndexMap<String, Track>,
}

/// Every env of a run holds one of these.
pub type SharedCurriculum = Arc<Mutex<Curriculum>>;

impl Curriculum {
    /// Loads the trajectories and the progress file (written with fresh progress when missing).
    pub fn new(spec: CurriculumSpec) -> Result<Curriculum, EnvError> {
        let tracks = load_trajectories(&spec.trajectories)?;
        if tracks.is_empty() {
            return Err(EnvError::State(format!(
                "No successful trajectories under {}",
                spec.trajectories.display()
            )));
        }
        let progress = if spec.state_path.exists() {
            let text = std::fs::read_to_string(&spec.state_path)?;
            serde_json::from_str(&text)
                .map_err(|e| EnvError::State(format!("{}: {e}", spec.state_path.display())))?
        } else {
            tracks
                .iter()
                .map(|(name, paths)| {
                    let length = paths.len();
                    let track = Track {
                        length,
                        point: length.saturating_sub(stride_for(length, spec.stages)),
                        attempts: 0,
                        successes: 0,
                        stages_passed: 0,
                    };
                    (name.clone(), track)
                })
                .collect()
        };
        let curriculum = Curriculum {
            spec,
            tracks,
            progress,
        };
        curriculum.write()?;
        Ok(curriculum)
    }

    pub fn shared(self) -> SharedCurriculum {
        Arc::new(Mutex::new(self))
    }

    /// Name -> progress.
    pub fn progress(&self) -> &IndexMap<String, Track> {
        &self.progress
    }

    fn write(&self) -> Result<(), EnvError> {
        let path = &self.spec.state_path;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut out = Vec::new();
        let formatter = serde_json::ser::PrettyFormatter::with_indent(b" ");
        let mut ser = serde_json::Serializer::with_formatter(&mut out, formatter);
        self.progress
            .serialize(&mut ser)
            .map_err(|e| EnvError::State(e.to_string()))?;
        let tmp = PathBuf::from(format!("{}.{}.tmp", path.display(), std::process::id()));
        std::fs::write(&tmp, out)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// (savestate path, tag) for a curriculum start, or None (not this reset, or every track complete).
    pub fn next<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(PathBuf, CurriculumTag)> {
        if rng.random::<f64>() >= self.spec.fraction {
            return None;
        }
        let active: Vec<&String> = self
            .progress
            .iter()
            .filter(|(name, t)| t.point > 0 && self.tracks.contains_key(*name))
            .map(|(name, _)| name)
            .collect();
        if active.is_empty() {
            return None;
        }
        let name = active[rng.random_range(0..active.len())];
        let track = &self.progress[name];
        let stride = stride_for(track.length, self.spec.stages);
        let index = rng.random_range(track.point..track.length.min(track.point + stride));
        Some((self.tracks[name][index].clone(), (name.clone(), index)))
    }

    /// Counts an episode started from `tag`; advances the stage when the window is full.
    pub fn record(
        &mut self,
        tag: &CurriculumTag,
        success: bool,
    ) -> Result<CurriculumInfo, EnvError> {
        let (name, index) = (&tag.0, tag.1);
        let (window, threshold, stages) = (self.spec.window, self.spec.threshold, self.spec.stages);
        let track = self
            .progress
            .get_mut(name)
            .ok_or_else(|| EnvError::State(format!("unknown curriculum track {name:?}")))?;
        let stride = stride_for(track.length, stages);
        let mut changed = false;
        if track.point <= index && index < track.point + stride {
            track.attempts += 1;
            track.successes += success as u32;
            if track.attempts >= window {
                if track.successes as f64 >= threshold * track.attempts as f64 {
                    track.point = track.point.saturating_sub(stride);
                    track.stages_passed += 1;
                }
                track.attempts = 0;
                track.successes = 0;
            }
            changed = true;
        }
        let info = CurriculumInfo {
            track: name.clone(),
            index,
            length: track.length,
            point: track.point,
        };
        if changed {
            self.write()?;
        }
        Ok(info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn make_track(root: &Path, name: &str, length: usize, success: bool) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..length {
            std::fs::write(dir.join(format!("{i:04}.state")), b"x").unwrap();
        }
        let meta = serde_json::json!({"start_state": name, "success": success});
        std::fs::write(dir.join(TRAJECTORY_FILE), meta.to_string()).unwrap();
    }

    fn spec(root: &Path, run: &Path) -> CurriculumSpec {
        let mut s = CurriculumSpec::new(root.to_path_buf(), run.join("curriculum.json"), 1.0, 0.6);
        s.stages = 4;
        s.window = 2;
        s
    }

    #[test]
    fn loads_only_successful_tracks() {
        let tmp = tempfile::tempdir().unwrap();
        make_track(tmp.path(), "1_44w_0", 7, true);
        make_track(tmp.path(), "bad", 5, false);
        let tracks = load_trajectories(tmp.path()).unwrap();
        assert_eq!(tracks.keys().collect::<Vec<_>>(), ["1_44w_0"]);
        assert_eq!(tracks["1_44w_0"].len(), 7);
        assert!(tracks["1_44w_0"][0].ends_with("0000.state"));
    }

    #[test]
    fn variant_tracks_of_one_state_are_separate() {
        let tmp = tempfile::tempdir().unwrap();
        make_track(tmp.path(), "2_0es_0", 4, true);
        let variant = tmp.path().join("2_0es_0~s3");
        std::fs::create_dir_all(&variant).unwrap();
        for i in 0..6 {
            std::fs::write(variant.join(format!("{i:04}.state")), b"x").unwrap();
        }
        let meta = serde_json::json!({"start_state": "2_0es_0", "success": true});
        std::fs::write(variant.join(TRAJECTORY_FILE), meta.to_string()).unwrap();
        let tracks = load_trajectories(tmp.path()).unwrap();
        assert_eq!(tracks.keys().collect::<Vec<_>>(), ["2_0es_0", "2_0es_0~s3"]);
        assert_eq!((tracks["2_0es_0"].len(), tracks["2_0es_0~s3"].len()), (4, 6));
    }

    #[test]
    fn stage_advances_backward_and_completes() {
        let tmp = tempfile::tempdir().unwrap();
        let trajectories = tmp.path().join("t");
        make_track(&trajectories, "s", 8, true);
        let mut c = Curriculum::new(spec(&trajectories, &tmp.path().join("run"))).unwrap();
        assert_eq!(stride_for(8, 4), 2);
        assert_eq!(c.progress()["s"].point, 6);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for expected in [4, 2, 0] {
            let (path, tag) = c.next(&mut rng).unwrap();
            assert!(tag.1 >= c.progress()["s"].point && tag.1 < c.progress()["s"].point + 2);
            assert!(path.ends_with(format!("{:04}.state", tag.1)));
            c.record(&tag, true).unwrap();
            let info = c.record(&tag, true).unwrap();
            assert_eq!(info.point, expected);
        }
        assert!(c.next(&mut rng).is_none());
        let saved: IndexMap<String, Track> =
            serde_json::from_str(&std::fs::read_to_string(&c.spec.state_path).unwrap()).unwrap();
        assert_eq!(saved["s"].stages_passed, 3);
        // A run that resumes reads the progress file.
        let again = Curriculum::new(c.spec.clone()).unwrap();
        assert_eq!(again.progress()["s"].point, 0);
    }

    #[test]
    fn failures_restart_the_stage_and_old_stages_dont_count() {
        let tmp = tempfile::tempdir().unwrap();
        let trajectories = tmp.path().join("t");
        make_track(&trajectories, "s", 8, true);
        let mut c = Curriculum::new(spec(&trajectories, &tmp.path().join("run"))).unwrap();
        let tag = ("s".to_owned(), 6);
        c.record(&tag, false).unwrap();
        let info = c.record(&tag, false).unwrap();
        assert_eq!(info.point, 6);
        assert_eq!(c.progress()["s"].attempts, 0);
        c.record(&("s".to_owned(), 2), true).unwrap();
        assert_eq!(c.progress()["s"].attempts, 0);
        assert!(c.next(&mut ChaCha8Rng::seed_from_u64(1)).is_some());
        let mut none = spec(&trajectories, &tmp.path().join("run2"));
        none.fraction = 0.0;
        let c = Curriculum::new(none).unwrap();
        assert!(c.next(&mut ChaCha8Rng::seed_from_u64(1)).is_none());
    }
}
