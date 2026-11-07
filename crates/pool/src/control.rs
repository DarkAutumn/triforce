//! Per-env controls the pool shares with its env: the render flag, frame publishing (double buffer + counter,
//! optionally held on its latest frame), frame pacing and the record flag.

use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Holds a frame rate (`play_grid.FramePacer`): each frame is due one period after the previous one. Falling more
/// than [`FramePacer::RESYNC`] behind restarts the schedule instead of fast-forwarding to catch up; `fps <= 0`
/// disables pacing.
#[derive(Clone, Debug)]
pub struct FramePacer {
    fps: f64,
    period: Option<Duration>,
    next: Option<Instant>,
}

impl FramePacer {
    pub const RESYNC: Duration = Duration::from_millis(250);

    pub fn new(fps: f64) -> FramePacer {
        FramePacer {
            fps,
            period: (fps > 0.0).then(|| Duration::from_secs_f64(1.0 / fps)),
            next: None,
        }
    }

    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// Advances the schedule by one frame at `now`; returns how long to sleep before the frame is due.
    pub fn tick(&mut self, now: Instant) -> Option<Duration> {
        let period = self.period?;
        let next = *self.next.get_or_insert(now) + period;
        self.next = Some(next);
        if next > now {
            Some(next - now)
        } else {
            if now - next > Self::RESYNC {
                self.next = Some(now);
            }
            None
        }
    }
}

/// The latest completed frame.
struct Front {
    buf: Vec<u8>,
    counter: u64,
}

struct Shared {
    index: usize,
    frame_bytes: usize,
    render: AtomicBool,
    record: AtomicBool,
    /// Publishing is suspended: the front keeps the latest frame (a finished run's last screen).
    hold: AtomicBool,
    fps: AtomicU64,
    pacer: Mutex<FramePacer>,
    /// The buffer the env renders into; swapped with the front on publish (allocated on first use).
    back: Mutex<Vec<u8>>,
    front: Mutex<Front>,
    counter: AtomicU64,
}

/// One env's controls, shared between the pool (caller side) and the env (worker side). Cheap to clone.
///
/// The env calls [`EnvControl::render`] before each real (non-look-ahead) emulated frame and renders only if it
/// returns true, then [`EnvControl::end_frame`] (or `publish` + `pace`) after it. [`EnvControl::recording`] asks
/// the env to record the inputs of its episodes, checked when an episode starts.
#[derive(Clone)]
pub struct EnvControl {
    shared: Arc<Shared>,
    stop: Arc<AtomicBool>,
}

impl EnvControl {
    pub(crate) fn new(index: usize, fps: f64, frame_bytes: usize, stop: Arc<AtomicBool>) -> Self {
        EnvControl {
            shared: Arc::new(Shared {
                index,
                frame_bytes,
                render: AtomicBool::new(false),
                record: AtomicBool::new(false),
                hold: AtomicBool::new(false),
                fps: AtomicU64::new(fps.to_bits()),
                pacer: Mutex::new(FramePacer::new(fps)),
                back: Mutex::new(Vec::new()),
                front: Mutex::new(Front {
                    buf: Vec::new(),
                    counter: 0,
                }),
                counter: AtomicU64::new(0),
            }),
            stop,
        }
    }

    /// A control outside any pool (tests, single envs): never stopping.
    pub fn standalone(index: usize, fps: f64, frame_bytes: usize) -> Self {
        Self::new(index, fps, frame_bytes, Arc::new(AtomicBool::new(false)))
    }

    /// The env's index in its pool.
    pub fn index(&self) -> usize {
        self.shared.index
    }

    /// Bytes of one frame.
    pub fn frame_bytes(&self) -> usize {
        self.shared.frame_bytes
    }

    /// Whether the next emulated frame should be rendered.
    pub fn render(&self) -> bool {
        self.shared.render.load(Ordering::Relaxed)
    }

    pub fn set_render(&self, on: bool) {
        self.shared.render.store(on, Ordering::Relaxed);
    }

    /// Whether episodes starting now should be recorded.
    pub fn recording(&self) -> bool {
        self.shared.record.load(Ordering::Relaxed)
    }

    pub fn set_recording(&self, on: bool) {
        self.shared.record.store(on, Ordering::Relaxed);
    }

    pub fn fps(&self) -> f64 {
        f64::from_bits(self.shared.fps.load(Ordering::Relaxed))
    }

    /// Paces the env at `fps` real frames per second from its next frame (`<= 0`: unthrottled).
    pub fn set_fps(&self, fps: f64) {
        self.shared.fps.store(fps.to_bits(), Ordering::Relaxed);
    }

    /// Holds the latest published frame (`publish` does nothing) until released.
    pub fn set_hold(&self, on: bool) {
        self.shared.hold.store(on, Ordering::Release);
    }

    pub fn holding(&self) -> bool {
        self.shared.hold.load(Ordering::Acquire)
    }

    /// The pool is shutting down: pacing no longer sleeps, and a long-running env may stop early.
    pub fn stopping(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Renders into the back buffer with `fill` (which returns false if it had no frame) and makes it the
    /// latest frame. Returns whether a frame was published (never while the frame is held).
    pub fn publish(&self, fill: impl FnOnce(&mut [u8]) -> bool) -> bool {
        let shared = &*self.shared;
        if self.holding() {
            return false;
        }
        let mut back = shared.back.lock();
        back.resize(shared.frame_bytes, 0);
        if !fill(&mut back) {
            return false;
        }
        let mut front = shared.front.lock();
        std::mem::swap(&mut front.buf, &mut *back);
        front.counter += 1;
        shared.counter.store(front.counter, Ordering::Release);
        true
    }

    /// Sleeps until the next frame is due at the current fps (no-op when unthrottled or stopping).
    pub fn pace(&self) {
        let fps = self.fps();
        if fps.is_nan() || fps <= 0.0 {
            return;
        }
        let delay = {
            let mut pacer = self.shared.pacer.lock();
            if pacer.fps().to_bits() != fps.to_bits() {
                *pacer = FramePacer::new(fps);
            }
            pacer.tick(Instant::now())
        };
        let Some(delay) = delay else { return };
        let deadline = Instant::now() + delay;
        // Pool::close unparks every worker; park_timeout may also wake spuriously.
        while !self.stopping() {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            std::thread::park_timeout(deadline - now);
        }
    }

    /// After every real frame: publishes it if it was rendered, then paces.
    pub fn end_frame(&self, rendered: bool, fill: impl FnOnce(&mut [u8]) -> bool) {
        if rendered {
            self.publish(fill);
        }
        self.pace();
    }

    /// Frames published so far (0: none yet).
    pub fn frame_counter(&self) -> u64 {
        self.shared.counter.load(Ordering::Acquire)
    }

    /// Copies the latest frame into `out` if it is newer than `since`; returns its counter.
    /// Panics if `out` is not `frame_bytes` long.
    pub fn frame(&self, since: u64, out: &mut [u8]) -> Option<u64> {
        assert_eq!(out.len(), self.shared.frame_bytes, "frame buffer size");
        let front = self.shared.front.lock();
        if front.counter <= since {
            return None;
        }
        out.copy_from_slice(&front.buf);
        Some(front.counter)
    }
}

impl std::fmt::Debug for EnvControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvControl")
            .field("index", &self.index())
            .field("render", &self.render())
            .field("recording", &self.recording())
            .field("fps", &self.fps())
            .field("frames", &self.frame_counter())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pacer_schedules_one_period_per_frame_and_resyncs() {
        let t0 = Instant::now();
        let ms = Duration::from_millis;
        let mut p = FramePacer::new(100.0);
        assert_eq!(p.tick(t0), Some(ms(10)));
        // On time: the next frame is due 10 ms after the previous deadline.
        assert_eq!(p.tick(t0 + ms(10)), Some(ms(10)));
        // 5 ms late: no sleep, the schedule keeps its phase.
        assert_eq!(p.tick(t0 + ms(35)), None);
        assert_eq!(p.tick(t0 + ms(35)), Some(ms(5)));
        // More than 0.25 s behind: restart from now.
        assert_eq!(p.tick(t0 + ms(400)), None);
        assert_eq!(p.tick(t0 + ms(400)), Some(ms(10)));
        assert_eq!(FramePacer::new(0.0).tick(t0), None);
        assert_eq!(FramePacer::new(-1.0).tick(t0), None);
    }

    #[test]
    fn frames_are_double_buffered_with_a_counter() {
        let c = EnvControl::standalone(0, 0.0, 4);
        let mut out = [0u8; 4];
        assert_eq!(c.frame(0, &mut out), None);
        assert!(!c.publish(|_| false));
        assert!(c.publish(|b| {
            b.copy_from_slice(&[1, 2, 3, 4]);
            true
        }));
        assert_eq!(c.frame(0, &mut out), Some(1));
        assert_eq!(out, [1, 2, 3, 4]);
        assert_eq!(c.frame(1, &mut out), None);
        c.publish(|b| {
            b.copy_from_slice(&[5, 6, 7, 8]);
            true
        });
        assert_eq!(c.frame_counter(), 2);
        assert_eq!(c.frame(1, &mut out), Some(2));
        assert_eq!(out, [5, 6, 7, 8]);
    }

    #[test]
    fn a_held_frame_stays_until_released() {
        let c = EnvControl::standalone(0, 0.0, 1);
        let mut out = [0u8; 1];
        let frame = |v: u8| move |b: &mut [u8]| {
            b[0] = v;
            true
        };
        c.publish(frame(1));
        c.set_hold(true);
        assert!(!c.publish(frame(2)));
        assert_eq!((c.frame(0, &mut out), out), (Some(1), [1]));
        c.set_hold(false);
        assert!(c.publish(frame(3)));
        assert_eq!((c.frame(1, &mut out), out), (Some(2), [3]));
    }
}
