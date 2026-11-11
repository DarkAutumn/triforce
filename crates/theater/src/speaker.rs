//! Where a cell's sound goes: a [`Ring`] of samples between the cell that plays and whatever takes them, and
//! the [`Speaker`], the default sound device taking them.
//!
//! The cell makes samples at the pace its frames are played (the screen's clock), the device takes them at
//! its own.  The two clocks drift, so the ring tells the cell what rate to make samples at
//! ([`Ring::feed_rate`]: the device's, up to half a percent more or less) to stay about [`Ring::target`]
//! samples ahead.  A ring that ran dry is topped up with silence first, one that grew far past its target
//! drops its oldest samples.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parking_lot::Mutex;

/// Seconds of sound the ring aims to hold: the latency a listener gets on top of the device's own.
const TARGET_SECONDS: f64 = 0.06;
/// The most the feed rate departs from the device's (inaudible as pitch).
const MAX_NUDGE: f64 = 0.005;
/// A ring this many targets full drops down to its target.
const OVERFULL: usize = 4;

/// Mono samples (`-1.0..=1.0`) at `rate` per second, first in first out.
pub struct Ring {
    rate: u32,
    samples: Mutex<VecDeque<f32>>,
}

impl Ring {
    pub fn new(rate: u32) -> Ring {
        Ring { rate, samples: Mutex::new(VecDeque::new()) }
    }

    /// Samples per second the taker plays.
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Samples the ring aims to hold.
    pub fn target(&self) -> usize {
        (f64::from(self.rate) * TARGET_SECONDS) as usize
    }

    pub fn len(&self) -> usize {
        self.samples.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds `samples`.  An empty ring (the start, or the taker ran it dry) gets its target of silence first.
    pub fn push(&self, samples: &[f32]) {
        let target = self.target();
        let mut ring = self.samples.lock();
        if ring.is_empty() {
            ring.resize(target, 0.0);
        }
        ring.extend(samples);
        if ring.len() > OVERFULL * target {
            let excess = ring.len() - target;
            ring.drain(..excess);
        }
    }

    /// Moves the oldest samples into `out`, as many as there are; returns how many.
    pub fn pop(&self, out: &mut [f32]) -> usize {
        let mut ring = self.samples.lock();
        let count = out.len().min(ring.len());
        for (slot, sample) in out.iter_mut().zip(ring.drain(..count)) {
            *slot = sample;
        }
        count
    }

    /// The rate to make the next samples at so the ring drifts back to its target.
    pub fn feed_rate(&self) -> f64 {
        let (target, len) = (self.target() as f64, self.len() as f64);
        let short = ((target - len) / target).clamp(-1.0, 1.0);
        f64::from(self.rate) * (1.0 + MAX_NUDGE * short)
    }
}

/// The default sound device, playing a [`Ring`] (silence while it is empty) until dropped.
pub struct Speaker {
    ring: Arc<Ring>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Speaker {
    /// Opens the default output device at its own sample rate.  The error says what is missing.
    pub fn open() -> Result<Speaker, String> {
        let stop = Arc::new(AtomicBool::new(false));
        let (ready, opened) = mpsc::channel();
        // The device's stream is not `Send`: it lives and dies on this thread.
        let thread = std::thread::Builder::new()
            .name("speaker".into())
            .spawn({
                let stop = stop.clone();
                move || match stream() {
                    Ok((stream, ring)) => {
                        let _ = ready.send(Ok(ring));
                        while !stop.load(Ordering::Acquire) {
                            std::thread::park_timeout(Duration::from_millis(200));
                        }
                        drop(stream);
                    }
                    Err(error) => {
                        let _ = ready.send(Err(error));
                    }
                }
            })
            .map_err(|e| format!("the speaker thread did not start: {e}"))?;
        match opened.recv() {
            Ok(Ok(ring)) => Ok(Speaker { ring, stop, thread: Some(thread) }),
            Ok(Err(error)) => Err(error),
            Err(_) => Err("the speaker thread died while opening the sound device".into()),
        }
    }

    /// Where to put the samples it plays.
    pub fn ring(&self) -> &Arc<Ring> {
        &self.ring
    }
}

impl Drop for Speaker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            let _ = thread.join();
        }
    }
}

/// A playing output stream on the default device and the ring it takes its samples from.
fn stream() -> Result<(cpal::Stream, Arc<Ring>), String> {
    let device = cpal::default_host()
        .default_output_device()
        .ok_or("no sound output device")?;
    let supported = device
        .default_output_config()
        .map_err(|e| format!("the sound device has no output format: {e}"))?;
    let config = supported.config();
    let ring = Arc::new(Ring::new(config.sample_rate.0));
    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => output::<f32>(&device, &config, ring.clone()),
        cpal::SampleFormat::I16 => output::<i16>(&device, &config, ring.clone()),
        cpal::SampleFormat::U16 => output::<u16>(&device, &config, ring.clone()),
        other => return Err(format!("the sound device wants {other} samples: not supported")),
    }?;
    stream.play().map_err(|e| format!("the sound device did not start: {e}"))?;
    Ok((stream, ring))
}

fn output<T>(device: &cpal::Device, config: &cpal::StreamConfig, ring: Arc<Ring>) -> Result<cpal::Stream, String>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let channels = usize::from(config.channels).max(1);
    let mut mono = Vec::new();
    device
        .build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                mono.clear();
                mono.resize(data.len() / channels, 0.0);
                // What the ring lacks stays silent.
                ring.pop(&mut mono);
                for (frame, &sample) in data.chunks_mut(channels).zip(&mono) {
                    frame.fill(T::from_sample(sample));
                }
            },
            // A device error (unplugged, the server gone) leaves silence; there is nobody to tell here.
            |_error| {},
            None,
        )
        .map_err(|e| format!("the sound device did not open: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ring_hands_samples_out_in_order_and_starts_a_target_of_silence_ahead() {
        let ring = Ring::new(48_000);
        assert_eq!(ring.target(), 2880);
        ring.push(&[0.5, -0.5]);
        assert_eq!(ring.len(), 2882, "silence first: the taker never starts on an empty ring");
        let mut out = vec![1.0; 2884];
        assert_eq!(ring.pop(&mut out), 2882);
        assert!(out[..2880].iter().all(|&x| x == 0.0));
        assert_eq!(out[2880..], [0.5, -0.5, 1.0, 1.0], "what the ring lacks is left alone");
        // Dry again: the next push starts over a target ahead.
        ring.push(&[0.25]);
        assert_eq!(ring.len(), 2881);
    }

    #[test]
    fn the_feed_rate_steers_the_ring_back_to_its_target() {
        let ring = Ring::new(48_000);
        ring.push(&[]);
        assert_eq!(ring.feed_rate(), 48_000.0, "on target");
        ring.pop(&mut vec![0.0; 1440]);
        assert_eq!(ring.feed_rate(), 48_000.0 * 1.0025, "half empty: a quarter percent faster");
        ring.push(&vec![0.0; 1440 + 2880]);
        assert_eq!(ring.feed_rate(), 48_000.0 * 0.995, "twice the target: as slow as it goes");
    }

    #[test]
    fn an_overfull_ring_drops_its_oldest_samples() {
        let ring = Ring::new(48_000);
        ring.push(&vec![0.5; 3 * 2880]);
        assert_eq!(ring.len(), 4 * 2880, "full, not overfull");
        ring.push(&[0.75]);
        assert_eq!(ring.len(), 2880);
        let mut out = vec![0.0; 2880];
        ring.pop(&mut out);
        assert_eq!((out[0], out[2879]), (0.5, 0.75), "the newest stay");
    }
}
