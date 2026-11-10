//! Long native calls that Ctrl-C can interrupt: the work runs on a scoped thread (GIL released) while the calling
//! thread checks Python's signals between short waits. Python handles signals on the main thread only, so the
//! work itself can't see them.

use parking_lot::{Condvar, Mutex};
use pyo3::prelude::*;
use std::time::Duration;

/// How long the caller waits between signal checks.
const TICK: Duration = Duration::from_millis(100);

/// Runs `work` and returns its result. A signal (KeyboardInterrupt) while it runs calls `interrupt` once (it should
/// make `work` stop soon: set a stop flag, cancel the rollouts), waits for `work` to end, then raises the signal's
/// error.
pub fn interruptible<T: Send>(
    py: Python<'_>,
    work: impl FnOnce() -> T + Send,
    interrupt: impl FnOnce(),
) -> PyResult<T> {
    let done = (Mutex::new(false), Condvar::new());
    std::thread::scope(|scope| {
        let handle = scope.spawn(|| {
            let value = work();
            *done.0.lock() = true;
            done.1.notify_all();
            value
        });
        let mut interrupt = Some(interrupt);
        let mut raised = None;
        loop {
            let finished = py.detach(|| {
                let mut finished = done.0.lock();
                if !*finished {
                    done.1.wait_for(&mut finished, TICK);
                }
                *finished
            });
            if finished {
                break;
            }
            if raised.is_none()
                && let Err(error) = py.check_signals()
            {
                if let Some(stop) = interrupt.take() {
                    stop();
                }
                raised = Some(error);
            }
        }
        let value = py.detach(move || handle.join()).map_err(|payload| {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_else(|| "(non-string panic payload)".into());
            pyo3::exceptions::PyRuntimeError::new_err(format!("the native call panicked: {message}"))
        })?;
        match raised {
            Some(error) => Err(error),
            None => Ok(value),
        }
    })
}
