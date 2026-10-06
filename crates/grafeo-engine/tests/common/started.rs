//! Work started on another thread, for tests that check what waits for what.
//! Include it with `#[path = "common/started.rs"] mod started;`.

use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

/// How long work that should wait gets to finish in the middle anyway.
pub const BRIEFLY: Duration = Duration::from_millis(300);

/// Work running on another thread.
pub struct Started<T> {
    handle: JoinHandle<T>,
    done: mpsc::Receiver<()>,
}

impl<T: Send + 'static> Started<T> {
    /// Starts `work` and returns once it runs.
    pub fn spawn(work: impl FnOnce() -> T + Send + 'static) -> Self {
        let (running, started) = mpsc::channel();
        let (finished, done) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let _ = running.send(());
            let result = work();
            let _ = finished.send(());
            result
        });
        started.recv().expect("the work started");
        Self { handle, done }
    }

    /// Whether the work finishes within [`BRIEFLY`].
    pub fn finishes_briefly(&self) -> bool {
        self.done.recv_timeout(BRIEFLY).is_ok()
    }

    /// Waits for the work and returns its result.
    pub fn join(self) -> T {
        self.handle.join().expect("the work panicked")
    }
}
