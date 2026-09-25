//! Runtime liveness, observable from outside the process.
//!
//! `run()` blocks until the transport closes, so a caller can tell "running"
//! from "exited" and nothing more. A supervisor needs the state in between: an
//! agent whose transport is stuck in reconnect-backoff is still a live process
//! doing nothing, and looks identical to a healthy one without this.

use std::time::{Duration, Instant};
use tokio::sync::watch;

/// What the runtime is currently doing, from a supervisor's point of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Health {
    /// Building or performing the first connect. Not yet able to receive.
    Starting,
    /// Transport connected; messages can arrive.
    Ready,
    /// Transport dropped and is retrying. Still a live process, but deaf —
    /// this is the state that is invisible without a health signal.
    Reconnecting,
    /// Shut down. Terminal.
    Stopped,
}

impl Health {
    /// Whether the agent can currently receive messages.
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    /// Whether the runtime has stopped for good.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Stopped)
    }
}

impl std::fmt::Display for Health {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Starting => "starting",
            Self::Ready => "ready",
            Self::Reconnecting => "reconnecting",
            Self::Stopped => "stopped",
        };
        f.write_str(s)
    }
}

/// Writer half. Held by whatever owns the connection lifecycle.
#[derive(Clone, Debug)]
pub struct HealthReporter {
    tx: watch::Sender<Health>,
}

impl HealthReporter {
    pub fn new() -> (Self, HealthWatcher) {
        let (tx, rx) = watch::channel(Health::Starting);
        (Self { tx }, HealthWatcher { rx })
    }

    /// Publish a state. Sends only on change, so a reconnect loop hammering
    /// `Reconnecting` doesn't wake every observer on each attempt.
    ///
    /// `Stopped` latches. `disconnect` can time out and retain a live listener,
    /// which then keeps publishing from a runtime that has already returned —
    /// so without this a supervisor sees a terminal agent go `Ready` again, and
    /// [`wait_ready`](HealthWatcher::wait_ready) loses the terminal short-circuit
    /// it relies on.
    pub fn set(&self, health: Health) {
        self.tx.send_if_modified(|current| {
            if *current == health || current.is_terminal() {
                false
            } else {
                *current = health;
                true
            }
        });
    }

    pub fn get(&self) -> Health {
        *self.tx.borrow()
    }
}

/// Reader half. Cheap to clone; a supervisor can hold one per agent.
#[derive(Clone, Debug)]
pub struct HealthWatcher {
    rx: watch::Receiver<Health>,
}

impl HealthWatcher {
    /// Current state, without waiting.
    pub fn get(&self) -> Health {
        *self.rx.borrow()
    }

    pub fn is_ready(&self) -> bool {
        self.get().is_ready()
    }

    /// Wait for the next change. `None` once the runtime is gone.
    ///
    /// Prefer this over polling: at a thousand agents, polling is a thousand
    /// timers doing nothing.
    pub async fn changed(&mut self) -> Option<Health> {
        self.rx.changed().await.ok()?;
        Some(*self.rx.borrow())
    }

    /// Wait until ready, or give up after `timeout`.
    ///
    /// Returns the state observed at timeout, so a caller can report whether it
    /// was still connecting or stuck reconnecting.
    pub async fn wait_ready(&mut self, timeout: Duration) -> std::result::Result<(), Health> {
        // `Instant + Duration` panics on overflow, and `panic = "abort"` makes
        // `wait_ready(Duration::MAX)` — a plausible "wait forever" — a process kill.
        let deadline = Instant::now().checked_add(timeout);
        loop {
            let current = self.get();
            if current.is_ready() {
                return Ok(());
            }
            if current.is_terminal() {
                return Err(current);
            }
            let remaining = match deadline {
                Some(deadline) => deadline.saturating_duration_since(Instant::now()),
                // The deadline overflowed the clock, so it is unreachable.
                None => Duration::MAX,
            };
            if remaining.is_zero() {
                return Err(current);
            }
            if tokio::time::timeout(remaining, self.rx.changed())
                .await
                .is_err()
            {
                return Err(self.get());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn starts_in_starting_and_tracks_changes() {
        let (reporter, watcher) = HealthReporter::new();
        assert_eq!(watcher.get(), Health::Starting);
        assert!(!watcher.is_ready());

        reporter.set(Health::Ready);
        assert_eq!(watcher.get(), Health::Ready);
        assert!(watcher.is_ready());
    }

    /// A reconnect loop sets `Reconnecting` on every attempt; observers must
    /// only wake on an actual transition.
    #[tokio::test]
    async fn repeated_same_state_does_not_notify() {
        let (reporter, mut watcher) = HealthReporter::new();
        reporter.set(Health::Reconnecting);
        assert_eq!(watcher.changed().await, Some(Health::Reconnecting));

        reporter.set(Health::Reconnecting);
        reporter.set(Health::Reconnecting);

        let woke = tokio::time::timeout(Duration::from_millis(50), watcher.changed()).await;
        assert!(woke.is_err(), "identical state must not notify");
    }

    #[tokio::test]
    async fn wait_ready_returns_once_ready() {
        let (reporter, mut watcher) = HealthReporter::new();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            reporter.set(Health::Ready);
        });
        assert!(watcher.wait_ready(Duration::from_secs(2)).await.is_ok());
    }

    /// The distinction the supervisor needs: on timeout, report what it was
    /// stuck doing rather than a bare failure.
    #[tokio::test]
    async fn wait_ready_reports_the_stuck_state() {
        let (reporter, mut watcher) = HealthReporter::new();
        reporter.set(Health::Reconnecting);
        assert_eq!(
            watcher.wait_ready(Duration::from_millis(30)).await,
            Err(Health::Reconnecting)
        );
    }

    /// A stopped runtime never becomes ready — fail immediately rather than
    /// burning the full timeout.
    #[tokio::test]
    async fn wait_ready_gives_up_on_stopped() {
        let (reporter, mut watcher) = HealthReporter::new();
        reporter.set(Health::Stopped);
        let started = Instant::now();
        assert_eq!(
            watcher.wait_ready(Duration::from_secs(10)).await,
            Err(Health::Stopped)
        );
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    /// Dropping the runtime must not hang an observer waiting on it.
    #[tokio::test]
    async fn changed_ends_when_the_reporter_is_dropped() {
        let (reporter, mut watcher) = HealthReporter::new();
        drop(reporter);
        assert_eq!(watcher.changed().await, None);
    }
}
