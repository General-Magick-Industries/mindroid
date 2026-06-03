//! Session-level coordination of concurrent pipeline runs.
//!
//! [`SessionCoordinator`] enforces a [`RunStrategy`] for a single session:
//!
//! - [`RunStrategy::Concurrent`] — all runs proceed in parallel; no cancellation.
//! - [`RunStrategy::LatestWins`] — a new run cancels the currently-active one.
//! - [`RunStrategy::Sequential`] — runs are queued and executed one at a time.

use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use dashmap::DashMap;
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::core::strategy::RunStrategy;
use crate::error::{MindroidError, Result};

// ---------------------------------------------------------------------------
// CoordinatorPermit
// ---------------------------------------------------------------------------

/// A permit returned by [`SessionCoordinator::acquire`].
///
/// Holds a [`CancellationToken`] scoped to the current pipeline run. For
/// [`RunStrategy::Sequential`], dropping the permit signals the coordinator
/// that the run has finished and the next queued run may proceed.
#[derive(Debug)]
pub struct CoordinatorPermit {
    cancel: CancellationToken,
    /// For Sequential: dropping this sender advances the queue.
    _completion: Option<oneshot::Sender<()>>,
}

impl CoordinatorPermit {
    /// Returns the cancellation token for this run.
    pub fn token(&self) -> &CancellationToken {
        &self.cancel
    }
}

// ---------------------------------------------------------------------------
// SessionCoordinator
// ---------------------------------------------------------------------------

/// Enforces a [`RunStrategy`] across concurrent pipeline runs in one session.
pub struct SessionCoordinator {
    strategy: RunStrategy,
    /// LatestWins: the token for the currently-active run (if any).
    active_cancel: Mutex<Option<CancellationToken>>,
    /// Sequential: sender side of the permit-request queue.
    /// `Option` so `close()` can drop/take it.
    sequential_tx: Mutex<Option<mpsc::Sender<oneshot::Sender<CoordinatorPermit>>>>,
    /// Shutdown signal shared with the Sequential background worker.
    shutdown: CancellationToken,
    closed: AtomicBool,
    #[allow(dead_code)]
    max_queue_depth: usize,
}

impl SessionCoordinator {
    /// Default maximum number of waiters in the Sequential queue.
    const DEFAULT_QUEUE_DEPTH: usize = 16;

    /// Create a new coordinator with the given strategy.
    ///
    /// For `Sequential`, a background task is spawned immediately to service
    /// the queue.
    pub fn new(strategy: RunStrategy) -> Self {
        let max_queue_depth = Self::DEFAULT_QUEUE_DEPTH;
        let shutdown = CancellationToken::new();
        let sequential_tx = if strategy == RunStrategy::Sequential {
            let (tx, rx) = mpsc::channel::<oneshot::Sender<CoordinatorPermit>>(max_queue_depth);
            tokio::spawn(sequential_worker(rx, shutdown.clone()));
            Some(tx)
        } else {
            None
        };

        Self {
            strategy,
            active_cancel: Mutex::new(None),
            sequential_tx: Mutex::new(sequential_tx),
            shutdown,
            closed: AtomicBool::new(false),
            max_queue_depth,
        }
    }

    /// Acquire a permit to run the pipeline.
    ///
    /// Behaviour depends on the configured [`RunStrategy`]:
    ///
    /// - `Concurrent` — always succeeds immediately.
    /// - `LatestWins` — cancels the previous permit (if any) and returns a new one.
    /// - `Sequential` — queues the caller; returns only when all prior runs complete.
    ///
    /// Returns `Err` if the coordinator is closed or the Sequential queue is full.
    pub async fn acquire(&self) -> Result<CoordinatorPermit> {
        if self.closed.load(Ordering::Acquire) {
            return Err(MindroidError::pipeline("coordinator closed"));
        }

        match self.strategy {
            RunStrategy::Concurrent => {
                let cancel = CancellationToken::new();
                Ok(CoordinatorPermit { cancel, _completion: None })
            }

            RunStrategy::LatestWins => {
                let cancel = CancellationToken::new();
                let mut guard = self.active_cancel.lock().await;
                // Cancel whatever was running before.
                if let Some(prev) = guard.take() {
                    prev.cancel();
                }
                *guard = Some(cancel.clone());
                drop(guard);
                Ok(CoordinatorPermit { cancel, _completion: None })
            }

            RunStrategy::Sequential => {
                // Re-check closed after taking the lock to avoid a TOCTOU race
                // where close() drops the sender between our check above and
                // the try_send below.
                let guard = self.sequential_tx.lock().await;

                if self.closed.load(Ordering::Acquire) {
                    return Err(MindroidError::pipeline("coordinator closed"));
                }

                let tx = guard
                    .as_ref()
                    .ok_or_else(|| MindroidError::pipeline("coordinator closed"))?;

                let (reply_tx, reply_rx) = oneshot::channel::<CoordinatorPermit>();

                tx.try_send(reply_tx)
                    .map_err(|_| MindroidError::pipeline("coordinator queue full"))?;

                drop(guard);

                // Wait for the background worker to grant us the permit.
                reply_rx
                    .await
                    .map_err(|_| MindroidError::pipeline("coordinator closed"))
            }
        }
    }

    /// Shut down the coordinator.
    ///
    /// - Sets the `closed` flag so future `acquire()` calls fail immediately.
    /// - Cancels the shutdown token, which signals the Sequential background
    ///   worker to stop granting permits; any queued waiters receive an error.
    /// - Drops the Sequential sender so no new requests can be enqueued.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        // Signal the background worker to abort (it will stop granting permits
        // to any items already in the queue).
        self.shutdown.cancel();
        // Drop the sender so no new requests can be enqueued.
        if let Ok(mut guard) = self.sequential_tx.try_lock() {
            guard.take();
        }
    }
}

// ---------------------------------------------------------------------------
// PerKey coordinator
// ---------------------------------------------------------------------------

/// Per-key coordinator: maps each key (e.g. `sender_id`) to an independent
/// [`SessionCoordinator`] that enforces the shared [`RunStrategy`].
///
/// Keys are created lazily on first `acquire`. The expected cardinality is
/// bounded by active users (typically <10K entries) — negligible memory.
/// TTL-based eviction is omitted deliberately; call [`PerKey::sweep_idle`] manually
/// if cleanup is needed.
pub struct PerKey<K: Hash + Eq + Clone + Send + Sync + 'static> {
    strategy: RunStrategy,
    coordinators: DashMap<K, Arc<SessionCoordinator>>,
}

impl<K: Hash + Eq + Clone + Send + Sync + 'static> PerKey<K> {
    /// Create a new per-key coordinator. All keys share the given strategy.
    pub fn new(strategy: RunStrategy) -> Self {
        Self {
            strategy,
            coordinators: DashMap::new(),
        }
    }

    /// Acquire a permit for the given key.
    ///
    /// Creates a new [`SessionCoordinator`] on first access for this key.
    pub async fn acquire(&self, key: &K) -> Result<CoordinatorPermit> {
        let coordinator = self.coordinators
            .entry(key.clone())
            .or_insert_with(|| Arc::new(SessionCoordinator::new(self.strategy)))
            .value()
            .clone();
        coordinator.acquire().await
    }

    /// Close all coordinators. Subsequent `acquire` calls will fail.
    pub fn close_all(&self) {
        for entry in self.coordinators.iter() {
            entry.value().close();
        }
    }

    /// Remove coordinators that have no active permits.
    /// Call this periodically if you want to reclaim memory from inactive keys.
    pub fn sweep_idle(&self) {
        self.coordinators.retain(|_k, v| Arc::strong_count(v) > 1);
    }
}

// ---------------------------------------------------------------------------
// Sequential background worker
// ---------------------------------------------------------------------------

/// Background task that drives the Sequential queue.
///
/// Receives `oneshot::Sender<CoordinatorPermit>` from callers, grants the
/// permit, and waits for the permit to be dropped (completion signal) before
/// moving to the next item.
///
/// When `shutdown` is cancelled, the worker drains any remaining items in the
/// queue by *dropping* their reply senders (so callers get an error) rather
/// than granting them permits.
async fn sequential_worker(
    mut rx: mpsc::Receiver<oneshot::Sender<CoordinatorPermit>>,
    shutdown: CancellationToken,
) {
    loop {
        // Wait for next request or shutdown.
        let reply_tx = tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                // Drain the remaining queue items by dropping them (error for waiters).
                rx.close();
                while rx.recv().await.is_some() {
                    // drop the sender → waiter's reply_rx gets RecvError → Err
                }
                return;
            }
            item = rx.recv() => {
                match item {
                    Some(tx) => tx,
                    None => return, // channel closed (coordinator dropped)
                }
            }
        };

        // Check shutdown again before granting — the token may have been
        // cancelled after we selected on rx.recv().
        if shutdown.is_cancelled() {
            // Drop reply_tx without sending → waiter gets error.
            continue;
        }

        // Create a fresh cancellation token and a completion oneshot.
        let cancel = CancellationToken::new();
        let (done_tx, done_rx) = oneshot::channel::<()>();

        let permit = CoordinatorPermit {
            cancel,
            _completion: Some(done_tx),
        };

        // If the caller already gave up (reply_tx dropped), skip.
        if reply_tx.send(permit).is_err() {
            continue;
        }

        // Wait until the permit is dropped (done_tx dropped → done_rx resolves),
        // or until shutdown is requested.
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                // The permit was granted but we're shutting down; the permit
                // holder keeps its token (its run may still complete), but we
                // stop accepting new queue items.
                rx.close();
                while rx.recv().await.is_some() {}
                return;
            }
            _ = done_rx => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::timeout;

    // Helper: coordinator with tiny queue for queue-depth tests.
    fn small_sequential(depth: usize) -> Arc<SessionCoordinator> {
        let (tx, rx) = mpsc::channel::<oneshot::Sender<CoordinatorPermit>>(depth);
        let shutdown = CancellationToken::new();
        tokio::spawn(sequential_worker(rx, shutdown.clone()));
        Arc::new(SessionCoordinator {
            strategy: RunStrategy::Sequential,
            active_cancel: Mutex::new(None),
            sequential_tx: Mutex::new(Some(tx)),
            shutdown,
            closed: AtomicBool::new(false),
            max_queue_depth: depth,
        })
    }

    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_concurrent_allows_parallel() {
        let coord = SessionCoordinator::new(RunStrategy::Concurrent);
        // All three should succeed immediately with no blocking.
        let p1 = coord.acquire().await.unwrap();
        let p2 = coord.acquire().await.unwrap();
        let p3 = coord.acquire().await.unwrap();
        // Tokens are independent — none cancelled.
        assert!(!p1.token().is_cancelled());
        assert!(!p2.token().is_cancelled());
        assert!(!p3.token().is_cancelled());
    }

    #[tokio::test]
    async fn test_latest_wins_cancels_previous() {
        let coord = SessionCoordinator::new(RunStrategy::LatestWins);
        let permit_a = coord.acquire().await.unwrap();
        let token_a = permit_a.token().clone();

        // Not cancelled yet.
        assert!(!token_a.is_cancelled());

        let _permit_b = coord.acquire().await.unwrap();

        // Now A should be cancelled.
        assert!(token_a.is_cancelled());
    }

    #[tokio::test]
    async fn test_sequential_ordering() {
        let coord = Arc::new(SessionCoordinator::new(RunStrategy::Sequential));

        // Acquire first permit — this occupies the worker.
        let permit_a = coord.acquire().await.unwrap();

        // Spawn a task that tries to acquire B.
        let coord2 = Arc::clone(&coord);
        let b_handle = tokio::spawn(async move { coord2.acquire().await });

        // Give the spawn a moment to reach the worker queue.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // B should NOT be resolved yet (A is still held).
        assert!(!b_handle.is_finished());

        // Drop A → signals completion → worker grants B.
        drop(permit_a);

        let permit_b = timeout(Duration::from_millis(200), b_handle)
            .await
            .expect("timeout waiting for B")
            .expect("join error")
            .expect("acquire error");

        assert!(!permit_b.token().is_cancelled());
    }

    #[tokio::test]
    async fn test_queue_depth_rejects() {
        // Queue of depth 2: we can have 1 active + 2 waiting = 3 in flight.
        // The 4th should be rejected.
        let coord = small_sequential(2);

        // Occupy the worker.
        let _permit_a = coord.acquire().await.unwrap();

        // Fill the queue (these send to the channel but block waiting for the permit).
        let c1 = Arc::clone(&coord);
        let _h1 = tokio::spawn(async move { c1.acquire().await });
        let c2 = Arc::clone(&coord);
        let _h2 = tokio::spawn(async move { c2.acquire().await });

        // Give time for the queue to fill.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Now the channel is full; next try_send should fail.
        let result = coord.acquire().await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("queue full") || err.to_string().contains("closed"));
    }

    #[tokio::test]
    async fn test_close_unblocks_waiters() {
        let coord = Arc::new(SessionCoordinator::new(RunStrategy::Sequential));

        // Hold the first permit so the worker is blocked.
        let _permit_a = coord.acquire().await.unwrap();

        // Spawn a waiter.
        let coord2 = Arc::clone(&coord);
        let waiter = tokio::spawn(async move { coord2.acquire().await });

        // Give the waiter time to enter the queue.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Close the coordinator.
        coord.close();
        // Drop the active permit to let the worker advance (and discover the
        // channel is closed).
        drop(_permit_a);

        let result = timeout(Duration::from_millis(200), waiter)
            .await
            .expect("timeout")
            .expect("join error");

        // The waiter should get an error (coordinator closed / channel gone).
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_acquire_after_close_returns_error() {
        let coord = SessionCoordinator::new(RunStrategy::Sequential);
        coord.close();
        let result = coord.acquire().await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("closed"), "expected 'closed' in: {err}");
    }

    #[tokio::test]
    async fn test_perkey_independent_sessions() {
        let pk = PerKey::new(RunStrategy::LatestWins);
        let permit_a = pk.acquire(&"user_a".to_string()).await.unwrap();
        let permit_b = pk.acquire(&"user_b".to_string()).await.unwrap();
        // Independent keys — neither should be cancelled
        assert!(!permit_a.token().is_cancelled());
        assert!(!permit_b.token().is_cancelled());
    }

    #[tokio::test]
    async fn test_perkey_same_key_enforced() {
        let pk = PerKey::new(RunStrategy::LatestWins);
        let permit_a = pk.acquire(&"user_x".to_string()).await.unwrap();
        let token_a = permit_a.token().clone();
        let _permit_b = pk.acquire(&"user_x".to_string()).await.unwrap();
        // Same key with LatestWins — first should be cancelled
        assert!(token_a.is_cancelled());
    }

    #[tokio::test]
    async fn test_perkey_close_all() {
        let pk = PerKey::new(RunStrategy::Concurrent);
        let _p1 = pk.acquire(&"a".to_string()).await.unwrap();
        pk.close_all();
        // Existing coordinators are closed, but new key creates new coordinator
        // Let's test that existing key fails:
        let result2 = pk.acquire(&"a".to_string()).await;
        assert!(result2.is_err(), "acquire on closed coordinator should fail");
    }
}
