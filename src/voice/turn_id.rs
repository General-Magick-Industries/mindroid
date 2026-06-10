//! TurnId + outbound stamping & telemetry — P7 correlation (task 7).
//!
//! Provides a monotonic [`TurnId`] value type so outbound frames can be
//! correlated with the prompt that triggered them.  The actual stamping onto
//! the wire-format `ChannelMessage` happens in `services/mindroid-voice`
//! (task 9); this module supplies the library-side primitive.

use std::fmt;
use tracing::field;

// ---------------------------------------------------------------------------
// TurnId
// ---------------------------------------------------------------------------

/// A monotonically-increasing identifier for a single voice turn.
///
/// The inner `u64` is public so callers can serialize / deserialize it without
/// extra boilerplate, but construction from raw integers is intentional only
/// through [`TurnCounter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TurnId(pub u64);

impl fmt::Display for TurnId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// TurnCounter — monotonic allocator
// ---------------------------------------------------------------------------

/// A simple monotonic allocator for [`TurnId`] values.
///
/// The first id returned by [`TurnCounter::next`] is **`TurnId(1)`**.
/// `TurnId(0)` is reserved as the "no turn" sentinel so callers can use
/// `Option<TurnId>` or a default-zeroed struct field without ambiguity.
///
/// ```
/// # use mindroid::voice::turn_id::{TurnId, TurnCounter};
/// let mut counter = TurnCounter::default();
/// assert_eq!(counter.next_id(), TurnId(1));
/// assert_eq!(counter.next_id(), TurnId(2));
/// ```
#[derive(Debug, Default)]
pub struct TurnCounter {
    last: u64,
}

impl TurnCounter {
    /// Create a new counter; the first [`TurnId`] emitted will be `TurnId(1)`.
    pub fn new() -> Self {
        Self { last: 0 }
    }

    /// Allocate the next [`TurnId`].  Always strictly greater than the
    /// previously returned value.
    pub fn next_id(&mut self) -> TurnId {
        self.last += 1;
        TurnId(self.last)
    }
}

// ---------------------------------------------------------------------------
// Tracing helper
// ---------------------------------------------------------------------------

impl TurnId {
    /// Record the turn id as a field on the *current* tracing span.
    ///
    /// Adds the field `turn_id = <n>` to whatever span is active.  If no span
    /// is active this is a no-op (the tracing crate silently drops
    /// unrecorded fields).
    ///
    /// ```no_run
    /// # use mindroid::voice::turn_id::{TurnId, TurnCounter};
    /// let mut counter = TurnCounter::new();
    /// let turn = counter.next_id();
    /// let span = tracing::info_span!("process_turn", turn_id = tracing::field::Empty);
    /// let _guard = span.enter();
    /// turn.record_on_current_span();
    /// ```
    pub fn record_on_current_span(&self) {
        tracing::Span::current().record("turn_id", field::display(self));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_starts_at_one() {
        let mut c = TurnCounter::new();
        assert_eq!(c.next_id(), TurnId(1));
    }

    #[test]
    fn counter_is_monotonically_increasing() {
        let mut c = TurnCounter::new();
        let ids: Vec<TurnId> = (0..5).map(|_| c.next_id()).collect();
        // Each successive element must be strictly greater than the previous.
        for window in ids.windows(2) {
            assert!(
                window[1] > window[0],
                "{:?} not > {:?}",
                window[1],
                window[0]
            );
        }
    }

    #[test]
    fn counter_default_matches_new() {
        let mut a = TurnCounter::new();
        let mut b = TurnCounter::default();
        assert_eq!(a.next_id(), b.next_id());
        assert_eq!(a.next_id(), b.next_id());
    }

    #[test]
    fn turn_id_ordering_and_equality() {
        assert!(TurnId(1) < TurnId(2));
        assert!(TurnId(100) > TurnId(99));
        assert_eq!(TurnId(7), TurnId(7));
        assert_ne!(TurnId(7), TurnId(8));
    }

    #[test]
    fn turn_id_display() {
        assert_eq!(TurnId(0).to_string(), "0");
        assert_eq!(TurnId(42).to_string(), "42");
        assert_eq!(TurnId(u64::MAX).to_string(), u64::MAX.to_string());
    }

    #[test]
    fn turn_id_is_copy() {
        let a = TurnId(3);
        let b = a; // copy
        assert_eq!(a, b);
    }
}
