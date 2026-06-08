use serde::{Deserialize, Serialize};

/// Strategy for handling overlapping pipeline runs within a session.
///
/// This is a type definition only — execution logic lives in a future
/// `SessionCoordinator`, not in `Pipeline` itself.
///
/// # Variants
///
/// - `Concurrent` — All inputs run their pipelines to completion (current behavior).
/// - `LatestWins` — A new input cancels the previous pipeline run.
/// - `Sequential` — Inputs are queued and processed one at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RunStrategy {
    /// All inputs run their pipelines to completion (default, current behavior).
    #[default]
    Concurrent,
    /// A new input cancels the previous pipeline run via `CancellationToken`.
    LatestWins,
    /// Inputs are queued and processed one at a time.
    Sequential,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_concurrent() {
        assert_eq!(RunStrategy::default(), RunStrategy::Concurrent);
    }

    #[test]
    fn test_serde_roundtrip() {
        let strategies = [
            RunStrategy::Concurrent,
            RunStrategy::LatestWins,
            RunStrategy::Sequential,
        ];
        for strategy in &strategies {
            let json = serde_json::to_string(strategy).unwrap();
            let deserialized: RunStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(*strategy, deserialized);
        }
    }

    #[test]
    fn test_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&RunStrategy::LatestWins).unwrap(),
            "\"latest_wins\""
        );
        assert_eq!(
            serde_json::to_string(&RunStrategy::Sequential).unwrap(),
            "\"sequential\""
        );
    }
}
