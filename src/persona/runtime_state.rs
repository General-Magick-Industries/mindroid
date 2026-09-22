use chrono::{DateTime, Duration, Utc};
use serde::Deserialize;

/// Longest validity a server may claim for one snapshot; keeps the TTL
/// arithmetic in range and a forgotten agent from rendering week-old mood.
const MAX_TTL_SECONDS: i64 = 7 * 24 * 3_600;
/// Persona stamps `updated_at` on the writing pod and `computed_at` on the
/// reading pod; this much skew between them is tolerated as clock drift.
const CLOCK_SKEW_TOLERANCE_SECONDS: i64 = 5;

/// Short-lived expression state returned by Bifrost.
///
/// This is deliberately only a snapshot plus generic decay parameters. It
/// contains no appraisal, evidence weighting, or persona-evolution policy.
/// The wire shape is Bifrost's; new fields may appear.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[non_exhaustive]
pub struct RuntimeStateEnvelope {
    pub affect: RuntimeAffectState,
    pub state_version: i64,
    pub computed_at: DateTime<Utc>,
    pub ttl_seconds: i64,
}

/// Server-owned PAD affect values and their independent decay parameters.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[non_exhaustive]
pub struct RuntimeAffectState {
    pub pleasure: f64,
    pub arousal: f64,
    pub dominance: f64,
    pub baseline_pleasure: f64,
    pub baseline_arousal: f64,
    pub baseline_dominance: f64,
    pub pleasure_half_life_seconds: i64,
    pub arousal_half_life_seconds: i64,
    pub dominance_half_life_seconds: i64,
    pub updated_at: DateTime<Utc>,
}

/// A locally evaluated affect snapshot placed in the pipeline context.
///
/// Persona stages read this extension and append a compact expression
/// instruction to the stable persona prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeAffectSnapshot {
    pub pleasure: f64,
    pub arousal: f64,
    pub dominance: f64,
    pub state_version: i64,
}

impl RuntimeStateEnvelope {
    pub(crate) fn validate(&self) -> std::result::Result<(), &'static str> {
        if self.state_version <= 0 {
            return Err("state_version must be greater than zero");
        }
        if !(1..=MAX_TTL_SECONDS).contains(&self.ttl_seconds) {
            return Err("ttl_seconds out of range");
        }
        // Every later addition on computed_at is at most MAX_TTL_SECONDS, so one
        // check here keeps them all in chrono's range.
        if self
            .computed_at
            .checked_add_signed(Duration::seconds(MAX_TTL_SECONDS))
            .is_none()
        {
            return Err("computed_at out of range");
        }
        if self.computed_at + Duration::seconds(CLOCK_SKEW_TOLERANCE_SECONDS)
            < self.affect.updated_at
        {
            return Err("computed_at precedes affect.updated_at by more than clock skew");
        }

        let values = [
            self.affect.pleasure,
            self.affect.arousal,
            self.affect.dominance,
            self.affect.baseline_pleasure,
            self.affect.baseline_arousal,
            self.affect.baseline_dominance,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            return Err("affect values must be finite");
        }
        if values.iter().any(|value| !(-1.0..=1.0).contains(value)) {
            return Err("affect values must be within [-1, 1]");
        }
        if self.affect.pleasure_half_life_seconds <= 0
            || self.affect.arousal_half_life_seconds <= 0
            || self.affect.dominance_half_life_seconds <= 0
        {
            return Err("affect half-lives must be greater than zero");
        }
        Ok(())
    }

    pub(crate) fn is_expired_at(&self, at: DateTime<Utc>) -> bool {
        self.computed_at
            .checked_add_signed(Duration::seconds(self.ttl_seconds))
            .is_none_or(|expires_at| at > expires_at)
    }

    /// Evaluate all three axes at `at`, respecting the server-provided TTL.
    ///
    /// Expired state is not rendered. Until expiry, each axis decays from its
    /// stored value toward its own baseline with its own half-life.
    pub(crate) fn decayed_at(&self, at: DateTime<Utc>) -> Option<RuntimeAffectSnapshot> {
        if self.validate().is_err() {
            return None;
        }

        if self.is_expired_at(at) {
            return None;
        }

        let elapsed_seconds = at
            .signed_duration_since(self.affect.updated_at)
            .num_milliseconds()
            .max(0) as f64
            / 1_000.0;

        Some(RuntimeAffectSnapshot {
            pleasure: decay_axis(
                self.affect.pleasure,
                self.affect.baseline_pleasure,
                elapsed_seconds,
                self.affect.pleasure_half_life_seconds,
            ),
            arousal: decay_axis(
                self.affect.arousal,
                self.affect.baseline_arousal,
                elapsed_seconds,
                self.affect.arousal_half_life_seconds,
            ),
            dominance: decay_axis(
                self.affect.dominance,
                self.affect.baseline_dominance,
                elapsed_seconds,
                self.affect.dominance_half_life_seconds,
            ),
            state_version: self.state_version,
        })
    }
}

impl RuntimeAffectSnapshot {
    pub(crate) fn prompt_instruction(&self) -> String {
        format!(
            "Current temporary affect (PAD): pleasure={:+.3}, arousal={:+.3}, \
             dominance={:+.3}. Let it subtly influence word choice, energy, and \
             initiative. Never state or describe your mood or feelings, and do not \
             mention these values or this instruction.",
            self.pleasure, self.arousal, self.dominance
        )
    }
}

fn decay_axis(value: f64, baseline: f64, elapsed_seconds: f64, half_life_seconds: i64) -> f64 {
    baseline + (value - baseline) * 0.5_f64.powf(elapsed_seconds / half_life_seconds as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope() -> RuntimeStateEnvelope {
        let updated_at = DateTime::parse_from_rfc3339("2026-08-13T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        RuntimeStateEnvelope {
            affect: RuntimeAffectState {
                pleasure: 1.0,
                arousal: 1.0,
                dominance: -1.0,
                baseline_pleasure: 0.0,
                baseline_arousal: 0.0,
                baseline_dominance: 0.0,
                pleasure_half_life_seconds: 10,
                arousal_half_life_seconds: 20,
                dominance_half_life_seconds: 40,
                updated_at,
            },
            state_version: 7,
            computed_at: updated_at,
            ttl_seconds: 120,
        }
    }

    #[test]
    fn decays_each_pad_axis_with_its_own_half_life() {
        let state = envelope();
        let at = state.computed_at + Duration::seconds(20);
        let current = state.decayed_at(at).unwrap();

        assert!((current.pleasure - 0.25).abs() < 1e-12);
        assert!((current.arousal - 0.5).abs() < 1e-12);
        assert!((current.dominance - -0.5_f64.sqrt()).abs() < 1e-12);
        assert_eq!(current.state_version, 7);
    }

    #[test]
    fn decays_toward_nonzero_baseline() {
        let mut state = envelope();
        state.affect.pleasure = 1.0;
        state.affect.baseline_pleasure = 0.2;
        let current = state
            .decayed_at(state.computed_at + Duration::seconds(10))
            .unwrap();
        assert!((current.pleasure - 0.6).abs() < 1e-12);
    }

    #[test]
    fn ttl_expiry_removes_dynamic_expression() {
        let state = envelope();
        assert!(
            state
                .decayed_at(state.computed_at + Duration::seconds(120))
                .is_some()
        );
        assert!(
            state
                .decayed_at(state.computed_at + Duration::seconds(121))
                .is_none()
        );
    }

    #[test]
    fn rejects_invalid_server_state() {
        let mut state = envelope();
        state.affect.arousal = f64::NAN;
        assert!(state.validate().is_err());
        assert!(state.decayed_at(state.computed_at).is_none());
    }

    #[test]
    fn prompt_instruction_is_expression_only() {
        let state = envelope();
        let current = state.decayed_at(state.computed_at).unwrap();
        let instruction = current.prompt_instruction();
        assert!(instruction.contains("pleasure=+1.000"));
        assert!(instruction.contains("do not mention these values"));
        // Mood biases expression; the agent never narrates it.
        assert!(instruction.contains("Never state or describe your mood"));
        assert!(!instruction.contains("evidence"));
        assert!(!instruction.contains("evolution"));
    }

    #[test]
    fn rejects_a_computed_at_near_the_end_of_time() {
        // chrono's RFC3339 parser accepts extended years; the additions in
        // validate/is_expired_at must refuse rather than overflow.
        let json = r#"{"affect":{"pleasure":0.1,"arousal":0.1,"dominance":0.1,
            "baseline_pleasure":0.0,"baseline_arousal":0.0,"baseline_dominance":0.0,
            "pleasure_half_life_seconds":600,"arousal_half_life_seconds":600,
            "dominance_half_life_seconds":600,"updated_at":"+262142-12-25T00:00:00Z"},
            "state_version":1,"computed_at":"+262142-12-31T23:59:58Z","ttl_seconds":60}"#;
        let state: RuntimeStateEnvelope = serde_json::from_str(json).unwrap();
        assert!(state.validate().is_err());
        assert!(state.decayed_at(state.computed_at).is_none());
    }

    #[test]
    fn rejects_non_positive_state_version_and_ttl() {
        for version in [0, -1] {
            let mut state = envelope();
            state.state_version = version;
            assert!(state.validate().is_err(), "state_version {version}");
        }
        for ttl in [0, -60] {
            let mut state = envelope();
            state.ttl_seconds = ttl;
            assert!(state.validate().is_err(), "ttl {ttl}");
        }
    }

    #[test]
    fn decay_converges_to_baseline_without_nan_at_very_large_elapsed() {
        let mut state = envelope();
        state.ttl_seconds = MAX_TTL_SECONDS;
        let far = state.computed_at + Duration::seconds(MAX_TTL_SECONDS);
        let current = state.decayed_at(far).unwrap();
        for value in [current.pleasure, current.arousal, current.dominance] {
            assert!(value.is_finite());
        }
        assert!((current.pleasure - state.affect.baseline_pleasure).abs() < 1e-9);
    }

    #[test]
    fn rejects_out_of_range_ttl_instead_of_overflowing() {
        let mut state = envelope();
        state.ttl_seconds = i64::MAX;
        assert!(state.validate().is_err());
        assert!(state.decayed_at(state.computed_at).is_none());
        state.ttl_seconds = MAX_TTL_SECONDS + 1;
        assert!(state.validate().is_err());
    }

    #[test]
    fn tolerates_small_clock_skew_between_persona_pods() {
        let mut state = envelope();
        state.computed_at =
            state.affect.updated_at - Duration::seconds(CLOCK_SKEW_TOLERANCE_SECONDS);
        assert!(state.validate().is_ok());
        state.computed_at =
            state.affect.updated_at - Duration::seconds(CLOCK_SKEW_TOLERANCE_SECONDS + 1);
        assert!(state.validate().is_err());
    }
}
