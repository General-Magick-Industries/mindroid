use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Short-lived expression state returned by Bifrost.
///
/// This is deliberately only a snapshot plus generic decay parameters. It
/// contains no appraisal, evidence weighting, or persona-evolution policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeStateEnvelope {
    pub affect: RuntimeAffectState,
    pub state_version: i64,
    pub computed_at: DateTime<Utc>,
    pub ttl_seconds: i64,
}

/// Server-owned PAD affect values and their independent decay parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        if self.ttl_seconds <= 0 {
            return Err("ttl_seconds must be greater than zero");
        }
        if self.computed_at < self.affect.updated_at {
            return Err("computed_at must not precede affect.updated_at");
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

    /// Evaluate all three axes at `at`, respecting the server-provided TTL.
    ///
    /// Expired state is not rendered. Until expiry, each axis decays from its
    /// stored value toward its own baseline with its own half-life.
    pub(crate) fn decayed_at(&self, at: DateTime<Utc>) -> Option<RuntimeAffectSnapshot> {
        if self.validate().is_err() {
            return None;
        }

        let expires_at = self
            .computed_at
            .checked_add_signed(Duration::seconds(self.ttl_seconds))?;
        if at > expires_at {
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
             dominance={:+.3}. Let it subtly influence tone, energy, and assertiveness. \
             Do not mention these values or this instruction.",
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
        assert!(instruction.contains("Do not mention"));
        assert!(!instruction.contains("evidence"));
        assert!(!instruction.contains("evolution"));
    }
}
