use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::persona::models::{EffectiveSources, EffectiveTrait, TraitValue};

use super::parser::LocalTraitDef;

/// Dyadic (per-user) learned trait overrides stored as a JSON file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DyadicLearnedTraits {
    pub user_id: String,
    pub traits: HashMap<String, TraitValue>,
    pub updated_at: String,
}

/// Blend authored traits with optional dyadic overrides.
///
/// Rules:
/// - lock = "HARD": authored value wins, dyadic ignored.
/// - lock = "SOFT": dyadic used but clamped to ±0.3 of authored.
/// - no lock: dyadic used if present, else authored.
pub fn blend_traits(
    authored: &HashMap<String, LocalTraitDef>,
    dyadic: Option<&DyadicLearnedTraits>,
) -> Vec<EffectiveTrait> {
    let mut result = Vec::with_capacity(authored.len());

    for (trait_ref, def) in authored {
        let authored_value = TraitValue {
            numeric_value: Some(def.value),
            string_value: None,
            string_list_value: None,
        };

        let dyadic_value = dyadic.and_then(|d| d.traits.get(trait_ref)).cloned();

        let (blended_value, dyadic_source, was_clamped) = match def.lock.as_deref() {
            Some("HARD") => {
                // Always use authored, ignore dyadic entirely
                (authored_value.clone(), None, false)
            }
            Some("SOFT") => {
                if let Some(dv) = dyadic_value.clone() {
                    if let Some(dyadic_num) = dv.numeric_value {
                        let clamped = clamp_soft(def.value, dyadic_num);
                        let was_clamped = (clamped - dyadic_num).abs() > f64::EPSILON;
                        (
                            TraitValue {
                                numeric_value: Some(clamped),
                                string_value: None,
                                string_list_value: None,
                            },
                            Some(dv),
                            was_clamped,
                        )
                    } else {
                        // Dyadic value is non-numeric; fall back to authored
                        (authored_value.clone(), Some(dv), false)
                    }
                } else {
                    (authored_value.clone(), None, false)
                }
            }
            _ => {
                // No lock: prefer dyadic if present
                if let Some(dv) = dyadic_value.clone() {
                    (dv.clone(), Some(dv), false)
                } else {
                    (authored_value.clone(), None, false)
                }
            }
        };

        result.push(EffectiveTrait {
            trait_ref: trait_ref.clone(),
            value: blended_value,
            sources: EffectiveSources {
                authored: Some(authored_value),
                global_learned: None,
                dyadic_learned: dyadic_source,
                lock: def.lock.clone(),
                was_clamped,
            },
        });
    }

    result
}

/// Clamp `dyadic` within ±0.3 of `authored`.
fn clamp_soft(authored: f64, dyadic: f64) -> f64 {
    let min = authored - 0.3;
    let max = authored + 0.3;
    dyadic.clamp(min, max)
}
