//! Gate combinators: [`OrGate`] and [`AndGate`].
//!
//! These combinators compose multiple [`Gate`] implementations with OR / AND logic.

use async_trait::async_trait;

use crate::core::context::Context;
use crate::{PipelineStage, Result};

use super::Gate;

/// Passes if **any** gate passes (OR logic). Halts only if **all** gates halt.
///
/// On error: fail-open (counts as a pass) and logs a warning.
pub struct OrGate {
    gates: Vec<Box<dyn Gate>>,
}

impl OrGate {
    pub fn new(gates: Vec<Box<dyn Gate>>) -> Self {
        Self { gates }
    }
}

#[async_trait]
impl PipelineStage for OrGate {
    fn name(&self) -> &str {
        "OrGate"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        for gate in &self.gates {
            let passed = gate.classify(ctx).await.unwrap_or_else(|e| {
                tracing::warn!("OrGate: gate returned error ({e}), treating as pass (fail-open)");
                true
            });
            if passed {
                // At least one gate passed — let the message through.
                ctx.response = Some(ctx.message.content.clone());
                return Ok(());
            }
        }
        ctx.halted = true;
        Ok(())
    }
}

/// Passes only if **all** gates pass (AND logic). Halts on the first failure.
///
/// On error: fail-open (counts as a pass) and logs a warning.
pub struct AndGate {
    gates: Vec<Box<dyn Gate>>,
}

impl AndGate {
    pub fn new(gates: Vec<Box<dyn Gate>>) -> Self {
        Self { gates }
    }
}

#[async_trait]
impl PipelineStage for AndGate {
    fn name(&self) -> &str {
        "AndGate"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        for gate in &self.gates {
            let passed = gate.classify(ctx).await.unwrap_or_else(|e| {
                tracing::warn!("AndGate: gate returned error ({e}), treating as pass (fail-open)");
                true
            });
            if !passed {
                ctx.halted = true;
                return Ok(());
            }
        }
        ctx.response = Some(ctx.message.content.clone());
        Ok(())
    }
}
