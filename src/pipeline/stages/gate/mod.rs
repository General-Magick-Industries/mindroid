mod combinators;
mod coordination;
mod relevance;

use async_trait::async_trait;

use crate::{PipelineContext, Result};

/// A boolean classifier that decides whether the pipeline should proceed.
///
/// Implement this trait on any type to use it with [`OrGate`] and [`AndGate`].
#[async_trait]
pub trait Gate: Send + Sync {
    /// Return `true` to pass, `false` to halt.
    async fn classify(&self, ctx: &PipelineContext) -> Result<bool>;
}

pub use combinators::{AndGate, OrGate};
pub use coordination::CoordinationGate;
pub use relevance::RelevanceGate;
