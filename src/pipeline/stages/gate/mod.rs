mod combinators;
mod coordination;
mod relevance;

use async_trait::async_trait;

use crate::core::context::Context;
use crate::Result;

/// A boolean classifier that decides whether the pipeline should proceed.
///
/// Implement this trait on any type to use it with [`OrGate`] and [`AndGate`].
#[async_trait]
pub trait Gate: Send + Sync {
    /// Return `true` to pass, `false` to halt.
    async fn classify(&self, ctx: &Context) -> Result<bool>;
}

pub use combinators::{AndGate, OrGate};
pub use coordination::CoordinationGate;
pub use relevance::RelevanceGate;
