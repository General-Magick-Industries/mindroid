pub mod builder;
pub mod config;
pub mod content;
pub mod context;
pub mod coordinator;
pub mod error;
pub mod events;
pub mod extension_map;
pub(crate) mod factory;
pub mod health;
pub mod message;
pub mod models;
/// Shared HTTP/URL helpers. Gated on the features that pull in `reqwest`.
#[cfg(any(
    feature = "apikey",
    feature = "llm-client",
    feature = "llm-hosted",
    feature = "persistence",
    feature = "persona",
    feature = "speech"
))]
pub(crate) mod net;
pub mod routine;
pub mod runtime;
pub mod strategy;

#[cfg(test)]
mod tests;
