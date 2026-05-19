#[cfg(feature = "llm-hosted")]
pub mod magickmind;
#[cfg(feature = "persistence")]
pub mod memory;
#[cfg(feature = "llm-local")]
pub mod ollama;
