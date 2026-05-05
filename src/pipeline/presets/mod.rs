#[cfg(feature = "llm-hosted")]
pub mod magickmind;
#[cfg(feature = "llm-local")]
pub mod ollama;
pub mod sqlite;

