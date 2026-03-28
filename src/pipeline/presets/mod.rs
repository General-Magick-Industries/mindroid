#[cfg(feature = "llm-local")]
pub mod ollama;
#[cfg(feature = "llm-hosted")]
pub mod magickmind;
