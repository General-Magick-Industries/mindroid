#[cfg(feature = "llm-hosted")]
pub mod magickmind;
#[cfg(feature = "persistence")]
pub mod memory;
#[cfg(feature = "llm-local")]
pub mod ollama;
#[cfg(feature = "llm-client")]
pub mod vision;
#[cfg(all(feature = "speech", feature = "llm-client"))]
pub mod voice;
