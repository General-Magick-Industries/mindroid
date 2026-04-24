pub mod core;
pub mod prelude;

// Re-export everything from prelude at crate root for backward compatibility.
// Users can also import from `mindroid::prelude::*` or from module paths directly.
pub use prelude::*;

// Core trait modules (always available)
pub mod auth;
pub mod memory;
pub mod observer;
pub mod pipeline;
pub mod skills;
pub mod tools;
pub mod transport;

// Optional implementation modules
#[cfg(feature = "llm-client")]
pub mod llm_client;

#[cfg(feature = "persona")]
pub mod persona;

#[cfg(feature = "corpus")]
pub mod corpus;

#[cfg(feature = "http-client")]
pub mod http;

#[cfg(feature = "identity")]
pub mod identity;
