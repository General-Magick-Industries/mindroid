mod cache;
mod client;
pub mod local;
pub mod models;
mod prepared_client;
mod prompt;
mod provider;
mod stage;

pub use client::MagickmindPersonaClient;
pub use local::LocalPersonaProvider;
pub use models::{
    EffectivePersonalityResponse, EffectiveSources, EffectiveTrait, PersonaSchema,
    PreparedPersonaResponse, TraitValue,
};
pub use prepared_client::{MagickmindAgentPersonaClient, PersonaCaller};
pub use prompt::build_system_prompt;
pub use provider::{PersonaProvider, PreparedPrompt};
pub use stage::PersonaContextBuilder;
