use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::auth::Auth;
use crate::config::MindroidConfig;
use crate::error::{MindroidError, Result};
use crate::memory::Memory;
use crate::observer::Observer;
use crate::pipeline::Pipeline;
use crate::transport::Transport;

use super::message::{MessageContext, MessageHandler, TransportSender};
use super::routine::{FnRoutine, Routine, RoutineContext};
use super::runtime::Runtime;

#[cfg(feature = "persona")]
use crate::persona::{LocalPersonaProvider, MagickmindPersonaClient, PersonaContextBuilder, PersonaProvider};

#[cfg(feature = "identity")]
use crate::identity::{IdentityResolutionStage, IdentityResolver};

/// Builder for constructing a `Runtime`.
///
/// Resolution order for each subsystem: **code > config > built-in default**.
///
/// Pass a [`MindroidConfig`] via [`.config()`](RuntimeBuilder::config) to use
/// it as a fallback for any subsystem not set explicitly in code. The config
/// is never loaded automatically — you must load and pass it yourself.
///
/// # Examples
///
/// Pure code (no config file):
/// ```ignore
/// Runtime::builder()
///     .transport(StdioTransport::new())
///     .pipeline(ollama_pipeline("http://localhost:11434", "llama3.2"))
///     .auth(StaticAuth::new("dev"))
///     .build()?
/// ```
///
/// Config as fallback for anything not set in code:
/// ```ignore
/// let config = MindroidConfig::from_file("./mindroid.toml")?;
/// Runtime::builder()
///     .config(config)
///     .pipeline(my_custom_pipeline) // overrides [pipeline] in config
///     .build()?
/// ```
///
/// Fully config-driven:
/// ```ignore
/// let config = MindroidConfig::from_file("./mindroid.toml")?;
/// Runtime::builder().config(config).build()?
/// ```
pub struct RuntimeBuilder {
    pub(crate) transport: Option<Box<dyn Transport>>,
    pub(crate) pipeline: Option<Pipeline>,
    pub(crate) auth: Option<Arc<dyn Auth>>,
    pub(crate) memory: Option<Box<dyn Memory>>,
    pub(crate) observers: Vec<Box<dyn Observer>>,
    pub(crate) config: Option<MindroidConfig>,
    pub(crate) handler: Option<MessageHandler>,
    pub(crate) transport_sender: Option<TransportSender>,
    pub(crate) channel_buffer: usize,
    pub(crate) routines: Vec<Box<dyn Routine>>,
    #[cfg(feature = "persona")]
    pub(crate) persona_provider: Option<Arc<dyn PersonaProvider>>,
    #[cfg(feature = "identity")]
    pub(crate) identity_resolver: Option<Arc<IdentityResolver>>,
}

impl RuntimeBuilder {
    pub(crate) fn new() -> Self {
        Self {
            transport: None,
            pipeline: None,
            auth: None,
            memory: None,
            observers: Vec::new(),
            config: None,
            handler: None,
            transport_sender: None,
            channel_buffer: 256,
            routines: Vec::new(),
            #[cfg(feature = "persona")]
            persona_provider: None,
            #[cfg(feature = "identity")]
            identity_resolver: None,
        }
    }

    /// Set a config to use as fallback for any subsystem not set in code.
    ///
    /// The config is never loaded automatically — you must load it yourself
    /// (e.g. [`MindroidConfig::from_file`]) and pass it here.
    pub fn config(mut self, config: MindroidConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn transport(mut self, transport: impl Transport + 'static) -> Self {
        self.transport = Some(Box::new(transport));
        self
    }

    pub fn pipeline(mut self, pipeline: Pipeline) -> Self {
        self.pipeline = Some(pipeline);
        self
    }

    pub fn auth(mut self, auth: impl Auth + 'static) -> Self {
        self.auth = Some(Arc::new(auth));
        self
    }

    /// Set a pre-wrapped `Arc<dyn Auth>`.
    ///
    /// Use this when you already have an `Arc<dyn Auth>` (e.g. shared
    /// with other components). Avoids double-wrapping.
    pub fn auth_shared(mut self, auth: Arc<dyn Auth>) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Get a shared reference to the auth provider, if one has been set.
    ///
    /// Useful after [`Runtime::from_config`] to build components that need
    /// auth (e.g. `MagickmindClient`, `MagickmindMemory`) without constructing
    /// it manually.
    pub fn auth_arc(&self) -> Option<Arc<dyn Auth>> {
        self.auth.clone()
    }

    /// Get a reference to the config stored in the builder.
    pub fn config_ref(&self) -> Option<&MindroidConfig> {
        self.config.as_ref()
    }

    pub fn memory(mut self, memory: impl Memory + 'static) -> Self {
        self.memory = Some(Box::new(memory));
        self
    }

    pub fn observer(mut self, observer: impl Observer + 'static) -> Self {
        self.observers.push(Box::new(observer));
        self
    }

    pub fn transport_sender(mut self, sender: TransportSender) -> Self {
        self.transport_sender = Some(sender);
        self
    }

    pub fn channel_buffer(mut self, size: usize) -> Self {
        self.channel_buffer = size;
        self
    }

    /// Get the persona provider, if one was configured via `[persona]` config.
    #[cfg(feature = "persona")]
    pub fn persona_provider(&self) -> Option<Arc<dyn PersonaProvider>> {
        self.persona_provider.clone()
    }

    /// Build a `PersonaContextBuilder` pipeline stage from the configured
    /// persona provider.
    ///
    /// This is async because it fetches the persona schema (name, role,
    /// background story, tones) once at construction time.
    ///
    /// Returns `None` if no persona provider is configured.
    #[cfg(feature = "persona")]
    pub async fn build_persona_stage(&self) -> Result<Option<PersonaContextBuilder>> {
        let config = self.config.as_ref();
        let persona_id = config.and_then(|c| c.persona.persona_id.as_deref());

        match (&self.persona_provider, persona_id) {
            (Some(provider), Some(pid)) => {
                let stage = PersonaContextBuilder::new(provider.clone(), pid).await?;
                Ok(Some(stage))
            }
            _ => Ok(None),
        }
    }

    /// Build an `IdentityResolutionStage` from the configured resolver.
    ///
    /// Returns `None` if identity resolution is not configured.
    #[cfg(feature = "identity")]
    pub fn build_identity_stage(&self) -> Option<IdentityResolutionStage> {
        self.identity_resolver
            .as_ref()
            .map(|r| IdentityResolutionStage::new(r.clone()))
    }

    /// Register a `Routine` trait implementation.
    pub fn add_routine(mut self, routine: impl Routine + 'static) -> Self {
        self.routines.push(Box::new(routine));
        self
    }

    /// Register a routine using poll and act closures.
    pub fn on_routine<PF, PFut, AF, AFut>(
        mut self,
        name: impl Into<String>,
        interval: std::time::Duration,
        poll_fn: PF,
        act_fn: AF,
    ) -> Self
    where
        PF: Fn() -> PFut + Send + Sync + 'static,
        PFut: Future<Output = Result<Option<String>>> + Send + 'static,
        AF: Fn(RoutineContext, String) -> AFut + Send + Sync + 'static,
        AFut: Future<Output = ()> + Send + 'static,
    {
        self.routines
            .push(Box::new(FnRoutine::new(name, interval, poll_fn, act_fn)));
        self
    }

    /// Set the message handler. Called for each incoming message.
    pub fn on_message<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(MessageContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.handler = Some(Box::new(move |ctx| Box::pin(handler(ctx))));
        self
    }

    /// Assemble the `Runtime`.
    ///
    /// For each subsystem not set explicitly in code, falls back to the config
    /// (if one was provided via [`.config()`](RuntimeBuilder::config)), then
    /// to built-in defaults where they exist.
    pub fn build(mut self) -> Result<Runtime> {
        let raw_config = self.config.take().unwrap_or_default();

        // 1. Auth: code > config > default (static "dev-token")
        let auth: Arc<dyn Auth> = match self.auth.take() {
            Some(a) => a,
            None => build_auth(&raw_config)?,
        };

        // 2. Transport: code > config > default (stdio)
        let transport: Box<dyn Transport> = match self.transport.take() {
            Some(t) => t,
            None => build_transport(&raw_config, &auth)?,
        };

        // 3. Observers: code > config > none
        // If observers were explicitly added in code, use those only.
        // Otherwise fall back to what the config specifies.
        let observers = if !self.observers.is_empty() {
            self.observers
        } else {
            build_observers(&raw_config)
        };

        // 4. Pipeline: code > config > empty pipeline
        let pipeline = match self.pipeline.take() {
            Some(p) => p,
            None => build_pipeline(&raw_config, &auth)?.unwrap_or_default(),
        };

        let handler = self.handler.take().unwrap_or_else(|| {
            Box::new(|ctx: MessageContext| {
                Box::pin(async move {
                    if let Err(e) = ctx.process_and_respond().await {
                        tracing::error!("Handler error: {e}");
                    }
                }) as Pin<Box<dyn Future<Output = ()> + Send>>
            })
        });

        let transport_sender = self.transport_sender.unwrap_or_else(TransportSender::noop);

        Ok(Runtime {
            transport,
            pipeline: Arc::new(pipeline),
            observers: Arc::new(observers),
            agent_config: Arc::new(raw_config.agent),
            handler,
            transport_sender: Arc::new(transport_sender),
            channel_buffer: self.channel_buffer,
            routines: self.routines,
            routine_handles: Vec::new(),
        })
    }
}

// -- from_config: eager build for advanced use cases -------------------------

impl Runtime {
    /// Eagerly build a runtime from config, returning a [`RuntimeBuilder`]
    /// with auth, transport, memory, and observer already constructed.
    ///
    /// Use this when you need access to the auth provider before calling `.build()`
    /// — for example, to share it with a `MagickmindClient` or build a persona
    /// stage. In all other cases, prefer:
    ///
    /// ```ignore
    /// let config = MindroidConfig::from_file("./mindroid.toml")?;
    /// Runtime::builder().config(config).build()?
    /// ```
    ///
    /// After `from_config`, call `.pipeline()` and/or `.on_message()` to
    /// complete the builder, then `.build()`:
    ///
    /// ```ignore
    /// let config = MindroidConfig::from_file("./mindroid.toml")?;
    /// let mut builder = Runtime::from_config(config)?;
    ///
    /// // auth is available here for building pipeline components
    /// let auth = builder.auth_arc().unwrap();
    /// let magickmind = Arc::new(MagickmindClient::new(base_url, auth));
    ///
    /// let runtime = builder
    ///     .pipeline(my_pipeline)
    ///     .build()?;
    /// ```
    pub fn from_config(config: MindroidConfig) -> Result<RuntimeBuilder> {
        let mut builder = RuntimeBuilder::new();

        // Eagerly resolve auth so it's available via auth_arc()
        // before build() is called (needed when sharing auth with
        // pipeline components like MagickmindClient).
        let auth = build_auth(&config)?;
        builder.auth = Some(auth.clone());
        builder.transport = Some(build_transport(&config, &auth)?);
        // Memory resolved from config but not stored in Runtime (reserved for future use).
        builder.observers = build_observers(&config);

        // Persona provider (feature-gated)
        #[cfg(feature = "persona")]
        {
            match config.persona.persona_type.as_deref() {
                Some("magickmind") => {
                    if config.persona.persona_id.is_some() {
                        let base_url = config
                            .persona
                            .base_url
                            .as_deref()
                            .or(config.memory.base_url.as_deref())
                            .or(config.auth.base_url.as_deref())
                            .ok_or_else(|| {
                                MindroidError::config(
                                    "persona.base_url, memory.base_url, or auth.base_url is required for magickmind persona",
                                )
                            })?;
                        let client = MagickmindPersonaClient::new(base_url, auth);
                        builder.persona_provider =
                            Some(Arc::new(client) as Arc<dyn PersonaProvider>);
                    } else {
                        return Err(MindroidError::config(
                            "persona.persona_id is required when persona.type = \"magickmind\"",
                        ));
                    }
                }
                Some("local") => {
                    if let Some(persona_id) = config.persona.persona_id.as_deref() {
                        let data_dir = config
                            .persona
                            .data_dir
                            .as_deref()
                            .unwrap_or("~/.mindroid/personas");
                        let expanded = super::factory::expand_tilde(data_dir);
                        let provider = LocalPersonaProvider::load(
                            expanded.to_str().unwrap_or(data_dir),
                            persona_id,
                        )?;
                        builder.persona_provider =
                            Some(Arc::new(provider) as Arc<dyn PersonaProvider>);
                    }
                }
                _ => {}
            }
        }

        #[cfg(feature = "identity")]
        {
            if config.identity.enabled {
                let registry_path = config
                    .identity
                    .registry_path
                    .as_deref()
                    .unwrap_or("~/.mindroid/identities/registry.json");
                let expanded = super::factory::expand_tilde(registry_path);
                let mut resolver = IdentityResolver::load(&expanded)?;
                if let Some(links) = &config.identity.links {
                    resolver.load_config_links(links);
                }
                builder.identity_resolver = Some(Arc::new(resolver));
            }
        }

        builder.config = Some(config);
        Ok(builder)
    }
}

// -- subsystem builders (delegated to factory module) -------------------------

use super::factory::{build_auth, build_transport, build_observers, build_pipeline};
