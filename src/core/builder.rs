use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::auth::Auth;
use crate::config::MindroidConfig;
use crate::core::strategy::RunStrategy;
use crate::error::{MindroidError, Result};
use crate::memory::Memory;
use crate::observer::Observer;
use crate::pipeline::Pipeline;
use crate::transport::Transport;

use super::message::{MessageContext, MessageHandler, TransportSender};
use super::routine::{FnRoutine, Routine, RoutineContext};
use super::runtime::Runtime;

#[cfg(feature = "persona")]
use crate::persona::{
    LocalPersonaProvider, MagickmindPersonaClient, MagickmindPersonaStage, PersonaContextBuilder,
    PersonaProvider,
};

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
/// Resolved `[episodes]` settings, shared by the two ingest-stage builders.
#[cfg(feature = "persona")]
struct EpisodeIngestParams {
    base_url: String,
    credential_kind: crate::models::CredentialKind,
    allow_insecure: bool,
    skip_persona: bool,
    scope: crate::config::IngestScope,
}

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
    pub(crate) strategy: RunStrategy,
    #[cfg(feature = "persona")]
    pub(crate) persona_provider: Option<Arc<dyn PersonaProvider>>,
    /// MagickMind-prepared persona: `(base_url, persona_id, cache_ttl_secs, allow_insecure)`.
    /// Set when `persona.type = "magickmind-prepared"`. Built into a
    /// `MagickmindPersonaStage` on demand.
    #[cfg(feature = "persona")]
    pub(crate) magickmind_persona: Option<(String, String, Option<u64>, bool)>,
    /// Agent-scoped prepared persona:
    /// `(base_url, agent_id, credential_kind, cache_ttl_secs, allow_insecure)`.
    /// Set when `persona.type = "magickmind-prepared-agent"`. Built into a
    /// `MagickmindAgentPersonaStage` on demand. `agent_id` is empty on the
    /// end-user route, where the token subject supplies it.
    #[cfg(feature = "persona")]
    pub(crate) magickmind_agent_persona: Option<(
        String,
        String,
        crate::models::CredentialKind,
        Option<u64>,
        bool,
    )>,
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
            strategy: RunStrategy::default(),
            #[cfg(feature = "persona")]
            persona_provider: None,
            #[cfg(feature = "persona")]
            magickmind_persona: None,
            #[cfg(feature = "persona")]
            magickmind_agent_persona: None,
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

    /// Set the run strategy for handling overlapping pipeline runs.
    /// Default: `Concurrent` (all messages processed in parallel).
    pub fn strategy(mut self, strategy: RunStrategy) -> Self {
        self.strategy = strategy;
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

    /// Build a `MagickmindPersonaStage` from the configured MagickMind-prepared persona.
    ///
    /// Returns `None` unless `persona.type = "magickmind-prepared"` was configured (and
    /// auth has been resolved). Unlike [`build_persona_stage`](Self::build_persona_stage),
    /// this performs no network call — the server computes the prompt per-request.
    #[cfg(feature = "persona")]
    pub fn build_magickmind_persona_stage(&self) -> Option<MagickmindPersonaStage> {
        let (base_url, persona_id, ttl_secs, allow_insecure) = self.magickmind_persona.as_ref()?;
        let auth = self.auth.clone()?;
        let mut stage = MagickmindPersonaStage::new(base_url, persona_id, auth)
            .with_allow_insecure(*allow_insecure);
        if let Some(secs) = ttl_secs {
            stage = stage.with_ttl(std::time::Duration::from_secs(*secs));
        }
        Some(stage)
    }

    /// Build a `MagickmindAgentPersonaStage` from the configured
    /// agent-scoped prepared persona.
    ///
    /// Returns `None` unless `persona.type = "magickmind-prepared-agent"` was
    /// configured (and auth resolved). The route follows `auth.type`: a
    /// service-user credential names `agent.agent_id` in the path, while
    /// `auth.type = "enduser"` uses the id-less end-user route. No network call
    /// is made here — the server computes the prompt per-request.
    #[cfg(feature = "persona")]
    pub fn build_magickmind_agent_persona_stage(
        &self,
    ) -> Option<crate::persona::MagickmindAgentPersonaStage> {
        let (base_url, agent_id, credential_kind, ttl_secs, allow_insecure) =
            self.magickmind_agent_persona.as_ref()?;
        let auth = self.auth.clone()?;
        let mut stage = crate::persona::MagickmindAgentPersonaStage::new(base_url, agent_id, auth)
            .with_credential_kind(*credential_kind)
            .with_allow_insecure(*allow_insecure);
        if let Some(secs) = ttl_secs {
            stage = stage.with_ttl(std::time::Duration::from_secs(*secs));
        }
        Some(stage)
    }

    /// A `MagickmindClient` fully wired from config (base URL, auth, credential
    /// kind, api key). `None` if no auth or `auth.base_url` is set.
    #[cfg(feature = "llm-hosted")]
    pub fn magickmind_client(
        &self,
    ) -> Option<crate::pipeline::presets::magickmind::MagickmindClient> {
        let config = self.config.as_ref()?;
        let auth = self.auth.clone()?;
        let base_url = config.auth.base_url.as_deref()?;
        let kind = crate::core::factory::credential_kind_from_config(config);
        let mut client =
            crate::pipeline::presets::magickmind::MagickmindClient::new(base_url, auth)
                .with_credential_kind(kind);
        if let Some(api_key) = &config.auth.api_key {
            client = client.with_api_key(api_key);
        }
        Some(client)
    }

    /// Resolve episode-ingest settings, or `Ok(None)` when `episodes.enabled`
    /// is false.
    ///
    /// Enabling episodes with an unusable configuration is an error, not a
    /// silent no-op: ingest failures are swallowed by design (a memory outage
    /// must not stop the agent), so a bad URL would otherwise warn once per
    /// message forever while storing nothing. This mirrors the persona path,
    /// which also refuses to start rather than degrade invisibly.
    #[cfg(feature = "persona")]
    fn episode_ingest_params(&self) -> Result<Option<EpisodeIngestParams>> {
        let Some(config) = self.config.as_ref() else {
            return Ok(None);
        };
        if !config.episodes.enabled {
            return Ok(None);
        }

        let base_url = config
            .episodes
            .base_url
            .as_deref()
            .or(config.memory.base_url.as_deref())
            .or(config.auth.base_url.as_deref())
            .ok_or_else(|| {
                MindroidError::config(
                    "episodes.base_url, memory.base_url, or auth.base_url is required when \
                     episodes.enabled = true",
                )
            })?
            .to_string();

        crate::core::net::require_secure_url(
            &base_url,
            config.episodes.allow_insecure,
            "episodes.allow_insecure",
        )?;

        let credential_kind = crate::core::factory::credential_kind_from_config(config);
        Ok(Some(EpisodeIngestParams {
            base_url,
            credential_kind,
            allow_insecure: config.episodes.allow_insecure,
            skip_persona: config.episodes.skip_persona,
            scope: config.episodes.scope,
        }))
    }

    /// Build the **inbound** episode-ingest stage from `[episodes]` config.
    ///
    /// Place it early — before any gate that may halt — so every received
    /// message is remembered. Returns `Ok(None)` when `episodes.enabled` is
    /// false or auth is unresolved; returns `Err` when episodes are enabled but
    /// misconfigured, rather than silently ingesting nothing.
    #[cfg(feature = "persona")]
    pub fn build_episode_ingest_stage(&self) -> Result<Option<crate::episode::EpisodeIngestStage>> {
        let Some(p) = self.episode_ingest_params()? else {
            return Ok(None);
        };
        let Some(auth) = self.auth.clone() else {
            return Ok(None);
        };
        Ok(Some(
            crate::episode::EpisodeIngestStage::new(&p.base_url, auth, p.credential_kind)
                .with_allow_insecure(p.allow_insecure)
                .with_skip_persona(p.skip_persona)
                .with_scope(p.scope),
        ))
    }

    /// Build the **outbound** reply-ingest stage from `[episodes]` config.
    ///
    /// Place it after response generation, but **before** any stage that can
    /// fail the pipeline — a stage returning `Err` aborts the run, and the
    /// agent's own turn would never reach episodic memory.
    ///
    /// Same `Ok(None)` / `Err` contract as [`build_episode_ingest_stage`](Self::build_episode_ingest_stage).
    #[cfg(feature = "persona")]
    pub fn build_episode_reply_ingest_stage(
        &self,
    ) -> Result<Option<crate::episode::EpisodeReplyIngestStage>> {
        let Some(p) = self.episode_ingest_params()? else {
            return Ok(None);
        };
        let Some(auth) = self.auth.clone() else {
            return Ok(None);
        };
        // The reply is the agent's own message, so it is always in scope — the
        // scope setting governs which *inbound* traffic is recorded.
        Ok(Some(
            crate::episode::EpisodeReplyIngestStage::new(&p.base_url, auth, p.credential_kind)
                .with_allow_insecure(p.allow_insecure)
                .with_skip_persona(p.skip_persona),
        ))
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

        let coordinator = Arc::new(crate::core::coordinator::PerKey::new(self.strategy));

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
            coordinator,
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
                Some("magickmind-prepared") => {
                    let persona_id = config.persona.persona_id.clone().ok_or_else(|| {
                        MindroidError::config(
                            "persona.persona_id is required when persona.type = \"magickmind-prepared\"",
                        )
                    })?;
                    let base_url = config
                        .persona
                        .base_url
                        .as_deref()
                        .or(config.memory.base_url.as_deref())
                        .or(config.auth.base_url.as_deref())
                        .ok_or_else(|| {
                            MindroidError::config(
                                "persona.base_url, memory.base_url, or auth.base_url is required for magickmind-prepared persona",
                            )
                        })?
                        .to_string();
                    // Auth headers ride on every prepare request — fail fast on
                    // a non-TLS base_url unless explicitly opted in.
                    crate::core::net::require_secure_url(
                        &base_url,
                        config.persona.allow_insecure,
                        "persona.allow_insecure",
                    )?;
                    builder.magickmind_persona = Some((
                        base_url,
                        persona_id,
                        config.persona.cache_ttl_secs,
                        config.persona.allow_insecure,
                    ));
                }
                Some("magickmind-prepared-agent") => {
                    let base_url = config
                        .persona
                        .base_url
                        .as_deref()
                        .or(config.memory.base_url.as_deref())
                        .or(config.auth.base_url.as_deref())
                        .ok_or_else(|| {
                            MindroidError::config(
                                "persona.base_url, memory.base_url, or auth.base_url is required for magickmind-prepared-agent persona",
                            )
                        })?
                        .to_string();
                    // Auth headers ride on every prepare request — fail fast on
                    // a non-TLS base_url unless explicitly opted in.
                    crate::core::net::require_secure_url(
                        &base_url,
                        config.persona.allow_insecure,
                        "persona.allow_insecure",
                    )?;

                    // The credential decides the route. An end-user JWT is the
                    // agent itself (id-less route); a service-user credential
                    // names the agent in the path and so requires agent_id.
                    let credential_kind =
                        crate::core::factory::credential_kind_from_config(&config);
                    let is_enduser = credential_kind == crate::models::CredentialKind::EndUser;
                    let agent_id = if is_enduser {
                        String::new()
                    } else {
                        if config.agent.agent_id.is_empty() {
                            return Err(MindroidError::config(
                                "agent.agent_id is required when persona.type = \
                                 \"magickmind-prepared-agent\" with a service-user credential; \
                                 with auth.type = \"enduser\" the agent is taken from the token \
                                 subject instead",
                            ));
                        }
                        // The id becomes a URL path segment. Percent-encoding
                        // handles separators, but `.` and `..` are path
                        // navigation and would silently drop the preceding
                        // segment — rewriting the route instead of 404ing.
                        if !config
                            .agent
                            .agent_id
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
                        {
                            return Err(MindroidError::config(format!(
                                "agent.agent_id {:?} must contain only ASCII letters, digits, \
                                 '-' or '_' — it is sent as a URL path segment",
                                config.agent.agent_id
                            )));
                        }
                        config.agent.agent_id.clone()
                    };
                    builder.magickmind_agent_persona = Some((
                        base_url,
                        agent_id,
                        credential_kind,
                        config.persona.cache_ttl_secs,
                        config.persona.allow_insecure,
                    ));
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

use super::factory::{build_auth, build_observers, build_pipeline, build_transport};
