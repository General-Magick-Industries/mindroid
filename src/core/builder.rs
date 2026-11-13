use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::auth::Auth;
use crate::config::MindroidConfig;
use crate::core::strategy::RunStrategy;
#[cfg(feature = "persona")]
use crate::error::MindroidError;
use crate::error::Result;
use crate::memory::Memory;
use crate::observer::Observer;
use crate::pipeline::Pipeline;
use crate::transport::Transport;

use super::message::{MessageContext, MessageHandler, TransportSender};
use super::routine::{FnRoutine, Routine, RoutineContext};
use super::runtime::Runtime;

#[cfg(feature = "persona")]
use crate::persona::{
    LocalPersonaProvider, MagickmindPersonaStage, PersonaContextBuilder, PersonaProvider,
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
    /// Set when `persona.type = "magickmind-prepared"`, or the deprecated
    /// `"magickmind"` without an agent id or end-user auth. Built into a
    /// `MagickmindPersonaStage` on demand.
    #[cfg(feature = "persona")]
    pub(crate) magickmind_persona: Option<(String, String, Option<u64>, bool)>,
    /// Agent-scoped prepared persona:
    /// `(base_url, agent_id, credential_kind, cache_ttl_secs, allow_insecure)`.
    /// Set when `persona.type = "magickmind-prepared-agent"`, or the deprecated
    /// `"magickmind"` with an agent id or end-user auth. Built into a
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

    /// Build a `PersonaContextBuilder` pipeline stage for a `persona.type = "local"`
    /// persona, which formats the prompt in-process.
    ///
    /// This is async because it fetches the persona schema (name, role,
    /// background story, tones) once at construction time.
    ///
    /// Returns `None` if no persona provider is configured. MagickMind personas,
    /// including the deprecated `"magickmind"` type, use the server-prepared
    /// prompt instead: build them with
    /// [`build_magickmind_agent_persona_stage`](Self::build_magickmind_agent_persona_stage)
    /// or [`build_magickmind_persona_stage`](Self::build_magickmind_persona_stage).
    #[cfg(feature = "persona")]
    pub async fn build_persona_stage(&self) -> Result<Option<PersonaContextBuilder>> {
        let config = self.config.as_ref();
        if config.and_then(|c| c.persona.persona_type.as_deref()) == Some("magickmind") {
            // This used to be the "magickmind" stage, so an app that calls only
            // this builder would otherwise lose its persona without a word.
            tracing::warn!(
                "build_persona_stage() serves persona.type = \"local\" only and returns None \
                 for \"magickmind\", which now uses the server-prepared prompt; build it with \
                 build_magickmind_agent_persona_stage() or build_magickmind_persona_stage()"
            );
            return Ok(None);
        }
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
    /// Returns `None` unless `persona.type = "magickmind-prepared"` (or the deprecated
    /// `"magickmind"` without an agent id or end-user auth) was configured and
    /// auth has been resolved. Unlike [`build_persona_stage`](Self::build_persona_stage),
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
    /// Returns `None` unless `persona.type = "magickmind-prepared-agent"` (or the
    /// deprecated `"magickmind"` with an agent id or end-user auth) was
    /// configured and auth resolved. The route follows `auth.type`: a
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
    /// kind, api key). `None` if no config, auth, or `auth.base_url` is set.
    #[cfg(feature = "llm-hosted")]
    pub fn magickmind_client(
        &self,
    ) -> Option<crate::pipeline::presets::magickmind::MagickmindClient> {
        let config = self.config.as_ref()?;
        let auth = self.auth.clone()?;
        let base_url = config.auth.base_url.as_deref()?;
        let kind = crate::core::factory::credential_kind_from_config(config);
        let mut client = crate::pipeline::presets::magickmind::MagickmindClient::try_new(
            base_url,
            auth,
            config.auth.allow_insecure,
        )
        .map_err(|e| tracing::error!("Refusing to build MagickmindClient: {e}"))
        .ok()?
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

    /// Build a [`MemoryClient`](crate::pipeline::presets::memory::MemoryClient)
    /// from the `[memory]` config section, so the configured backend can be used
    /// in a pipeline (history fetch + message save).
    ///
    /// Mirrors [`build_persona_stage`](Self::build_persona_stage): it constructs
    /// a fresh memory backend from config rather than taking the one the runtime
    /// holds, so both can coexist over the same store. Returns `None` when no
    /// config is present.
    #[cfg(feature = "persistence")]
    pub fn build_memory_client(
        &self,
    ) -> Result<Option<Arc<crate::pipeline::presets::memory::MemoryClient>>> {
        let Some(config) = self.config.as_ref() else {
            return Ok(None);
        };
        let auth = self
            .auth
            .clone()
            .unwrap_or_else(|| Arc::new(crate::auth::static_id::StaticAuth::new("local")));
        let memory: Arc<dyn Memory> = Arc::from(super::factory::build_memory(config, &auth)?);
        Ok(Some(Arc::new(
            crate::pipeline::presets::memory::MemoryClient::new(memory),
        )))
    }

    /// Build the matched artifact pieces from the `[artifacts]` config section:
    /// the offload stage and the `get_artifact` tool, both sharing one store
    /// (see [`artifacts_from_store`](crate::pipeline::presets::artifacts::artifacts_from_store)).
    ///
    /// `scope` is the trusted session/channel id the tool loads under. Returns
    /// `None` when no config is present or `[artifacts] type = "none"`.
    #[cfg(feature = "artifacts")]
    pub fn build_artifact_set(
        &self,
        scope: impl Into<String>,
    ) -> Result<Option<crate::pipeline::presets::artifacts::ArtifactSet>> {
        let Some(config) = self.config.as_ref() else {
            return Ok(None);
        };
        if config.artifacts.artifacts_type.as_deref().unwrap_or("none") == "none" {
            return Ok(None);
        }
        let store = super::factory::build_artifact_store(config)?;
        Ok(Some(
            crate::pipeline::presets::artifacts::artifacts_from_store(store, scope),
        ))
    }

    /// Build an [`ArtifactManager`](crate::artifacts::ArtifactManager) from the
    /// `[artifacts]` config — the cloneable orchestration handle. Use this when a
    /// per-turn handler needs to re-mint the offload stage / load tool from
    /// `ArtifactManager::from_*`. Returns `None` for no config / `type = "none"`.
    #[cfg(feature = "artifacts")]
    pub fn build_artifact_manager(&self) -> Result<Option<crate::artifacts::ArtifactManager>> {
        let Some(config) = self.config.as_ref() else {
            return Ok(None);
        };
        if config.artifacts.artifacts_type.as_deref().unwrap_or("none") == "none" {
            return Ok(None);
        }
        let store = super::factory::build_artifact_store(config)?;
        Ok(Some(crate::artifacts::ArtifactManager::new(store)))
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
        let (health, health_watcher) = crate::core::health::HealthReporter::new();

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
            health,
            health_watcher,
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
        // Resolve auth from config, then delegate — the transport, persona
        // provider, and every `build_*` helper are wired from that auth.
        let auth = build_auth(&config)?;
        Self::from_config_with_auth(config, auth)
    }

    /// Like [`from_config`](Self::from_config), but uses the supplied `auth`
    /// instead of building it from `[auth]` config. The injected auth wires the
    /// transport, persona provider, and all `build_*` helpers.
    ///
    /// For spawned agents that inject a per-instance credential (e.g. a fetched
    /// end-user token): give each agent its own `Arc<dyn Auth>` so nothing is
    /// shared. `config` still supplies routing (`auth.type` → credential kind),
    /// base URLs, and every other setting; only the token comes from `auth`.
    ///
    /// This is the injection point [`auth_shared`](Self::auth_shared) can't
    /// reach on its own — `from_config` builds the transport eagerly, so the
    /// auth must be supplied here, before that happens.
    pub fn from_config_with_auth(
        config: MindroidConfig,
        auth: Arc<dyn Auth>,
    ) -> Result<RuntimeBuilder> {
        // The credential comes from the caller and the routing kind from config;
        // if they disagree, every backend call is misrouted and fails at the
        // server as an opaque 401. Fail here instead, where the cause is legible.
        let configured = crate::core::factory::credential_kind_from_config(&config);
        if auth.kind() != configured {
            return Err(crate::error::MindroidError::config(format!(
                "injected credential is {:?} but auth.type resolves to {:?}: set \
                 auth.type = \"enduser\" for an end-user credential, or inject a \
                 service-user credential",
                auth.kind(),
                configured
            )));
        }

        let mut builder = RuntimeBuilder::new();

        // Eagerly wire auth into the transport (and expose via auth_arc()) so
        // the same credential drives the connection and every build_* helper.
        builder.auth = Some(auth.clone());
        builder.transport = Some(build_transport(&config, &auth)?);
        // Memory resolved from config but not stored in Runtime (reserved for future use).
        builder.observers = build_observers(&config);

        // Persona provider (feature-gated)
        #[cfg(feature = "persona")]
        {
            match config.persona.persona_type.as_deref() {
                // Deprecated alias. It used to format the prompt in-process from
                // the effective personality, which drifted from the server's
                // wording; it now takes Bifrost's prepared prompt so there is
                // one copy of that wording.
                Some("magickmind") => {
                    let agent_keyed = crate::core::factory::credential_kind_from_config(&config)
                        == crate::models::CredentialKind::EndUser
                        || !config.agent.agent_id.is_empty();
                    if agent_keyed {
                        builder.magickmind_agent_persona =
                            Some(prepared_agent_persona(&config, "magickmind")?);
                        // Old configs set persona_id beside agent_id; the agent
                        // route takes the agent's active persona, so a different
                        // persona_id would change the personality silently.
                        let ignored = if config.persona.persona_id.is_some() {
                            "; persona.persona_id is ignored on this route, which uses the \
                             agent's active persona"
                        } else {
                            ""
                        };
                        tracing::warn!(
                            "persona.type = \"magickmind\" is deprecated and now resolves to \
                             \"magickmind-prepared-agent\" (the server-prepared prompt); set \
                             persona.type = \"magickmind-prepared-agent\" explicitly{ignored}"
                        );
                    } else {
                        builder.magickmind_persona = Some(prepared_persona(&config, "magickmind")?);
                        tracing::warn!(
                            "persona.type = \"magickmind\" is deprecated and now resolves to \
                             \"magickmind-prepared\" (the server-prepared prompt), whose \
                             persona-keyed route serves authored trait values only, with no \
                             evolution or per-user adjustment; set agent.agent_id to get the \
                             evolved per-user prompt, and set persona.type explicitly"
                        );
                    }
                }
                Some("magickmind-prepared") => {
                    builder.magickmind_persona =
                        Some(prepared_persona(&config, "magickmind-prepared")?);
                }
                Some("magickmind-prepared-agent") => {
                    builder.magickmind_agent_persona = Some(prepared_agent_persona(
                        &config,
                        "magickmind-prepared-agent",
                    )?);
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

// -- server-prepared persona settings ------------------------------------------

/// The prepare endpoint's base URL: `persona.base_url`, then `memory.base_url`,
/// then `auth.base_url`. `persona_type` only names the setting in the error.
#[cfg(feature = "persona")]
fn prepared_persona_base_url(config: &MindroidConfig, persona_type: &str) -> Result<String> {
    let base_url = config
        .persona
        .base_url
        .as_deref()
        .or(config.memory.base_url.as_deref())
        .or(config.auth.base_url.as_deref())
        .ok_or_else(|| {
            MindroidError::config(format!(
                "persona.base_url, memory.base_url, or auth.base_url is required for \
                 {persona_type} persona"
            ))
        })?
        .to_string();
    // Auth headers ride on every prepare request — fail fast on a non-TLS
    // base_url unless explicitly opted in.
    crate::core::net::require_secure_url(
        &base_url,
        config.persona.allow_insecure,
        "persona.allow_insecure",
    )?;
    Ok(base_url)
}

/// Settings for the persona-keyed prepare route, as stored in
/// `RuntimeBuilder::magickmind_persona`.
#[cfg(feature = "persona")]
fn prepared_persona(
    config: &MindroidConfig,
    persona_type: &str,
) -> Result<(String, String, Option<u64>, bool)> {
    let persona_id = config.persona.persona_id.clone().ok_or_else(|| {
        MindroidError::config(format!(
            "persona.persona_id is required when persona.type = \"{persona_type}\""
        ))
    })?;
    let base_url = prepared_persona_base_url(config, persona_type)?;
    Ok((
        base_url,
        persona_id,
        config.persona.cache_ttl_secs,
        config.persona.allow_insecure,
    ))
}

/// Settings for the agent-keyed prepare route, as stored in
/// `RuntimeBuilder::magickmind_agent_persona`.
#[cfg(feature = "persona")]
fn prepared_agent_persona(
    config: &MindroidConfig,
    persona_type: &str,
) -> Result<(
    String,
    String,
    crate::models::CredentialKind,
    Option<u64>,
    bool,
)> {
    let base_url = prepared_persona_base_url(config, persona_type)?;

    // The credential decides the route. An end-user JWT is the agent itself
    // (id-less route); a service-user credential names the agent in the path
    // and so requires agent_id.
    let credential_kind = crate::core::factory::credential_kind_from_config(config);
    let agent_id = if credential_kind == crate::models::CredentialKind::EndUser {
        String::new()
    } else {
        if config.agent.agent_id.is_empty() {
            return Err(MindroidError::config(format!(
                "agent.agent_id is required when persona.type = \"{persona_type}\" with a \
                 service-user credential; with auth.type = \"enduser\" the agent is taken \
                 from the token subject instead"
            )));
        }
        // The id becomes a URL path segment. Percent-encoding handles
        // separators, but `.` and `..` are path navigation and would silently
        // drop the preceding segment — rewriting the route instead of 404ing.
        if !config
            .agent
            .agent_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        {
            return Err(MindroidError::config(format!(
                "agent.agent_id {:?} must contain only ASCII letters, digits, '-' or '_' — \
                 it is sent as a URL path segment",
                config.agent.agent_id
            )));
        }
        config.agent.agent_id.clone()
    };
    Ok((
        base_url,
        agent_id,
        credential_kind,
        config.persona.cache_ttl_secs,
        config.persona.allow_insecure,
    ))
}

// -- subsystem builders (delegated to factory module) -------------------------

use super::factory::{build_auth, build_observers, build_pipeline, build_transport};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::static_id::StaticAuth;

    fn config_with_auth_token(token: &str) -> MindroidConfig {
        let mut c = MindroidConfig::default();
        c.auth.auth_type = Some("static".into());
        c.auth.token = Some(token.into());
        c
    }

    /// The whole point of the injected form: the supplied credential is used
    /// verbatim, NOT rebuilt from `[auth]`. A spawned agent's per-instance token
    /// must survive a config that names a different one.
    #[tokio::test]
    async fn from_config_with_auth_prefers_the_injected_credential_over_config() {
        let injected: Arc<dyn Auth> = Arc::new(StaticAuth::new("injected-token"));
        let builder =
            Runtime::from_config_with_auth(config_with_auth_token("config-token"), injected)
                .expect("builder should construct");

        let got = builder
            .auth_arc()
            .expect("injected auth must be reachable via auth_arc")
            .get_token()
            .await
            .unwrap();

        assert_eq!(got, "injected-token");
    }

    /// `auth_arc()` is the sharing point (e.g. handing the same credential to a
    /// backend client), so it must hand back the same object, not a rebuild.
    #[tokio::test]
    async fn auth_arc_returns_the_same_instance_that_was_injected() {
        let injected: Arc<dyn Auth> = Arc::new(StaticAuth::new("shared"));
        let builder =
            Runtime::from_config_with_auth(config_with_auth_token("ignored"), injected.clone())
                .expect("builder should construct");

        let from_builder = builder.auth_arc().expect("auth_arc must be populated");
        assert!(
            Arc::ptr_eq(&injected, &from_builder),
            "auth_arc must share the injected Arc, not clone its contents"
        );
    }

    /// A service-user credential against an ordinary config is the common case
    /// and must stay unaffected by the reconciliation check.
    #[tokio::test]
    async fn a_service_user_credential_against_a_service_user_config_is_accepted() {
        let injected: Arc<dyn Auth> = Arc::new(StaticAuth::new("t"));
        assert_eq!(injected.kind(), crate::models::CredentialKind::ServiceUser);

        Runtime::from_config_with_auth(config_with_auth_token("t"), injected)
            .expect("matching kinds must construct");
    }

    /// Config claims end-user routing, but the injected credential is a service
    /// user: every backend call would be misrouted and fail as an opaque 401.
    #[tokio::test]
    async fn a_credential_that_contradicts_the_configured_kind_is_rejected() {
        let mut config = config_with_auth_token("t");
        config.auth.auth_type = Some("enduser".into());

        let injected: Arc<dyn Auth> = Arc::new(StaticAuth::new("service-token"));
        let msg = match Runtime::from_config_with_auth(config, injected) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a contradicting credential must not construct"),
        };
        assert!(
            msg.contains("auth.type"),
            "error should name the fix: {msg}"
        );
    }

    /// The plain `from_config` path still builds auth from config — the injected
    /// form is an override, not a replacement.
    #[tokio::test]
    async fn from_config_builds_auth_from_config() {
        let builder = Runtime::from_config(config_with_auth_token("config-token"))
            .expect("builder should construct");

        let got = builder
            .auth_arc()
            .expect("from_config must populate auth")
            .get_token()
            .await
            .unwrap();

        assert_eq!(got, "config-token");
    }

    /// Auth and transport are wired eagerly, but the config is also retained so
    /// `build()` can still resolve the pipeline, memory, and everything else the
    /// injected credential doesn't cover.
    #[test]
    fn from_config_with_auth_retains_the_config_for_later_resolution() {
        let injected: Arc<dyn Auth> = Arc::new(StaticAuth::new("t"));
        let builder =
            Runtime::from_config_with_auth(config_with_auth_token("cfg-token"), injected).unwrap();

        let stored = builder
            .config_ref()
            .expect("config must remain available to build()");
        assert_eq!(stored.auth.token.as_deref(), Some("cfg-token"));
    }

    #[cfg(feature = "persona")]
    fn magickmind_persona_config() -> MindroidConfig {
        let mut c = config_with_auth_token("t");
        c.persona.persona_type = Some("magickmind".into());
        c.persona.persona_id = Some("persona-1".into());
        c.persona.base_url = Some("https://mm.example.com".into());
        c
    }

    #[cfg(feature = "persona")]
    fn config_error(config: MindroidConfig) -> String {
        match Runtime::from_config(config) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("config must be rejected"),
        }
    }

    /// The deprecated type must take the server's prompt, not format one
    /// locally: an agent id selects the evolved per-user (agent-keyed) route.
    #[cfg(feature = "persona")]
    #[tokio::test]
    async fn magickmind_with_an_agent_id_uses_the_agent_keyed_prepared_prompt() {
        let mut config = magickmind_persona_config();
        config.agent.agent_id = "agent-1".into();
        config.persona.cache_ttl_secs = Some(60);
        let builder = Runtime::from_config(config).unwrap();

        let (_, agent_id, kind, ttl, _) = builder.magickmind_agent_persona.clone().unwrap();
        assert_eq!(agent_id, "agent-1");
        assert_eq!(kind, crate::models::CredentialKind::ServiceUser);
        assert_eq!(
            ttl,
            Some(60),
            "cache_ttl_secs now applies to \"magickmind\""
        );
        assert!(builder.build_magickmind_agent_persona_stage().is_some());
        assert!(builder.build_magickmind_persona_stage().is_none());
        assert!(builder.persona_provider().is_none());
        assert!(builder.build_persona_stage().await.unwrap().is_none());
    }

    #[cfg(feature = "magickmind")]
    #[tokio::test]
    async fn magickmind_with_enduser_auth_uses_the_id_less_agent_route() {
        let mut config = magickmind_persona_config();
        config.auth.auth_type = Some("enduser".into());
        config.auth.base_url = Some("https://mm.example.com".into());
        let builder = Runtime::from_config(config).unwrap();

        let (_, agent_id, kind, _, _) = builder.magickmind_agent_persona.clone().unwrap();
        assert_eq!(agent_id, "", "the token subject names the agent");
        assert_eq!(kind, crate::models::CredentialKind::EndUser);
        assert!(builder.build_magickmind_agent_persona_stage().is_some());
        assert!(builder.build_magickmind_persona_stage().is_none());
        assert!(builder.build_persona_stage().await.unwrap().is_none());
    }

    /// With nothing naming an agent, the only route left is persona-keyed,
    /// which still needs the persona id the old mapping required.
    #[cfg(feature = "persona")]
    #[tokio::test]
    async fn magickmind_without_an_agent_uses_the_persona_keyed_prepared_prompt() {
        let builder = Runtime::from_config(magickmind_persona_config()).unwrap();

        let (_, persona_id, _, _) = builder.magickmind_persona.clone().unwrap();
        assert_eq!(persona_id, "persona-1");
        assert!(builder.build_magickmind_persona_stage().is_some());
        assert!(builder.build_magickmind_agent_persona_stage().is_none());
        assert!(builder.build_persona_stage().await.unwrap().is_none());

        let mut config = magickmind_persona_config();
        config.persona.persona_id = None;
        assert_eq!(
            config_error(config),
            "Config error: persona.persona_id is required when persona.type = \"magickmind\""
        );
    }

    /// The old mapping sent auth headers over any scheme; the prepared routes
    /// refuse plaintext unless opted in, and "magickmind" now does too.
    #[cfg(feature = "persona")]
    #[test]
    fn magickmind_refuses_a_plaintext_base_url_unless_allowed() {
        let mut config = magickmind_persona_config();
        config.persona.base_url = Some("http://mm.example.com".into());
        let msg = config_error(config.clone());
        assert!(msg.contains("persona.allow_insecure"), "{msg}");

        config.persona.allow_insecure = true;
        let builder = Runtime::from_config(config).unwrap();
        assert!(builder.build_magickmind_persona_stage().is_some());
    }

    #[cfg(feature = "persona")]
    #[test]
    fn magickmind_rejects_an_agent_id_that_is_not_a_path_segment_like_the_agent_type() {
        let mut config = magickmind_persona_config();
        config.agent.agent_id = "../x".into();
        let via_alias = config_error(config.clone());

        config.persona.persona_type = Some("magickmind-prepared-agent".into());
        let via_explicit = config_error(config);

        assert!(via_alias.contains("URL path segment"), "{via_alias}");
        assert_eq!(via_alias, via_explicit);
    }

    /// Collects formatted log output so a test can assert on a warning.
    #[cfg(feature = "persona")]
    #[derive(Clone, Default)]
    struct CapturedLogs(Arc<std::sync::Mutex<Vec<u8>>>);

    #[cfg(feature = "persona")]
    impl std::io::Write for CapturedLogs {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[cfg(feature = "persona")]
    impl CapturedLogs {
        /// Routes this thread's events here until the guard drops.
        fn capture(&self) -> tracing::subscriber::DefaultGuard {
            let logs = self.clone();
            tracing::subscriber::set_default(
                tracing_subscriber::fmt()
                    .with_writer(move || logs.clone())
                    .with_ansi(false)
                    .with_max_level(tracing::Level::WARN)
                    .finish(),
            )
        }

        fn text(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

    /// The warning is the only signal that an old config changed route, so it
    /// must name what it resolved to, and what is now ignored or lost.
    #[cfg(feature = "persona")]
    #[tokio::test]
    async fn magickmind_warns_with_the_type_it_resolved_to() {
        let agent_keyed = CapturedLogs::default();
        {
            let _guard = agent_keyed.capture();
            let mut config = magickmind_persona_config();
            config.agent.agent_id = "agent-1".into();
            let builder = Runtime::from_config(config).unwrap();
            builder.build_persona_stage().await.unwrap();
        }
        let text = agent_keyed.text();
        assert!(
            text.contains("now resolves to \"magickmind-prepared-agent\""),
            "{text}"
        );
        assert!(text.contains("persona.persona_id is ignored"), "{text}");
        assert!(
            text.contains("build_magickmind_agent_persona_stage()"),
            "build_persona_stage must point at the prepared builders: {text}"
        );

        let persona_keyed = CapturedLogs::default();
        {
            let _guard = persona_keyed.capture();
            Runtime::from_config(magickmind_persona_config()).unwrap();
        }
        let text = persona_keyed.text();
        assert!(
            text.contains("now resolves to \"magickmind-prepared\""),
            "{text}"
        );
        assert!(text.contains("authored trait values only"), "{text}");
        assert!(text.contains("set agent.agent_id"), "{text}");

        let explicit = CapturedLogs::default();
        {
            let _guard = explicit.capture();
            let mut config = magickmind_persona_config();
            config.agent.agent_id = "agent-1".into();
            config.persona.persona_type = Some("magickmind-prepared-agent".into());
            Runtime::from_config(config).unwrap();
        }
        assert!(
            !explicit.text().contains("deprecated"),
            "{}",
            explicit.text()
        );
    }

    /// The explicit types now share their settings helpers with the alias; the
    /// errors their users see must not change wording.
    #[cfg(feature = "persona")]
    #[test]
    fn the_explicit_prepared_types_keep_their_error_messages() {
        let mut config = config_with_auth_token("t");
        config.persona.persona_type = Some("magickmind-prepared".into());
        assert_eq!(
            config_error(config.clone()),
            "Config error: persona.persona_id is required when persona.type = \
             \"magickmind-prepared\""
        );

        config.persona.persona_id = Some("persona-1".into());
        assert_eq!(
            config_error(config.clone()),
            "Config error: persona.base_url, memory.base_url, or auth.base_url is required \
             for magickmind-prepared persona"
        );

        config.persona.persona_type = Some("magickmind-prepared-agent".into());
        config.persona.base_url = Some("https://mm.example.com".into());
        assert_eq!(
            config_error(config),
            "Config error: agent.agent_id is required when persona.type = \
             \"magickmind-prepared-agent\" with a service-user credential; with auth.type = \
             \"enduser\" the agent is taken from the token subject instead"
        );
    }

    /// "local" is not a Bifrost path and keeps formatting in-process.
    #[cfg(feature = "persona")]
    #[tokio::test]
    async fn local_persona_still_uses_the_in_process_context_builder() {
        let mut config = config_with_auth_token("t");
        config.persona.persona_type = Some("local".into());
        config.persona.persona_id = Some("guide".into());
        config.persona.data_dir = Some(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/examples/local_persona/personas"
            )
            .into(),
        );
        let builder = Runtime::from_config(config).unwrap();

        assert!(builder.persona_provider().is_some());
        assert!(builder.build_magickmind_persona_stage().is_none());
        assert!(builder.build_magickmind_agent_persona_stage().is_none());
        assert!(builder.build_persona_stage().await.unwrap().is_some());
    }
}
