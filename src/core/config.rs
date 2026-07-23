use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::{MindroidError, Result};

/// Top-level configuration loaded from TOML or built programmatically.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MindroidConfig {
    pub agent: AgentConfig,
    pub transport: TransportConfig,
    pub pipeline: PipelineConfig,
    pub auth: AuthConfig,
    pub memory: MemoryConfig,
    pub observer: ObserverConfig,
    pub persona: PersonaConfig,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,
    #[serde(default)]
    pub identity: IdentityConfig,
    #[serde(default)]
    pub episodes: EpisodesConfig,
}

impl MindroidConfig {
    /// Load config from a TOML file path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content =
            std::fs::read_to_string(path.as_ref()).map_err(|e| MindroidError::Config {
                message: format!("Failed to read config file: {e}"),
                source: Some(Box::new(e)),
            })?;
        Self::from_toml_str(&content)
    }

    /// Parse config from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self> {
        let mut config: Self = toml::from_str(s).map_err(|e| MindroidError::Config {
            message: format!("TOML parse error: {e}"),
            source: Some(Box::new(e)),
        })?;
        config.apply_env_overrides();
        Ok(config)
    }

    /// Resolve config: explicit path → `MINDROID_CONFIG` env → `./mindroid.toml` → `~/.mindroid/config.toml` → defaults.
    pub fn resolve(path: Option<&str>) -> Result<Self> {
        if let Some(p) = path {
            return Self::from_file(p);
        }
        if let Ok(p) = std::env::var("MINDROID_CONFIG") {
            return Self::from_file(p);
        }
        if Path::new("./mindroid.toml").exists() {
            return Self::from_file("./mindroid.toml");
        }
        if let Some(home) = std::env::var_os("HOME") {
            let home_config = Path::new(&home).join(".mindroid/config.toml");
            if home_config.exists() {
                return Self::from_file(home_config);
            }
        }
        let mut config = Self::default();
        config.apply_env_overrides();
        Ok(config)
    }

    /// Resolve config from CLI args, then env, then file discovery.
    ///
    /// Checks for `--config <path>` in `std::env::args()`, then falls through
    /// to [`resolve`](Self::resolve) (which checks `MINDROID_CONFIG` env var,
    /// `./mindroid.toml`, `~/.mindroid/config.toml`, and defaults).
    ///
    /// Usage in examples/binaries:
    /// ```ignore
    /// let config = mindroid::MindroidConfig::resolve_from_args()?;
    /// ```
    ///
    /// Run with:
    /// ```sh
    /// cargo run --example my_example --features full -- --config ./my-config.toml
    /// ```
    pub fn resolve_from_args() -> Result<Self> {
        let config_path = std::env::args().skip_while(|a| a != "--config").nth(1);
        Self::resolve(config_path.as_deref())
    }

    fn apply_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("MINDROID_API_KEY") {
            self.auth.api_key = Some(v);
        }
        if let Ok(v) = std::env::var("MINDROID_EMAIL") {
            self.auth.email = Some(v);
        }
        if let Ok(v) = std::env::var("MINDROID_PASSWORD") {
            self.auth.password = Some(v);
        }
        if let Ok(v) = std::env::var("MINDROID_BASE_URL") {
            self.pipeline.base_url = Some(v.clone());
            self.transport.url = Some(v);
        }
        if let Ok(v) = std::env::var("MINDROID_AGENT_ID") {
            self.agent.agent_id = v;
        }
        if let Ok(v) = std::env::var("MINDROID_PERSONA_ID") {
            self.persona.persona_id = Some(v);
        }
        if let Ok(v) = std::env::var("MINDROID_PERSONA_BASE_URL") {
            self.persona.base_url = Some(v);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentConfig {
    pub agent_id: String,
    pub name: String,
    pub model_type: String,
    pub model_ids: Vec<String>,
    pub compute_power: u8,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            agent_id: String::new(),
            name: "Mindroid Agent".into(),
            model_type: "chat".into(),
            model_ids: Vec::new(),
            compute_power: 50,
            metadata: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TransportConfig {
    #[serde(rename = "type")]
    pub transport_type: Option<String>,
    pub url: Option<String>,
    pub channels: Vec<String>,
    /// Permit sending the auth token over plaintext `ws://` (local development only).
    /// Production deployments must use `wss://`.
    #[serde(default)]
    pub allow_insecure: bool,
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PipelineConfig {
    #[serde(rename = "type")]
    pub pipeline_type: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AuthConfig {
    /// Credential kind. Supported: `"static"`, `"apikey"`, `"enduser"`.
    ///
    /// `"enduser"` carries a pre-minted end-user JWT in `token` and marks this
    /// runtime as the agent itself, routing `magickmind-prepared-agent` persona
    /// calls to the id-less end-user prepare route.
    #[serde(rename = "type")]
    pub auth_type: Option<String>,
    pub email: Option<String>,
    pub password: Option<String>,
    pub api_key: Option<String>,
    pub token: Option<String>,
    pub base_url: Option<String>,
    /// Permit sending credentials over plaintext `http://` (local development
    /// only). Production deployments must use `https://`.
    #[serde(default)]
    pub allow_insecure: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    #[serde(rename = "type")]
    pub memory_type: Option<String>,
    pub path: Option<String>,
    pub base_url: Option<String>,
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ObserverConfig {
    #[serde(rename = "type")]
    pub observer_type: Option<String>,
    pub level: Option<String>,
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

/// Configuration for the persona subsystem.
///
/// When `type = "magickmind"`, the runtime will fetch the effective personality
/// from the magickmind runtime service per-request and format the prompt in-process.
/// When `type = "magickmind-prepared"`, the runtime delegates prompt construction to the
/// MagickMind server's `POST /v1/persona/{id}/prepare` endpoint and uses the returned
/// `system_prompt` verbatim (the server owns trait banding / formatting). Keyed by persona id.
/// When `type = "magickmind-prepared-agent"`, same server-prepared prompt but keyed by
/// **agent id**: the route follows `auth.type` — a service-user credential names the agent in
/// the path (`agent.agent_id` required), while `auth.type = "enduser"` uses the id-less
/// end-user route where the agent is the token subject.
/// When `type = "local"`, the persona is loaded from a local `persona.md` file.
/// When `type = "markdown"`, the system prompt is loaded from a local file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PersonaConfig {
    /// Persona provider type. Supported: `"magickmind"`, `"magickmind-prepared"`, `"local"`.
    #[serde(rename = "type")]
    pub persona_type: Option<String>,
    /// The persona ID to fetch.
    pub persona_id: Option<String>,
    /// Base URL for the magickmind API. Falls back to `memory.base_url` if absent.
    pub base_url: Option<String>,
    /// Directory containing local persona files (for `type = "local"`).
    /// Defaults to `~/.mindroid/personas`.
    pub data_dir: Option<String>,
    /// Cache TTL (seconds) for server-prepared system prompts (`type = "magickmind-prepared"`).
    /// Defaults to 600 (10 minutes). `0` disables caching.
    pub cache_ttl_secs: Option<u64>,
    /// Permit sending auth headers over plaintext `http://` (local development only).
    /// Production deployments must use `https://`.
    pub allow_insecure: bool,
}

/// Configuration for episodic-memory ingest.
///
/// When `enabled`, every inbound message and the agent's own reply is sent to
/// MagickMind's episode `/process` endpoint. The route follows `auth.type`, as
/// with the persona stage: `auth.type = "enduser"` uses the id-less end-user
/// route (owner is the token subject), otherwise the service-user route with
/// `agent.agent_id` as the owner.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EpisodesConfig {
    /// Enable episodic-memory ingest. Default: false.
    pub enabled: bool,
    /// Base URL for the episode API. Falls back to `memory.base_url`, then
    /// `auth.base_url`, if absent.
    pub base_url: Option<String>,
    /// Permit sending auth headers over plaintext `http://` (local dev only).
    pub allow_insecure: bool,
    /// Ask the server not to resolve the agent's persona for each ingested
    /// message. The server otherwise attaches a persona snapshot per message,
    /// which costs a lookup. Default: false (persona attached).
    pub skip_persona: bool,
    /// Which inbound messages are ingested. Default: [`IngestScope::All`].
    pub scope: IngestScope,
}

/// Which inbound messages reach episodic memory.
///
/// In a group magickspace an agent may be present without being addressed. The
/// default records everything, which is what episodic memory is usually for —
/// but it means messages from participants who never addressed the agent, and
/// who may not know it is listening, are stored durably. Operators who cannot
/// give those participants notice should narrow this.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IngestScope {
    /// Every received message, addressed to the agent or not. The default, and
    /// the behaviour before this setting existed.
    #[default]
    All,
    /// Only messages that passed the agent's gate — i.e. that were addressed to
    /// it. Bystander traffic in group channels is not recorded.
    ///
    /// The runtime cannot enforce this from inside the ingest stage, which runs
    /// before any gate; it is enforced by *where* the integrator calls the
    /// stage. See the `persona_agent` example.
    Addressed,
    /// Only 1:1 channels. Group and broadcast traffic is not recorded.
    DirectOnly,
}

/// Configuration for cross-platform identity resolution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct IdentityConfig {
    /// Enable identity resolution. Default: false.
    pub enabled: bool,
    /// Path to the identity registry JSON file.
    /// Default: ~/.mindroid/identities/registry.json
    pub registry_path: Option<String>,
    /// Pre-configured identity links: canonical_name → ["platform:platform_id", ...]
    pub links: Option<HashMap<String, Vec<String>>>,
}

/// Shared LLM provider credentials (endpoint + auth).
///
/// Define once in `[providers.<name>]`, reference from multiple `[models.*]` entries.
///
/// Auth style is derived automatically: if `api_key` is present → Bearer,
/// absent → None. Override with `auth_style = "x-api-key"` for cortex-service style.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderConfig {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    /// Explicit auth style override: `"bearer"`, `"x-api-key"`, or `"none"`.
    /// If absent, derived from `api_key` presence.
    pub auth_style: Option<String>,
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

/// Per-call-site LLM configuration that inherits from a named provider.
///
/// Any field set here overrides the corresponding provider field.
///
/// ```toml
/// [models.main]
/// provider = "cortex"
/// model = "gpt-4"
/// compute_power = 80
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// References a `[providers.<name>]` entry.
    pub provider: String,
    pub model: Option<String>,
    /// Override provider's `api_key`.
    pub api_key: Option<String>,
    /// Override provider's `base_url`.
    pub base_url: Option<String>,
    pub compute_power: Option<u8>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

// -- Tools config ------------------------------------------------------------

/// Configuration for the tools subsystem.
///
/// ```toml
/// [tools.shell]
/// enabled = true
/// timeout_secs = 30
///
/// [tools.open]
/// enabled = true
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolsConfig {
    pub shell: ShellToolConfig,
    pub open: OpenToolConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ShellToolConfig {
    pub enabled: bool,
    pub timeout_secs: u64,
    /// Optional free-text hints about this system's setup injected into the
    /// tool description so the agent picks the right commands on the first try.
    ///
    /// Example:
    /// ```toml
    /// [tools.shell]
    /// instructions = """
    /// Desktop: i3wm on X11.
    /// Screen lock: i3lock.
    /// Media player: Spotify (playerctl works).
    /// Brightness: brightnessctl.
    /// """
    /// ```
    pub instructions: Option<String>,
    /// Commands (binary names) that are permitted to run.
    /// An empty list means "allow all" — for deployments that sandbox externally.
    /// Defaults to a curated safe set.
    #[serde(default = "ShellToolConfig::default_allowed_commands")]
    pub allowed_commands: Vec<String>,
}

impl ShellToolConfig {
    fn default_allowed_commands() -> Vec<String> {
        [
            "ls", "cat", "echo", "pwd", "find", "grep", "head", "tail", "wc", "sort", "uniq",
            "sed", "awk", "curl", "git", "cargo", "rustc", "python3", "node", "npm", "which",
            "env", "printenv", "date", "uname", "whoami", "id", "mkdir", "touch", "rm", "cp", "mv",
            "chmod", "chown", "ln", "stat", "ps", "kill", "pkill", "top", "df", "du", "free",
            "lsof", "ss", "netstat", "ping", "wget", "tar", "gzip", "gunzip", "zip", "unzip", "jq",
            "tr", "cut", "paste", "diff", "patch", "xargs", "tee", "read", "test", "true", "false",
            "sh", "bash", "zsh",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }
}

impl Default for ShellToolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout_secs: 30,
            instructions: None,
            allowed_commands: Self::default_allowed_commands(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenToolConfig {
    pub enabled: bool,
    /// URI schemes (without `://`) that may be opened.
    /// An empty list means "allow all schemes".
    /// Defaults to `["http", "https"]`.
    #[serde(default = "OpenToolConfig::default_allowed_schemes")]
    pub allowed_schemes: Vec<String>,
}

impl OpenToolConfig {
    fn default_allowed_schemes() -> Vec<String> {
        vec!["http".to_string(), "https".to_string()]
    }
}

impl Default for OpenToolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            allowed_schemes: Self::default_allowed_schemes(),
        }
    }
}

// -- LLM config resolver (requires llm-client feature) -----------------------

#[cfg(feature = "llm-client")]
impl MindroidConfig {
    /// Resolve a named model into a ready-to-use [`LlmClientConfig`].
    ///
    /// Looks up `[models.<name>]`, inherits from its `[providers.<provider>]`,
    /// and merges overrides.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let llm_config = config.llm("gate")?;
    /// let client = LlmClient::new(llm_config);
    /// ```
    pub fn llm(&self, name: &str) -> Result<crate::llm_client::LlmClientConfig> {
        use crate::llm_client::{AuthStyle, LlmClientConfig};

        let model_cfg = self
            .models
            .get(name)
            .ok_or_else(|| MindroidError::config(format!("model config '{name}' not found")))?;

        let provider = self.providers.get(&model_cfg.provider).ok_or_else(|| {
            MindroidError::config(format!(
                "provider '{}' referenced by model '{name}' not found",
                model_cfg.provider
            ))
        })?;

        // Merge: model overrides provider
        let base_url = model_cfg
            .base_url
            .as_ref()
            .or(provider.base_url.as_ref())
            .ok_or_else(|| {
                MindroidError::config(format!(
                    "no base_url for model '{name}' or provider '{}'",
                    model_cfg.provider
                ))
            })?;

        let api_key = model_cfg.api_key.clone().or(provider.api_key.clone());

        // Auth style: explicit field > derived from api_key presence
        let auth_style = if let Some(ref style) = provider.auth_style {
            match style.as_str() {
                "none" => AuthStyle::None,
                "x-api-key" => AuthStyle::XApiKey,
                _ => AuthStyle::Bearer,
            }
        } else if api_key.is_some() {
            AuthStyle::Bearer
        } else {
            AuthStyle::None
        };

        let mut llm_config = LlmClientConfig::new(format!("{base_url}/v1"));
        llm_config.api_key = api_key;
        llm_config.default_model = model_cfg.model.clone();
        llm_config.auth_style = auth_style;
        llm_config.default_temperature = model_cfg.temperature;
        llm_config.default_max_tokens = model_cfg.max_tokens;

        if let Some(cp) = model_cfg.compute_power {
            llm_config
                .custom_headers
                .insert("X-Compute-Power".into(), cp.to_string());
        }

        Ok(llm_config)
    }
}
