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
    pub corpus: CorpusConfig,
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
        if let Ok(v) = std::env::var("MINDROID_CORPUS_ID") {
            self.corpus.corpus_id = Some(v);
        }
    }

    /// Resolve a named provider, optionally overriding `base_url` from the component.
    ///
    /// This is the non-LLM equivalent of [`llm()`](Self::llm). Components like
    /// `[persona]`, `[corpus]`, `[memory]` reference a `[providers.*]` entry by name;
    /// this method looks it up and merges any component-level `base_url` override.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The named provider does not exist in `[providers]`
    /// - Neither the component nor the provider specifies a `base_url`
    pub fn resolve_provider(
        &self,
        provider_name: &str,
        component_base_url: Option<&str>,
    ) -> Result<ResolvedProvider> {
        let provider = self.providers.get(provider_name).ok_or_else(|| {
            MindroidError::config(format!("provider '{provider_name}' not found"))
        })?;

        let base_url = component_base_url
            .map(|s| s.to_string())
            .or_else(|| provider.base_url.clone())
            .ok_or_else(|| {
                MindroidError::config(format!(
                    "no base_url for provider '{provider_name}' or component override"
                ))
            })?;

        Ok(ResolvedProvider {
            base_url,
            api_key: provider.api_key.clone(),
            auth_type: provider.auth_type.clone(),
            auth_style: provider.auth_style.clone(),
            email: provider.email.clone(),
            password: provider.password.clone(),
        })
    }
}

/// A provider resolved from `[providers.*]` with component-level overrides applied.
///
/// Returned by [`MindroidConfig::resolve_provider`]. Contains everything needed
/// to construct an HTTP client or `Auth` implementation for a component.
#[derive(Debug, Clone)]
pub struct ResolvedProvider {
    pub base_url: String,
    pub api_key: Option<String>,
    pub auth_type: Option<String>,
    pub auth_style: Option<String>,
    pub email: Option<String>,
    pub password: Option<String>,
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
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PipelineConfig {
    #[serde(rename = "type")]
    pub pipeline_type: Option<String>,
    /// Named provider from `[providers.*]` for base_url + auth.
    pub provider: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AuthConfig {
    #[serde(rename = "type")]
    pub auth_type: Option<String>,
    pub email: Option<String>,
    pub password: Option<String>,
    pub api_key: Option<String>,
    pub token: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    #[serde(rename = "type")]
    pub memory_type: Option<String>,
    /// Named provider from `[providers.*]` for base_url + auth.
    pub provider: Option<String>,
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
/// from the magickmind runtime service per-request.
/// When `type = "local"`, the persona is loaded from a local `persona.md` file.
/// When `type = "markdown"`, the system prompt is loaded from a local file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PersonaConfig {
    /// Persona provider type. Supported: `"magickmind"`, `"local"`.
    #[serde(rename = "type")]
    pub persona_type: Option<String>,
    /// Named provider from `[providers.*]` for base_url + auth.
    pub provider: Option<String>,
    /// The persona ID to fetch.
    pub persona_id: Option<String>,
    /// Base URL override. When `provider` is set, overrides the provider's base_url.
    /// Legacy: falls back to `memory.base_url` / `auth.base_url` if no provider.
    pub base_url: Option<String>,
    /// Directory containing local persona files (for `type = "local"`).
    /// Defaults to `~/.mindroid/personas`.
    pub data_dir: Option<String>,
}

/// Configuration for the corpus (RAG) subsystem.
///
/// ```toml
/// [corpus]
/// corpus_id = "abc-123"
/// base_url = "https://api.magickmind.io"  # optional, falls back to pipeline.base_url
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CorpusConfig {
    /// The corpus ID to query for document context.
    pub corpus_id: Option<String>,
    /// Named provider from `[providers.*]` for base_url + auth.
    pub provider: Option<String>,
    /// Base URL override. When `provider` is set, overrides the provider's base_url.
    /// Legacy: falls back to `auth.base_url` if no provider.
    pub base_url: Option<String>,
    /// API key override for corpus queries.
    /// When `provider` is set, falls back to the provider's api_key.
    /// Legacy: falls back to `auth.api_key` if no provider.
    pub api_key: Option<String>,
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

/// Shared provider credentials (endpoint + auth).
///
/// Define once in `[providers.<name>]`, reference from `[models.*]`, `[persona]`,
/// `[corpus]`, `[memory]`, or any component that talks to a remote service.
///
/// Auth style is derived automatically: if `api_key` is present → Bearer,
/// absent → None. Override with `auth_style = "x-api-key"` for cortex-service style.
///
/// For providers that require login (email + password → token exchange), set
/// `auth_type = "apikey"` with `email` and `password`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderConfig {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    /// Explicit auth style override: `"bearer"`, `"x-api-key"`, or `"none"`.
    /// If absent, derived from `api_key` presence.
    pub auth_style: Option<String>,
    /// Auth type for this provider: `"apikey"` (email+password login),
    /// `"static"` (fixed token), or absent (derived from other fields).
    pub auth_type: Option<String>,
    /// Email for `auth_type = "apikey"` login flow.
    pub email: Option<String>,
    /// Password for `auth_type = "apikey"` login flow.
    pub password: Option<String>,
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
    /// MCP server connections. Each entry connects to one external MCP server.
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
}

/// Configuration for a single MCP server connection.
///
/// ```toml
/// [[tools.mcp_servers]]
/// name = "context7"
/// url = "https://mcp.context7.com/mcp"
/// api_key_env = "CONTEXT7_API_KEY"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Unique name used as tool prefix (e.g. "context7" → "context7_query-docs").
    pub name: String,
    /// MCP server endpoint URL.
    pub url: String,
    /// Inline API key (prefer `api_key_env` to avoid secrets in config files).
    pub api_key: Option<String>,
    /// Environment variable name containing the API key.
    /// Resolved at runtime; takes precedence over `api_key`.
    pub api_key_env: Option<String>,
    /// Whether this server is enabled. Default: true.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

impl McpServerConfig {
    /// Resolve the API key: `api_key_env` env var first, then inline `api_key`.
    pub fn resolve_api_key(&self) -> Option<String> {
        if let Some(ref env_name) = self.api_key_env {
            if let Ok(val) = std::env::var(env_name) {
                return Some(val);
            }
        }
        self.api_key.clone()
    }
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
