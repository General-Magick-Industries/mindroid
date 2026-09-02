use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::process::Command;

use crate::config::OpenToolConfig;
use crate::error::Result;

use super::Tool;

/// Opens a URL in the default browser or launches an application by name.
///
/// On Linux uses `xdg-open`; on macOS uses `open`; on Windows uses `cmd /C start`.
/// The process is spawned and detached — the tool returns immediately
/// without waiting for the app to close.
pub struct OpenTool {
    config: OpenToolConfig,
}

impl OpenTool {
    pub fn new() -> Self {
        Self {
            config: OpenToolConfig::default(),
        }
    }

    pub fn with_config(config: OpenToolConfig) -> Self {
        Self { config }
    }
}

impl Default for OpenTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for OpenTool {
    fn name(&self) -> &str {
        "open"
    }

    fn description(&self) -> &str {
        "Open a URL in the default browser, launch an application by name, or use \
         an app URI scheme to open content in the native app. \
         Prefer app URI schemes over web URLs when available — e.g. \
         'spotify:collection' for liked songs, 'spotify:playlist:ID' for a playlist, \
         'spotify:search:query' to search — these open directly in the Spotify app. \
         Use web URLs only as a fallback."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "App URI scheme (e.g. spotify:collection), URL (e.g. https://...), or app name (e.g. spotify, firefox)"
                }
            },
            "required": ["target"]
        })
    }

    async fn execute(&self, args: Value, _ctx: &super::ToolContext) -> Result<String> {
        let target = args
            .get("target")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if target.is_empty() {
            return Ok("Error: no target provided".into());
        }

        // Validate URI scheme when the allowlist is non-empty.
        if !self.config.allowed_schemes.is_empty() {
            match target.find("://") {
                None => {
                    return Ok(format!(
                        "Error: '{target}' has no URI scheme. \
                         Allowed schemes: {allowed}",
                        allowed = self.config.allowed_schemes.join(", ")
                    ));
                }
                Some(pos) => {
                    let scheme = &target[..pos];
                    if !self.config.allowed_schemes.iter().any(|s| s == scheme) {
                        return Ok(format!(
                            "Error: URI scheme '{scheme}' is not allowed. \
                             Allowed schemes: {allowed}",
                            allowed = self.config.allowed_schemes.join(", ")
                        ));
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        let result = Command::new("cmd")
            .args(["/C", "start", "", target.as_str()])
            .spawn();

        #[cfg(target_os = "macos")]
        let result = Command::new("open").arg(&target).spawn();

        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        let result = Command::new("xdg-open").arg(&target).spawn();

        match result {
            Ok(_) => Ok(format!("Opened: {target}")),
            Err(e) => Ok(format!("Error: could not open '{target}': {e}")),
        }
    }
}
