use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::process::Command;
use tokio::time::{Duration, timeout};

use crate::config::ShellToolConfig;
use crate::error::Result;

use super::Tool;

/// Maximum bytes kept from stdout/stderr before truncating.
const MAX_OUTPUT: usize = 1024 * 1024; // 1 MB

/// Runs shell commands on the local computer subject to an allowlist.
///
/// On Unix uses `/bin/sh -c`; on Windows uses `cmd /C`.
/// Environment is scrubbed to a safe allowlist so agent secrets don't leak.
/// Privileged commands (sudo/doas) are always blocked.
pub struct ShellTool {
    timeout_secs: u64,
    description: String,
    config: ShellToolConfig,
}

impl ShellTool {
    pub fn new(timeout_secs: u64) -> Self {
        Self::with_instructions(timeout_secs, None)
    }

    pub fn with_instructions(timeout_secs: u64, instructions: Option<String>) -> Self {
        Self::with_config(timeout_secs, instructions, ShellToolConfig::default())
    }

    pub fn with_config(
        timeout_secs: u64,
        instructions: Option<String>,
        config: ShellToolConfig,
    ) -> Self {
        let os = std::env::consts::OS; // "linux", "windows", "macos"
        let shell = if cfg!(target_os = "windows") {
            "cmd.exe".to_string()
        } else {
            std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".into())
        };

        let discovery_hint = if instructions.is_none() {
            " When unsure which tool is available for a task, probe first with \
             `which tool1 tool2 tool3` to find what is installed, then use the one that exists."
        } else {
            ""
        };

        let custom = instructions.as_deref().unwrap_or("");
        let instructions_block = if custom.is_empty() {
            String::new()
        } else {
            format!(" System setup: {}", custom.trim())
        };

        let description = format!(
            "Run a shell command on this computer (OS: {os}, shell: {shell}). \
             Use it to check running apps, control media playback, adjust system \
             settings, open programs, manage files, and query system state. \
             NOTE: sudo is not available — use userspace tools directly.{discovery_hint}{instructions_block}"
        );
        Self {
            timeout_secs,
            description,
            config,
        }
    }
}

impl Default for ShellTool {
    fn default() -> Self {
        Self::new(30)
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory to run the command in"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: Value, _ctx: &super::ToolContext) -> Result<String> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if command.is_empty() {
            return Ok("Error: no command provided".into());
        }

        if let Some(reason) = blocked_reason(&command, &self.config.allowed_commands) {
            return Ok(format!("Error: {reason}"));
        }

        let working_dir = args
            .get("working_dir")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let timeout_secs = self.timeout_secs;
        let run = async move {
            #[cfg(target_os = "windows")]
            let mut cmd = Command::new("cmd");
            #[cfg(target_os = "windows")]
            cmd.args(["/C", &command]);

            #[cfg(not(target_os = "windows"))]
            let mut cmd = Command::new("sh");
            #[cfg(not(target_os = "windows"))]
            cmd.arg("-c").arg(&command);

            cmd.env_clear().envs(safe_env_vars());

            if let Some(ref dir) = working_dir {
                cmd.current_dir(dir);
            }

            let output = cmd.output().await;

            match output {
                Ok(out) => {
                    let stdout = truncate(String::from_utf8_lossy(&out.stdout).into_owned());
                    let stderr = truncate(String::from_utf8_lossy(&out.stderr).into_owned());
                    let exit = out.status.code().unwrap_or(-1);

                    if stdout.is_empty() && !stderr.is_empty() {
                        format!("exit={exit}\nstderr: {stderr}")
                    } else if !stderr.is_empty() {
                        format!("exit={exit}\n{stdout}\nstderr: {stderr}")
                    } else {
                        format!("exit={exit}\n{stdout}")
                    }
                }
                Err(e) => format!("Error: failed to spawn command: {e}"),
            }
        };

        Ok(
            match timeout(Duration::from_secs(timeout_secs), run).await {
                Ok(result) => result,
                Err(_) => format!("Error: command timed out after {timeout_secs}s"),
            },
        )
    }
}

/// Returns an error message if the command should be blocked, or `None` if it's allowed.
///
/// Checks each shell segment's first word against the allowlist.  An empty
/// `allowed_commands` slice means "allow all".  `sudo`/`doas` are always
/// blocked regardless of the allowlist as an additional safety net.
fn blocked_reason(command: &str, allowed_commands: &[String]) -> Option<String> {
    for segment in command.split([';', '|', '&']) {
        let word = segment.split_whitespace().next().unwrap_or("");
        if word.is_empty() {
            continue;
        }
        if word == "sudo" || word == "doas" {
            return Some(
                "sudo/doas are not available. Use userspace tools directly — \
                 e.g. `brightnessctl set 50%` for brightness, \
                 `playerctl play-pause` for media, \
                 `systemctl --user` for user-level services."
                    .to_string(),
            );
        }
        // Extract just the binary name from a possible absolute path (e.g. /usr/bin/sudo -> sudo).
        let binary = word.rsplit('/').next().unwrap_or(word);
        if !allowed_commands.is_empty() && !allowed_commands.iter().any(|a| a == binary) {
            return Some(format!(
                "Command '{binary}' is not in the allowed commands list. \
                 Allowed commands: {allowed}",
                allowed = allowed_commands.join(", ")
            ));
        }
    }
    None
}

/// Only pass through environment variables that are safe and useful.
/// Prevents API keys, tokens, and other secrets from leaking into subprocesses.
fn safe_env_vars() -> Vec<(String, String)> {
    #[cfg(target_os = "windows")]
    let safe: &[&str] = &[
        "PATH",
        "USERPROFILE",
        "USERNAME",
        "TEMP",
        "TMP",
        "APPDATA",
        "LOCALAPPDATA",
        "SYSTEMROOT",
        "COMSPEC",
        "PATHEXT",
    ];

    #[cfg(not(target_os = "windows"))]
    let safe: &[&str] = &[
        "PATH",
        "HOME",
        "USER",
        "SHELL",
        "TERM",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "XDG_RUNTIME_DIR",
        "XDG_SESSION_TYPE",
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "DBUS_SESSION_BUS_ADDRESS",
        "PULSE_SERVER",
    ];

    safe.iter()
        .filter_map(|k| std::env::var(k).ok().map(|v| (k.to_string(), v)))
        .collect()
}

fn truncate(s: String) -> String {
    if s.len() <= MAX_OUTPUT {
        return s;
    }
    // Walk back to a valid UTF-8 boundary
    let mut end = MAX_OUTPUT;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}... [truncated]", &s[..end])
}
