use std::path::PathBuf;
use std::sync::Arc;

use crate::auth::Auth;
use crate::config::MindroidConfig;
use crate::error::{MindroidError, Result};
use crate::memory::Memory;
use crate::observer::Observer;
use crate::pipeline::Pipeline;
use crate::transport::Transport;

/// Expand a leading `~/` to the user's home directory.
pub(crate) fn expand_tilde(path: &str) -> PathBuf {
    if path.starts_with("~/") {
        dirs::home_dir()
            .map(|h| h.join(&path[2..]))
            .unwrap_or_else(|| PathBuf::from(path))
    } else {
        PathBuf::from(path)
    }
}

/// The [`CredentialKind`] implied by `auth.type`: `"enduser"` acts as an end
/// user, everything else as a service user. Pass it to adapters (the Centrifugo
/// transport via `with_credential_kind`, the magickmind clients via
/// `with_credential_kind`) so route and connect behavior follow the credential.
///
/// [`CredentialKind`]: crate::models::CredentialKind
pub fn credential_kind_from_config(config: &MindroidConfig) -> crate::models::CredentialKind {
    if config.auth.auth_type.as_deref() == Some("enduser") {
        crate::models::CredentialKind::EndUser
    } else {
        crate::models::CredentialKind::ServiceUser
    }
}

pub(crate) fn build_auth(config: &MindroidConfig) -> Result<Arc<dyn Auth>> {
    let auth_type = config.auth.auth_type.as_deref().unwrap_or("static");

    match auth_type {
        "static" => {
            let token = config.auth.token.as_deref().unwrap_or("dev-token");
            Ok(Arc::new(crate::auth::static_id::StaticAuth::new(token)))
        }

        // A pre-minted end-user JWT (from `POST /v1/end-users/tokens`). Carried
        // as a bearer token like `static`, but named separately so the runtime
        // routes magickmind calls to the end-user API surface, where the agent
        // is the token subject rather than a path parameter.
        //
        // Gated on `magickmind`: that routing is the only thing distinguishing
        // this from `static`, and it exists only in backend-gated code. Without
        // the feature the variant would be silently meaningless, so reject it.
        #[cfg(feature = "magickmind")]
        "enduser" => {
            // A pre-minted end-user JWT. The end-user vs service-user routing
            // decision is made by the transport and magickmind clients from
            // `auth.type`, not here.
            let token =
                config.auth.token.as_deref().ok_or_else(|| {
                    MindroidError::config("auth.token is required for enduser auth")
                })?;

            // Always EndUserAuth — `rotate_token` chooses how the credential is
            // kept fresh, not what type holds it. A plain static holder would
            // report `ServiceUser` (routing this token to the wrong surface) and
            // would never latch a rejection, so a supervisor that stops
            // redelivering yields an agent that 401s every call while reporting
            // healthy.
            let base_url = config.auth.base_url.as_deref().ok_or_else(|| {
                MindroidError::config(
                    "auth.base_url is required for enduser auth (it is the host the \
                     credential belongs to, and where rotation posts)",
                )
            })?;

            let mut auth = crate::auth::enduser::EndUserAuth::try_new(
                base_url,
                token,
                config.auth.token_expires_at.as_deref(),
                config.auth.allow_insecure,
            )?;
            if let Some(ttl) = config.auth.token_ttl_seconds {
                auth = auth.with_ttl_seconds(ttl);
            }
            if let Some(path) = &config.auth.token_file {
                auth = auth.with_token_file(path);
            }

            if config.auth.rotate_token {
                // Self-refresh: a supervised token is barred from the refresh
                // route, so that pairing is a guaranteed 403 — terminal. Fail at
                // boot rather than an hour in.
                auth.ensure_self_refreshable()?;
            } else {
                // Supervised: a control plane re-mints and redelivers.
                auth = auth.supervised();
            }
            Ok(Arc::new(auth))
        }

        #[cfg(feature = "apikey")]
        "apikey" => {
            let base_url = config.auth.base_url.as_deref().ok_or_else(|| {
                MindroidError::config("auth.base_url is required for apikey auth")
            })?;
            let email = config.auth.email.as_deref().unwrap_or("");
            let password = config.auth.password.as_deref().unwrap_or("");
            Ok(Arc::new(crate::auth::apikey::ApiKeyAuth::try_new(
                base_url,
                email,
                password,
                config.auth.allow_insecure,
            )?))
        }

        other => Err(MindroidError::config(format!(
            "unknown or disabled auth type: '{other}'"
        ))),
    }
}

pub(crate) fn build_transport(
    config: &MindroidConfig,
    auth: &Arc<dyn Auth>,
) -> Result<Box<dyn Transport>> {
    let _ = auth; // used in feature-gated branches (centrifugo)
    let t_type = config
        .transport
        .transport_type
        .as_deref()
        .unwrap_or("stdio");

    match t_type {
        "stdio" => Ok(Box::new(crate::transport::stdio::StdioTransport::new())),

        #[cfg(feature = "transport-ws")]
        "centrifugo" => {
            let url = config.transport.url.as_deref().ok_or_else(|| {
                MindroidError::config("transport.url is required for centrifugo transport")
            })?;
            let agent_id = &config.agent.agent_id;
            let kind = credential_kind_from_config(config);
            Ok(Box::new(
                crate::transport::centrifugo::CentrifugoTransport::new(url, agent_id, auth.clone())
                    .with_credential_kind(kind)
                    .with_allow_insecure(config.transport.allow_insecure),
            ))
        }

        other => Err(MindroidError::config(format!(
            "unknown or disabled transport type: '{other}'"
        ))),
    }
}

#[allow(dead_code)] // Will be used when memory field is restored on Runtime
pub(crate) fn build_memory(
    config: &MindroidConfig,
    auth: &Arc<dyn Auth>,
) -> Result<Box<dyn Memory>> {
    let _ = auth; // used in feature-gated branches (magickmind)
    let m_type = config.memory.memory_type.as_deref().unwrap_or("none");

    match m_type {
        "none" => Ok(Box::new(crate::memory::NoMemory)),

        #[cfg(feature = "persistence")]
        "magickmind" => {
            let base_url = config
                .memory
                .base_url
                .as_deref()
                .or(config.auth.base_url.as_deref())
                .ok_or_else(|| {
                    MindroidError::config(
                        "memory.base_url (or auth.base_url) is required for magickmind memory",
                    )
                })?;
            Ok(Box::new(crate::memory::magickmind::MagickmindMemory::new(
                base_url,
                auth.clone(),
            )))
        }

        #[cfg(feature = "persistence")]
        "sqlite" => {
            let path = config.memory.path.as_deref().unwrap_or("./mindroid.db");
            Ok(Box::new(crate::memory::sqlite::SqliteMemory::new(path)?))
        }

        other => Err(MindroidError::config(format!(
            "unknown or disabled memory type: '{other}'"
        ))),
    }
}

/// Build the artifact store from config. Returns `Arc` (not `Box`) because the
/// store is shared into BOTH the persistence stage AND the load tool/executor.
#[cfg(feature = "artifacts")]
pub fn build_artifact_store(
    config: &MindroidConfig,
) -> Result<Arc<dyn crate::artifacts::ArtifactStore>> {
    let a_type = config.artifacts.artifacts_type.as_deref().unwrap_or("none");
    match a_type {
        "none" => Ok(Arc::new(crate::artifacts::NoArtifactStore)),
        "local" => {
            let path = config
                .artifacts
                .path
                .as_deref()
                .unwrap_or("./mindroid-artifacts");
            Ok(Arc::new(crate::artifacts::LocalArtifactStore::new(
                expand_tilde(path),
            )))
        }
        other => Err(MindroidError::config(format!(
            "unknown or disabled artifacts type: '{other}'"
        ))),
    }
}

pub(crate) fn build_observers(config: &MindroidConfig) -> Vec<Box<dyn Observer>> {
    let mut observers: Vec<Box<dyn Observer>> = Vec::new();
    let o_type = config.observer.observer_type.as_deref().unwrap_or("none");

    if o_type == "log" {
        observers.push(Box::new(crate::observer::log::LogObserver::new()));
    }

    observers
}

pub(crate) fn build_pipeline(
    config: &MindroidConfig,
    auth: &Arc<dyn Auth>,
) -> Result<Option<Pipeline>> {
    let _ = auth; // used in feature-gated branches (magickmind)
    let p_type = match config.pipeline.pipeline_type.as_deref() {
        Some(t) => t,
        None => return Ok(None),
    };

    match p_type {
        #[cfg(feature = "llm-local")]
        "ollama" => {
            let base_url = config
                .pipeline
                .base_url
                .as_deref()
                .unwrap_or("http://localhost:11434");
            let model = config.pipeline.model.as_deref().unwrap_or("llama3.2");
            Ok(Some(crate::pipeline::presets::ollama::ollama_pipeline(
                base_url, model,
            )?))
        }

        #[cfg(feature = "llm-hosted")]
        "magickmind" => {
            let base_url = config
                .pipeline
                .base_url
                .as_deref()
                .or(config.auth.base_url.as_deref())
                .ok_or_else(|| {
                    MindroidError::config("pipeline.base_url is required for magickmind pipeline")
                })?;
            let api_key = config
                .pipeline
                .api_key
                .as_deref()
                .or(config.auth.api_key.as_deref())
                .unwrap_or("");
            Ok(Some(
                crate::pipeline::presets::magickmind::magickmind_pipeline(
                    auth.clone(),
                    base_url,
                    api_key,
                    config.agent.compute_power,
                )?,
            ))
        }

        #[cfg(feature = "llm-hosted")]
        "magickmind_ollama" => {
            let magickmind_url = config
                .pipeline
                .base_url
                .as_deref()
                .or(config.auth.base_url.as_deref())
                .ok_or_else(|| {
                    MindroidError::config(
                        "pipeline.base_url is required for magickmind_ollama pipeline",
                    )
                })?;
            let ollama_url = config
                .pipeline
                .options
                .get("ollama_url")
                .and_then(|v| v.as_str())
                .unwrap_or("http://localhost:11434");
            let model = config.pipeline.model.as_deref().unwrap_or("llama3.2");
            Ok(Some(
                crate::pipeline::presets::magickmind::magickmind_ollama_pipeline(
                    auth.clone(),
                    magickmind_url,
                    ollama_url,
                    model,
                )?,
            ))
        }

        other => Err(MindroidError::config(format!(
            "unknown or disabled pipeline type: '{other}'"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_tilde_with_home() {
        let result = expand_tilde("~/foo/bar");
        // Should not start with ~/ anymore
        assert!(!result.to_str().unwrap().starts_with("~/"));
        assert!(result.to_str().unwrap().ends_with("foo/bar"));
    }

    #[test]
    fn test_expand_tilde_absolute() {
        assert_eq!(expand_tilde("/abs/path"), PathBuf::from("/abs/path"));
    }

    #[test]
    fn test_expand_tilde_relative() {
        assert_eq!(expand_tilde("relative"), PathBuf::from("relative"));
    }
}
