use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::error::{MindroidError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalUser {
    pub canonical_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    pub identities: Vec<PlatformIdentity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformIdentity {
    pub platform: String,
    pub platform_id: String,
    pub linked_at: DateTime<Utc>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct IdentityRegistry {
    users: HashMap<String, CanonicalUser>,
}

pub struct IdentityResolver {
    registry: RwLock<IdentityRegistry>,
    /// Reverse index: (platform, platform_id) → canonical_id
    index: RwLock<HashMap<(String, String), String>>,
    registry_path: PathBuf,
}

impl IdentityResolver {
    /// Load the identity registry from disk. If the file doesn't exist, start empty.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let registry: IdentityRegistry = if path.exists() {
            let content = std::fs::read_to_string(&path).map_err(|e| {
                MindroidError::config(format!("Failed to read identity registry: {e}"))
            })?;
            serde_json::from_str(&content).map_err(|e| {
                MindroidError::config(format!("Failed to parse identity registry: {e}"))
            })?
        } else {
            IdentityRegistry::default()
        };

        // Build reverse index
        let mut index = HashMap::new();
        for user in registry.users.values() {
            for identity in &user.identities {
                index.insert(
                    (identity.platform.clone(), identity.platform_id.clone()),
                    user.canonical_id.clone(),
                );
            }
        }

        Ok(Self {
            registry: RwLock::new(registry),
            index: RwLock::new(index),
            registry_path: path,
        })
    }

    /// Resolve a (platform, platform_id) pair to a canonical user ID.
    ///
    /// If no mapping exists, auto-creates a new canonical user and persists it.
    pub async fn resolve(&self, platform: &str, platform_id: &str) -> String {
        // Fast path: check index under read lock
        {
            let index = self.index.read().await;
            if let Some(canonical_id) = index.get(&(platform.to_string(), platform_id.to_string()))
            {
                return canonical_id.clone();
            }
        }

        // Slow path: create new canonical user
        let short_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        let canonical_id = short_id.clone();

        let identity = PlatformIdentity {
            platform: platform.to_string(),
            platform_id: platform_id.to_string(),
            linked_at: Utc::now(),
        };

        let user = CanonicalUser {
            canonical_id: canonical_id.clone(),
            display_name: None,
            identities: vec![identity],
        };

        {
            let mut registry = self.registry.write().await;
            registry.users.insert(canonical_id.clone(), user);
        }
        {
            let mut index = self.index.write().await;
            index.insert(
                (platform.to_string(), platform_id.to_string()),
                canonical_id.clone(),
            );
        }

        if let Err(e) = self.persist().await {
            tracing::warn!("IdentityResolver: failed to persist registry: {e}");
        }

        canonical_id
    }

    /// Add a platform identity to an existing canonical user.
    pub async fn link(&self, canonical_id: &str, platform: &str, platform_id: &str) -> Result<()> {
        let identity = PlatformIdentity {
            platform: platform.to_string(),
            platform_id: platform_id.to_string(),
            linked_at: Utc::now(),
        };

        {
            let mut registry = self.registry.write().await;
            let user = registry.users.get_mut(canonical_id).ok_or_else(|| {
                MindroidError::config(format!(
                    "IdentityResolver: canonical user '{}' not found",
                    canonical_id
                ))
            })?;
            // Avoid duplicate entries
            let already_linked = user
                .identities
                .iter()
                .any(|i| i.platform == platform && i.platform_id == platform_id);
            if !already_linked {
                user.identities.push(identity);
            }
        }
        {
            let mut index = self.index.write().await;
            index.insert(
                (platform.to_string(), platform_id.to_string()),
                canonical_id.to_string(),
            );
        }

        self.persist().await
    }

    /// Load pre-configured identity links from config.
    ///
    /// Format: canonical_name → ["platform:platform_id", ...]
    /// If canonical_name doesn't exist in the registry, it is created.
    pub fn load_config_links(&mut self, links: &HashMap<String, Vec<String>>) {
        let registry = self.registry.get_mut();
        let index = self.index.get_mut();

        for (canonical_name, platform_ids) in links {
            // Ensure canonical user exists
            let user = registry
                .users
                .entry(canonical_name.clone())
                .or_insert_with(|| CanonicalUser {
                    canonical_id: canonical_name.clone(),
                    display_name: Some(canonical_name.clone()),
                    identities: Vec::new(),
                });

            for entry in platform_ids {
                // Parse "platform:platform_id"
                if let Some((platform, platform_id)) = entry.split_once(':') {
                    let already_linked = user
                        .identities
                        .iter()
                        .any(|i| i.platform == platform && i.platform_id == platform_id);
                    if !already_linked {
                        user.identities.push(PlatformIdentity {
                            platform: platform.to_string(),
                            platform_id: platform_id.to_string(),
                            linked_at: Utc::now(),
                        });
                    }
                    index.insert(
                        (platform.to_string(), platform_id.to_string()),
                        canonical_name.clone(),
                    );
                } else {
                    tracing::warn!(
                        "IdentityResolver: invalid link entry '{}' for '{}' (expected 'platform:id')",
                        entry,
                        canonical_name
                    );
                }
            }
        }
    }

    /// Persist the registry to disk atomically (write to .tmp, then rename).
    async fn persist(&self) -> Result<()> {
        let registry = self.registry.read().await;
        let json = serde_json::to_string_pretty(&*registry).map_err(|e| {
            MindroidError::config(format!("Failed to serialize identity registry: {e}"))
        })?;
        drop(registry);

        // Ensure parent directory exists
        if let Some(parent) = self.registry_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                MindroidError::config(format!("Failed to create identity registry directory: {e}"))
            })?;
        }

        let tmp_path = self.registry_path.with_extension("json.tmp");
        tokio::fs::write(&tmp_path, json.as_bytes())
            .await
            .map_err(|e| {
                MindroidError::config(format!("Failed to write identity registry tmp file: {e}"))
            })?;
        tokio::fs::rename(&tmp_path, &self.registry_path)
            .await
            .map_err(|e| {
                MindroidError::config(format!("Failed to rename identity registry file: {e}"))
            })?;

        Ok(())
    }
}
