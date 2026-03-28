use std::collections::HashMap;

use tokio::sync::RwLock;
use tokio::time::Instant;

use super::models::EffectivePersonalityResponse;

/// In-memory TTL cache for effective personality responses.
///
/// Keys are `"{persona_id}:{user_id}"` (or `"{persona_id}:"` when no user_id).
/// TTL is derived from the server's `ttl_seconds` field on each response.
pub struct PersonaCache {
    entries: RwLock<HashMap<String, CacheEntry>>,
}

struct CacheEntry {
    response: EffectivePersonalityResponse,
    expires_at: Instant,
}

impl PersonaCache {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
        }
    }

    /// Look up a cached effective personality. Returns `None` on miss or expiry.
    ///
    /// Expired entries are evicted eagerly on access.
    pub async fn get(
        &self,
        persona_id: &str,
        user_id: Option<&str>,
    ) -> Option<EffectivePersonalityResponse> {
        let key = cache_key(persona_id, user_id);
        let now = Instant::now();

        // Fast path: read lock only.
        {
            let entries = self.entries.read().await;
            match entries.get(&key) {
                Some(entry) if entry.expires_at > now => return Some(entry.response.clone()),
                None => return None,
                _ => {} // entry exists but is expired — fall through to evict
            }
        }

        // Entry is expired: upgrade to write lock and remove it.
        let mut entries = self.entries.write().await;
        // Re-check under write lock in case another task refreshed it.
        if entries.get(&key).is_some_and(|e| e.expires_at > now) {
            return Some(entries[&key].response.clone());
        }
        entries.remove(&key);
        None
    }

    /// Store an effective personality response, using the server's `ttl_seconds`.
    ///
    /// Skips insertion when `ttl_seconds <= 0` (server says don't cache).
    pub async fn set(&self, response: EffectivePersonalityResponse) {
        if response.ttl_seconds == 0 {
            return;
        }

        let key = cache_key(&response.persona_id, response.user_id.as_deref());
        let ttl = std::time::Duration::from_secs(response.ttl_seconds);
        let entry = CacheEntry {
            response,
            expires_at: Instant::now() + ttl,
        };

        let mut entries = self.entries.write().await;

        // Evict expired entries when cache grows beyond threshold
        if entries.len() > 200 {
            let now = Instant::now();
            entries.retain(|_, e| e.expires_at > now);
        }

        entries.insert(key, entry);
    }

    /// Explicitly invalidate a cached entry.
    #[allow(dead_code)]
    pub(crate) async fn invalidate(&self, persona_id: &str, user_id: Option<&str>) {
        let key = cache_key(persona_id, user_id);
        let mut entries = self.entries.write().await;
        entries.remove(&key);
    }
}

impl Default for PersonaCache {
    fn default() -> Self {
        Self::new()
    }
}

fn cache_key(persona_id: &str, user_id: Option<&str>) -> String {
    format!("{}:{}", persona_id, user_id.unwrap_or(""))
}
