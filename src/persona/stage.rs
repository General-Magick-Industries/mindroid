use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tokio::time::Instant;
use tracing::debug;

use crate::core::context::Context;
use crate::error::Result;
use crate::models::{LlmMessage, SenderType};
use crate::pipeline::PipelineStage;
#[cfg(feature = "transport-audio")]
use crate::pipeline::extensions::TextInput;

use super::cache::PersonaCache;
use super::models::PersonaSchema;
use super::prompt::build_system_prompt;
use super::provider::PersonaProvider;

/// A pipeline stage that resolves a system prompt for the current message and
/// assembles `ctx.llm_messages`.
///
/// Replaces `SimpleContextBuilder` when a persona is configured. The prompt
/// comes from one of two sources, decided per provider:
///
/// - **Prepared** — the provider returns a server-assembled prompt from
///   [`PersonaProvider::prepared_prompt`], used verbatim.
/// - **Assembled** — the provider supplies a schema and blended traits, which
///   [`build_system_prompt`] composes client-side.
///
/// Either way the sender's user_id is passed through for dyadic (per-user)
/// adaptation.
pub struct PersonaContextBuilder {
    provider: Arc<dyn PersonaProvider>,
    cache: Arc<PersonaCache>,
    /// TTL cache for prepared prompts, keyed by `"{id}:{user_id}"`.
    ///
    /// Keyed by the id this stage was built with — an agent id on the prepared
    /// path — never by the `persona_id` the server resolves to. Two agents can
    /// share a persona, so persona-keyed entries would serve one agent's prompt
    /// to another, potentially across tenants.
    prompt_cache: Arc<RwLock<HashMap<String, (String, Instant)>>>,
    persona_id: String,
    /// Static persona info, fetched once. `None` for providers that return a
    /// prepared prompt — they have no schema to fetch.
    persona_info: Option<PersonaSchema>,
    /// Pre-fetched conversation history injected at construction time.
    history: Arc<Vec<LlmMessage>>,
}

impl PersonaContextBuilder {
    /// Create a new `PersonaContextBuilder`.
    ///
    /// `id` is a persona id for providers that assemble client-side, and an
    /// **agent id** for providers backed by the prepare endpoint.
    ///
    /// Probes the provider once to learn which path it uses. For assembling
    /// providers this also fetches the persona schema (name, role,
    /// background_story, tones); for prepared providers it fetches nothing and
    /// the prompt is resolved per-request in `process()`.
    pub async fn new(provider: Arc<dyn PersonaProvider>, id: &str) -> Result<Self> {
        // A prepared provider has no schema to fetch.
        let persona_info = if provider.is_prepared() {
            None
        } else {
            Some(provider.get_persona(id).await?)
        };

        Ok(Self {
            provider,
            cache: Arc::new(PersonaCache::new()),
            prompt_cache: Arc::new(RwLock::new(HashMap::new())),
            persona_id: id.to_string(),
            persona_info,
            history: Arc::new(Vec::new()),
        })
    }

    /// Provide conversation history for inclusion in the LLM prompt.
    pub fn with_history(mut self, history: Arc<Vec<LlmMessage>>) -> Self {
        self.history = history;
        self
    }

    /// Look up a cached prompt, evicting it if expired.
    async fn cached_prompt(&self, key: &str) -> Option<String> {
        let now = Instant::now();

        {
            let entries = self.prompt_cache.read().await;
            match entries.get(key) {
                Some((prompt, expires_at)) if *expires_at > now => return Some(prompt.clone()),
                None => return None,
                _ => {}
            }
        }

        let mut entries = self.prompt_cache.write().await;
        if let Some((prompt, expires_at)) = entries.get(key)
            && *expires_at > now
        {
            return Some(prompt.clone());
        }
        entries.remove(key);
        None
    }

    /// Store a prompt under `key` for `ttl_seconds`.
    async fn store_prompt(&self, key: String, prompt: &str, ttl_seconds: u64) {
        let expires_at = Instant::now() + std::time::Duration::from_secs(ttl_seconds);
        let mut entries = self.prompt_cache.write().await;

        if entries.len() > 200 {
            let now = Instant::now();
            entries.retain(|_, (_, exp)| *exp > now);
        }

        entries.insert(key, (prompt.to_string(), expires_at));
    }

    /// Resolve the system prompt for this request, via whichever path the
    /// provider supports.
    async fn resolve_system_prompt(&self, user_id: Option<&str>) -> Result<String> {
        if self.persona_info.is_none() {
            let key = format!("{}:{}", self.persona_id, user_id.unwrap_or(""));

            if let Some(prompt) = self.cached_prompt(&key).await {
                debug!(
                    "PersonaContextBuilder: prompt cache hit for id={} user={:?}",
                    self.persona_id, user_id
                );
                return Ok(prompt);
            }

            if let Some(prepared) = self
                .provider
                .prepared_prompt(&self.persona_id, user_id)
                .await?
            {
                debug!(
                    "PersonaContextBuilder: prepared prompt for id={} user={:?} (ttl={}s)",
                    self.persona_id, user_id, prepared.ttl_seconds
                );
                if prepared.ttl_seconds > 0 {
                    self.store_prompt(key, &prepared.system_prompt, prepared.ttl_seconds)
                        .await;
                }
                return Ok(prepared.system_prompt);
            }
        }

        let effective = if let Some(cached) = self.cache.get(&self.persona_id, user_id).await {
            debug!(
                "PersonaContextBuilder: cache hit for persona={} user={:?}",
                self.persona_id, user_id
            );
            cached
        } else {
            debug!(
                "PersonaContextBuilder: cache miss, fetching effective personality for persona={} user={:?}",
                self.persona_id, user_id
            );
            let resp = self
                .provider
                .get_effective_personality(&self.persona_id, user_id)
                .await?;
            self.cache.set(resp.clone()).await;
            resp
        };

        let persona =
            self.persona_info
                .as_ref()
                .ok_or_else(|| crate::error::MindroidError::Api {
                    message: "persona schema unavailable and provider returned no prepared prompt"
                        .to_string(),
                    status_code: None,
                })?;

        Ok(build_system_prompt(persona, &effective.traits))
    }
}

#[async_trait]
impl PipelineStage for PersonaContextBuilder {
    fn name(&self) -> &str {
        "PersonaContextBuilder"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        // Determine user_id for dyadic blending (only for user-sent messages)
        // Prefer canonical user ID from identity resolution if available
        let canonical_id: Option<String>;
        #[cfg(feature = "identity")]
        {
            canonical_id = ctx
                .get_ext::<crate::identity::CanonicalUserId>()
                .map(|c| c.0.clone());
        }
        #[cfg(not(feature = "identity"))]
        {
            canonical_id = None;
        }

        let user_id = if ctx.message.sender_type == SenderType::User {
            canonical_id
                .as_deref()
                .or(Some(ctx.message.sender_id.as_str()))
        } else {
            None
        };

        let system_prompt = self.resolve_system_prompt(user_id).await?;

        // Assemble LLM messages: system prompt first, then history, then user message
        let mut messages = vec![LlmMessage::system(&system_prompt)];
        messages.extend(self.history.as_ref().clone());

        #[cfg(feature = "transport-audio")]
        let user_text = ctx
            .get_ext::<TextInput>()
            .map(|t| t.0.as_str())
            .unwrap_or(&ctx.message.content);
        #[cfg(not(feature = "transport-audio"))]
        let user_text = &ctx.message.content;

        messages.push(LlmMessage::user(user_text));

        debug!(
            "PersonaContextBuilder: {} history messages, {} total llm_messages",
            self.history.len(),
            messages.len(),
        );

        ctx.llm_messages = messages;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persona::models::{EffectivePersonalityResponse, PersonaSchema};
    use crate::persona::provider::PreparedPrompt;

    struct PreparedProvider;

    #[async_trait]
    impl PersonaProvider for PreparedProvider {
        fn name(&self) -> &str {
            "test-prepared"
        }
        fn is_prepared(&self) -> bool {
            true
        }
        async fn get_persona(&self, _id: &str) -> Result<PersonaSchema> {
            panic!("prepared provider must not fetch a schema")
        }
        async fn get_effective_personality(
            &self,
            _id: &str,
            _user_id: Option<&str>,
        ) -> Result<EffectivePersonalityResponse> {
            panic!("prepared provider must not fetch traits")
        }
        async fn prepared_prompt(
            &self,
            id: &str,
            _user_id: Option<&str>,
        ) -> Result<Option<PreparedPrompt>> {
            Ok(Some(PreparedPrompt {
                system_prompt: format!("server prompt for {id}"),
                ttl_seconds: 60,
            }))
        }
    }

    struct AssemblingProvider;

    #[async_trait]
    impl PersonaProvider for AssemblingProvider {
        fn name(&self) -> &str {
            "test-assembling"
        }
        async fn get_persona(&self, id: &str) -> Result<PersonaSchema> {
            Ok(PersonaSchema {
                id: id.into(),
                artifact_id: None,
                name: "Aria".into(),
                role: "guide".into(),
                traits: Vec::new(),
                tones: Vec::new(),
                background_story: String::new(),
                created_by: String::new(),
                updated_by: String::new(),
                active_version: None,
            })
        }
        async fn get_effective_personality(
            &self,
            id: &str,
            user_id: Option<&str>,
        ) -> Result<EffectivePersonalityResponse> {
            Ok(EffectivePersonalityResponse {
                persona_id: id.into(),
                user_id: user_id.map(String::from),
                traits: Vec::new(),
                computed_at: "2026-01-01T00:00:00Z".into(),
                ttl_seconds: 0,
            })
        }
    }

    #[tokio::test]
    async fn prepared_provider_uses_server_prompt_verbatim() {
        let stage = PersonaContextBuilder::new(Arc::new(PreparedProvider), "agent-1")
            .await
            .unwrap();
        let prompt = stage.resolve_system_prompt(Some("u1")).await.unwrap();
        assert_eq!(prompt, "server prompt for agent-1");
    }

    struct CountingPreparedProvider {
        calls: std::sync::atomic::AtomicUsize,
        ttl_seconds: u64,
    }

    #[async_trait]
    impl PersonaProvider for CountingPreparedProvider {
        fn name(&self) -> &str {
            "test-counting"
        }
        fn is_prepared(&self) -> bool {
            true
        }
        async fn get_persona(&self, _id: &str) -> Result<PersonaSchema> {
            unreachable!()
        }
        async fn get_effective_personality(
            &self,
            _id: &str,
            _user_id: Option<&str>,
        ) -> Result<EffectivePersonalityResponse> {
            unreachable!()
        }
        async fn prepared_prompt(
            &self,
            _id: &str,
            _user_id: Option<&str>,
        ) -> Result<Option<PreparedPrompt>> {
            let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(Some(PreparedPrompt {
                system_prompt: format!("prompt #{n}"),
                ttl_seconds: self.ttl_seconds,
            }))
        }
    }

    #[tokio::test]
    async fn prepared_prompt_is_cached_within_ttl() {
        let provider = Arc::new(CountingPreparedProvider {
            calls: Default::default(),
            ttl_seconds: 300,
        });
        let stage = PersonaContextBuilder::new(provider.clone(), "agent-1")
            .await
            .unwrap();

        let first = stage.resolve_system_prompt(Some("u1")).await.unwrap();
        let second = stage.resolve_system_prompt(Some("u1")).await.unwrap();

        assert_eq!(first, "prompt #0");
        assert_eq!(second, "prompt #0", "second call should hit the cache");
        assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn prompt_cache_is_scoped_per_user() {
        let provider = Arc::new(CountingPreparedProvider {
            calls: Default::default(),
            ttl_seconds: 300,
        });
        let stage = PersonaContextBuilder::new(provider.clone(), "agent-1")
            .await
            .unwrap();

        let a = stage.resolve_system_prompt(Some("alice")).await.unwrap();
        let b = stage.resolve_system_prompt(Some("bob")).await.unwrap();

        assert_ne!(a, b, "distinct users must not share a cached prompt");
        assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn zero_ttl_disables_caching() {
        let provider = Arc::new(CountingPreparedProvider {
            calls: Default::default(),
            ttl_seconds: 0,
        });
        let stage = PersonaContextBuilder::new(provider.clone(), "agent-1")
            .await
            .unwrap();

        stage.resolve_system_prompt(Some("u1")).await.unwrap();
        stage.resolve_system_prompt(Some("u1")).await.unwrap();

        assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn assembling_provider_builds_prompt_client_side() {
        let stage = PersonaContextBuilder::new(Arc::new(AssemblingProvider), "persona-1")
            .await
            .unwrap();
        let prompt = stage.resolve_system_prompt(Some("u1")).await.unwrap();
        assert_eq!(prompt, "You are Aria, a guide.");
    }
}
