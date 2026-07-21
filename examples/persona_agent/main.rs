//! Persona-driven agent: magickmind persona + runtime integration with @mention gate.
//!
//! Demonstrates using a rich, versioned persona from the magickmind persona
//! service instead of a flat persona string. `PersonaContextBuilder` fetches a
//! server-assembled system prompt per-request (with dyadic per-user
//! adaptation) from the prepare endpoint.
//!
//! Configured with `persona.type = "magickmind-prepared"`, which is keyed by
//! `agent.agent_id` — not a persona id. The server resolves which persona the
//! agent uses. See the config files for the legacy client-side-blending path.
//!
//! The agent only responds when mentioned with `@name` or `@agent_id`.
//!
//! Flow:
//!   1. MentionGate — check for @mention, skip if not mentioned (no API calls)
//!   2. Context preparation — fetch MagickMind context (chat history, knowledge)
//!   3. PersonaContextBuilder — resolve the persona system prompt
//!   4. LLM processor — streaming inference
//!   5. PostProcessor — format response
//!   6. MagickmindPersistence — save response to MagickMind
//!
//! Run with:
//!   cargo run --example persona_agent --features full -- --config examples/persona_agent/dazael.toml

use std::sync::Arc;

use async_trait::async_trait;
use mindroid::llm_client::LlmClient;
use mindroid::pipeline::presets::magickmind::{
    MagickmindClient, MagickmindContext, MagickmindPersistence,
};
use mindroid::{
    ContextPreparer, GenericLlmProcessor, LlmMessage, MindroidConfig, Pipeline, PipelineContext,
    PipelineStage, PostProcessor, PrepareOutcome, Result, Runtime,
};

// ---------------------------------------------------------------------------
// MentionGate — halts the pipeline unless the agent is @mentioned
// ---------------------------------------------------------------------------

/// A simple pipeline stage that checks if the incoming message contains
/// an @mention of the agent's name or ID. Halts the pipeline if not mentioned.
///
/// Matches (case-insensitive):
///   - `@AgentName` or `@agent_name`
///   - `@agent_id`
struct MentionGate {
    /// Lowercase patterns to match (e.g. `["@trip planner", "@699530a01f0bdf56bf139ddc"]`)
    patterns: Vec<String>,
}

impl MentionGate {
    fn new(agent_name: &str, agent_id: &str) -> Self {
        let name_lower = agent_name.to_lowercase();
        let id_lower = agent_id.to_lowercase();

        let mut patterns = vec![
            format!("@{}", name_lower),   // @Agent Name
            format!("<@{}>", id_lower),   // <@agent_id> (Discord/Slack style)
            format!("<@{}>", name_lower), // <@agent name>
            format!("@{}", id_lower),     // @agent_id (plain)
        ];

        // Deduplicate
        patterns.sort();
        patterns.dedup();

        Self { patterns }
    }
}

#[async_trait]
impl PipelineStage for MentionGate {
    fn name(&self) -> &str {
        "MentionGate"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let content = ctx.message.content.to_lowercase();

        tracing::info!(
            "MentionGate: checking message content: {:?}",
            ctx.message.content
        );
        tracing::info!(
            "MentionGate: matching against patterns: {:?}",
            self.patterns
        );

        let mentioned = self.patterns.iter().any(|p| content.contains(p));

        if mentioned {
            tracing::info!("MentionGate: agent was @mentioned — proceeding");
        } else {
            tracing::info!("MentionGate: no @mention found — halting");
            ctx.halted = true;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SharedPersonaStage — delegates to Arc<PersonaContextBuilder>
// ---------------------------------------------------------------------------

/// Wraps `Arc<PersonaContextBuilder>` so it can be used as a `PipelineStage`
/// inside a `Pipeline`. This lets us share a single, pre-initialised persona
/// stage across requests without re-fetching the persona schema each time.
///
/// Per-request conversation history is injected by setting `self.history` in
/// a new `PersonaContextBuilder` built from the shared Arc — but since
/// `PersonaContextBuilder` doesn't implement `Clone`, we instead build the
/// respond pipeline per-request with history passed via `with_history()`.
///
/// Note: this struct is NOT used in the current example because the respond
/// pipeline is built per-request (see `main()`). It is kept here for reference.
#[cfg(any())]
struct SharedPersonaStage(Arc<mindroid::PersonaContextBuilder>);

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,mindroid=debug")
        .init();

    let config = MindroidConfig::resolve_from_args()?;

    // Extract LLM config before from_config consumes it
    let respond_llm = config.llm("respond")?;

    // Auto-build identity, transport, memory, observer from config
    let builder = Runtime::from_config(config)?;

    // MagickmindClient for context preparation and persistence
    let identity = builder.auth_arc().unwrap();
    let config = builder.config_ref().unwrap();
    let magickmind_url = config
        .auth
        .base_url
        .as_deref()
        .unwrap_or("https://dev-magickmind.magickmind.ai");
    let mut magickmind_client = MagickmindClient::new(magickmind_url, identity);
    if let Some(api_key) = &config.auth.api_key {
        magickmind_client = magickmind_client.with_api_key(api_key);
    }
    let magickmind = Arc::new(magickmind_client);

    // Mention gate: only respond when @mentioned
    let mention_gate = MentionGate::new(&config.agent.name, &config.agent.agent_id);

    // Context preparer: fetch chat history and knowledge from MagickMind
    let agent_id = config.agent.agent_id.clone();
    let context_preparer = Arc::new(
        ContextPreparer::new()
            .add_provider(MagickmindContext::new(magickmind.clone()).with_self_id(agent_id)),
    );

    // Gate pipeline: cheap @mention check (no API calls)
    let gate_pipeline = Arc::new(Pipeline::new().add_stage(mention_gate));

    // Build the persona stage once. On the prepared path nothing is fetched here —
    // the prompt is resolved per-request. The legacy path fetches the schema now.
    // The respond pipeline is built per-request so per-request history can be
    // injected into PersonaContextBuilder via with_history().
    let persona_client = builder.build_persona_stage().await?.expect(
        "persona_agent requires a [persona] config section \
             (type = \"magickmind-prepared\" with agent.agent_id, \
             or type = \"magickmind\" with persona.persona_id)",
    );

    // We need the underlying client/cache/id to rebuild per-request.
    // For simplicity, wrap the built PersonaContextBuilder in Arc and use it
    // directly as a stage (without per-request history injection). The persona
    // stage will use an empty history Arc — chat history is persisted via
    // MagickmindPersistence and the LLM's context window.
    let persona_stage = Arc::new(persona_client);
    let respond_llm = Arc::new(respond_llm);

    // Wire up the runtime
    let mut runtime = builder
        .on_message(move |ctx| {
            let preparer = Arc::clone(&context_preparer);
            let gate = Arc::clone(&gate_pipeline);
            let magickmind = Arc::clone(&magickmind);
            let respond_llm = Arc::clone(&respond_llm);
            let persona = Arc::clone(&persona_stage);

            async move {
                tracing::info!(
                    "Received message from {}: {:?}",
                    ctx.message.sender_id,
                    ctx.message.content
                );

                // Step 1: Check @mention first (cheap, no API calls)
                let mut pctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());

                if let Err(e) = ctx.run_with_context(&gate, &mut pctx).await {
                    tracing::error!("Mention gate failed: {e}");
                    return;
                }
                if pctx.halted {
                    tracing::info!("Not @mentioned — staying silent");
                    return;
                }
                tracing::info!("@mentioned — proceeding");

                // Step 2: Fetch context from MagickMind (only after mention check passes)
                let history = match preparer.prepare(&ctx.message).await {
                    PrepareOutcome::Complete(msgs) => Arc::new(msgs),
                    PrepareOutcome::Degraded { messages, warnings } => {
                        for w in &warnings {
                            tracing::warn!("Context provider '{}' failed: {}", w.provider, w.error);
                        }
                        Arc::new(messages)
                    }
                    PrepareOutcome::Failed(warnings) => {
                        for w in &warnings {
                            tracing::error!(
                                "Context provider '{}' failed: {}",
                                w.provider,
                                w.error
                            );
                        }
                        tracing::error!("All context providers failed — aborting");
                        return;
                    }
                };

                // Step 3: Build the respond pipeline per-request with history injected.
                // PersonaContextBuilder is wrapped in Arc — Pipeline::add_stage accepts
                // Arc<T> when T: PipelineStage.
                let respond_pipeline = Pipeline::new()
                    .add_stage(PersonaWithHistory {
                        persona: Arc::clone(&persona),
                        history,
                    })
                    .add_streaming_stage(match LlmClient::new((*respond_llm).clone()) {
                        Ok(c) => GenericLlmProcessor::new(c),
                        Err(e) => {
                            tracing::error!("LlmClient init failed: {e}");
                            return;
                        }
                    })
                    .add_stage(PostProcessor)
                    .add_stage(MagickmindPersistence::new(Arc::clone(&magickmind)));

                pctx.reset_output();

                match ctx.run_with_context(&respond_pipeline, &mut pctx).await {
                    Ok(None) => {
                        tracing::info!("No response generated");
                        return;
                    }
                    Ok(Some(_)) => {}
                    Err(e) => {
                        tracing::error!("Pipeline failed: {e}");
                        return;
                    }
                }

                let response = pctx.response.as_deref().unwrap_or("").trim().to_string();
                if !response.is_empty()
                    && let Err(e) = ctx.respond(&response).await
                {
                    tracing::error!("Failed to send response: {e}");
                }
            }
        })
        .build()?;

    runtime.run().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// PersonaWithHistory — delegates to a shared PersonaContextBuilder with
// per-request history injected by temporarily setting self.history.
// Since PersonaContextBuilder.process() reads self.history, we need a
// separate stage that wraps it and provides per-request history.
// ---------------------------------------------------------------------------

use mindroid::PersonaContextBuilder;

struct PersonaWithHistory {
    persona: Arc<PersonaContextBuilder>,
    history: Arc<Vec<LlmMessage>>,
}

#[async_trait]
impl PipelineStage for PersonaWithHistory {
    fn name(&self) -> &str {
        "PersonaWithHistory"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        // Build a temporary view with history by cloning and calling with_history.
        // PersonaContextBuilder fields are Arc-wrapped so cloning is cheap conceptually,
        // but PersonaContextBuilder doesn't implement Clone.
        // Instead, we manually set llm_messages using the persona stage's logic
        // by calling process() and then prepending history to llm_messages.
        self.persona.process(ctx).await?;

        // Inject history before the user message: insert after the system prompt.
        if !self.history.is_empty() {
            let system_msgs: Vec<_> = ctx.llm_messages.drain(..).collect();
            let mut new_messages = Vec::with_capacity(system_msgs.len() + self.history.len());
            // Keep system messages first
            let mut rest_start = 0;
            for (i, msg) in system_msgs.iter().enumerate() {
                if msg.role == "system".into() {
                    new_messages.push(msg.clone());
                    rest_start = i + 1;
                } else {
                    break;
                }
            }
            // Insert history
            new_messages.extend(self.history.iter().cloned());
            // Append remaining messages (the user message)
            new_messages.extend(system_msgs[rest_start..].iter().cloned());
            ctx.llm_messages = new_messages;
        }

        Ok(())
    }
}
