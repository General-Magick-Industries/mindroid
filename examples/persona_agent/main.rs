//! Persona-driven agent: magickmind persona + runtime integration with @mention gate.
//!
//! Demonstrates using a rich, versioned persona from the magickmind persona
//! service instead of a flat persona string. The `PersonaContextBuilder`
//! fetches the effective personality per-request (with dyadic per-user
//! adaptation) and builds a structured system prompt from blended traits.
//!
//! The agent only responds when mentioned with `@name` or `@agent_id`.
//!
//! Flow:
//!   1. MentionGate — check for @mention, skip if not mentioned (no API calls)
//!   2. Context preparation — fetch MagickMind context (chat history, knowledge)
//!   3. PersonaContextBuilder — fetch effective personality, build system prompt
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
    ContextPreparer, ConversationHistory, GenericLlmProcessor, MindroidConfig, Pipeline,
    PipelineContext, PipelineStage, PostProcessor, PrepareOutcome, Result, Runtime,
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
// SharedStage — lets an Arc<dyn PipelineStage> be added to a Pipeline
// ---------------------------------------------------------------------------

/// Wraps `Arc<dyn PipelineStage>` so a single shared stage instance (here the
/// persona stage, which owns a prompt cache worth preserving across requests)
/// can be added to a `Pipeline` built once at startup.
struct SharedStage(Arc<dyn PipelineStage>);

#[async_trait]
impl PipelineStage for SharedStage {
    fn name(&self) -> &str {
        self.0.name()
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        self.0.process(ctx).await
    }
}

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
        .unwrap_or("https://magickmind.example.com");
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

    // Local test toggle: set MINDROID_DISABLE_GATE=1 (or =true) to answer every
    // message on the channel (no @mention required) — e.g. plain messages from a
    // website magickspace. Any other value (or unset) keeps the @mention gate.
    let gate_enabled =
        !std::env::var("MINDROID_DISABLE_GATE").is_ok_and(|v| matches!(v.as_str(), "1" | "true"));

    // Build the persona stage once. Three modes are supported via [persona] config:
    //   type = "magickmind"                → PersonaContextBuilder formats the prompt in-process
    //   type = "magickmind-prepared"       → MagickmindPersonaStage delegates prompt construction
    //                                        to the server (persona-keyed) and uses the returned
    //                                        system_prompt verbatim
    //   type = "magickmind-prepared-agent" → MagickmindAgentPersonaStage, same server-prepared
    //                                        prompt but agent-keyed; the route follows auth.type
    //                                        (service-user names agent_id, "enduser" uses the
    //                                        id-less end-user route)
    // All implement PipelineStage, so we hold whichever is configured as a
    // trait object. Per-request conversation history is injected via the
    // ConversationHistory context extension, so the respond pipeline is built
    // once and shared across requests.
    let persona_stage: Arc<dyn PipelineStage> = if let Some(agent_prepared) =
        builder.build_magickmind_agent_persona_stage()
    {
        tracing::info!(
            "Using agent-scoped prepared persona stage (type = \"magickmind-prepared-agent\")"
        );
        Arc::new(agent_prepared)
    } else if let Some(prepared) = builder.build_magickmind_persona_stage() {
        tracing::info!("Using MagickMind-prepared persona stage (type = \"magickmind-prepared\")");
        Arc::new(prepared)
    } else {
        let persona_client = builder.build_persona_stage().await?.expect(
                "persona_agent requires [persona] config with type = \"magickmind\", \"magickmind-prepared\", or \"magickmind-prepared-agent\"",
            );
        tracing::info!("Using magickmind persona stage (type = \"magickmind\")");
        Arc::new(persona_client)
    };

    // Episode ingest (optional, [episodes] config). The inbound stage runs
    // before the mention gate so EVERY received message is remembered, not just
    // ones the agent replies to. The reply stage runs after generation to
    // capture the agent's own message, which mindroid drops from the inbound
    // path (runtime skips the agent's own messages to avoid loops).
    let inbound_ingest = builder.build_episode_ingest_stage()?.map(Arc::new);
    if inbound_ingest.is_some() {
        tracing::info!("Episode ingest enabled");
    }

    // Respond pipeline, built once: persona → LLM → post-process →
    // (optional) reply ingest → persist.
    //
    // Reply ingest goes BEFORE persistence: a stage returning Err aborts the
    // run, and MagickmindPersistence fails when the chat-history service is
    // down. Ordered the other way, the agent's own turn would be dropped from
    // episodic memory during exactly the outage best-effort ingest exists to
    // survive — leaving a half-turn (user message stored, reply missing).
    let mut respond = Pipeline::new()
        .add_stage(SharedStage(Arc::clone(&persona_stage)))
        .add_streaming_stage(GenericLlmProcessor::new(LlmClient::new(respond_llm)?))
        .add_stage(PostProcessor);
    if let Some(reply_ingest) = builder.build_episode_reply_ingest_stage()? {
        respond = respond.add_stage(reply_ingest);
    }
    let respond_pipeline =
        Arc::new(respond.add_stage(MagickmindPersistence::new(Arc::clone(&magickmind))));

    // Wire up the runtime
    let mut runtime = builder
        .on_message(move |ctx| {
            let preparer = Arc::clone(&context_preparer);
            let gate = Arc::clone(&gate_pipeline);
            let respond = Arc::clone(&respond_pipeline);
            let inbound_ingest = inbound_ingest.clone();

            async move {
                tracing::info!(
                    "Received message from {}: {:?}",
                    ctx.message.sender_id,
                    ctx.message.content
                );

                let mut pctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());

                // Step 0: Remember this message before anything can halt. Every
                // received message is ingested, whether or not the agent replies.
                // Ingest is best-effort: the stage logs its own failures and
                // always returns Ok, so there is nothing to handle here.
                if let Some(ingest) = &inbound_ingest {
                    let _ = ingest.process(&mut pctx).await;
                }

                // Step 1: Check @mention first (cheap, no API calls).
                // Skipped when MINDROID_DISABLE_GATE is set (local mindroid test).
                if gate_enabled {
                    if let Err(e) = ctx.run_with_context(&gate, &mut pctx).await {
                        tracing::error!("Mention gate failed: {e}");
                        return;
                    }
                    if pctx.halted {
                        tracing::info!("Not @mentioned — staying silent");
                        return;
                    }
                    tracing::info!("@mentioned — proceeding");
                } else {
                    tracing::info!("Mention gate disabled — responding to all messages");
                }

                // Step 2: Fetch context from MagickMind (only after mention check passes)
                let history = match preparer.prepare(&ctx.message).await {
                    PrepareOutcome::Complete(msgs) => msgs,
                    PrepareOutcome::Degraded { messages, warnings } => {
                        for w in &warnings {
                            tracing::warn!("Context provider '{}' failed: {}", w.provider, w.error);
                        }
                        messages
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

                // Step 3: Run the shared respond pipeline. This request's history
                // rides in the ConversationHistory extension, which the persona
                // stage splices between the system prompt and the user message.
                pctx.reset_output();
                pctx.set_ext(ConversationHistory(history));

                match ctx.run_with_context(&respond, &mut pctx).await {
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
