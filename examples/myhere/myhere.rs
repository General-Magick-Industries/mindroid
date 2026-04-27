use std::io::Write;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde::Deserialize;

use mindroid::llm_client::{LlmClient, LlmClientConfig};
use mindroid::memory::sqlite::SqliteMemory;
use mindroid::pipeline::presets::magickmind::{MagickmindClient, MagickmindPersistence};
use mindroid::{
    ContextPreparer, ContextProvider, LlmMessage, Memory, Message, MessageContext, Pipeline,
    PipelineContext, PipelineStage, PostProcessor, Result, SimpleContextBuilder, StreamEvent,
    ToolExecutorStage, ToolRegistry,
};

// ── PersistenceBackend enum ──────────────────────────────────────────────────

/// Flexible persistence backend supporting both SQLite and MagickMind.
#[derive(Clone)]
pub enum PersistenceBackend {
    Sqlite(Arc<SqliteMemory>),
    Magickmind(Arc<MagickmindClient>),
}

impl PersistenceBackend {
    pub fn sqlite(memory: Arc<SqliteMemory>) -> Self {
        Self::Sqlite(memory)
    }

    pub fn magickmind(client: Arc<MagickmindClient>) -> Self {
        Self::Magickmind(client)
    }
}

// ── IsFinal extension ────────────────────────────────────────────────────────

/// `true`  = fast brain answered sufficiently, skip smart brain.
/// `false` = question needs deep reasoning, escalate to smart brain.
pub struct IsFinal(pub bool);

#[derive(Deserialize)]
pub struct FastBrainOutput {
    pub is_final: bool,
    pub response: String,
}

// ── IsFinalExtractor ─────────────────────────────────────────────────────────

pub struct IsFinalExtractor;

#[async_trait]
impl PipelineStage for IsFinalExtractor {
    fn name(&self) -> &str {
        "IsFinalExtractor"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let raw = ctx.response.as_deref().unwrap_or("{}");
        match serde_json::from_str::<FastBrainOutput>(raw) {
            Ok(parsed) => {
                ctx.set_ext(IsFinal(parsed.is_final));
                ctx.response = Some(parsed.response);
            }
            Err(_) => {
                let trimmed = raw.trim();
                if trimmed.eq_ignore_ascii_case("false") {
                    ctx.set_ext(IsFinal(false));
                    ctx.response = Some("Let me hand this to my smart brain for a minute.".into());
                } else if trimmed.eq_ignore_ascii_case("true") {
                    ctx.set_ext(IsFinal(true));
                    ctx.response = None;
                } else {
                    ctx.set_ext(IsFinal(true));
                }
            }
        }
        Ok(())
    }
}

// ── BrainRouterGate ──────────────────────────────────────────────────────────

pub struct BrainRouterGate;

#[async_trait]
impl PipelineStage for BrainRouterGate {
    fn name(&self) -> &str {
        "BrainRouterGate"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let needs_smart = ctx.get_ext::<IsFinal>().map(|f| !f.0).unwrap_or(true);
        if needs_smart {
            tracing::info!("BrainRouterGate: escalating to smart brain");
            ctx.halted = true;
        }
        Ok(())
    }
}

// ── SqliteContextProvider ────────────────────────────────────────────────────

/// Loads recent chat history from SQLite and converts it to LlmMessages.
/// Messages from `agent_id` are mapped to the `assistant` role; all others to `user`.
pub struct SqliteContextProvider {
    pub memory: Arc<SqliteMemory>,
    pub agent_id: String,
    pub limit: usize,
}

#[async_trait]
impl ContextProvider for SqliteContextProvider {
    fn name(&self) -> &str {
        "SqliteContextProvider"
    }

    async fn fetch(&self, message: &Message) -> Result<Vec<LlmMessage>> {
        let channel_id = if message.channel_id.is_empty() {
            "stdio".to_string()
        } else {
            message.channel_id.clone()
        };

        let history = self
            .memory
            .get_history(&channel_id, self.limit)
            .await?;

        let llm_messages = history
            .into_iter()
            .map(|msg| {
                if msg.sender_id == self.agent_id || msg.sender_id.is_empty() {
                    LlmMessage::assistant(msg.content)
                } else {
                    LlmMessage::user(msg.content)
                }
            })
            .collect();

        Ok(llm_messages)
    }
}

// ── Animation helper ────────────────────────────────────────────────────────

/// Spawns an animated loading indicator that cycles through `.`, `..`, `...`
pub fn spawn_loading_animation(initial_label: &str) -> tokio::task::JoinHandle<()> {
    let label = initial_label.to_string();
    tokio::spawn(async move {
        let mut stage = 0;
        loop {
            let dots = match stage {
                0 => "|",
                1 => "/",
                2 => "—",
                3 => "\\",
                _ => "*",
            };
            print!("\r{}\x1b[90m{}\x1b[0m\x1b[K", label, dots);
            let _ = std::io::stdout().flush();

            stage = (stage + 1) % 4;
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }
    })
}

// ── SqlitePersistence ────────────────────────────────────────────────────────

/// Saves both the incoming user message and the agent's response to SQLite.
pub struct SqlitePersistence {
    pub memory: Arc<SqliteMemory>,
    pub agent_id: String,
}

#[async_trait]
impl PipelineStage for SqlitePersistence {
    fn name(&self) -> &str {
        "SqlitePersistence"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let channel_id = if ctx.message.channel_id.is_empty() {
            "stdio".to_string()
        } else {
            ctx.message.channel_id.clone()
        };
        let user_id = &ctx.message.sender_id;

        // Save user message
        self.memory
            .save_message(&channel_id, user_id, &ctx.message.content, None)
            .await?;

        // Save agent response
        if let Some(response) = ctx.response.as_deref().filter(|r| !r.is_empty()) {
            self.memory
                .save_message(&channel_id, &self.agent_id, response, None)
                .await?;
        }

        Ok(())
    }
}

// ── MyHereStage (MyHere as a reusable stage) ────────────────────────────────

/// MyHere dual-brain pipeline packaged as a PipelineStage.
/// Can be used within MyThere's pipeline to delegate reasoning to MyHere.
pub struct MyHereStage {
    context_preparer: Arc<ContextPreparer>,
    persistence: PersistenceBackend,
    fast_llm: LlmClientConfig,
    smart_llm: LlmClientConfig,
    fast_persona: Arc<str>,
    smart_persona: Arc<str>,
    tool_registry: Arc<ToolRegistry>,
    agent_id: String,
    persist: bool,
}

impl MyHereStage {
    pub fn new(
        context_preparer: Arc<ContextPreparer>,
        persistence: PersistenceBackend,
        fast_llm: LlmClientConfig,
        smart_llm: LlmClientConfig,
        fast_persona: Arc<str>,
        smart_persona: Arc<str>,
        tool_registry: Arc<ToolRegistry>,
        agent_id: String,
    ) -> Self {
        Self {
            context_preparer,
            persistence,
            fast_llm,
            smart_llm,
            fast_persona,
            smart_persona,
            tool_registry,
            agent_id,
            persist: false,
        }
    }

    pub fn with_persistence(mut self, persist: bool) -> Self {
        self.persist = persist;
        self
    }
}

#[async_trait]
impl PipelineStage for MyHereStage {
    fn name(&self) -> &str {
        "MyHereStage"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        // Fetch additional context from SQLite
        let additional_context = match self.context_preparer.prepare(&ctx.message).await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("MyHere: context fetch failed, continuing: {e}");
                Vec::new()
            }
        };
        let history = Arc::new(additional_context);

        // ── Fast brain ────────────────────────────────────────────────────
        let fast_client = match LlmClient::new(self.fast_llm.clone()) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("MyHere fast brain LLM init failed: {e}");
                return Err(e.into());
            }
        };

        let mut fast_ctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());

        SimpleContextBuilder::with_prompt_and_history(self.fast_persona.as_ref(), history.clone())
            .process(&mut fast_ctx)
            .await?;

        ToolExecutorStage::new(fast_client, Arc::clone(&self.tool_registry))
            .process(&mut fast_ctx)
            .await?;

        IsFinalExtractor.process(&mut fast_ctx).await?;

        let needs_smart = fast_ctx.get_ext::<IsFinal>().map(|f| !f.0).unwrap_or(true);

        // If fast brain final, update outer context and return
        if !needs_smart {
            PostProcessor.process(&mut fast_ctx).await?;
            if self.persist {
                match &self.persistence {
                    PersistenceBackend::Sqlite(memory) => {
                        SqlitePersistence {
                            memory: Arc::clone(memory),
                            agent_id: self.agent_id.clone(),
                        }
                        .process(&mut fast_ctx)
                        .await?;
                    }
                    PersistenceBackend::Magickmind(magickmind) => {
                        MagickmindPersistence::new(Arc::clone(magickmind))
                            .process(&mut fast_ctx)
                            .await?;
                    }
                }
            }

            ctx.response = fast_ctx.response;
            return Ok(());
        }

        // ── Smart brain (if needed) ───────────────────────────────────────
        tracing::info!("MyHere: fast brain escalated, running smart brain");

        let smart_client = match LlmClient::new(self.smart_llm.clone()) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("MyHere smart brain LLM init failed: {e}");
                return Err(e.into());
            }
        };

        let fast_response = fast_ctx.response.clone().unwrap_or_default();
        let mut smart_history = (*history).clone();
        if !fast_response.is_empty() {
            smart_history.push(LlmMessage::assistant(fast_response));
        }
        let smart_history = Arc::new(smart_history);

        let mut smart_ctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());

        SimpleContextBuilder::with_prompt_and_history(self.smart_persona.as_ref(), smart_history)
            .process(&mut smart_ctx)
            .await?;

        ToolExecutorStage::new(smart_client, Arc::clone(&self.tool_registry))
            .process(&mut smart_ctx)
            .await?;

        PostProcessor.process(&mut smart_ctx).await?;
        if self.persist {
            match &self.persistence {
                PersistenceBackend::Sqlite(memory) => {
                    SqlitePersistence {
                        memory: Arc::clone(memory),
                        agent_id: self.agent_id.clone(),
                    }
                    .process(&mut smart_ctx)
                    .await?;
                }
                PersistenceBackend::Magickmind(magickmind) => {
                    MagickmindPersistence::new(Arc::clone(magickmind))
                        .process(&mut smart_ctx)
                        .await?;
                }
            }
        }

        ctx.response = smart_ctx.response;
        Ok(())
    }
}

// ── MyHere pipeline builder ──────────────────────────────────────────────────

/// Creates an empty tool registry.
/// Downstream pipelines can add tools as needed.
pub fn create_tool_registry() -> ToolRegistry {
    ToolRegistry::new()
}

/// Builder for MyHere pipeline with extensibility points for custom stages.
/// Allows downstream pipelines (e.g., MyThere) to inject additional stages.
pub struct MyHerePipelineBuilder {
    context_preparer: Arc<ContextPreparer>,
    persistence: PersistenceBackend,
    fast_llm: LlmClientConfig,
    smart_llm: LlmClientConfig,
    fast_persona: Arc<str>,
    smart_persona: Arc<str>,
    tool_registry: Arc<ToolRegistry>,
    agent_id: String,
    persist: bool,
}

impl MyHerePipelineBuilder {
    pub fn new(
        context_preparer: Arc<ContextPreparer>,
        persistence: PersistenceBackend,
        fast_llm: LlmClientConfig,
        smart_llm: LlmClientConfig,
        fast_persona: Arc<str>,
        smart_persona: Arc<str>,
        tool_registry: Arc<ToolRegistry>,
        agent_id: String,
    ) -> Self {
        Self {
            context_preparer,
            persistence,
            fast_llm,
            smart_llm,
            fast_persona,
            smart_persona,
            tool_registry,
            agent_id,
            persist: false,
        }
    }

    pub fn with_persistence(mut self, persist: bool) -> Self {
        self.persist = persist;
        self
    }

    pub fn build(self) -> impl Fn(MessageContext) -> futures::future::BoxFuture<'static, ()> + Send + 'static {
        let context_preparer = self.context_preparer;
        let persistence = self.persistence;
        let fast_llm = self.fast_llm;
        let smart_llm = self.smart_llm;
        let fast_persona = self.fast_persona;
        let smart_persona = self.smart_persona;
        let tool_registry = self.tool_registry;
        let agent_id = self.agent_id.clone();
        let persist = self.persist;

        move |ctx| {
            let preparer = Arc::clone(&context_preparer);
            let fast_llm = fast_llm.clone();
            let smart_llm = smart_llm.clone();
            let fast_persona = Arc::clone(&fast_persona);
            let smart_persona = Arc::clone(&smart_persona);
            let tool_registry = Arc::clone(&tool_registry);
            let agent_id = agent_id.clone();
            let persistence = persistence.clone();

            Box::pin(async move {
                // ── 1. Fetch local SQLite context ─────────────────────────────
                let context = match preparer.prepare(&ctx.message).await {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!("Context fetch failed, continuing without history: {e}");
                        Vec::new()
                    }
                };
                let history = Arc::new(context);

                // ── 2. Fast brain pipeline ────────────────────────────────────
                let fast_client = match LlmClient::new(fast_llm) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::error!("Fast brain LLM init failed: {e}");
                        return;
                    }
                };

                let mut fast_pipeline = Pipeline::new()
                    .add_stage(SimpleContextBuilder::with_prompt_and_history(
                        fast_persona.as_ref(),
                        history.clone(),
                    ))
                    .add_streaming_stage(ToolExecutorStage::new(
                        fast_client,
                        Arc::clone(&tool_registry),
                    ))
                    .add_stage(IsFinalExtractor)
                    .add_stage(BrainRouterGate);

                fast_pipeline = fast_pipeline.add_stage(PostProcessor);
                if persist {
                    match &persistence {
                        PersistenceBackend::Sqlite(memory) => {
                            fast_pipeline = fast_pipeline.add_stage(SqlitePersistence {
                                memory: Arc::clone(memory),
                                agent_id: agent_id.clone(),
                            });
                        }
                        PersistenceBackend::Magickmind(magickmind) => {
                            fast_pipeline = fast_pipeline
                                .add_stage(MagickmindPersistence::new(Arc::clone(magickmind)));
                        }
                    }
                }

                let mut pctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());

                print!("\n\x1b[90mMyHere [Fast]: \x1b[0m");
                std::io::stdout().flush().ok();
                let animation = spawn_loading_animation("\x1b[90mMyHere [Fast]: \x1b[0m");

                let (needs_smart, fast_response) =
                    match ctx.run_with_context(&fast_pipeline, &mut pctx).await {
                        Ok(resp) => {
                            let is_final = pctx.get_ext::<IsFinal>().map(|f| f.0).unwrap_or(true);
                            let needs_smart = !is_final;
                            (needs_smart, resp.unwrap_or_default())
                        }
                        Err(e) => {
                            tracing::error!("Fast brain pipeline failed: {e}");
                            return;
                        }
                    };

                animation.abort();
                let response = fast_response.trim().to_string();
                if !response.is_empty() {
                    let is_stdio = ctx.message.channel_id.is_empty() || ctx.message.channel_id == "stdio";
                    if is_stdio {
                        print!("\rMyHere [Fast]: {response}\n\n");
                        std::io::stdout().flush().ok();
                    } else if let Err(e) = ctx.respond(&response).await {
                        tracing::error!("Failed to send fast brain response: {e}");
                    }
                }


                if !needs_smart {
                    println!("\x1b[90mType your messages below:\x1b[0m");
                    return;
                }

                // ── 3. Smart brain pipeline ───────────────────────────────────
                tracing::info!("Fast brain escalated — running smart brain");

                let smart_client = match LlmClient::new(smart_llm) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::error!("Smart brain LLM init failed: {e}");
                        return;
                    }
                };

                // Include fast brain's response in context for smart brain
                let mut smart_history = (*history).clone();
                if !response.is_empty() {
                    smart_history.push(LlmMessage::assistant(response.clone()));
                }
                let smart_history = Arc::new(smart_history);

                let mut smart_pipeline = Pipeline::new()
                    .add_stage(SimpleContextBuilder::with_prompt_and_history(
                        smart_persona.as_ref(),
                        smart_history,
                    ))
                    .add_streaming_stage(ToolExecutorStage::new(
                        smart_client,
                        Arc::clone(&tool_registry),
                    ))
                    .add_stage(PostProcessor);
                if persist {
                    match &persistence {
                        PersistenceBackend::Sqlite(memory) => {
                            smart_pipeline = smart_pipeline.add_stage(SqlitePersistence {
                                memory: Arc::clone(memory),
                                agent_id: agent_id.clone(),
                            });
                        }
                        PersistenceBackend::Magickmind(magickmind) => {
                            smart_pipeline = smart_pipeline
                                .add_stage(MagickmindPersistence::new(Arc::clone(magickmind)));
                        }
                    }
                }

                pctx.reset_output();

                print!("\x1b[90mMyHere [Smart]: \x1b[0m");
                std::io::stdout().flush().ok();
                let animation = spawn_loading_animation("\x1b[90mMyHere [Smart]: \x1b[0m");

                let mut stream = ctx.run_streaming_with_context(&smart_pipeline, &mut pctx);
                let mut full_response = String::new();

                let mut first_chunk = true;
                while let Some(event) = stream.next().await {
                    match &event {
                        StreamEvent::Chunk { content } => {
                            if first_chunk {
                                animation.abort();
                                print!("\rMyHere [Smart]: ");
                                first_chunk = false;
                            }
                            print!("{content}");
                            let _ = std::io::Write::flush(&mut std::io::stdout());
                            full_response.push_str(content);
                        }
                        StreamEvent::Complete { content, .. } => {
                            if !content.is_empty() {
                                full_response = content.clone();
                            }
                        }
                        StreamEvent::Error { message } => {
                            tracing::error!("Smart brain stream error: {message}");
                        }
                        _ => {}
                    }
                }

                let response = full_response.trim().to_string();
                if !response.is_empty() {
                    if let Err(e) = ctx.respond(&response).await {
                        tracing::error!("Failed to send smart brain response: {e}");
                    }
                }

                println!("\n");
                println!("\x1b[90mType your messages below:\x1b[0m");
            })
        }
    }
}

/// Convenience function: builds the MyHere pipeline with no custom stages.
/// For custom stages, use `MyHerePipelineBuilder` directly.
pub fn build_myhere_pipeline(
    context_preparer: Arc<ContextPreparer>,
    persistence: PersistenceBackend,
    fast_llm: LlmClientConfig,
    smart_llm: LlmClientConfig,
    fast_persona: Arc<str>,
    smart_persona: Arc<str>,
    tool_registry: Arc<ToolRegistry>,
    agent_id: String,
) -> impl Fn(MessageContext) -> futures::future::BoxFuture<'static, ()> + Send + 'static {
    MyHerePipelineBuilder::new(
        context_preparer,
        persistence,
        fast_llm,
        smart_llm,
        fast_persona,
        smart_persona,
        tool_registry,
        agent_id,
    )
    .build()
}
