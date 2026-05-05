use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde::Deserialize;

use mindroid::config::MemoryConfig;
use mindroid::llm_client::{LlmClient, LlmClientConfig};
use mindroid::memory::sqlite::SqliteMemory;
use mindroid::pipeline::presets::magickmind::{
    MagickmindClient, MagickmindContext, MagickmindPersistence,
};
use mindroid::pipeline::presets::sqlite::{
    SqliteClient, SqliteContext, SqlitePersistence,
};
use mindroid::{
    AgentConfig, ContextPreparer, LlmMessage, Message, Pipeline,
    PipelineContext, PipelineStage, PostProcessor, Result, SimpleContextBuilder, StreamEvent,
    StreamingStage, ToolExecutorStage, ToolRegistry,
};

// ── Config ───────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct Config {
    pub memory: MemoryConfig,
}

impl Config {
    pub fn from_file(path: &str) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config = toml::from_str(&content)?;
        Ok(config)
    }
}

// ── PersistenceBackend ───────────────────────────────────────────────────

#[derive(Clone)]
pub enum PersistenceBackend {
    Sqlite(Arc<SqliteClient>),
    Magickmind(Arc<MagickmindClient>),
}

impl PersistenceBackend {
    pub fn sqlite(memory: Arc<SqliteMemory>) -> Self {
        let client = Arc::new(SqliteClient::new(memory));
        Self::Sqlite(client)
    }

    pub fn magickmind(client: Arc<MagickmindClient>) -> Self {
        Self::Magickmind(client)
    }

    pub fn from_config(
        config: &MemoryConfig,
        magickmind: Option<Arc<MagickmindClient>>,
    ) -> anyhow::Result<Self> {
        match config.memory_type.as_deref().unwrap_or("sqlite") {
            "sqlite" => {
                let path = config.path.as_deref().unwrap_or("./memory.db");
                let memory = SqliteMemory::new(path)?;
                Ok(Self::sqlite(Arc::new(memory)))
            }
            "magickmind" => {
                let client = magickmind
                    .ok_or_else(|| anyhow::anyhow!("MagickMind client required"))?;
                Ok(Self::Magickmind(client))
            }
            t => Err(anyhow::anyhow!("Unknown memory backend type: {}", t)),
        }
    }
}

// ── Context builder helper ───────────────────────────────────────────────

pub fn build_context_preparer(
    persistence: &Option<PersistenceBackend>,
    agent_id: &str,
) -> ContextPreparer {
    let mut preparer = ContextPreparer::new();

    if let Some(p) = persistence {
        match p {
            PersistenceBackend::Sqlite(client) => {
                preparer = preparer.add_provider(
                    SqliteContext::new(client.clone())
                        .with_agent_id(agent_id.to_string()),
                );
            }
            PersistenceBackend::Magickmind(client) => {
                preparer = preparer.add_provider(
                    MagickmindContext::new(client.clone())
                        .with_self_id(agent_id.to_string()),
                );
            }
        }
    }

    preparer
}

// ── Persistence stage helper ─────────────────────────────────────────────

pub fn add_persistence_stage(
    pipeline: Pipeline,
    persistence: Option<PersistenceBackend>,
) -> Pipeline {
    match persistence {
        None => pipeline,
        Some(PersistenceBackend::Sqlite(client)) => {
            pipeline.add_stage(SqlitePersistence::new(client))
        }
        Some(PersistenceBackend::Magickmind(client)) => {
            pipeline.add_stage(MagickmindPersistence::new(client))
        }
    }
}

// ── IsFinal logic ────────────────────────────────────────────────────────

pub struct IsFinal(pub bool);

#[derive(Deserialize)]
pub struct FastBrainOutput {
    pub is_final: bool,
    pub response: String,
}

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
                    ctx.response =
                        Some("Let me hand this to my smart brain for a minute.".into());
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

// ── PrepareHistoryStage ──────────────────────────────────────────────────

pub struct PreparedHistory(pub Arc<Vec<LlmMessage>>);

pub struct PrepareHistoryStage {
    context_preparer: Arc<ContextPreparer>,
}

impl PrepareHistoryStage {
    pub fn new(context_preparer: Arc<ContextPreparer>) -> Self {
        Self { context_preparer }
    }
}

#[async_trait]
impl PipelineStage for PrepareHistoryStage {
    fn name(&self) -> &str {
        "PrepareHistoryStage"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let history = self
            .context_preparer
            .prepare(&ctx.message)
            .await
            .unwrap_or_default();

        ctx.set_ext(PreparedHistory(Arc::new(history)));
        Ok(())
    }
}

// ── DualBrainStage ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DualBrainEventMode {
    None,
    BrainMarkers,
    SpinnerFriendly,
}
pub struct DualBrainStage {
    fast_llm: LlmClientConfig,
    smart_llm: LlmClientConfig,
    fast_persona: Arc<str>,
    smart_persona: Arc<str>,
    tool_registry: Arc<ToolRegistry>,
    event_mode: DualBrainEventMode,
}

impl DualBrainStage {
    pub fn new(
        fast_llm: LlmClientConfig,
        smart_llm: LlmClientConfig,
        fast_persona: Arc<str>,
        smart_persona: Arc<str>,
        tool_registry: Arc<ToolRegistry>,
    ) -> Self {
        Self {
            fast_llm,
            smart_llm,
            fast_persona,
            smart_persona,
            tool_registry,
            event_mode: DualBrainEventMode::None,
        }
    }

    pub fn with_event_mode(mut self, mode: DualBrainEventMode) -> Self {
        self.event_mode = mode;
        self
    }

    fn history_from_ctx(ctx: &PipelineContext) -> Arc<Vec<LlmMessage>> {
        ctx.get_ext::<PreparedHistory>()
            .map(|h| Arc::clone(&h.0))
            .unwrap_or_else(|| Arc::new(Vec::new()))
    }

    fn build_smart_history(history: &Arc<Vec<LlmMessage>>, fast_response: &str) -> Arc<Vec<LlmMessage>> {
        let mut smart_history = (**history).clone();
        let fast_response = fast_response.trim();
        if !fast_response.is_empty() {
            smart_history.push(LlmMessage::assistant(fast_response.to_string()));
        }
        Arc::new(smart_history)
    }

    async fn run_fast_brain(
        &self,
        message: Arc<Message>,
        agent_config: Arc<AgentConfig>,
        history: Arc<Vec<LlmMessage>>,
    ) -> Result<(bool, String)> {
        let fast_client = LlmClient::new(self.fast_llm.clone())?;
        let mut fast_ctx = PipelineContext::new(message, agent_config);

        SimpleContextBuilder::with_prompt_and_history(self.fast_persona.as_ref(), history)
            .process(&mut fast_ctx)
            .await?;

        ToolExecutorStage::new(fast_client, Arc::clone(&self.tool_registry))
            .process(&mut fast_ctx)
            .await?;

        IsFinalExtractor.process(&mut fast_ctx).await?;

        let needs_smart = fast_ctx.get_ext::<IsFinal>().map(|f| !f.0).unwrap_or(true);
        Ok((needs_smart, fast_ctx.response.unwrap_or_default()))
    }

    async fn init_smart_ctx(
        &self,
        message: Arc<Message>,
        agent_config: Arc<AgentConfig>,
        smart_history: Arc<Vec<LlmMessage>>,
    ) -> Result<PipelineContext> {
        let mut smart_ctx = PipelineContext::new(message, agent_config);
        SimpleContextBuilder::with_prompt_and_history(self.smart_persona.as_ref(), smart_history)
            .process(&mut smart_ctx)
            .await?;
        Ok(smart_ctx)
    }

    async fn run_smart_brain_nonstreaming(
        &self,
        message: Arc<Message>,
        agent_config: Arc<AgentConfig>,
        smart_history: Arc<Vec<LlmMessage>>,
    ) -> Result<String> {
        let smart_client = LlmClient::new(self.smart_llm.clone())?;
        let mut smart_ctx = self.init_smart_ctx(message, agent_config, smart_history).await?;

        ToolExecutorStage::new(smart_client, Arc::clone(&self.tool_registry))
            .process(&mut smart_ctx)
            .await?;

        Ok(smart_ctx.response.unwrap_or_default())
    }
}

#[async_trait]
impl PipelineStage for DualBrainStage {
    fn name(&self) -> &str {
        "DualBrainStage"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let history = Self::history_from_ctx(ctx);

        let (needs_smart, fast_response) = self
            .run_fast_brain(
                ctx.message.clone(),
                ctx.agent_config.clone(),
                Arc::clone(&history),
            )
            .await?;

        if !needs_smart {
            ctx.response = Some(fast_response);
            return Ok(());
        }

        let smart_history = Self::build_smart_history(&history, &fast_response);
        let smart_response = self
            .run_smart_brain_nonstreaming(ctx.message.clone(), ctx.agent_config.clone(), smart_history)
            .await?;

        ctx.response = Some(smart_response);
        Ok(())
    }
}

impl StreamingStage for DualBrainStage {
    fn stream<'a>(&'a self, outer: &'a mut PipelineContext) -> BoxStream<'a, StreamEvent> {
        Box::pin(async_stream::stream! {
            let history = Self::history_from_ctx(outer);
            if self.event_mode != DualBrainEventMode::None {
                yield StreamEvent::Thinking { content: "MyHere [Fast]".to_string() };
            }

            // ── Fast brain (non-streaming) ───────────────────────────────
            let (needs_smart, fast_response) = match self
                .run_fast_brain(
                    outer.message.clone(),
                    outer.agent_config.clone(),
                    Arc::clone(&history),
                )
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    yield StreamEvent::Error { message: e.to_string() };
                    return;
                }
            };

            if !needs_smart {
                outer.response = Some(fast_response.clone());
                if self.event_mode == DualBrainEventMode::SpinnerFriendly && !fast_response.is_empty() {
                    yield StreamEvent::Chunk { content: fast_response.clone() };
                }
                yield StreamEvent::Complete {
                    content: fast_response,
                    usage: None,
                };
                return;
            }

            let fast_preface = fast_response.trim().to_string();
            if self.event_mode == DualBrainEventMode::SpinnerFriendly && !fast_preface.is_empty() {
                yield StreamEvent::Chunk { content: format!("{fast_preface}\n") };
            }

            // ── Smart brain (streaming) ──────────────────────────────────
            if self.event_mode != DualBrainEventMode::None {
                yield StreamEvent::Thinking { content: "MyHere [Smart]".to_string() };
            }
            let smart_history = Self::build_smart_history(&history, &fast_preface);

            let smart_client = match LlmClient::new(self.smart_llm.clone()) {
                Ok(c) => c,
                Err(e) => {
                    yield StreamEvent::Error { message: e.to_string() };
                    return;
                }
            };

            let mut smart_ctx = match self
                .init_smart_ctx(outer.message.clone(), outer.agent_config.clone(), smart_history)
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    yield StreamEvent::Error { message: e.to_string() };
                    return;
                }
            };

            let tool_stage = ToolExecutorStage::new(smart_client, Arc::clone(&self.tool_registry));

            let collected = {
                let mut stream = tool_stage.stream(&mut smart_ctx);
                let mut collected = String::new();
                while let Some(event) = stream.next().await {
                    match &event {
                        StreamEvent::Chunk { content } => collected.push_str(content),
                        StreamEvent::Complete { content, .. } => {
                            if !content.is_empty() {
                                collected = content.clone();
                            }
                        }
                        _ => {}
                    }
                    yield event;
                }
                collected
            };

            smart_ctx.response = Some(collected);
            outer.response = smart_ctx.response;
        })
    }
}


// ── MyHereStage (MyHere as a reusable stage) ────────────────────────────────

/// MyHere dual-brain pipeline packaged as a PipelineStage.
/// Can be used within MyThere's pipeline to delegate reasoning to MyHere.

pub struct MyHereStage {
    context_preparer: Option<Arc<ContextPreparer>>,
    fast_llm: LlmClientConfig,
    smart_llm: LlmClientConfig,
    fast_persona: Arc<str>,
    smart_persona: Arc<str>,
    tool_registry: Arc<ToolRegistry>,
    persistence: Option<PersistenceBackend>,
}

impl MyHereStage {
    pub fn new(
        context_preparer: Option<Arc<ContextPreparer>>,
        persistence: Option<PersistenceBackend>,
        fast_llm: LlmClientConfig,
        smart_llm: LlmClientConfig,
        fast_persona: Arc<str>,
        smart_persona: Arc<str>,
        tool_registry: Arc<ToolRegistry>,
    ) -> Self {
        Self {
            context_preparer,
            persistence,
            fast_llm,
            smart_llm,
            fast_persona,
            smart_persona,
            tool_registry,
        }
    }
}

#[async_trait]
impl PipelineStage for MyHereStage {
    fn name(&self) -> &str {
        "MyHereStage"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let mut pipeline = Pipeline::new();

        if let Some(preparer) = &self.context_preparer {
            pipeline = pipeline.add_stage(PrepareHistoryStage::new(preparer.clone()));
        }

        pipeline = pipeline
            .add_stage(DualBrainStage::new(
                self.fast_llm.clone(),
                self.smart_llm.clone(),
                self.fast_persona.clone(),
                self.smart_persona.clone(),
                self.tool_registry.clone(),
            ))
            .add_stage(PostProcessor);

        let pipeline = add_persistence_stage(pipeline, self.persistence.clone());

        let mut inner = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());
        ctx.response = pipeline.run(&mut inner).await?;

        Ok(())
    }
}

// ── Tool registry ────────────────────────────────────────────────────────

pub fn create_tool_registry() -> ToolRegistry {
    ToolRegistry::new()
}