pub mod context;
pub mod coordination;
pub mod extensions;
pub mod presets;
pub mod stages;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

use crate::config::AgentConfig;
use crate::error::Result;
use crate::models::{LlmMessage, Message, StreamEvent};

/// Context passed through pipeline stages, accumulating data at each step.
pub struct PipelineContext {
    /// The incoming message being processed.
    pub message: Arc<Message>,
    /// Agent configuration.
    pub agent_config: Arc<AgentConfig>,
    /// LLM conversation messages, built by ContextBuilder stage.
    pub llm_messages: Vec<LlmMessage>,
    /// Response text produced by the pipeline (set by Processor and PostProcessor stages).
    pub response: Option<String>,
    /// When set to `true`, the pipeline stops executing further stages.
    pub halted: bool,

    /// Typed extension map for feature-specific pipeline state.
    ///
    /// Use [`PipelineContext::set_ext`], [`PipelineContext::get_ext`], and
    /// [`PipelineContext::take_ext`] to store and retrieve values keyed by type.
    pub(crate) extensions: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl PipelineContext {
    pub fn new(message: Arc<Message>, agent_config: Arc<AgentConfig>) -> Self {
        Self {
            message,
            agent_config,
            llm_messages: Vec::new(),
            response: None,
            halted: false,
            extensions: HashMap::new(),
        }
    }

    /// Clear output fields for reuse across pipeline runs.
    /// Preserves message and agent_config.
    /// Clears llm_messages so each pipeline builds its own prompt.
    pub fn reset_output(&mut self) {
        self.llm_messages.clear();
        self.response = None;
        self.halted = false;
        self.extensions.clear();
    }

    /// Store a typed value in the extension map.
    pub fn set_ext<T: Send + Sync + 'static>(&mut self, value: T) {
        self.extensions.insert(TypeId::of::<T>(), Box::new(value));
    }

    /// Retrieve a shared reference to a typed value from the extension map.
    pub fn get_ext<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.extensions.get(&TypeId::of::<T>())?.downcast_ref()
    }

    /// Remove and return a typed value from the extension map.
    pub fn take_ext<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.extensions
            .remove(&TypeId::of::<T>())
            .and_then(|b| b.downcast().ok().map(|b| *b))
    }
}

/// A single stage in the processing pipeline.
#[async_trait]
pub trait PipelineStage: Send + Sync {
    fn name(&self) -> &str;
    async fn process(&self, ctx: &mut PipelineContext) -> Result<()>;
}

/// A pipeline stage that supports streaming output (e.g., LLM token streaming).
/// Only one streaming stage is allowed per pipeline.
///
/// Implementors must also implement `PipelineStage::process()` as the
/// non-streaming fallback (collect all chunks and set `ctx.raw_response`).
pub trait StreamingStage: PipelineStage {
    fn stream<'a>(&'a self, ctx: &'a mut PipelineContext) -> BoxStream<'a, StreamEvent>;
}

enum StageEntry {
    Normal(Box<dyn PipelineStage>),
    Streaming(Box<dyn StreamingStage>),
}

/// A composable, ordered pipeline of processing stages.
///
/// Stages run sequentially. At most one stage may be a `StreamingStage`;
/// stages before it run normally, the streaming stage streams, and stages
/// after run on the collected result.
pub struct Pipeline {
    stages: Vec<StageEntry>,
    streaming_idx: Option<usize>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            streaming_idx: None,
        }
    }

    pub fn add_stage(mut self, stage: impl PipelineStage + 'static) -> Self {
        self.stages.push(StageEntry::Normal(Box::new(stage)));
        self
    }

    pub fn add_streaming_stage(mut self, stage: impl StreamingStage + 'static) -> Self {
        let idx = self.stages.len();
        self.streaming_idx = Some(idx);
        self.stages.push(StageEntry::Streaming(Box::new(stage)));
        self
    }

    /// Run the pipeline to completion (non-streaming). Returns the final response.
    ///
    /// For the streaming stage, calls its `PipelineStage::process()` fallback
    /// rather than streaming, which should collect the full response into
    /// `ctx.response`.
    pub async fn run(&self, ctx: &mut PipelineContext) -> Result<Option<String>> {
        let pipeline_start = Instant::now();
        info!("Pipeline::run starting ({} stages)", self.stages.len());

        for (i, entry) in self.stages.iter().enumerate() {
            match entry {
                StageEntry::Normal(stage) => {
                    let name = stage.name();
                    debug!(
                        "Pipeline stage [{}/{}] '{}' starting",
                        i + 1,
                        self.stages.len(),
                        name
                    );
                    let start = Instant::now();
                    stage.process(ctx).await?;
                    let elapsed = start.elapsed();
                    info!(
                        "Pipeline stage [{}/{}] '{}' completed in {:.2?}",
                        i + 1,
                        self.stages.len(),
                        name,
                        elapsed
                    );
                    if ctx.halted {
                        info!(
                            "Pipeline halted by stage [{}/{}] '{}'",
                            i + 1,
                            self.stages.len(),
                            name
                        );
                        break;
                    }
                }
                StageEntry::Streaming(stage) => {
                    let name = stage.name();
                    debug!(
                        "Pipeline stage [{}/{}] '{}' (streaming fallback) starting",
                        i + 1,
                        self.stages.len(),
                        name
                    );
                    let start = Instant::now();
                    stage.process(ctx).await?;
                    let elapsed = start.elapsed();
                    info!(
                        "Pipeline stage [{}/{}] '{}' (streaming fallback) completed in {:.2?}",
                        i + 1,
                        self.stages.len(),
                        name,
                        elapsed
                    );
                    if ctx.halted {
                        info!(
                            "Pipeline halted by stage [{}/{}] '{}'",
                            i + 1,
                            self.stages.len(),
                            name
                        );
                        break;
                    }
                }
            }
        }

        let total = pipeline_start.elapsed();
        info!("Pipeline::run completed in {:.2?}", total);

        Ok(ctx.response.take())
    }

    /// Run the pipeline with streaming. Returns a stream of `StreamEvent`s.
    ///
    /// Pre-streaming stages run before the first event is yielded.
    /// Post-streaming stages run after the stream completes (their effects
    /// are signaled via a final `Complete` event).
    #[allow(clippy::collapsible_if)]
    pub fn run_streaming<'a>(&'a self, ctx: &'a mut PipelineContext) -> BoxStream<'a, StreamEvent> {
        let stream = async_stream::stream! {
            let pipeline_start = Instant::now();
            let total_stages = self.stages.len();
            info!("Pipeline::run_streaming starting ({total_stages} stages)");

            let split = self.streaming_idx.unwrap_or(total_stages);

            // Run pre-streaming stages
            for (i, entry) in self.stages[..split].iter().enumerate() {
                if let StageEntry::Normal(stage) = entry {
                    let name = stage.name();
                    debug!("Pipeline stage [{}/{}] '{}' starting", i + 1, total_stages, name);
                    let start = Instant::now();
                    if let Err(e) = stage.process(ctx).await {
                        warn!("Pipeline stage '{}' failed: {}", name, e);
                        yield StreamEvent::Error { message: e.to_string() };
                        return;
                    }
                    info!("Pipeline stage [{}/{}] '{}' completed in {:.2?}", i + 1, total_stages, name, start.elapsed());
                    if ctx.halted {
                        info!("Pipeline halted by stage [{}/{}] '{}'", i + 1, total_stages, name);
                        return;
                    }
                }
            }

            // Skip streaming and post-streaming stages if halted
            if ctx.halted {
                info!("Pipeline::run_streaming completed (halted) in {:.2?}", pipeline_start.elapsed());
                return;
            }

            // Run streaming stage
            if let Some(idx) = self.streaming_idx {
                if let StageEntry::Streaming(stage) = &self.stages[idx] {
                    let name = stage.name();
                    debug!("Pipeline stage [{}/{}] '{}' (streaming) starting", idx + 1, total_stages, name);
                    let stream_start = Instant::now();
                    let mut collected = String::new();
                    let mut chunk_count: u32 = 0;
                    {
                        let mut event_stream: BoxStream<'_, StreamEvent> = stage.stream(ctx);
                        while let Some(event) = event_stream.next().await {
                            match &event {
                                StreamEvent::Chunk { content } => {
                                    collected.push_str(content);
                                    chunk_count += 1;
                                }
                                StreamEvent::Complete { content, .. }
                                    if !content.is_empty() =>
                                {
                                    collected = content.clone();
                                }
                                StreamEvent::Error { message } => {
                                    warn!("Pipeline stage '{}' stream error: {}", name, message);
                                }
                                _ => {}
                            }
                            yield event;
                        }
                    }
                    info!(
                        "Pipeline stage [{}/{}] '{}' (streaming) completed in {:.2?} ({} chunks, {} bytes)",
                        idx + 1, total_stages, name, stream_start.elapsed(), chunk_count, collected.len()
                    );
                    if ctx.response.is_none() {
                        ctx.response = Some(collected);
                    }
                }
            }

            // Run post-streaming stages
            let post_start = self.streaming_idx.map(|i| i + 1).unwrap_or(total_stages);
            for (i, entry) in self.stages[post_start..].iter().enumerate() {
                if let StageEntry::Normal(stage) = entry {
                    let stage_num = post_start + i + 1;
                    let name = stage.name();
                    debug!("Pipeline stage [{}/{}] '{}' starting", stage_num, total_stages, name);
                    let start = Instant::now();
                    if let Err(e) = stage.process(ctx).await {
                        warn!("Pipeline stage '{}' failed: {}", name, e);
                        yield StreamEvent::Error { message: e.to_string() };
                        return;
                    }
                    info!("Pipeline stage [{}/{}] '{}' completed in {:.2?}", stage_num, total_stages, name, start.elapsed());
                    if ctx.halted {
                        info!("Pipeline halted by stage [{}/{}] '{}'", stage_num, total_stages, name);
                        return;
                    }
                }
            }

            info!("Pipeline::run_streaming completed in {:.2?}", pipeline_start.elapsed());
        };

        Box::pin(stream)
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipeline")
            .field("stages", &format!("{} stages", self.stages.len()))
            .field("streaming_idx", &self.streaming_idx)
            .finish()
    }
}
