pub mod combinators;
pub mod context;
pub mod coordination;
pub mod extensions;
pub mod presets;
pub mod stages;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use std::fmt;
use std::time::Instant;
use tracing::{debug, info, warn};

use crate::core::context::Context;
use crate::core::events::PipelineEvent;
use crate::error::Result;
use crate::models::StreamEvent;

/// Backward-compatible alias for [`crate::core::context::Context`].
pub type PipelineContext = crate::core::context::Context;

/// A single stage in the processing pipeline.
#[async_trait]
pub trait PipelineStage: Send + Sync {
    fn name(&self) -> &str;
    async fn process(&self, ctx: &mut Context) -> Result<()>;
}

/// A pipeline stage that supports streaming output (e.g., LLM token streaming).
/// Only one streaming stage is allowed per pipeline.
///
/// Implementors must also implement `PipelineStage::process()` as the
/// non-streaming fallback (collect all chunks and set `ctx.raw_response`).
pub trait StreamingStage: PipelineStage {
    fn stream<'a>(&'a self, ctx: &'a mut Context) -> BoxStream<'a, StreamEvent>;
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
    pub async fn run(&self, ctx: &mut Context) -> Result<Option<String>> {
        let pipeline_start = Instant::now();
        info!("Pipeline::run starting ({} stages)", self.stages.len());
        ctx.emit_event(PipelineEvent::PipelineStarted {
            stage_count: self.stages.len(),
        });

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
                    ctx.emit_event(PipelineEvent::StageStarted {
                        stage_name: name.to_string(),
                        stage_index: i,
                    });
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
                    ctx.emit_event(PipelineEvent::StageCompleted {
                        stage_name: name.to_string(),
                        stage_index: i,
                        elapsed,
                    });
                    if ctx.halted {
                        info!(
                            "Pipeline halted by stage [{}/{}] '{}'",
                            i + 1,
                            self.stages.len(),
                            name
                        );
                        break;
                    }
                    if ctx.cancel.is_cancelled() {
                        info!(
                            "Pipeline cancelled after stage [{}/{}] '{}'",
                            i + 1,
                            self.stages.len(),
                            name
                        );
                        ctx.emit_event(PipelineEvent::Cancelled {
                            stage_name: name.to_string(),
                        });
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
                    ctx.emit_event(PipelineEvent::StageStarted {
                        stage_name: name.to_string(),
                        stage_index: i,
                    });
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
                    ctx.emit_event(PipelineEvent::StageCompleted {
                        stage_name: name.to_string(),
                        stage_index: i,
                        elapsed,
                    });
                    if ctx.halted {
                        info!(
                            "Pipeline halted by stage [{}/{}] '{}'",
                            i + 1,
                            self.stages.len(),
                            name
                        );
                        break;
                    }
                    if ctx.cancel.is_cancelled() {
                        info!(
                            "Pipeline cancelled after stage [{}/{}] '{}'",
                            i + 1,
                            self.stages.len(),
                            name
                        );
                        ctx.emit_event(PipelineEvent::Cancelled {
                            stage_name: name.to_string(),
                        });
                        break;
                    }
                }
            }
        }

        let total = pipeline_start.elapsed();
        info!("Pipeline::run completed in {:.2?}", total);
        ctx.emit_event(PipelineEvent::PipelineCompleted { elapsed: total });

        Ok(ctx.response.take())
    }

    /// Run the pipeline with streaming. Returns a stream of `StreamEvent`s.
    ///
    /// Pre-streaming stages run before the first event is yielded.
    /// Post-streaming stages run after the stream completes (their effects
    /// are signaled via a final `Complete` event).
    #[allow(clippy::collapsible_if)]
    pub fn run_streaming<'a>(&'a self, ctx: &'a mut Context) -> BoxStream<'a, StreamEvent> {
        let stream = async_stream::stream! {
            let pipeline_start = Instant::now();
            let total_stages = self.stages.len();
            info!("Pipeline::run_streaming starting ({total_stages} stages)");
            ctx.emit_event(PipelineEvent::PipelineStarted {
                stage_count: total_stages,
            });

            let split = self.streaming_idx.unwrap_or(total_stages);

            // Run pre-streaming stages
            for (i, entry) in self.stages[..split].iter().enumerate() {
                if let StageEntry::Normal(stage) = entry {
                    let name = stage.name();
                    debug!("Pipeline stage [{}/{}] '{}' starting", i + 1, total_stages, name);
                    ctx.emit_event(PipelineEvent::StageStarted {
                        stage_name: name.to_string(),
                        stage_index: i,
                    });
                    let start = Instant::now();
                    if let Err(e) = stage.process(ctx).await {
                        warn!("Pipeline stage '{}' failed: {}", name, e);
                        ctx.emit_event(PipelineEvent::StageError {
                            stage_name: name.to_string(),
                            error: e.to_string(),
                        });
                        yield StreamEvent::Error { message: e.to_string() };
                        return;
                    }
                    let elapsed = start.elapsed();
                    info!("Pipeline stage [{}/{}] '{}' completed in {:.2?}", i + 1, total_stages, name, elapsed);
                    ctx.emit_event(PipelineEvent::StageCompleted {
                        stage_name: name.to_string(),
                        stage_index: i,
                        elapsed,
                    });
                    if ctx.halted {
                        info!("Pipeline halted by stage [{}/{}] '{}'", i + 1, total_stages, name);
                        return;
                    }
                    if ctx.cancel.is_cancelled() {
                        info!("Pipeline cancelled after stage [{}/{}] '{}'", i + 1, total_stages, name);
                        ctx.emit_event(PipelineEvent::Cancelled {
                            stage_name: name.to_string(),
                        });
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
                    // Capture events sender before the mutable borrow for streaming
                    let events_tx = ctx.events_sender().cloned();
                    let emit = |event: PipelineEvent| {
                        if let Some(ref tx) = events_tx {
                            let _ = tx.send(event);
                        }
                    };
                    emit(PipelineEvent::StageStarted {
                        stage_name: name.to_string(),
                        stage_index: idx,
                    });
                    let stream_start = Instant::now();
                    let mut collected = String::new();
                    let mut chunk_count: u32 = 0;
                    let cancel = ctx.cancel.clone();
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
                            if cancel.is_cancelled() {
                                info!("Pipeline streaming cancelled during stage '{}'", name);
                                emit(PipelineEvent::Cancelled {
                                    stage_name: name.to_string(),
                                });
                                break;
                            }
                        }
                    }
                    let stream_elapsed = stream_start.elapsed();
                    info!(
                        "Pipeline stage [{}/{}] '{}' (streaming) completed in {:.2?} ({} chunks, {} bytes)",
                        idx + 1, total_stages, name, stream_elapsed, chunk_count, collected.len()
                    );
                    emit(PipelineEvent::StageCompleted {
                        stage_name: name.to_string(),
                        stage_index: idx,
                        elapsed: stream_elapsed,
                    });
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
                    ctx.emit_event(PipelineEvent::StageStarted {
                        stage_name: name.to_string(),
                        stage_index: post_start + i,
                    });
                    let start = Instant::now();
                    if let Err(e) = stage.process(ctx).await {
                        warn!("Pipeline stage '{}' failed: {}", name, e);
                        ctx.emit_event(PipelineEvent::StageError {
                            stage_name: name.to_string(),
                            error: e.to_string(),
                        });
                        yield StreamEvent::Error { message: e.to_string() };
                        return;
                    }
                    let elapsed = start.elapsed();
                    info!("Pipeline stage [{}/{}] '{}' completed in {:.2?}", stage_num, total_stages, name, elapsed);
                    ctx.emit_event(PipelineEvent::StageCompleted {
                        stage_name: name.to_string(),
                        stage_index: post_start + i,
                        elapsed,
                    });
                    if ctx.halted {
                        info!("Pipeline halted by stage [{}/{}] '{}'", stage_num, total_stages, name);
                        return;
                    }
                    if ctx.cancel.is_cancelled() {
                        info!("Pipeline cancelled after stage [{}/{}] '{}'", stage_num, total_stages, name);
                        ctx.emit_event(PipelineEvent::Cancelled {
                            stage_name: name.to_string(),
                        });
                        return;
                    }
                }
            }

            ctx.emit_event(PipelineEvent::PipelineCompleted {
                elapsed: pipeline_start.elapsed(),
            });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;
    use crate::models::Message;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// A stage that records whether it was called.
    struct RecorderStage {
        called: Arc<AtomicBool>,
    }

    #[async_trait]
    impl PipelineStage for RecorderStage {
        fn name(&self) -> &str {
            "recorder"
        }
        async fn process(&self, _ctx: &mut Context) -> crate::error::Result<()> {
            self.called.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    /// A stage that cancels the context.
    struct CancelStage;

    #[async_trait]
    impl PipelineStage for CancelStage {
        fn name(&self) -> &str {
            "cancel"
        }
        async fn process(&self, ctx: &mut Context) -> crate::error::Result<()> {
            ctx.cancel.cancel();
            Ok(())
        }
    }

    fn make_test_context() -> Context {
        let msg = Arc::new(Message::new("test", "user1", "ch1"));
        let config = Arc::new(AgentConfig::default());
        Context::new(msg, config)
    }

    #[tokio::test]
    async fn test_pipeline_cancellation() {
        let second_called = Arc::new(AtomicBool::new(false));
        let pipeline = Pipeline::new()
            .add_stage(CancelStage)
            .add_stage(RecorderStage {
                called: second_called.clone(),
            });

        let mut ctx = make_test_context();
        let _ = pipeline.run(&mut ctx).await;

        // Second stage should NOT have been called
        assert!(
            !second_called.load(Ordering::SeqCst),
            "Second stage should not run after cancellation"
        );
    }

    #[tokio::test]
    async fn test_pipeline_streaming_cancellation() {
        use futures::StreamExt;

        let second_called = Arc::new(AtomicBool::new(false));
        let pipeline = Pipeline::new()
            .add_stage(CancelStage)
            .add_stage(RecorderStage {
                called: second_called.clone(),
            });

        let mut ctx = make_test_context();
        // Consume the stream fully
        let mut stream = pipeline.run_streaming(&mut ctx);
        while stream.next().await.is_some() {}

        // Second stage should NOT have been called
        assert!(
            !second_called.load(Ordering::SeqCst),
            "Second stage should not run after cancellation in streaming mode"
        );
    }
}
