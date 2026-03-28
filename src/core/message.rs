use async_trait::async_trait;
use futures::stream::BoxStream;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::config::AgentConfig;
use crate::error::Result;
use crate::models::{Message, Response, StreamEvent};
use crate::observer::Observer;
use crate::pipeline::{Pipeline, PipelineContext};

/// Context available to message handlers. Provides access to the pipeline,
/// transport, and memory for processing a single message.
pub struct MessageContext {
    pub message: Arc<Message>,
    pub agent_config: Arc<AgentConfig>,
    pub(crate) pipeline: Arc<Pipeline>,
    pub(crate) transport: Arc<TransportSender>,
    pub(crate) observers: Arc<Vec<Box<dyn Observer>>>,
}

impl MessageContext {
    /// Run the pipeline on this message and return the result.
    pub async fn process(&self) -> Result<Option<String>> {
        let mut ctx = PipelineContext::new(self.message.clone(), self.agent_config.clone());
        self.pipeline.run(&mut ctx).await
    }

    /// Run the pipeline with streaming and return a stream of events.
    pub fn process_streaming(&self) -> BoxStream<'_, StreamEvent> {
        let agent_config = self.agent_config.clone();
        let message = self.message.clone();
        let pipeline = self.pipeline.clone();

        let stream = async_stream::stream! {
            let mut ctx = PipelineContext::new(message, agent_config);
            let mut inner = pipeline.run_streaming(&mut ctx);
            use futures::StreamExt;
            while let Some(event) = inner.next().await {
                yield event;
            }
        };

        Box::pin(stream)
    }

    /// Run any pipeline on this message and return the result.
    pub async fn run_pipeline(&self, pipeline: &Pipeline) -> Result<Option<String>> {
        let mut ctx = PipelineContext::new(self.message.clone(), self.agent_config.clone());
        pipeline.run(&mut ctx).await
    }

    /// Run any pipeline with streaming and return a stream of events.
    pub fn run_pipeline_streaming<'a>(
        &'a self,
        pipeline: &'a Pipeline,
    ) -> BoxStream<'a, StreamEvent> {
        let agent_config = self.agent_config.clone();
        let message = self.message.clone();

        let stream = async_stream::stream! {
            let mut ctx = PipelineContext::new(message, agent_config);
            let mut inner = pipeline.run_streaming(&mut ctx);
            use futures::StreamExt;
            while let Some(event) = inner.next().await {
                yield event;
            }
        };

        Box::pin(stream)
    }

    /// Send a response back through the transport.
    pub async fn respond(&self, content: &str) -> Result<Option<String>> {
        let response = Response::new(content, &self.message.channel_id, &self.agent_config.agent_id)
            .reply_to(&self.message.id);

        let result = self.transport.send(&response).await?;

        for obs in self.observers.iter() {
            obs.on_response_sent(&self.message.channel_id, content).await;
        }

        Ok(result)
    }

    /// Run the pipeline and return the full [`PipelineContext`].
    ///
    /// Use this when you need access to fields beyond `final_response`,
    /// such as `audio_output` produced by [`TtsStage`].
    pub async fn process_context(&self) -> Result<PipelineContext> {
        let mut ctx = PipelineContext::new(self.message.clone(), self.agent_config.clone());
        self.pipeline.run(&mut ctx).await?;
        Ok(ctx)
    }

    /// Convenience: run pipeline then send the response.
    /// Returns `Ok(None)` when the pipeline was halted with no response.
    pub async fn process_and_respond(&self) -> Result<Option<String>> {
        let content = self.process().await?;
        if let Some(ref text) = content {
            self.respond(text).await?;
        }
        Ok(content)
    }

    /// Run a pipeline with a developer-provided PipelineContext.
    /// Allows sharing context (llm_messages, extensions) across pipeline runs.
    pub async fn run_with_context(
        &self,
        pipeline: &Pipeline,
        pctx: &mut PipelineContext,
    ) -> Result<Option<String>> {
        pipeline.run(pctx).await
    }

    /// Run a pipeline with streaming using a developer-provided PipelineContext.
    /// Allows sharing context (llm_messages, extensions) across pipeline runs.
    pub fn run_streaming_with_context<'a>(
        &'a self,
        pipeline: &'a Pipeline,
        pctx: &'a mut PipelineContext,
    ) -> BoxStream<'a, StreamEvent> {
        pipeline.run_streaming(pctx)
    }
}

/// Wraps a transport sender so MessageContext can send responses
/// without holding a mutable reference to the transport.
///
/// Transport implementations should provide a `TransportSender` via a channel
/// or shared state. The default noop implementation discards responses.
pub struct TransportSender {
    inner: Box<dyn TransportSend>,
}

#[async_trait]
pub trait TransportSend: Send + Sync {
    async fn send(&self, response: &Response) -> Result<Option<String>>;
}

struct NoopTransportSend;

#[async_trait]
impl TransportSend for NoopTransportSend {
    async fn send(&self, _response: &Response) -> Result<Option<String>> {
        Ok(None)
    }
}

impl TransportSender {
    pub fn new(inner: impl TransportSend + 'static) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }

    pub fn noop() -> Self {
        Self::new(NoopTransportSend)
    }

    pub async fn send(&self, response: &Response) -> Result<Option<String>> {
        self.inner.send(response).await
    }
}

pub(crate) type MessageHandler =
    Box<dyn Fn(MessageContext) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;
