use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;

use crate::core::context::Context;
use crate::llm_client::{ChatRequest, LlmClient};
use crate::{MindroidError, PipelineStage, Result, StreamEvent, StreamingStage};

/// Collect a stream of [`StreamEvent`]s into a `(content, Option<error_message>)` tuple.
///
/// Useful for non-streaming pipeline execution or any code that needs to
/// consume an LLM stream to completion.
pub async fn collect_stream(
    mut stream: BoxStream<'static, StreamEvent>,
) -> (String, Option<String>) {
    let mut collected = String::new();
    while let Some(event) = stream.next().await {
        match event {
            StreamEvent::Chunk { content } => collected.push_str(&content),
            StreamEvent::Complete { content, .. } if !content.is_empty() => {
                collected = content;
            }
            StreamEvent::Error { message } => {
                return (collected, Some(message));
            }
            _ => {}
        }
    }
    (collected, None)
}

/// A generic LLM processor that works with any [`LlmClient`] instance.
///
/// Uses the `default_model` from the client's [`LlmClientConfig`] rather than
/// reading model IDs from the pipeline context. This makes it backend-agnostic —
/// works with Ollama, OpenAI, litellm, vLLM, OpenRouter, or any OpenAI-compatible
/// endpoint.
///
/// # Example
///
/// ```ignore
/// use mindroid::llm_client::{LlmClient, LlmClientConfig, AuthStyle};
/// use mindroid::pipeline::stages::{SimpleContextBuilder, GenericLlmProcessor, PostProcessor};
/// use mindroid::Pipeline;
///
/// let mut config = LlmClientConfig::new("http://localhost:11434/v1");
/// config.default_model = Some("llama3.2".to_string());
/// config.auth_style = AuthStyle::None;
/// let client = LlmClient::new(config);
///
/// let pipeline = Pipeline::new()
///     .add_stage(SimpleContextBuilder)
///     .add_streaming_stage(GenericLlmProcessor::new(client))
///     .add_stage(PostProcessor);
/// ```
pub struct GenericLlmProcessor {
    client: LlmClient,
}

impl GenericLlmProcessor {
    pub fn new(client: LlmClient) -> Self {
        Self { client }
    }
}

#[async_trait]
impl PipelineStage for GenericLlmProcessor {
    fn name(&self) -> &str {
        "GenericLlmProcessor"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let (collected, error) = collect_stream(self.client.stream_chat(ChatRequest {
            messages: &ctx.llm_messages,
            model: None,
            temperature: None,
            max_tokens: None,
            stream: true,
            response_format: None,
        }))
        .await;

        if let Some(msg) = error {
            return Err(MindroidError::Pipeline {
                stage: "GenericLlmProcessor".into(),
                message: msg,
                source: None,
            });
        }

        ctx.response = Some(collected);
        Ok(())
    }
}

impl StreamingStage for GenericLlmProcessor {
    fn stream<'a>(&'a self, ctx: &'a mut Context) -> BoxStream<'a, StreamEvent> {
        self.client.stream_chat(ChatRequest {
            messages: &ctx.llm_messages,
            model: None,
            temperature: None,
            max_tokens: None,
            stream: true,
            response_format: None,
        })
    }
}
