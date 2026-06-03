//! OpenAI-compatible LLM client for Mindroid.
//!
//! Wraps [`async_openai`] to provide a unified client that works with any
//! OpenAI-compatible endpoint (OpenAI, litellm, vLLM, Ollama, cortex, OpenRouter, etc.)
//! with configurable `base_url` and auth style.

use std::collections::HashMap;
use std::fmt;

use crate::core::content::{ContentPart, ContentSource};
use crate::{LlmMessage, Role, StreamEvent, TokenUsage};
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequestArgs,
        FinishReason, ImageUrl, ResponseFormat,
    },
};
use futures::StreamExt;
use futures::stream::BoxStream;
use tracing::debug;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// How to attach credentials to outgoing requests.
#[derive(Debug, Clone, Default)]
pub enum AuthStyle {
    /// `Authorization: Bearer <key>` (default, used by OpenAI/OpenRouter/etc.)
    #[default]
    Bearer,
    /// `x-api-key: Bearer <key>` (cortex-service style)
    XApiKey,
    /// No auth header (e.g. local Ollama)
    None,
}

/// Endpoint and default parameter configuration for [`LlmClient`].
#[derive(Debug, Clone)]
pub struct LlmClientConfig {
    /// Base URL including `/v1`, e.g. `"https://api.openai.com/v1"`.
    pub base_url: String,
    /// API key (used according to `auth_style`).
    pub api_key: Option<String>,
    /// Fallback model when `ChatRequest.model` is `None`.
    pub default_model: Option<String>,
    /// Fallback temperature.
    pub default_temperature: Option<f32>,
    /// Fallback max_tokens.
    pub default_max_tokens: Option<u32>,
    /// How to send credentials.
    pub auth_style: AuthStyle,
    /// Extra headers to include on every request.
    pub custom_headers: HashMap<String, String>,
}

impl LlmClientConfig {
    /// Create a new config with the required `base_url`.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: None,
            default_model: None,
            default_temperature: None,
            default_max_tokens: None,
            auth_style: AuthStyle::Bearer,
            custom_headers: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// Per-call parameters for a chat request.
pub struct ChatRequest<'a> {
    pub messages: &'a [LlmMessage],
    /// Overrides `LlmClientConfig.default_model`.
    pub model: Option<&'a str>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stream: bool,
    /// Optional response format constraint (JSON mode or JSON schema).
    pub response_format: Option<ResponseFormat>,
}

// ---------------------------------------------------------------------------
// Multimodal helpers
// ---------------------------------------------------------------------------

/// Returns true if content contains any non-text parts that need multimodal formatting.
fn has_multimodal_content(content: &[ContentPart]) -> bool {
    content.iter().any(|p| !p.is_text())
}

/// Convert `ContentPart`s to OpenAI user message content parts for multimodal messages.
/// Text-only messages should use the fast text path instead.
fn content_parts_to_openai(
    content: &[ContentPart],
) -> Vec<ChatCompletionRequestUserMessageContentPart> {
    content
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text { text } => Some(ChatCompletionRequestUserMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartText { text: text.clone() },
            )),
            ContentPart::Image { source, mime_type } => {
                let url = match source {
                    ContentSource::Uri { uri } => uri.clone(),
                    ContentSource::Inline { data } => {
                        #[cfg(feature = "transport-ws")]
                        {
                            use base64::Engine;
                            let encoded =
                                base64::engine::general_purpose::STANDARD.encode(data);
                            format!("data:{};base64,{}", mime_type, encoded)
                        }
                        #[cfg(not(feature = "transport-ws"))]
                        {
                            let _ = (data, mime_type);
                            tracing::warn!(
                                "Skipping inline image in OpenAI conversion: base64 encoding \
                                 requires the `transport-ws` feature"
                            );
                            return None;
                        }
                    }
                };
                Some(ChatCompletionRequestUserMessageContentPart::ImageUrl(
                    ChatCompletionRequestMessageContentPartImage {
                        image_url: ImageUrl { url, detail: None },
                    },
                ))
            }
            other => {
                tracing::warn!(
                    "Skipping unsupported content type in OpenAI conversion: {:?}",
                    other
                );
                None
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// LlmClient
// ---------------------------------------------------------------------------

/// OpenAI-compatible LLM client backed by `async-openai`.
pub struct LlmClient {
    config: LlmClientConfig,
    client: Client<OpenAIConfig>,
}

impl LlmClient {
    pub fn new(config: LlmClientConfig) -> crate::Result<Self> {
        let api_key = match (&config.auth_style, &config.api_key) {
            (AuthStyle::None, _) => "ollama".to_string(),
            (_, Some(key)) => key.clone(),
            _ => String::new(),
        };

        let openai_config = OpenAIConfig::new()
            .with_api_base(&config.base_url)
            .with_api_key(&api_key);

        // Build custom reqwest client for extra headers
        let mut default_headers = reqwest::header::HeaderMap::new();
        for (k, v) in &config.custom_headers {
            if let (Ok(name), Ok(val)) = (
                reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                reqwest::header::HeaderValue::from_str(v),
            ) {
                default_headers.insert(name, val);
            }
        }

        // For XApiKey auth, add the custom header
        if let (AuthStyle::XApiKey, Some(key)) = (&config.auth_style, &config.api_key)
            && let Ok(val) = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
        {
            default_headers.insert("x-api-key", val);
        }

        let http_client = reqwest::ClientBuilder::new()
            .default_headers(default_headers)
            .build()
            .map_err(|e| crate::MindroidError::Other(anyhow::Error::from(e)))?;

        let client = Client::with_config(openai_config).with_http_client(http_client);

        Ok(Self { config, client })
    }

    fn resolve_model<'a>(&'a self, req_model: Option<&'a str>) -> &'a str {
        req_model
            .or(self.config.default_model.as_deref())
            .unwrap_or("gpt-4o-mini")
    }

    fn resolve_temperature(&self, req_temp: Option<f32>) -> Option<f32> {
        req_temp.or(self.config.default_temperature)
    }

    fn resolve_max_tokens(&self, req_max: Option<u32>) -> Option<u32> {
        req_max.or(self.config.default_max_tokens)
    }

    fn convert_messages(messages: &[LlmMessage]) -> Vec<ChatCompletionRequestMessage> {
        messages
            .iter()
            .filter_map(|msg| {
                let text = msg.text();
                match msg.role {
                    Role::System => ChatCompletionRequestSystemMessageArgs::default()
                        .content(text.as_str())
                        .build()
                        .ok()
                        .map(Into::into),
                    Role::User | Role::Unknown => {
                        if has_multimodal_content(&msg.content) {
                            // Multimodal: use array content format
                            ChatCompletionRequestUserMessageArgs::default()
                                .content(content_parts_to_openai(&msg.content))
                                .build()
                                .ok()
                                .map(Into::into)
                        } else {
                            // Text-only: fast path
                            ChatCompletionRequestUserMessageArgs::default()
                                .content(text.as_str())
                                .build()
                                .ok()
                                .map(Into::into)
                        }
                    }
                    Role::Assistant => ChatCompletionRequestAssistantMessageArgs::default()
                        .content(text.as_str())
                        .build()
                        .ok()
                        .map(Into::into),
                    Role::Tool => ChatCompletionRequestUserMessageArgs::default()
                        .content(text.as_str())
                        .build()
                        .ok()
                        .map(Into::into),
                }
            })
            .collect()
    }

    fn pipeline_err(message: String) -> crate::MindroidError {
        crate::MindroidError::Pipeline {
            stage: "LlmClient".into(),
            message,
            source: None,
        }
    }

    // ── Non-streaming ────────────────────────────────────────────────────

    /// Send a non-streaming chat request. Returns `(content, Option<TokenUsage>)`.
    pub async fn chat(&self, req: ChatRequest<'_>) -> crate::Result<(String, Option<TokenUsage>)> {
        let messages = Self::convert_messages(req.messages);
        let model = self.resolve_model(req.model);

        let mut builder = CreateChatCompletionRequestArgs::default();
        builder.model(model).messages(messages);

        if let Some(temp) = self.resolve_temperature(req.temperature) {
            builder.temperature(temp);
        }
        if let Some(max) = self.resolve_max_tokens(req.max_tokens) {
            builder.max_completion_tokens(max);
        }

        let mut request = builder
            .build()
            .map_err(|e| Self::pipeline_err(format!("Failed to build request: {e}")))?;

        if let Some(fmt) = req.response_format {
            request.response_format = Some(fmt);
        }

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .map_err(|e| Self::pipeline_err(format!("API error: {e}")))?;

        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let usage = response.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok((content, usage))
    }

    // ── Streaming ────────────────────────────────────────────────────────

    /// Send a streaming chat request. Returns a `BoxStream<'static, StreamEvent>`.
    pub fn stream_chat(&self, req: ChatRequest<'_>) -> BoxStream<'static, StreamEvent> {
        let messages = Self::convert_messages(req.messages);
        let model = self.resolve_model(req.model).to_string();
        let temperature = self.resolve_temperature(req.temperature);
        let max_tokens = self.resolve_max_tokens(req.max_tokens);
        let response_format = req.response_format;
        let client = self.client.clone();

        let stream = async_stream::stream! {
            debug!("stream_chat: model={model}, messages={}", messages.len());

            let mut request = match CreateChatCompletionRequestArgs::default()
                .model(&model)
                .messages(messages)
                .build()
            {
                Ok(r) => r,
                Err(e) => {
                    yield StreamEvent::Error {
                        message: format!("Failed to build request: {e}"),
                    };
                    return;
                }
            };

            if let Some(temp) = temperature {
                request.temperature = Some(temp);
            }
            if let Some(max) = max_tokens {
                request.max_completion_tokens = Some(max);
            }
            if let Some(ref fmt) = response_format {
                request.response_format = Some(ResponseFormat::clone(fmt));
            }

            debug!("stream_chat: model={model}");

            let mut response_stream = match client.chat().create_stream(request).await {
                Ok(s) => s,
                Err(e) => {
                    yield StreamEvent::Error {
                        message: format!("API error: {e}"),
                    };
                    return;
                }
            };

            let mut accumulated = String::new();
            let mut final_usage: Option<TokenUsage> = None;

            while let Some(result) = response_stream.next().await {
                match result {
                    Err(e) => {
                        yield StreamEvent::Error { message: format!("{e}") };
                        return;
                    }
                    Ok(chunk) => {
                        // Capture usage if present
                        if let Some(ref u) = chunk.usage {
                            final_usage = Some(TokenUsage {
                                prompt_tokens: u.prompt_tokens,
                                completion_tokens: u.completion_tokens,
                                total_tokens: u.total_tokens,
                            });
                        }

                        if let Some(choice) = chunk.choices.first() {
                            // Emit content delta
                            if let Some(ref content) = choice.delta.content
                                && !content.is_empty()
                            {
                                accumulated.push_str(content);
                                yield StreamEvent::Chunk {
                                    content: content.clone(),
                                };
                            }

                            // Check for terminal finish reasons
                            match choice.finish_reason {
                                Some(FinishReason::Stop) | Some(FinishReason::ToolCalls) => {
                                    yield StreamEvent::Complete {
                                        content: accumulated.clone(),
                                        usage: final_usage.clone(),
                                    };
                                    return;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            // Stream ended without explicit stop
            yield StreamEvent::Complete {
                content: accumulated,
                usage: final_usage,
            };
        };

        Box::pin(stream)
    }
}

impl fmt::Debug for LlmClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmClient")
            .field("base_url", &self.config.base_url)
            .field("default_model", &self.config.default_model)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let config = LlmClientConfig::new("http://localhost:11434/v1");
        assert_eq!(config.base_url, "http://localhost:11434/v1");
        assert!(config.api_key.is_none());
        assert!(matches!(config.auth_style, AuthStyle::Bearer));
    }

    #[test]
    fn resolve_model_priority() {
        let mut config = LlmClientConfig::new("http://localhost/v1");
        config.default_model = Some("gpt-4".into());
        let client = LlmClient::new(config).unwrap();

        assert_eq!(client.resolve_model(Some("claude-3")), "claude-3");
        assert_eq!(client.resolve_model(None), "gpt-4");

        let client2 = LlmClient::new(LlmClientConfig::new("http://localhost/v1")).unwrap();
        assert_eq!(client2.resolve_model(None), "gpt-4o-mini");
    }

    #[test]
    fn convert_messages_all_roles() {
        let messages = vec![
            LlmMessage::system("You are helpful"),
            LlmMessage::user("Hello"),
            LlmMessage::assistant("Hi there"),
        ];

        let converted = LlmClient::convert_messages(&messages);
        assert_eq!(converted.len(), 3);
    }

    #[test]
    fn convert_messages_empty() {
        let messages: Vec<LlmMessage> = vec![];
        let converted = LlmClient::convert_messages(&messages);
        assert!(converted.is_empty());
    }

    #[test]
    fn custom_headers_applied() {
        let mut config = LlmClientConfig::new("http://localhost/v1");
        config.custom_headers = HashMap::from([("X-Custom".into(), "value".into())]);
        let _client = LlmClient::new(config).unwrap();
    }

    #[test]
    fn no_auth_uses_dummy_key() {
        let mut config = LlmClientConfig::new("http://localhost:11434/v1");
        config.auth_style = AuthStyle::None;
        let _client = LlmClient::new(config).unwrap();
    }

    #[test]
    fn text_only_uses_fast_path() {
        let msg = LlmMessage::user("hello world");
        assert!(!has_multimodal_content(&msg.content));
    }

    #[test]
    fn image_uri_detected_as_multimodal() {
        let msg = LlmMessage {
            role: Role::User,
            content: vec![
                ContentPart::text("Look at this:"),
                ContentPart::Image {
                    source: ContentSource::Uri {
                        uri: "https://example.com/cat.jpg".into(),
                    },
                    mime_type: "image/jpeg".into(),
                },
            ],
        };
        assert!(has_multimodal_content(&msg.content));
        let parts = content_parts_to_openai(&msg.content);
        assert_eq!(parts.len(), 2);
    }
}
