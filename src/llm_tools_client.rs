//! OpenAI-compatible client for NATIVE (JSON) tool calling.
//!
//! Deliberately separate from [`LlmClient`](crate::llm_client::LlmClient):
//! native tool rounds need request messages [`LlmMessage`] cannot represent —
//! an assistant turn carrying `tool_calls`, and `role: tool` results keyed by
//! `tool_call_id` — so this client speaks async-openai request types directly
//! and the caller
//! ([`ToolExecutorJsonStage`](crate::pipeline::stages::ToolExecutorJsonStage))
//! owns the round-trip conversation. Requests are non-streaming: a tool round
//! cannot be surfaced until it is complete anyway (the prompt-XML executor
//! buffers whole rounds for the same reason).

use std::fmt;

use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionMessageToolCalls, ChatCompletionRequestMessage, ChatCompletionTool,
        ChatCompletionTools, CreateChatCompletionRequestArgs, FunctionObject,
    },
};

use crate::core::prompt_text::sanitize_line;
use crate::llm_client::{AuthStyle, LLM_REQUEST_TIMEOUT, LlmClient, LlmClientConfig};
use crate::tools::{MAX_REMOTE_TOOL_PROMPT_BYTES, ToolRegistry};
use crate::{LlmMessage, TokenUsage};

/// Fold control characters out of every string a publisher placed in a schema —
/// property keys and their descriptions both reach the model.
fn sanitize_schema_text(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::String(s) => *s = sanitize_line(s),
        serde_json::Value::Array(items) => items.iter_mut().for_each(sanitize_schema_text),
        serde_json::Value::Object(map) => {
            *map = map
                .iter()
                .map(|(k, v)| {
                    let mut v = v.clone();
                    sanitize_schema_text(&mut v);
                    (sanitize_line(k), v)
                })
                .collect();
        }
        _ => {}
    }
}

/// One structured tool call the model made.
#[derive(Debug, Clone)]
pub struct NativeToolCall {
    /// Provider-issued id; a returning `role: tool` result must echo it.
    pub id: String,
    pub name: String,
    /// The arguments JSON exactly as the model produced it (unparsed).
    pub arguments: String,
}

/// One completed round: the model's prose plus any tool calls it made.
#[derive(Debug, Default)]
pub struct ToolsChatOutcome {
    pub content: String,
    pub tool_calls: Vec<NativeToolCall>,
    pub usage: Option<TokenUsage>,
}

/// OpenAI-compatible client whose requests carry a native `tools` array.
#[derive(Clone)]
pub struct ToolsLlmClient {
    config: LlmClientConfig,
    client: Client<OpenAIConfig>,
}

impl ToolsLlmClient {
    pub fn new(config: LlmClientConfig) -> crate::Result<Self> {
        let api_key = match (&config.auth_style, &config.api_key) {
            (AuthStyle::None, _) => "ollama".to_string(),
            (_, Some(key)) => key.clone(),
            _ => String::new(),
        };

        let openai_config = OpenAIConfig::new()
            .with_api_base(&config.base_url)
            .with_api_key(&api_key);

        let mut default_headers = reqwest::header::HeaderMap::new();
        for (k, v) in &config.custom_headers {
            if let (Ok(name), Ok(val)) = (
                reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                reqwest::header::HeaderValue::from_str(v),
            ) {
                default_headers.insert(name, val);
            }
        }
        if let (AuthStyle::XApiKey, Some(key)) = (&config.auth_style, &config.api_key)
            && let Ok(val) = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
        {
            default_headers.insert("x-api-key", val);
        }

        // reqwest strips `Authorization` across origins but not `x-api-key`,
        // so a redirect would hand the key to another host.
        let http_client = reqwest::ClientBuilder::new()
            .default_headers(default_headers)
            .redirect(reqwest::redirect::Policy::none())
            .timeout(LLM_REQUEST_TIMEOUT)
            .build()
            .map_err(|e| crate::MindroidError::Other(anyhow::Error::from(e)))?;

        let client = Client::with_config(openai_config).with_http_client(http_client);
        Ok(Self { config, client })
    }

    /// Convert pipeline history into request messages (shared conversion with
    /// [`LlmClient`], so multimodal parts render identically).
    pub fn base_messages(messages: &[LlmMessage]) -> Vec<ChatCompletionRequestMessage> {
        LlmClient::convert_messages(messages)
    }

    /// Render a registry's tools as the request's native function specs.
    ///
    /// Publisher-supplied text on a remote tool is neutralized and budgeted
    /// here, because the JSON path never renders `ToolRegistry::system_prompt`
    /// — the single point where the XML path does the same (MM-456).
    pub fn tool_specs(registry: &ToolRegistry) -> Vec<ChatCompletionTools> {
        let mut remote_bytes = 0usize;
        let mut specs = Vec::new();
        for tool in registry.tools() {
            let mut name = tool.name().to_string();
            let mut description = tool.description().to_string();
            let mut parameters = tool.parameters_schema();

            if tool.is_remote() {
                name = sanitize_line(&name);
                description = sanitize_line(&description);
                sanitize_schema_text(&mut parameters);
                let cost = name.len() + description.len() + parameters.to_string().len();
                if remote_bytes + cost > MAX_REMOTE_TOOL_PROMPT_BYTES {
                    continue;
                }
                remote_bytes += cost;
            }

            specs.push(ChatCompletionTools::Function(ChatCompletionTool {
                function: FunctionObject {
                    name,
                    description: Some(description),
                    parameters: Some(parameters),
                    strict: None,
                },
            }));
        }
        specs
    }

    fn pipeline_err(message: String) -> crate::MindroidError {
        crate::MindroidError::Pipeline {
            stage: "ToolsLlmClient".into(),
            message,
            source: None,
        }
    }

    /// One chat round with native tools attached. An empty `tools` slice sends
    /// a plain request (no `tools` field), which some endpoints require.
    pub async fn chat_with_tools(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
        tools: &[ChatCompletionTools],
        model: Option<&str>,
    ) -> crate::Result<ToolsChatOutcome> {
        let model = model
            .or(self.config.default_model.as_deref())
            .unwrap_or("gpt-4o-mini");

        let mut builder = CreateChatCompletionRequestArgs::default();
        builder.model(model).messages(messages);
        if let Some(temp) = self.config.default_temperature {
            builder.temperature(temp);
        }
        if let Some(max) = self.config.default_max_tokens {
            builder.max_completion_tokens(max);
        }
        let mut request = builder
            .build()
            .map_err(|e| Self::pipeline_err(format!("Failed to build request: {e}")))?;
        if !tools.is_empty() {
            request.tools = Some(tools.to_vec());
        }

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .map_err(|e| Self::pipeline_err(format!("API error: {e}")))?;

        let usage = response.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        let mut outcome = ToolsChatOutcome {
            usage,
            ..Default::default()
        };
        if let Some(choice) = response.choices.into_iter().next() {
            outcome.content = choice.message.content.unwrap_or_default();
            outcome.tool_calls = choice
                .message
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .filter_map(|call| match call {
                    ChatCompletionMessageToolCalls::Function(f) => Some(NativeToolCall {
                        id: f.id,
                        name: f.function.name,
                        arguments: f.function.arguments,
                    }),
                    // Custom tool calls are an OpenAI-specific shape this
                    // runtime does not advertise; nothing should produce one.
                    other => {
                        tracing::warn!("Ignoring unsupported tool-call shape: {other:?}");
                        None
                    }
                })
                .collect();
        }
        Ok(outcome)
    }
}

impl fmt::Debug for ToolsLlmClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolsLlmClient")
            .field("base_url", &self.config.base_url)
            .field("default_model", &self.config.default_model)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn construction_with_all_auth_styles() {
        for style in [AuthStyle::Bearer, AuthStyle::XApiKey, AuthStyle::None] {
            let mut config = LlmClientConfig::new("http://localhost/v1");
            config.auth_style = style;
            config.api_key = Some("k".into());
            assert!(ToolsLlmClient::new(config).is_ok());
        }
    }

    #[test]
    fn tool_specs_render_name_description_schema() {
        let registry = ToolRegistry::new().register(
            crate::tools::RemoteTool::new("highlight_element", "Spotlight an element").schema(
                json!({
                    "type": "object",
                    "properties": {"target": {"type": "string", "description": "element key"}}
                }),
            ),
        );

        let specs = ToolsLlmClient::tool_specs(&registry);
        assert_eq!(specs.len(), 1);
        let ChatCompletionTools::Function(tool) = &specs[0] else {
            panic!("expected a function spec");
        };
        assert_eq!(tool.function.name, "highlight_element");
        assert_eq!(
            tool.function.description.as_deref(),
            Some("Spotlight an element")
        );
        let params = tool.function.parameters.as_ref().unwrap();
        assert!(params["properties"]["target"].is_object());
    }

    #[test]
    fn empty_registry_yields_no_specs() {
        assert!(ToolsLlmClient::tool_specs(&ToolRegistry::new()).is_empty());
    }
}
