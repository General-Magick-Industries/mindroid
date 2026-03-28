use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use std::sync::Arc;
use tracing::{debug, warn};

use crate::error::{MindroidError, Result};
use crate::llm_client::{ChatRequest, LlmClient};
use crate::models::{LlmMessage, StreamEvent};
use crate::pipeline::{PipelineContext, PipelineStage, StreamingStage};

/// Truncate a string to at most `max_bytes` bytes, respecting UTF-8 char boundaries.
fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}
use crate::tools::ToolRegistry;

/// A parsed tool call extracted from LLM output.
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Parses tool call requests from LLM output text.
pub trait ToolCallParser: Send + Sync {
    /// Extract tool calls from a chunk of LLM output.
    /// Returns a list of [`ParsedToolCall`] values.
    fn parse(&self, text: &str) -> Vec<ParsedToolCall>;
}

/// XML-based tool call parser.
///
/// Parses `<tool_call>{"name": "...", "args": {...}}</tool_call>` blocks.
/// Tolerates missing closing tags and malformed JSON (attempts repair).
pub struct XmlToolCallParser;

impl ToolCallParser for XmlToolCallParser {
    fn parse(&self, text: &str) -> Vec<ParsedToolCall> {
        parse_tool_calls(text)
            .into_iter()
            .map(|(name, arguments)| ParsedToolCall { name, arguments })
            .collect()
    }
}

/// Maximum number of tool-call → result rounds before giving up.
const DEFAULT_MAX_ITERATIONS: usize = 20;

/// Prompt appended as a user message when `max_iterations` is reached, asking
/// the LLM to summarise its findings rather than call more tools.
const SUMMARY_PROMPT: &str = "You have gathered enough information from the tools. \
    Please summarize your findings and answer the original question concisely.";

/// A streaming pipeline stage that gives the LLM access to local computer tools.
///
/// Replaces [`GenericLlmProcessor`](crate::pipeline::stages::GenericLlmProcessor)
/// in the pipeline. It injects tool descriptions into the system prompt, then
/// runs a loop:
///
/// 1. Call the LLM, streaming chunks to the caller.
/// 2. If the response contains `<tool_call>` blocks, execute each tool.
/// 3. Feed the results back as a user message and repeat.
/// 4. Stop when the LLM produces a response with no tool calls.
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use mindroid::{Pipeline, tools::{ToolRegistry, ShellTool, OpenTool}};
/// use mindroid::pipeline::stages::{SimpleContextBuilder, PostProcessor, ToolExecutorStage};
/// use mindroid::llm_client::{LlmClient, LlmClientConfig, AuthStyle};
///
/// let mut cfg = LlmClientConfig::new("http://localhost:11434/v1");
/// cfg.default_model = Some("llama3.2".into());
/// cfg.auth_style = AuthStyle::None;
/// let client = LlmClient::new(cfg);
///
/// let registry = Arc::new(
///     ToolRegistry::new()
///         .register(ShellTool::default())
///         .register(OpenTool),
/// );
///
/// let pipeline = Pipeline::new()
///     .add_stage(SimpleContextBuilder)
///     .add_streaming_stage(ToolExecutorStage::new(client, registry))
///     .add_stage(PostProcessor);
/// ```
pub struct ToolExecutorStage {
    client: LlmClient,
    registry: Arc<ToolRegistry>,
    max_iterations: usize,
    parser: Arc<dyn ToolCallParser>,
}

impl ToolExecutorStage {
    pub fn new(client: LlmClient, registry: Arc<ToolRegistry>) -> Self {
        Self {
            client,
            registry,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            parser: Arc::new(XmlToolCallParser),
        }
    }

    /// Override the maximum number of tool-call iterations (default: 10).
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Override the tool call parser (default: [`XmlToolCallParser`]).
    pub fn with_parser(mut self, parser: impl ToolCallParser + 'static) -> Self {
        self.parser = Arc::new(parser);
        self
    }
}

#[async_trait]
impl PipelineStage for ToolExecutorStage {
    fn name(&self) -> &str {
        "ToolExecutorStage"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        // Non-streaming fallback: delegates to run_tool_loop (no yielded events).
        // Operates on local `messages` to avoid borrowing `ctx` through a
        // BoxStream lifetime (which would block the final write to ctx.raw_response).
        let messages = build_messages_with_tools(&ctx.llm_messages, &self.registry);
        let (mut messages, mut final_content, hit_max) = run_tool_loop(
            &self.client,
            &self.registry,
            self.parser.as_ref(),
            messages,
            self.max_iterations,
        )
        .await?;

        if hit_max {
            messages.push(LlmMessage::user(SUMMARY_PROMPT.to_string()));
            let mut llm_stream = self.client.stream_chat(ChatRequest {
                messages: &messages,
                model: None,
                temperature: None,
                max_tokens: None,
                stream: true,
                response_format: None,
            });
            final_content = collect_llm_text(&mut llm_stream).await?;
        }

        ctx.response = Some(final_content);
        Ok(())
    }
}

impl StreamingStage for ToolExecutorStage {
    fn stream<'a>(&'a self, ctx: &'a mut PipelineContext) -> BoxStream<'a, StreamEvent> {
        Box::pin(async_stream::stream! {
            // Clone messages so we can extend them with tool rounds.
            // The original ctx.llm_messages stays untouched.
            let mut messages = build_messages_with_tools(&ctx.llm_messages, &self.registry);

            let mut final_content = String::new();
            let mut hit_max = false;

            for iteration in 0..self.max_iterations {
                debug!("ToolExecutorStage: iteration {}", iteration + 1);

                // stream_chat converts messages into owned OpenAI types immediately
                // and returns a BoxStream<'static, StreamEvent>, so the borrow of
                // `messages` ends as soon as stream_chat returns — before any .await.
                let mut llm_stream = self.client.stream_chat(ChatRequest {
                    messages: &messages,
                    model: None,
                    temperature: None,
                    max_tokens: None,
                    stream: true,
                    response_format: None,
                });

                let mut response_text = String::new();
                // Collect chunks without yielding — we don't know yet whether this
                // iteration is a tool-call round or the final answer. Chunks are only
                // forwarded to the caller once we confirm no tool calls are present.
                let mut collected_chunks: Vec<String> = Vec::new();

                while let Some(event) = llm_stream.next().await {
                    match event {
                        StreamEvent::Chunk { ref content } => {
                            response_text.push_str(content);
                            collected_chunks.push(content.clone());
                        }
                        StreamEvent::Complete { ref content, .. } => {
                            if !content.is_empty() {
                                response_text = content.clone();
                            }
                            // Don't yield Complete yet — we may still have tool rounds.
                        }
                        StreamEvent::Error { .. } => {
                            yield event;
                            return;
                        }
                        other => {
                            yield other;
                        }
                    }
                }

                // Check whether the LLM wants to call any tools.
                let calls: Vec<(String, serde_json::Value)> = self.parser.parse(&response_text)
                    .into_iter()
                    .map(|c| (c.name, c.arguments))
                    .collect();

                tracing::info!(
                    "ToolExecutorStage: iteration {} response ({} chars): {:?}",
                    iteration + 1,
                    response_text.len(),
                    truncate_str(&response_text, 300)
                );
                tracing::info!(
                    "ToolExecutorStage: found {} tool call(s): {:?}",
                    calls.len(),
                    calls.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()
                );

                if calls.is_empty() {
                    // No tool calls — this is the final answer. Now it's safe to
                    // stream the chunks to the caller (no tool XML will be spoken).
                    for chunk in collected_chunks {
                        yield StreamEvent::Chunk { content: chunk };
                    }
                    final_content = response_text;
                    break;
                }

                // Append the assistant turn (which contains the tool calls).
                messages.push(LlmMessage::assistant(&response_text));

                // Execute each tool and collect results.
                let mut results_msg = String::new();
                for (name, args) in calls {
                    yield StreamEvent::ToolCall {
                        name: name.clone(),
                        arguments: args.to_string(),
                    };

                    let result = match self.registry.get(&name) {
                        Some(tool) => match tool.execute(args).await {
                            Ok(out) => out,
                            Err(e) => format!("Error: {e}"),
                        },
                        None => format!("Error: unknown tool '{name}'"),
                    };

                    tracing::info!("ToolExecutorStage: tool '{}' executed → {} bytes: {:?}", name, result.len(), truncate_str(&result, 120));

                    yield StreamEvent::ToolResult {
                        name: name.clone(),
                        result: result.clone(),
                    };

                    results_msg.push_str(&format!(
                        "<tool_result name=\"{name}\">{result}</tool_result>\n"
                    ));
                }

                // Feed the tool results back as a user message.
                messages.push(LlmMessage::user(results_msg));

                if iteration + 1 >= self.max_iterations {
                    warn!(
                        "ToolExecutorStage: reached max iterations ({}), stopping",
                        self.max_iterations
                    );
                    hit_max = true;
                    break;
                }
            }

            if hit_max {
                messages.push(LlmMessage::user(SUMMARY_PROMPT.to_string()));
                let mut llm_stream = self.client.stream_chat(ChatRequest {
                    messages: &messages,
                    model: None,
                    temperature: None,
                    max_tokens: None,
                    stream: true,
                    response_format: None,
                });
                let mut summary = String::new();
                while let Some(event) = llm_stream.next().await {
                    match event {
                        StreamEvent::Chunk { ref content } => {
                            summary.push_str(content);
                            yield event;
                        }
                        StreamEvent::Complete { ref content, .. } => {
                            if !content.is_empty() {
                                summary = content.clone();
                            }
                        }
                        StreamEvent::Error { .. } => {
                            yield event;
                            return;
                        }
                        other => yield other,
                    }
                }
                final_content = summary;
            }

            ctx.response = Some(final_content.clone());
            yield StreamEvent::Complete { content: final_content, usage: None };
        })
    }
}

/// Extract the first balanced `{...}` object from `s`, ignoring braces
/// inside JSON strings. Returns `None` if no balanced object is found.
/// Handles the common LLM issue of emitting extra closing braces or
/// trailing characters after the JSON object.
fn extract_balanced_json(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let bytes = s.as_bytes();
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escaped = false;

    for i in start..bytes.len() {
        let ch = bytes[i] as char;
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Attempt to close a truncated JSON string so it parses correctly.
/// Handles the common case where the LLM stops mid-string inside a tool call.
fn repair_json(s: &str) -> String {
    let mut result = s.trim().to_string();
    let mut in_string = false;
    let mut escaped = false;
    let mut open_braces: i32 = 0;

    for ch in result.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            '{' if !in_string => open_braces += 1,
            '}' if !in_string => open_braces -= 1,
            _ => {}
        }
    }

    if in_string {
        result.push('"');
    }
    for _ in 0..open_braces.max(0) {
        result.push('}');
    }
    result
}

/// Drain a streaming LLM response into a plain `String`, returning an error on
/// `StreamEvent::Error`. Used by both the non-streaming and streaming paths to
/// avoid repeating the same match loop.
async fn collect_llm_text(
    stream: &mut (impl futures::Stream<Item = StreamEvent> + Unpin),
) -> Result<String> {
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        match event {
            StreamEvent::Chunk { content } => text.push_str(&content),
            StreamEvent::Complete { content, .. } if !content.is_empty() => {
                text = content;
            }
            StreamEvent::Error { message } => {
                return Err(MindroidError::Pipeline {
                    stage: "ToolExecutorStage".into(),
                    message,
                    source: None,
                });
            }
            _ => {}
        }
    }
    Ok(text)
}

/// Run the tool-call iteration loop without streaming events to callers.
///
/// Returns `(messages, final_content, hit_max)`:
/// - `messages` is the updated conversation (including all tool rounds).
/// - `final_content` is the LLM's last plain-text response (empty if `hit_max`).
/// - `hit_max` is `true` when the loop was stopped by `max_iterations`.
async fn run_tool_loop(
    client: &LlmClient,
    registry: &ToolRegistry,
    parser: &dyn ToolCallParser,
    mut messages: Vec<LlmMessage>,
    max_iterations: usize,
) -> Result<(Vec<LlmMessage>, String, bool)> {
    let mut final_content = String::new();
    let mut hit_max = false;

    for iteration in 0..max_iterations {
        let mut llm_stream = client.stream_chat(ChatRequest {
            messages: &messages,
            model: None,
            temperature: None,
            max_tokens: None,
            stream: true,
            response_format: None,
        });

        let response_text = collect_llm_text(&mut llm_stream).await?;

        let calls: Vec<(String, serde_json::Value)> = parser
            .parse(&response_text)
            .into_iter()
            .map(|c| (c.name, c.arguments))
            .collect();
        if calls.is_empty() {
            final_content = response_text;
            break;
        }

        messages.push(LlmMessage::assistant(&response_text));
        let mut results_msg = String::new();
        for (name, args) in calls {
            let result = match registry.get(&name) {
                Some(tool) => tool
                    .execute(args)
                    .await
                    .unwrap_or_else(|e| format!("Error: {e}")),
                None => format!("Error: unknown tool '{name}'"),
            };
            results_msg.push_str(&format!(
                "<tool_result name=\"{name}\">{result}</tool_result>\n"
            ));
        }
        messages.push(LlmMessage::user(results_msg));

        if iteration + 1 >= max_iterations {
            warn!(
                "ToolExecutorStage: reached max iterations ({})",
                max_iterations
            );
            hit_max = true;
            break;
        }
    }

    Ok((messages, final_content, hit_max))
}

/// Clone the pipeline messages and inject tool descriptions into the system prompt.
fn build_messages_with_tools(source: &[LlmMessage], registry: &ToolRegistry) -> Vec<LlmMessage> {
    let mut messages = source.to_vec();
    let tool_prompt = registry.system_prompt();
    if !tool_prompt.is_empty() {
        if let Some(sys) = messages.iter_mut().find(|m| m.role == crate::Role::System) {
            sys.content.push_str("\n\n");
            sys.content.push_str(&tool_prompt);
        } else {
            messages.insert(0, LlmMessage::system(tool_prompt));
        }
    }
    messages
}

/// Parse all `<tool_call>{"name": "...", "args": {...}}</tool_call>` blocks from text.
/// Tolerates missing closing `</tool_call>` tag — some LLMs omit it.
fn parse_tool_calls(text: &str) -> Vec<(String, serde_json::Value)> {
    let mut calls = Vec::new();
    let mut haystack = text;

    while let Some(start) = haystack.find("<tool_call>") {
        let after_open = &haystack[start + "<tool_call>".len()..];

        let (json_str, rest) = if let Some(end) = after_open.find("</tool_call>") {
            // Closing tag present — extract JSON between the tags.
            (
                after_open[..end].trim(),
                &after_open[end + "</tool_call>".len()..],
            )
        } else {
            // No closing tag — treat the rest of the text as JSON (LLM omitted it).
            (after_open.trim(), "")
        };

        // Extract only the balanced JSON object — LLMs sometimes emit extra
        // closing braces, stray `>` chars, or other trailing junk.
        let json_str = extract_balanced_json(json_str).unwrap_or(json_str);

        let parsed = serde_json::from_str::<serde_json::Value>(json_str)
            .or_else(|_| serde_json::from_str::<serde_json::Value>(&repair_json(json_str)));

        match parsed {
            Ok(val) => {
                if let (Some(name), Some(args)) =
                    (val.get("name").and_then(|n| n.as_str()), val.get("args"))
                {
                    calls.push((name.to_string(), args.clone()));
                }
            }
            Err(_) => {
                debug!(
                    "parse_tool_calls: failed to parse JSON: {:?}",
                    truncate_str(json_str, 120)
                );
            }
        }

        if rest.is_empty() {
            break;
        }
        haystack = rest;
    }

    calls
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_single_tool_call() {
        let text = r#"Let me check the system.
<tool_call>{"name": "shell", "args": {"command": "pgrep spotify"}}</tool_call>"#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "shell");
        assert_eq!(calls[0].1, json!({"command": "pgrep spotify"}));
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let text = r#"<tool_call>{"name": "shell", "args": {"command": "ls"}}</tool_call>
Some text.
<tool_call>{"name": "open", "args": {"target": "spotify"}}</tool_call>"#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, "shell");
        assert_eq!(calls[1].0, "open");
    }

    #[test]
    fn parse_no_tool_calls() {
        let calls = parse_tool_calls("Just a normal response with no tools.");
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_malformed_json_ignored() {
        let calls = parse_tool_calls("<tool_call>not valid json</tool_call>");
        assert!(calls.is_empty());
    }
}
