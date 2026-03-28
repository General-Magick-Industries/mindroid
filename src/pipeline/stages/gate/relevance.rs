//! LLM-based relevance gate for multi-agent pipelines.
//!
//! The [`RelevanceGate`] is a pipeline stage that decides whether an agent
//! should engage with an incoming message. It uses a lightweight LLM call
//! (e.g. local Ollama) with **structured JSON output** to get a deterministic
//! true/false decision.
//!
//! # Design
//!
//! The gate treats every participant as a first-class citizen — it does not
//! check `sender_type` or distinguish between humans and agents. It decides
//! purely on:
//!
//! - **Role** — what is this agent's domain of expertise?
//! - **Content** — is the message relevant to that domain?
//! - **Context** — given the conversation so far, should this agent speak now?
//!
//! The LLM is constrained to return `{"relevant": true}` or `{"relevant": false}`
//! via JSON schema enforcement, eliminating ambiguity from free-text classification.
//!
//! When the gate decides the agent should not engage, it sets `ctx.halted = true`
//! and the pipeline stops. Otherwise the pipeline continues to the next stage.
//!
//! # Example
//!
//! ```ignore
//! use mindroid::Pipeline;
//! use mindroid::pipeline::stages::gate::RelevanceGate;
//! use mindroid::pipeline::stages::{GenericLlmProcessor, PostProcessor};
//! use mindroid::llm_client::{AuthStyle, LlmClient, LlmClientConfig};
//!
//! let gate = RelevanceGate::new(
//!     "budget planning — budgets, expenses, savings, income, financial goals",
//!     "http://localhost:11434",
//!     "smallthinker",
//! );
//!
//! let mut config = LlmClientConfig::new("http://localhost:11434/v1");
//! config.auth_style = AuthStyle::None;
//! config.default_model = Some("llama3.2".to_string());
//! let client = LlmClient::new(config).unwrap();
//!
//! let pipeline = Pipeline::new()
//!     .add_stage(gate)
//!     .add_streaming_stage(GenericLlmProcessor::new(client))
//!     .add_stage(PostProcessor);
//! ```

use async_trait::async_trait;
use std::sync::Arc;

use async_openai::types::chat::{ResponseFormat, ResponseFormatJsonSchema};
use serde::Deserialize;

use crate::llm_client::{AuthStyle, ChatRequest, LlmClient, LlmClientConfig};
use crate::{LlmMessage, MindroidError, PipelineContext, PipelineStage, Result, Role};

use super::Gate;

fn gate_err(message: String) -> MindroidError {
    MindroidError::Pipeline {
        stage: "RelevanceGate".to_string(),
        message,
        source: None,
    }
}

/// JSON response from the gate LLM call.
#[derive(Deserialize)]
struct GateResponse {
    relevant: bool,
}

/// An LLM-based gate that halts the pipeline when the incoming message
/// is not relevant to the agent's role.
///
/// The gate runs a cheap, fast LLM call (typically a small local model)
/// with structured JSON output to get a deterministic boolean decision.
/// It does not distinguish between human and agent senders — relevance
/// is determined purely by content, role, and conversation history.
///
/// # Customisation
///
/// For custom gate logic, implement [`PipelineStage`] directly and set
/// `ctx.halted = true` to stop the pipeline. This struct is a convenience
/// for the common LLM-classification pattern.
pub struct RelevanceGate {
    /// A short description of the agent's domain of expertise.
    role: String,
    /// Additional instructions appended to the classification prompt.
    custom_instructions: Option<String>,
    /// LLM client for the classification call (typically local Ollama).
    client: LlmClient,
    /// Model to use for classification.
    model: String,
    /// When true, halt the pipeline if the LLM call fails.
    /// When false (default), let the message through on error.
    strict: bool,
    /// Conversation history injected at construction time.
    history: Arc<Vec<LlmMessage>>,
}

impl RelevanceGate {
    /// Create a new relevance gate.
    ///
    /// - `role`: a short description of the agent's domain (e.g. "budget planning").
    /// - `ollama_url`: base URL for the local Ollama instance.
    /// - `model`: which model to use for classification (e.g. "smallthinker").
    pub fn new(role: &str, ollama_url: &str, model: &str) -> crate::Result<Self> {
        let mut config = LlmClientConfig::new(format!("{ollama_url}/v1"));
        config.auth_style = AuthStyle::None;
        let client = LlmClient::new(config)?;

        Ok(Self {
            role: role.to_string(),
            custom_instructions: None,
            client,
            model: model.to_string(),
            strict: false,
            history: Arc::new(Vec::new()),
        })
    }

    /// Create from a pre-resolved [`LlmClientConfig`] (e.g. from [`MindroidConfig::llm`]).
    ///
    /// Uses `default_model` from the config as the classification model,
    /// falling back to `"smallthinker"`.
    pub fn from_config(
        role: &str,
        llm_config: crate::llm_client::LlmClientConfig,
    ) -> crate::Result<Self> {
        let model = llm_config
            .default_model
            .clone()
            .unwrap_or_else(|| "smallthinker".into());
        let client = LlmClient::new(llm_config)?;
        Ok(Self {
            role: role.to_string(),
            custom_instructions: None,
            client,
            model,
            strict: false,
            history: Arc::new(Vec::new()),
        })
    }

    /// Create a relevance gate using any pre-configured [`LlmClient`].
    ///
    /// Use this when the classifier runs on a remote API, needs custom auth,
    /// or shares a client with other stages.
    pub fn with_client(role: &str, client: LlmClient, model: &str) -> Self {
        Self {
            role: role.to_string(),
            custom_instructions: None,
            client,
            model: model.to_string(),
            strict: false,
            history: Arc::new(Vec::new()),
        }
    }

    /// Provide conversation history for context-aware relevance decisions.
    pub fn with_history(mut self, history: Arc<Vec<LlmMessage>>) -> Self {
        self.history = history;
        self
    }

    /// Append custom instructions to the classification prompt.
    ///
    /// Use this to add domain-specific rules, e.g.:
    /// ```ignore
    /// gate.instructions("Always engage when the word 'cost' appears.")
    /// ```
    pub fn instructions(mut self, instructions: &str) -> Self {
        self.custom_instructions = Some(instructions.to_string());
        self
    }

    /// When strict mode is enabled, the gate halts the pipeline if the LLM
    /// call fails (network error, timeout, etc). By default, the gate lets
    /// the message through on error so the agent can still respond.
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Returns `true` if the gate is in strict mode.
    pub fn is_strict(&self) -> bool {
        self.strict
    }

    /// Classify the message in `ctx` against this gate's role.
    ///
    /// Returns `true` if relevant (gate passes), `false` if not relevant (gate would halt).
    /// On LLM / parse failure: returns `Err` when strict, `Ok(true)` otherwise.
    pub async fn run_classification(&self, ctx: &PipelineContext) -> crate::Result<bool> {
        let messages = self.build_prompt(&ctx.message.content);

        tracing::debug!(
            "RelevanceGate [{}]: prompt system={:?}",
            self.role,
            messages.first().map(|m| &m.content),
        );

        let result = self
            .client
            .chat(ChatRequest {
                messages: &messages,
                model: Some(&self.model),
                temperature: Some(0.0),
                max_tokens: None,
                stream: false,
                response_format: Some(Self::response_format()),
            })
            .await;

        let raw = match result {
            Ok((text, _usage)) => text,
            Err(e) => {
                if self.strict {
                    return Err(gate_err(format!("Classification failed: {e}")));
                }
                tracing::warn!("RelevanceGate: LLM call failed ({e}), letting message through");
                return Ok(true);
            }
        };

        let relevant = match serde_json::from_str::<GateResponse>(&raw) {
            Ok(resp) => resp.relevant,
            Err(e) => {
                tracing::warn!(
                    "RelevanceGate [{}]: failed to parse JSON ({e}), raw={raw}; letting message through",
                    self.role,
                );
                true
            }
        };

        Ok(relevant)
    }

    fn build_prompt(&self, message_content: &str) -> Vec<LlmMessage> {
        let mut system = format!(
            "Your role: {role}\n\
             \n\
             Given the conversation context and the latest message, is the latest message \
             directed at your role? Should YOU be the one to respond?\n\
             You MUST respond with JSON: {{\"relevant\": true}} or {{\"relevant\": false}}",
            role = self.role,
        );

        if let Some(ref extra) = self.custom_instructions {
            system.push_str("\n\n");
            system.push_str(extra);
        }

        let mut messages = vec![LlmMessage::system(&system)];

        // Include conversation history so the gate understands the flow.
        // Only include user/assistant messages (skip system context like knowledge/corpus
        // to keep the prompt small for the gate model).
        for msg in self.history.iter() {
            if msg.role == Role::User || msg.role == Role::Assistant {
                messages.push(msg.clone());
            }
        }

        // The latest message to classify
        messages.push(LlmMessage::user(message_content));
        messages
    }

    fn response_format() -> ResponseFormat {
        ResponseFormat::JsonSchema {
            json_schema: ResponseFormatJsonSchema {
                name: "gate_decision".to_string(),
                description: Some(
                    "Whether the message is relevant to the agent's role".to_string(),
                ),
                schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "relevant": {
                            "type": "boolean"
                        }
                    },
                    "required": ["relevant"],
                    "additionalProperties": false
                })),
                strict: Some(true),
            },
        }
    }
}

#[async_trait]
impl PipelineStage for RelevanceGate {
    fn name(&self) -> &str {
        "RelevanceGate"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let relevant = self.run_classification(ctx).await?;

        if relevant {
            tracing::info!(
                "RelevanceGate [{}]: engaging with message from {}",
                self.role,
                ctx.message.sender_id,
            );
            // Signal that the gate passed so callers can distinguish
            // "gate passed" (Some) from "gate halted" (None).
            ctx.response = Some(ctx.message.content.clone());
        } else {
            tracing::info!("RelevanceGate [{}]: not relevant", self.role,);
            ctx.halted = true;
        }

        Ok(())
    }
}

#[async_trait]
impl Gate for RelevanceGate {
    async fn classify(&self, ctx: &PipelineContext) -> Result<bool> {
        self.run_classification(ctx).await
    }
}
