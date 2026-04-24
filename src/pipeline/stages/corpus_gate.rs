//! LLM-based corpus gate that classifies whether a message needs document retrieval.
//!
//! Stores a [`CorpusGateDecision`] in `PipelineContext` extensions for
//! downstream stages (like [`CorpusDistillStage`](super::corpus_distill::CorpusDistillStage))
//! to read.

use async_trait::async_trait;

use std::sync::Arc;

use crate::error::Result;
use crate::llm_client::{ChatRequest, LlmClient};
use crate::models::LlmMessage;
use crate::pipeline::{PipelineContext, PipelineStage};

/// Typed extension stored by `CorpusGateStage`, read by `CorpusDistillStage`.
///
/// `true` means the message needs corpus retrieval, `false` means skip.
pub struct CorpusGateDecision(pub bool);

const DEFAULT_GATE_PROMPT: &str = "\
You are a message classifier. Determine if the user's message \
requires looking up reference documents to answer properly.\n\n\
Reply with ONLY \"yes\" or \"no\".\n\n\
Say \"no\" for: greetings, small talk, thank you, yes/no answers, \
and simple conversational messages.\n\
Say \"yes\" for: questions about features, APIs, configuration, \
troubleshooting, how-to requests, or anything that needs factual \
knowledge to answer.";

/// Pipeline stage that uses a lightweight LLM call to decide whether
/// corpus retrieval is needed for the current message.
///
/// Unlike [`RelevanceGate`](super::gate::RelevanceGate), this stage
/// does **not** halt the pipeline — it stores a [`CorpusGateDecision`]
/// extension that downstream stages can check.
///
/// # Example
///
/// ```ignore
/// use mindroid::pipeline::stages::corpus_gate::CorpusGateStage;
///
/// let pipeline = Pipeline::new()
///     .add_stage(CorpusGateStage::new(gate_llm))
///     .add_stage(CorpusDistillStage::new(corpus, corpus_id, Some(distill_llm)))
///     .add_streaming_stage(GenericLlmProcessor::new(main_llm))
///     .add_stage(PostProcessor);
/// ```
pub struct CorpusGateStage {
    llm: Arc<LlmClient>,
    system_prompt: String,
}

impl CorpusGateStage {
    /// Create a gate with the default classification prompt.
    pub fn new(llm: Arc<LlmClient>) -> Self {
        Self {
            llm,
            system_prompt: DEFAULT_GATE_PROMPT.to_string(),
        }
    }

    /// Create a gate with a custom classification prompt.
    pub fn with_prompt(llm: Arc<LlmClient>, system_prompt: impl Into<String>) -> Self {
        Self {
            llm,
            system_prompt: system_prompt.into(),
        }
    }
}

#[async_trait]
impl PipelineStage for CorpusGateStage {
    fn name(&self) -> &str {
        "CorpusGate"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        let messages = vec![
            LlmMessage::system(&self.system_prompt),
            LlmMessage::user(&ctx.message.content),
        ];

        let needs_corpus = match self
            .llm
            .chat(ChatRequest {
                messages: &messages,
                model: None,
                temperature: Some(0.0),
                max_tokens: Some(3),
                stream: false,
                response_format: None,
            })
            .await
        {
            Ok((answer, _)) => {
                let needs = answer.trim().to_lowercase().contains("yes");
                tracing::info!(
                    "CorpusGate: \"{}\" → {}",
                    ctx.message.content,
                    if needs { "QUERY" } else { "SKIP" }
                );
                needs
            }
            Err(e) => {
                tracing::warn!("CorpusGate classification failed, querying anyway: {e}");
                true
            }
        };

        ctx.set_ext(CorpusGateDecision(needs_corpus));
        Ok(())
    }
}
