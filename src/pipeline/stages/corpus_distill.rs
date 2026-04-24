//! Corpus retrieval and optional distillation stage.
//!
//! Queries a corpus for relevant documents and optionally summarises them
//! through a secondary LLM before appending to the pipeline context.
//!
//! Respects [`CorpusGateDecision`](super::corpus_gate::CorpusGateDecision)
//! if present — skips retrieval when the gate says no.

use std::sync::Arc;

use async_trait::async_trait;

use crate::corpus::CorpusClient;
use crate::error::Result;
use crate::llm_client::{ChatRequest, LlmClient};
use crate::models::LlmMessage;
use crate::pipeline::PipelineContext;
use crate::pipeline::PipelineStage;

use super::corpus_gate::CorpusGateDecision;

const DEFAULT_DISTILL_PROMPT: &str = "\
You are a context distillation assistant. Given a user \
question and retrieved documents, extract only the \
information relevant to answering the question. Be concise \
and preserve key facts, names, and numbers. Do not answer \
the question — only summarise the relevant context.";

/// Pipeline stage that queries a corpus for relevant documents,
/// optionally distills them through a secondary LLM, and appends
/// the result as a system message in `ctx.llm_messages`.
///
/// If a [`CorpusGateDecision`] extension is present and `false`,
/// this stage skips retrieval entirely.
///
/// # Example
///
/// ```ignore
/// use mindroid::pipeline::stages::corpus_distill::CorpusDistillStage;
///
/// // With distillation
/// let stage = CorpusDistillStage::new(corpus_client, "my-corpus-id", Some(distill_llm));
///
/// // Without distillation (raw docs appended)
/// let stage = CorpusDistillStage::new(corpus_client, "my-corpus-id", None);
/// ```
pub struct CorpusDistillStage {
    corpus: Arc<CorpusClient>,
    corpus_id: String,
    distill_llm: Option<Arc<LlmClient>>,
    distill_prompt: String,
}

impl CorpusDistillStage {
    /// Create a new corpus stage.
    ///
    /// Pass `Some(llm)` to distill raw documents through a secondary LLM,
    /// or `None` to append raw documents directly.
    pub fn new(
        corpus: Arc<CorpusClient>,
        corpus_id: impl Into<String>,
        distill_llm: Option<Arc<LlmClient>>,
    ) -> Self {
        Self {
            corpus,
            corpus_id: corpus_id.into(),
            distill_llm,
            distill_prompt: DEFAULT_DISTILL_PROMPT.to_string(),
        }
    }

    /// Override the default distillation system prompt.
    pub fn with_distill_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.distill_prompt = prompt.into();
        self
    }
}

#[async_trait]
impl PipelineStage for CorpusDistillStage {
    fn name(&self) -> &str {
        "CorpusDistill"
    }

    async fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        // Respect gate decision if present
        if let Some(CorpusGateDecision(false)) = ctx.get_ext::<CorpusGateDecision>() {
            tracing::debug!("CorpusDistill: gate decision is SKIP, skipping retrieval");
            return Ok(());
        }

        if ctx.message.content.is_empty() {
            tracing::debug!("CorpusDistill: empty message content, skipping");
            return Ok(());
        }

        // Query corpus
        let raw = match self
            .corpus
            .query(&self.corpus_id, &ctx.message.content, None)
            .await
        {
            Ok(text) if !text.is_empty() => text,
            Ok(_) => {
                tracing::debug!("CorpusDistill: corpus returned empty result");
                return Ok(());
            }
            Err(e) => {
                tracing::warn!("CorpusDistill: corpus query failed, continuing without: {e}");
                return Ok(());
            }
        };

        // Optionally distill through secondary LLM
        let context_text = if let Some(ref llm) = self.distill_llm {
            let distill_messages = vec![
                LlmMessage::system(&self.distill_prompt),
                LlmMessage::user(format!(
                    "User question: {}\n\nDocuments:\n{}",
                    ctx.message.content, raw
                )),
            ];

            match llm
                .chat(ChatRequest {
                    messages: &distill_messages,
                    model: None,
                    temperature: Some(0.0),
                    max_tokens: None,
                    stream: false,
                    response_format: None,
                })
                .await
            {
                Ok((summary, usage)) => {
                    tracing::info!(
                        "CorpusDistill: distilled {} → {} bytes (tokens: {:?})",
                        raw.len(),
                        summary.len(),
                        usage
                    );
                    summary
                }
                Err(e) => {
                    tracing::warn!("CorpusDistill: distillation failed, using raw: {e}");
                    raw
                }
            }
        } else {
            raw
        };

        ctx.llm_messages.push(LlmMessage::system(format!(
            "Reference documents:\n{context_text}"
        )));

        Ok(())
    }
}
