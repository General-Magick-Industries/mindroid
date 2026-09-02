//! Generic inbound ingest stage: composes a [`Source`] and an [`Encoder`].
//!
//! This is the swappable replacement for the hardcoded `AttachMedia` stage. It
//! collects raw incoming attachments from the context (the [`FileInputs`]
//! extension, or `message.metadata` image fields), resolves each via its
//! `Source`, encodes via its `Encoder`, and appends the resulting parts to the
//! last user message. Behavior is entirely in the two injected impls.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::debug;

use crate::core::context::Context;
use crate::ingest::{Base64Source, Encoder, MediaEncoder, RawInput, Source};
use crate::models::Role;
use crate::pipeline::extensions::FileInputs;
use crate::{PipelineStage, Result};

/// Composes a `Source` (resolve raw input → bytes/uri) and an `Encoder`
/// (bytes/uri → `ContentPart`s), appending the result to the last user message.
pub struct IngestStage {
    source: Arc<dyn Source>,
    encoder: Arc<dyn Encoder>,
}

impl IngestStage {
    pub fn new(source: Arc<dyn Source>, encoder: Arc<dyn Encoder>) -> Self {
        Self { source, encoder }
    }

    /// The default pairing: [`Base64Source`] + [`MediaEncoder`], reproducing the
    /// built-in `AttachMedia` behavior.
    pub fn default_media() -> Self {
        Self {
            source: Arc::new(Base64Source),
            encoder: Arc::new(MediaEncoder),
        }
    }
}

/// Collect raw incoming attachments from the context, mirroring the input paths
/// of the legacy `resolve_files`: the [`FileInputs`] extension first, then the
/// `message.metadata` image fields (`image_url`, then `image_data` + `image_mime`).
fn collect_raw_inputs(ctx: &Context) -> Vec<RawInput> {
    if let Some(files) = ctx.get_ext::<FileInputs>() {
        return files
            .0
            .iter()
            .map(|f| RawInput::Bytes {
                data: f.data.clone(),
                mime_type: f.mime_type.clone(),
            })
            .collect();
    }

    let mime = || {
        ctx.message
            .metadata
            .get("image_mime")
            .and_then(|v| v.as_str())
            .unwrap_or("image/png")
            .to_string()
    };

    if let Some(uri) = ctx
        .message
        .metadata
        .get("image_url")
        .and_then(|v| v.as_str())
    {
        return vec![RawInput::Uri {
            uri: uri.to_string(),
            mime_type: mime(),
        }];
    }

    if let Some(b64) = ctx
        .message
        .metadata
        .get("image_data")
        .and_then(|v| v.as_str())
    {
        return vec![RawInput::Base64 {
            b64: b64.to_string(),
            mime_type: mime(),
        }];
    }

    Vec::new()
}

#[async_trait]
impl PipelineStage for IngestStage {
    fn name(&self) -> &str {
        "IngestStage"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let inputs = collect_raw_inputs(ctx);
        if inputs.is_empty() {
            debug!("IngestStage: no attachments; pass-through");
            return Ok(());
        }

        let mut parts = Vec::new();
        for input in &inputs {
            let resolved = self.source.resolve(input).await?;
            parts.extend(self.encoder.encode(&resolved).await?);
        }

        match ctx
            .llm_messages
            .iter_mut()
            .rev()
            .find(|m| m.role == Role::User)
        {
            Some(user_msg) => user_msg.content.extend(parts),
            None => debug!("IngestStage: no user message to attach to; pass-through"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;
    use crate::core::content::{ContentPart, ContentSource};
    use crate::ingest::ResolvedSource;
    use crate::models::{LlmMessage, Message};
    use crate::pipeline::extensions::{FileInput, FileInputs};

    fn ctx_with_user(content: &str) -> Context {
        let msg = Arc::new(Message::new(content, "user", "ch"));
        let mut ctx = Context::new(msg, Arc::new(AgentConfig::default()));
        ctx.llm_messages.push(LlmMessage::user(content));
        ctx
    }

    #[tokio::test]
    async fn ingest_appends_image_to_user_message() {
        let mut ctx = ctx_with_user("look");
        ctx.set_ext(FileInputs::one(FileInput::image(
            vec![1, 2, 3],
            "image/png",
        )));

        IngestStage::default_media()
            .process(&mut ctx)
            .await
            .unwrap();

        let last = ctx.llm_messages.last().unwrap();
        assert!(matches!(
            last.content.last().unwrap(),
            ContentPart::Image { .. }
        ));
    }

    #[tokio::test]
    async fn ingest_is_noop_without_inputs() {
        let mut ctx = ctx_with_user("hi");
        IngestStage::default_media()
            .process(&mut ctx)
            .await
            .unwrap();
        let last = ctx.llm_messages.last().unwrap();
        assert_eq!(last.content.len(), 1); // still text-only
    }

    #[tokio::test]
    async fn base64_source_decodes_and_media_encoder_dispatches() {
        use base64::{Engine, engine::general_purpose::STANDARD};
        let b64 = STANDARD.encode([0xDE, 0xAD, 0xBE, 0xEF]);
        let resolved = Base64Source
            .resolve(&RawInput::Base64 {
                b64,
                mime_type: "image/jpeg".into(),
            })
            .await
            .unwrap();
        match &resolved.source {
            ContentSource::Inline { data } => assert_eq!(data, &[0xDE, 0xAD, 0xBE, 0xEF]),
            _ => panic!("expected inline bytes"),
        }
        let parts = MediaEncoder.encode(&resolved).await.unwrap();
        assert!(matches!(parts[0], ContentPart::Image { .. }));
    }

    #[tokio::test]
    async fn swapped_encoder_changes_behavior_without_stage_change() {
        // A custom Encoder that always emits a File part proves behavior lives in
        // the impl, not the stage wiring.
        struct AlwaysFile;
        #[async_trait]
        impl Encoder for AlwaysFile {
            async fn encode(&self, r: &ResolvedSource) -> Result<Vec<ContentPart>> {
                Ok(vec![ContentPart::file(
                    r.source.clone(),
                    r.mime_type.clone(),
                    Some("forced file".into()),
                )])
            }
        }

        let mut ctx = ctx_with_user("look");
        ctx.set_ext(FileInputs::one(FileInput::image(vec![1], "image/png")));
        let stage = IngestStage::new(Arc::new(Base64Source), Arc::new(AlwaysFile));
        stage.process(&mut ctx).await.unwrap();

        let last = ctx.llm_messages.last().unwrap();
        assert!(matches!(
            last.content.last().unwrap(),
            ContentPart::File { .. }
        ));
    }
}
