use async_trait::async_trait;
use std::sync::Arc;

use tracing::debug;

use crate::core::context::Context;
#[cfg(feature = "transport-audio")]
use crate::pipeline::extensions::TextInput;
use crate::skills::skillset::SkillSet;
use crate::{LlmMessage, PipelineStage, Result};

#[cfg(feature = "llm-client")]
use crate::core::content::{ContentPart, ContentSource};
#[cfg(feature = "llm-client")]
use crate::models::Role;
#[cfg(feature = "llm-client")]
use crate::pipeline::extensions::FileInputs;

/// A simple context builder that creates LLM messages from an optional
/// system prompt, pre-fetched conversation history, and incoming message content.
///
/// # Examples
///
/// ```ignore
/// // No system prompt, no history
/// SimpleContextBuilder::new()
///
/// // With a system prompt
/// SimpleContextBuilder::with_prompt("You are a helpful assistant.")
///
/// // With pre-fetched conversation history
/// SimpleContextBuilder::with_history(history.clone())
///
/// // With both
/// SimpleContextBuilder::with_prompt_and_history("You are helpful.", history.clone())
/// ```
pub struct SimpleContextBuilder {
    system_prompt: Option<String>,
    history: Arc<Vec<LlmMessage>>,
}

impl SimpleContextBuilder {
    pub fn new() -> Self {
        Self {
            system_prompt: None,
            history: Arc::new(Vec::new()),
        }
    }

    pub fn with_prompt(prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: Some(prompt.into()),
            history: Arc::new(Vec::new()),
        }
    }

    pub fn with_history(history: Arc<Vec<LlmMessage>>) -> Self {
        Self {
            system_prompt: None,
            history,
        }
    }

    pub fn with_prompt_and_history(
        prompt: impl Into<String>,
        history: Arc<Vec<LlmMessage>>,
    ) -> Self {
        Self {
            system_prompt: Some(prompt.into()),
            history,
        }
    }

    /// Append a skill index to the system prompt.
    ///
    /// When skills are present, the index is appended after the base prompt.
    /// No-op if the skill set is empty.
    pub fn with_skills(mut self, skills: &SkillSet) -> Self {
        if !skills.is_empty() {
            let base = self.system_prompt.unwrap_or_default();
            self.system_prompt = Some(format!("{}\n\n{}", base, skills.index()));
        }
        self
    }
}

impl Default for SimpleContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStage for SimpleContextBuilder {
    fn name(&self) -> &str {
        "SimpleContextBuilder"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let mut messages = Vec::new();

        if let Some(ref prompt) = self.system_prompt {
            messages.push(LlmMessage::system(prompt));
        }

        messages.extend(self.history.as_ref().clone());

        #[cfg(feature = "transport-audio")]
        let user_text = ctx
            .get_ext::<TextInput>()
            .map(|t| t.0.as_str())
            .unwrap_or(&ctx.message.content);
        #[cfg(not(feature = "transport-audio"))]
        let user_text = &ctx.message.content;

        // Same rule as the persona builder: a turn nothing has authenticated
        // and claimed is a participant talking, and must not be able to arrive
        // shaped like executed tool output.
        let user_text = if crate::pipeline::claimed_this_message(ctx) {
            user_text.to_string()
        } else {
            crate::core::prompt_text::neutralize_block(user_text)
        };
        messages.push(LlmMessage::user(user_text));
        let current_user = messages.len() - 1;

        debug!(
            "SimpleContextBuilder: {} history messages, {} total llm_messages",
            self.history.len(),
            messages.len(),
        );
        for (i, msg) in messages.iter().enumerate() {
            debug!(
                "  llm_messages[{}] role={} len={}",
                i,
                msg.role.as_str(),
                msg.text().len()
            );
        }

        ctx.llm_messages = messages;
        ctx.set(crate::pipeline::extensions::CurrentUserMessage(
            current_user,
        ));

        Ok(())
    }
}

/// A resolved attachment: its byte/URL source, MIME type, and optional filename.
#[cfg(feature = "llm-client")]
struct ResolvedFile {
    source: ContentSource,
    mime_type: String,
    filename: Option<String>,
}

#[cfg(feature = "llm-client")]
impl ResolvedFile {
    /// Map MIME type to the matching `ContentPart` variant.
    fn into_part(self) -> ContentPart {
        let ResolvedFile {
            source,
            mime_type,
            filename,
        } = self;
        if mime_type.starts_with("image/") {
            ContentPart::image(source, mime_type)
        } else if mime_type.starts_with("audio/") {
            ContentPart::audio(source, mime_type, None)
        } else if mime_type.starts_with("video/") {
            ContentPart::video(source, mime_type, None)
        } else {
            ContentPart::file(source, mime_type, filename)
        }
    }
}

/// Resolves attachments for the current pipeline context.
///
/// Mirrors `resolve_audio` in the STT stage, with input paths (in order):
/// 1. The [`FileInputs`] extension — one or more inline attachments set
///    programmatically.
/// 2. `ctx.message.metadata["image_url"]` — a hosted URL written by a transport,
///    yielding a `ContentSource::Uri`.
/// 3. `ctx.message.metadata["image_data"]` — base64 inline bytes written by a
///    transport, with MIME from `metadata["image_mime"]` (default `image/png`).
///    Requires the `transport-ws` feature.
///
/// The metadata fallbacks are checked only when the extension is absent. Returns
/// an empty vec when nothing is available, so [`AttachMedia`] can no-op.
#[cfg(feature = "llm-client")]
fn resolve_files(ctx: &Context) -> Vec<ResolvedFile> {
    if let Some(files) = ctx.get_ext::<FileInputs>() {
        return files
            .0
            .iter()
            .map(|f| ResolvedFile {
                source: ContentSource::Inline {
                    data: f.data.clone(),
                },
                mime_type: f.mime_type.clone(),
                filename: f.filename.clone(),
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
        return vec![ResolvedFile {
            source: ContentSource::Uri {
                uri: uri.to_string(),
            },
            mime_type: mime(),
            filename: None,
        }];
    }

    #[cfg(feature = "transport-ws")]
    {
        use base64::{Engine, engine::general_purpose::STANDARD};
        if let Some(b64) = ctx
            .message
            .metadata
            .get("image_data")
            .and_then(|v| v.as_str())
        {
            match STANDARD.decode(b64) {
                Ok(data) => {
                    return vec![ResolvedFile {
                        source: ContentSource::Inline { data },
                        mime_type: mime(),
                        filename: None,
                    }];
                }
                Err(e) => debug!("AttachMedia: failed to base64-decode image_data: {e}"),
            }
        }
    }

    Vec::new()
}

/// Attaches media (images, audio, video, or files) to the current user message.
///
/// A small, composable mutator stage. It does **not** build messages itself —
/// place it *after* a context builder (e.g. [`SimpleContextBuilder`]) that has
/// populated `ctx.llm_messages`. It resolves any attachments (see
/// [`resolve_files`]), maps each MIME type to the matching `ContentPart` variant
/// (`image/*` → Image, `audio/*` → Audio, `video/*` → Video, else File), and
/// appends them to the last user message.
///
/// Attachments come from either the [`FileInputs`] context extension (set
/// programmatically) or `message.metadata` (written by a transport) — mirroring
/// how audio flows through `AudioInput` / `metadata["audio_data"]`.
///
/// If there are no attachments, or no user message to attach to, the stage is a
/// no-op pass-through — so the same pipeline serves plain chat and media turns.
///
/// # Note
///
/// Inline bytes are base64-encoded by the LLM client only when the `transport-ws`
/// feature is enabled. The client currently only converts `image/*` to the wire
/// format; other types are attached here but dropped during conversion.
///
/// # Examples
///
/// ```ignore
/// use mindroid::pipeline::extensions::{FileInput, FileInputs};
///
/// let pipeline = Pipeline::new()
///     .add_stage(SimpleContextBuilder::with_prompt_and_history(SYSTEM, history))
///     .add_stage(AttachMedia)
///     .add_streaming_stage(GenericLlmProcessor::new(client))
///     .add_stage(PostProcessor);
///
/// ctx.set_ext(FileInputs::one(FileInput::image(png_bytes, "image/png")));
/// let answer = pipeline.run(&mut ctx).await?;
/// ```
#[cfg(feature = "llm-client")]
pub struct AttachMedia;

#[cfg(feature = "llm-client")]
#[async_trait]
impl PipelineStage for AttachMedia {
    fn name(&self) -> &str {
        "AttachMedia"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let files = resolve_files(ctx);
        if files.is_empty() {
            debug!("AttachMedia: no attachments; pass-through");
            return Ok(());
        }

        match ctx
            .llm_messages
            .iter_mut()
            .rev()
            .find(|m| m.role == Role::User)
        {
            Some(user_msg) => {
                for f in files {
                    user_msg.content.push(f.into_part());
                }
            }
            None => debug!("AttachMedia: no user message to attach to; pass-through"),
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "llm-client"))]
mod media_tests {
    use super::*;
    use crate::config::AgentConfig;
    use crate::models::Message;
    use crate::pipeline::extensions::{FileInput, FileInputs};

    fn ctx_with(content: &str) -> Context {
        let msg = Arc::new(Message::new(content, "user", "ch"));
        Context::new(msg, Arc::new(AgentConfig::default()))
    }

    /// SimpleContextBuilder + AttachMedia compose: text built by the former, the
    /// image appended by the latter.
    #[tokio::test]
    async fn attach_media_appends_image_to_user_message() {
        let mut ctx = ctx_with("What is this?");
        ctx.set_ext(FileInputs::one(FileInput::image(
            vec![1, 2, 3],
            "image/png",
        )));

        SimpleContextBuilder::new().process(&mut ctx).await.unwrap();
        AttachMedia.process(&mut ctx).await.unwrap();

        let msg = ctx.llm_messages.last().unwrap();
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content.len(), 2);
        assert!(matches!(&msg.content[0], ContentPart::Text { text } if text == "What is this?"));
        assert!(matches!(
            &msg.content[1],
            ContentPart::Image { mime_type, .. } if mime_type == "image/png"
        ));
    }

    /// MIME dispatch: each FileInput maps to its matching ContentPart variant,
    /// and multiple attachments all land on the user message.
    #[tokio::test]
    async fn attach_media_dispatches_mime_for_multiple_files() {
        let mut ctx = ctx_with("look at these");
        ctx.set_ext(FileInputs::new(vec![
            FileInput::image(vec![1], "image/png"),
            FileInput {
                data: vec![2],
                mime_type: "audio/wav".into(),
                filename: None,
            },
            FileInput {
                data: vec![3],
                mime_type: "video/mp4".into(),
                filename: None,
            },
            FileInput {
                data: vec![4],
                mime_type: "application/pdf".into(),
                filename: Some("doc.pdf".into()),
            },
        ]));

        SimpleContextBuilder::new().process(&mut ctx).await.unwrap();
        AttachMedia.process(&mut ctx).await.unwrap();

        let parts = &ctx.llm_messages.last().unwrap().content;
        // [text, image, audio, video, file]
        assert_eq!(parts.len(), 5);
        assert!(matches!(parts[1], ContentPart::Image { .. }));
        assert!(matches!(parts[2], ContentPart::Audio { .. }));
        assert!(matches!(parts[3], ContentPart::Video { .. }));
        assert!(matches!(
            &parts[4],
            ContentPart::File { filename: Some(f), mime_type, .. }
                if f == "doc.pdf" && mime_type == "application/pdf"
        ));
    }

    /// Mirroring audio: an image arriving via `metadata["image_data"]` (base64)
    /// is picked up as a fallback when no FileInputs extension is set.
    #[cfg(feature = "transport-ws")]
    #[tokio::test]
    async fn attach_media_reads_metadata_fallback() {
        use base64::{Engine, engine::general_purpose::STANDARD};

        let mut msg = Message::new("what is this?", "user", "ch");
        msg.metadata.insert(
            "image_data".into(),
            serde_json::Value::String(STANDARD.encode([7u8, 8, 9])),
        );
        msg.metadata.insert(
            "image_mime".into(),
            serde_json::Value::String("image/jpeg".into()),
        );
        let mut ctx = Context::new(Arc::new(msg), Arc::new(AgentConfig::default()));

        SimpleContextBuilder::new().process(&mut ctx).await.unwrap();
        AttachMedia.process(&mut ctx).await.unwrap();

        let last = ctx.llm_messages.last().unwrap();
        assert_eq!(last.content.len(), 2);
        assert!(matches!(
            &last.content[1],
            ContentPart::Image { source: ContentSource::Inline { data }, mime_type, .. }
                if data == &[7, 8, 9] && mime_type == "image/jpeg"
        ));
    }

    /// A hosted image URL in `metadata["image_url"]` becomes a Uri image part.
    #[tokio::test]
    async fn attach_media_reads_url_metadata() {
        let mut msg = Message::new("what is this?", "user", "ch");
        msg.metadata.insert(
            "image_url".into(),
            serde_json::Value::String("https://example.com/cat.jpg".into()),
        );
        msg.metadata.insert(
            "image_mime".into(),
            serde_json::Value::String("image/jpeg".into()),
        );
        let mut ctx = Context::new(Arc::new(msg), Arc::new(AgentConfig::default()));

        SimpleContextBuilder::new().process(&mut ctx).await.unwrap();
        AttachMedia.process(&mut ctx).await.unwrap();

        let last = ctx.llm_messages.last().unwrap();
        assert!(matches!(
            &last.content[1],
            ContentPart::Image { source: ContentSource::Uri { uri }, mime_type, .. }
                if uri == "https://example.com/cat.jpg" && mime_type == "image/jpeg"
        ));
    }

    /// Without attachments, AttachMedia is a no-op — message stays text-only.
    #[tokio::test]
    async fn attach_media_is_noop_without_files() {
        let mut ctx = ctx_with("hello");
        SimpleContextBuilder::new().process(&mut ctx).await.unwrap();
        AttachMedia.process(&mut ctx).await.unwrap();

        let msg = ctx.llm_messages.last().unwrap();
        assert_eq!(msg.content.len(), 1);
        assert!(msg.content[0].is_text());
    }

    /// Attachments land on the last *user* message, leaving history intact.
    #[tokio::test]
    async fn attach_media_targets_last_user_message() {
        let history = Arc::new(vec![
            LlmMessage::user("first question"),
            LlmMessage::assistant("first answer"),
        ]);
        let mut ctx = ctx_with("follow-up about the image");
        ctx.set_ext(FileInputs::one(FileInput::image(vec![9], "image/png")));

        SimpleContextBuilder::with_history(history)
            .process(&mut ctx)
            .await
            .unwrap();
        AttachMedia.process(&mut ctx).await.unwrap();

        // [user(first), assistant(first), user(follow-up + image)]
        assert_eq!(ctx.llm_messages.len(), 3);
        assert_eq!(ctx.llm_messages[0].text(), "first question");
        assert_eq!(ctx.llm_messages[1].text(), "first answer");
        let last = ctx.llm_messages.last().unwrap();
        assert_eq!(last.content.len(), 2);
        assert!(matches!(&last.content[1], ContentPart::Image { .. }));
        // history's user message is untouched
        assert_eq!(ctx.llm_messages[0].content.len(), 1);
    }
}
