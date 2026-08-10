/// Audio input bytes stored in [`PipelineContext`] extensions.
///
/// Set by `AudioTransport` or any stage that produces raw audio input.
/// Consumed by `SttStage`.
#[cfg(feature = "transport-audio")]
pub struct AudioInput(pub Vec<u8>);

/// Synthesized audio output bytes stored in [`PipelineContext`] extensions.
///
/// Set by `TtsStage`. Can be read by an audio output transport or inline
/// playback helper.
#[cfg(feature = "transport-audio")]
pub struct AudioOutput(pub Vec<u8>);

/// Text transcript stored in [`PipelineContext`] extensions.
///
/// Set by `SttStage` after transcription so downstream stages (e.g.
/// `SimpleContextBuilder`) see the spoken text rather than the raw audio
/// placeholder in `message.content`.
#[cfg(feature = "transport-audio")]
pub struct TextInput(pub String);

/// Index of the current inbound user turn in [`Context::llm_messages`].
///
/// Context builders set this after assembling history so stages that transform
/// message content never mistake an older user message for the active turn.
/// Custom context builders should set it when they append the inbound message.
///
/// [`Context::llm_messages`]: crate::Context::llm_messages
#[derive(Debug, Clone, Copy)]
pub struct CurrentUserMessage(pub usize);

pub(crate) struct PersistedUserTurn(pub String);

pub(crate) struct CorrelatedRemoteResult;

/// A single binary attachment (image, audio, video, or arbitrary file) to send
/// to the LLM, stored in [`PipelineContext`] extensions as part of [`FileInputs`].
///
/// Set by a transport or any upstream stage. Consumed by `AttachMedia`, which
/// maps `mime_type` to the matching [`crate::core::content::ContentPart`] variant
/// (`image/*` → Image, `audio/*` → Audio, `video/*` → Video, else File) and
/// appends it to the user [`crate::LlmMessage`].
///
/// `data` is the raw bytes; `mime_type` is e.g. `"image/png"`. Inline bytes are
/// base64-encoded into a `data:` URL by the LLM client, which requires the
/// `transport-ws` feature. `filename` is used for the `File` variant.
///
/// Note: the LLM client currently only converts `image/*` to the OpenAI wire
/// format; other types are accepted here but dropped during conversion until the
/// client gains support for them.
#[cfg(feature = "llm-client")]
#[derive(Debug, Clone)]
pub struct FileInput {
    pub data: Vec<u8>,
    pub mime_type: String,
    pub filename: Option<String>,
}

#[cfg(feature = "llm-client")]
impl FileInput {
    /// Convenience constructor for an image attachment (no filename).
    pub fn image(data: Vec<u8>, mime_type: impl Into<String>) -> Self {
        Self {
            data,
            mime_type: mime_type.into(),
            filename: None,
        }
    }
}

/// Context extension holding one or more [`FileInput`] attachments for the
/// current turn. Consumed by `AttachMedia`.
#[cfg(feature = "llm-client")]
#[derive(Debug, Clone, Default)]
pub struct FileInputs(pub Vec<FileInput>);

#[cfg(feature = "llm-client")]
impl FileInputs {
    pub fn new(files: Vec<FileInput>) -> Self {
        Self(files)
    }

    /// Wrap a single file as a one-element [`FileInputs`].
    pub fn one(file: FileInput) -> Self {
        Self(vec![file])
    }
}

#[cfg(all(test, feature = "llm-client"))]
mod tests {
    use super::*;

    #[test]
    fn image_constructor_sets_mime_and_leaves_filename_unset() {
        let f = FileInput::image(b"png-bytes".to_vec(), "image/png");
        assert_eq!(f.data, b"png-bytes");
        assert_eq!(f.mime_type, "image/png");
        assert!(f.filename.is_none(), "images carry no filename");
    }

    #[test]
    fn one_wraps_a_single_file_and_new_preserves_order() {
        let single = FileInputs::one(FileInput::image(vec![1], "image/png"));
        assert_eq!(single.0.len(), 1);

        let many = FileInputs::new(vec![
            FileInput::image(vec![1], "image/png"),
            FileInput::image(vec![2], "image/jpeg"),
        ]);
        let mimes: Vec<&str> = many.0.iter().map(|f| f.mime_type.as_str()).collect();
        assert_eq!(mimes, ["image/png", "image/jpeg"]);
    }

    /// `Default` is what a stage gets when no attachment was set, so it must be
    /// empty rather than a one-element placeholder.
    #[test]
    fn default_is_empty() {
        assert!(FileInputs::default().0.is_empty());
    }
}
