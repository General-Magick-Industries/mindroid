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
