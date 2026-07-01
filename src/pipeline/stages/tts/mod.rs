#[cfg(feature = "speech")]
mod cartesia;
#[cfg(all(feature = "speech", feature = "transport-ws"))]
mod cartesia_streaming;
#[cfg(feature = "speech")]
mod deepgram;
#[cfg(feature = "llm-client")]
mod openai;

#[cfg(feature = "speech")]
pub use cartesia::{CartesiaTts, CartesiaTtsConfig, GenerationConfig};
#[cfg(all(feature = "speech", feature = "transport-ws"))]
pub use cartesia_streaming::CartesiaStreamingTts;
#[cfg(feature = "speech")]
pub use deepgram::{DeepgramTts, DeepgramTtsConfig};
#[cfg(feature = "llm-client")]
pub use openai::{OpenAiTts, OpenAiTtsConfig};

use async_trait::async_trait;
use std::sync::Arc;

use tracing::warn;

#[cfg(feature = "transport-audio")]
use futures::StreamExt;
#[cfg(feature = "transport-audio")]
use rodio;

use crate::core::context::Context;
use crate::error::Result;
#[cfg(feature = "transport-audio")]
use crate::pipeline::extensions::AudioOutput;
use crate::{MindroidError, PipelineStage};

/// Converts text to synthesized audio bytes.
#[async_trait]
pub trait TtsProvider: Send + Sync {
    /// Returns raw audio bytes (MP3 or WAV depending on provider).
    async fn synthesize(&self, text: &str) -> Result<Vec<u8>>;
}

/// Pipeline stage that synthesizes the agent response to audio.
///
/// Reads `ctx.response`, calls the configured [`TtsProvider`], and stores
/// the resulting audio bytes in `ctx.audio_output`. A downstream audio
/// transport or inline playback helper can then consume `ctx.audio_output`.
///
/// Skips synthesis and logs a warning if there is no response text.
pub struct TtsStage {
    provider: Arc<dyn TtsProvider>,
}

impl TtsStage {
    pub fn new(provider: impl TtsProvider + 'static) -> Self {
        Self {
            provider: Arc::new(provider),
        }
    }

    pub fn from_arc(provider: Arc<dyn TtsProvider>) -> Self {
        Self { provider }
    }
}

#[async_trait]
impl PipelineStage for TtsStage {
    fn name(&self) -> &str {
        "TtsStage"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let text = ctx.response.as_deref().unwrap_or("");

        if text.trim().is_empty() {
            warn!("TtsStage: no response text to synthesize, skipping");
            return Ok(());
        }

        let clean = strip_markdown(text);

        let audio =
            self.provider
                .synthesize(&clean)
                .await
                .map_err(|e| MindroidError::Pipeline {
                    stage: "TtsStage".into(),
                    message: format!("TTS synthesis failed: {e}"),
                    source: None,
                })?;

        #[cfg(feature = "transport-audio")]
        {
            ctx.set_ext(AudioOutput(audio));
        }
        #[cfg(not(feature = "transport-audio"))]
        {
            let _ = audio;
        }
        Ok(())
    }
}

// ─── StreamingTtsTransformer ──────────────────────────────────────────────────

/// Standalone transformer that synthesizes audio sentence-by-sentence as
/// LLM tokens arrive, playing each sentence immediately without waiting for the
/// full response.
///
/// Users who want streaming TTS compose this manually by calling
/// [`StreamingTtsTransformer::transform`] on any [`BoxStream`] of
/// [`StreamEvent`]s. It is no longer a first-class pipeline concept.
///
/// All [`StreamEvent`]s are re-emitted unchanged so handlers and
/// post-streaming stages still see the full output. TTS is triggered only on
/// [`StreamEvent::Chunk`] events; tool call and result events pass through
/// without being synthesized.
#[cfg(feature = "transport-audio")]
pub struct StreamingTtsTransformer {
    tts: Arc<dyn TtsProvider>,
    cmd_tx: std::sync::mpsc::SyncSender<AudioPlaybackCmd>,
}

#[cfg(feature = "transport-audio")]
impl StreamingTtsTransformer {
    pub fn new(tts: impl TtsProvider + 'static) -> Self {
        Self::from_arc(Arc::new(tts))
    }

    pub fn from_arc(tts: Arc<dyn TtsProvider>) -> Self {
        let cmd_tx = spawn_audio_playback_thread("StreamingTtsTransformer", 4);
        Self { tts, cmd_tx }
    }
}

#[cfg(feature = "transport-audio")]
impl StreamingTtsTransformer {
    /// Transform a stream of [`StreamEvent`]s, synthesizing and playing audio
    /// sentence-by-sentence as chunks arrive. All events are re-emitted
    /// unchanged so callers still see the full output.
    pub fn transform<'a>(
        &'a self,
        stream: futures::stream::BoxStream<'a, crate::models::StreamEvent>,
    ) -> futures::stream::BoxStream<'a, crate::models::StreamEvent> {
        let tts = Arc::clone(&self.tts);
        let cmd_tx = self.cmd_tx.clone();

        Box::pin(async_stream::stream! {
            const MICRO_BATCH_CHARS: usize = 180;

            let mut buf = String::new();
            let mut stream = stream;

            while let Some(event) = stream.next().await {
                match &event {
                    crate::models::StreamEvent::Chunk { content } => {
                        buf.push_str(content);
                        // Spawn a TTS task every ~180 chars without blocking chunk consumption.
                        if buf.len() >= MICRO_BATCH_CHARS {
                            let chunk = buf.trim().to_string();
                            buf.clear();
                            if !chunk.is_empty() {
                                let tts = Arc::clone(&tts);
                                let cmd_tx = cmd_tx.clone();
                                tokio::spawn(async move {
                                    if let Ok(audio) = tts.synthesize(&chunk).await {
                                        let (done_tx, _done_rx) = std::sync::mpsc::sync_channel(1);
                                        let _ = cmd_tx.send(AudioPlaybackCmd { bytes: audio, done_tx });
                                    }
                                });
                            }
                        }
                    }
                    crate::models::StreamEvent::Complete { .. } => {
                        // Flush remainder as a background task — don't block the pipeline.
                        let remainder = buf.trim().to_string();
                        if !remainder.is_empty() {
                            let tts = Arc::clone(&tts);
                            let cmd_tx = cmd_tx.clone();
                            tokio::spawn(async move {
                                if let Ok(audio) = tts.synthesize(&remainder).await {
                                    let (done_tx, _done_rx) = std::sync::mpsc::sync_channel(1);
                                    let _ = cmd_tx.send(AudioPlaybackCmd { bytes: audio, done_tx });
                                }
                            });
                        }
                    }
                    _ => {}
                }
                yield event;
            }
        })
    }
}

// ─── AudioOutputStage ─────────────────────────────────────────────────────────

#[cfg(feature = "transport-audio")]
struct AudioPlaybackCmd {
    bytes: Vec<u8>,
    done_tx: std::sync::mpsc::SyncSender<()>,
}

/// Spawn a background thread that receives [`AudioPlaybackCmd`]s and plays each
/// one through the default audio output device. A fresh `OutputStream` + `Sink`
/// is created per command to avoid silent-playback issues on Linux ALSA/PipeWire
/// after the first command completes.
///
/// `caller` is used only in error-log messages; `buffer_size` sets the
/// sync-channel capacity (use 1 for back-pressure, 4 for look-ahead).
#[cfg(feature = "transport-audio")]
fn spawn_audio_playback_thread(
    caller: &'static str,
    buffer_size: usize,
) -> std::sync::mpsc::SyncSender<AudioPlaybackCmd> {
    let (cmd_tx, cmd_rx) = std::sync::mpsc::sync_channel::<AudioPlaybackCmd>(buffer_size);
    std::thread::spawn(move || {
        while let Ok(cmd) = cmd_rx.recv() {
            let (_stream, handle) = match rodio::OutputStream::try_default() {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!("{caller}: output device error: {e}");
                    let _ = cmd.done_tx.send(());
                    continue;
                }
            };
            let sink = match rodio::Sink::try_new(&handle) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("{caller}: sink error: {e}");
                    let _ = cmd.done_tx.send(());
                    continue;
                }
            };
            match rodio::Decoder::new(std::io::Cursor::new(cmd.bytes)) {
                Ok(source) => {
                    sink.append(source);
                    sink.sleep_until_end();
                }
                Err(e) => tracing::error!("{caller}: decode error: {e}"),
            }
            std::thread::sleep(std::time::Duration::from_millis(80));
            let _ = cmd.done_tx.send(());
        }
    });
    cmd_tx
}

/// Pipeline stage that plays synthesized audio through the default output device.
///
/// Reads `ctx.audio_output` (set by [`TtsStage`]), sends the bytes to a
/// dedicated audio thread that holds a persistent [`rodio::OutputStream`] for
/// its lifetime, and blocks until playback finishes. Using a persistent stream
/// keeps one audio device registration with the OS instead of opening and
/// closing it on every response.
///
/// Place this stage after `TtsStage` in the pipeline:
/// ```ignore
/// Pipeline::new()
///     /* ... STT, LLM stages ... */
///     .add_stage(TtsStage::new(provider))
///     .add_stage(AudioOutputStage::new())
/// ```
#[cfg(feature = "transport-audio")]
pub struct AudioOutputStage {
    cmd_tx: std::sync::mpsc::SyncSender<AudioPlaybackCmd>,
}

#[cfg(feature = "transport-audio")]
impl AudioOutputStage {
    pub fn new() -> Self {
        // Buffer size 1 provides back-pressure: AudioOutputStage plays one clip
        // at a time and waits for done_rx before accepting the next command.
        let cmd_tx = spawn_audio_playback_thread("AudioOutputStage", 1);
        Self { cmd_tx }
    }
}

#[cfg(feature = "transport-audio")]
impl Default for AudioOutputStage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "transport-audio")]
#[async_trait]
impl PipelineStage for AudioOutputStage {
    fn name(&self) -> &str {
        "AudioOutputStage"
    }

    async fn process(&self, ctx: &mut Context) -> Result<()> {
        let audio = match ctx.take_ext::<AudioOutput>() {
            Some(a) => a.0,
            None => {
                warn!("AudioOutputStage: no audio to play, skipping");
                return Ok(());
            }
        };
        let (done_tx, done_rx) = std::sync::mpsc::sync_channel(1);
        self.cmd_tx
            .send(AudioPlaybackCmd {
                bytes: audio,
                done_tx,
            })
            .map_err(|_| MindroidError::Pipeline {
                stage: "AudioOutputStage".into(),
                message: "Audio playback thread is not running".into(),
                source: None,
            })?;
        tokio::task::spawn_blocking(move || done_rx.recv())
            .await
            .map_err(|e| MindroidError::Pipeline {
                stage: "AudioOutputStage".into(),
                message: format!("Playback task panicked: {e}"),
                source: None,
            })?
            .map_err(|_| MindroidError::Pipeline {
                stage: "AudioOutputStage".into(),
                message: "Audio playback thread disconnected".into(),
                source: None,
            })
    }
}

// ─── Markdown stripping ───────────────────────────────────────────────────────

/// Strip common markdown syntax so TTS reads clean prose instead of symbols.
fn strip_markdown(text: &str) -> String {
    let mut out = String::with_capacity(text.len());

    for line in text.lines() {
        let trimmed = line.trim_start();

        // Fenced code blocks — drop the fence lines, keep content as-is
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            continue;
        }

        // Headings: # Title → "Title"
        let line = if let Some(rest) = trimmed
            .strip_prefix("######")
            .or_else(|| trimmed.strip_prefix("#####"))
            .or_else(|| trimmed.strip_prefix("####"))
            .or_else(|| trimmed.strip_prefix("###"))
            .or_else(|| trimmed.strip_prefix("##"))
            .or_else(|| trimmed.strip_prefix('#'))
        {
            rest.trim()
        } else {
            line
        };

        // Blockquotes: > text → "text"
        let line = line.trim_start_matches('>').trim_start();

        // Horizontal rules
        let stripped = line.trim();
        if stripped == "---" || stripped == "***" || stripped == "___" {
            continue;
        }

        // Bullet / numbered list markers: "- item", "* item", "1. item"
        let line = if let Some(rest) = line
            .trim_start()
            .strip_prefix("- ")
            .or_else(|| line.trim_start().strip_prefix("* "))
            .or_else(|| line.trim_start().strip_prefix("+ "))
        {
            rest
        } else {
            line
        };

        // Remove numbered list prefix "1. ", "12. "
        let line = {
            let t = line.trim_start();
            let dot_pos = t.find(". ");
            if let Some(pos) = dot_pos {
                if t[..pos].chars().all(|c| c.is_ascii_digit()) {
                    &t[pos + 2..]
                } else {
                    line
                }
            } else {
                line
            }
        };

        // Inline: bold (**x** / __x__), italic (*x* / _x_), code (`x`)
        let line = inline_strip(line);

        if !line.trim().is_empty() {
            out.push_str(line.trim());
            out.push(' ');
        }
    }

    out.trim().to_string()
}

/// Remove inline markdown spans: **bold**, *italic*, `code`, [text](url).
fn inline_strip(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            // Bold **x** or __x__
            '*' | '_' if i + 1 < chars.len() && chars[i + 1] == chars[i] => {
                let marker = chars[i];
                i += 2;
                while i < chars.len() {
                    if chars[i] == marker && i + 1 < chars.len() && chars[i + 1] == marker {
                        i += 2;
                        break;
                    }
                    result.push(chars[i]);
                    i += 1;
                }
            }
            // Italic *x* or _x_
            '*' | '_' => {
                let marker = chars[i];
                i += 1;
                while i < chars.len() {
                    if chars[i] == marker {
                        i += 1;
                        break;
                    }
                    result.push(chars[i]);
                    i += 1;
                }
            }
            // Inline code `x`
            '`' => {
                i += 1;
                while i < chars.len() {
                    if chars[i] == '`' {
                        i += 1;
                        break;
                    }
                    result.push(chars[i]);
                    i += 1;
                }
            }
            // Links [text](url) → text
            '[' => {
                i += 1;
                let mut link_text = String::new();
                while i < chars.len() && chars[i] != ']' {
                    link_text.push(chars[i]);
                    i += 1;
                }
                // skip ](url)
                if i < chars.len() && chars[i] == ']' {
                    i += 1;
                    if i < chars.len() && chars[i] == '(' {
                        i += 1;
                        while i < chars.len() && chars[i] != ')' {
                            i += 1;
                        }
                        if i < chars.len() {
                            i += 1;
                        }
                    }
                }
                result.push_str(&link_text);
            }
            c => {
                result.push(c);
                i += 1;
            }
        }
    }

    result
}
