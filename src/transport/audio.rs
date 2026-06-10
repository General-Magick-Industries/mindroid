use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SizedSample, Stream};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use voice_activity_detector::VoiceActivityDetector;

use crate::MindroidError;
use crate::error::Result;
use crate::models::{ChannelType, Message, MessageType, Response, SenderType};
use crate::transport::Transport;
use crate::voice::types::VadConfig;

// ─── VAD backend ──────────────────────────────────────────────────────────────

/// Which VAD backend to use for voice activity detection.
#[derive(Debug, Clone, Default)]
pub enum VadBackend {
    /// Silero VAD via the `voice_activity_detector` crate. Default.
    #[default]
    Silero,
    /// Cobra VAD (Picovoice). Requires the `cobra` feature and an access key.
    Cobra { access_key: String },
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`AudioTransport`].
///
/// VAD tuning lives in `vad: VadConfig`. Transport-specific knobs (backend
/// selection) remain as top-level fields. Construct with
/// [`AudioTransportConfig::default()`] or override individual `vad` sub-fields:
///
/// ```rust,ignore
/// AudioTransportConfig {
///     vad: VadConfig {
///         silence_duration: Duration::from_millis(800),
///         ..AudioTransportConfig::default().vad
///     },
///     ..AudioTransportConfig::default()
/// }
/// ```
#[derive(Debug, Clone)]
pub struct AudioTransportConfig {
    /// All VAD tuning parameters (thresholds, durations, limits).
    ///
    /// Note: the default silence window for this transport is **1200 ms** —
    /// deliberately different from `VadConfig::default()` (500 ms) because the
    /// transport is optimised for conversational microphone input where a longer
    /// pause is needed to reliably detect end-of-turn.
    pub vad: VadConfig,
    /// Which VAD backend to use. Default: [`VadBackend::Silero`].
    pub vad_backend: VadBackend,
}

impl Default for AudioTransportConfig {
    fn default() -> Self {
        Self {
            // Transport-specific VAD defaults — silence is 1200 ms (NOT the
            // VadConfig::default() 500 ms) because microphone input requires a
            // longer inter-turn gap to avoid premature cutoff.
            vad: VadConfig {
                speech_threshold: 0.5,
                speech_end_threshold: 0.3,
                silence_duration: Duration::from_millis(1200),
                speech_pad: Duration::from_millis(300),
                min_speech: Duration::from_millis(300),
                max_utterance: Duration::from_secs(30),
            },
            vad_backend: VadBackend::Silero,
        }
    }
}

// ─── Transport ────────────────────────────────────────────────────────────────

/// Transport that captures audio from the default microphone and emits
/// utterances as [`Message`]s with `message_type = Audio`.
///
/// Each message carries the raw WAV bytes base64-encoded under
/// `metadata["audio_data"]`. [`SttStage`](crate::pipeline::stages::stt::SttStage)
/// reads this automatically when the `audio` feature is enabled.
///
/// Audio output (TTS playback) is handled separately by
/// [`TtsStage`](crate::pipeline::stages::tts::TtsStage).
///
/// # Note on ONNX Runtime
///
/// This transport depends on [`voice_activity_detector`] which uses the `ort`
/// crate (ONNX Runtime). On first build, `ort` may download the ONNX Runtime
/// binary. Set `ORT_DYLIB_PATH` to point to an existing `libonnxruntime.so` to
/// skip the download.
pub struct AudioTransport {
    agent_id: String,
    config: AudioTransportConfig,
    connected: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
}

impl AudioTransport {
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            config: AudioTransportConfig::default(),
            connected: Arc::new(AtomicBool::new(false)),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn with_config(agent_id: impl Into<String>, config: AudioTransportConfig) -> Self {
        Self {
            agent_id: agent_id.into(),
            config,
            connected: Arc::new(AtomicBool::new(false)),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[async_trait]
impl Transport for AudioTransport {
    fn name(&self) -> &str {
        "audio"
    }

    async fn connect(&mut self) -> Result<()> {
        cpal::default_host()
            .default_input_device()
            .ok_or_else(|| MindroidError::Transport {
                message: "No default audio input device found".into(),
                source: None,
            })?;
        self.shutdown.store(false, Ordering::SeqCst);
        self.connected.store(true, Ordering::SeqCst);
        info!("AudioTransport: connected");
        Ok(())
    }

    async fn disconnect(&mut self) -> Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        self.connected.store(false, Ordering::SeqCst);
        info!("AudioTransport: disconnected");
        Ok(())
    }

    async fn listen(&self, tx: mpsc::Sender<Message>) -> Result<()> {
        let config = self.config.clone();
        let agent_id = self.agent_id.clone();
        let shutdown = self.shutdown.clone();

        // All cpal and VAD work runs inside a detached blocking thread because
        // cpal::Stream is !Send and cannot live across .await points.
        // We detach (don't await) so the runtime's message loop can start
        // immediately and receive the messages we send via tx.
        tokio::task::spawn_blocking(move || {
            let result = (|| -> Result<()> {
                let host = cpal::default_host();
                let device =
                    host.default_input_device()
                        .ok_or_else(|| MindroidError::Transport {
                            message: "No default audio input device found".into(),
                            source: None,
                        })?;

                let stream_config = preferred_input_config(&device)?;
                let sample_rate = stream_config.sample_rate().0;

                if sample_rate != 16_000 && sample_rate != 8_000 {
                    return Err(MindroidError::Transport {
                        message: format!(
                            "Sample rate {sample_rate} Hz is not supported by Silero VAD \
                             (needs 8000 or 16000). Configure your device or add resampling."
                        ),
                        source: None,
                    });
                }

                // Cobra requires 16kHz; reject early if the device fell back to 8kHz.
                if matches!(config.vad_backend, VadBackend::Cobra { .. }) && sample_rate != 16_000 {
                    return Err(MindroidError::Transport {
                        message: format!(
                            "Cobra VAD requires 16000 Hz sample rate, but device is at \
                             {sample_rate} Hz. Configure your input device to use 16 kHz."
                        ),
                        source: None,
                    });
                }

                info!(
                    "AudioTransport: listening on '{}' at {} Hz {} ch",
                    device.name().unwrap_or_default(),
                    sample_rate,
                    stream_config.channels(),
                );

                let chunk_size: usize = if sample_rate == 16_000 { 512 } else { 256 };

                let (frame_tx, frame_rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(512);

                // build_input_stream now owns the accumulator + mono downmix so
                // the channel always delivers exactly chunk_size mono samples.
                let stream = build_input_stream(&device, &stream_config, frame_tx, chunk_size)?;
                stream.play().map_err(|e| MindroidError::Transport {
                    message: format!("Failed to start audio stream: {e}"),
                    source: None,
                })?;

                // Initialise the selected VAD backend.
                let mut processor: Box<dyn VadProcessor> = match &config.vad_backend {
                    VadBackend::Silero => {
                        let vad = VoiceActivityDetector::builder()
                            .sample_rate(sample_rate)
                            .chunk_size(chunk_size)
                            .build()
                            .map_err(|e| MindroidError::Transport {
                                message: format!("VAD init failed: {e}"),
                                source: None,
                            })?;
                        Box::new(SileroProcessor { detector: vad })
                    }
                    VadBackend::Cobra { access_key } => {
                        #[cfg(feature = "transport-audio")]
                        let proc: Box<dyn VadProcessor> = {
                            warn!(
                                "AudioTransport: Cobra VAD stub active — pv_cobra crate is not \
                                 available on crates.io (all versions yanked). VAD will return \
                                 silence probability 0.0. Add a real pv_cobra dep to enable \
                                 Cobra processing."
                            );
                            Box::new(CobraProcessor {
                                _access_key: access_key.clone(),
                            })
                        };
                        #[cfg(not(feature = "transport-audio"))]
                        let proc: Box<dyn VadProcessor> = {
                            let _ = access_key;
                            warn!(
                                "AudioTransport: Cobra VAD requested but 'cobra' feature is not \
                                 enabled — falling back to Silero"
                            );
                            let vad = VoiceActivityDetector::builder()
                                .sample_rate(sample_rate)
                                .chunk_size(chunk_size)
                                .build()
                                .map_err(|e| MindroidError::Transport {
                                    message: format!("VAD init failed: {e}"),
                                    source: None,
                                })?;
                            Box::new(SileroProcessor { detector: vad })
                        };
                        proc
                    }
                };

                vad_loop(
                    frame_rx,
                    tx,
                    config,
                    agent_id,
                    sample_rate,
                    chunk_size,
                    processor.as_mut(),
                    shutdown,
                );

                drop(stream);
                Ok(())
            })();

            if let Err(e) = result {
                error!("AudioTransport: audio loop error: {e}");
            }
        });

        Ok(())
    }

    /// AudioTransport is input-only. Text responses are ignored here.
    /// Audio output is handled by [`TtsStage`] and a separate playback helper.
    async fn send(&self, _response: &Response) -> Result<Option<String>> {
        Ok(None)
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }
}

// ─── VAD processor trait ──────────────────────────────────────────────────────

trait VadProcessor {
    /// Process one frame of f32 mono audio. Returns voice probability in [0.0, 1.0].
    fn process(&mut self, samples: &[f32]) -> f32;
}

struct SileroProcessor {
    detector: VoiceActivityDetector,
}

impl VadProcessor for SileroProcessor {
    fn process(&mut self, samples: &[f32]) -> f32 {
        self.detector.predict(samples.iter().copied())
    }
}

/// Placeholder for Cobra VAD. The `pv_cobra` crate has all crates.io versions
/// yanked; when an accessible version is available, replace this stub with the
/// real implementation that calls `pv_cobra::Cobra::process`.
#[cfg(feature = "transport-audio")]
struct CobraProcessor {
    /// Forwarded to the real Cobra SDK when available.
    _access_key: String,
}

#[cfg(feature = "transport-audio")]
impl VadProcessor for CobraProcessor {
    fn process(&mut self, _samples: &[f32]) -> f32 {
        // Stub — real impl converts samples to i16 and calls cobra.process().
        0.0
    }
}

// ─── cpal helpers ─────────────────────────────────────────────────────────────

/// Try to get an input config at 16 kHz (ideal for Silero VAD + Whisper).
/// Falls back to the device default if 16 kHz is not available.
fn preferred_input_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
    if let Ok(mut configs) = device.supported_input_configs()
        && let Some(range) =
            configs.find(|c| c.min_sample_rate().0 <= 16_000 && c.max_sample_rate().0 >= 16_000)
    {
        return Ok(range.with_sample_rate(cpal::SampleRate(16_000)));
    }
    device
        .default_input_config()
        .map_err(|e| MindroidError::Transport {
            message: format!("No supported input config: {e}"),
            source: None,
        })
}

/// Build a cpal input stream.
///
/// Each format branch owns an accumulator that normalises samples to f32,
/// downmixes to mono, and only emits exactly `chunk_size` samples at a time.
/// This prevents panics on platforms (e.g. Windows/WASAPI) that deliver
/// variable-size buffers.
fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    frame_tx: std::sync::mpsc::SyncSender<Vec<f32>>,
    chunk_size: usize,
) -> Result<Stream> {
    let stream_config = config.config();
    let err_fn = |e: cpal::StreamError| error!("AudioTransport: stream error: {e}");

    let stream = match config.sample_format() {
        SampleFormat::F32 => {
            build_stream_for_format(device, &stream_config, frame_tx, chunk_size, err_fn, |s| s)
        }
        SampleFormat::I16 => build_stream_for_format(
            device,
            &stream_config,
            frame_tx,
            chunk_size,
            err_fn,
            |s: i16| s as f32 / i16::MAX as f32,
        ),
        SampleFormat::I32 => build_stream_for_format(
            device,
            &stream_config,
            frame_tx,
            chunk_size,
            err_fn,
            |s: i32| s as f32 / i32::MAX as f32,
        ),
        SampleFormat::U8 => build_stream_for_format(
            device,
            &stream_config,
            frame_tx,
            chunk_size,
            err_fn,
            |s: u8| (s as f32 - 128.0) / 128.0,
        ),
        other => {
            return Err(MindroidError::Transport {
                message: format!("Unsupported sample format: {other:?}"),
                source: None,
            });
        }
    }
    .map_err(|e| MindroidError::Transport {
        message: format!("Failed to build input stream: {e}"),
        source: None,
    })?;

    Ok(stream)
}

/// Generic helper that builds a typed cpal input stream.
///
/// `normalize` converts one sample of type `T` to an `f32` in `[-1.0, 1.0]`.
/// The callback accumulates mono-downmixed frames and emits exactly
/// `chunk_size` samples at a time to `sender`.
fn build_stream_for_format<T, N, E>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sender: std::sync::mpsc::SyncSender<Vec<f32>>,
    chunk_size: usize,
    err_fn: E,
    normalize: N,
) -> std::result::Result<Stream, cpal::BuildStreamError>
where
    T: SizedSample + Send + 'static,
    N: Fn(T) -> f32 + Send + 'static,
    E: FnMut(cpal::StreamError) + Send + 'static,
{
    let channels = config.channels as usize;
    let mut accumulator: Vec<f32> = Vec::new();
    device.build_input_stream(
        config,
        move |data: &[T], _| {
            for frame in data.chunks(channels) {
                let mono = frame.iter().map(|&s| normalize(s)).sum::<f32>() / channels as f32;
                accumulator.push(mono);
            }
            while accumulator.len() >= chunk_size {
                let chunk: Vec<f32> = accumulator.drain(..chunk_size).collect();
                let _ = sender.try_send(chunk);
            }
        },
        err_fn,
        None,
    )
}

// ─── VAD loop ─────────────────────────────────────────────────────────────────

/// Core VAD loop driven by [`crate::voice::AudioFrontend`].
///
/// The externally-computed probability (from the [`VadProcessor`] backend) is
/// fed into [`AudioFrontend::process`] along with the raw f32 frame.
/// [`FrontendEvent::UtteranceComplete`] carries the accumulated samples; we
/// WAV-encode them and forward exactly as before.
///
/// The local-mic transport has no agent-playback path, so `agent_speaking` is
/// always `false` — the barge-in gate inside the frontend stays inert.
/// A monotonic `Instant::now()` is supplied at each call site (no clock
/// knowledge required inside the pure `voice/` core).
///
/// Frame-count arithmetic (at 16 kHz / 512-sample chunk):
///   silence : 1200 ms × 16000 / 1000 = 19200 samples → div_ceil(512) = 38 frames
///   pad     :  300 ms × 16000 / 1000 =  4800 samples → div_ceil(512) = 10 frames (≈ 300 ms)
///   min     :  300 ms × 16000 / 1000 =  4800 samples (threshold on buffer length)
///   max     :   30 s  × 16000       = 480000 samples (hard cap)
///
/// ## 8 kHz compatibility
///
/// The real `sample_rate` is passed to the builder via `.sample_rate_hz(..)`, so
/// `AudioFrontend` derives `min_utterance_samples` / `max_utterance_samples`
/// correctly at both 16 kHz and 8 kHz — preserving behavioral parity with the
/// former hand-rolled loop.
#[allow(clippy::too_many_arguments)]
fn vad_loop(
    frame_rx: std::sync::mpsc::Receiver<Vec<f32>>,
    msg_tx: mpsc::Sender<Message>,
    config: AudioTransportConfig,
    agent_id: String,
    sample_rate: u32,
    chunk_size: usize,
    processor: &mut dyn VadProcessor,
    shutdown: Arc<AtomicBool>,
) {
    use crate::voice::encode_wav;
    use crate::voice::{
        AudioFrontend, FrontendEvent,
        types::{BargeInMode, TurnDetection},
    };

    // chunk_duration_ms is the same for both supported rates:
    //   16000 Hz / 512 samples = 32 ms
    //    8000 Hz / 256 samples = 32 ms
    let chunk_duration_ms = (chunk_size as u64 * 1_000) / sample_rate as u64;

    let mut frontend = AudioFrontend::builder(config.vad.clone(), chunk_duration_ms)
        // Pass the real rate so Duration→sample thresholds are correct at 8 kHz too.
        .sample_rate_hz(sample_rate)
        // Local-mic transport has no agent playback; barge-in gate is inert.
        .barge_in_mode(BargeInMode::Disabled)
        // Silence-based auto-completion mirrors the old hand-rolled logic.
        .turn_detection(TurnDetection::Local(config.vad.clone()))
        .build();

    loop {
        match frame_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(frame) => {
                // frame is already mono with exactly chunk_size samples
                // (guaranteed by the accumulator in build_input_stream).
                let prob = processor.process(&frame);

                // Monotonic clock at the call site — pure voice/ core does not
                // call Instant::now() itself.
                let now = std::time::Instant::now();

                // Local-mic transport has no agent playback, so agent_speaking
                // is always false; the barge-in gate remains inert.
                let events = frontend.process(&frame, prob, false, now);

                for event in events {
                    match event {
                        FrontendEvent::SpeechStarted => {
                            debug!("AudioTransport: speech started (prob={prob:.3})");
                        }
                        FrontendEvent::UtteranceComplete { samples, .. } => {
                            let dur = samples.len() as f32 / sample_rate as f32;
                            debug!("AudioTransport: utterance complete ({dur:.1}s)");
                            let wav = encode_wav(&samples, sample_rate);
                            let msg = build_message(wav, &agent_id);
                            if msg_tx.blocking_send(msg).is_err() {
                                info!("AudioTransport: channel closed, exiting");
                                return;
                            }
                        }
                        // BargeIn is not reachable (mode = Disabled), but
                        // handle exhaustively for forward-compatibility.
                        FrontendEvent::BargeIn { .. } => {}
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if shutdown.load(Ordering::Relaxed) {
                    info!("AudioTransport: shutdown requested, exiting VAD loop");
                    return;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                info!("AudioTransport: audio stream closed");
                return;
            }
        }
    }
}

// ─── Message construction ─────────────────────────────────────────────────────

fn build_message(wav_bytes: Vec<u8>, agent_id: &str) -> Message {
    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        "audio_data".to_string(),
        serde_json::Value::String(STANDARD.encode(&wav_bytes)),
    );
    Message {
        id: uuid::Uuid::new_v4().to_string(),
        content: String::new(),
        sender_id: "microphone".to_string(),
        sender_type: SenderType::User,
        channel_id: agent_id.to_string(),
        channel_type: ChannelType::Direct,
        message_type: MessageType::Audio,
        timestamp: chrono::Utc::now(),
        metadata,
        platform: Some("microphone".into()),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Parity test: the migrated `AudioTransportConfig::default()` must produce
    /// byte-identical derived frame thresholds to the pre-migration flat fields.
    ///
    /// Arithmetic at 16 kHz / 512-sample chunk:
    ///   silence : 1200 ms × 16000 Hz / 1000 = 19200 samples → div_ceil(512) = 38 frames
    ///   pad     :  300 ms × 16000 Hz / 1000 =  4800 samples → div_ceil(512) = 10 frames
    ///   min     :  300 ms × 16000 Hz / 1000 =  4800 samples  (buffer-length threshold)
    ///   max     :   30  s × 16000 Hz        = 480000 samples (hard cap)
    #[test]
    fn audio_transport_config_default_parity() {
        const SAMPLE_RATE: usize = 16_000;
        const CHUNK_SIZE: usize = 512;

        let cfg = AudioTransportConfig::default();
        let vad = &cfg.vad;

        // Silence frame count (same arithmetic as vad_loop).
        let silence_samples = (vad.silence_duration.as_millis() as usize * SAMPLE_RATE) / 1000;
        let silence_frames = silence_samples.div_ceil(CHUNK_SIZE);
        assert_eq!(silence_frames, 38, "silence_frames mismatch");

        // Pad frame count.
        let pad_samples = (vad.speech_pad.as_millis() as usize * SAMPLE_RATE) / 1000;
        let pad_frames = pad_samples.div_ceil(CHUNK_SIZE);
        assert_eq!(pad_frames, 10, "pad_frames mismatch");

        // Minimum utterance samples.
        let min_samples = (vad.min_speech.as_millis() as usize * SAMPLE_RATE) / 1000;
        assert_eq!(min_samples, 4800, "min_utterance_samples mismatch");

        // Maximum utterance samples.
        let max_samples = (vad.max_utterance.as_millis() as usize * SAMPLE_RATE) / 1000;
        assert_eq!(max_samples, 480_000, "max_utterance_samples mismatch");
    }
}
