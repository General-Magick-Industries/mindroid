//! Desktop audio implementation using `cpal` (microphone input) and `rodio`
//! (speaker output).
//!
//! Feature-gated behind `transport-audio`. No resampling is performed here —
//! it is the caller's responsibility to ensure the sample rate matches the
//! hardware device's native rate.
//!
//! # cpal / rodio threading notes
//!
//! `cpal::Stream` is `!Send`, so it must live on the thread that built it.
//! `CpalAudioSource` keeps the stream alive in a `spawn_blocking` thread and
//! bridges audio data to async via `tokio::sync::mpsc`.
//!
//! `rodio::OutputStream` is also `!Send`; `CpalAudioSink` stores it on a
//! dedicated OS thread and exposes async wrappers via a command channel.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SizedSample};
use futures::Stream;
use std::pin::Pin;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info};

use crate::core::error::MindroidError;
use crate::omni::audio::{AudioSink, AudioSource};
use crate::omni::types::AudioChunk;

// ─── Sink command channel ──────────────────────────────────────────────────────

/// Commands sent to the rodio playback thread.
enum SinkCmd {
    Play(AudioChunk),
    Stop,
    Flush(tokio::sync::oneshot::Sender<()>),
}

// ─── CpalAudioSource ──────────────────────────────────────────────────────────

/// [`AudioSource`] backed by `cpal` microphone capture.
///
/// On construction a `cpal` input stream is started in a `spawn_blocking`
/// thread. Raw PCM frames are converted to [`AudioChunk`] and forwarded
/// through a `tokio::sync::mpsc` channel. [`AudioSource::stream`] wraps
/// that receiver as an async [`Stream`].
///
/// Note: no resampling — the stream uses the device's native sample rate.
/// Callers requiring a specific rate must resample externally.
pub struct CpalAudioSource {
    /// Async receiver of captured chunks.
    rx: Mutex<Option<mpsc::Receiver<AudioChunk>>>,
    sample_rate: u32,
    /// Signals the capture thread to stop.
    shutdown: Arc<AtomicBool>,
}

impl CpalAudioSource {
    /// Start capturing from the default input device.
    pub fn new(sample_rate: u32) -> Result<Self, MindroidError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| MindroidError::Transport {
                message: "No default audio input device found".into(),
                source: None,
            })?;

        // Try to get the requested sample rate; fall back to device default.
        let stream_config = preferred_input_config_at(&device, sample_rate)?;
        let actual_rate = stream_config.sample_rate().0;

        info!(
            "CpalAudioSource: opening '{}' at {} Hz {} ch",
            device.name().unwrap_or_default(),
            actual_rate,
            stream_config.channels(),
        );

        let (chunk_tx, chunk_rx) = mpsc::channel::<AudioChunk>(256);
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        // cpal::Stream is !Send, so we keep it alive in a blocking thread.
        tokio::task::spawn_blocking(move || {
            let (raw_tx, raw_rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(512);
            let config = stream_config.config();
            let channels = config.channels as usize;

            let err_fn = |e: cpal::StreamError| error!("CpalAudioSource: stream error: {e}");

            let stream_result = match stream_config.sample_format() {
                SampleFormat::F32 => build_source_stream::<f32, _, _>(
                    &device,
                    &config,
                    raw_tx,
                    channels,
                    err_fn,
                    |s| s,
                ),
                SampleFormat::I16 => build_source_stream::<i16, _, _>(
                    &device,
                    &config,
                    raw_tx,
                    channels,
                    err_fn,
                    |s| s as f32 / i16::MAX as f32,
                ),
                SampleFormat::I32 => build_source_stream::<i32, _, _>(
                    &device,
                    &config,
                    raw_tx,
                    channels,
                    err_fn,
                    |s| s as f32 / i32::MAX as f32,
                ),
                SampleFormat::U8 => build_source_stream::<u8, _, _>(
                    &device,
                    &config,
                    raw_tx,
                    channels,
                    err_fn,
                    |s| (s as f32 - 128.0) / 128.0,
                ),
                other => {
                    error!("CpalAudioSource: unsupported sample format {other:?}");
                    return;
                }
            };

            let stream = match stream_result {
                Ok(s) => s,
                Err(e) => {
                    error!("CpalAudioSource: failed to build stream: {e}");
                    return;
                }
            };

            if let Err(e) = stream.play() {
                error!("CpalAudioSource: failed to start stream: {e}");
                return;
            }

            // Relay f32 mono frames to async chunk_tx as raw i16 PCM bytes.
            loop {
                if shutdown_clone.load(Ordering::Relaxed) {
                    info!("CpalAudioSource: shutdown, stopping capture");
                    break;
                }
                match raw_rx.recv_timeout(std::time::Duration::from_millis(50)) {
                    Ok(frames) => {
                        // Convert f32 → i16 PCM bytes.
                        let data: Vec<u8> = frames
                            .iter()
                            .flat_map(|&s| {
                                let s16 = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                                s16.to_le_bytes()
                            })
                            .collect();
                        let chunk = AudioChunk {
                            data,
                            sample_rate: actual_rate,
                            channels: 1,
                            bits_per_sample: 16,
                        };
                        if chunk_tx.blocking_send(chunk).is_err() {
                            info!("CpalAudioSource: receiver dropped, stopping");
                            break;
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        info!("CpalAudioSource: raw channel closed");
                        break;
                    }
                }
            }
            drop(stream);
        });

        Ok(Self {
            rx: Mutex::new(Some(chunk_rx)),
            sample_rate: actual_rate,
            shutdown,
        })
    }
}

impl Drop for CpalAudioSource {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

impl AudioSource for CpalAudioSource {
    /// Returns an async stream of captured [`AudioChunk`]s.
    ///
    /// The stream ends when the capture thread stops or is dropped. Only
    /// the first call yields the stream; subsequent calls return an empty
    /// stream (the receiver is consumed on first call).
    fn stream(&self) -> Pin<Box<dyn Stream<Item = AudioChunk> + Send + '_>> {
        let rx = self
            .rx
            .lock()
            .expect("CpalAudioSource: mutex poisoned")
            .take();
        match rx {
            Some(receiver) => Box::pin(ReceiverStream::new(receiver)),
            // Already consumed — return an immediately-finished stream.
            None => Box::pin(futures::stream::empty()),
        }
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

// ─── CpalAudioSink ────────────────────────────────────────────────────────────

/// [`AudioSink`] backed by a `rodio` sink for speaker playback.
///
/// `rodio::OutputStream` is `!Send`, so this type manages a dedicated OS
/// thread that owns the output stream and processes commands from an `mpsc`
/// channel.
///
/// - [`play`](AudioSink::play)  — appends PCM data to the rodio queue.
/// - [`stop`](AudioSink::stop)  — calls `sink.clear()` for instant barge-in.
/// - [`flush`](AudioSink::flush) — blocks (in a `spawn_blocking` thread) until
///   the rodio queue is drained.
pub struct CpalAudioSink {
    /// Sender for playback commands to the rodio thread.
    cmd_tx: std::sync::mpsc::SyncSender<SinkCmd>,
    sample_rate: u32,
}

impl CpalAudioSink {
    /// Open the default output device and start the playback thread.
    pub fn new(sample_rate: u32) -> Result<Self, MindroidError> {
        let (cmd_tx, cmd_rx) = std::sync::mpsc::sync_channel::<SinkCmd>(64);

        // rodio::OutputStream is !Send — lives on a dedicated OS thread.
        std::thread::spawn(move || {
            let (_stream, handle) = match rodio::OutputStream::try_default() {
                Ok(v) => v,
                Err(e) => {
                    error!("CpalAudioSink: output device error: {e}");
                    return;
                }
            };
            let sink = match rodio::Sink::try_new(&handle) {
                Ok(s) => s,
                Err(e) => {
                    error!("CpalAudioSink: sink init error: {e}");
                    return;
                }
            };

            info!("CpalAudioSink: rodio sink ready at {sample_rate} Hz");

            while let Ok(cmd) = cmd_rx.recv() {
                match cmd {
                    SinkCmd::Play(chunk) => {
                        // Wrap raw i16-LE PCM in a rodio SamplesBuffer.
                        let samples: Vec<i16> = chunk
                            .data
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .copied()
                            .map(i16::from_le_bytes)
                            .collect();
                        let source = rodio::buffer::SamplesBuffer::new(
                            chunk.channels,
                            chunk.sample_rate,
                            samples,
                        );
                        sink.append(source);
                    }
                    SinkCmd::Stop => {
                        sink.clear();
                    }
                    SinkCmd::Flush(done_tx) => {
                        sink.sleep_until_end();
                        let _ = done_tx.send(());
                    }
                }
            }
        });

        Ok(Self {
            cmd_tx,
            sample_rate,
        })
    }
}

#[async_trait]
impl AudioSink for CpalAudioSink {
    /// Append PCM audio data to the rodio playback queue.
    ///
    /// The chunk must contain raw 16-bit little-endian signed PCM.
    async fn play(&self, chunk: AudioChunk) -> Result<(), MindroidError> {
        self.cmd_tx
            .try_send(SinkCmd::Play(chunk))
            .map_err(|e| MindroidError::Transport {
                message: format!("CpalAudioSink: play command failed: {e}"),
                source: None,
            })
    }

    /// Wait for all queued audio to finish playing.
    async fn flush(&self) -> Result<(), MindroidError> {
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        self.cmd_tx
            .try_send(SinkCmd::Flush(done_tx))
            .map_err(|e| MindroidError::Transport {
                message: format!("CpalAudioSink: flush command failed: {e}"),
                source: None,
            })?;
        // Wait for drain confirmation without blocking the async runtime.
        done_rx.await.map_err(|_| MindroidError::Transport {
            message: "CpalAudioSink: flush response channel dropped".into(),
            source: None,
        })
    }

    /// Immediately discard all queued audio (barge-in support).
    async fn stop(&self) -> Result<(), MindroidError> {
        self.cmd_tx
            .try_send(SinkCmd::Stop)
            .map_err(|e| MindroidError::Transport {
                message: format!("CpalAudioSink: stop command failed: {e}"),
                source: None,
            })
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

// ─── CpalAudio ────────────────────────────────────────────────────────────────

/// Convenience handle that bundles a [`CpalAudioSource`] and a
/// [`CpalAudioSink`] for the same nominal sample rate.
///
/// The actual capture sample rate is chosen by the OS (matching the hardware
/// device); use [`CpalAudioSource::sample_rate`] to inspect it.
///
/// ```ignore
/// let audio = CpalAudio::new(16_000)?;
/// let stream = audio.source().stream();
/// audio.sink().play(chunk).await?;
/// audio.sink().stop().await?;
/// ```
pub struct CpalAudio {
    input: CpalAudioSource,
    output: CpalAudioSink,
}

impl CpalAudio {
    /// Open the default input and output devices at `sample_rate`.
    pub fn new(sample_rate: u32) -> Result<Self, MindroidError> {
        let input = CpalAudioSource::new(sample_rate)?;
        let output = CpalAudioSink::new(sample_rate)?;
        Ok(Self { input, output })
    }

    /// Return a reference to the microphone source.
    pub fn source(&self) -> &CpalAudioSource {
        &self.input
    }

    /// Return a reference to the speaker sink.
    pub fn sink(&self) -> &CpalAudioSink {
        &self.output
    }
}

// ─── cpal helpers ─────────────────────────────────────────────────────────────

/// Try to open an input config at the requested `sample_rate`.
/// Falls back to the device default if the rate is not supported.
fn preferred_input_config_at(
    device: &cpal::Device,
    sample_rate: u32,
) -> Result<cpal::SupportedStreamConfig, MindroidError> {
    if let Ok(mut configs) = device.supported_input_configs()
        && let Some(range) = configs
            .find(|c| c.min_sample_rate().0 <= sample_rate && c.max_sample_rate().0 >= sample_rate)
    {
        return Ok(range.with_sample_rate(cpal::SampleRate(sample_rate)));
    }
    device
        .default_input_config()
        .map_err(|e| MindroidError::Transport {
            message: format!("CpalAudioSource: no supported input config: {e}"),
            source: None,
        })
}

/// Build a typed cpal input stream that mono-downmixes and forwards `chunk_size`-
/// aligned `Vec<f32>` frames to `sender`.
fn build_source_stream<T, N, E>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sender: std::sync::mpsc::SyncSender<Vec<f32>>,
    channels: usize,
    err_fn: E,
    normalize: N,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: SizedSample + Send + 'static,
    N: Fn(T) -> f32 + Send + 'static,
    E: FnMut(cpal::StreamError) + Send + 'static,
{
    // Accumulate ≥ 512 samples before flushing so consumers get reasonably-
    // sized chunks, but flush the remainder when the buffer is non-empty.
    const CHUNK: usize = 512;
    let mut acc: Vec<f32> = Vec::new();
    device.build_input_stream(
        config,
        move |data: &[T], _| {
            for frame in data.chunks(channels) {
                let mono = frame.iter().map(|&s| normalize(s)).sum::<f32>() / channels as f32;
                acc.push(mono);
            }
            while acc.len() >= CHUNK {
                let chunk: Vec<f32> = acc.drain(..CHUNK).collect();
                let _ = sender.try_send(chunk);
            }
        },
        err_fn,
        None,
    )
}
