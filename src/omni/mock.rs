//! Reusable mock implementations for omni module testing.
//!
//! These mocks are intended for use in downstream integration tests and
//! third-party crate tests — not just internal unit tests. Import them with:
//!
//! ```rust,ignore
//! use mindroid::omni::mock::{MockOmniProvider, MockAudioSource, MockAudioSink};
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use mindroid::omni::mock::MockOmniProvider;
//! use mindroid::omni::types::OmniEvent;
//!
//! let (mut provider, tx) = MockOmniProvider::new();
//! tx.send(OmniEvent::TurnComplete).await.unwrap();
//! drop(tx);
//! // ... build and run an OmniSession with `provider` ...
//! ```

use crate::core::error::MindroidError;
use crate::omni::audio::{AudioSink, AudioSource};
use crate::omni::provider::OmniProvider;
use crate::omni::types::{AudioChunk, OmniConfig, OmniEvent};
use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;
use serde_json::Value;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

// ── MockOmniProvider ──────────────────────────────────────────────────────────

/// A reusable [`OmniProvider`] mock suitable for integration tests.
///
/// - Records every `send_audio`, `send_text`, and `send_tool_result` call.
/// - Events are driven by the [`mpsc::Sender`] returned from [`MockOmniProvider::new`].
/// - Tracks whether `disconnect` was called via [`MockOmniProvider::was_disconnected`].
///
/// # Usage
///
/// ```rust,ignore
/// let (mut provider, tx) = MockOmniProvider::new();
/// let received_audio = provider.received_audio();
///
/// tx.send(OmniEvent::TurnComplete).await.unwrap();
/// drop(tx); // closes the event stream
///
/// // Build and run your session...
///
/// assert_eq!(received_audio.lock().unwrap().len(), 0);
/// ```
pub struct MockOmniProvider {
    /// Receiver end — consumed by the first call to `events()`.
    events_rx: Option<mpsc::Receiver<OmniEvent>>,
    /// Audio chunks recorded by `send_audio`.
    pub received_audio: Arc<Mutex<Vec<AudioChunk>>>,
    /// Text strings recorded by `send_text`.
    pub received_text: Arc<Mutex<Vec<String>>>,
    /// `(call_id, result)` pairs recorded by `send_tool_result`.
    pub received_tool_results: Arc<Mutex<Vec<(String, Value)>>>,
    /// Set to `true` when `disconnect` is called.
    disconnected: Arc<Mutex<bool>>,
}

impl MockOmniProvider {
    /// Creates a new mock provider and its event-injection sender.
    ///
    /// The caller should push [`OmniEvent`]s through the returned [`mpsc::Sender`]
    /// and then `drop` it to signal end-of-stream to the session loop.
    pub fn new() -> (Self, mpsc::Sender<OmniEvent>) {
        let (tx, rx) = mpsc::channel::<OmniEvent>(64);
        let provider = Self {
            events_rx: Some(rx),
            received_audio: Arc::new(Mutex::new(Vec::new())),
            received_text: Arc::new(Mutex::new(Vec::new())),
            received_tool_results: Arc::new(Mutex::new(Vec::new())),
            disconnected: Arc::new(Mutex::new(false)),
        };
        (provider, tx)
    }

    /// Returns a clone of the shared audio-chunk recording.
    pub fn received_audio(&self) -> Arc<Mutex<Vec<AudioChunk>>> {
        Arc::clone(&self.received_audio)
    }

    /// Returns a clone of the shared text recording.
    pub fn received_text(&self) -> Arc<Mutex<Vec<String>>> {
        Arc::clone(&self.received_text)
    }

    /// Returns a clone of the shared tool-result recording.
    pub fn received_tool_results(&self) -> Arc<Mutex<Vec<(String, Value)>>> {
        Arc::clone(&self.received_tool_results)
    }

    /// Returns a clone of the disconnection flag.
    pub fn disconnected_flag(&self) -> Arc<Mutex<bool>> {
        Arc::clone(&self.disconnected)
    }

    /// Returns `true` if `disconnect()` has been called.
    pub fn was_disconnected(&self) -> bool {
        *self.disconnected.lock().unwrap()
    }
}

impl Default for MockOmniProvider {
    fn default() -> Self {
        Self::new().0
    }
}

#[async_trait]
impl OmniProvider for MockOmniProvider {
    async fn connect(&mut self, _config: &OmniConfig) -> Result<(), MindroidError> {
        Ok(())
    }

    async fn send_audio(&self, chunk: AudioChunk) -> Result<(), MindroidError> {
        self.received_audio.lock().unwrap().push(chunk);
        Ok(())
    }

    async fn send_text(&self, text: &str) -> Result<(), MindroidError> {
        self.received_text.lock().unwrap().push(text.to_string());
        Ok(())
    }

    async fn send_tool_result(&self, call_id: &str, result: Value) -> Result<(), MindroidError> {
        self.received_tool_results
            .lock()
            .unwrap()
            .push((call_id.to_string(), result));
        Ok(())
    }

    async fn end_audio_stream(&self) -> Result<(), MindroidError> {
        Ok(())
    }

    fn events(&mut self) -> Pin<Box<dyn Stream<Item = OmniEvent> + Send>> {
        let rx = self
            .events_rx
            .take()
            .expect("MockOmniProvider::events() must only be called once");
        Box::pin(ReceiverStream::new(rx))
    }

    async fn disconnect(&mut self) -> Result<(), MindroidError> {
        *self.disconnected.lock().unwrap() = true;
        Ok(())
    }
}

// ── MockAudioSource ───────────────────────────────────────────────────────────

/// A reusable [`AudioSource`] mock that yields a fixed sequence of [`AudioChunk`]s.
///
/// # Example
///
/// ```rust,ignore
/// use mindroid::omni::mock::MockAudioSource;
/// use mindroid::omni::types::AudioChunk;
///
/// let source = MockAudioSource::new(vec![
///     AudioChunk { data: vec![1, 2], sample_rate: 16_000, channels: 1, bits_per_sample: 16 },
/// ]);
/// ```
pub struct MockAudioSource {
    chunks: Vec<AudioChunk>,
    sample_rate: u32,
}

impl MockAudioSource {
    /// Creates a source that yields `chunks` in order, then ends.
    pub fn new(chunks: Vec<AudioChunk>) -> Self {
        Self {
            chunks,
            sample_rate: 16_000,
        }
    }

    /// Creates a source with a custom sample rate.
    pub fn with_sample_rate(chunks: Vec<AudioChunk>, sample_rate: u32) -> Self {
        Self {
            chunks,
            sample_rate,
        }
    }

    /// Convenience constructor for a silent (empty) source.
    pub fn empty() -> Self {
        Self::new(vec![])
    }
}

impl AudioSource for MockAudioSource {
    fn stream(&self) -> Pin<Box<dyn Stream<Item = AudioChunk> + Send + '_>> {
        let chunks = self.chunks.clone();
        Box::pin(stream! {
            for chunk in chunks {
                yield chunk;
            }
        })
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

// ── MockAudioSink ─────────────────────────────────────────────────────────────

/// A reusable [`AudioSink`] mock that records every method call.
///
/// Each call appends a string to the internal `calls` log:
/// - `play(chunk)` → `"play:<first_byte>"` (e.g. `"play:42"`)
/// - `flush()` → `"flush"`
/// - `stop()` → `"stop"`
///
/// # Example
///
/// ```rust,ignore
/// use mindroid::omni::mock::MockAudioSink;
///
/// let sink = MockAudioSink::new();
/// let calls = sink.calls();
///
/// // ... run session with sink ...
///
/// assert!(calls.lock().unwrap().contains(&"flush".to_string()));
/// ```
pub struct MockAudioSink {
    /// Ordered log of method calls.
    pub calls: Arc<Mutex<Vec<String>>>,
    sample_rate: u32,
}

impl MockAudioSink {
    /// Creates a new sink recording at 16 kHz.
    pub fn new() -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
            sample_rate: 16_000,
        }
    }

    /// Creates a new sink recording at the given sample rate.
    pub fn with_sample_rate(sample_rate: u32) -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
            sample_rate,
        }
    }

    /// Returns a clone of the shared call-log handle.
    pub fn calls(&self) -> Arc<Mutex<Vec<String>>> {
        Arc::clone(&self.calls)
    }

    /// Returns `true` if `flush` appears in the call log.
    pub fn was_flushed(&self) -> bool {
        self.calls.lock().unwrap().contains(&"flush".to_string())
    }

    /// Returns `true` if `stop` appears in the call log.
    pub fn was_stopped(&self) -> bool {
        self.calls.lock().unwrap().contains(&"stop".to_string())
    }

    /// Returns the number of `play` calls recorded.
    pub fn play_count(&self) -> usize {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.starts_with("play:"))
            .count()
    }
}

impl Default for MockAudioSink {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AudioSink for MockAudioSink {
    async fn play(&self, chunk: AudioChunk) -> Result<(), MindroidError> {
        let label = chunk.data.first().copied().unwrap_or(0);
        self.calls.lock().unwrap().push(format!("play:{label}"));
        Ok(())
    }

    async fn flush(&self) -> Result<(), MindroidError> {
        self.calls.lock().unwrap().push("flush".to_string());
        Ok(())
    }

    async fn stop(&self) -> Result<(), MindroidError> {
        self.calls.lock().unwrap().push("stop".to_string());
        Ok(())
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use serde_json::json;

    fn make_chunk(n: u8) -> AudioChunk {
        AudioChunk {
            data: vec![n],
            sample_rate: 16_000,
            channels: 1,
            bits_per_sample: 16,
        }
    }

    // ── MockOmniProvider ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn mock_provider_records_audio() {
        let (mut p, _tx) = MockOmniProvider::new();
        p.connect(&OmniConfig::default()).await.unwrap();

        p.send_audio(make_chunk(1)).await.unwrap();
        p.send_audio(make_chunk(2)).await.unwrap();

        let chunks = p.received_audio.lock().unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].data, vec![1]);
        assert_eq!(chunks[1].data, vec![2]);
    }

    #[tokio::test]
    async fn mock_provider_records_text() {
        let (mut p, _tx) = MockOmniProvider::new();
        p.connect(&OmniConfig::default()).await.unwrap();

        p.send_text("hello").await.unwrap();
        p.send_text("world").await.unwrap();

        let texts = p.received_text.lock().unwrap();
        assert_eq!(texts.as_slice(), &["hello", "world"]);
    }

    #[tokio::test]
    async fn mock_provider_records_tool_results() {
        let (mut p, _tx) = MockOmniProvider::new();
        p.connect(&OmniConfig::default()).await.unwrap();

        p.send_tool_result("call-1", json!({"temp": 21}))
            .await
            .unwrap();
        p.send_tool_result("call-2", json!("ok")).await.unwrap();

        let results = p.received_tool_results.lock().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "call-1");
        assert_eq!(results[0].1, json!({"temp": 21}));
        assert_eq!(results[1].0, "call-2");
        assert_eq!(results[1].1, json!("ok"));
    }

    #[tokio::test]
    async fn mock_provider_event_stream() {
        let (mut p, tx) = MockOmniProvider::new();
        let mut stream = p.events();

        tx.send(OmniEvent::TurnComplete).await.unwrap();
        tx.send(OmniEvent::Interrupted).await.unwrap();
        drop(tx);

        let e1 = stream.next().await.unwrap();
        assert!(matches!(e1, OmniEvent::TurnComplete));

        let e2 = stream.next().await.unwrap();
        assert!(matches!(e2, OmniEvent::Interrupted));

        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn mock_provider_disconnect_flag() {
        let (mut p, _tx) = MockOmniProvider::new();
        assert!(!p.was_disconnected());
        p.disconnect().await.unwrap();
        assert!(p.was_disconnected());
    }

    #[test]
    fn mock_provider_accessor_helpers() {
        let (p, _tx) = MockOmniProvider::new();
        // All accessors return Arc handles to the same data
        let _audio = p.received_audio();
        let _text = p.received_text();
        let _results = p.received_tool_results();
        let _flag = p.disconnected_flag();
    }

    // ── MockAudioSource ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn mock_audio_source_yields_chunks() {
        let src = MockAudioSource::new(vec![make_chunk(10), make_chunk(20), make_chunk(30)]);
        assert_eq!(src.sample_rate(), 16_000);

        let collected: Vec<AudioChunk> = src.stream().collect().await;
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].data, vec![10]);
        assert_eq!(collected[1].data, vec![20]);
        assert_eq!(collected[2].data, vec![30]);
    }

    #[tokio::test]
    async fn mock_audio_source_empty() {
        let src = MockAudioSource::empty();
        let collected: Vec<AudioChunk> = src.stream().collect().await;
        assert!(collected.is_empty());
    }

    #[test]
    fn mock_audio_source_custom_sample_rate() {
        let src = MockAudioSource::with_sample_rate(vec![], 44_100);
        assert_eq!(src.sample_rate(), 44_100);
    }

    // ── MockAudioSink ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn mock_audio_sink_records_calls() {
        let sink = MockAudioSink::new();
        assert_eq!(sink.sample_rate(), 16_000);

        sink.play(make_chunk(5)).await.unwrap();
        sink.play(make_chunk(6)).await.unwrap();
        sink.flush().await.unwrap();
        sink.stop().await.unwrap();

        let calls = sink.calls.lock().unwrap().clone();
        assert_eq!(calls, vec!["play:5", "play:6", "flush", "stop"]);
    }

    #[tokio::test]
    async fn mock_audio_sink_helper_methods() {
        let sink = MockAudioSink::new();
        assert!(!sink.was_flushed());
        assert!(!sink.was_stopped());
        assert_eq!(sink.play_count(), 0);

        sink.play(make_chunk(1)).await.unwrap();
        sink.play(make_chunk(2)).await.unwrap();
        sink.flush().await.unwrap();

        assert!(sink.was_flushed());
        assert!(!sink.was_stopped());
        assert_eq!(sink.play_count(), 2);

        sink.stop().await.unwrap();
        assert!(sink.was_stopped());
    }

    #[test]
    fn mock_audio_sink_custom_rate() {
        let sink = MockAudioSink::with_sample_rate(48_000);
        assert_eq!(sink.sample_rate(), 48_000);
    }
}
