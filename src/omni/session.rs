use crate::core::config::AgentConfig;
use crate::core::error::MindroidError;
use crate::omni::audio::{AudioSink, AudioSource};
use crate::omni::provider::OmniProvider;
use crate::omni::types::{
    AudioChunk, BargeInMode, OmniConfig, OmniEvent, SessionState, TurnDetection,
};
use crate::tools::Tool;
use futures::StreamExt;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

// Local-VAD imports — only compiled when the Silero ONNX feature is present.
#[cfg(feature = "transport-audio")]
use {
    crate::omni::vad::VadInference,
    crate::voice::frontend::{AudioFrontend, FrontendEvent},
    crate::voice::types::VadConfig,
    std::time::Instant,
};

/// An omnimodal session that connects an [`OmniProvider`] with optional audio
/// I/O and tools, managing session lifecycle state.
pub struct OmniSession {
    provider: Box<dyn OmniProvider>,
    audio_source: Option<Arc<dyn AudioSource>>,
    audio_sink: Option<Arc<dyn AudioSink>>,
    tools: Vec<Arc<dyn Tool>>,
    config: OmniConfig,
    // Reserved for future use: persona prompts, identity resolution, etc.
    #[allow(dead_code)]
    agent_config: Arc<AgentConfig>,
    state: SessionState,
    cancel: CancellationToken,
}

impl OmniSession {
    /// Create a new [`OmniSessionBuilder`].
    pub fn builder() -> OmniSessionBuilder {
        OmniSessionBuilder::new()
    }

    /// Current session state.
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Connect to the provider and run the main event loop until cancelled,
    /// the provider stream ends, or a fatal error occurs.
    pub async fn run(&mut self) -> Result<(), MindroidError> {
        // 1. Connect
        self.state = SessionState::Connecting;
        self.provider.connect(&self.config).await?;
        self.state = SessionState::Listening;

        // 2. Take the events stream (owned — no lifetime tie to provider)
        let mut provider_events = self.provider.events();

        // 3. Optional audio source stream
        let mut audio_stream = self.audio_source.as_ref().map(|s| s.stream());

        // 4. Determine whether local VAD paths are active.
        let has_local_turn_detection =
            matches!(&self.config.turn_detection, TurnDetection::Local(_));
        let has_local_barge_in = matches!(&self.config.barge_in, BargeInMode::LocalVad);
        // Only read under `transport-audio`, like the VAD channels below.
        #[cfg_attr(not(feature = "transport-audio"), allow(unused_variables))]
        let use_local_vad = has_local_turn_detection || has_local_barge_in;

        // 5. Set up the VAD inference offload.
        //
        //    The channel types are always declared so the select! arm compiles
        //    regardless of features.  The channels are populated only when the
        //    `transport-audio` feature (and hence `VadInference`) is available.
        //
        //    `vad_out_rx`: receives `(f32_samples, probability)` back from the
        //    blocking worker.
        //    `vad_in_tx`: sends raw `AudioChunk`s to the worker.
        let mut vad_out_rx: Option<mpsc::Receiver<(Vec<f32>, f32)>> = None;
        let mut vad_in_tx: Option<mpsc::Sender<AudioChunk>> = None;

        #[cfg(feature = "transport-audio")]
        if use_local_vad {
            let sample_rate = self
                .audio_source
                .as_ref()
                .map(|s| s.sample_rate())
                .unwrap_or(16_000);

            // 32 ms frame — Silero's preferred stride.
            let chunk_size = (sample_rate as u64 * 32 / 1_000) as usize;

            let (in_tx, in_rx) = mpsc::channel::<AudioChunk>(8);
            let (out_tx, out_rx) = mpsc::channel::<(Vec<f32>, f32)>(8);

            match VadInference::new(sample_rate, chunk_size) {
                Ok(mut vad) => {
                    // Hand the inference to a blocking thread; it owns `in_rx` and
                    // `out_tx` for its lifetime.
                    tokio::task::spawn_blocking(move || {
                        let mut in_rx = in_rx;
                        while let Some(chunk) = in_rx.blocking_recv() {
                            // Raw bytes → i16 (little-endian, as produced by CPAL).
                            let i16_samples: Vec<i16> = chunk
                                .data
                                .chunks_exact(2)
                                .map(|b| i16::from_le_bytes([b[0], b[1]]))
                                .collect();
                            // f32 copy for AudioFrontend::process.
                            let f32_samples: Vec<f32> = i16_samples
                                .iter()
                                .map(|&s| s as f32 / i16::MAX as f32)
                                .collect();
                            let probability = vad.predict(&i16_samples);
                            // If the async side is gone, the send silently fails.
                            let _ = out_tx.blocking_send((f32_samples, probability));
                        }
                    });
                    vad_in_tx = Some(in_tx);
                    vad_out_rx = Some(out_rx);
                }
                Err(_) => {
                    // VadInference construction failed — fall back to provider-only.
                }
            }
        }

        // 6. Build AudioFrontend when local VAD channels are live.
        //    Declared always so the select! body can refer to it unconditionally;
        //    actual construction is feature-gated.
        #[cfg(feature = "transport-audio")]
        let mut audio_frontend: Option<AudioFrontend> = if vad_in_tx.is_some() {
            let sample_rate = self
                .audio_source
                .as_ref()
                .map(|s| s.sample_rate())
                .unwrap_or(16_000);
            let vad_config = match &self.config.turn_detection {
                TurnDetection::Local(cfg) => cfg.clone(),
                _ => VadConfig::default(),
            };
            Some(
                AudioFrontend::builder(vad_config, 32)
                    .sample_rate_hz(sample_rate)
                    .barge_in_mode(self.config.barge_in.clone())
                    .turn_detection(self.config.turn_detection.clone())
                    .build(),
            )
        } else {
            None
        };

        // 7. The select! loop.
        //
        //    The VAD results arm is always present syntactically (tokio::select!
        //    does not support #[cfg] on arms).  When `vad_out_rx` is None the
        //    arm resolves to `std::future::pending()` and never fires.
        loop {
            tokio::select! {
                biased;

                // Highest priority: cancellation.
                _ = self.cancel.cancelled() => {
                    self.state = SessionState::Closed;
                    if let Some(ref sink) = self.audio_sink {
                        let _ = sink.stop().await;
                    }
                    let _ = self.provider.disconnect().await;
                    break;
                }

                // VAD inference results.  When `vad_out_rx` is None this arm
                // immediately resolves to `pending()` and is never selected.
                vad_result = async {
                    match vad_out_rx.as_mut() {
                        Some(rx) => rx.recv().await,
                        None => std::future::pending::<Option<(Vec<f32>, f32)>>().await,
                    }
                } => {
                    // All processing in here is feature-gated because `AudioFrontend`
                    // and `FrontendEvent` only exist with `transport-audio`.
                    #[cfg(feature = "transport-audio")]
                    {
                        if let Some((f32_samples, probability)) = vad_result {
                            if let Some(ref mut fe) = audio_frontend {
                                let agent_speaking = self.state == SessionState::Speaking;
                                let now = Instant::now();
                                let events =
                                    fe.process(&f32_samples, probability, agent_speaking, now);
                                for event in events {
                                    match event {
                                        FrontendEvent::BargeIn { .. } => {
                                            // Local barge-in: stop the sink immediately and
                                            // return to Listening.  Provider-side history
                                            // truncation is deferred (not in scope here).
                                            if let Some(ref sink) = self.audio_sink {
                                                sink.stop().await?;
                                            }
                                            self.state = SessionState::Listening;
                                        }
                                        FrontendEvent::UtteranceComplete { .. } => {
                                            // Local turn detection: tell the provider that
                                            // the user's turn is complete so it can start
                                            // generating a response.
                                            self.provider.end_audio_stream().await?;
                                        }
                                        FrontendEvent::SpeechStarted => {
                                            // Speech onset — no session-level action needed;
                                            // the frontend tracks state internally.
                                        }
                                    }
                                }
                            }
                        } else {
                            // Worker exited — disable the channel so we always pend.
                            vad_out_rx = None;
                        }
                    }
                    // When the feature is absent the arm type-checks as `Option<…>`
                    // but the body is compiled away; nothing happens.
                    #[cfg(not(feature = "transport-audio"))]
                    let _ = vad_result;
                }

                // Audio input forwarding.
                chunk = async {
                    match audio_stream.as_mut() {
                        Some(s) => s.next().await,
                        None => std::future::pending().await,
                    }
                } => {
                    if let Some(chunk) = chunk {
                        // Always forward to provider.
                        self.provider.send_audio(chunk.clone()).await?;

                        // When local VAD is active, also send a copy to the inference
                        // worker.  `try_send` — skip the frame if the inbox is full
                        // rather than blocking the audio loop.
                        if let Some(ref tx) = vad_in_tx {
                            let _ = tx.try_send(chunk);
                        }
                    } else {
                        // Audio stream ended.
                        audio_stream = None;
                        if !has_local_turn_detection {
                            // Provider-side turn detection: signal end immediately.
                            self.provider.end_audio_stream().await?;
                        } else {
                            // Local turn detection: close the sender so the VAD
                            // worker drains and terminates; end_audio_stream will be
                            // called from the UtteranceComplete event once VAD fires.
                            vad_in_tx = None;
                        }
                    }
                }

                // Provider event handling.
                event = provider_events.next() => {
                    match event {
                        Some(OmniEvent::AudioChunk(chunk)) => {
                            self.state = SessionState::Speaking;
                            if let Some(ref sink) = self.audio_sink {
                                sink.play(chunk).await?;
                            }
                        }
                        Some(OmniEvent::ToolCall { id, name, args }) => {
                            self.state = SessionState::ToolCall;
                            let result = self.execute_tool(&name, args).await;
                            let result_value = match result {
                                Ok(r) => Value::String(r),
                                Err(e) => serde_json::json!({"error": e.to_string()}),
                            };
                            self.provider.send_tool_result(&id, result_value).await?;
                            // Return to Listening after tool call is dispatched.
                            self.state = SessionState::Listening;
                        }
                        Some(OmniEvent::Interrupted) => {
                            if let Some(ref sink) = self.audio_sink {
                                sink.stop().await?;
                            }
                            self.state = SessionState::Listening;
                        }
                        Some(OmniEvent::TurnComplete) => {
                            if let Some(ref sink) = self.audio_sink {
                                sink.flush().await?;
                            }
                            self.state = SessionState::Listening;
                        }
                        Some(OmniEvent::Error(e)) => {
                            self.state = SessionState::Closed;
                            if let Some(ref sink) = self.audio_sink {
                                let _ = sink.stop().await;
                            }
                            // Arc<MindroidError> → MindroidError: unwrap if sole owner,
                            // otherwise re-wrap as a pipeline error preserving the message.
                            let err = Arc::try_unwrap(e)
                                .unwrap_or_else(|arc| MindroidError::pipeline(arc.to_string()));
                            return Err(err);
                        }
                        Some(OmniEvent::Transcript { .. }) => {
                            // Transcript events — log/ignore for now.
                            // TODO: expose transcripts via a callback/channel.
                        }
                        None => {
                            // Provider stream exhausted — clean exit.
                            self.state = SessionState::Closed;
                            break;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Find a tool by name and invoke it with the given arguments.
    async fn execute_tool(&self, name: &str, args: Value) -> Result<String, MindroidError> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == name)
            .ok_or_else(|| MindroidError::pipeline(format!("tool not found: {name}")))?;
        tool.execute(args, &crate::tools::ToolContext::default())
            .await
    }

    /// Test-only entry point that injects a pre-populated VAD results channel,
    /// bypassing the `VadInference` worker.  Used to verify that the run loop
    /// correctly handles `FrontendEvent::BargeIn` and `FrontendEvent::UtteranceComplete`
    /// without requiring the `transport-audio` Silero feature at test time.
    ///
    /// The `vad_rx` channel should send `(f32_samples, probability)` pairs and
    /// then be closed (dropped) to allow `run()` to complete naturally.
    #[cfg(all(test, feature = "transport-audio"))]
    pub(crate) async fn run_with_vad_rx(
        &mut self,
        vad_rx: mpsc::Receiver<(Vec<f32>, f32)>,
    ) -> Result<(), MindroidError> {
        use crate::voice::frontend::AudioFrontend;
        use crate::voice::types::VadConfig;

        self.state = SessionState::Connecting;
        self.provider.connect(&self.config).await?;
        self.state = SessionState::Listening;

        let mut provider_events = self.provider.events();
        let mut audio_stream = self.audio_source.as_ref().map(|s| s.stream());

        let has_local_turn_detection =
            matches!(&self.config.turn_detection, TurnDetection::Local(_));
        let has_local_barge_in = matches!(&self.config.barge_in, BargeInMode::LocalVad);

        let mut vad_out_rx: Option<mpsc::Receiver<(Vec<f32>, f32)>> = Some(vad_rx);

        // Build an AudioFrontend for the injected VAD path.
        let sample_rate = self
            .audio_source
            .as_ref()
            .map(|s| s.sample_rate())
            .unwrap_or(16_000);
        let vad_config = match &self.config.turn_detection {
            TurnDetection::Local(cfg) => cfg.clone(),
            _ => VadConfig::default(),
        };
        let mut audio_frontend: Option<AudioFrontend> =
            if has_local_turn_detection || has_local_barge_in {
                Some(
                    AudioFrontend::builder(vad_config, 32)
                        .sample_rate_hz(sample_rate)
                        .barge_in_mode(self.config.barge_in.clone())
                        .turn_detection(self.config.turn_detection.clone())
                        .build(),
                )
            } else {
                None
            };

        loop {
            tokio::select! {
                biased;

                _ = self.cancel.cancelled() => {
                    self.state = SessionState::Closed;
                    if let Some(ref sink) = self.audio_sink {
                        let _ = sink.stop().await;
                    }
                    let _ = self.provider.disconnect().await;
                    break;
                }

                vad_result = async {
                    match vad_out_rx.as_mut() {
                        Some(rx) => rx.recv().await,
                        None => std::future::pending::<Option<(Vec<f32>, f32)>>().await,
                    }
                } => {
                    if let Some((f32_samples, probability)) = vad_result {
                        if let Some(ref mut fe) = audio_frontend {
                            let agent_speaking = self.state == SessionState::Speaking;
                            let now = std::time::Instant::now();
                            let events =
                                fe.process(&f32_samples, probability, agent_speaking, now);
                            for event in events {
                                match event {
                                    FrontendEvent::BargeIn { .. } => {
                                        if let Some(ref sink) = self.audio_sink {
                                            sink.stop().await?;
                                        }
                                        self.state = SessionState::Listening;
                                    }
                                    FrontendEvent::UtteranceComplete { .. } => {
                                        self.provider.end_audio_stream().await?;
                                    }
                                    FrontendEvent::SpeechStarted => {}
                                }
                            }
                        }
                    } else {
                        vad_out_rx = None;
                    }
                }

                chunk = async {
                    match audio_stream.as_mut() {
                        Some(s) => s.next().await,
                        None => std::future::pending().await,
                    }
                } => {
                    if let Some(chunk) = chunk {
                        self.provider.send_audio(chunk).await?;
                    } else {
                        audio_stream = None;
                        if !has_local_turn_detection {
                            self.provider.end_audio_stream().await?;
                        }
                    }
                }

                event = provider_events.next() => {
                    match event {
                        Some(OmniEvent::AudioChunk(chunk)) => {
                            self.state = SessionState::Speaking;
                            if let Some(ref sink) = self.audio_sink {
                                sink.play(chunk).await?;
                            }
                        }
                        Some(OmniEvent::ToolCall { id, name, args }) => {
                            self.state = SessionState::ToolCall;
                            let result = self.execute_tool(&name, args).await;
                            let result_value = match result {
                                Ok(r) => Value::String(r),
                                Err(e) => serde_json::json!({"error": e.to_string()}),
                            };
                            self.provider.send_tool_result(&id, result_value).await?;
                            self.state = SessionState::Listening;
                        }
                        Some(OmniEvent::Interrupted) => {
                            if let Some(ref sink) = self.audio_sink {
                                sink.stop().await?;
                            }
                            self.state = SessionState::Listening;
                        }
                        Some(OmniEvent::TurnComplete) => {
                            if let Some(ref sink) = self.audio_sink {
                                sink.flush().await?;
                            }
                            self.state = SessionState::Listening;
                        }
                        Some(OmniEvent::Error(e)) => {
                            self.state = SessionState::Closed;
                            if let Some(ref sink) = self.audio_sink {
                                let _ = sink.stop().await;
                            }
                            let err = Arc::try_unwrap(e)
                                .unwrap_or_else(|arc| MindroidError::pipeline(arc.to_string()));
                            return Err(err);
                        }
                        Some(OmniEvent::Transcript { .. }) => {}
                        None => {
                            self.state = SessionState::Closed;
                            break;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Fluent builder for [`OmniSession`].
pub struct OmniSessionBuilder {
    provider: Option<Box<dyn OmniProvider>>,
    audio_source: Option<Arc<dyn AudioSource>>,
    audio_sink: Option<Arc<dyn AudioSink>>,
    tools: Vec<Arc<dyn Tool>>,
    config: OmniConfig,
    agent_config: Option<Arc<AgentConfig>>,
    cancel: Option<CancellationToken>,
}

impl OmniSessionBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        Self {
            provider: None,
            audio_source: None,
            audio_sink: None,
            tools: Vec::new(),
            config: OmniConfig::default(),
            agent_config: None,
            cancel: None,
        }
    }

    /// Set the omni provider (required).
    pub fn provider(mut self, p: impl OmniProvider + 'static) -> Self {
        self.provider = Some(Box::new(p));
        self
    }

    /// Set the audio source (optional — omit for server/text-only mode).
    pub fn audio_source(mut self, s: impl AudioSource + 'static) -> Self {
        self.audio_source = Some(Arc::new(s));
        self
    }

    /// Set the audio sink (optional — omit for server/text-only mode).
    pub fn audio_sink(mut self, s: impl AudioSink + 'static) -> Self {
        self.audio_sink = Some(Arc::new(s));
        self
    }

    /// Add a single tool.
    pub fn tool(mut self, t: impl Tool + 'static) -> Self {
        self.tools.push(Arc::new(t));
        self
    }

    /// Replace the tool list wholesale.
    pub fn tools(mut self, t: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = t;
        self
    }

    /// Set the full [`OmniConfig`].
    pub fn config(mut self, c: OmniConfig) -> Self {
        self.config = c;
        self
    }

    /// Set the agent config.
    pub fn agent_config(mut self, c: Arc<AgentConfig>) -> Self {
        self.agent_config = Some(c);
        self
    }

    /// Override turn detection on the current config.
    pub fn turn_detection(mut self, td: TurnDetection) -> Self {
        self.config.turn_detection = td;
        self
    }

    /// Override barge-in mode on the current config.
    pub fn barge_in(mut self, bi: BargeInMode) -> Self {
        self.config.barge_in = bi;
        self
    }

    /// Set the cancellation token. If not provided, a fresh token is created.
    pub fn cancel_token(mut self, ct: CancellationToken) -> Self {
        self.cancel = Some(ct);
        self
    }

    /// Build the [`OmniSession`].
    ///
    /// # Errors
    ///
    /// Returns [`MindroidError::Config`] if no provider has been set.
    pub fn build(self) -> Result<OmniSession, MindroidError> {
        let provider = self
            .provider
            .ok_or_else(|| MindroidError::config("OmniSession requires a provider"))?;
        let agent_config = self
            .agent_config
            .unwrap_or_else(|| Arc::new(AgentConfig::default()));
        Ok(OmniSession {
            provider,
            audio_source: self.audio_source,
            audio_sink: self.audio_sink,
            tools: self.tools,
            config: self.config,
            agent_config,
            state: SessionState::Connecting,
            cancel: self.cancel.unwrap_or_default(),
        })
    }
}

impl Default for OmniSessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::omni::types::{AudioChunk, OmniEvent};
    use async_stream::stream;
    use async_trait::async_trait;
    use futures::Stream;
    use serde_json::Value;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    // ── Helper ────────────────────────────────────────────────────────────────

    fn make_chunk(n: u8) -> AudioChunk {
        AudioChunk {
            data: vec![n],
            sample_rate: 16_000,
            channels: 1,
            bits_per_sample: 16,
        }
    }

    // ── Recording mock provider ───────────────────────────────────────────────
    // Tracks send_audio calls and send_tool_result calls; events are driven via
    // an mpsc channel the test owns.

    struct RecordingProvider {
        pub recorded_chunks: Arc<Mutex<Vec<AudioChunk>>>,
        pub recorded_tool_results: Arc<Mutex<Vec<(String, Value)>>>,
        pub disconnected: Arc<Mutex<bool>>,
        event_rx: Option<mpsc::Receiver<OmniEvent>>,
    }

    impl RecordingProvider {
        fn new() -> (Self, mpsc::Sender<OmniEvent>) {
            let (tx, rx) = mpsc::channel::<OmniEvent>(64);
            let p = RecordingProvider {
                recorded_chunks: Arc::new(Mutex::new(Vec::new())),
                recorded_tool_results: Arc::new(Mutex::new(Vec::new())),
                disconnected: Arc::new(Mutex::new(false)),
                event_rx: Some(rx),
            };
            (p, tx)
        }
    }

    #[async_trait]
    impl OmniProvider for RecordingProvider {
        async fn connect(&mut self, _config: &OmniConfig) -> Result<(), MindroidError> {
            Ok(())
        }

        async fn send_audio(&self, chunk: AudioChunk) -> Result<(), MindroidError> {
            self.recorded_chunks.lock().unwrap().push(chunk);
            Ok(())
        }

        async fn send_text(&self, _text: &str) -> Result<(), MindroidError> {
            Ok(())
        }

        async fn send_tool_result(
            &self,
            call_id: &str,
            result: Value,
        ) -> Result<(), MindroidError> {
            self.recorded_tool_results
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
                .event_rx
                .take()
                .expect("events() called more than once");
            Box::pin(ReceiverStream::new(rx))
        }

        async fn disconnect(&mut self) -> Result<(), MindroidError> {
            *self.disconnected.lock().unwrap() = true;
            Ok(())
        }
    }

    // ── Minimal no-op mock provider (for builder tests) ───────────────────────

    struct NoOpProvider;

    #[async_trait]
    impl OmniProvider for NoOpProvider {
        async fn connect(&mut self, _config: &OmniConfig) -> Result<(), MindroidError> {
            Ok(())
        }

        async fn send_audio(&self, _chunk: AudioChunk) -> Result<(), MindroidError> {
            Ok(())
        }

        async fn send_text(&self, _text: &str) -> Result<(), MindroidError> {
            Ok(())
        }

        async fn send_tool_result(
            &self,
            _call_id: &str,
            _result: Value,
        ) -> Result<(), MindroidError> {
            Ok(())
        }

        async fn end_audio_stream(&self) -> Result<(), MindroidError> {
            Ok(())
        }

        fn events(&mut self) -> Pin<Box<dyn Stream<Item = OmniEvent> + Send>> {
            Box::pin(futures::stream::empty())
        }

        async fn disconnect(&mut self) -> Result<(), MindroidError> {
            Ok(())
        }
    }

    // ── Recording audio sink ──────────────────────────────────────────────────

    struct RecordingSink {
        pub calls: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingSink {
        fn new() -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl AudioSink for RecordingSink {
        async fn play(&self, chunk: AudioChunk) -> Result<(), MindroidError> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("play:{}", chunk.data[0]));
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
            16_000
        }
    }

    // ── Recording audio source ────────────────────────────────────────────────

    struct FixedAudioSource {
        chunks: Vec<AudioChunk>,
    }

    impl AudioSource for FixedAudioSource {
        fn stream(&self) -> Pin<Box<dyn Stream<Item = AudioChunk> + Send + '_>> {
            let chunks = self.chunks.clone();
            Box::pin(stream! {
                for chunk in chunks {
                    yield chunk;
                }
            })
        }

        fn sample_rate(&self) -> u32 {
            16_000
        }
    }

    // ── Minimal no-op mock audio source ──────────────────────────────────────

    struct NoOpAudioSource;

    impl AudioSource for NoOpAudioSource {
        fn stream(&self) -> Pin<Box<dyn Stream<Item = AudioChunk> + Send + '_>> {
            Box::pin(futures::stream::empty())
        }

        fn sample_rate(&self) -> u32 {
            16_000
        }
    }

    // ── Minimal no-op mock audio sink ─────────────────────────────────────────

    struct NoOpSink;

    #[async_trait]
    impl AudioSink for NoOpSink {
        async fn play(&self, _chunk: AudioChunk) -> Result<(), MindroidError> {
            Ok(())
        }

        async fn flush(&self) -> Result<(), MindroidError> {
            Ok(())
        }

        async fn stop(&self) -> Result<(), MindroidError> {
            Ok(())
        }

        fn sample_rate(&self) -> u32 {
            16_000
        }
    }

    // ── Mock tool ─────────────────────────────────────────────────────────────

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echoes arguments back as a string"
        }

        fn parameters_schema(&self) -> Value {
            serde_json::json!({ "type": "object", "properties": {} })
        }

        async fn execute(
            &self,
            args: Value,
            _ctx: &crate::tools::ToolContext,
        ) -> crate::error::Result<String> {
            Ok(args.to_string())
        }
    }

    // ── Builder tests ─────────────────────────────────────────────────────────

    /// Building without a provider must return an error.
    #[test]
    fn test_builder_requires_provider() {
        match OmniSession::builder().build() {
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("provider"),
                    "error message should mention 'provider', got: {msg}"
                );
            }
            Ok(_) => panic!("expected Err when no provider is set, got Ok"),
        }
    }

    /// Building with all options set should succeed.
    #[test]
    fn test_builder_with_all_options() {
        let agent_cfg = Arc::new(AgentConfig {
            agent_id: "test-agent".into(),
            name: "Test".into(),
            ..AgentConfig::default()
        });
        let cancel = CancellationToken::new();

        let result = OmniSession::builder()
            .provider(NoOpProvider)
            .audio_source(NoOpAudioSource)
            .audio_sink(NoOpSink)
            .tool(EchoTool)
            .config(OmniConfig::default())
            .agent_config(agent_cfg)
            .turn_detection(TurnDetection::Manual)
            .barge_in(BargeInMode::Disabled)
            .cancel_token(cancel)
            .build();

        assert!(result.is_ok(), "expected Ok with all options set");
    }

    /// Building without audio (server / text-only mode) must still succeed.
    #[test]
    fn test_builder_audio_optional() {
        let result = OmniSession::builder().provider(NoOpProvider).build();
        assert!(
            result.is_ok(),
            "expected Ok without audio (server mode), got: {:?}",
            result.err()
        );

        let session = result.unwrap();
        assert!(session.audio_source.is_none());
        assert!(session.audio_sink.is_none());
    }

    /// A freshly built session must be in the `Connecting` state.
    #[test]
    fn test_initial_state() {
        let session = OmniSession::builder()
            .provider(NoOpProvider)
            .build()
            .expect("build should succeed");
        assert_eq!(session.state(), SessionState::Connecting);
    }

    /// When no agent_config is supplied, the builder should use `AgentConfig::default()`.
    #[test]
    fn test_default_agent_config() {
        let session = OmniSession::builder()
            .provider(NoOpProvider)
            .build()
            .expect("build should succeed");

        // AgentConfig::default() gives name "Mindroid Agent"
        assert_eq!(session.agent_config.name, "Mindroid Agent");
    }

    // ── run() tests ───────────────────────────────────────────────────────────

    /// Audio chunks from the source are forwarded to the provider via send_audio.
    #[tokio::test]
    async fn test_run_audio_forwarding() {
        let (provider, tx) = RecordingProvider::new();
        let recorded = Arc::clone(&provider.recorded_chunks);

        // Source yields 3 chunks; drop the event sender to end the stream after
        // the audio source is exhausted.
        let source = FixedAudioSource {
            chunks: vec![make_chunk(1), make_chunk(2), make_chunk(3)],
        };

        let mut session = OmniSession::builder()
            .provider(provider)
            .audio_source(source)
            .build()
            .unwrap();

        // Drop the event sender — stream ends immediately, so run() will exit
        // once the audio source AND the provider stream are both done.
        drop(tx);

        session.run().await.expect("run() should succeed");

        let chunks = recorded.lock().unwrap();
        assert_eq!(chunks.len(), 3, "all 3 audio chunks should be forwarded");
        assert_eq!(chunks[0].data, vec![1]);
        assert_eq!(chunks[1].data, vec![2]);
        assert_eq!(chunks[2].data, vec![3]);
    }

    /// A ToolCall event executes the named tool and sends back the result.
    #[tokio::test]
    async fn test_run_tool_execution() {
        let (provider, tx) = RecordingProvider::new();
        let recorded_results = Arc::clone(&provider.recorded_tool_results);

        let mut session = OmniSession::builder()
            .provider(provider)
            .tool(EchoTool)
            .build()
            .unwrap();

        // Push a ToolCall event then close the stream.
        tx.send(OmniEvent::ToolCall {
            id: "call-1".to_string(),
            name: "echo".to_string(),
            args: serde_json::json!({ "msg": "hello" }),
        })
        .await
        .unwrap();
        drop(tx); // ends the provider event stream → run() exits cleanly

        session.run().await.expect("run() should succeed");

        let results = recorded_results.lock().unwrap();
        assert_eq!(results.len(), 1, "one tool result should be sent");
        assert_eq!(results[0].0, "call-1");
        // EchoTool returns args.to_string() wrapped in a Value::String
        assert_eq!(
            results[0].1,
            Value::String(r#"{"msg":"hello"}"#.to_string())
        );
    }

    /// AudioChunk followed by Interrupted → sink.stop() is called, state → Listening.
    #[tokio::test]
    async fn test_run_barge_in_server() {
        let (provider, tx) = RecordingProvider::new();

        let sink = RecordingSink::new();
        let sink_calls = Arc::clone(&sink.calls);

        let mut session = OmniSession::builder()
            .provider(provider)
            .audio_sink(sink)
            .build()
            .unwrap();

        tx.send(OmniEvent::AudioChunk(make_chunk(42)))
            .await
            .unwrap();
        tx.send(OmniEvent::Interrupted).await.unwrap();
        drop(tx);

        session.run().await.expect("run() should succeed");

        let calls = sink_calls.lock().unwrap().clone();
        assert!(
            calls.contains(&"stop".to_string()),
            "sink.stop() should be called on Interrupted; calls = {calls:?}"
        );
        assert_eq!(session.state(), SessionState::Closed);
    }

    /// AudioChunk followed by TurnComplete → sink.flush() called, state → Listening.
    #[tokio::test]
    async fn test_run_turn_complete() {
        let (provider, tx) = RecordingProvider::new();

        let sink = RecordingSink::new();
        let sink_calls = Arc::clone(&sink.calls);

        let mut session = OmniSession::builder()
            .provider(provider)
            .audio_sink(sink)
            .build()
            .unwrap();

        tx.send(OmniEvent::AudioChunk(make_chunk(7))).await.unwrap();
        tx.send(OmniEvent::TurnComplete).await.unwrap();
        drop(tx);

        session.run().await.expect("run() should succeed");

        let calls = sink_calls.lock().unwrap().clone();
        assert!(
            calls.contains(&"flush".to_string()),
            "sink.flush() should be called on TurnComplete; calls = {calls:?}"
        );
        // After stream ends the session is Closed, but it passed through Listening
        // after TurnComplete — we verify flush was called as evidence of that path.
        assert_eq!(session.state(), SessionState::Closed);
    }

    /// Cancellation token fires → clean shutdown: sink.stop() called, disconnect called.
    #[tokio::test]
    async fn test_run_cancellation() {
        let (provider, _tx) = RecordingProvider::new();
        let disconnected = Arc::clone(&provider.disconnected);

        let sink = RecordingSink::new();
        let sink_calls = Arc::clone(&sink.calls);

        let cancel = CancellationToken::new();

        let mut session = OmniSession::builder()
            .provider(provider)
            .audio_sink(sink)
            .cancel_token(cancel.clone())
            .build()
            .unwrap();

        // Cancel in a spawned task after a tiny yield so run() enters the loop.
        tokio::spawn(async move {
            tokio::task::yield_now().await;
            cancel.cancel();
        });

        session
            .run()
            .await
            .expect("run() should succeed after cancellation");

        let calls = sink_calls.lock().unwrap().clone();
        assert!(
            calls.contains(&"stop".to_string()),
            "sink.stop() should be called on cancellation; calls = {calls:?}"
        );
        assert!(
            *disconnected.lock().unwrap(),
            "disconnect() should be called on cancellation"
        );
        assert_eq!(session.state(), SessionState::Closed);
    }

    /// No audio source or sink (server mode) — provider events are still processed.
    #[tokio::test]
    async fn test_run_no_audio() {
        let (provider, tx) = RecordingProvider::new();
        let recorded_results = Arc::clone(&provider.recorded_tool_results);

        // No audio_source or audio_sink — server/text-only mode.
        let mut session = OmniSession::builder()
            .provider(provider)
            .tool(EchoTool)
            .build()
            .unwrap();

        tx.send(OmniEvent::ToolCall {
            id: "srv-call".to_string(),
            name: "echo".to_string(),
            args: serde_json::json!("ping"),
        })
        .await
        .unwrap();
        tx.send(OmniEvent::TurnComplete).await.unwrap();
        drop(tx);

        session
            .run()
            .await
            .expect("server mode run() should succeed");

        let results = recorded_results.lock().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "srv-call");
        assert_eq!(session.state(), SessionState::Closed);
    }

    /// Provider sends an Error event → run() returns Err, sink.stop() was called.
    #[tokio::test]
    async fn test_run_error_cleanup() {
        let (provider, tx) = RecordingProvider::new();

        let sink = RecordingSink::new();
        let sink_calls = Arc::clone(&sink.calls);

        let mut session = OmniSession::builder()
            .provider(provider)
            .audio_sink(sink)
            .build()
            .unwrap();

        let err = Arc::new(MindroidError::pipeline("provider exploded"));
        tx.send(OmniEvent::Error(err)).await.unwrap();
        // Don't drop tx — run() should return Err before reading further.

        let result = session.run().await;
        assert!(result.is_err(), "run() should return Err on Error event");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("provider exploded"),
            "error message should propagate; got: {msg}"
        );

        let calls = sink_calls.lock().unwrap().clone();
        assert!(
            calls.contains(&"stop".to_string()),
            "sink.stop() should be called before returning Err; calls = {calls:?}"
        );
        assert_eq!(session.state(), SessionState::Closed);
    }

    // ── Local-VAD integration tests (transport-audio feature required) ────────
    //
    // These tests bypass the Silero ONNX worker by using `run_with_vad_rx`,
    // which accepts a pre-populated channel of (f32_samples, probability) pairs
    // that the AudioFrontend processes directly in the run loop.

    /// When `BargeInMode::LocalVad` is configured and the AudioFrontend fires a
    /// `BargeIn` event (sustained speech while agent is speaking), the session
    /// must call `sink.stop()` and return to `Listening`.
    ///
    /// We drive the frontend to a barge-in by:
    /// 1. Sending an AudioChunk provider event so `state == Speaking`.
    /// 2. Injecting 10 consecutive high-probability speech frames into `vad_rx`
    ///    (the InterruptionGate threshold inside AudioFrontend).
    /// 3. Then dropping the provider event sender to end the session.
    #[cfg(feature = "transport-audio")]
    #[tokio::test]
    async fn test_local_barge_in_stops_sink() {
        let (provider, tx) = RecordingProvider::new();
        let sink = RecordingSink::new();
        let sink_calls = Arc::clone(&sink.calls);

        // Configure local barge-in.
        let mut session = OmniSession::builder()
            .provider(provider)
            .audio_sink(sink)
            .barge_in(BargeInMode::LocalVad)
            .turn_detection(TurnDetection::Server) // server-side turn, local barge-in only
            .build()
            .unwrap();

        // Channel that feeds synthetic VAD results to the run loop.
        let (vad_tx, vad_rx) = mpsc::channel::<(Vec<f32>, f32)>(32);

        // Step 1: make the session think the agent is speaking by sending an
        //         AudioChunk event from the provider.
        tx.send(OmniEvent::AudioChunk(make_chunk(1))).await.unwrap();

        // Spawn the session so it starts processing.
        let session_task = tokio::spawn(async move {
            session
                .run_with_vad_rx(vad_rx)
                .await
                .expect("run should succeed");
            session
        });

        // Give the loop a moment to process the AudioChunk (state → Speaking).
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        // Step 2: inject 10 high-probability speech frames while agent is speaking.
        // AudioFrontend's InterruptionGate fires BargeIn on the 10th consecutive
        // speech frame.  Each frame is 512 samples @ 16 kHz (32 ms).
        let frame = vec![0.1f32; 512];
        for _ in 0..10 {
            vad_tx.send((frame.clone(), 0.9)).await.unwrap();
            tokio::task::yield_now().await;
        }

        // Step 3: close VAD channel and provider event stream so the loop exits.
        drop(vad_tx);
        drop(tx);

        let _ = session_task.await.unwrap();

        let calls = sink_calls.lock().unwrap().clone();
        assert!(
            calls.contains(&"stop".to_string()),
            "sink.stop() should be called on local BargeIn; calls = {calls:?}"
        );
    }

    /// When `TurnDetection::Local` is configured and the AudioFrontend fires
    /// `UtteranceComplete`, the session must call `provider.end_audio_stream()`.
    ///
    /// We drive the frontend to UtteranceComplete by:
    /// 1. Injecting enough speech frames to exceed min_speech (≥ 10 frames × 512
    ///    samples = 5120 > 4800 minimum).
    /// 2. Then injecting 38 silence frames to satisfy the silence threshold.
    #[cfg(feature = "transport-audio")]
    #[tokio::test]
    async fn test_local_turn_complete_calls_end_audio_stream() {
        use crate::voice::types::VadConfig;
        use std::time::Duration;

        // VadConfig tuned to match the test parameters (silence=38×32ms, pad=10×32ms).
        let vad_cfg = VadConfig {
            speech_threshold: 0.5,
            speech_end_threshold: 0.3,
            silence_duration: Duration::from_millis(38 * 32),
            speech_pad: Duration::from_millis(10 * 32),
            min_speech: Duration::from_millis(300),
            max_utterance: Duration::from_secs(30),
        };

        let (_provider, tx) = RecordingProvider::new();
        let recorded_eos = Arc::new(Mutex::new(0u32));

        // Wrap RecordingProvider to count end_audio_stream calls.
        struct CountingProvider {
            inner: RecordingProvider,
            count: Arc<Mutex<u32>>,
        }

        #[async_trait]
        impl OmniProvider for CountingProvider {
            async fn connect(&mut self, cfg: &OmniConfig) -> Result<(), MindroidError> {
                self.inner.connect(cfg).await
            }
            async fn send_audio(&self, chunk: AudioChunk) -> Result<(), MindroidError> {
                self.inner.send_audio(chunk).await
            }
            async fn send_text(&self, t: &str) -> Result<(), MindroidError> {
                self.inner.send_text(t).await
            }
            async fn send_tool_result(&self, id: &str, r: Value) -> Result<(), MindroidError> {
                self.inner.send_tool_result(id, r).await
            }
            async fn end_audio_stream(&self) -> Result<(), MindroidError> {
                *self.count.lock().unwrap() += 1;
                Ok(())
            }
            fn events(&mut self) -> Pin<Box<dyn Stream<Item = OmniEvent> + Send>> {
                self.inner.events()
            }
            async fn disconnect(&mut self) -> Result<(), MindroidError> {
                self.inner.disconnect().await
            }
        }

        let (inner_provider, tx2) = RecordingProvider::new();
        let eos_count = Arc::clone(&recorded_eos);
        let counting = CountingProvider {
            inner: inner_provider,
            count: Arc::clone(&eos_count),
        };

        let mut session = OmniSession::builder()
            .provider(counting)
            .barge_in(BargeInMode::Disabled)
            .turn_detection(TurnDetection::Local(vad_cfg))
            .build()
            .unwrap();

        let (vad_tx, vad_rx) = mpsc::channel::<(Vec<f32>, f32)>(64);

        let session_task = tokio::spawn(async move {
            session
                .run_with_vad_rx(vad_rx)
                .await
                .expect("run should succeed");
        });

        // Feed 12 speech frames (12 × 512 = 6144 samples > 4800 min) then 38
        // silence frames to trigger UtteranceComplete in the frontend.
        let speech_frame = vec![0.1f32; 512];
        let silence_frame = vec![0.0f32; 512];

        for _ in 0..12 {
            vad_tx.send((speech_frame.clone(), 0.8)).await.unwrap();
            tokio::task::yield_now().await;
        }
        for _ in 0..38 {
            vad_tx.send((silence_frame.clone(), 0.1)).await.unwrap();
            tokio::task::yield_now().await;
        }

        // Close channels to let the loop exit.
        drop(vad_tx);
        drop(tx);
        drop(tx2);

        session_task.await.unwrap();

        assert!(
            *eos_count.lock().unwrap() >= 1,
            "end_audio_stream should be called at least once on local UtteranceComplete"
        );
    }
}
