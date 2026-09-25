//! Open a session only while someone is talking.
//!
//! The microphone stays open and a local speech detector watches it. The first
//! detected speech opens an [`OmniSession`], replaying the last moment of audio so
//! the opening words are not lost. After [`idle_timeout`](VoiceGateBuilder::idle_timeout)
//! without speech the session is closed, which flushes its transcripts to memory;
//! the next speech opens a fresh session that seeds that history back in. With
//! [`provider_closes`](VoiceGateBuilder::provider_closes) the gate only ever opens:
//! the session lives until the provider ends it, and the next speech reopens.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::time::Instant;

use futures::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

use crate::core::error::MindroidError;
use crate::memory::Memory;
use crate::omni::audio::{AudioSink, AudioSource};
use crate::omni::provider::OmniProvider;
use crate::omni::session::{OmniSession, ToolContextInit};
use crate::omni::types::{AudioChunk, OmniConfig};
use crate::pipeline::stages::stt::SttProvider;
use crate::tools::{Tool, ToolContext};
use crate::voice::types::VadConfig;
use crate::voice::vad::{VadDecision, VadStateMachine};

/// Speech probability for one microphone chunk, in `[0, 1]`.
///
/// Called inline on the gate's loop for every chunk, so an implementation must be
/// cheap per call (Silero is about a millisecond per 32 ms frame).
pub trait SpeechDetector: Send {
    fn speech_probability(&mut self, chunk: &AudioChunk) -> f32;
}

/// Silero VAD. The model only takes 16 kHz audio in 512-sample frames, so the
/// microphone's PCM16 is decimated to 16 kHz (first channel only) and framed here;
/// a chunk too short to complete a frame reports the previous probability.
#[cfg(feature = "transport-audio")]
pub struct SileroDetector {
    vad: crate::omni::vad::VadInference,
    pending: Vec<i16>,
    last: f32,
}

#[cfg(feature = "transport-audio")]
impl SileroDetector {
    const RATE: u32 = 16_000;
    const FRAME: usize = 512;

    /// `sample_rate` is the microphone's rate and must be a multiple of 16 kHz.
    ///
    /// # Errors
    ///
    /// Returns [`MindroidError::Transport`] for an unsupported rate or when the
    /// ONNX model cannot be loaded.
    pub fn new(sample_rate: u32) -> Result<Self, MindroidError> {
        if sample_rate == 0 || !sample_rate.is_multiple_of(Self::RATE) {
            return Err(MindroidError::Transport {
                message: format!(
                    "SileroDetector: capture rate {sample_rate} Hz is not a multiple of 16 kHz"
                ),
                source: None,
            });
        }
        Ok(Self {
            vad: crate::omni::vad::VadInference::new(Self::RATE, Self::FRAME)?,
            pending: Vec::with_capacity(Self::FRAME * 2),
            last: 0.0,
        })
    }
}

#[cfg(feature = "transport-audio")]
impl SpeechDetector for SileroDetector {
    fn speech_probability(&mut self, chunk: &AudioChunk) -> f32 {
        let step =
            (chunk.sample_rate / Self::RATE).max(1) as usize * chunk.channels.max(1) as usize;
        self.pending.extend(
            chunk
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .step_by(step)
                .map(|b| i16::from_le_bytes(*b)),
        );
        let mut best: Option<f32> = None;
        while self.pending.len() >= Self::FRAME {
            let frame: Vec<i16> = self.pending.drain(..Self::FRAME).collect();
            let p = self.vad.predict(&frame);
            best = Some(best.map_or(p, |b| b.max(p)));
        }
        if let Some(p) = best {
            self.last = p;
        }
        self.last
    }
}

type ProviderFactory = Box<dyn Fn() -> Box<dyn OmniProvider> + Send + Sync>;

/// One session's audio, fed by the gate. `stream()` ends when the gate closes it.
struct ChannelAudioSource {
    rx: Mutex<Option<mpsc::Receiver<AudioChunk>>>,
    sample_rate: u32,
    channels: u16,
}

impl AudioSource for ChannelAudioSource {
    fn stream(&self) -> Pin<Box<dyn Stream<Item = AudioChunk> + Send + '_>> {
        let rx = self
            .rx
            .lock()
            .expect("ChannelAudioSource: mutex poisoned")
            .take();
        match rx {
            Some(rx) => Box::pin(ReceiverStream::new(rx)),
            None => Box::pin(futures::stream::empty()),
        }
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }
}

struct Live {
    tx: mpsc::Sender<AudioChunk>,
    cancel: CancellationToken,
    task: JoinSet<Result<(), MindroidError>>,
}

/// Runs [`OmniSession`]s on demand: one per burst of conversation, opened by
/// speech and closed by silence. Build with [`VoiceGate::builder`].
pub struct VoiceGate {
    provider: ProviderFactory,
    detector: Box<dyn SpeechDetector>,
    audio_source: Arc<dyn AudioSource>,
    audio_sink: Option<Arc<dyn AudioSink>>,
    tools: Vec<Arc<dyn Tool>>,
    config: OmniConfig,
    memory: Option<Arc<dyn Memory>>,
    conversation: Option<(String, String, String)>,
    history_limit: Option<usize>,
    transcriber: Option<Arc<dyn SttProvider>>,
    tool_ctx_init: Option<ToolContextInit>,
    vad: VadConfig,
    idle_timeout: Option<Duration>,
    preroll: Duration,
    cancel: CancellationToken,
}

impl VoiceGate {
    pub fn builder() -> VoiceGateBuilder {
        VoiceGateBuilder::default()
    }

    /// Watch the microphone until it closes or the cancel token fires. Every
    /// session opened along the way is closed and flushed before this returns.
    ///
    /// # Errors
    ///
    /// Returns the error from building a session; a session that fails while
    /// running is logged and the gate keeps listening.
    pub async fn run(mut self) -> Result<(), MindroidError> {
        let source = Arc::clone(&self.audio_source);
        let mut mic = source.stream();
        let rate = source.sample_rate();
        let channels = source.channels();
        let bytes_per_ms = (rate as usize * 2 * channels as usize / 1000).max(1);
        let preroll_cap = bytes_per_ms * self.preroll.as_millis() as usize;

        let mut vad: Option<VadStateMachine> = None;
        let mut preroll: VecDeque<AudioChunk> = VecDeque::new();
        let mut preroll_bytes = 0usize;
        let mut live: Option<Live> = None;
        let mut last_speech = Instant::now();
        let mut opened = 0usize;
        tracing::info!("voice gate: waiting for speech");

        loop {
            tokio::select! {
                biased;

                _ = self.cancel.cancelled() => {
                    Self::close(&mut live, "cancelled").await;
                    return Ok(());
                }

                ended = async {
                    match live.as_mut() {
                        Some(l) => l.task.join_next().await,
                        None => std::future::pending().await,
                    }
                } => {
                    match ended {
                        Some(Ok(Ok(()))) => tracing::info!("voice gate: session ended on its own"),
                        Some(Ok(Err(e))) => tracing::warn!(%e, "voice gate: session failed"),
                        Some(Err(e)) => tracing::warn!(%e, "voice gate: session task failed"),
                        None => {}
                    }
                    live = None;
                    if let Some(v) = vad.as_mut() {
                        v.reset();
                    }
                    tracing::info!("voice gate: waiting for speech");
                }

                // No idle timeout means the session closes only when the provider
                // ends it: the arm is still declared, but `pending()` never resolves.
                _ = async {
                    match self.idle_timeout {
                        Some(d) => tokio::time::sleep_until(last_speech + d).await,
                        None => std::future::pending::<()>().await,
                    }
                }, if live.is_some() => {
                    Self::close(&mut live, "idle").await;
                    if let Some(v) = vad.as_mut() {
                        v.reset();
                    }
                }

                chunk = mic.next() => {
                    let Some(chunk) = chunk else {
                        Self::close(&mut live, "microphone closed").await;
                        return Ok(());
                    };
                    let chunk_ms = (chunk.data.len() / bytes_per_ms).max(1) as u64;
                    let sm = vad.get_or_insert_with(|| VadStateMachine::new(self.vad.clone(), chunk_ms));
                    let decision = sm.process(self.detector.speech_probability(&chunk));
                    tracing::trace!(?decision, first = chunk.data.first(), live = live.is_some(), "voice gate chunk");
                    if matches!(decision, VadDecision::SpeechStarted | VadDecision::SpeechContinues) {
                        last_speech = Instant::now();
                    }
                    match live.as_ref() {
                        Some(l) => {
                            if l.tx.try_send(chunk).is_err() {
                                tracing::debug!("voice gate: session audio queue full, chunk dropped");
                            }
                        }
                        None => {
                            preroll_bytes += chunk.data.len();
                            preroll.push_back(chunk);
                            while preroll_bytes > preroll_cap && preroll.len() > 1 {
                                preroll_bytes -= preroll.pop_front().map_or(0, |c| c.data.len());
                            }
                            if decision == VadDecision::SpeechStarted {
                                opened += 1;
                                let replay: Vec<AudioChunk> = preroll.drain(..).collect();
                                preroll_bytes = 0;
                                live = Some(self.open(rate, channels, replay, opened)?);
                            }
                        }
                    }
                }
            }
        }
    }

    fn open(
        &self,
        rate: u32,
        channels: u16,
        replay: Vec<AudioChunk>,
        n: usize,
    ) -> Result<Live, MindroidError> {
        let (tx, rx) = mpsc::channel(256);
        for chunk in replay {
            let _ = tx.try_send(chunk);
        }
        let cancel = self.cancel.child_token();
        let mut b = OmniSession::builder()
            .provider_boxed((self.provider)())
            .audio_source(ChannelAudioSource {
                rx: Mutex::new(Some(rx)),
                sample_rate: rate,
                channels,
            })
            .tools(self.tools.clone())
            .config(self.config.clone())
            .cancel_token(cancel.clone());
        if let Some(s) = &self.audio_sink {
            b = b.audio_sink_shared(Arc::clone(s));
        }
        if let Some(m) = &self.memory {
            b = b.memory(Arc::clone(m));
        }
        if let Some((c, s, a)) = &self.conversation {
            b = b.conversation(c.clone(), s.clone(), a.clone());
        }
        if let Some(limit) = self.history_limit {
            b = b.history_limit(limit);
        }
        if let Some(t) = &self.transcriber {
            b = b.transcriber(Arc::clone(t));
        }
        if let Some(init) = &self.tool_ctx_init {
            let init = Arc::clone(init);
            b = b.tool_context_init(move |ctx| init(ctx));
        }
        let mut session = b.build()?;
        tracing::info!(session = n, "voice gate: speech detected, opening session");
        let mut task = JoinSet::new();
        task.spawn(async move { session.run().await });
        Ok(Live { tx, cancel, task })
    }

    async fn close(live: &mut Option<Live>, why: &str) {
        let Some(mut l) = live.take() else {
            return;
        };
        tracing::info!(why, "voice gate: closing session");
        drop(l.tx);
        l.cancel.cancel();
        while let Some(res) = l.task.join_next().await {
            match res {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!(%e, "voice gate: session failed on close"),
                Err(e) => tracing::warn!(%e, "voice gate: session task failed on close"),
            }
        }
        tracing::info!("voice gate: waiting for speech");
    }
}

/// Builder for [`VoiceGate`]. Requires a provider factory, a detector and an
/// audio source; everything else is optional and passed to each session as-is.
pub struct VoiceGateBuilder {
    provider: Option<ProviderFactory>,
    detector: Option<Box<dyn SpeechDetector>>,
    audio_source: Option<Arc<dyn AudioSource>>,
    audio_sink: Option<Arc<dyn AudioSink>>,
    tools: Vec<Arc<dyn Tool>>,
    config: OmniConfig,
    memory: Option<Arc<dyn Memory>>,
    conversation: Option<(String, String, String)>,
    history_limit: Option<usize>,
    transcriber: Option<Arc<dyn SttProvider>>,
    tool_ctx_init: Option<ToolContextInit>,
    vad: VadConfig,
    idle_timeout: Option<Duration>,
    preroll: Duration,
    cancel: Option<CancellationToken>,
}

impl Default for VoiceGateBuilder {
    fn default() -> Self {
        Self {
            provider: None,
            detector: None,
            audio_source: None,
            audio_sink: None,
            tools: Vec::new(),
            config: OmniConfig::default(),
            memory: None,
            conversation: None,
            history_limit: None,
            transcriber: None,
            tool_ctx_init: None,
            vad: VadConfig::default(),
            idle_timeout: Some(Duration::from_secs(15)),
            preroll: Duration::from_secs(1),
            cancel: None,
        }
    }
}

impl VoiceGateBuilder {
    /// Called once per session opened; each session needs a fresh connection.
    pub fn provider<F>(mut self, factory: F) -> Self
    where
        F: Fn() -> Box<dyn OmniProvider> + Send + Sync + 'static,
    {
        self.provider = Some(Box::new(factory));
        self
    }

    pub fn detector(mut self, d: impl SpeechDetector + 'static) -> Self {
        self.detector = Some(Box::new(d));
        self
    }

    pub fn audio_source(mut self, s: impl AudioSource + 'static) -> Self {
        self.audio_source = Some(Arc::new(s));
        self
    }

    pub fn audio_sink(mut self, s: impl AudioSink + 'static) -> Self {
        self.audio_sink = Some(Arc::new(s));
        self
    }

    pub fn tools(mut self, t: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = t;
        self
    }

    pub fn config(mut self, c: OmniConfig) -> Self {
        self.config = c;
        self
    }

    pub fn memory(mut self, m: Arc<dyn Memory>) -> Self {
        self.memory = Some(m);
        self
    }

    pub fn conversation(
        mut self,
        channel_id: impl Into<String>,
        sender_id: impl Into<String>,
        agent_id: impl Into<String>,
    ) -> Self {
        self.conversation = Some((channel_id.into(), sender_id.into(), agent_id.into()));
        self
    }

    pub fn history_limit(mut self, n: usize) -> Self {
        self.history_limit = Some(n);
        self
    }

    pub fn transcriber(mut self, stt: Arc<dyn SttProvider>) -> Self {
        self.transcriber = Some(stt);
        self
    }

    /// Populate the [`ToolContext`] for every tool call in every session this gate
    /// opens (e.g. deposit `AgentCredentials`).
    pub fn tool_context_init<F>(mut self, init: F) -> Self
    where
        F: Fn(&ToolContext) + Send + Sync + 'static,
    {
        self.tool_ctx_init = Some(Arc::new(init));
        self
    }

    /// Thresholds for the gate's own speech detection (not the provider's).
    pub fn vad(mut self, v: VadConfig) -> Self {
        self.vad = v;
        self
    }

    /// Silence after the last detected speech that closes the session. Default 15 s.
    pub fn idle_timeout(mut self, d: Duration) -> Self {
        self.idle_timeout = Some(d);
        self
    }

    /// Never close on silence: the session ends only when the provider ends it
    /// (or the gate is cancelled). Local speech still opens the next one.
    pub fn provider_closes(mut self) -> Self {
        self.idle_timeout = None;
        self
    }

    /// Audio replayed into a new session from before speech was detected. Default 1 s.
    pub fn preroll(mut self, d: Duration) -> Self {
        self.preroll = d;
        self
    }

    pub fn cancel_token(mut self, ct: CancellationToken) -> Self {
        self.cancel = Some(ct);
        self
    }

    /// # Errors
    ///
    /// Returns [`MindroidError::Config`] when the provider factory, detector or
    /// audio source is missing.
    pub fn build(self) -> Result<VoiceGate, MindroidError> {
        Ok(VoiceGate {
            provider: self
                .provider
                .ok_or_else(|| MindroidError::config("VoiceGate requires a provider factory"))?,
            detector: self
                .detector
                .ok_or_else(|| MindroidError::config("VoiceGate requires a speech detector"))?,
            audio_source: self
                .audio_source
                .ok_or_else(|| MindroidError::config("VoiceGate requires an audio source"))?,
            audio_sink: self.audio_sink,
            tools: self.tools,
            config: self.config,
            memory: self.memory,
            conversation: self.conversation,
            history_limit: self.history_limit,
            transcriber: self.transcriber,
            tool_ctx_init: self.tool_ctx_init,
            vad: self.vad,
            idle_timeout: self.idle_timeout,
            preroll: self.preroll,
            cancel: self.cancel.unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::omni::mock::MockOmniProvider;
    use crate::omni::types::OmniEvent;

    /// Speech when the first byte is 1.
    struct Scripted;

    impl SpeechDetector for Scripted {
        fn speech_probability(&mut self, chunk: &AudioChunk) -> f32 {
            if chunk.data.first() == Some(&1) {
                0.9
            } else {
                0.0
            }
        }
    }

    /// Yields chunks with a real-time gap between them so idle timers can fire.
    struct Paced {
        chunks: Vec<AudioChunk>,
        gap: Duration,
    }

    impl AudioSource for Paced {
        fn stream(&self) -> Pin<Box<dyn Stream<Item = AudioChunk> + Send + '_>> {
            let gap = self.gap;
            Box::pin(futures::stream::unfold(
                self.chunks.clone().into_iter(),
                move |mut it| async move {
                    let next = it.next()?;
                    tokio::time::sleep(gap).await;
                    Some((next, it))
                },
            ))
        }

        fn sample_rate(&self) -> u32 {
            16_000
        }
    }

    fn chunk(first: u8) -> AudioChunk {
        let mut data = vec![0u8; 320]; // 10 ms at 16 kHz mono
        data[0] = first;
        AudioChunk {
            data,
            sample_rate: 16_000,
            channels: 1,
            bits_per_sample: 16,
        }
    }

    type Opened = Arc<Mutex<Vec<(Arc<Mutex<Vec<AudioChunk>>>, mpsc::Sender<OmniEvent>)>>>;

    fn factory(opened: &Opened) -> impl Fn() -> Box<dyn OmniProvider> + Send + Sync + 'static {
        let opened = Arc::clone(opened);
        move || {
            let (provider, events) = MockOmniProvider::new();
            opened
                .lock()
                .unwrap()
                .push((provider.received_audio(), events));
            Box::new(provider)
        }
    }

    fn gate(chunks: Vec<AudioChunk>, idle: Duration, opened: &Opened) -> VoiceGate {
        VoiceGate::builder()
            .provider(factory(opened))
            .detector(Scripted)
            .audio_source(Paced {
                chunks,
                gap: Duration::from_millis(10),
            })
            .vad(VadConfig {
                silence_duration: Duration::from_millis(50),
                ..VadConfig::default()
            })
            .idle_timeout(idle)
            .build()
            .unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn speech_opens_one_session_and_replays_the_preroll() {
        let opened: Opened = Arc::default();
        let chunks = vec![
            chunk(0),
            chunk(0),
            chunk(0),
            chunk(1),
            chunk(1),
            chunk(1),
            chunk(0),
        ];
        gate(chunks, Duration::from_secs(5), &opened)
            .run()
            .await
            .unwrap();

        let opened = opened.lock().unwrap();
        assert_eq!(opened.len(), 1, "exactly one session for one burst");
        let heard = opened[0].0.lock().unwrap();
        assert!(
            heard.len() >= 4,
            "preroll plus speech reached the provider: {}",
            heard.len()
        );
        assert_eq!(
            heard[0].data[0], 0,
            "the first chunk is replayed silence from before speech"
        );
        assert!(heard.iter().any(|c| c.data[0] == 1));
    }

    #[tokio::test(start_paused = true)]
    async fn silence_closes_the_session_and_speech_reopens_a_new_one() {
        let opened: Opened = Arc::default();
        let mut chunks = vec![chunk(1), chunk(1)];
        chunks.extend(std::iter::repeat_n(chunk(0), 60)); // >600 ms of silence, well past the idle timeout
        chunks.extend([chunk(1), chunk(1), chunk(0)]);
        gate(chunks, Duration::from_millis(200), &opened)
            .run()
            .await
            .unwrap();

        assert_eq!(
            opened.lock().unwrap().len(),
            2,
            "idle closed the first, speech opened a second"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn silence_alone_never_opens_a_session() {
        let opened: Opened = Arc::default();
        gate(vec![chunk(0); 5], Duration::from_secs(5), &opened)
            .run()
            .await
            .unwrap();
        assert!(opened.lock().unwrap().is_empty());
    }

    /// A 48 kHz stereo microphone, the common Windows default, must be accepted and
    /// framed down to what Silero takes; silence must score low.
    #[cfg(feature = "transport-audio")]
    #[test]
    fn silero_detector_takes_a_48k_stereo_mic() {
        let mut d = SileroDetector::new(48_000).unwrap();
        let silence = AudioChunk {
            data: vec![0u8; 48_000 * 2 * 2 / 10], // 100 ms stereo
            sample_rate: 48_000,
            channels: 2,
            bits_per_sample: 16,
        };
        let p = d.speech_probability(&silence);
        assert!(p < 0.3, "silence scored {p}");
        assert!(SileroDetector::new(44_100).is_err());
    }

    #[test]
    fn builder_requires_the_three_inputs() {
        assert!(VoiceGate::builder().build().is_err());
    }
}
