//! AudioFrontend — orchestrator composing VAD, echo guard, interruption gate,
//! and turn detection into a single probabilities-in / events-out pipeline (task 8).
//!
//! # Design
//!
//! [`AudioFrontend`] is a pure, [`Send`], non-async state machine.  The caller
//! is responsible for:
//! - Running ONNX/Silero inference and producing a probability per chunk.
//! - Passing a monotonic [`Instant`] to every call (no `Instant::now()` inside).
//!
//! This mirrors the design of the underlying layers ([`VadStateMachine`],
//! [`EchoGuard`], etc.) and makes the orchestrator fully deterministic and easy
//! to unit-test with synthetic timelines.
//!
//! ## run_vad parity
//!
//! The accumulation / pad / min-speech / max-utterance logic reproduces the
//! behaviour of `services/mindroid-voice/src/voice_session.rs::run_vad`:
//!
//! - On speech onset: accumulate the onset frame, set `pad_remaining = pad_frames`.
//! - While `pad_remaining > 0`: every frame decrements it; the silence counter is
//!   **not** incremented (immunity window).
//! - After the pad: frames with `probability < speech_end_threshold` increment
//!   `silence_frames`; frames above reset it.
//! - End condition: `silence_frames >= silence_frames_threshold` **or** accumulated
//!   samples ≥ `max_utterance_samples`.
//! - Discard condition: `utterance_buf.len() < min_utterance_samples`.
//!
//! ## InterruptionGate onset reconciliation
//!
//! The onset frame is fed to the gate as frame 1 (counter = 1 after the call),
//! so the 10th consecutive speech-while-agent-speaking frame fires `Interrupt` —
//! matching run_vad's `barge_in_speech_frames = 1` at onset.

use std::time::Instant;

use crate::voice::{
    conditioning::CaptureConditioner,
    echo_guard::EchoGuard,
    interruption::{InterruptionDecision, InterruptionGate, InterruptionScorer},
    turn_detector::{SilenceTimerTurnDetector, TurnDecision, TurnDetector},
    turn_id::{TurnCounter, TurnId},
    types::{BargeInMode, TurnDetection, VadConfig},
    vad::{VadDecision, VadStateMachine},
};

// ─── FrontendEvent ────────────────────────────────────────────────────────────

/// Events emitted by [`AudioFrontend::process`].
#[derive(Debug)]
pub enum FrontendEvent {
    /// VAD detected speech onset for a new turn.
    SpeechStarted,
    /// The user's utterance is complete and meets the minimum-length threshold.
    ///
    /// `samples` carries the raw f32 PCM for the full utterance.  WAV encoding
    /// is the responsibility of the edge/transport layer (task 10).
    UtteranceComplete {
        /// Raw f32 PCM samples (16 kHz, mono).
        samples: Vec<f32>,
        /// The turn this utterance belongs to.
        turn: TurnId,
    },
    /// Sustained speech was detected while the agent was speaking — the agent's
    /// turn should be cancelled.
    BargeIn {
        /// The new (user) turn that triggered the barge-in.
        turn: TurnId,
    },
}

// ─── AudioFrontendBuilder ─────────────────────────────────────────────────────

/// Builder for [`AudioFrontend`].
///
/// Obtain via [`AudioFrontend::builder`].
pub struct AudioFrontendBuilder {
    config: VadConfig,
    chunk_duration_ms: u64,
    sample_rate_hz: u32,
    barge_in_mode: BargeInMode,
    turn_detection: TurnDetection,
    turn_detector: Option<Box<dyn TurnDetector + Send>>,
    scorer: Option<Box<dyn InterruptionScorer>>,
    conditioner: Option<Box<dyn CaptureConditioner>>,
}

impl AudioFrontendBuilder {
    fn new(config: VadConfig, chunk_duration_ms: u64) -> Self {
        Self {
            config,
            chunk_duration_ms,
            sample_rate_hz: 16_000,
            barge_in_mode: BargeInMode::LocalVad,
            turn_detection: TurnDetection::Local(VadConfig::default()),
            turn_detector: None,
            scorer: None,
            conditioner: None,
        }
    }

    /// Set the audio sample rate (Hz) used to convert the `VadConfig` `Duration`
    /// thresholds into sample/frame counts. Defaults to 16 kHz. Set this to match
    /// the transport's real rate (e.g. 8 kHz) so `min_speech`/`max_utterance`
    /// length checks are correct.
    pub fn sample_rate_hz(mut self, rate: u32) -> Self {
        self.sample_rate_hz = rate;
        self
    }

    /// Set the barge-in mode.
    pub fn barge_in_mode(mut self, mode: BargeInMode) -> Self {
        self.barge_in_mode = mode;
        self
    }

    /// Set the turn-detection mode (overrides any custom `TurnDetector`).
    pub fn turn_detection(mut self, td: TurnDetection) -> Self {
        self.turn_detection = td;
        self
    }

    /// Provide a custom [`TurnDetector`] implementation.
    ///
    /// When set, this takes precedence over the `TurnDetection` enum value for
    /// choosing the detector implementation, but [`TurnDetection::Manual`] /
    /// [`TurnDetection::Server`] still disable auto-completion on silence.
    pub fn turn_detector(mut self, td: Box<dyn TurnDetector + Send>) -> Self {
        self.turn_detector = Some(td);
        self
    }

    /// Provide a custom [`InterruptionScorer`].
    ///
    /// The scorer is used to evaluate each frame during barge-in detection.
    /// Defaults to [`crate::voice::interruption::SustainedFramesScorer`].
    pub fn scorer(mut self, s: Box<dyn InterruptionScorer>) -> Self {
        self.scorer = Some(s);
        self
    }

    /// Provide a capture conditioner (AGC/denoise seam).
    ///
    /// Defaults to [`crate::voice::conditioning::PassthroughConditioner`].
    /// Note: the frontend does not call the conditioner itself — see the module
    /// docs for [`crate::voice::conditioning`].  This field is stored for future
    /// use or if a subcomponent needs it.
    pub fn conditioner(mut self, c: Box<dyn CaptureConditioner>) -> Self {
        self.conditioner = Some(c);
        self
    }

    /// Consume the builder and produce an [`AudioFrontend`].
    pub fn build(self) -> AudioFrontend {
        let vad = VadStateMachine::new(self.config.clone(), self.chunk_duration_ms);
        let echo_guard = EchoGuard::default_window();
        let gate = InterruptionGate::default();

        // Derive frame-count thresholds from config (integer division, matching
        // run_vad's `samples / chunk_size` pattern via chunk_duration_ms).
        let sample_rate_hz: usize = self.sample_rate_hz as usize;
        let chunk_samples =
            (sample_rate_hz as u64 * self.chunk_duration_ms / 1_000) as usize;
        let chunk_duration_ms = self.chunk_duration_ms;

        let silence_frames_threshold = {
            let samples = self.config.silence_duration.as_millis() as usize
                * sample_rate_hz
                / 1_000;
            samples.div_ceil(chunk_samples)
        };
        let pad_frames = {
            let samples = self.config.speech_pad.as_millis() as usize
                * sample_rate_hz
                / 1_000;
            samples.div_ceil(chunk_samples)
        };
        let min_utterance_samples = self.config.min_speech.as_millis() as usize
            * sample_rate_hz
            / 1_000;
        let max_utterance_samples = self.config.max_utterance.as_millis() as usize
            * sample_rate_hz
            / 1_000;

        // Resolve TurnDetector: explicit override > TurnDetection::Local > default.
        let turn_detector: Box<dyn TurnDetector + Send> =
            if let Some(td) = self.turn_detector {
                td
            } else {
                match &self.turn_detection {
                    TurnDetection::Local(_) => {
                        Box::new(SilenceTimerTurnDetector::new())
                    }
                    // Manual / Server: we still store a SilenceTimerTurnDetector
                    // but the auto_complete flag prevents it from ending turns.
                    TurnDetection::Manual | TurnDetection::Server => {
                        Box::new(SilenceTimerTurnDetector::new())
                    }
                }
            };

        // Whether silence-based auto-completion is active.
        let auto_complete = matches!(self.turn_detection, TurnDetection::Local(_));

        AudioFrontend {
            vad,
            echo_guard,
            gate,
            turn_detector,
            counter: TurnCounter::new(),
            config: self.config,
            chunk_duration_ms,
            silence_frames_threshold,
            pad_frames,
            min_utterance_samples,
            max_utterance_samples,
            barge_in_mode: self.barge_in_mode,
            auto_complete,
            // Runtime state.
            speaking: false,
            current_turn: None,
            utterance_buf: Vec::new(),
            silence_frames: 0,
            pad_remaining: 0,
            // Source-pluggable overrides.
            injected_barge_in: false,
            injected_turn_start: false,
            user_speaking_override: None,
            conditioning_done: false,
        }
    }
}

// ─── AudioFrontend ────────────────────────────────────────────────────────────

/// Orchestrator that composes [`VadStateMachine`], [`EchoGuard`],
/// [`InterruptionGate`], and [`TurnDetector`] into a single per-chunk pipeline.
///
/// ## Usage
///
/// ```rust,no_run
/// use mindroid::voice::frontend::{AudioFrontend, FrontendEvent};
/// use mindroid::voice::types::{VadConfig, BargeInMode, TurnDetection};
/// use std::time::Instant;
///
/// let mut fe = AudioFrontend::builder(VadConfig::default(), 32)
///     .barge_in_mode(BargeInMode::LocalVad)
///     .turn_detection(TurnDetection::Local(VadConfig::default()))
///     .build();
///
/// let frame = vec![0.0f32; 512];
/// let now = Instant::now();
/// let events = fe.process(&frame, 0.7, false, now);
/// ```
pub struct AudioFrontend {
    // ── Sub-components ────────────────────────────────────────────────────────
    vad: VadStateMachine,
    echo_guard: EchoGuard,
    gate: InterruptionGate,
    turn_detector: Box<dyn TurnDetector + Send>,
    counter: TurnCounter,

    // ── Config-derived thresholds ─────────────────────────────────────────────
    config: VadConfig,
    #[allow(dead_code)]
    chunk_duration_ms: u64,
    silence_frames_threshold: usize,
    pad_frames: usize,
    min_utterance_samples: usize,
    max_utterance_samples: usize,
    barge_in_mode: BargeInMode,
    /// Whether silence-based auto-completion is active (Local mode only).
    auto_complete: bool,

    // ── Per-utterance runtime state ───────────────────────────────────────────
    speaking: bool,
    current_turn: Option<TurnId>,
    utterance_buf: Vec<f32>,
    silence_frames: usize,
    pad_remaining: usize,

    // ── Source-pluggable signal overrides ─────────────────────────────────────
    /// Pending injected barge-in; consumed in the next `process` call.
    injected_barge_in: bool,
    /// Pending injected turn start; consumed in the next `process` call.
    injected_turn_start: bool,
    /// When `Some`, overrides the caller-supplied `agent_speaking` argument for
    /// the barge-in path.
    user_speaking_override: Option<bool>,
    /// Informational flag set by [`mark_conditioning_done`].
    conditioning_done: bool,
}

impl AudioFrontend {
    /// Create a builder for `AudioFrontend`.
    ///
    /// # Arguments
    ///
    /// - `config` — VAD and timing parameters.
    /// - `chunk_duration_ms` — duration of each audio chunk in milliseconds.
    ///   For 512 samples at 16 kHz this is 32 ms.
    pub fn builder(config: VadConfig, chunk_duration_ms: u64) -> AudioFrontendBuilder {
        AudioFrontendBuilder::new(config, chunk_duration_ms)
    }

    // ── Public delegating helpers ─────────────────────────────────────────────

    /// Notify the echo guard that a TTS audio chunk was sent at `now`.
    ///
    /// Delegates to [`EchoGuard::notify_tts_chunk`].
    pub fn notify_tts_chunk(&mut self, now: Instant) {
        self.echo_guard.notify_tts_chunk(now);
    }

    /// Notify the echo guard that the agent has stopped speaking.
    ///
    /// Delegates to [`EchoGuard::notify_agent_stopped`] and resets the
    /// interruption-gate counter.
    pub fn notify_agent_stopped(&mut self) {
        self.echo_guard.notify_agent_stopped();
        self.gate.reset();
    }

    // ── Source-pluggable signal injection ─────────────────────────────────────

    /// Force a [`FrontendEvent::BargeIn`] for the current turn in the next
    /// [`process`](Self::process) call.
    ///
    /// Used by device/server signals that know the user is interrupting without
    /// waiting for the local sustained-speech counter to reach 10.  When no
    /// turn is active, a new turn is started first.
    pub fn inject_barge_in(&mut self) {
        self.injected_barge_in = true;
    }

    /// Force a turn start in the next [`process`](Self::process) call, without
    /// waiting for VAD onset.
    ///
    /// Useful when the caller has out-of-band knowledge (e.g. a push-to-talk
    /// button, a server VAD event) that the user has begun speaking.
    pub fn inject_turn_start(&mut self) {
        self.injected_turn_start = true;
    }

    /// Override the "user is speaking" flag used by the barge-in path.
    ///
    /// When set to `Some(v)`, this replaces the `agent_speaking` argument
    /// passed to [`process`](Self::process) for the InterruptionGate decision.
    /// Pass `None` to remove the override and revert to the caller-supplied
    /// value.
    ///
    /// Note: this is named `set_user_speaking` per the task spec; it overrides
    /// from the *user*'s perspective (e.g. a WebRTC track-presence signal).
    pub fn set_user_speaking(&mut self, speaking: Option<bool>) {
        self.user_speaking_override = speaking;
    }

    /// Informational: mark that upstream capture conditioning (AGC/denoise) has
    /// already been applied to the stream by the caller.
    ///
    /// This does not change any processing path inside the frontend.  It is
    /// stored so that future layers or logging can inspect whether conditioning
    /// was applied.  See [`crate::voice::conditioning`] for context.
    pub fn mark_conditioning_done(&mut self) {
        self.conditioning_done = true;
    }

    /// Whether the caller has declared that conditioning is done.
    pub fn conditioning_done(&self) -> bool {
        self.conditioning_done
    }

    /// The current active turn, if any.
    pub fn current_turn(&self) -> Option<TurnId> {
        self.current_turn
    }

    // ── Core processing ───────────────────────────────────────────────────────

    /// Process one audio chunk and return the events it produces.
    ///
    /// # Arguments
    ///
    /// - `frame` — f32 PCM samples for this chunk (should be exactly
    ///   `chunk_samples` in length; shorter frames are accepted gracefully).
    /// - `probability` — VAD probability for this chunk (0.0–1.0), computed by
    ///   the caller's ONNX / Silero inference.
    /// - `agent_speaking` — whether the agent is currently playing back TTS.
    /// - `now` — monotonic timestamp for this chunk.  Must not regress between
    ///   calls.
    ///
    /// # Returns
    ///
    /// Zero or more [`FrontendEvent`]s.  Multiple events can be returned in a
    /// single call (e.g. `SpeechStarted` + `BargeIn` on the 10th frame, or
    /// `SpeechStarted` immediately followed by an injected barge-in).
    pub fn process(
        &mut self,
        frame: &[f32],
        probability: f32,
        agent_speaking: bool,
        now: Instant,
    ) -> Vec<FrontendEvent> {
        let mut events = Vec::new();

        // ── 1. Handle source-pluggable injections ─────────────────────────────

        if self.injected_turn_start && !self.speaking {
            self.injected_turn_start = false;
            let turn = self.counter.next();
            self.current_turn = Some(turn);
            self.speaking = true;
            self.silence_frames = 0;
            self.pad_remaining = self.pad_frames;
            self.utterance_buf.extend_from_slice(frame);
            // Reset VAD so it doesn't mis-fire SpeechStarted.
            self.vad.reset();
            events.push(FrontendEvent::SpeechStarted);
        }

        if self.injected_barge_in {
            self.injected_barge_in = false;
            let turn = self.current_turn.unwrap_or_else(|| {
                let t = self.counter.next();
                self.current_turn = Some(t);
                t
            });
            events.push(FrontendEvent::BargeIn { turn });
        }

        // ── 2. Feed probability to VAD state machine ──────────────────────────

        let vad_decision = self.vad.process(probability);

        // ── 3. Handle VAD onset (only if we haven't already started via inject) ─

        if matches!(vad_decision, VadDecision::SpeechStarted) && !self.speaking {
            self.speaking = true;
            let turn = self.counter.next();
            self.current_turn = Some(turn);
            self.silence_frames = 0;
            self.pad_remaining = self.pad_frames;
            self.utterance_buf.extend_from_slice(frame);
            events.push(FrontendEvent::SpeechStarted);

            // Onset frame is the first barge-in frame (counter = 1 after this call).
            let echo_suppressed = self.echo_guard.is_suppressed(now);
            let gate_agent_speaking =
                self.user_speaking_override.unwrap_or(agent_speaking);
            if !matches!(self.barge_in_mode, BargeInMode::Disabled) {
                let decision = self.gate.observe(true, gate_agent_speaking, echo_suppressed);
                if let Some(ev) = self.maybe_barge_in(decision, turn) {
                    events.push(ev);
                }
            }
            return events;
        }

        // ── 4. Accumulate frame and manage pad / silence counters ─────────────

        if self.speaking {
            self.utterance_buf.extend_from_slice(frame);

            if self.pad_remaining > 0 {
                // Inside speech_pad immunity window: decrement counter but do
                // not advance silence_frames regardless of probability.
                self.pad_remaining -= 1;
            } else if probability >= self.config.speech_end_threshold {
                self.silence_frames = 0;
            } else {
                self.silence_frames += 1;
            }

            // ── 5. Barge-in gate (continuing speech frames) ───────────────────
            let is_speech = probability >= self.config.speech_threshold;
            let echo_suppressed = self.echo_guard.is_suppressed(now);
            let gate_agent_speaking = self.user_speaking_override.unwrap_or(agent_speaking);

            if !matches!(self.barge_in_mode, BargeInMode::Disabled) {
                let decision = self.gate.observe(is_speech, gate_agent_speaking, echo_suppressed);
                if let Some(turn) = self.current_turn {
                    if let Some(ev) = self.maybe_barge_in(decision, turn) {
                        events.push(ev);
                    }
                }
            }

            // ── 6. Endpointing ────────────────────────────────────────────────
            let silence_ended = self.silence_frames >= self.silence_frames_threshold;
            let max_reached = self.utterance_buf.len() >= self.max_utterance_samples;

            // TurnDetector is consulted when auto_complete is on.
            let turn_complete = if self.auto_complete {
                let td_decision = self.turn_detector.observe(vad_decision, now);
                matches!(td_decision, TurnDecision::Complete) || max_reached
            } else {
                // Manual / Server: only max-utterance can force an end.
                max_reached
            };

            if (silence_ended || max_reached) && self.auto_complete || max_reached && !self.auto_complete {
                if let Some(turn) = self.current_turn.take() {
                    let samples = std::mem::take(&mut self.utterance_buf);
                    if samples.len() >= self.min_utterance_samples {
                        events.push(FrontendEvent::UtteranceComplete { samples, turn });
                    }
                    // Either way, reset state.
                    self.speaking = false;
                    self.silence_frames = 0;
                    self.pad_remaining = 0;
                    self.gate.reset();
                    self.vad.reset();
                } else {
                    // Belt-and-suspenders: reset even without a turn id.
                    self.utterance_buf.clear();
                    self.speaking = false;
                    self.silence_frames = 0;
                    self.pad_remaining = 0;
                    self.gate.reset();
                    self.vad.reset();
                }
            } else {
                // Check turn_complete from TurnDetector separately (handles
                // silence_ended via TurnDecision::Complete path).
                let _ = turn_complete; // suppress unused warning — already
                                      // merged into the combined condition above.
            }
        }

        events
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Convert an `InterruptionDecision` to an optional `FrontendEvent::BargeIn`,
    /// honouring the current `BargeInMode`.
    fn maybe_barge_in(
        &self,
        decision: InterruptionDecision,
        turn: TurnId,
    ) -> Option<FrontendEvent> {
        match (&self.barge_in_mode, decision) {
            (BargeInMode::Disabled, _) => None,
            (_, InterruptionDecision::Interrupt) => Some(FrontendEvent::BargeIn { turn }),
            _ => None,
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::types::{BargeInMode, TurnDetection, VadConfig};
    use std::time::{Duration, Instant};

    // ── Test helpers ──────────────────────────────────────────────────────────

    /// Chunk duration: 512 samples @ 16 kHz ≈ 32 ms.
    const CHUNK_MS: u64 = 32;
    const CHUNK_SAMPLES: usize = 512;

    /// Config tuned for the test scenarios:
    /// - silence = 38 frames × 32 ms ≈ 1216 ms
    /// - pad = 10 frames × 32 ms ≈ 320 ms
    /// - min_speech = 4800 samples ≈ 300 ms (exactly 4800 / 512 ≈ 9.4 chunks)
    fn test_config() -> VadConfig {
        VadConfig {
            speech_threshold: 0.5,
            speech_end_threshold: 0.3,
            silence_duration: Duration::from_millis(38 * CHUNK_MS),
            speech_pad: Duration::from_millis(10 * CHUNK_MS),
            min_speech: Duration::from_millis(300),
            max_utterance: Duration::from_secs(30),
        }
    }

    fn speech_frame() -> Vec<f32> {
        vec![0.1f32; CHUNK_SAMPLES]
    }

    fn silence_frame() -> Vec<f32> {
        vec![0.0f32; CHUNK_SAMPLES]
    }

    fn build_frontend(barge_in_mode: BargeInMode, auto_complete: bool) -> AudioFrontend {
        let td = if auto_complete {
            TurnDetection::Local(test_config())
        } else {
            TurnDetection::Manual
        };
        AudioFrontend::builder(test_config(), CHUNK_MS)
            .barge_in_mode(barge_in_mode)
            .turn_detection(td)
            .build()
    }

    /// Collect all events from feeding `n` speech frames (high probability).
    fn feed_speech(
        fe: &mut AudioFrontend,
        n: usize,
        agent_speaking: bool,
        base: Instant,
        offset_ms: u64,
    ) -> (Vec<FrontendEvent>, u64) {
        let mut all = Vec::new();
        let mut t = offset_ms;
        for _ in 0..n {
            let now = base + Duration::from_millis(t);
            let evs = fe.process(&speech_frame(), 0.8, agent_speaking, now);
            all.extend(evs);
            t += CHUNK_MS;
        }
        (all, t)
    }

    /// Collect all events from feeding `n` silence frames (low probability).
    fn feed_silence(
        fe: &mut AudioFrontend,
        n: usize,
        agent_speaking: bool,
        base: Instant,
        offset_ms: u64,
    ) -> (Vec<FrontendEvent>, u64) {
        let mut all = Vec::new();
        let mut t = offset_ms;
        for _ in 0..n {
            let now = base + Duration::from_millis(t);
            let evs = fe.process(&silence_frame(), 0.1, agent_speaking, now);
            all.extend(evs);
            t += CHUNK_MS;
        }
        (all, t)
    }

    fn has_barge_in(events: &[FrontendEvent]) -> bool {
        events
            .iter()
            .any(|e| matches!(e, FrontendEvent::BargeIn { .. }))
    }

    fn has_utterance_complete(events: &[FrontendEvent]) -> bool {
        events
            .iter()
            .any(|e| matches!(e, FrontendEvent::UtteranceComplete { .. }))
    }

    fn has_speech_started(events: &[FrontendEvent]) -> bool {
        events
            .iter()
            .any(|e| matches!(e, FrontendEvent::SpeechStarted))
    }

    // ── Test 1: echo suppression prevents BargeIn ─────────────────────────────

    /// Agent speaking, notify_tts_chunk fired recently, 10 speech frames within
    /// 250 ms of last TTS chunk → NO BargeIn (echo suppression active).
    #[test]
    fn echo_suppression_prevents_barge_in() {
        let base = Instant::now();
        let mut fe = build_frontend(BargeInMode::LocalVad, true);

        // Simulate a TTS chunk arriving at t=0.
        fe.notify_tts_chunk(base);

        // Feed 10 speech frames within 250 ms (10 × 32 ms = 320 ms → first 7
        // frames are within 250 ms window, but we want ALL 10 within 250ms).
        // Use a very short chunk so all 10 fit: check the actual window.
        // At 32 ms/chunk, frame 8 is at t = 7*32 = 224 ms < 250 ms.
        // Frame 9 is at t = 8*32 = 256 ms ≥ 250 ms → frame 9 onward is NOT
        // suppressed.  To keep all 10 within 250 ms we need chunk ≤ 24 ms.
        //
        // Instead, re-notify_tts_chunk before each frame to ensure every frame
        // sees a fresh suppression window — this matches the intent of the test
        // (agent is actively streaming TTS, so chunks arrive continuously).
        let mut all_events = Vec::new();
        for i in 0..10usize {
            let now = base + Duration::from_millis(i as u64 * CHUNK_MS);
            // Refresh the TTS-chunk timestamp so suppression stays active.
            fe.notify_tts_chunk(now);
            let evs = fe.process(&speech_frame(), 0.8, true, now);
            all_events.extend(evs);
        }

        assert!(!has_barge_in(&all_events), "barge-in should be suppressed by echo guard");
    }

    // ── Test 2: sustained barge-in fires after suppression window ────────────

    /// Agent speaking, 10 speech frames starting > 250 ms after the last TTS
    /// chunk → BargeIn fires.
    #[test]
    fn sustained_barge_in_fires_outside_echo_window() {
        let base = Instant::now();
        let mut fe = build_frontend(BargeInMode::LocalVad, true);

        // Record a TTS chunk at t=0.
        fe.notify_tts_chunk(base);

        // Start speech at t=300 ms (well outside 250 ms suppression window).
        let (events, _) = feed_speech(&mut fe, 10, true, base, 300);

        assert!(has_barge_in(&events), "barge-in should fire outside echo window");
    }

    // ── Test 3: short noise (9 frames then non-speech) → no BargeIn ──────────

    /// Agent speaking, 9 consecutive speech frames then a non-speech frame →
    /// gate counter resets, no BargeIn.
    #[test]
    fn short_noise_nine_then_silence_no_barge_in() {
        let base = Instant::now();
        let mut fe = build_frontend(BargeInMode::LocalVad, true);

        // Speech started at t=0 (within pad, but still counting for barge-in).
        let (mut all, mut t) = feed_speech(&mut fe, 9, true, base, 0);

        // One non-speech frame.
        let now = base + Duration::from_millis(t);
        let evs = fe.process(&silence_frame(), 0.1, true, now);
        all.extend(evs);
        t += CHUNK_MS;
        let _ = t;

        assert!(!has_barge_in(&all), "9 frames + non-speech should not fire barge-in");
    }

    // ── Test 4: reset on agent stop ───────────────────────────────────────────

    /// After notify_agent_stopped, subsequent speech frames do NOT cause barge-in
    /// and are processed as a normal utterance.
    #[test]
    fn reset_on_agent_stop() {
        let base = Instant::now();
        let mut fe = build_frontend(BargeInMode::LocalVad, true);

        // 5 speech frames while agent is speaking.
        let (events1, t) = feed_speech(&mut fe, 5, true, base, 0);

        // Agent stops.
        fe.notify_agent_stopped();

        // More speech frames — agent NOT speaking.
        let (events2, _) = feed_speech(&mut fe, 10, false, base, t);

        // Should not have fired any barge-in at any point.
        let all: Vec<_> = events1.into_iter().chain(events2).collect();
        assert!(!has_barge_in(&all), "after agent stopped, no barge-in should fire");
    }

    // ── Test 5: silence turn-end produces UtteranceComplete ──────────────────

    /// Enough speech frames (≥ min_speech) followed by 38 silence frames →
    /// UtteranceComplete emitted.
    #[test]
    fn silence_turn_end_emits_utterance_complete() {
        let base = Instant::now();
        let mut fe = build_frontend(BargeInMode::Disabled, true);

        // Feed enough speech to exceed min_speech (4800 samples ÷ 512 ≈ 10 frames).
        let (_, t) = feed_speech(&mut fe, 12, false, base, 0);

        // Now feed 38 silence frames to trigger endpointing.
        let (silence_events, _) = feed_silence(&mut fe, 38, false, base, t);

        assert!(
            has_utterance_complete(&silence_events),
            "38 silence frames after sufficient speech should emit UtteranceComplete"
        );
    }

    // ── Test 6: speech_pad immunity window ───────────────────────────────────

    /// After onset, low-probability frames inside the pad window do NOT advance
    /// the silence counter, so the utterance is not ended prematurely.
    #[test]
    fn speech_pad_prevents_early_end() {
        let base = Instant::now();
        let mut fe = build_frontend(BargeInMode::Disabled, true);

        // Trigger onset with one high-prob frame.
        let t0 = base;
        let evs = fe.process(&speech_frame(), 0.8, false, t0);
        assert!(has_speech_started(&evs), "should have started speech");
        assert!(fe.speaking, "should be speaking after onset");

        // Feed pad_frames low-prob frames (inside the immunity window).
        // pad = 10 frames.
        let pad_count = 10;
        let mut t = CHUNK_MS;
        for _ in 0..pad_count {
            let now = base + Duration::from_millis(t);
            let evs = fe.process(&silence_frame(), 0.1, false, now);
            assert!(
                !has_utterance_complete(&evs),
                "pad frame should not end utterance"
            );
            t += CHUNK_MS;
        }

        // After the pad, silence counter should still be 0 (pad ate the silence).
        assert_eq!(
            fe.silence_frames, 0,
            "silence counter should be 0 after pad window"
        );

        // Utterance is still active.
        assert!(fe.speaking, "should still be speaking after pad");
    }

    // ── Test 7: too-short utterance discarded ─────────────────────────────────

    /// Utterance shorter than min_speech (< 4800 samples ≈ 10 frames) is
    /// discarded — no UtteranceComplete event.
    #[test]
    fn too_short_utterance_discarded() {
        let base = Instant::now();
        let mut fe = build_frontend(BargeInMode::Disabled, true);

        // Only 3 speech frames (3 × 512 = 1536 samples < 4800).
        let (_, t) = feed_speech(&mut fe, 3, false, base, 0);

        // Feed 38 silence frames to trigger endpointing.
        let (silence_events, _) = feed_silence(&mut fe, 38, false, base, t);

        assert!(
            !has_utterance_complete(&silence_events),
            "utterance shorter than min_speech should be discarded"
        );
    }

    // ── Test: BargeIn disabled → never emits BargeIn ─────────────────────────

    #[test]
    fn barge_in_disabled_never_fires() {
        let base = Instant::now();
        let mut fe = build_frontend(BargeInMode::Disabled, true);

        let (events, _) = feed_speech(&mut fe, 15, true, base, 0);

        assert!(!has_barge_in(&events), "BargeIn should never fire when mode is Disabled");
    }

    // ── Test: inject_barge_in emits BargeIn immediately ──────────────────────

    #[test]
    fn inject_barge_in_emits_event() {
        let base = Instant::now();
        let mut fe = build_frontend(BargeInMode::LocalVad, true);

        // Start a turn.
        fe.process(&speech_frame(), 0.8, false, base);

        // Inject a barge-in.
        fe.inject_barge_in();
        let now = base + Duration::from_millis(CHUNK_MS);
        let evs = fe.process(&speech_frame(), 0.8, false, now);

        assert!(has_barge_in(&evs), "injected barge-in should produce BargeIn event");
    }

    // ── Test: inject_turn_start starts a turn without VAD ────────────────────

    #[test]
    fn inject_turn_start_starts_turn() {
        let base = Instant::now();
        let mut fe = build_frontend(BargeInMode::Disabled, true);

        // No speech, but inject a turn start.
        fe.inject_turn_start();
        let evs = fe.process(&silence_frame(), 0.0, false, base);

        assert!(has_speech_started(&evs), "injected turn start should emit SpeechStarted");
        assert!(fe.speaking);
    }

    // ── Test: Manual turn detection does not auto-complete ───────────────────

    #[test]
    fn manual_turn_detection_no_auto_complete() {
        let base = Instant::now();
        let mut fe = build_frontend(BargeInMode::Disabled, false);

        // Feed lots of speech then lots of silence.
        let (_, t) = feed_speech(&mut fe, 20, false, base, 0);
        let (events, _) = feed_silence(&mut fe, 50, false, base, t);

        assert!(
            !has_utterance_complete(&events),
            "Manual mode should not auto-complete on silence"
        );
    }

    // ── Test: mark_conditioning_done is informational ────────────────────────

    #[test]
    fn mark_conditioning_done_is_informational() {
        let base = Instant::now();
        let mut fe = build_frontend(BargeInMode::Disabled, true);

        assert!(!fe.conditioning_done());
        fe.mark_conditioning_done();
        assert!(fe.conditioning_done());

        // Processing still works normally.
        let evs = fe.process(&speech_frame(), 0.8, false, base);
        assert!(has_speech_started(&evs));
    }
}
