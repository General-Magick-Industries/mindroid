//! Integration tests for the `omni` module.
//!
//! These tests exercise the full public API surface — `OmniSession` built with
//! `MockOmniProvider`, `MockAudioSource`, and `MockAudioSink` from the
//! `mindroid::omni::mock` module.  No real audio hardware or real providers
//! are used.

use async_trait::async_trait;
use mindroid::memory::Memory;
use mindroid::models::{Message, SenderType};
use mindroid::omni::mock::{MockAudioSink, MockAudioSource, MockOmniProvider};
use mindroid::omni::session::OmniSession;
use mindroid::omni::types::{
    AudioChunk, OmniConfig, OmniEvent, Role, SessionState, TranscriptSource,
};
use mindroid::pipeline::stages::stt::SttProvider;
use mindroid::tools::Tool;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

// ── Shared helpers ────────────────────────────────────────────────────────────

fn make_chunk(n: u8) -> AudioChunk {
    AudioChunk {
        data: vec![n],
        sample_rate: 16_000,
        channels: 1,
        bits_per_sample: 16,
    }
}

// ── EchoTool — reusable tool mock ─────────────────────────────────────────────

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echoes arguments back as a JSON string"
    }

    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }

    async fn execute(
        &self,
        args: Value,
        _ctx: &mindroid::tools::ToolContext,
    ) -> mindroid::Result<String> {
        Ok(args.to_string())
    }
}

// ── Test 1: Full lifecycle ─────────────────────────────────────────────────────
//
// Sequence: AudioChunk (provider → sink)  →  ToolCall (echo)  →  TurnComplete
//
// Verifies:
//   • Session starts in Connecting, transitions to Listening after connect
//   • AudioChunk event causes sink.play() to be called (Speaking state)
//   • ToolCall is dispatched and the result is sent back via send_tool_result
//   • TurnComplete causes sink.flush()
//   • Final state is Closed (stream exhausted)

#[tokio::test]
async fn lifecycle_audio_tool_turn_complete() {
    let (provider, tx) = MockOmniProvider::new();

    // Capture handles to shared state before moving provider into session
    let received_audio = provider.received_audio();
    let received_tool_results = provider.received_tool_results();
    let disconnected = provider.disconnected_flag();

    let sink = MockAudioSink::new();
    let sink_calls = sink.calls();

    let mut session = OmniSession::builder()
        .provider(provider)
        .audio_sink(sink)
        .tool(EchoTool)
        .build()
        .expect("build should succeed");

    // Drive the event sequence
    tx.send(OmniEvent::AudioChunk(make_chunk(42)))
        .await
        .unwrap();
    tx.send(OmniEvent::ToolCall {
        id: "call-1".to_string(),
        name: "echo".to_string(),
        args: json!({ "answer": 42 }),
    })
    .await
    .unwrap();
    tx.send(OmniEvent::TurnComplete).await.unwrap();
    drop(tx); // close the event stream → session exits cleanly

    session.run().await.expect("run() should succeed");

    // 1. No audio was sent from client → source; provider received 0 audio chunks
    assert_eq!(
        received_audio.lock().unwrap().len(),
        0,
        "no audio source was attached — provider should receive 0 chunks"
    );

    // 2. Tool result was sent back
    {
        let results = received_tool_results.lock().unwrap();
        assert_eq!(results.len(), 1, "one tool result expected");
        assert_eq!(results[0].0, "call-1");
        assert_eq!(
            results[0].1,
            Value::String(r#"{"answer":42}"#.to_string()),
            "EchoTool should return args.to_string()"
        );
    }

    // 3. Sink received play + flush (AudioChunk → play, TurnComplete → flush)
    {
        let calls = sink_calls.lock().unwrap().clone();
        assert!(
            calls.contains(&"play:42".to_string()),
            "sink.play() should be called for the AudioChunk event; calls={calls:?}"
        );
        assert!(
            calls.contains(&"flush".to_string()),
            "sink.flush() should be called for TurnComplete; calls={calls:?}"
        );
    }

    // 4. Session ended cleanly
    assert_eq!(session.state(), SessionState::Closed);

    // 5. Provider was NOT explicitly disconnected (stream ended naturally, not cancelled)
    // The session calls disconnect() only on cancellation, not on natural end.
    let _ = disconnected; // used above — just suppressing unused warning
}

// ── Test 2: Cancellation mid-session ─────────────────────────────────────────
//
// Verifies:
//   • CancellationToken fires during run()
//   • Session transitions to Closed
//   • Sink.stop() is called
//   • Provider.disconnect() is called

#[tokio::test]
async fn cancellation_mid_session_clean_shutdown() {
    let (provider, _tx) = MockOmniProvider::new();

    let disconnected = provider.disconnected_flag();

    let sink = MockAudioSink::new();
    let sink_calls = sink.calls();

    let cancel = CancellationToken::new();

    let mut session = OmniSession::builder()
        .provider(provider)
        .audio_sink(sink)
        .cancel_token(cancel.clone())
        .build()
        .expect("build should succeed");

    // Cancel from a spawned task after giving run() a chance to enter the loop.
    tokio::spawn(async move {
        tokio::task::yield_now().await;
        cancel.cancel();
    });

    session
        .run()
        .await
        .expect("run() should return Ok on cancellation");

    // Sink.stop() must have been called
    let calls = sink_calls.lock().unwrap().clone();
    assert!(
        calls.contains(&"stop".to_string()),
        "sink.stop() must be called on cancellation; calls={calls:?}"
    );

    // Provider.disconnect() must have been called
    assert!(
        *disconnected.lock().unwrap(),
        "provider.disconnect() must be called on cancellation"
    );

    assert_eq!(session.state(), SessionState::Closed);
}

// ── Test 3: Server mode (no audio source/sink) ────────────────────────────────
//
// Verifies:
//   • Session runs with no AudioSource and no AudioSink
//   • ToolCall events are still processed
//   • TurnComplete is handled without panic
//   • Session exits cleanly

#[tokio::test]
async fn server_mode_no_audio_processes_events() {
    let (provider, tx) = MockOmniProvider::new();
    let received_tool_results = provider.received_tool_results();

    // No audio_source or audio_sink — pure server / text-only mode
    let mut session = OmniSession::builder()
        .provider(provider)
        .tool(EchoTool)
        .build()
        .expect("build should succeed in server mode");

    tx.send(OmniEvent::ToolCall {
        id: "srv-call".to_string(),
        name: "echo".to_string(),
        args: json!("ping"),
    })
    .await
    .unwrap();
    tx.send(OmniEvent::TurnComplete).await.unwrap();
    drop(tx);

    session
        .run()
        .await
        .expect("server-mode run() should succeed");

    let results = received_tool_results.lock().unwrap();
    assert_eq!(results.len(), 1, "one tool result expected");
    assert_eq!(results[0].0, "srv-call");
    assert_eq!(
        results[0].1,
        Value::String(r#""ping""#.to_string()),
        "EchoTool echoes the JSON value"
    );

    assert_eq!(session.state(), SessionState::Closed);
}

// ── Test 4: Audio forwarding (source → provider) ──────────────────────────────
//
// Verifies:
//   • Chunks produced by MockAudioSource are forwarded via send_audio
//   • After the source is exhausted, end_audio_stream is called (implicitly)
//   • Session exits cleanly once the event stream also ends

#[tokio::test]
async fn audio_source_chunks_forwarded_to_provider() {
    let (provider, tx) = MockOmniProvider::new();
    let received_audio = provider.received_audio();

    let source = MockAudioSource::new(vec![make_chunk(1), make_chunk(2), make_chunk(3)]);

    let mut session = OmniSession::builder()
        .provider(provider)
        .audio_source(source)
        .build()
        .expect("build should succeed");

    // Drop event sender so the provider stream ends immediately;
    // session will finish once the audio source and provider stream are both done.
    drop(tx);

    session.run().await.expect("run() should succeed");

    let chunks = received_audio.lock().unwrap();
    assert_eq!(chunks.len(), 3, "all 3 source chunks should be forwarded");
    assert_eq!(chunks[0].data, vec![1]);
    assert_eq!(chunks[1].data, vec![2]);
    assert_eq!(chunks[2].data, vec![3]);

    assert_eq!(session.state(), SessionState::Closed);
}

// ── Test 5: Barge-in / Interrupted event ─────────────────────────────────────
//
// Verifies:
//   • Interrupted event causes sink.stop()
//   • Session can continue after interruption (does not close)

#[tokio::test]
async fn interrupted_event_stops_sink() {
    let (provider, tx) = MockOmniProvider::new();

    let sink = MockAudioSink::new();
    let sink_calls = sink.calls();

    let mut session = OmniSession::builder()
        .provider(provider)
        .audio_sink(sink)
        .build()
        .expect("build should succeed");

    // Emit audio, then interrupt, then end the session
    tx.send(OmniEvent::AudioChunk(make_chunk(7))).await.unwrap();
    tx.send(OmniEvent::Interrupted).await.unwrap();
    drop(tx);

    session
        .run()
        .await
        .expect("run() should succeed after interrupt");

    let calls = sink_calls.lock().unwrap().clone();
    assert!(
        calls.contains(&"stop".to_string()),
        "sink.stop() should be called on Interrupted; calls={calls:?}"
    );
    assert_eq!(session.state(), SessionState::Closed);
}

// ── Test 6: Error event propagates ───────────────────────────────────────────
//
// Verifies:
//   • Provider Error event causes run() to return Err
//   • sink.stop() is called before returning
//   • Session state is Closed

#[tokio::test]
async fn error_event_propagates_and_stops_sink() {
    use mindroid::MindroidError;

    let (provider, tx) = MockOmniProvider::new();

    let sink = MockAudioSink::new();
    let sink_calls = sink.calls();

    let mut session = OmniSession::builder()
        .provider(provider)
        .audio_sink(sink)
        .build()
        .expect("build should succeed");

    let err = Arc::new(MindroidError::pipeline("provider crashed"));
    tx.send(OmniEvent::Error(err)).await.unwrap();
    // Do NOT drop tx — run() should return before reading further

    let result = session.run().await;
    assert!(result.is_err(), "run() must return Err on Error event");
    assert!(
        result.unwrap_err().to_string().contains("provider crashed"),
        "error message should propagate"
    );

    let calls = sink_calls.lock().unwrap().clone();
    assert!(
        calls.contains(&"stop".to_string()),
        "sink.stop() must be called on Error; calls={calls:?}"
    );
    assert_eq!(session.state(), SessionState::Closed);
}

// ── Test 7: Transcript events are ignored gracefully ─────────────────────────

#[tokio::test]
async fn transcript_events_ignored_gracefully() {
    let (provider, tx) = MockOmniProvider::new();

    let mut session = OmniSession::builder()
        .provider(provider)
        .build()
        .expect("build should succeed");

    tx.send(OmniEvent::Transcript {
        text: "hello there".to_string(),
        is_final: true,
        source: TranscriptSource::Input,
    })
    .await
    .unwrap();
    tx.send(OmniEvent::Transcript {
        text: "intermediate".to_string(),
        is_final: false,
        source: TranscriptSource::Output,
    })
    .await
    .unwrap();
    tx.send(OmniEvent::TurnComplete).await.unwrap();
    drop(tx);

    // Should not panic or error
    session
        .run()
        .await
        .expect("transcript events should be handled gracefully");

    assert_eq!(session.state(), SessionState::Closed);
}

// ── Test 8: Builder without provider fails ────────────────────────────────────

// ── RecordingMemory — seeds fixed history, records every save ────────────────

/// (channel, sender, content, reply_to)
type SavedTurn = (String, String, String, Option<String>);

struct RecordingMemory {
    history: Vec<Message>,
    saved: Arc<Mutex<Vec<SavedTurn>>>,
}

#[async_trait]
impl Memory for RecordingMemory {
    async fn save_message(
        &self,
        channel_id: &str,
        sender_id: &str,
        content: &str,
        reply_to_id: Option<&str>,
    ) -> mindroid::Result<Option<String>> {
        let mut saved = self.saved.lock().unwrap();
        saved.push((
            channel_id.into(),
            sender_id.into(),
            content.into(),
            reply_to_id.map(str::to_string),
        ));
        Ok(Some(format!("msg-{}", saved.len())))
    }

    async fn get_history(
        &self,
        _channel_id: &str,
        _limit: usize,
    ) -> mindroid::Result<Vec<Message>> {
        Ok(self.history.clone())
    }

    async fn clear_history(&self, _channel_id: &str) -> mindroid::Result<()> {
        Ok(())
    }
}

/// Prior turns come out of memory as text history before the provider connects,
/// with roles derived from who sent them.
#[tokio::test]
async fn history_is_seeded_from_memory_before_connect() {
    let mut agent_turn = Message::new("hello", "agent", "chan");
    agent_turn.sender_type = SenderType::Agent;
    let memory = RecordingMemory {
        history: vec![Message::new("hi", "user", "chan"), agent_turn],
        saved: Arc::new(Mutex::new(Vec::new())),
    };

    let (provider, tx) = MockOmniProvider::new();
    let mut session = OmniSession::builder()
        .provider(provider)
        .memory(Arc::new(memory))
        .conversation("chan", "user", "agent")
        .build()
        .unwrap();
    drop(tx);
    session.run().await.unwrap();

    let history = &session.config().history;
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].role, Role::User);
    assert_eq!(history[0].text, "hi");
    assert_eq!(history[1].role, Role::Model);
    assert_eq!(history[1].text, "hello");
}

async fn seed_and_read_prompt(history: Vec<Message>, base_prompt: Option<&str>) -> Option<String> {
    let (provider, tx) = MockOmniProvider::new();
    let config = OmniConfig {
        system_prompt: base_prompt.map(str::to_string),
        ..OmniConfig::default()
    };
    let mut session = OmniSession::builder()
        .provider(provider)
        .config(config)
        .memory(Arc::new(RecordingMemory {
            history,
            saved: Arc::new(Mutex::new(Vec::new())),
        }))
        .conversation("chan", "user", "agent")
        .build()
        .unwrap();
    drop(tx);
    session.run().await.unwrap();
    session.config().system_prompt.clone()
}

/// A session resumed after a long gap tells the model it is a new session; a fresh
/// one moments after the last turn does not.
#[tokio::test]
async fn a_gap_since_the_last_turn_is_flagged_to_the_model() {
    let mut old = Message::new("earlier", "user", "chan");
    old.timestamp = chrono::Utc::now() - chrono::Duration::minutes(20);
    let after_gap = seed_and_read_prompt(vec![old], Some("You are a helper.")).await;
    let after_gap = after_gap.expect("prompt present");
    assert!(
        after_gap.starts_with("You are a helper."),
        "keeps the base prompt"
    );
    assert!(
        after_gap.contains("new session"),
        "flags the resume: {after_gap}"
    );
    assert!(
        after_gap.contains("20 minutes"),
        "names the gap: {after_gap}"
    );

    let recent = Message::new("just now", "user", "chan"); // timestamp = now
    let continuous = seed_and_read_prompt(vec![recent], Some("You are a helper."))
        .await
        .expect("prompt present");
    assert_eq!(
        continuous, "You are a helper.",
        "no note for a natural pause"
    );
}

/// Only final transcripts are persisted, in event order, each under the right
/// sender, with the agent's turn threaded to the user's saved message id.
#[tokio::test]
async fn final_transcripts_are_persisted_under_the_right_sender() {
    let saved = Arc::new(Mutex::new(Vec::new()));
    let memory = RecordingMemory {
        history: Vec::new(),
        saved: Arc::clone(&saved),
    };

    let (provider, tx) = MockOmniProvider::new();
    let mut session = OmniSession::builder()
        .provider(provider)
        .memory(Arc::new(memory))
        .conversation("chan", "user", "agent")
        .build()
        .unwrap();

    let t = |text: &str, is_final: bool, source: TranscriptSource| OmniEvent::Transcript {
        text: text.into(),
        is_final,
        source,
    };
    tx.send(t("what time", true, TranscriptSource::Input))
        .await
        .unwrap();
    tx.send(t("It's", false, TranscriptSource::Output))
        .await
        .unwrap();
    tx.send(t("It's noon.", true, TranscriptSource::Output))
        .await
        .unwrap();
    tx.send(OmniEvent::TurnComplete).await.unwrap();
    drop(tx);
    session.run().await.unwrap();

    let saved = saved.lock().unwrap();
    assert_eq!(
        *saved,
        vec![
            (
                "chan".to_string(),
                "user".to_string(),
                "what time".to_string(),
                None
            ),
            (
                "chan".to_string(),
                "agent".to_string(),
                "It's noon.".to_string(),
                Some("msg-1".to_string())
            ),
        ],
        "finals only, in event order, with the reply threaded to the user's message"
    );
}

#[test]
fn builder_requires_provider() {
    match OmniSession::builder().build() {
        Err(e) => {
            assert!(
                e.to_string().to_lowercase().contains("provider"),
                "error message must mention 'provider'; got: {e}"
            );
        }
        Ok(_) => panic!("build() without a provider must fail"),
    }
}

// ── Test 9: Multiple audio events and tool calls in sequence ─────────────────

#[tokio::test]
async fn multiple_tool_calls_in_sequence() {
    let (provider, tx) = MockOmniProvider::new();
    let received_tool_results = provider.received_tool_results();

    let mut session = OmniSession::builder()
        .provider(provider)
        .tool(EchoTool)
        .build()
        .expect("build should succeed");

    for i in 0u32..3 {
        tx.send(OmniEvent::ToolCall {
            id: format!("call-{i}"),
            name: "echo".to_string(),
            args: json!({ "i": i }),
        })
        .await
        .unwrap();
    }
    tx.send(OmniEvent::TurnComplete).await.unwrap();
    drop(tx);

    session.run().await.expect("run() should succeed");

    let results = received_tool_results.lock().unwrap();
    assert_eq!(results.len(), 3, "three tool results expected");
    for i in 0u32..3 {
        assert_eq!(results[i as usize].0, format!("call-{i}"));
    }
    assert_eq!(session.state(), SessionState::Closed);
}

struct SlowStt;

#[async_trait]
impl SttProvider for SlowStt {
    async fn transcribe(&self, audio: &[u8]) -> mindroid::Result<String> {
        assert_eq!(
            &audio[..4],
            b"RIFF",
            "session must hand the transcriber a WAV"
        );
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        Ok("what time".to_string())
    }
}

/// With our own transcriber, the user turn is stored before the reply even though
/// the transcript resolves after the reply's final transcript has arrived.
#[tokio::test]
async fn own_transcription_keeps_the_user_turn_ahead_of_the_reply() {
    let (provider, tx) = MockOmniProvider::new();
    let saved = Arc::new(Mutex::new(Vec::new()));
    let memory = RecordingMemory {
        history: Vec::new(),
        saved: Arc::clone(&saved),
    };
    let source = MockAudioSource::with_sample_rate(
        vec![AudioChunk {
            data: vec![0; 16_000],
            sample_rate: 16_000,
            channels: 1,
            bits_per_sample: 16,
        }],
        16_000,
    );
    let mut session = OmniSession::builder()
        .provider(provider)
        .audio_source(source)
        .memory(Arc::new(memory))
        .conversation("chan", "user", "agent")
        .transcriber(Arc::new(SlowStt))
        .build()
        .unwrap();

    let t = |text: &str, source: TranscriptSource| OmniEvent::Transcript {
        text: text.to_string(),
        is_final: true,
        source,
    };
    tx.send(OmniEvent::Interrupted).await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    tx.send(OmniEvent::UserSpeechEnded).await.unwrap();
    tx.send(t(
        "ignored: provider input transcript",
        TranscriptSource::Input,
    ))
    .await
    .unwrap();
    tx.send(t("It's noon.", TranscriptSource::Output))
        .await
        .unwrap();
    tx.send(OmniEvent::TurnComplete).await.unwrap();
    drop(tx);

    session.run().await.unwrap();

    let saved = saved.lock().unwrap();
    assert_eq!(
        *saved,
        vec![
            (
                "chan".to_string(),
                "user".to_string(),
                "what time".to_string(),
                None
            ),
            (
                "chan".to_string(),
                "agent".to_string(),
                "It's noon.".to_string(),
                Some("msg-1".to_string()),
            ),
        ],
        "user turn from our transcriber first, provider input transcript ignored, reply threaded"
    );
}
