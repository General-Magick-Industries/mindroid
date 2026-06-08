use crate::core::error::MindroidError;
use crate::omni::types::{AudioChunk, OmniConfig, OmniEvent};
use async_trait::async_trait;
use futures::Stream;
use serde_json::Value;
use std::pin::Pin;

/// Bidirectional connection to an omnimodal AI provider.
#[async_trait]
pub trait OmniProvider: Send + Sync + 'static {
    async fn connect(&mut self, config: &OmniConfig) -> Result<(), MindroidError>;
    async fn send_audio(&self, chunk: AudioChunk) -> Result<(), MindroidError>;
    async fn send_text(&self, text: &str) -> Result<(), MindroidError>;
    async fn send_tool_result(&self, call_id: &str, result: Value) -> Result<(), MindroidError>;
    async fn end_audio_stream(&self) -> Result<(), MindroidError>;
    /// Returns an OWNED stream (no lifetime tie to &self).
    /// Provider internally uses mpsc — call events() once before the select! loop.
    fn events(&mut self) -> Pin<Box<dyn Stream<Item = OmniEvent> + Send>>;
    async fn disconnect(&mut self) -> Result<(), MindroidError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    // ── MockOmniProvider ──────────────────────────────────────────────────────

    struct MockOmniProvider {
        /// Chunks accumulated by send_audio
        pub recorded_chunks: Arc<Mutex<Vec<AudioChunk>>>,
        /// Tool results accumulated by send_tool_result
        pub recorded_tool_results: Arc<Mutex<Vec<(String, Value)>>>,
        /// Receiver half — consumed by events()
        event_rx: Option<mpsc::Receiver<OmniEvent>>,
    }

    impl MockOmniProvider {
        /// Returns the provider and the sender the test can use to push events.
        /// The provider does NOT hold a sender clone, so dropping the returned
        /// `tx` is sufficient to close the channel and terminate the stream.
        fn new() -> (Self, mpsc::Sender<OmniEvent>) {
            let (tx, rx) = mpsc::channel::<OmniEvent>(64);
            let provider = MockOmniProvider {
                recorded_chunks: Arc::new(Mutex::new(Vec::new())),
                recorded_tool_results: Arc::new(Mutex::new(Vec::new())),
                event_rx: Some(rx),
            };
            (provider, tx)
        }
    }

    #[async_trait]
    impl OmniProvider for MockOmniProvider {
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
                .expect("events() must only be called once");
            Box::pin(ReceiverStream::new(rx))
        }

        async fn disconnect(&mut self) -> Result<(), MindroidError> {
            Ok(())
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_chunk(data: Vec<u8>) -> AudioChunk {
        AudioChunk {
            data,
            sample_rate: 16_000,
            channels: 1,
            bits_per_sample: 16,
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// Sending 3 audio chunks via send_audio should record them all.
    #[tokio::test]
    async fn test_send_audio_records_chunks() {
        let (mut provider, _tx) = MockOmniProvider::new();
        let config = OmniConfig::default();
        provider.connect(&config).await.unwrap();

        provider
            .send_audio(make_chunk(vec![1, 2, 3]))
            .await
            .unwrap();
        provider
            .send_audio(make_chunk(vec![4, 5, 6]))
            .await
            .unwrap();
        provider
            .send_audio(make_chunk(vec![7, 8, 9]))
            .await
            .unwrap();

        let chunks = provider.recorded_chunks.lock().unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].data, vec![1, 2, 3]);
        assert_eq!(chunks[1].data, vec![4, 5, 6]);
        assert_eq!(chunks[2].data, vec![7, 8, 9]);
    }

    /// Events pushed through the mpsc sender should be yielded by the stream.
    #[tokio::test]
    async fn test_event_stream_yields_tool_call_and_turn_complete() {
        let (mut provider, tx) = MockOmniProvider::new();
        let config = OmniConfig::default();
        provider.connect(&config).await.unwrap();

        // Obtain the stream before sending events (mirrors real select! usage).
        let mut stream = provider.events();

        // Push a ToolCall event followed by TurnComplete, then drop the sender
        // so the stream terminates.
        tx.send(OmniEvent::ToolCall {
            id: "call-1".to_string(),
            name: "get_weather".to_string(),
            args: json!({ "city": "Berlin" }),
        })
        .await
        .unwrap();
        tx.send(OmniEvent::TurnComplete).await.unwrap();
        drop(tx);

        let first = stream.next().await.expect("expected ToolCall event");
        assert!(
            matches!(first, OmniEvent::ToolCall { ref id, .. } if id == "call-1"),
            "unexpected first event: {first:?}",
        );

        let second = stream.next().await.expect("expected TurnComplete event");
        assert!(
            matches!(second, OmniEvent::TurnComplete),
            "unexpected second event: {second:?}",
        );

        // Stream should be exhausted after the sender was dropped.
        assert!(stream.next().await.is_none());

        provider.disconnect().await.unwrap();
    }

    /// connect and disconnect succeed on a fresh mock.
    #[tokio::test]
    async fn test_connect_disconnect_succeed() {
        let (mut provider, _tx) = MockOmniProvider::new();
        let config = OmniConfig::default();
        assert!(provider.connect(&config).await.is_ok());
        assert!(provider.disconnect().await.is_ok());
    }

    /// send_tool_result stores (id, value) pairs.
    #[tokio::test]
    async fn test_send_tool_result_recorded() {
        let (mut provider, _tx) = MockOmniProvider::new();
        let config = OmniConfig::default();
        provider.connect(&config).await.unwrap();

        provider
            .send_tool_result("call-42", json!({ "temp": 21 }))
            .await
            .unwrap();
        provider
            .send_tool_result("call-99", json!("ok"))
            .await
            .unwrap();

        let results = provider.recorded_tool_results.lock().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "call-42");
        assert_eq!(results[0].1, json!({ "temp": 21 }));
        assert_eq!(results[1].0, "call-99");
        assert_eq!(results[1].1, json!("ok"));
    }
}
