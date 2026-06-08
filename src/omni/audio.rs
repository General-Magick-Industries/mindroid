use crate::core::error::MindroidError;
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

use crate::omni::types::AudioChunk;

/// Raw audio input stream (e.g. microphone). No async methods — no #[async_trait].
pub trait AudioSource: Send + Sync + 'static {
    fn stream(&self) -> Pin<Box<dyn Stream<Item = AudioChunk> + Send + '_>>;
    fn sample_rate(&self) -> u32;
    fn channels(&self) -> u16 {
        1
    }
}

/// Raw audio output (e.g. speaker)
#[async_trait]
pub trait AudioSink: Send + Sync + 'static {
    async fn play(&self, chunk: AudioChunk) -> Result<(), MindroidError>;
    async fn flush(&self) -> Result<(), MindroidError>;
    async fn stop(&self) -> Result<(), MindroidError>;
    fn sample_rate(&self) -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_stream::stream;
    use futures::StreamExt;
    use std::sync::{Arc, Mutex};

    fn make_chunk(n: u8) -> AudioChunk {
        AudioChunk {
            data: vec![n],
            sample_rate: 16_000,
            channels: 1,
            bits_per_sample: 16,
        }
    }

    // --- MockAudioSource ---

    struct MockAudioSource {
        chunks: Vec<AudioChunk>,
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
            16_000
        }
    }

    #[tokio::test]
    async fn mock_audio_source_yields_three_chunks() {
        let src = MockAudioSource {
            chunks: vec![make_chunk(1), make_chunk(2), make_chunk(3)],
        };

        assert_eq!(src.sample_rate(), 16_000);
        assert_eq!(src.channels(), 1); // default

        let collected: Vec<AudioChunk> = src.stream().collect().await;
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].data, vec![1]);
        assert_eq!(collected[1].data, vec![2]);
        assert_eq!(collected[2].data, vec![3]);
    }

    #[tokio::test]
    async fn mock_audio_source_stream_ends() {
        let src = MockAudioSource { chunks: vec![] };
        let collected: Vec<AudioChunk> = src.stream().collect().await;
        assert!(collected.is_empty());
    }

    // --- MockAudioSink ---

    struct MockAudioSink {
        calls: Arc<Mutex<Vec<String>>>,
        rate: u32,
    }

    impl MockAudioSink {
        fn new(rate: u32) -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
                rate,
            }
        }
    }

    #[async_trait]
    impl AudioSink for MockAudioSink {
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
            self.rate
        }
    }

    #[tokio::test]
    async fn mock_audio_sink_records_calls() {
        let sink = MockAudioSink::new(44_100);

        assert_eq!(sink.sample_rate(), 44_100);

        sink.play(make_chunk(10)).await.unwrap();
        sink.play(make_chunk(20)).await.unwrap();
        sink.flush().await.unwrap();
        sink.stop().await.unwrap();

        let calls = sink.calls.lock().unwrap().clone();
        assert_eq!(calls, vec!["play:10", "play:20", "flush", "stop"]);
    }

    #[tokio::test]
    async fn mock_audio_sink_trait_object_safe() {
        // Verify AudioSink is usable as a trait object
        let sink: Box<dyn AudioSink> = Box::new(MockAudioSink::new(16_000));
        assert_eq!(sink.sample_rate(), 16_000);
        sink.play(make_chunk(42)).await.unwrap();
        sink.flush().await.unwrap();
        sink.stop().await.unwrap();
    }

    #[test]
    fn mock_audio_source_trait_object_safe() {
        // Verify AudioSource is usable as a trait object
        let src: Box<dyn AudioSource> = Box::new(MockAudioSource {
            chunks: vec![make_chunk(7)],
        });
        assert_eq!(src.sample_rate(), 16_000);
        assert_eq!(src.channels(), 1);
    }
}
