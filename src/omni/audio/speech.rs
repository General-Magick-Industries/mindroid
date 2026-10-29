//! Speech probability per microphone chunk, scored off the async runtime.

use tokio::sync::mpsc;

use crate::omni::types::AudioChunk;
#[cfg(feature = "transport-audio")]
use crate::{
    core::error::MindroidError, omni::audio::resample::Resampler, omni::vad::VadInference,
};

/// Speech probability for one microphone chunk, in `[0, 1]`.
///
/// Called on a blocking worker thread, one chunk at a time and in order, so an
/// implementation may do synchronous CPU work such as ONNX inference. It must
/// still keep up with the microphone: chunks that arrive while the worker is busy
/// are dropped, not queued past the channel's bound.
pub trait SpeechDetector: Send {
    fn speech_probability(&mut self, chunk: &AudioChunk) -> f32;
}

/// Chunks queued for, or holding, a score. Small on purpose: a detector that
/// falls behind should shed chunks at the sender rather than build up latency.
const QUEUE: usize = 8;

/// Run `detector` on a blocking worker thread, where inference cannot starve
/// the async loop that feeds it. Every chunk sent in comes back with its
/// probability, in order. The worker stops once the sender is dropped and what
/// it holds is scored, or as soon as the receiver is dropped.
pub(crate) fn spawn_worker(
    mut detector: Box<dyn SpeechDetector>,
) -> (mpsc::Sender<AudioChunk>, mpsc::Receiver<(AudioChunk, f32)>) {
    let (chunk_tx, mut chunk_rx) = mpsc::channel::<AudioChunk>(QUEUE);
    let (score_tx, score_rx) = mpsc::channel(QUEUE);
    tokio::task::spawn_blocking(move || {
        while let Some(chunk) = chunk_rx.blocking_recv() {
            let probability = detector.speech_probability(&chunk);
            if score_tx.blocking_send((chunk, probability)).is_err() {
                break;
            }
        }
    });
    (chunk_tx, score_rx)
}

/// Length of one scored frame. At 16 kHz it is exactly one Silero window.
pub(crate) const FRAME_MS: u32 = 32;

/// Re-cuts microphone chunks into [`FRAME_MS`] frames. The frontend counts
/// frames, so frames of any other length would scale its barge-in and silence
/// timings by the device's chunk size.
#[derive(Debug)]
pub(crate) struct Framer {
    sample_bytes: usize,
    frame_bytes: usize,
    sample_rate: u32,
    channels: u16,
    pending: Vec<u8>,
}

impl Framer {
    pub(crate) fn new(sample_rate: u32, channels: u16) -> Self {
        let sample_bytes = usize::from(channels.max(1)) * 2;
        let samples = sample_rate as usize * FRAME_MS as usize / 1_000;
        Self {
            sample_bytes,
            frame_bytes: samples * sample_bytes,
            sample_rate,
            channels,
            pending: Vec::new(),
        }
    }

    /// A chunk's trailing partial sample is dropped so it cannot misalign
    /// every frame after it.
    pub(crate) fn push(&mut self, chunk: &AudioChunk) -> Vec<AudioChunk> {
        let usable = chunk.data.len() - chunk.data.len() % self.sample_bytes;
        self.pending.extend_from_slice(&chunk.data[..usable]);
        let whole = self.pending.len() - self.pending.len() % self.frame_bytes;
        let frames = self.pending[..whole]
            .chunks_exact(self.frame_bytes)
            .map(|data| AudioChunk {
                data: data.to_vec(),
                sample_rate: self.sample_rate,
                channels: self.channels,
                bits_per_sample: 16,
            })
            .collect();
        self.pending.drain(..whole);
        frames
    }
}

/// Silero VAD. The model takes 8 or 16 kHz audio in fixed frames, so the first
/// channel of the microphone's PCM16 is resampled to the model's rate and framed
/// here; a chunk too short to complete a frame reports the previous probability.
#[cfg(feature = "transport-audio")]
pub struct SileroDetector {
    vad: VadInference,
    resampler: Resampler,
    frame: usize,
    pending: Vec<i16>,
    last: f32,
}

#[cfg(feature = "transport-audio")]
impl SileroDetector {
    /// `sample_rate` is the microphone's rate: 8 kHz, or a multiple of 16 kHz up
    /// to 192 kHz.
    /// Every chunk scored afterwards must arrive at that rate.
    ///
    /// # Errors
    ///
    /// Returns [`MindroidError::Transport`] for an unsupported rate or when the
    /// ONNX model cannot be loaded.
    pub fn new(sample_rate: u32) -> Result<Self, MindroidError> {
        let (rate, frame) = if sample_rate == 8_000 {
            (8_000, 256)
        } else {
            (16_000, 512)
        };
        let resampler =
            Resampler::new(sample_rate, rate).map_err(|e| MindroidError::Transport {
                message: format!("SileroDetector: capture rate {sample_rate} Hz is unsupported"),
                source: Some(Box::new(e)),
            })?;
        Ok(Self {
            vad: VadInference::new(rate, frame)?,
            resampler,
            frame,
            pending: Vec::with_capacity(frame * 2),
            last: 0.0,
        })
    }
}

#[cfg(feature = "transport-audio")]
impl SpeechDetector for SileroDetector {
    fn speech_probability(&mut self, chunk: &AudioChunk) -> f32 {
        let first_channel = chunk
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .step_by(usize::from(chunk.channels.max(1)))
            .map(|b| i16::from_le_bytes(*b));
        self.resampler.process(first_channel, &mut self.pending);
        let whole = self.pending.len() - self.pending.len() % self.frame;
        let best = self.pending[..whole]
            .chunks_exact(self.frame)
            .map(|frame| self.vad.predict(frame))
            .reduce(f32::max);
        self.pending.drain(..whole);
        if let Some(p) = best {
            self.last = p;
        }
        self.last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Doubling;

    impl SpeechDetector for Doubling {
        fn speech_probability(&mut self, chunk: &AudioChunk) -> f32 {
            f32::from(chunk.data[0]) * 2.0
        }
    }

    fn chunk(first: u8) -> AudioChunk {
        AudioChunk {
            data: vec![first, 0],
            sample_rate: 16_000,
            channels: 1,
            bits_per_sample: 16,
        }
    }

    #[tokio::test]
    async fn worker_scores_every_chunk_in_order_and_stops_with_its_sender() {
        let (tx, mut rx) = spawn_worker(Box::new(Doubling));
        for n in 1..=3 {
            tx.send(chunk(n)).await.unwrap();
        }
        drop(tx);
        let mut scored = Vec::new();
        while let Some((chunk, p)) = rx.recv().await {
            scored.push((chunk.data[0], p));
        }
        assert_eq!(scored, [(1, 2.0), (2, 4.0), (3, 6.0)]);
    }

    /// `CpalAudioSource` sends 512-sample chunks, 10.7 ms at 48 kHz; the frontend
    /// must still see 32 ms frames, with the remainder carried to the next push.
    #[test]
    fn framer_cuts_small_mic_chunks_into_frame_ms_frames() {
        let mut framer = Framer::new(48_000, 1);
        let mic = |samples: usize| AudioChunk {
            data: vec![1u8; samples * 2],
            sample_rate: 48_000,
            channels: 1,
            bits_per_sample: 16,
        };
        assert!(framer.push(&mic(512)).is_empty());
        assert!(framer.push(&mic(512)).is_empty());
        let frames = framer.push(&mic(512 + 1_536 + 100));
        assert_eq!(frames.len(), 2);
        assert!(frames.iter().all(|f| f.data.len() == 1_536 * 2));
        assert_eq!(framer.pending.len(), 100 * 2);

        let mut odd = Framer::new(16_000, 1);
        let mut chunk = mic(511);
        chunk.data.push(7);
        assert!(odd.push(&chunk).is_empty());
        assert_eq!(odd.pending.len(), 511 * 2);

        let mut stereo = Framer::new(16_000, 2);
        let frames = stereo.push(&AudioChunk {
            data: vec![0u8; 512 * 2 * 2],
            sample_rate: 16_000,
            channels: 2,
            bits_per_sample: 16,
        });
        assert_eq!(frames.len(), 1);
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

    #[cfg(feature = "transport-audio")]
    #[test]
    fn silero_detector_runs_the_8k_model_on_an_8k_mic() {
        let mut d = SileroDetector::new(8_000).unwrap();
        let silence = AudioChunk {
            data: vec![0u8; 8_000 * 2 / 10],
            sample_rate: 8_000,
            channels: 1,
            bits_per_sample: 16,
        };
        assert!(d.speech_probability(&silence) < 0.3);
        assert!(SileroDetector::new(24_000).is_err());
    }
}
