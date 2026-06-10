//! Shared WAV encoding helper.
//!
//! Gated behind the `transport-audio` feature because it depends on `hound`.
//! The pure `voice/` types (VAD state machine, frontend, etc.) do not require
//! this module and remain always-available.

#[cfg(feature = "transport-audio")]
use tracing::error;

/// Encode f32 PCM samples into a WAV byte buffer (16-bit, mono).
///
/// Returns an empty `Vec` and logs an error if the `hound` writer fails.
/// This is intentionally the same implementation as the former
/// `transport/audio.rs::encode_wav` so that callers get byte-identical output.
#[cfg(feature = "transport-audio")]
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut buf = Vec::<u8>::new();
    {
        let cursor = std::io::Cursor::new(&mut buf);
        let mut writer = match hound::WavWriter::new(cursor, spec) {
            Ok(w) => w,
            Err(e) => {
                error!("voice::wav: WAV writer init failed: {e}");
                return Vec::new();
            }
        };
        for &s in samples {
            let s16 = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            if writer.write_sample(s16).is_err() {
                break;
            }
        }
        // finalize() writes the correct RIFF/data header sizes.
        if let Err(e) = writer.finalize() {
            error!("voice::wav: WAV finalize failed: {e}");
            return Vec::new();
        }
    }
    buf
}

#[cfg(all(test, feature = "transport-audio"))]
mod tests {
    use super::*;

    #[test]
    fn encode_wav_produces_riff_header() {
        let samples: Vec<f32> = vec![0.0; 512];
        let wav = encode_wav(&samples, 16_000);
        // Minimal sanity: non-empty and starts with RIFF magic.
        assert!(!wav.is_empty());
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
    }

    #[test]
    fn encode_wav_empty_samples() {
        let wav = encode_wav(&[], 16_000);
        // An empty WAV is valid — hound writes a correct zero-length file.
        assert!(
            !wav.is_empty(),
            "hound writes a valid header even for 0 samples"
        );
        assert_eq!(&wav[0..4], b"RIFF");
    }

    #[test]
    fn encode_wav_clamps_samples() {
        // Values outside [-1.0, 1.0] must not panic.
        let samples = vec![-2.0f32, 2.0f32, 0.5f32];
        let wav = encode_wav(&samples, 16_000);
        assert!(!wav.is_empty());
    }
}
