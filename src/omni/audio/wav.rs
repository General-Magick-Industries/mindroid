//! PCM16 WAV encoding: a canonical 44-byte header followed by the samples.

/// Wrap little-endian PCM16 in a WAV container.
///
/// `pcm` is interleaved when `channels > 1`; a channel count of 0 is treated as
/// 1. A trailing partial frame is dropped so the data chunk holds whole frames.
///
/// # Panics
///
/// If `pcm` is larger than a WAV file can describe (about 4 GiB).
pub fn encode_pcm16(pcm: &[u8], sample_rate: u32, channels: u16) -> Vec<u8> {
    let channels = channels.max(1);
    let pcm = &pcm[..pcm.len() - pcm.len() % (usize::from(channels) * 2)];
    let mut wav = Vec::with_capacity(HEADER_LEN + pcm.len());
    write_header(&mut wav, pcm.len(), sample_rate, channels);
    wav.extend_from_slice(pcm);
    wav
}

/// Quantize mono samples in `[-1.0, 1.0]` to PCM16 and encode them as a WAV.
/// Samples outside that range are clamped.
///
/// # Panics
///
/// If `samples` is larger than a WAV file can describe (about 2^31 samples).
pub fn encode_mono_f32(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let mut wav = Vec::with_capacity(HEADER_LEN + samples.len() * 2);
    write_header(&mut wav, samples.len() * 2, sample_rate, 1);
    wav.extend(
        samples
            .iter()
            .flat_map(|s| ((s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16).to_le_bytes()),
    );
    wav
}

const HEADER_LEN: usize = 44;

fn write_header(wav: &mut Vec<u8>, data_len: usize, sample_rate: u32, channels: u16) {
    let data_len = u32::try_from(data_len)
        .ok()
        .filter(|n| *n <= u32::MAX - 36)
        .expect("PCM data exceeds the 4 GiB a WAV header can describe");
    let block = channels * 2;
    for part in [
        b"RIFF".as_slice(),
        &(36 + data_len).to_le_bytes(),
        b"WAVEfmt ",
        &16u32.to_le_bytes(),
        &1u16.to_le_bytes(),
        &channels.to_le_bytes(),
        &sample_rate.to_le_bytes(),
        &sample_rate.saturating_mul(u32::from(block)).to_le_bytes(),
        &block.to_le_bytes(),
        &16u16.to_le_bytes(),
        b"data",
        &data_len.to_le_bytes(),
    ] {
        wav.extend_from_slice(part);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_the_canonical_pcm_layout() {
        let wav = encode_pcm16(&[1, 0, 2, 0], 16_000, 1);
        let expected: Vec<u8> = [
            b"RIFF".as_slice(),
            &40u32.to_le_bytes(),
            b"WAVEfmt ",
            &16u32.to_le_bytes(),
            &1u16.to_le_bytes(),
            &1u16.to_le_bytes(),
            &16_000u32.to_le_bytes(),
            &32_000u32.to_le_bytes(),
            &2u16.to_le_bytes(),
            &16u16.to_le_bytes(),
            b"data",
            &4u32.to_le_bytes(),
            &[1, 0, 2, 0],
        ]
        .concat();
        assert_eq!(wav, expected);
    }

    #[test]
    fn stereo_drops_a_trailing_partial_frame() {
        let wav = encode_pcm16(&[1, 0, 2, 0, 3, 0], 48_000, 2);
        assert_eq!(wav.len(), HEADER_LEN + 4);
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 2);
        assert_eq!(u32::from_le_bytes(wav[28..32].try_into().unwrap()), 192_000);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 4);
    }

    #[test]
    fn empty_input_is_a_valid_empty_wav() {
        let wav = encode_mono_f32(&[], 16_000);
        assert_eq!(wav.len(), HEADER_LEN);
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 36);
    }

    #[test]
    fn f32_samples_are_clamped_and_quantized() {
        let wav = encode_mono_f32(&[-2.0, 2.0, 0.5, 0.0], 16_000);
        let samples: Vec<i16> = wav[HEADER_LEN..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i16::from_le_bytes(*b))
            .collect();
        assert_eq!(samples, [-32_767, 32_767, 16_383, 0]);
        assert_eq!(
            wav[..HEADER_LEN],
            encode_pcm16(&[0; 8], 16_000, 1)[..HEADER_LEN]
        );
    }
}
