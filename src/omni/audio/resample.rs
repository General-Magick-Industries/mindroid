//! Integer-factor sample-rate reduction with an anti-alias filter.

use std::f64::consts::PI;

use thiserror::Error;

/// Filter length per unit of decimation factor. 32 gives a transition band
/// narrow enough that the stop band starts just above the output's Nyquist
/// frequency, at 97 taps for 48 → 16 kHz.
const TAPS_PER_FACTOR: usize = 32;

/// Pass-band edge as a fraction of the output's Nyquist frequency.
const CUTOFF: f64 = 0.9;

/// Why a [`Resampler`] cannot convert between two rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ResampleError {
    #[error("0 Hz cannot be resampled")]
    ZeroRate,
    #[error("{from} Hz is not an integer multiple of {to} Hz")]
    NotIntegerMultiple { from: u32, to: u32 },
}

/// Lowers mono PCM16 from one rate to another that divides it exactly.
///
/// A Blackman-windowed sinc low-pass removes what the output rate cannot carry
/// before samples are dropped, so it is filtered out instead of folding back
/// into the audible band. State carries across [`process`](Self::process)
/// calls: a stream fed in chunks of any size yields the same samples as the
/// whole stream fed at once. The output lags the input by `16 * factor` input
/// samples (1 ms at 48 → 16 kHz).
#[derive(Debug, Clone, PartialEq)]
pub struct Resampler {
    from: u32,
    factor: usize,
    taps: Vec<f32>,
    buf: Vec<f32>,
    next: usize,
}

impl Resampler {
    /// A resampler from `from` Hz down to `to` Hz.
    ///
    /// # Errors
    ///
    /// [`ResampleError::ZeroRate`] when either rate is zero, and
    /// [`ResampleError::NotIntegerMultiple`] when `from` is not an integer
    /// multiple of `to` — which includes every `from < to`, since only
    /// downsampling is supported.
    pub fn new(from: u32, to: u32) -> Result<Self, ResampleError> {
        if from == 0 || to == 0 {
            return Err(ResampleError::ZeroRate);
        }
        if !from.is_multiple_of(to) {
            return Err(ResampleError::NotIntegerMultiple { from, to });
        }
        let factor = (from / to) as usize;
        let taps = low_pass(factor);
        let len = taps.len();
        Ok(Self {
            from,
            factor,
            taps,
            buf: vec![0.0; len - 1],
            next: len,
        })
    }

    /// The input rate this resampler was built for.
    pub fn from_rate(&self) -> u32 {
        self.from
    }

    /// Filter `input` and append the samples it completes at the output rate
    /// to `out`. Over a stream, `n` input samples yield `n.div_ceil(factor)`
    /// output samples.
    pub fn process(&mut self, input: impl IntoIterator<Item = i16>, out: &mut Vec<i16>) {
        if self.factor == 1 {
            out.extend(input);
            return;
        }
        self.buf.extend(input.into_iter().map(f32::from));
        let len = self.taps.len();
        let ends = (self.next..self.buf.len() + 1).step_by(self.factor);
        let produced = ends.len();
        let (buf, taps) = (&self.buf, &self.taps);
        out.extend(ends.map(|end| {
            let y: f32 = buf[end - len..end]
                .iter()
                .zip(taps)
                .map(|(x, h)| x * h)
                .sum();
            y.round() as i16
        }));
        self.next += produced * self.factor;
        let consumed = self.next - len;
        self.buf.drain(..consumed);
        self.next -= consumed;
    }

    /// Forget the filter history, as if no audio had been processed.
    pub fn reset(&mut self) {
        let len = self.taps.len();
        self.buf.clear();
        self.buf.resize(len - 1, 0.0);
        self.next = len;
    }
}

/// Blackman-windowed sinc with unity DC gain, cut off at [`CUTOFF`] of the
/// output's Nyquist frequency.
fn low_pass(factor: usize) -> Vec<f32> {
    if factor == 1 {
        return vec![1.0];
    }
    let len = TAPS_PER_FACTOR * factor + 1;
    let span = (len - 1) as f64;
    let fc = CUTOFF / (2.0 * factor as f64);
    let raw: Vec<f64> = (0..len)
        .map(|n| {
            let t = n as f64 - span / 2.0;
            let sinc = if t == 0.0 {
                2.0 * fc
            } else {
                (2.0 * PI * fc * t).sin() / (PI * t)
            };
            let w = n as f64 / span;
            sinc * (0.42 - 0.5 * (2.0 * PI * w).cos() + 0.08 * (4.0 * PI * w).cos())
        })
        .collect();
    let gain: f64 = raw.iter().sum();
    raw.iter().map(|h| (h / gain) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tones(rate: u32, len: usize, parts: &[(f64, f64)]) -> Vec<i16> {
        (0..len)
            .map(|n| {
                let t = n as f64 / f64::from(rate);
                let v: f64 = parts
                    .iter()
                    .map(|(freq, amp)| amp * (2.0 * PI * freq * t).sin())
                    .sum();
                (v * f64::from(i16::MAX)).round() as i16
            })
            .collect()
    }

    /// Amplitude of each DFT bin from DC to Nyquist, as a fraction of full scale.
    fn spectrum(x: &[i16]) -> Vec<f64> {
        let len = x.len() as f64;
        (0..=x.len() / 2)
            .map(|k| {
                let (re, im) = x.iter().enumerate().fold((0.0, 0.0), |(re, im), (n, &s)| {
                    let phase = 2.0 * PI * k as f64 * n as f64 / len;
                    let s = f64::from(s);
                    (re + s * phase.cos(), im - s * phase.sin())
                });
                2.0 * re.hypot(im) / len / f64::from(i16::MAX)
            })
            .collect()
    }

    fn resample(input: &[i16], from: u32, to: u32) -> Vec<i16> {
        let mut r = Resampler::new(from, to).unwrap();
        let mut out = Vec::new();
        r.process(input.iter().copied(), &mut out);
        out
    }

    /// A lone 1 kHz tone has nothing above 8 kHz to alias, so it cannot show
    /// whether the filter works. A 12 kHz partner can: dropping samples without
    /// filtering folds it onto 4 kHz at full strength.
    #[test]
    fn a_1khz_tone_resampled_48k_to_16k_carries_nothing_else() {
        let input = tones(48_000, 48_000, &[(1_000.0, 0.25), (12_000.0, 0.25)]);
        // 1600 samples at 16 kHz put both 1 kHz and the 4 kHz alias on exact bins.
        let window = 160..1_760;
        let (tone_bin, alias_bin) = (100, 400);
        let floor = 0.25 * 10f64.powf(-60.0 / 20.0);

        let naive: Vec<i16> = input.iter().copied().step_by(3).collect();
        assert!(
            spectrum(&naive[window.clone()])[alias_bin] > 0.2,
            "without a filter the 12 kHz tone folds onto 4 kHz"
        );

        let out = resample(&input, 48_000, 16_000);
        let bins = spectrum(&out[window]);
        let gain_db = 20.0 * (bins[tone_bin] / 0.25).log10();
        assert!(gain_db.abs() < 0.1, "1 kHz passes at {gain_db:.3} dB");
        for (k, &amp) in bins.iter().enumerate().filter(|&(k, _)| k != tone_bin) {
            assert!(
                amp < floor,
                "bin {k} ({} Hz) holds {amp:.2e}, less than 60 dB below the tone",
                k * 10
            );
        }
    }

    #[test]
    fn chunked_input_matches_one_shot() {
        let input = tones(48_000, 9_600, &[(440.0, 0.5), (9_000.0, 0.3)]);
        let whole = resample(&input, 48_000, 16_000);

        let mut r = Resampler::new(48_000, 16_000).unwrap();
        let mut chunked = Vec::new();
        let mut rest = input.as_slice();
        for size in [1, 2, 7, 480, 3, 1_001, 64].into_iter().cycle() {
            if rest.is_empty() {
                break;
            }
            let (head, tail) = rest.split_at(size.min(rest.len()));
            r.process(head.iter().copied(), &mut chunked);
            rest = tail;
        }
        assert_eq!(chunked, whole);
    }

    #[test]
    fn output_length_is_the_input_over_the_factor() {
        assert_eq!(resample(&[0; 4_800], 48_000, 16_000).len(), 1_600);
        assert_eq!(resample(&[0; 4_801], 48_000, 24_000).len(), 2_401);
    }

    #[test]
    fn dc_passes_at_unity_gain() {
        let out = resample(&[1_000; 960], 48_000, 16_000);
        assert!(out[32..].iter().all(|&s| (999..=1_001).contains(&s)));
    }

    #[test]
    fn equal_rates_pass_through() {
        let input = tones(16_000, 160, &[(1_000.0, 0.5)]);
        assert_eq!(resample(&input, 16_000, 16_000), input);
    }

    #[test]
    fn reset_forgets_history() {
        let input = tones(48_000, 300, &[(1_000.0, 0.5)]);
        let mut r = Resampler::new(48_000, 16_000).unwrap();
        let mut first = Vec::new();
        r.process(input.iter().copied(), &mut first);
        r.reset();
        let mut second = Vec::new();
        r.process(input.iter().copied(), &mut second);
        assert_eq!(first, second);
    }

    #[test]
    fn rates_that_do_not_divide_are_refused() {
        assert_eq!(
            Resampler::new(44_100, 24_000),
            Err(ResampleError::NotIntegerMultiple {
                from: 44_100,
                to: 24_000
            })
        );
        assert!(Resampler::new(24_000, 16_000).is_err());
        assert!(Resampler::new(16_000, 48_000).is_err());
        assert_eq!(Resampler::new(0, 16_000), Err(ResampleError::ZeroRate));
        assert_eq!(Resampler::new(16_000, 0), Err(ResampleError::ZeroRate));
    }
}
