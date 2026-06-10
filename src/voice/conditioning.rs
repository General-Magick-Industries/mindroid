//! Capture conditioning seam — AGC / denoise pre-processing (task 8 / task 13).
//!
//! # Design
//!
//! [`CaptureConditioner`] is the seam for audio pre-processing (automatic gain
//! control, noise suppression, etc.) that logically runs at the **caller's
//! inference edge** — i.e. before the caller computes the VAD probability and
//! before calling [`AudioFrontend::process`].  The trait is defined here so that
//! the frontend can accept a `Box<dyn CaptureConditioner>` and the real
//! implementation (task 13) can slot in without touching the core orchestrator.
//!
//! Task 8 ships only the trait + the no-op [`PassthroughConditioner`].  When a
//! device or browser client has already conditioned the stream upstream it can
//! call [`AudioFrontend::mark_conditioning_done`] to let the frontend know;
//! that is purely informational and does not change any processing path inside
//! the frontend.
//!
//! ## Why conditioning is NOT inside the frontend
//!
//! The frontend is a *probabilities-in* component: it consumes pre-computed VAD
//! probabilities (offloaded ONNX inference) and raw f32 PCM for accumulation.
//! Conditioning (which operates on i16 PCM before resampling / inference) sits
//! one step earlier in the pipeline.  Placing it here would couple I/O concerns
//! into the pure state machine.

// ─── CaptureConditioner ───────────────────────────────────────────────────────

/// Pre-processing seam applied to raw captured audio before VAD inference.
///
/// Implementations should process the frame **in-place**.  The default
/// no-op implementation ([`PassthroughConditioner`]) leaves samples unchanged.
///
/// # Thread safety
///
/// All implementations must be [`Send`] so they can be moved to a capture
/// thread.
pub trait CaptureConditioner: Send {
    /// Process one raw PCM frame in-place.
    ///
    /// `frame` — 16-bit PCM samples for one audio chunk (e.g. 512 samples at
    /// 16 kHz → 32 ms).  Modify in-place to apply AGC, noise suppression, etc.
    fn process(&mut self, frame: &mut [i16]);
}

// ─── PassthroughConditioner ───────────────────────────────────────────────────

/// No-op [`CaptureConditioner`] that leaves audio samples unmodified.
///
/// This is the default implementation used by [`AudioFrontend`] when no real
/// conditioner has been provided.  The real AGC/denoise implementation ships
/// with task 13.
///
/// [`AudioFrontend`]: crate::voice::frontend::AudioFrontend
#[derive(Debug, Default)]
pub struct PassthroughConditioner;

impl CaptureConditioner for PassthroughConditioner {
    /// No-op: returns immediately without modifying `frame`.
    fn process(&mut self, _frame: &mut [i16]) {}
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_leaves_samples_unchanged() {
        let mut cond = PassthroughConditioner;
        let original = vec![100i16, -200, 0, 32767, -32768];
        let mut frame = original.clone();
        cond.process(&mut frame);
        assert_eq!(frame, original);
    }

    #[test]
    fn passthrough_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<PassthroughConditioner>();
    }

    #[test]
    fn boxed_trait_object_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Box<dyn CaptureConditioner>>();
    }
}
