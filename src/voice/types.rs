use std::time::Duration;

#[derive(Debug, Clone)]
pub enum TurnDetection {
    Server,
    Local(VadConfig),
    Manual,
}

#[derive(Debug, Clone)]
pub enum BargeInMode {
    LocalVad,
    ServerOnly,
    Disabled,
}

#[derive(Debug, Clone)]
pub struct VadConfig {
    pub speech_threshold: f32,
    pub speech_end_threshold: f32,
    pub silence_duration: Duration,
    pub speech_pad: Duration,
    /// Minimum utterance length to forward. Shorter segments are discarded as noise.
    /// Default: 300ms.
    pub min_speech: Duration,
    pub max_utterance: Duration,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            speech_threshold: 0.5,
            speech_end_threshold: 0.3,
            silence_duration: Duration::from_millis(500),
            speech_pad: Duration::from_millis(300),
            min_speech: Duration::from_millis(300),
            max_utterance: Duration::from_secs(30),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn vad_config_default() {
        let cfg = VadConfig::default();
        assert_eq!(cfg.speech_threshold, 0.5);
        assert_eq!(cfg.speech_end_threshold, 0.3);
        assert_eq!(cfg.silence_duration, Duration::from_millis(500));
        assert_eq!(cfg.speech_pad, Duration::from_millis(300));
        assert_eq!(cfg.min_speech, Duration::from_millis(300));
        assert_eq!(cfg.max_utterance, Duration::from_secs(30));
    }

    #[test]
    fn vad_config_min_speech_field() {
        // Verify min_speech can be overridden independently.
        let cfg = VadConfig {
            min_speech: Duration::from_millis(500),
            ..VadConfig::default()
        };
        assert_eq!(cfg.min_speech, Duration::from_millis(500));
        // Other fields unchanged.
        assert_eq!(cfg.speech_threshold, 0.5);
        assert_eq!(cfg.silence_duration, Duration::from_millis(500));
    }

    #[test]
    fn turn_detection_local_vad() {
        let vad = VadConfig::default();
        let td = TurnDetection::Local(vad);
        let _cloned = td.clone();
        assert!(matches!(td, TurnDetection::Local(_)));
    }
}
