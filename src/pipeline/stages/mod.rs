mod context;
#[cfg(feature = "llm-client")]
pub mod corpus_gate;
#[cfg(all(feature = "corpus", feature = "llm-client"))]
pub mod corpus_distill;
#[cfg(feature = "llm-client")]
pub mod gate;
#[cfg(feature = "llm-client")]
mod llm_processor;
mod post_processor;
pub mod stt;
#[cfg(feature = "llm-client")]
mod tool_executor;
pub mod tts;

pub use context::SimpleContextBuilder;
#[cfg(feature = "llm-client")]
pub use gate::{AndGate, CoordinationGate, Gate, OrGate, RelevanceGate};
#[cfg(feature = "llm-client")]
pub use llm_processor::{GenericLlmProcessor, collect_stream};
pub use post_processor::PostProcessor;
#[cfg(feature = "speech")]
pub use stt::DeepgramStt;
#[cfg(feature = "speech")]
pub use stt::DeepgramSttConfig;
#[cfg(feature = "llm-client")]
pub use stt::OpenAiStt;
#[cfg(feature = "llm-client")]
pub use stt::OpenAiSttConfig;
pub use stt::{SttProvider, SttStage};
#[cfg(feature = "llm-client")]
pub use tool_executor::{ParsedToolCall, ToolCallParser, ToolExecutorStage, XmlToolCallParser};
#[cfg(feature = "transport-audio")]
pub use tts::AudioOutputStage;
#[cfg(feature = "speech")]
pub use tts::DeepgramTts;
#[cfg(feature = "speech")]
pub use tts::DeepgramTtsConfig;
#[cfg(feature = "llm-client")]
pub use tts::OpenAiTts;
#[cfg(feature = "llm-client")]
pub use tts::OpenAiTtsConfig;
#[cfg(feature = "transport-audio")]
pub use tts::StreamingTtsTransformer;
pub use tts::{TtsProvider, TtsStage};
#[cfg(feature = "llm-client")]
pub use corpus_gate::{CorpusGateDecision, CorpusGateStage};
#[cfg(all(feature = "corpus", feature = "llm-client"))]
pub use corpus_distill::CorpusDistillStage;
