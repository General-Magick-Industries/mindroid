mod context;
mod post_processor;
#[cfg(feature = "llm-client")]
mod llm_processor;
#[cfg(feature = "llm-client")]
pub mod gate;
pub mod stt;
pub mod tts;
#[cfg(feature = "llm-client")]
mod tool_executor;

pub use context::SimpleContextBuilder;
pub use post_processor::PostProcessor;
pub use stt::{SttProvider, SttStage};
pub use tts::{TtsProvider, TtsStage};
#[cfg(feature = "transport-audio")]
pub use tts::AudioOutputStage;
#[cfg(feature = "transport-audio")]
pub use tts::StreamingTtsTransformer;
#[cfg(feature = "llm-client")]
pub use stt::OpenAiStt;
#[cfg(feature = "llm-client")]
pub use stt::OpenAiSttConfig;
#[cfg(feature = "speech")]
pub use stt::DeepgramStt;
#[cfg(feature = "speech")]
pub use stt::DeepgramSttConfig;
#[cfg(feature = "llm-client")]
pub use tts::OpenAiTts;
#[cfg(feature = "llm-client")]
pub use tts::OpenAiTtsConfig;
#[cfg(feature = "speech")]
pub use tts::DeepgramTts;
#[cfg(feature = "speech")]
pub use tts::DeepgramTtsConfig;
#[cfg(feature = "llm-client")]
pub use llm_processor::{GenericLlmProcessor, collect_stream};
#[cfg(feature = "llm-client")]
pub use gate::{AndGate, CoordinationGate, Gate, OrGate, RelevanceGate};
#[cfg(feature = "llm-client")]
pub use tool_executor::{ToolExecutorStage, ToolCallParser, ParsedToolCall, XmlToolCallParser};
