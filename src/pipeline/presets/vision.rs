use crate::Pipeline;
use crate::llm_client::LlmClient;
use crate::pipeline::stages::{GenericLlmProcessor, PostProcessor, SimpleContextBuilder};

/// Build a vision-capable pipeline from a pre-configured [`LlmClient`].
///
/// The caller is responsible for configuring the client with a vision-capable
/// model (e.g. `llava`, `gpt-4o`, `claude-3-5-sonnet`) and any required auth
/// before passing it in.
///
/// Stages: `SimpleContextBuilder` → `GenericLlmProcessor` (streaming) → `PostProcessor`.
///
/// # Example
///
/// ```ignore
/// use mindroid::llm_client::{AuthStyle, LlmClient, LlmClientConfig};
/// use mindroid::pipeline::presets::vision::vision_pipeline;
///
/// let mut config = LlmClientConfig::new("http://localhost:11434/v1");
/// config.auth_style = AuthStyle::None;
/// config.default_model = Some("llava".to_string());
/// let client = LlmClient::new(config)?;
///
/// let pipeline = vision_pipeline(client)?;
/// ```
pub fn vision_pipeline(client: LlmClient) -> crate::Result<Pipeline> {
    Ok(Pipeline::new()
        .add_stage(SimpleContextBuilder::new())
        .add_streaming_stage(GenericLlmProcessor::new(client))
        .add_stage(PostProcessor))
}
