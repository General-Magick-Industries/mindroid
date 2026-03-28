use crate::{Pipeline};
use crate::llm_client::{AuthStyle, LlmClient, LlmClientConfig};
use crate::pipeline::stages::{SimpleContextBuilder, PostProcessor, GenericLlmProcessor};

pub fn ollama_pipeline(base_url: &str, model: &str) -> crate::Result<Pipeline> {
    let mut config = LlmClientConfig::new(format!("{base_url}/v1"));
    config.auth_style = AuthStyle::None;
    config.default_model = Some(model.to_string());
    let client = LlmClient::new(config)?;

    Ok(Pipeline::new()
        .add_stage(SimpleContextBuilder::new())
        .add_streaming_stage(GenericLlmProcessor::new(client))
        .add_stage(PostProcessor))
}
