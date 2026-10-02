use crate::Pipeline;
use crate::llm_client::LlmClient;
use crate::pipeline::stages::{
    GenericLlmProcessor, IngestStage, PostProcessor, SimpleContextBuilder,
};

/// Build a vision-capable pipeline from a pre-configured [`LlmClient`].
///
/// The caller is responsible for configuring the client with a vision-capable
/// model (e.g. `llava`, `gpt-4o`, `claude-3-5-sonnet`) and any required auth
/// before passing it in.
///
/// Stages: `SimpleContextBuilder` → `IngestStage` → `GenericLlmProcessor`
/// (streaming) → `PostProcessor`.
///
/// `SimpleContextBuilder` builds the text messages; `IngestStage` appends the
/// image (if any) to the user message — the two compose modularly so the same
/// pipeline serves plain chat and image turns.
///
/// Provide the image to analyze by setting the [`crate::FileInput`] extension
/// on the context before running; the text prompt defaults to the incoming
/// `message.content`. Inline image bytes require the `transport-ws` feature for
/// base64 encoding — otherwise pass a hosted image URL via a custom builder.
///
/// # Example
///
/// ```ignore
/// use mindroid::llm_client::{AuthStyle, LlmClient, LlmClientConfig};
/// use mindroid::{FileInput, FileInputs, vision_pipeline};
///
/// let mut config = LlmClientConfig::new("http://localhost:11434/v1");
/// config.auth_style = AuthStyle::None;
/// config.default_model = Some("llava".to_string());
/// let client = LlmClient::new(config)?;
///
/// let pipeline = vision_pipeline(client)?;
///
/// ctx.set_ext(FileInputs(vec![FileInput::image(png_bytes, "image/png")]));
/// let answer = pipeline.run(&mut ctx).await?;
/// ```
pub fn vision_pipeline(client: LlmClient) -> crate::Result<Pipeline> {
    Ok(Pipeline::new()
        .add_stage(SimpleContextBuilder::new())
        .add_stage(IngestStage::default_media())
        .add_streaming_stage(GenericLlmProcessor::new(client))
        .add_stage(PostProcessor))
}
