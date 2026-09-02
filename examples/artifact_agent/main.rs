//! Artifact agent — the idiomatic, config-driven shape.
//!
//! Unlike `artifact_permutations` (a bespoke REPL that hand-wires everything),
//! this example is built the way mindroid intends:
//!
//!   * `MindroidConfig` + `Runtime::from_config` own transport, auth, and the
//!     persistence backends — you describe them in `config.toml`, not in code.
//!   * `[memory] type = "sqlite"` → conversation history persists across runs.
//!   * `[artifacts] path = ...` → image bytes are offloaded to disk; chat history
//!     keeps only a compact `get_artifact` reference.
//!
//! `build_memory_client()` surfaces the config-built SQLite memory for the pipeline.
//!
//! ## Custom store: store-defined metadata
//!
//! Rather than the config-built plain store, this example wires a
//! [`DescribedLocalStore`] — a custom `ArtifactStore` that wraps the built-in
//! `LocalArtifactStore` (the decorator pattern) and, on `save`, attaches metadata
//! describing the storage: `{ "type": "locally saved artifact", "directory": ... }`.
//! That metadata rides along on the offloaded `File` reference and is persisted in
//! history — a small demo of how a store returns more than a bare id, with no
//! framework change.
//!
//! Per turn the pipeline is:
//!   SimpleContextBuilder(history) → IngestStage → XmlToolExecutorStage(get_artifact)
//!     → PostProcessor → ArtifactOffload
//! then the turn (with any artifact reference) is saved back to SQLite.
//!
//! Run (stdio transport — type a message, Enter):
//!   cargo run -p mindroid-example-artifact-agent -- --config examples/artifact_agent/config.toml
//!
//! Attach a generated image frame with:  /snap <your question>
//! History and offloaded references survive a restart (same SQLite db + store dir).
//!
//! `/snap` is just this example's way of producing image bytes. Attachment is
//! generic: anything that sets the `FileInputs` extension on the turn's
//! `PipelineContext` — a transport, an upload handler, a file reader, another
//! capture device — flows through the same IngestStage → ArtifactOffload chain.

use std::sync::Arc;

use anyhow::{Context as _, Result};
use async_trait::async_trait;
use mindroid::artifacts::{
    Artifact, ArtifactManager, ArtifactStore, LocalArtifactStore, StoredArtifact,
};
use mindroid::core::content::ContentMetadata;
use mindroid::llm_client::LlmClient;
use mindroid::models::Role;
use mindroid::pipeline::extensions::{FileInput, FileInputs};
use mindroid::pipeline::presets::memory::MemoryClient;
use mindroid::pipeline::stages::{
    ArtifactOffload, GenericLlmProcessor, IngestStage, PostProcessor, SimpleContextBuilder,
    XmlToolExecutorStage,
};
use mindroid::tools::{GetArtifactTool, ToolRegistry};
use mindroid::{MindroidConfig, Pipeline, PipelineContext, Runtime};

const SCOPE: &str = "stdio";

/// Command that triggers a frame snap. Text after it is an optional prompt;
/// `/snap` alone sends just the image (no text), like attaching an image with
/// no message.
const SNAP_CMD: &str = "/snap";

/// Strip the `/snap` command from a line, returning the optional trailing prompt.
/// Returns `Some(prompt)` when the line is a snap command, `None` otherwise.
fn snap_prompt(content: &str) -> Option<&str> {
    let rest = content.strip_prefix(SNAP_CMD)?;
    // Must be exactly `/snap` or `/snap` followed by whitespace — not `/snapfoo`.
    if rest.is_empty() || rest.starts_with(char::is_whitespace) {
        Some(rest.trim())
    } else {
        None
    }
}

const HISTORY_LIMIT: usize = 30;

/// A custom `ArtifactStore` demonstrating store-defined metadata.
///
/// It wraps the built-in [`LocalArtifactStore`] (the *decorator* pattern — reuse
/// all its disk logic) and, on `save`, attaches metadata describing *how* and
/// *where* the artifact was stored:
///
/// ```json
/// "metadata": { "type": "locally saved artifact", "directory": "<base>/<scope>" }
/// ```
///
/// That metadata rides along on the `File` reference and is persisted in history —
/// nothing in the framework had to change; the store simply chose what to return.
struct DescribedLocalStore {
    inner: LocalArtifactStore,
    base: String,
}

impl DescribedLocalStore {
    fn new(base: impl Into<String>) -> Self {
        let base = base.into();
        Self {
            inner: LocalArtifactStore::new(&base),
            base,
        }
    }
}

#[async_trait]
impl ArtifactStore for DescribedLocalStore {
    async fn save(
        &self,
        scope: &str,
        data: &[u8],
        mime_type: &str,
    ) -> Result<StoredArtifact, mindroid::MindroidError> {
        // Delegate the actual disk write to the wrapped local store...
        let stored = self.inner.save(scope, data, mime_type).await?;
        // ...then attach our own metadata describing the storage.
        //
        // Metadata is visible to the model by default. Prefix a key with `_` to keep
        // it code-only — here `type` is shown to the model, while `_directory`
        // (a filesystem path — noise to the LLM) stays hidden.
        let mut metadata = ContentMetadata::new();
        metadata.insert("type".into(), "locally saved artifact".into());
        metadata.insert("_directory".into(), format!("{}/{scope}", self.base).into());
        Ok(stored.with_metadata(metadata))
    }

    async fn load(&self, scope: &str, id: &str) -> Result<Artifact, mindroid::MindroidError> {
        self.inner.load(scope, id).await
    }

    async fn delete(&self, scope: &str, id: &str) -> Result<(), mindroid::MindroidError> {
        self.inner.delete(scope, id).await
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("warn,mindroid::observer=info")
        .init();

    let config = MindroidConfig::resolve_from_args()?;
    let agent_id = config.agent.agent_id.clone();
    // Capture the artifact store path before `from_config` consumes the config;
    // the custom DescribedLocalStore is built from it below.
    let artifact_path = config.artifacts.path.clone();
    let system_prompt =
        "You are a helpful assistant that can see images the user attaches. When the user \
         references an attached image you can't see, call get_artifact with its id to view it."
            .to_string();

    // Resolve the LLM client from [models.respond] + [providers.*].
    let llm_config = config
        .llm("respond")
        .context("config needs [models.respond] + [providers.*]")?;
    let client = LlmClient::new(llm_config)?;

    // Runtime::from_config builds transport (stdio), auth, memory, observers.
    let builder = Runtime::from_config(config)?;

    // The new helpers surface the config-built persistence pieces for the pipeline.
    let memory: Arc<MemoryClient> = builder
        .build_memory_client()?
        .context("config needs [memory] type = \"sqlite\" (or another backend)")?;
    // Instead of the config-built plain store (via `builder.build_artifact_manager()`),
    // wire our custom DescribedLocalStore so offloaded artifacts carry
    // { type, directory } metadata. The manager is cloneable — re-mint the offload
    // stage + load tool from it each turn.
    let artifacts: Option<ArtifactManager> = artifact_path.map(|base| {
        let store: Arc<dyn ArtifactStore> = Arc::new(DescribedLocalStore::new(base));
        ArtifactManager::new(store)
    });

    let client = Arc::new(client);
    let system_prompt = Arc::new(system_prompt);
    let agent_id = Arc::new(agent_id);

    let mut runtime = builder
        .on_message(move |ctx| {
            let client = Arc::clone(&client);
            let memory = Arc::clone(&memory);
            let system_prompt = Arc::clone(&system_prompt);
            let agent_id = Arc::clone(&agent_id);
            // Cloning the manager is cheap (shares one Arc store).
            let artifact_mgr = artifacts.clone();

            async move {
                let channel = ctx.message.channel_id.clone();

                // 0. CAPTURE — `/snap <prompt>` generates a frame for this turn.
                //    The frame is attached as a FileInputs extension below; the
                //    message content is rewritten to just the prompt text.
                let snapped: Option<(Vec<u8>, String)> = snap_prompt(&ctx.message.content)
                    .and_then(|_| {
                        println!("(generating a frame...)");
                        match capture_frame() {
                            Ok((bytes, mime)) => {
                                println!("(captured {} bytes, {mime})", bytes.len());
                                Some((bytes, mime))
                            }
                            Err(e) => {
                                eprintln!("frame capture failed: {e}");
                                None
                            }
                        }
                    });

                // 1. READ — load history from SQLite (artifact refs round-trip via from_stored).
                let history = match memory
                    .prepare_context(&channel, &agent_id, HISTORY_LIMIT)
                    .await
                {
                    Ok(h) => Arc::new(h),
                    Err(e) => {
                        tracing::error!("history load failed: {e}");
                        Arc::new(Vec::new())
                    }
                };

                // 2. BUILD the per-turn pipeline with history injected.
                let base = Pipeline::new()
                    .add_stage(SimpleContextBuilder::with_prompt_and_history(
                        system_prompt.as_str(),
                        history.clone(),
                    ))
                    .add_stage(IngestStage::default_media());

                // With an artifact store: tool loop (get_artifact) + offload after.
                // Without: a plain LLM processor. Both re-minted from the manager.
                let pipeline = if let Some(mgr) = &artifact_mgr {
                    let tool = GetArtifactTool::from_manager(mgr.clone(), SCOPE);
                    let registry = Arc::new(ToolRegistry::new().register(tool));
                    base.add_streaming_stage(XmlToolExecutorStage::new((*client).clone(), registry))
                        .add_stage(PostProcessor)
                        .add_stage(ArtifactOffload::from_manager(mgr.clone()))
                } else {
                    base.add_streaming_stage(GenericLlmProcessor::new((*client).clone()))
                        .add_stage(PostProcessor)
                };

                // The turn's message — rewritten to drop the `/snap ` prefix when a
                // frame was captured, so the LLM sees only the prompt text.
                let turn_message = if snapped.is_some() {
                    // Rewrite content to just the prompt (may be empty → image-only turn).
                    let prompt = snap_prompt(&ctx.message.content).unwrap_or("").to_string();
                    let mut m = (*ctx.message).clone();
                    m.content = prompt;
                    Arc::new(m)
                } else {
                    ctx.message.clone()
                };

                let mut pctx = PipelineContext::new(turn_message, ctx.agent_config.clone());

                // Attach the captured frame so IngestStage → ArtifactOffload offloads it.
                if let Some((data, mime)) = snapped {
                    pctx.set_ext(FileInputs::one(FileInput::image(data, mime)));
                }

                // `run` returns the response via Option (it `take()`s ctx.response),
                // so capture it here rather than re-reading pctx.response after.
                let answer = match ctx.run_with_context(&pipeline, &mut pctx).await {
                    Ok(resp) => resp.unwrap_or_default().trim().to_string(),
                    Err(e) => {
                        tracing::error!("pipeline error: {e}");
                        return;
                    }
                };

                // 3. SAVE — persist the (possibly offloaded) user turn + the answer.
                //    `to_stored()` writes the artifact reference as JSON, not bytes.
                let user_content = pctx
                    .llm_messages
                    .iter()
                    .rev()
                    .find(|m| m.role == Role::User)
                    .map(|m| m.to_stored())
                    .unwrap_or_else(|| ctx.message.content.clone());
                if let Err(e) = memory
                    .save_message(&channel, &ctx.message.sender_id, &user_content, None)
                    .await
                {
                    tracing::error!("save user failed: {e}");
                }
                if !answer.is_empty() {
                    let _ = memory
                        .save_message(&channel, &agent_id, &answer, Some(&ctx.message.id))
                        .await;
                    if let Err(e) = ctx.respond(&answer).await {
                        tracing::error!("respond failed: {e}");
                    }
                }
            }
        })
        .build()?;

    runtime.run().await?;
    Ok(())
}

// ── Synthetic frame capture ──────────────────────────────────────────────────
//
// `/snap <prompt>` generates one image and attaches it to the turn as a
// `FileInputs` extension. The already-wired IngestStage → ArtifactOffload chain
// then offloads the bytes to the local store and keeps only an
// `[artifact <id>]` reference in SQLite history — the LLM re-fetches via
// get_artifact on demand. Type:  /snap what do you see?
//
// The frame is generated rather than captured so the example has no camera
// dependency. Swap `capture_frame` for a real source (camera, file, upload) and
// nothing downstream changes — that substitutability is the point.

/// Generate a QVGA gradient frame with a distinguishable block, encoded as JPEG.
fn capture_frame() -> Result<(Vec<u8>, String)> {
    use image::codecs::jpeg::JpegEncoder;
    use image::{ImageBuffer, Rgb};

    const W: u32 = 320;
    const H: u32 = 240;
    const QUALITY: u8 = 80;

    let buf: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(W, H, |x, y| {
        // A red square on a teal gradient — enough for the model to describe.
        if (80..160).contains(&x) && (80..160).contains(&y) {
            Rgb([220, 40, 40])
        } else {
            Rgb([(x * 255 / W) as u8, (y * 255 / H) as u8, 160])
        }
    });

    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, QUALITY)
        .encode_image(&buf)
        .context("failed to JPEG-encode frame")?;
    Ok((jpeg, "image/jpeg".to_string()))
}
