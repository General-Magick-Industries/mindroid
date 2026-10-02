use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result};
use mindroid::auth::static_id::StaticAuth;
use mindroid::llm_client::{LlmClient, LlmClientConfig};
use mindroid::pipeline::presets::magickmind::{
    MagickmindClient, MagickmindContext, MagickmindPersistence,
};
use mindroid::tools::{DynamicRegistry, ManifestStage, RemoteCallTimeout};
use mindroid::transport::centrifugo::CentrifugoTransport;
use mindroid::{
    Auth, ContextPreparer, CredentialKind, MessageType, MindroidConfig, Pipeline, PipelineContext,
    PrepareOutcome, Runtime, SimpleContextBuilder, ToolRegistry, XmlToolExecutorStage,
};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

const SYSTEM_PROMPT: &str = "You are a small desk robot. Use your tools to act.";

pub struct AgentSetup {
    pub base_url: String,
    pub ws_url: String,
    pub agent_id: String,
    pub token: String,
    pub device_id: String,
    pub llm_base_url: String,
    pub allow_insecure: bool,
}

pub struct AgentHandle {
    cancel: CancellationToken,
    task: JoinHandle<()>,
}

impl AgentHandle {
    pub async fn stop(self) {
        self.cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(5), self.task).await;
    }
}

pub async fn start(setup: AgentSetup) -> Result<AgentHandle> {
    let auth: Arc<dyn Auth> = Arc::new(StaticAuth::new(setup.token));
    let transport = CentrifugoTransport::new(&setup.ws_url, &setup.agent_id, Arc::clone(&auth))
        .with_credential_kind(CredentialKind::EndUser)
        .with_trust_fanout_sender(true)
        .with_allow_insecure(setup.allow_insecure);

    let magickmind = Arc::new(
        MagickmindClient::try_new(&setup.base_url, Arc::clone(&auth), setup.allow_insecure)?
            .with_credential_kind(CredentialKind::EndUser),
    );
    let preparer = Arc::new(ContextPreparer::new().add_provider(
        MagickmindContext::new(Arc::clone(&magickmind)).with_self_id(&setup.agent_id),
    ));

    let mut llm = LlmClientConfig::new(format!("{}/v1", setup.llm_base_url));
    llm.default_model = Some("stub".into());
    let registry = DynamicRegistry::new(ToolRegistry::new());
    let executor =
        XmlToolExecutorStage::with_dynamic_registry(LlmClient::new(llm)?, registry.clone())
            .with_max_iterations(4);
    let manifest = Arc::new(
        Pipeline::new().add_stage(ManifestStage::new(registry).trust_sender(&setup.device_id)),
    );

    let mut config = MindroidConfig::default();
    config.agent.agent_id = setup.agent_id;

    let pending = executor.pending();
    let mut runtime = Runtime::builder()
        .config(config)
        .transport(transport)
        .add_routine(RemoteCallTimeout::new(pending).with_interval(Duration::from_secs(2)))
        .on_message(move |ctx| {
            let preparer = Arc::clone(&preparer);
            let manifest = Arc::clone(&manifest);
            let executor = executor.clone();
            let magickmind = Arc::clone(&magickmind);
            async move {
                let mut pctx = PipelineContext::new(ctx.message.clone(), ctx.agent_config.clone());
                if ctx.message.message_type == MessageType::ToolManifest {
                    if let Err(e) = ctx.run_with_context(&manifest, &mut pctx).await {
                        tracing::error!("manifest pipeline failed: {e}");
                    }
                    return;
                }
                let history = match preparer.prepare(&ctx.message).await {
                    PrepareOutcome::Complete(msgs) => msgs,
                    PrepareOutcome::Degraded { messages, warnings } => {
                        for w in &warnings {
                            tracing::warn!(
                                "context provider '{}' degraded: {}",
                                w.provider,
                                w.error
                            );
                        }
                        messages
                    }
                    PrepareOutcome::Failed(warnings) => {
                        for w in &warnings {
                            tracing::error!(
                                "context provider '{}' failed: {}",
                                w.provider,
                                w.error
                            );
                        }
                        return;
                    }
                };
                let turn = Pipeline::new()
                    .add_stage(executor.result_gate())
                    .add_stage(SimpleContextBuilder::with_prompt_and_history(
                        SYSTEM_PROMPT,
                        Arc::new(history),
                    ))
                    .add_streaming_stage(executor)
                    .add_stage(MagickmindPersistence::new(magickmind));
                if let Err(e) = ctx.run_with_context(&turn, &mut pctx).await {
                    tracing::error!("agent turn failed: {e}");
                }
            }
        })
        .build()?;

    let mut health = runtime.health();
    let cancel = CancellationToken::new();
    let stop = cancel.clone();
    let task = tokio::spawn(async move {
        if let Err(e) = runtime.run_until_cancelled(stop).await {
            tracing::error!("agent runtime exited: {e}");
        }
    });
    health
        .wait_ready(Duration::from_secs(15))
        .await
        .map_err(|state| anyhow::anyhow!("agent never became ready: {state:?}"))
        .context("starting the mindroid agent")?;
    Ok(AgentHandle { cancel, task })
}
