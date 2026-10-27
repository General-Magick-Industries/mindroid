//! Resumes a turn whose remote tool call was never answered.

use std::time::Duration;

use async_trait::async_trait;

use crate::core::context::Context;
use crate::core::routine::{Routine, RoutineContext};
use crate::error::Result;
use crate::models::{Message, MessageType, Response};
use crate::pipeline::extensions::CorrelatedRemoteResult;
use crate::pipeline::stages::{ExpiredRemoteCall, PendingRemoteCalls};

const DEFAULT_SWEEP_INTERVAL: Duration = Duration::from_secs(15);

/// Turns an expired remote tool call into a synthesized error result.
///
/// A remote call is emitted as the turn's response and answered by a later
/// inbound message. When the client never answers, the call expires after the
/// tool's [`remote_timeout`](crate::tools::Tool::remote_timeout) — which frees
/// the slot but leaves the conversation truncated, since the model was still
/// waiting on a tool. This routine feeds the expiry back in as
/// `<tool_result name="…">error: …</tool_result>`, so the pipeline runs to
/// completion and the model can say the tool never responded.
///
/// Wire it with the executor's own outstanding-call set, or expiries are only
/// logged:
///
/// ```ignore
/// let executor = ToolExecutorStage::new(client, registry);
/// let runtime = Runtime::builder()
///     .routine(RemoteCallTimeout::new(executor.pending()))
///     .pipeline(Pipeline::new().add_streaming_stage(executor))
///     .build()?;
/// ```
///
/// A late result arriving after expiry is dropped as unsolicited: the claim was
/// consumed here, so the model never sees two answers to one call.
pub struct RemoteCallTimeout {
    pending: PendingRemoteCalls,
    interval: Duration,
}

impl RemoteCallTimeout {
    pub fn new(pending: PendingRemoteCalls) -> Self {
        Self {
            pending,
            interval: DEFAULT_SWEEP_INTERVAL,
        }
    }

    /// How often to sweep. The tool's timeout sets *when* a call expires; this
    /// only bounds how late the notification is, so it need not be short.
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    async fn resume(&self, ctx: &RoutineContext, call: &ExpiredRemoteCall) -> Result<()> {
        let mut message = Message::new(
            format!(
                "<tool_result name=\"{}\">error: {}</tool_result>",
                call.name,
                call.reason.as_error()
            ),
            &call.sender_id,
            &call.channel_id,
        );
        message.message_type = MessageType::ToolResult;

        let message = std::sync::Arc::new(message);
        let mut pipeline_ctx = Context::new(message.clone(), ctx.agent_config.clone());
        // Claimed already: the sweep took the pending entry, so the gate must
        // not try to claim it a second time and drop the turn.
        pipeline_ctx.set(CorrelatedRemoteResult(message.id.clone()));

        if let Some(response) = ctx.pipeline.run(&mut pipeline_ctx).await?
            && !response.trim().is_empty()
        {
            ctx.send(
                &Response::new(response, &call.channel_id, &ctx.agent_config.agent_id)
                    .reply_to(&message.id),
            )
            .await?;
        }
        Ok(())
    }
}

#[async_trait]
impl Routine for RemoteCallTimeout {
    fn name(&self) -> &str {
        "remote_call_timeout"
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    async fn poll(&self) -> Result<Option<String>> {
        let expired = self.pending.sweep_expired();
        if expired.is_empty() {
            return Ok(None);
        }
        serde_json::to_string(&expired)
            .map(Some)
            .map_err(|e| crate::error::MindroidError::pipeline(e.to_string()))
    }

    async fn act(&self, ctx: RoutineContext, data: String) -> Result<()> {
        let expired: Vec<ExpiredRemoteCall> = serde_json::from_str(&data)
            .map_err(|e| crate::error::MindroidError::pipeline(e.to_string()))?;
        for call in expired {
            tracing::warn!(
                tool = %call.name,
                channel = %call.channel_id,
                reason = ?call.reason,
                "Remote tool call was never answered; resuming the turn with an error result"
            );
            // One unresumable turn must not strand the others in this sweep.
            if let Err(error) = self.resume(&ctx, &call).await {
                tracing::error!(tool = %call.name, %error, "Failed to resume a timed-out turn");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::stages::ExpiryReason;

    fn expired(name: &str, reason: ExpiryReason) -> ExpiredRemoteCall {
        ExpiredRemoteCall {
            channel_id: "chan1".into(),
            id: "call-1".into(),
            name: name.into(),
            sender_id: "client".into(),
            reason,
        }
    }

    /// The synthesized body carries no `call` attribute, so it does NOT satisfy
    /// `validated_tool_result` — the same shape the gate produces after it
    /// strips the attribute. It reaches the model because the claim marker
    /// exempts it from admission, so that exemption is the thing to pin.
    #[tokio::test]
    async fn the_synthesized_result_reaches_the_stages_rather_than_being_refused() {
        for reason in [ExpiryReason::TimedOut, ExpiryReason::Evicted] {
            let call = expired("take_photo", reason);
            let mut message = Message::new(
                format!(
                    "<tool_result name=\"{}\">error: {}</tool_result>",
                    call.name,
                    call.reason.as_error()
                ),
                &call.sender_id,
                &call.channel_id,
            );
            message.message_type = MessageType::ToolResult;
            let message = std::sync::Arc::new(message);

            let reached = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let pipeline = crate::pipeline::Pipeline::new().add_stage(Recorder {
                reached: reached.clone(),
            });

            let mut ctx = Context::new(
                message.clone(),
                std::sync::Arc::new(crate::config::AgentConfig::default()),
            );
            ctx.set(CorrelatedRemoteResult(message.id.clone()));
            pipeline.run(&mut ctx).await.unwrap();

            assert!(
                reached.load(std::sync::atomic::Ordering::SeqCst),
                "{reason:?}: a timed-out turn that never reaches a stage resumes nothing"
            );
            assert!(!ctx.halted);
        }
    }

    /// Without the claim, the same body is refused — so the exemption above is
    /// doing the work, not something incidental about the envelope.
    #[tokio::test]
    async fn the_same_body_without_a_claim_is_still_refused() {
        let mut message = Message::new(
            "<tool_result name=\"take_photo\">error: the client did not answer in time</tool_result>",
            "client",
            "chan1",
        );
        message.message_type = MessageType::ToolResult;

        let reached = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pipeline = crate::pipeline::Pipeline::new().add_stage(Recorder {
            reached: reached.clone(),
        });

        let mut ctx = Context::new(
            std::sync::Arc::new(message),
            std::sync::Arc::new(crate::config::AgentConfig::default()),
        );
        pipeline.run(&mut ctx).await.unwrap();

        assert!(!reached.load(std::sync::atomic::Ordering::SeqCst));
        assert!(ctx.halted);
    }

    /// The sweep already consumed the claim, so a gate wired ahead of the
    /// executor must not try to claim it again and drop the turn.
    #[tokio::test]
    async fn an_explicit_gate_does_not_reclaim_a_synthesized_result() {
        let pending = PendingRemoteCalls::default();
        let mut message = Message::new(
            "<tool_result name=\"take_photo\">error: the client did not answer in time</tool_result>",
            "client",
            "chan1",
        );
        message.message_type = MessageType::ToolResult;
        let message = std::sync::Arc::new(message);

        let mut ctx = Context::new(
            message.clone(),
            std::sync::Arc::new(crate::config::AgentConfig::default()),
        );
        ctx.set(CorrelatedRemoteResult(message.id.clone()));

        crate::pipeline::PipelineStage::process(
            &crate::pipeline::stages::RemoteResultGate::with_pending(pending),
            &mut ctx,
        )
        .await
        .unwrap();

        assert!(
            !ctx.halted,
            "the gate dropped the turn it was meant to pass"
        );
    }

    struct Recorder {
        reached: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    #[async_trait]
    impl crate::pipeline::PipelineStage for Recorder {
        fn name(&self) -> &str {
            "Recorder"
        }
        async fn process(&self, _ctx: &mut Context) -> Result<()> {
            self.reached
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test]
    async fn poll_reports_nothing_until_a_call_expires() {
        let pending = PendingRemoteCalls::default();
        let routine = RemoteCallTimeout::new(pending.clone());

        pending.record_for(
            "chan1",
            Some("client"),
            "call-1",
            "take_photo",
            Duration::from_secs(300),
        );
        assert!(routine.poll().await.unwrap().is_none());

        pending.record_for(
            "chan1",
            Some("client"),
            "call-2",
            "take_photo",
            Duration::ZERO,
        );
        let data = routine.poll().await.unwrap().expect("the expired call");
        let reported: Vec<ExpiredRemoteCall> = serde_json::from_str(&data).unwrap();

        assert_eq!(reported.len(), 1, "only the expired call is reported");
        assert_eq!(reported[0].id, "call-2");
        assert_eq!(reported[0].reason, ExpiryReason::TimedOut);
        assert!(
            routine.poll().await.unwrap().is_none(),
            "draining is one-shot, or the turn resumes once per sweep forever"
        );
    }
}
