use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::Arc;

use crate::config::AgentConfig;
use crate::core::context::Context;
use crate::core::extension_map::SharedExtensionMap;
use crate::error::Result;
use crate::models::Message;
use crate::pipeline::Pipeline;

use super::Tool;

/// A tool that delegates a message to another agent pipeline.
///
/// When invoked by the LLM, extracts the "message" argument, creates a
/// [`Message`] and [`Context`], runs the configured pipeline, and returns
/// the response string. An optional session map can be propagated so the
/// sub-pipeline shares the caller's session state.
pub struct DelegationTool {
    tool_name: String,
    description: String,
    pipeline: Arc<Pipeline>,
    agent_config: Arc<AgentConfig>,
    session: Option<SharedExtensionMap>,
}

impl DelegationTool {
    /// Create a new `DelegationTool` with the given name, description, pipeline,
    /// and agent configuration.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        pipeline: Arc<Pipeline>,
        agent_config: Arc<AgentConfig>,
    ) -> Self {
        Self {
            tool_name: name.into(),
            description: description.into(),
            pipeline,
            agent_config,
            session: None,
        }
    }

    /// Attach a session-scoped extension map that is propagated to the
    /// delegated pipeline's context.
    pub fn with_session(mut self, session: SharedExtensionMap) -> Self {
        self.session = Some(session);
        self
    }
}

#[async_trait]
impl Tool for DelegationTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to delegate to the agent pipeline"
                }
            },
            "required": ["message"]
        })
    }

    async fn execute(&self, args: Value, _ctx: &super::ToolContext) -> Result<String> {
        let message_text = args
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let msg = Arc::new(Message::new(message_text, "delegation_tool", "delegation"));

        let mut ctx = Context::new(msg, Arc::clone(&self.agent_config));
        if let Some(ref session) = self.session {
            ctx = ctx.with_session(session.clone());
        }

        let response = self.pipeline.run(&mut ctx).await?;
        Ok(response.unwrap_or_else(|| "no response".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicBool, Ordering};

    use crate::pipeline::{Pipeline, PipelineStage};

    /// A stage that records whether it was called.
    struct RecorderStage {
        called: Arc<AtomicBool>,
    }

    #[async_trait]
    impl PipelineStage for RecorderStage {
        fn name(&self) -> &str {
            "recorder"
        }

        async fn process(&self, _ctx: &mut Context) -> crate::error::Result<()> {
            self.called.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    /// A stage that sets ctx.response to a fixed string.
    struct ResponderStage {
        response: String,
    }

    #[async_trait]
    impl PipelineStage for ResponderStage {
        fn name(&self) -> &str {
            "responder"
        }

        async fn process(&self, ctx: &mut Context) -> crate::error::Result<()> {
            ctx.response = Some(self.response.clone());
            Ok(())
        }
    }

    fn make_tool(pipeline: Pipeline) -> DelegationTool {
        DelegationTool::new(
            "delegate",
            "Delegates to a sub-agent",
            Arc::new(pipeline),
            Arc::new(AgentConfig::default()),
        )
    }

    #[tokio::test]
    async fn test_delegation_runs_pipeline() {
        let called = Arc::new(AtomicBool::new(false));
        let pipeline = Pipeline::new().add_stage(RecorderStage {
            called: called.clone(),
        });

        let tool = make_tool(pipeline);
        let args = json!({ "message": "hello" });
        let _ = tool
            .execute(args, &crate::tools::ToolContext::default())
            .await
            .expect("execute failed");

        assert!(
            called.load(Ordering::SeqCst),
            "RecorderStage should have been called"
        );
    }

    #[tokio::test]
    async fn test_delegation_returns_response() {
        let expected = "delegated response";
        let pipeline = Pipeline::new().add_stage(ResponderStage {
            response: expected.to_string(),
        });

        let tool = make_tool(pipeline);
        let args = json!({ "message": "do something" });
        let result = tool
            .execute(args, &crate::tools::ToolContext::default())
            .await
            .expect("execute failed");

        assert_eq!(result, expected);
    }

    #[tokio::test]
    async fn test_delegation_returns_no_response_fallback() {
        // Pipeline with no stage that sets ctx.response
        let pipeline = Pipeline::new().add_stage(RecorderStage {
            called: Arc::new(AtomicBool::new(false)),
        });

        let tool = make_tool(pipeline);
        let args = json!({ "message": "hello" });
        let result = tool
            .execute(args, &crate::tools::ToolContext::default())
            .await
            .expect("execute failed");

        assert_eq!(result, "no response");
    }

    #[tokio::test]
    async fn test_delegation_with_session() {
        let session = SharedExtensionMap::new();
        session.set(42u32);

        let called = Arc::new(AtomicBool::new(false));
        let pipeline = Pipeline::new().add_stage(RecorderStage {
            called: called.clone(),
        });

        let tool = make_tool(pipeline).with_session(session);
        let args = json!({ "message": "hello" });
        let _ = tool
            .execute(args, &crate::tools::ToolContext::default())
            .await
            .expect("execute failed");

        assert!(called.load(Ordering::SeqCst));
    }
}
