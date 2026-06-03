#[cfg(test)]
mod model_tests {
    use crate::models::*;

    #[test]
    fn message_serde_roundtrip() {
        let msg = Message::new("hello world", "user-1", "channel-1");
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.content, "hello world");
        assert_eq!(decoded.sender_id, "user-1");
        assert_eq!(decoded.channel_id, "channel-1");
        assert!(!decoded.id.is_empty());
    }

    #[test]
    fn response_serde_roundtrip() {
        let resp = Response::new("reply", "ch-1", "agent-1").reply_to("msg-1");
        let json = serde_json::to_string(&resp).unwrap();
        let decoded: Response = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.content, "reply");
        assert_eq!(decoded.reply_to_id, Some("msg-1".to_string()));
    }

    #[test]
    fn stream_event_serde_roundtrip() {
        let events = vec![
            StreamEvent::Thinking {
                content: "hmm".into(),
            },
            StreamEvent::Chunk {
                content: "hello".into(),
            },
            StreamEvent::Complete {
                content: "done".into(),
                usage: None,
            },
            StreamEvent::Error {
                message: "oops".into(),
            },
            StreamEvent::Heartbeat,
        ];
        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            let decoded: StreamEvent = serde_json::from_str(&json).unwrap();
            // Verify roundtrip by re-serializing
            let json2 = serde_json::to_string(&decoded).unwrap();
            assert_eq!(json, json2);
        }
    }

    #[test]
    fn llm_message_constructors() {
        let sys = LlmMessage::system("you are helpful");
        assert_eq!(sys.role, Role::System);
        let user = LlmMessage::user("hello");
        assert_eq!(user.role, Role::User);
        let asst = LlmMessage::assistant("hi");
        assert_eq!(asst.role, Role::Assistant);
    }
}

#[cfg(test)]
mod config_tests {
    use crate::config::MindroidConfig;

    #[test]
    fn parse_minimal_toml() {
        let toml_str = r#"
[agent]
agent_id = "agent-123"
name = "Test Agent"

[transport]
type = "centrifugo"
url = "wss://example.com/ws"

[pipeline]
type = "magickmind"
base_url = "https://api.example.com"

[auth]
type = "apikey"
email = "test@test.com"
"#;
        let config = MindroidConfig::from_toml_str(toml_str).unwrap();
        assert_eq!(config.agent.agent_id, "agent-123");
        assert_eq!(config.agent.name, "Test Agent");
        assert_eq!(config.transport.transport_type, Some("centrifugo".into()));
        assert_eq!(config.pipeline.pipeline_type, Some("magickmind".into()));
        assert_eq!(config.auth.auth_type, Some("apikey".into()));
    }

    #[test]
    fn defaults_applied() {
        let config = MindroidConfig::from_toml_str("").unwrap();
        assert_eq!(config.agent.name, "Mindroid Agent");
        assert_eq!(config.agent.compute_power, 50);
        assert_eq!(config.agent.model_type, "chat");
    }
}

#[cfg(test)]
mod pipeline_tests {
    use std::sync::Arc;

    use crate::config::AgentConfig;
    use crate::error::Result;
    use crate::models::Message;
    use crate::core::context::Context;
    use crate::pipeline::{Pipeline, PipelineStage};
    use async_trait::async_trait;

    struct AppendStage {
        name: String,
        value: String,
    }

    #[async_trait]
    impl PipelineStage for AppendStage {
        fn name(&self) -> &str {
            &self.name
        }
        async fn process(&self, ctx: &mut Context) -> Result<()> {
            let current = ctx.response.take().unwrap_or_default();
            ctx.response = Some(format!("{current}{}", self.value));
            Ok(())
        }
    }

    #[tokio::test]
    async fn pipeline_stages_run_in_order() {
        let pipeline = Pipeline::new()
            .add_stage(AppendStage {
                name: "a".into(),
                value: "A".into(),
            })
            .add_stage(AppendStage {
                name: "b".into(),
                value: "B".into(),
            })
            .add_stage(AppendStage {
                name: "c".into(),
                value: "C".into(),
            });

        let msg = Message::new("test", "user", "ch");
        let mut ctx = Context::new(Arc::new(msg), Arc::new(AgentConfig::default()));
        let result = pipeline.run(&mut ctx).await.unwrap();
        assert_eq!(result, Some("ABC".to_string()));
    }

    struct FailStage;

    #[async_trait]
    impl PipelineStage for FailStage {
        fn name(&self) -> &str {
            "fail"
        }
        async fn process(&self, _ctx: &mut Context) -> Result<()> {
            Err(crate::error::MindroidError::Pipeline {
                stage: "fail".into(),
                message: "intentional".into(),
                source: None,
            })
        }
    }

    #[tokio::test]
    async fn pipeline_halts_on_error() {
        let pipeline = Pipeline::new()
            .add_stage(AppendStage {
                name: "a".into(),
                value: "A".into(),
            })
            .add_stage(FailStage)
            .add_stage(AppendStage {
                name: "c".into(),
                value: "C".into(),
            });

        let msg = Message::new("test", "user", "ch");
        let mut ctx = Context::new(Arc::new(msg), Arc::new(AgentConfig::default()));
        let result = pipeline.run(&mut ctx).await;
        assert!(result.is_err());
        // Stage C should not have run, so response should be "A"
        assert_eq!(ctx.response, Some("A".into()));
    }

    #[tokio::test]
    async fn response_field_returned() {
        struct SetResponse;

        #[async_trait]
        impl PipelineStage for SetResponse {
            fn name(&self) -> &str {
                "set_response"
            }
            async fn process(&self, ctx: &mut Context) -> Result<()> {
                ctx.response = Some("hello!".into());
                Ok(())
            }
        }

        let pipeline = Pipeline::new().add_stage(SetResponse);

        let msg = Message::new("test", "user", "ch");
        let mut ctx = Context::new(Arc::new(msg), Arc::new(AgentConfig::default()));
        let result = pipeline.run(&mut ctx).await.unwrap();
        assert_eq!(result, Some("hello!".to_string()));
    }

    /// Stage that halts the pipeline without setting a response.
    struct HaltStage;

    #[async_trait]
    impl PipelineStage for HaltStage {
        fn name(&self) -> &str {
            "halt"
        }
        async fn process(&self, ctx: &mut Context) -> Result<()> {
            ctx.halted = true;
            Ok(())
        }
    }

    /// Stage that halts the pipeline AND sets a canned response.
    struct HaltWithResponseStage;

    #[async_trait]
    impl PipelineStage for HaltWithResponseStage {
        fn name(&self) -> &str {
            "halt_with_response"
        }
        async fn process(&self, ctx: &mut Context) -> Result<()> {
            ctx.response = Some("canned".into());
            ctx.halted = true;
            Ok(())
        }
    }

    /// Stage that records whether it was called.
    struct SpyStage {
        called: Arc<std::sync::atomic::AtomicBool>,
    }

    #[async_trait]
    impl PipelineStage for SpyStage {
        fn name(&self) -> &str {
            "spy"
        }
        async fn process(&self, _ctx: &mut Context) -> Result<()> {
            self.called.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test]
    async fn pipeline_halts_when_flagged() {
        let pipeline = Pipeline::new().add_stage(HaltStage).add_stage(AppendStage {
            name: "after".into(),
            value: "X".into(),
        });

        let msg = Message::new("test", "user", "ch");
        let mut ctx = Context::new(Arc::new(msg), Arc::new(AgentConfig::default()));
        let result = pipeline.run(&mut ctx).await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn pipeline_halt_with_response() {
        let pipeline = Pipeline::new()
            .add_stage(HaltWithResponseStage)
            .add_stage(AppendStage {
                name: "after".into(),
                value: "X".into(),
            });

        let msg = Message::new("test", "user", "ch");
        let mut ctx = Context::new(Arc::new(msg), Arc::new(AgentConfig::default()));
        let result = pipeline.run(&mut ctx).await.unwrap();
        assert_eq!(result, Some("canned".to_string()));
    }

    #[tokio::test]
    async fn pipeline_halt_skips_remaining_stages() {
        let spy_called = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let pipeline = Pipeline::new()
            .add_stage(AppendStage {
                name: "first".into(),
                value: "A".into(),
            })
            .add_stage(HaltStage)
            .add_stage(SpyStage {
                called: spy_called.clone(),
            });

        let msg = Message::new("test", "user", "ch");
        let mut ctx = Context::new(Arc::new(msg), Arc::new(AgentConfig::default()));
        let result = pipeline.run(&mut ctx).await.unwrap();
        // Halted with no explicit response, but response was set by first stage
        assert_eq!(result, Some("A".to_string()));
        // SpyStage should NOT have been called
        assert!(!spy_called.load(std::sync::atomic::Ordering::SeqCst));
    }
}

#[cfg(test)]
#[cfg(feature = "llm-client")]
mod llm_config_tests {
    use crate::config::MindroidConfig;
    use crate::llm_client::AuthStyle;

    #[test]
    fn llm_resolves_provider_inheritance() {
        let toml_str = r#"
[providers.ollama]
base_url = "http://localhost:11434"

[models.gate]
provider = "ollama"
model = "smallthinker"
"#;
        let config = MindroidConfig::from_toml_str(toml_str).unwrap();
        let llm = config.llm("gate").unwrap();

        assert_eq!(llm.base_url, "http://localhost:11434/v1");
        assert_eq!(llm.default_model, Some("smallthinker".into()));
        assert!(llm.api_key.is_none());
        assert!(matches!(llm.auth_style, AuthStyle::None));
    }

    #[test]
    fn llm_model_overrides_provider_api_key() {
        let toml_str = r#"
[providers.cortex]
base_url = "https://api.magickmind.io"
api_key = "sk-shared"

[models.main]
provider = "cortex"
model = "gpt-4"
api_key = "sk-override"
"#;
        let config = MindroidConfig::from_toml_str(toml_str).unwrap();
        let llm = config.llm("main").unwrap();

        assert_eq!(llm.base_url, "https://api.magickmind.io/v1");
        assert_eq!(llm.api_key, Some("sk-override".into()));
        assert_eq!(llm.default_model, Some("gpt-4".into()));
        assert!(matches!(llm.auth_style, AuthStyle::Bearer));
    }

    #[test]
    fn llm_model_inherits_provider_api_key() {
        let toml_str = r#"
[providers.cortex]
base_url = "https://api.magickmind.io"
api_key = "sk-shared"

[models.ack]
provider = "cortex"
model = "gpt-4o-mini"
"#;
        let config = MindroidConfig::from_toml_str(toml_str).unwrap();
        let llm = config.llm("ack").unwrap();

        assert_eq!(llm.api_key, Some("sk-shared".into()));
    }

    #[test]
    fn llm_model_overrides_base_url() {
        let toml_str = r#"
[providers.openai]
base_url = "https://api.openai.com"
api_key = "sk-key"

[models.proxy]
provider = "openai"
base_url = "https://proxy.example.com"
model = "gpt-4"
"#;
        let config = MindroidConfig::from_toml_str(toml_str).unwrap();
        let llm = config.llm("proxy").unwrap();

        assert_eq!(llm.base_url, "https://proxy.example.com/v1");
    }

    #[test]
    fn llm_compute_power_as_custom_header() {
        let toml_str = r#"
[providers.cortex]
base_url = "https://api.magickmind.io"
api_key = "sk-key"

[models.main]
provider = "cortex"
model = "gpt-4"
compute_power = 80
"#;
        let config = MindroidConfig::from_toml_str(toml_str).unwrap();
        let llm = config.llm("main").unwrap();

        assert_eq!(
            llm.custom_headers.get("X-Compute-Power"),
            Some(&"80".to_string())
        );
    }

    #[test]
    fn llm_temperature_and_max_tokens() {
        let toml_str = r#"
[providers.ollama]
base_url = "http://localhost:11434"

[models.creative]
provider = "ollama"
model = "llama3"
temperature = 0.9
max_tokens = 2048
"#;
        let config = MindroidConfig::from_toml_str(toml_str).unwrap();
        let llm = config.llm("creative").unwrap();

        assert_eq!(llm.default_temperature, Some(0.9));
        assert_eq!(llm.default_max_tokens, Some(2048));
    }

    #[test]
    fn llm_explicit_auth_style_override() {
        let toml_str = r#"
[providers.cortex]
base_url = "https://api.magickmind.io"
api_key = "sk-key"
auth_style = "x-api-key"

[models.main]
provider = "cortex"
model = "gpt-4"
"#;
        let config = MindroidConfig::from_toml_str(toml_str).unwrap();
        let llm = config.llm("main").unwrap();

        assert!(matches!(llm.auth_style, AuthStyle::XApiKey));
    }

    #[test]
    fn llm_missing_model_name_errors() {
        let toml_str = r#"
[providers.ollama]
base_url = "http://localhost:11434"
"#;
        let config = MindroidConfig::from_toml_str(toml_str).unwrap();
        let result = config.llm("nonexistent");

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("nonexistent"),
            "error should mention the name: {err}"
        );
    }

    #[test]
    fn llm_missing_provider_errors() {
        let toml_str = r#"
[models.gate]
provider = "missing"
model = "smallthinker"
"#;
        let config = MindroidConfig::from_toml_str(toml_str).unwrap();
        let result = config.llm("gate");

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("missing"),
            "error should mention the provider: {err}"
        );
    }

    #[test]
    fn llm_missing_base_url_errors() {
        let toml_str = r#"
[providers.empty]

[models.test]
provider = "empty"
model = "test"
"#;
        let config = MindroidConfig::from_toml_str(toml_str).unwrap();
        let result = config.llm("test");

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("base_url"),
            "error should mention base_url: {err}"
        );
    }

    #[test]
    fn empty_providers_and_models_no_breakage() {
        let toml_str = r#"
[agent]
agent_id = "test"
"#;
        let config = MindroidConfig::from_toml_str(toml_str).unwrap();
        assert!(config.providers.is_empty());
        assert!(config.models.is_empty());
    }

    #[test]
    fn multiple_models_same_provider() {
        let toml_str = r#"
[providers.cortex]
base_url = "https://api.magickmind.io"
api_key = "sk-shared"

[models.fast]
provider = "cortex"
model = "gpt-4o-mini"
compute_power = 20

[models.smart]
provider = "cortex"
model = "gpt-4"
compute_power = 90
"#;
        let config = MindroidConfig::from_toml_str(toml_str).unwrap();

        let fast = config.llm("fast").unwrap();
        let smart = config.llm("smart").unwrap();

        // Same provider credentials
        assert_eq!(fast.api_key, smart.api_key);
        assert_eq!(fast.base_url, smart.base_url);

        // Different model and compute power
        assert_eq!(fast.default_model, Some("gpt-4o-mini".into()));
        assert_eq!(smart.default_model, Some("gpt-4".into()));
        assert_eq!(
            fast.custom_headers.get("X-Compute-Power"),
            Some(&"20".to_string())
        );
        assert_eq!(
            smart.custom_headers.get("X-Compute-Power"),
            Some(&"90".to_string())
        );
    }
}

#[cfg(test)]
mod trait_object_safety {
    use crate::auth::Auth;
    use crate::memory::Memory;
    use crate::observer::Observer;
    use crate::pipeline::{PipelineStage, StreamingStage};
    use crate::transport::Transport;

    // These functions just need to compile — they prove the traits are object-safe.
    fn _transport(_: Box<dyn Transport>) {}
    fn _auth(_: Box<dyn Auth>) {}
    fn _memory(_: Box<dyn Memory>) {}
    fn _observer(_: Box<dyn Observer>) {}
    fn _pipeline_stage(_: Box<dyn PipelineStage>) {}
    fn _streaming_stage(_: Box<dyn StreamingStage>) {}
}
