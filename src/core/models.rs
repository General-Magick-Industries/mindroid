use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    #[default]
    Text,
    Command,
    System,
    Image,
    Audio,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SenderType {
    #[default]
    User,
    Agent,
    System,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChannelType {
    #[default]
    Direct,
    Group,
    Broadcast,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub content: String,
    pub sender_id: String,
    #[serde(default)]
    pub sender_type: SenderType,
    pub channel_id: String,
    #[serde(default)]
    pub channel_type: ChannelType,
    #[serde(default)]
    pub message_type: MessageType,
    pub timestamp: DateTime<Utc>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub platform: Option<String>,
}

impl Message {
    pub fn new(
        content: impl Into<String>,
        sender_id: impl Into<String>,
        channel_id: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.into(),
            sender_id: sender_id.into(),
            sender_type: SenderType::User,
            channel_id: channel_id.into(),
            channel_type: ChannelType::Direct,
            message_type: MessageType::Text,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
            platform: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub content: String,
    pub channel_id: String,
    pub sender_id: String,
    pub reply_to_id: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Response {
    pub fn new(
        content: impl Into<String>,
        channel_id: impl Into<String>,
        sender_id: impl Into<String>,
    ) -> Self {
        Self {
            content: content.into(),
            channel_id: channel_id.into(),
            sender_id: sender_id.into(),
            reply_to_id: None,
            metadata: HashMap::new(),
        }
    }

    pub fn reply_to(mut self, message_id: impl Into<String>) -> Self {
        self.reply_to_id = Some(message_id.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    Thinking {
        content: String,
    },
    Chunk {
        content: String,
    },
    ToolCall {
        name: String,
        arguments: String,
    },
    ToolResult {
        name: String,
        result: String,
    },
    Complete {
        content: String,
        usage: Option<TokenUsage>,
    },
    Error {
        message: String,
    },
    Heartbeat,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
    #[serde(other)]
    Unknown,
}

impl Role {
    pub fn as_str(&self) -> &str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for Role {
    fn from(s: &str) -> Self {
        match s {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::Unknown,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: Role,
    pub content: String,
}

impl LlmMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}
