use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::core::content::ContentPart;

/// Which identity a credential acts as; adapters use it to pick service-user
/// (`/v1/...`) vs end-user (`/v1/end-user/...`) routes and connect behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum CredentialKind {
    #[default]
    ServiceUser,
    EndUser,
}

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
    #[serde(deserialize_with = "deserialize_content")]
    pub content: Vec<ContentPart>,
}

impl LlmMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentPart::text(content)],
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentPart::text(content)],
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentPart::text(content)],
        }
    }

    /// Join all text parts into a single string.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| part.as_text())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Append text to the last Text part, or push a new Text part.
    /// Critical for tool_executor.rs which does sys.content.push_str().
    pub fn append_text(&mut self, s: &str) {
        // Find last text part and append to it
        for part in self.content.iter_mut().rev() {
            if let ContentPart::Text { text } = part {
                text.push_str(s);
                return;
            }
        }
        // No text part found, push a new one
        self.content.push(ContentPart::text(s));
    }
}

/// Deserializes content that may be either a plain String (old format)
/// or a Vec<ContentPart> (new format).
fn deserialize_content<'de, D>(deserializer: D) -> std::result::Result<Vec<ContentPart>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum ContentOrString {
        Parts(Vec<ContentPart>),
        Plain(String),
    }

    match ContentOrString::deserialize(deserializer)? {
        ContentOrString::Parts(parts) => Ok(parts),
        ContentOrString::Plain(s) => Ok(vec![ContentPart::text(s)]),
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_message_backward_compat() {
        let msg = LlmMessage::user("hello");
        assert_eq!(msg.text(), "hello");
    }

    #[test]
    fn test_append_text_to_existing() {
        let mut msg = LlmMessage::user("hello");
        msg.append_text(" world");
        assert_eq!(msg.text(), "hello world");
    }

    #[test]
    fn test_append_text_to_empty() {
        let mut msg = LlmMessage {
            role: Role::User,
            content: vec![],
        };
        msg.append_text("new text");
        assert_eq!(msg.text(), "new text");
        assert_eq!(msg.content.len(), 1);
    }

    #[test]
    fn test_serde_roundtrip() {
        let msg = LlmMessage {
            role: Role::User,
            content: vec![
                ContentPart::text("hello"),
                ContentPart::Image {
                    source: crate::core::content::ContentSource::Uri {
                        uri: "https://example.com/img.png".into(),
                    },
                    mime_type: "image/png".into(),
                },
            ],
        };
        let json = serde_json::to_string(&msg).unwrap();
        let decoded: LlmMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.role, Role::User);
        assert_eq!(decoded.content.len(), 2);
        assert_eq!(decoded.text(), "hello");
    }

    #[test]
    fn test_serde_migration() {
        // Old format: content is a plain string
        let json = r#"{"role":"user","content":"hello"}"#;
        let msg: LlmMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content.len(), 1);
        assert_eq!(msg.text(), "hello");
    }
}
