// XML-backed memory for Mindroid

use async_trait::async_trait;
use chrono::Utc;
use std::fs;
use std::path::PathBuf;

use crate::{ChannelType, Memory, Message, MessageType, MindroidError, Result, SenderType};

pub struct MarkdownMemory {
    base_path: PathBuf,
}

impl MarkdownMemory {
    pub fn new(path: &str) -> Result<Self> {
        let base_path = PathBuf::from(path);
        fs::create_dir_all(&base_path)
            .map_err(|e| MindroidError::Other(anyhow::Error::from(e)))?;
        Ok(Self { base_path })
    }

    fn channel_file(&self, channel_id: &str) -> PathBuf {
        self.base_path.join(format!("{}.xml", channel_id))
    }

    fn format_message(&self, id: &str, sender_id: &str, content: &str, reply_to_id: Option<&str>, timestamp: &str) -> String {
        let reply = reply_to_id
            .map(|r| format!("    <reply_to>{}</reply_to>\n", Self::escape_xml(r)))
            .unwrap_or_default();
        format!(
            "  <message>\n    <id>{}</id>\n    <sender>{}</sender>\n    <timestamp>{}</timestamp>\n{}    <content>{}</content>\n  </message>\n",
            Self::escape_xml(id),
            Self::escape_xml(sender_id),
            Self::escape_xml(timestamp),
            reply,
            Self::escape_xml(content)
        )
    }

    fn escape_xml(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }

    fn unescape_xml(s: &str) -> String {
        s.replace("&apos;", "'")
            .replace("&quot;", "\"")
            .replace("&gt;", ">")
            .replace("&lt;", "<")
            .replace("&amp;", "&")
    }

    fn parse_xml_element(xml: &str, tag: &str) -> Option<String> {
        let open_tag = format!("<{}>", tag);
        let close_tag = format!("</{}>", tag);

        if let Some(start) = xml.find(&open_tag) {
            if let Some(end) = xml[start + open_tag.len()..].find(&close_tag) {
                let content = &xml[start + open_tag.len()..start + open_tag.len() + end];
                return Some(Self::unescape_xml(content));
            }
        }
        None
    }
}

#[async_trait]
impl Memory for MarkdownMemory {
    async fn save_message(
        &self,
        channel_id: &str,
        sender_id: &str,
        content: &str,
        reply_to_id: Option<&str>,
    ) -> Result<Option<String>> {
        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = Utc::now().to_rfc3339();
        let file_path = self.channel_file(channel_id);
        let id_clone = id.clone();
        let sender_id = sender_id.to_string();
        let content = content.to_string();
        let reply_to_id = reply_to_id.map(|s| s.to_string());

        tokio::task::spawn_blocking(move || {
            let message_xml = format!(
                "  <message>\n    <id>{}</id>\n    <sender>{}</sender>\n    <timestamp>{}</timestamp>\n{}    <content>{}</content>\n  </message>\n",
                MarkdownMemory::escape_xml(&id_clone),
                MarkdownMemory::escape_xml(&sender_id),
                MarkdownMemory::escape_xml(&timestamp),
                reply_to_id.as_ref().map(|r| format!("    <reply_to>{}</reply_to>\n", MarkdownMemory::escape_xml(r))).unwrap_or_default(),
                MarkdownMemory::escape_xml(&content)
            );

            let mut xml = fs::read_to_string(&file_path)
                .unwrap_or_else(|_| "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<messages>\n</messages>\n".to_string());

            let insert_pos = xml.rfind("</messages>")
                .ok_or_else(|| MindroidError::Other(anyhow::Error::msg("Invalid XML: missing </messages> tag")))?;

            xml.insert_str(insert_pos, &message_xml);

            fs::write(&file_path, xml)
                .map_err(|e| MindroidError::Other(anyhow::Error::from(e)))?;
            Ok::<(), MindroidError>(())
        })
        .await
        .map_err(|e| MindroidError::Other(anyhow::Error::from(e)))??;

        Ok(Some(id))
    }

    async fn get_history(&self, channel_id: &str, limit: usize) -> Result<Vec<Message>> {
        let file_path = self.channel_file(channel_id);
        let channel_id = channel_id.to_string();

        let messages = tokio::task::spawn_blocking(move || {
            let xml = fs::read_to_string(&file_path).unwrap_or_default();

            let mut messages = Vec::new();
            let start_tag = "<message>";
            let end_tag = "</message>";

            let mut pos = 0;
            while let Some(start) = xml[pos..].find(start_tag) {
                let start_idx = pos + start;
                if let Some(end) = xml[start_idx..].find(end_tag) {
                    let end_idx = start_idx + end + end_tag.len();
                    let message_block = &xml[start_idx..end_idx];

                    if let (Some(id), Some(sender_id), Some(timestamp), Some(content)) = (
                        MarkdownMemory::parse_xml_element(message_block, "id"),
                        MarkdownMemory::parse_xml_element(message_block, "sender"),
                        MarkdownMemory::parse_xml_element(message_block, "timestamp"),
                        MarkdownMemory::parse_xml_element(message_block, "content"),
                    ) {
                        let ts = chrono::DateTime::parse_from_rfc3339(&timestamp)
                            .map(|dt| dt.with_timezone(&Utc))
                            .unwrap_or_else(|_| Utc::now());

                        messages.push(Message {
                            id,
                            content,
                            sender_id,
                            sender_type: SenderType::default(),
                            channel_id: channel_id.clone(),
                            channel_type: ChannelType::default(),
                            message_type: MessageType::default(),
                            timestamp: ts,
                            metadata: std::collections::HashMap::new(),
                            platform: None,
                        });
                    }

                    pos = end_idx;
                } else {
                    break;
                }
            }

            messages.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
            if messages.len() > limit {
                messages = messages.into_iter().rev().take(limit).collect();
                messages.reverse();
            }

            Ok::<Vec<Message>, MindroidError>(messages)
        })
        .await
        .map_err(|e| MindroidError::Other(anyhow::Error::from(e)))??;

        Ok(messages)
    }

    async fn clear_history(&self, channel_id: &str) -> Result<()> {
        let file_path = self.channel_file(channel_id);

        tokio::task::spawn_blocking(move || {
            fs::remove_file(&file_path).or_else(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Ok(())
                } else {
                    Err(e)
                }
            })
            .map_err(|e| MindroidError::Other(anyhow::Error::from(e)))?;
            Ok::<(), MindroidError>(())
        })
        .await
        .map_err(|e| MindroidError::Other(anyhow::Error::from(e)))??;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn save_and_retrieve() {
        let tmpdir = TempDir::new().unwrap();
        let mem = MarkdownMemory::new(tmpdir.path().to_str().unwrap()).unwrap();
        let id = mem
            .save_message("chan1", "user1", "hello", None)
            .await
            .unwrap();
        assert!(id.is_some());

        let history = mem.get_history("chan1", 10).await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "hello");
        assert_eq!(history[0].sender_id, "user1");
        assert_eq!(history[0].channel_id, "chan1");
    }

    #[tokio::test]
    async fn save_with_special_chars() {
        let tmpdir = TempDir::new().unwrap();
        let mem = MarkdownMemory::new(tmpdir.path().to_str().unwrap()).unwrap();
        mem.save_message("chan1", "user1", "<hello> & \"world\"", None)
            .await
            .unwrap();

        let history = mem.get_history("chan1", 10).await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "<hello> & \"world\"");
    }

    #[tokio::test]
    async fn clear_history() {
        let tmpdir = TempDir::new().unwrap();
        let mem = MarkdownMemory::new(tmpdir.path().to_str().unwrap()).unwrap();
        mem.save_message("chan1", "user1", "msg1", None)
            .await
            .unwrap();
        mem.save_message("chan1", "user1", "msg2", None)
            .await
            .unwrap();

        mem.clear_history("chan1").await.unwrap();

        let history = mem.get_history("chan1", 10).await.unwrap();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn history_ordering() {
        let tmpdir = TempDir::new().unwrap();
        let mem = MarkdownMemory::new(tmpdir.path().to_str().unwrap()).unwrap();
        mem.save_message("chan1", "user1", "first", None)
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        mem.save_message("chan1", "user1", "second", None)
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        mem.save_message("chan1", "user1", "third", None)
            .await
            .unwrap();

        let history = mem.get_history("chan1", 10).await.unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].content, "first");
        assert_eq!(history[1].content, "second");
        assert_eq!(history[2].content, "third");
    }

    #[tokio::test]
    async fn history_limit() {
        let tmpdir = TempDir::new().unwrap();
        let mem = MarkdownMemory::new(tmpdir.path().to_str().unwrap()).unwrap();
        for i in 0..5 {
            mem.save_message("chan1", "user1", &format!("msg{}", i), None)
                .await
                .unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        let history = mem.get_history("chan1", 3).await.unwrap();
        assert_eq!(history.len(), 3);
    }
}
