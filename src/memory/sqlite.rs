// SQLite-backed memory for Mindroid

use async_trait::async_trait;
use chrono::Utc;
use rusqlite::Connection;
use std::sync::{Arc, Mutex};

use crate::{
    ChannelType, Memory, Message, MessageType, MindroidError, Result, SenderType,
};

pub struct SqliteMemory {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteMemory {
    pub fn new(path: &str) -> Result<Self> {
        let conn = Connection::open(path).map_err(|e| MindroidError::Other(anyhow::Error::from(e)))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                content TEXT NOT NULL,
                reply_to_id TEXT,
                timestamp TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel_id, timestamp);",
        )
        .map_err(|e| MindroidError::Other(anyhow::Error::from(e)))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }
}

#[async_trait]
impl Memory for SqliteMemory {
    async fn save_message(
        &self,
        channel_id: &str,
        sender_id: &str,
        content: &str,
        reply_to_id: Option<&str>,
    ) -> Result<Option<String>> {
        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = Utc::now().to_rfc3339();
        let conn = Arc::clone(&self.conn);
        let id_clone = id.clone();
        let channel_id = channel_id.to_string();
        let sender_id = sender_id.to_string();
        let content = content.to_string();
        let reply_to_id = reply_to_id.map(|s| s.to_string());

        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| {
                MindroidError::Other(anyhow::anyhow!("Mutex poisoned: {}", e))
            })?;
            conn.execute(
                "INSERT INTO messages (id, channel_id, sender_id, content, reply_to_id, timestamp) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![id_clone, channel_id, sender_id, content, reply_to_id, timestamp],
            )
            .map_err(|e| MindroidError::Other(anyhow::Error::from(e)))?;
            Ok::<(), MindroidError>(())
        })
        .await
        .map_err(|e| MindroidError::Other(anyhow::Error::from(e)))??;

        Ok(Some(id))
    }

    async fn get_history(&self, channel_id: &str, limit: usize) -> Result<Vec<Message>> {
        let conn = Arc::clone(&self.conn);
        let channel_id = channel_id.to_string();

        let rows = tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| {
                MindroidError::Other(anyhow::anyhow!("Mutex poisoned: {}", e))
            })?;
            let mut stmt = conn
                .prepare(
                    "SELECT id, channel_id, sender_id, content, reply_to_id, timestamp \
                     FROM messages WHERE channel_id = ?1 ORDER BY timestamp DESC LIMIT ?2",
                )
                .map_err(|e| MindroidError::Other(anyhow::Error::from(e)))?;

            let rows: std::result::Result<Vec<_>, _> = stmt
                .query_map(rusqlite::params![channel_id, limit as i64], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, String>(5)?,
                    ))
                })
                .map_err(|e| MindroidError::Other(anyhow::Error::from(e)))?
                .collect();

            rows.map_err(|e| MindroidError::Other(anyhow::Error::from(e)))
        })
        .await
        .map_err(|e| MindroidError::Other(anyhow::Error::from(e)))??;

        let mut messages: Vec<Message> = rows
            .into_iter()
            .map(|(id, channel_id, sender_id, content, _reply_to_id, timestamp)| {
                let ts = chrono::DateTime::parse_from_rfc3339(&timestamp)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());
                Message {
                    id,
                    content,
                    sender_id,
                    sender_type: SenderType::default(),
                    channel_id,
                    channel_type: ChannelType::default(),
                    message_type: MessageType::default(),
                    timestamp: ts,
                    metadata: std::collections::HashMap::new(),
                    platform: None,
                }
            })
            .collect();

        messages.reverse();
        Ok(messages)
    }

    async fn clear_history(&self, channel_id: &str) -> Result<()> {
        let conn = Arc::clone(&self.conn);
        let channel_id = channel_id.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(|e| {
                MindroidError::Other(anyhow::anyhow!("Mutex poisoned: {}", e))
            })?;
            conn.execute(
                "DELETE FROM messages WHERE channel_id = ?1",
                rusqlite::params![channel_id],
            )
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

    #[tokio::test]
    async fn save_and_retrieve() {
        let mem = SqliteMemory::new(":memory:").unwrap();
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
    async fn clear_history() {
        let mem = SqliteMemory::new(":memory:").unwrap();
        mem.save_message("chan1", "user1", "msg1", None).await.unwrap();
        mem.save_message("chan1", "user1", "msg2", None).await.unwrap();

        mem.clear_history("chan1").await.unwrap();

        let history = mem.get_history("chan1", 10).await.unwrap();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn history_ordering() {
        let mem = SqliteMemory::new(":memory:").unwrap();
        mem.save_message("chan1", "user1", "first", None).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        mem.save_message("chan1", "user1", "second", None).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        mem.save_message("chan1", "user1", "third", None).await.unwrap();

        let history = mem.get_history("chan1", 10).await.unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].content, "first");
        assert_eq!(history[1].content, "second");
        assert_eq!(history[2].content, "third");
    }

    #[tokio::test]
    async fn history_limit() {
        let mem = SqliteMemory::new(":memory:").unwrap();
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
