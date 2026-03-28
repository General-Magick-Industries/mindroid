//! Reminder tool and routine for time-delayed notifications.
//!
//! The LLM calls `set_reminder` to schedule a reminder. The `ReminderRoutine`
//! checks every second for due reminders and sends them through the transport.

use async_trait::async_trait;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tokio::sync::RwLock;

use crate::core::routine::{Routine, RoutineContext};
use crate::error::Result;
use crate::models::Response;
use crate::tools::Tool;


/// A single scheduled reminder.
#[derive(Debug, Clone)]
pub struct Reminder {
    pub message: String,
    pub due_at: Instant,
    pub created_at: Instant,
}

/// Shared reminder storage used by both the tool and the routine.
pub type ReminderStore = Arc<RwLock<Vec<Reminder>>>;

/// Create a new empty reminder store. Pass this to both `SetReminderTool` and `ReminderRoutine`.
pub fn new_reminder_store() -> ReminderStore {
    Arc::new(RwLock::new(Vec::new()))
}

// ---------------------------------------------------------------------------
// SetReminderTool
// ---------------------------------------------------------------------------

/// Tool that lets the LLM schedule a reminder for the user.
pub struct SetReminderTool {
    store: ReminderStore,
}

impl SetReminderTool {
    pub fn new(store: ReminderStore) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for SetReminderTool {
    fn name(&self) -> &str {
        "set_reminder"
    }

    fn description(&self) -> &str {
        "Set a reminder that will be delivered after a specified delay. \
         Use this when the user asks to be reminded of something."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The reminder message to deliver"
                },
                "delay_seconds": {
                    "type": "number",
                    "description": "Number of seconds to wait before delivering the reminder"
                }
            },
            "required": ["message", "delay_seconds"]
        })
    }

    async fn execute(&self, args: Value) -> Result<String> {
        let message = args
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let delay_secs = args
            .get("delay_seconds")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as u64;

        let now = Instant::now();
        let reminder = Reminder {
            message: message.clone(),
            due_at: now + Duration::from_secs(delay_secs),
            created_at: now,
        };

        self.store.write().await.push(reminder);

        Ok(format!("Reminder set: '{message}' in {delay_secs} seconds."))
    }
}

// ---------------------------------------------------------------------------
// ReminderRoutine
// ---------------------------------------------------------------------------

/// Routine that checks every second for due reminders and fires them.
pub struct ReminderRoutine {
    store: ReminderStore,
}

impl ReminderRoutine {
    pub fn new(store: ReminderStore) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Routine for ReminderRoutine {
    fn name(&self) -> &str {
        "reminders"
    }

    fn interval(&self) -> Duration {
        Duration::from_secs(1)
    }

    async fn poll(&self) -> Result<Option<String>> {
        let now = Instant::now();
        let mut store = self.store.write().await;

        let mut due_messages: Vec<String> = Vec::new();
        store.retain(|reminder| {
            if now >= reminder.due_at {
                due_messages.push(reminder.message.clone());
                false // remove from store
            } else {
                true // keep
            }
        });

        if due_messages.is_empty() {
            Ok(None)
        } else {
            Ok(Some(due_messages.join("\n")))
        }
    }

    async fn act(&self, ctx: RoutineContext, data: String) -> Result<()> {
        let response = Response::new(data, "reminder", &ctx.agent_config.agent_id);
        ctx.send(&response).await?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_set_reminder_stores_correctly() {
        let store = new_reminder_store();
        let tool = SetReminderTool::new(store.clone());

        let args = json!({ "message": "Take a break", "delay_seconds": 60 });
        let result = tool.execute(args).await.unwrap();

        assert!(result.contains("Take a break"));
        assert!(result.contains("60 seconds"));

        let reminders = store.read().await;
        assert_eq!(reminders.len(), 1);
        assert_eq!(reminders[0].message, "Take a break");
    }

    #[tokio::test]
    async fn test_poll_returns_none_when_no_reminders_due() {
        let store = new_reminder_store();
        {
            let mut s = store.write().await;
            s.push(Reminder {
                message: "Future reminder".to_string(),
                due_at: Instant::now() + Duration::from_secs(9999),
                created_at: Instant::now(),
            });
        }

        let routine = ReminderRoutine::new(store.clone());
        let result = routine.poll().await.unwrap();
        assert!(result.is_none());

        // Reminder should still be in store
        let reminders = store.read().await;
        assert_eq!(reminders.len(), 1);
    }

    #[tokio::test]
    async fn test_poll_returns_and_removes_due_reminders() {
        let store = new_reminder_store();
        {
            let mut s = store.write().await;
            // Already past due
            s.push(Reminder {
                message: "Past due".to_string(),
                due_at: Instant::now() - Duration::from_secs(1),
                created_at: Instant::now() - Duration::from_secs(10),
            });
            // Not yet due
            s.push(Reminder {
                message: "Future".to_string(),
                due_at: Instant::now() + Duration::from_secs(9999),
                created_at: Instant::now(),
            });
        }

        let routine = ReminderRoutine::new(store.clone());
        let result = routine.poll().await.unwrap();

        assert!(result.is_some());
        assert_eq!(result.unwrap(), "Past due");

        // Only the future reminder should remain
        let reminders = store.read().await;
        assert_eq!(reminders.len(), 1);
        assert_eq!(reminders[0].message, "Future");
    }

    #[test]
    fn test_parameters_schema_is_valid() {
        let store = new_reminder_store();
        let tool = SetReminderTool::new(store);
        let schema = tool.parameters_schema();

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["message"].is_object());
        assert!(schema["properties"]["delay_seconds"].is_object());

        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "message"));
        assert!(required.iter().any(|v| v == "delay_seconds"));
    }
}
