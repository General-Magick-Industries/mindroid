use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use lru::LruCache;
use tokio::sync::Mutex;

const MAX_ENGAGEMENTS: usize = 10_000;

/// Tracks per-channel engagement to prevent feedback loops in multi-agent conversations.
///
/// When multiple agents share a channel, they can get into loops where each agent
/// responds to the other's messages. The tracker prevents this by remembering which
/// `sender_id` the agent is currently engaged with per channel. Messages from other
/// senders are skipped until the engagement expires.
///
/// This works without knowing who is an agent — it simply says: "I'm talking to
/// sender X right now; messages from other senders are ignored until the cooldown
/// passes."
///
/// # Example
///
/// ```ignore
/// use std::time::Duration;
/// use mindroid::EngagementTracker;
///
/// let tracker = Arc::new(EngagementTracker::new(Duration::from_secs(120)));
///
/// builder.on_message(move |ctx| {
///     let tracker = Arc::clone(&tracker);
///     async move {
///         if !tracker.should_engage(&ctx.message.channel_id, &ctx.message.sender_id).await {
///             return; // Another sender is active — likely another agent
///         }
///
///         // ... run pipelines, send response ...
///
///         tracker.record(&ctx.message.channel_id, &ctx.message.sender_id).await;
///     }
/// });
/// ```
pub struct EngagementTracker {
    engagements: Mutex<LruCache<String, Engagement>>,
    cooldown: Duration,
}

struct Engagement {
    sender_id: String,
    responded_at: Instant,
}

impl EngagementTracker {
    pub fn new(cooldown: Duration) -> Self {
        Self {
            engagements: Mutex::new(LruCache::new(
                NonZeroUsize::new(MAX_ENGAGEMENTS).unwrap(),
            )),
            cooldown,
        }
    }

    /// Check if the agent should engage with this message.
    ///
    /// Returns `true` if:
    /// - No active engagement in this channel (new conversation)
    /// - The sender matches the active engagement (follow-up from the same user)
    /// - The active engagement has expired (cooldown passed)
    ///
    /// Returns `false` if:
    /// - There's an active, unexpired engagement with a **different** sender
    pub async fn should_engage(&self, channel_id: &str, sender_id: &str) -> bool {
        let mut engagements = self.engagements.lock().await;

        match engagements.get(channel_id) {
            Some(engagement) => {
                if engagement.responded_at.elapsed() >= self.cooldown
                    || engagement.sender_id == sender_id
                {
                    true // Engagement expired or same sender — likely a follow-up
                } else {
                    tracing::debug!(
                        "EngagementTracker: ignoring message from '{}' in channel '{}' \
                         (engaged with '{}', {}s remaining)",
                        sender_id,
                        channel_id,
                        engagement.sender_id,
                        (self.cooldown - engagement.responded_at.elapsed()).as_secs(),
                    );
                    false // Different sender during active engagement
                }
            }
            None => true, // No engagement — new conversation
        }
    }

    /// Record that the agent responded to a sender in a channel.
    ///
    /// Call this **after** the agent has sent a response, not before.
    /// This ensures only successful engagements are tracked.
    pub async fn record(&self, channel_id: &str, sender_id: &str) {
        let mut engagements = self.engagements.lock().await;
        engagements.put(
            channel_id.to_string(),
            Engagement {
                sender_id: sender_id.to_string(),
                responded_at: Instant::now(),
            },
        );
    }

    /// Clear engagement for a channel.
    ///
    /// Use when the conversation is complete or the agent should
    /// be open to engaging with new senders.
    pub async fn clear(&self, channel_id: &str) {
        let mut engagements = self.engagements.lock().await;
        engagements.pop(channel_id);
    }
}
