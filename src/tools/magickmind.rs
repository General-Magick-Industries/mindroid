//! Magickmind-backed tools.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{NaiveDate, TimeDelta, Utc};
use serde_json::{Value, json};

use crate::auth::Auth;
use crate::core::net::{error_excerpt, require_secure_url, secure_json_client};
use crate::error::{MindroidError, Result};
use crate::models::CredentialKind;
use crate::tools::{Tool, ToolContext};

/// The agent's credential, placed into [`ToolContext`] by a pipeline stage so
/// magickmind tools can call the backend as the agent. `auth.get_token()` refreshes
/// on demand, so it stays current across the run.
#[derive(Clone)]
pub struct AgentCredentials {
    pub agent_id: String,
    pub auth: Arc<dyn Auth>,
    pub credential_kind: CredentialKind,
}

/// Pipeline stage that puts [`AgentCredentials`] into the run scope so the tool
/// executor's [`ToolContext`] carries them. Place before the tool executor.
pub struct AgentCredentialsStage {
    credentials: AgentCredentials,
}

impl AgentCredentialsStage {
    pub fn new(credentials: AgentCredentials) -> Self {
        Self { credentials }
    }
}

#[async_trait]
impl crate::pipeline::PipelineStage for AgentCredentialsStage {
    fn name(&self) -> &str {
        "AgentCredentials"
    }

    async fn process(&self, ctx: &mut crate::core::context::Context) -> Result<()> {
        let tc = ctx.get_run::<ToolContext>().cloned().unwrap_or_default();
        tc.set(self.credentials.clone());
        ctx.set(tc);
        Ok(())
    }
}

/// Searches the agent's episodic memory via the backend and returns matching
/// episodes as text for the LLM.
pub struct EpisodicMemoryTool {
    base_url: String,
    allow_insecure: bool,
}

impl EpisodicMemoryTool {
    const HTTP_TIMEOUT_SECS: u64 = 10;

    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            allow_insecure: false,
        }
    }

    /// Permit sending auth headers over plaintext `http://` (local dev only).
    pub fn with_allow_insecure(mut self, allow_insecure: bool) -> Self {
        self.allow_insecure = allow_insecure;
        self
    }

    fn search_url(&self, kind: CredentialKind) -> String {
        match kind {
            CredentialKind::EndUser => format!("{}/v1/end-user/episodes/search", self.base_url),
            CredentialKind::ServiceUser => format!("{}/v1/episodes/search", self.base_url),
        }
    }
}

#[async_trait]
impl Tool for EpisodicMemoryTool {
    fn name(&self) -> &str {
        "search_episodic_memory"
    }

    fn description(&self) -> &str {
        "Search your episodic memory for past conversations relevant to a query. \
         Returns summarized episodes (topic, what happened, what worked, what to avoid). \
         By default only this conversation's memories are searched; disable \
         filter_by_session to search across all your conversations, and enable \
         filter_by_sender to only see memories from conversations the current \
         speaker took part in."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "What to search memory for." },
                "limit": { "type": "integer", "description": "Max episodes (optional)." },
                "filter_by_session": {
                    "type": "boolean",
                    "description": "Restrict to the current conversation space (default true)."
                },
                "filter_by_sender": {
                    "type": "boolean",
                    "description": "Restrict to conversations the current speaker took part in (default false)."
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<String> {
        let creds = ctx.get::<AgentCredentials>().ok_or_else(|| {
            MindroidError::config("search_episodic_memory needs AgentCredentials in ToolContext")
        })?;

        let query = args["query"].as_str().unwrap_or("").trim();
        if query.is_empty() {
            return Err(MindroidError::config("query is required"));
        }

        require_secure_url(&self.base_url, self.allow_insecure, "allow_insecure")?;

        let url = self.search_url(creds.credential_kind);
        let mut params: Vec<(&str, String)> = vec![("q", query.to_string())];
        params.extend(scope_params(&args, ctx));
        if let Some(limit) = args["limit"].as_i64() {
            params.push(("limit", limit.to_string()));
        }

        let headers = crate::auth::build_auth_header_map(creds.auth.as_ref()).await?;
        let resp = secure_json_client(Duration::from_secs(Self::HTTP_TIMEOUT_SECS))
            .get(&url)
            .headers(headers)
            .query(&params)
            .send()
            .await
            .map_err(|e| MindroidError::Api {
                message: e.to_string(),
                status_code: None,
            })?;

        let status = resp.status();
        crate::core::net::note_auth_status(creds.auth.as_ref(), status);
        if !status.is_success() {
            let text = error_excerpt(&resp.text().await.unwrap_or_default());
            return Err(MindroidError::Api {
                message: format!("episodic search failed: {text}"),
                status_code: Some(status.as_u16()),
            });
        }

        let body: SearchResponse = resp.json().await.map_err(|e| MindroidError::Api {
            message: format!("failed to parse episodic search response: {e}"),
            status_code: None,
        })?;

        Ok(if body.memory_content.trim().is_empty() {
            "No relevant memories found.".to_string()
        } else {
            body.memory_content
        })
    }
}

/// Scope filters for the search. The model chooses only WHETHER to filter;
/// the values come from the trusted per-message context, never from args
/// (ADR-0005) — so it cannot search another space or as another speaker.
/// The mindspace-scoped route reads `participant_id`; the cross-space route
/// reads `user_id`, hence the key switch.
fn scope_params(args: &Value, ctx: &ToolContext) -> Vec<(&'static str, String)> {
    let by_session = args["filter_by_session"].as_bool().unwrap_or(true);
    let by_sender = args["filter_by_sender"].as_bool().unwrap_or(false);
    let session_scoped = by_session && !ctx.channel_id.is_empty();

    let mut params = Vec::new();
    if session_scoped {
        params.push(("mindspace_id", ctx.channel_id.clone()));
    }
    if by_sender && !ctx.sender_id.is_empty() {
        let key = if session_scoped {
            "participant_id"
        } else {
            "user_id"
        };
        params.push((key, ctx.sender_id.clone()));
    }
    params
}

#[derive(serde::Deserialize)]
struct SearchResponse {
    #[serde(default)]
    memory_content: String,
}

/// Recalls episodes in a date window. Search ranks by relevance and cannot
/// filter on time -- no timestamp is indexed -- so a window is a separate call.
pub struct RecallTimeWindowTool {
    base_url: String,
    allow_insecure: bool,
}

const DATE_FORMAT: &str = "%Y-%m-%d";

impl RecallTimeWindowTool {
    const HTTP_TIMEOUT_SECS: u64 = 10;

    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            allow_insecure: false,
        }
    }

    /// Permit sending auth headers over plaintext `http://` (local dev only).
    pub fn with_allow_insecure(mut self, allow_insecure: bool) -> Self {
        self.allow_insecure = allow_insecure;
        self
    }

    /// End-user route only: the window is scoped to the token subject, so there
    /// is no owner for a caller to name.
    fn range_url(&self) -> String {
        format!("{}/v1/end-user/episodes/range", self.base_url)
    }
}

#[async_trait]
impl Tool for RecallTimeWindowTool {
    fn name(&self) -> &str {
        "recall_time_window"
    }

    fn description(&self) -> &str {
        "Recall what happened in a recent span of time, e.g. yesterday, the last \n         week, or the week before last. Use this instead of search_episodic_memory \n         whenever the question is about WHEN something happened rather than what it \n         was about -- memory search ranks by relevance and cannot filter by time. \n         The window is given in days back from now: yesterday is from 1 to 1, the \n         last week is from 7 to 0, the week before last is from 14 to 7. \n         Returns the episodes in the window, newest first."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "from_days_ago": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Start of the window, in days back from today. 7 = a week ago."
                },
                "to_days_ago": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "End of the window, in days back from today. Omit for today."
                },
                "limit": { "type": "integer", "description": "Max episodes (optional)." },
                "filter_by_session": {
                    "type": "boolean",
                    "description": "Restrict to the current conversation space (default true)."
                }
            },
            "required": ["from_days_ago"]
        })
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<String> {
        let creds = ctx.get::<AgentCredentials>().ok_or_else(|| {
            MindroidError::config("recall_time_window needs AgentCredentials in ToolContext")
        })?;

        if creds.credential_kind != CredentialKind::EndUser {
            return Err(MindroidError::config(
                "recall_time_window requires an end-user credential; the range route has no service-user form",
            ));
        }

        let (date_start, date_end) = window_from_args(&args, Utc::now().date_naive())?;

        require_secure_url(&self.base_url, self.allow_insecure, "allow_insecure")?;

        let mut params: Vec<(&str, String)> =
            vec![("date_start", date_start), ("date_end", date_end)];
        params.extend(session_param(&args, ctx));
        if let Some(limit) = args["limit"].as_i64().filter(|n| *n > 0) {
            params.push(("limit", limit.to_string()));
        }

        let headers = crate::auth::build_auth_header_map(creds.auth.as_ref()).await?;
        let resp = secure_json_client(Duration::from_secs(Self::HTTP_TIMEOUT_SECS))
            .get(self.range_url())
            .headers(headers)
            .query(&params)
            .send()
            .await
            .map_err(|e| MindroidError::Api {
                message: e.to_string(),
                status_code: None,
            })?;

        let status = resp.status();
        crate::core::net::note_auth_status(creds.auth.as_ref(), status);
        if !status.is_success() {
            let text = error_excerpt(&resp.text().await.unwrap_or_default());
            return Err(MindroidError::Api {
                message: format!("episode window lookup failed: {text}"),
                status_code: Some(status.as_u16()),
            });
        }

        let body: RangeResponse = resp.json().await.map_err(|e| MindroidError::Api {
            message: format!("failed to parse episode window response: {e}"),
            status_code: None,
        })?;

        Ok(render_episodes(&body.data))
    }
}

/// Scope for the window. The model chooses only WHETHER to narrow to the current
/// space; the id comes from the trusted per-message context, never from args
/// (ADR-0005), so it cannot aim recall at another space.
fn session_param(args: &Value, ctx: &ToolContext) -> Option<(&'static str, String)> {
    let by_session = args["filter_by_session"].as_bool().unwrap_or(true);
    (by_session && !ctx.channel_id.is_empty()).then(|| ("mindspace_id", ctx.channel_id.clone()))
}

/// Turns the model's relative offsets into the absolute window the API wants.
/// The model never sees or supplies a date: it has no reliable notion of today,
/// so letting it name one produces a plausible window from its training cutoff
/// and an empty result that reads as "nothing happened".
fn window_from_args(args: &Value, today: NaiveDate) -> Result<(String, String)> {
    let from = args["from_days_ago"].as_i64().ok_or_else(|| {
        MindroidError::config("from_days_ago is required, as a whole number of days")
    })?;
    let to = args["to_days_ago"].as_i64().unwrap_or(0);
    if from < 0 || to < 0 {
        return Err(MindroidError::config(
            "day offsets count backwards from today and cannot be negative",
        ));
    }
    if to > from {
        return Err(MindroidError::config(
            "to_days_ago must not be further back than from_days_ago",
        ));
    }

    let start = today
        .checked_sub_signed(TimeDelta::days(from))
        .ok_or_else(|| MindroidError::config("from_days_ago is too far in the past"))?;
    let end = today
        .checked_sub_signed(TimeDelta::days(to))
        .ok_or_else(|| MindroidError::config("to_days_ago is too far in the past"))?;

    Ok((
        start.format(DATE_FORMAT).to_string(),
        end.format(DATE_FORMAT).to_string(),
    ))
}

fn render_episodes(episodes: &[Episode]) -> String {
    if episodes.is_empty() {
        return "No episodes in that window.".to_string();
    }
    episodes
        .iter()
        .map(|e| {
            format!(
                "Topic: {}\nSubtopics: {}\nSummary: {}",
                e.topic,
                e.subtopics.join(", "),
                e.summarized_conversation
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
struct RangeResponse {
    #[serde(default)]
    data: Vec<Episode>,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
struct Episode {
    #[serde(default)]
    topic: String,
    #[serde(default)]
    subtopics: Vec<String>,
    #[serde(default)]
    summarized_conversation: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_url_is_end_user_only() {
        let tool = RecallTimeWindowTool::new("https://api.example.com/");
        assert_eq!(
            tool.range_url(),
            "https://api.example.com/v1/end-user/episodes/range"
        );
    }

    #[test]
    fn render_episodes_reports_an_empty_window() {
        assert_eq!(render_episodes(&[]), "No episodes in that window.");
    }

    #[test]
    fn render_episodes_keeps_topic_subtopics_and_summary() {
        let rendered = render_episodes(&[Episode {
            topic: "Garden planning".into(),
            subtopics: vec!["tomatoes".into(), "watering".into()],
            summarized_conversation: "Agreed to start seedlings indoors.".into(),
        }]);
        assert_eq!(
            rendered,
            "Topic: Garden planning
Subtopics: tomatoes, watering
Summary: Agreed to start seedlings indoors."
        );
    }

    fn day(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).expect("valid date")
    }

    #[test]
    fn window_counts_backwards_from_today() {
        let (start, end) =
            window_from_args(&json!({"from_days_ago": 7}), day(2026, 8, 20)).unwrap();
        assert_eq!((start.as_str(), end.as_str()), ("2026-08-13", "2026-08-20"));
    }

    #[test]
    fn window_supports_a_span_that_does_not_reach_today() {
        let (start, end) = window_from_args(
            &json!({"from_days_ago": 14, "to_days_ago": 7}),
            day(2026, 8, 20),
        )
        .unwrap();
        assert_eq!((start.as_str(), end.as_str()), ("2026-08-06", "2026-08-13"));
    }

    #[test]
    fn window_of_one_day_is_that_day() {
        let (start, end) = window_from_args(
            &json!({"from_days_ago": 1, "to_days_ago": 1}),
            day(2026, 8, 20),
        )
        .unwrap();
        assert_eq!((start.as_str(), end.as_str()), ("2026-08-19", "2026-08-19"));
    }

    #[test]
    fn window_crosses_a_month_boundary() {
        let (start, _) = window_from_args(&json!({"from_days_ago": 5}), day(2026, 3, 2)).unwrap();
        assert_eq!(start, "2026-02-25");
    }

    #[test]
    fn window_rejects_a_reversed_or_negative_span() {
        for bad in [
            json!({"from_days_ago": 3, "to_days_ago": 10}),
            json!({"from_days_ago": -1}),
            json!({"from_days_ago": 3, "to_days_ago": -1}),
        ] {
            assert!(
                window_from_args(&bad, day(2026, 8, 20)).is_err(),
                "accepted {bad}"
            );
        }
    }

    #[test]
    fn window_requires_the_start_offset() {
        for missing in [
            json!({}),
            json!({"to_days_ago": 3}),
            json!({"from_days_ago": "7"}),
        ] {
            assert!(
                window_from_args(&missing, day(2026, 8, 20)).is_err(),
                "accepted {missing}"
            );
        }
    }

    // ADR-0005: the model picks WHETHER to scope, never WHERE. A refactor that
    // read the space from args would silently widen recall to another space.
    #[test]
    fn window_scope_ignores_a_model_supplied_space() {
        let ctx = ctx_with("ms-1", "sender-1");
        let got = session_param(
            &json!({"mindspace_id": "other-space", "channel_id": "other-space"}),
            &ctx,
        );
        assert_eq!(got, Some(("mindspace_id", "ms-1".to_string())));
    }

    #[test]
    fn window_scope_can_be_widened_but_never_redirected() {
        let ctx = ctx_with("ms-1", "sender-1");
        assert_eq!(
            session_param(
                &json!({"filter_by_session": false, "mindspace_id": "other"}),
                &ctx
            ),
            None
        );
    }

    #[test]
    fn window_scope_is_omitted_without_a_channel() {
        assert_eq!(session_param(&json!({}), &ctx_with("", "sender-1")), None);
    }

    #[test]
    fn search_url_routes_by_credential_kind() {
        let tool = EpisodicMemoryTool::new("https://api.example.com/");
        assert_eq!(
            tool.search_url(CredentialKind::EndUser),
            "https://api.example.com/v1/end-user/episodes/search"
        );
        assert_eq!(
            tool.search_url(CredentialKind::ServiceUser),
            "https://api.example.com/v1/episodes/search"
        );
    }

    #[test]
    fn new_trims_trailing_slash() {
        let tool = EpisodicMemoryTool::new("https://api.example.com//");
        assert_eq!(
            tool.search_url(CredentialKind::ServiceUser),
            "https://api.example.com/v1/episodes/search"
        );
    }

    fn ctx_with(channel: &str, sender: &str) -> ToolContext {
        ToolContext {
            channel_id: channel.into(),
            sender_id: sender.into(),
            ..Default::default()
        }
    }

    #[test]
    fn scope_defaults_to_the_current_session_only() {
        let params = scope_params(&json!({"query": "q"}), &ctx_with("ms-1", "user-1"));
        assert_eq!(params, vec![("mindspace_id", "ms-1".to_string())]);
    }

    #[test]
    fn sender_filter_rides_the_session_scope_as_participant() {
        let params = scope_params(
            &json!({"filter_by_sender": true}),
            &ctx_with("ms-1", "user-1"),
        );
        assert_eq!(
            params,
            vec![
                ("mindspace_id", "ms-1".to_string()),
                ("participant_id", "user-1".to_string()),
            ]
        );
    }

    #[test]
    fn cross_session_sender_filter_uses_the_user_lens() {
        let params = scope_params(
            &json!({"filter_by_session": false, "filter_by_sender": true}),
            &ctx_with("ms-1", "user-1"),
        );
        assert_eq!(params, vec![("user_id", "user-1".to_string())]);
    }

    /// The values never come from args: a model naming ids gets them ignored.
    #[test]
    fn model_supplied_identities_are_ignored() {
        let params = scope_params(
            &json!({"filter_by_sender": true, "participant_id": "victim", "mindspace_id": "other"}),
            &ctx_with("ms-1", "user-1"),
        );
        assert_eq!(
            params,
            vec![
                ("mindspace_id", "ms-1".to_string()),
                ("participant_id", "user-1".to_string()),
            ]
        );
    }

    #[test]
    fn blank_context_values_add_no_filters() {
        let params = scope_params(
            &json!({"filter_by_session": true, "filter_by_sender": true}),
            &ctx_with("", ""),
        );
        assert!(params.is_empty());
    }
}
