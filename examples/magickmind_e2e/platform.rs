use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use reqwest::{Client, Method, StatusCode};
use serde_json::{Value, json};

#[derive(Debug)]
pub struct Reply {
    pub status: StatusCode,
    pub trace_id: String,
    pub body: Value,
}

impl Reply {
    pub fn ok(self, what: &str) -> Result<Value> {
        if !self.status.is_success() {
            bail!(
                "{what} returned {} (trace {}): {}",
                self.status,
                self.trace_id,
                self.body
            );
        }
        Ok(self.body)
    }
}

pub struct Bifrost {
    http: Client,
    base: String,
}

impl Bifrost {
    pub fn new(base: &str) -> Result<Self> {
        let http = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(20))
            .build()?;
        Ok(Self {
            http,
            base: base.trim_end_matches('/').to_string(),
        })
    }

    async fn call(
        &self,
        method: Method,
        path: &str,
        query: &[(&str, &str)],
        bearer: Option<&str>,
        body: Option<&Value>,
    ) -> Result<Reply> {
        let trace_id = uuid::Uuid::new_v4().simple().to_string();
        let span_id = &uuid::Uuid::new_v4().simple().to_string()[..16];
        let mut req = self
            .http
            .request(method.clone(), format!("{}{path}", self.base))
            .header("traceparent", format!("00-{trace_id}-{span_id}-01"));
        if !query.is_empty() {
            req = req.query(query);
        }
        if let Some(token) = bearer {
            req = req.bearer_auth(token);
        }
        if let Some(body) = body {
            req = req.json(body);
        }
        let resp = req
            .send()
            .await
            .with_context(|| format!("{method} {path} failed to send"))?;
        let status = resp.status();
        let trace_id = resp
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .filter(|v| !v.is_empty())
            .map_or(trace_id, str::to_string);
        let text = resp
            .text()
            .await
            .with_context(|| format!("{method} {path}: reading the response"))?;
        let body = serde_json::from_str(&text).unwrap_or(Value::String(text));
        Ok(Reply {
            status,
            trace_id,
            body,
        })
    }

    pub async fn login(&self, email: &str, password: &str) -> Result<String> {
        let body = self
            .call(
                Method::POST,
                "/v1/auth/login",
                &[],
                None,
                Some(&json!({ "email": email, "password": password })),
            )
            .await?
            .ok("service-user login")?;
        body["access_token"]
            .as_str()
            .map(str::to_string)
            .context("login response had no access_token")
    }

    pub async fn ensure_end_user(
        &self,
        jwt: &str,
        external_id: &str,
        name: &str,
        participant_type: &str,
    ) -> Result<String> {
        let page = self
            .call(
                Method::GET,
                "/v1/end-users",
                &[("external_id", external_id), ("limit", "20")],
                Some(jwt),
                None,
            )
            .await?
            .ok("listing end users")?;
        let existing = page["data"].as_array().and_then(|users| {
            users
                .iter()
                .find(|u| u["external_id"].as_str() == Some(external_id))
        });
        if let Some(user) = existing {
            if user["participant_type"].as_str() != Some(participant_type) {
                bail!(
                    "fixture {external_id} exists as {} but the harness needs {participant_type}",
                    user["participant_type"]
                );
            }
            return id_of(user, "end user");
        }
        let created = self
            .call(
                Method::POST,
                "/v1/end-users",
                &[],
                Some(jwt),
                Some(&json!({
                    "name": name,
                    "external_id": external_id,
                    "participant_type": participant_type,
                })),
            )
            .await?
            .ok("creating end user")?;
        id_of(&created, "end user")
    }

    pub async fn mint(&self, jwt: &str, subject_id: &str) -> Result<String> {
        let body = self
            .call(
                Method::POST,
                "/v1/end-users/tokens",
                &[],
                Some(jwt),
                Some(&json!({
                    "subject_id": subject_id,
                    "supervised": true,
                    "ttl_seconds": 3600,
                })),
            )
            .await?
            .ok("minting an end-user token")?;
        body["token"]
            .as_str()
            .map(str::to_string)
            .context("mint response had no token")
    }

    pub async fn create_space(
        &self,
        jwt: &str,
        name: &str,
        project_id: &str,
        participants: &[&str],
    ) -> Result<String> {
        let body = self
            .call(
                Method::POST,
                "/v1/magickspaces",
                &[],
                Some(jwt),
                Some(&json!({
                    "name": name,
                    "type": "GROUP",
                    "project_id": project_id,
                    "participant_ids": participants,
                })),
            )
            .await?
            .ok("creating the magickspace")?;
        id_of(&body, "magickspace")
    }

    pub async fn delete_space(&self, jwt: &str, space_id: &str) -> Result<()> {
        self.call(
            Method::DELETE,
            &format!("/v1/magickspaces/{space_id}"),
            &[],
            Some(jwt),
            None,
        )
        .await?
        .ok("deleting the magickspace")?;
        Ok(())
    }

    pub async fn send(&self, token: &str, space_id: &str, body: &Value) -> Result<Reply> {
        self.call(
            Method::POST,
            &format!("/v1/end-user/magickspaces/{space_id}/messages"),
            &[],
            Some(token),
            Some(body),
        )
        .await
    }

    pub async fn list_messages(&self, token: &str, space_id: &str) -> Result<Reply> {
        self.call(
            Method::GET,
            &format!("/v1/end-user/magickspaces/{space_id}/messages"),
            &[("limit", "100"), ("order", "asc")],
            Some(token),
            None,
        )
        .await
    }

    pub async fn prepare_context(&self, token: &str, space_id: &str) -> Result<Reply> {
        self.call(
            Method::POST,
            &format!("/v1/end-user/magickspaces/{space_id}/context"),
            &[],
            Some(token),
            Some(&json!({ "chat_history": { "limit": 100 } })),
        )
        .await
    }
}

fn id_of(v: &Value, what: &str) -> Result<String> {
    v["id"]
        .as_str()
        .map(str::to_string)
        .with_context(|| format!("{what} response had no id: {v}"))
}
