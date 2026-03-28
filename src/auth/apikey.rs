use async_trait::async_trait;

use crate::{Auth, MindroidError, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::debug;

#[derive(Debug, Serialize)]
struct LoginRequest<'a> {
    email: &'a str,
    password: &'a str,
}

#[derive(Debug, Serialize)]
struct RefreshRequest {
    refresh_token: String,
}

#[derive(Debug, Deserialize)]
struct AuthResponse {
    access_token: String,
    refresh_token: String,
    expires_in: u64,
}

#[derive(Debug)]
struct TokenState {
    access_token: String,
    refresh_token: String,
    expires_at: Instant,
}

pub struct ApiKeyAuth {
    base_url: String,
    email: String,
    password: String,
    client: reqwest::Client,
    state: Arc<RwLock<Option<TokenState>>>,
}

impl ApiKeyAuth {
    pub fn new(base_url: &str, email: &str, password: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            email: email.to_string(),
            password: password.to_string(),
            client: reqwest::Client::new(),
            state: Arc::new(RwLock::new(None)),
        }
    }

    async fn login(&self) -> Result<AuthResponse> {
        debug!("Logging in as {}", self.email);
        let url = format!("{}/v1/auth/login", self.base_url);
        let body = LoginRequest {
            email: &self.email,
            password: &self.password,
        };
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| MindroidError::Auth {
                message: format!("Login request failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            return Err(MindroidError::Auth {
                message: format!("Login failed (HTTP {status}): {text}"),
                source: None,
            });
        }

        resp.json::<AuthResponse>()
            .await
            .map_err(|e| MindroidError::Auth {
                message: format!("Failed to parse login response: {e}"),
                source: Some(Box::new(e)),
            })
    }

    async fn do_refresh(&self, refresh_token: &str) -> Result<AuthResponse> {
        debug!("Refreshing token");
        let url = format!("{}/v1/auth/refresh", self.base_url);
        let body = RefreshRequest {
            refresh_token: refresh_token.to_string(),
        };
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| MindroidError::Auth {
                message: format!("Refresh request failed: {e}"),
                source: Some(Box::new(e)),
            })?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            return Err(MindroidError::Auth {
                message: format!("Token refresh failed (HTTP {status}): {text}"),
                source: None,
            });
        }

        resp.json::<AuthResponse>()
            .await
            .map_err(|e| MindroidError::Auth {
                message: format!("Failed to parse refresh response: {e}"),
                source: Some(Box::new(e)),
            })
    }

    fn store_auth_response(auth: AuthResponse) -> TokenState {
        TokenState {
            access_token: auth.access_token,
            refresh_token: auth.refresh_token,
            expires_at: Instant::now() + Duration::from_secs(auth.expires_in),
        }
    }
}

#[async_trait]
impl Auth for ApiKeyAuth {
    async fn get_token(&self) -> Result<String> {
        // Check if we have a valid token (fast path)
        {
            let state = self.state.read().await;
            if let Some(ref s) = *state
                && s.expires_at > Instant::now() + Duration::from_secs(10)
            {
                return Ok(s.access_token.clone());
            }
        }

        // Need to login or refresh — take write lock
        let mut state = self.state.write().await;

        // Re-check after acquiring write lock (another task may have refreshed)
        if let Some(ref s) = *state
            && s.expires_at > Instant::now() + Duration::from_secs(10)
        {
            return Ok(s.access_token.clone());
        }

        let auth = if let Some(ref s) = *state {
            // Refresh existing session
            self.do_refresh(&s.refresh_token).await?
        } else {
            // Initial login
            self.login().await?
        };

        let token = auth.access_token.clone();
        *state = Some(Self::store_auth_response(auth));
        Ok(token)
    }

    async fn get_auth_headers(&self) -> Result<Vec<(String, String)>> {
        let token = self.get_token().await?;
        Ok(vec![(
            "Authorization".to_string(),
            format!("Bearer {token}"),
        )])
    }

    fn is_authenticated(&self) -> bool {
        // Non-async: use try_read to avoid blocking
        if let Ok(state) = self.state.try_read()
            && let Some(ref s) = *state
        {
            return s.expires_at > Instant::now();
        }
        false
    }

    async fn refresh(&self) -> Result<()> {
        let mut state = self.state.write().await;
        let auth = if let Some(ref s) = *state {
            self.do_refresh(&s.refresh_token).await?
        } else {
            self.login().await?
        };
        *state = Some(Self::store_auth_response(auth));
        Ok(())
    }
}
