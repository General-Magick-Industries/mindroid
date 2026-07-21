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
    /// Lifetime of the *refresh* token. Keycloak returns this; older gateways
    /// may not, hence the Option — absent means "unknown", not "never expires".
    #[serde(default)]
    refresh_expires_in: Option<u64>,
}

#[derive(Debug)]
struct TokenState {
    access_token: String,
    refresh_token: String,
    expires_at: Instant,
    /// None when the gateway didn't tell us — we then discover expiry only by
    /// a failed refresh, which the login fallback recovers from.
    refresh_expires_at: Option<Instant>,
}

impl TokenState {
    /// Whether the access token is still usable, with a skew margin so we
    /// don't hand out a token that expires mid-flight.
    fn access_valid_at(&self, now: Instant) -> bool {
        self.expires_at > now + EXPIRY_SKEW
    }

    /// Whether refreshing is worth attempting. A refresh token we *know* has
    /// expired should go straight to login instead of burning a round-trip on
    /// a guaranteed "Token is not active".
    fn refresh_worth_trying_at(&self, now: Instant) -> bool {
        match self.refresh_expires_at {
            Some(exp) => exp > now + EXPIRY_SKEW,
            None => true,
        }
    }
}

/// Margin applied to both token lifetimes to avoid racing an expiry.
const EXPIRY_SKEW: Duration = Duration::from_secs(10);

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
        let now = Instant::now();
        TokenState {
            access_token: auth.access_token,
            refresh_token: auth.refresh_token,
            expires_at: now + Duration::from_secs(auth.expires_in),
            refresh_expires_at: auth
                .refresh_expires_in
                .map(|s| now + Duration::from_secs(s)),
        }
    }

    /// Acquire a fresh `AuthResponse`, preferring refresh but always able to
    /// fall back to a full login.
    ///
    /// The fallback is what keeps a long-lived process recoverable: a refresh
    /// token can expire (SSO idle/max lifetime), be revoked, or be consumed by
    /// a rotation whose response we never saw. Without falling back to the
    /// credentials we already hold, any one of those bricks auth permanently
    /// and only a restart fixes it.
    async fn acquire(&self, state: Option<&TokenState>) -> Result<AuthResponse> {
        let Some(s) = state else {
            return self.login().await;
        };

        if !s.refresh_worth_trying_at(Instant::now()) {
            debug!("Refresh token expired; logging in instead of refreshing");
            return self.login().await;
        }

        match self.do_refresh(&s.refresh_token).await {
            Ok(auth) => Ok(auth),
            Err(e) => {
                debug!("Refresh failed ({e}); falling back to login");
                self.login().await
            }
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
                && s.access_valid_at(Instant::now())
            {
                return Ok(s.access_token.clone());
            }
        }

        // Need to login or refresh — take write lock
        let mut state = self.state.write().await;

        // Re-check after acquiring write lock (another task may have refreshed)
        if let Some(ref s) = *state
            && s.access_valid_at(Instant::now())
        {
            return Ok(s.access_token.clone());
        }

        // On failure, clear the state rather than leaving a token we now know
        // is unusable — the next caller then starts from a clean login.
        let auth = match self.acquire(state.as_ref()).await {
            Ok(auth) => auth,
            Err(e) => {
                *state = None;
                return Err(e);
            }
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
        let auth = match self.acquire(state.as_ref()).await {
            Ok(auth) => auth,
            Err(e) => {
                *state = None;
                return Err(e);
            }
        };
        *state = Some(Self::store_auth_response(auth));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn token_state(access_in: u64, refresh_in: Option<u64>) -> TokenState {
        let now = Instant::now();
        TokenState {
            access_token: "access".into(),
            refresh_token: "refresh".into(),
            expires_at: now + Duration::from_secs(access_in),
            refresh_expires_at: refresh_in.map(|s| now + Duration::from_secs(s)),
        }
    }

    #[test]
    fn access_token_within_skew_is_not_reused() {
        let s = token_state(300, None);
        assert!(s.access_valid_at(Instant::now()));
        // A token expiring inside the skew window must be treated as dead, so
        // we never hand out one that dies mid-request.
        assert!(!s.access_valid_at(Instant::now() + Duration::from_secs(295)));
    }

    #[test]
    fn unknown_refresh_expiry_is_still_worth_trying() {
        // Gateways that omit refresh_expires_in must not be assumed expired.
        assert!(token_state(300, None).refresh_worth_trying_at(Instant::now()));
    }

    #[test]
    fn known_expired_refresh_token_is_not_worth_trying() {
        let s = token_state(300, Some(600));
        assert!(s.refresh_worth_trying_at(Instant::now()));
        assert!(!s.refresh_worth_trying_at(Instant::now() + Duration::from_secs(595)));
    }

    #[test]
    fn rotation_stores_the_new_refresh_token() {
        let st = ApiKeyAuth::store_auth_response(AuthResponse {
            access_token: "a2".into(),
            refresh_token: "r2".into(),
            expires_in: 300,
            refresh_expires_in: Some(1800),
        });
        assert_eq!(st.refresh_token, "r2");
        assert!(st.refresh_expires_at.is_some());
    }

    /// Canned auth gateway. `refresh_status` lets a test make /v1/auth/refresh
    /// fail the way Keycloak does for a dead token.
    async fn spawn_gateway(
        refresh_status: u16,
        login_status: u16,
        login_hits: Arc<AtomicUsize>,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let hits = login_hits.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    let Ok(n) = sock.read(&mut buf).await else {
                        return;
                    };
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();

                    let (status, body) = if req.contains("/v1/auth/refresh") {
                        if refresh_status == 200 {
                            (200, r#"{"access_token":"from_refresh","refresh_token":"r2","expires_in":300}"#.to_string())
                        } else {
                            (
                                refresh_status,
                                r#"{"error":{"detail":"Token is not active"}}"#.to_string(),
                            )
                        }
                    } else if req.contains("/v1/auth/login") {
                        hits.fetch_add(1, Ordering::SeqCst);
                        if login_status == 200 {
                            (200, r#"{"access_token":"from_login","refresh_token":"r1","expires_in":300}"#.to_string())
                        } else {
                            (login_status, r#"{"error":"bad creds"}"#.to_string())
                        }
                    } else {
                        (404, "{}".to_string())
                    };

                    let resp = format!(
                        "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                    let _ = sock.flush().await;
                });
            }
        });

        format!("http://{addr}")
    }

    /// The regression this whole change exists for: a refresh that fails with
    /// Keycloak's "Token is not active" must recover via a full login instead
    /// of bricking auth until the process restarts.
    #[tokio::test]
    async fn failed_refresh_falls_back_to_login() {
        let hits = Arc::new(AtomicUsize::new(0));
        let base = spawn_gateway(400, 200, hits.clone()).await;
        let auth = ApiKeyAuth::new(&base, "user@example.com", "pw");

        // Seed a session whose access token is already dead, forcing a refresh.
        {
            let mut st = auth.state.write().await;
            *st = Some(TokenState {
                access_token: "stale".into(),
                refresh_token: "dead_refresh".into(),
                expires_at: Instant::now(),
                refresh_expires_at: None,
            });
        }

        let token = auth.get_token().await.expect("must recover via login");
        assert_eq!(token, "from_login");
        assert_eq!(hits.load(Ordering::SeqCst), 1, "login should be attempted");
    }

    /// A refresh token we already know is expired shouldn't cost a round-trip.
    #[tokio::test]
    async fn known_expired_refresh_goes_straight_to_login() {
        let hits = Arc::new(AtomicUsize::new(0));
        // Refresh would 500 if called; reaching login proves we skipped it.
        let base = spawn_gateway(500, 200, hits.clone()).await;
        let auth = ApiKeyAuth::new(&base, "user@example.com", "pw");

        {
            let mut st = auth.state.write().await;
            *st = Some(TokenState {
                access_token: "stale".into(),
                refresh_token: "expired".into(),
                expires_at: Instant::now(),
                refresh_expires_at: Some(Instant::now()),
            });
        }

        assert_eq!(auth.get_token().await.unwrap(), "from_login");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    /// When recovery is genuinely impossible, don't keep a token we know is
    /// dead — clearing lets the next call start from a clean login.
    #[tokio::test]
    async fn state_is_cleared_when_refresh_and_login_both_fail() {
        let hits = Arc::new(AtomicUsize::new(0));
        let base = spawn_gateway(400, 401, hits.clone()).await;
        let auth = ApiKeyAuth::new(&base, "user@example.com", "pw");

        {
            let mut st = auth.state.write().await;
            *st = Some(TokenState {
                access_token: "stale".into(),
                refresh_token: "dead".into(),
                expires_at: Instant::now(),
                refresh_expires_at: None,
            });
        }

        assert!(auth.get_token().await.is_err());
        assert!(
            auth.state.read().await.is_none(),
            "unusable state must not persist"
        );
    }

    #[tokio::test]
    async fn first_call_with_no_state_logs_in() {
        let hits = Arc::new(AtomicUsize::new(0));
        let base = spawn_gateway(400, 200, hits.clone()).await;
        let auth = ApiKeyAuth::new(&base, "user@example.com", "pw");

        assert_eq!(auth.get_token().await.unwrap(), "from_login");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn healthy_refresh_is_still_preferred_over_login() {
        let hits = Arc::new(AtomicUsize::new(0));
        let base = spawn_gateway(200, 200, hits.clone()).await;
        let auth = ApiKeyAuth::new(&base, "user@example.com", "pw");

        {
            let mut st = auth.state.write().await;
            *st = Some(TokenState {
                access_token: "stale".into(),
                refresh_token: "good".into(),
                expires_at: Instant::now(),
                refresh_expires_at: None,
            });
        }

        assert_eq!(auth.get_token().await.unwrap(), "from_refresh");
        assert_eq!(hits.load(Ordering::SeqCst), 0, "login must not be used");
    }
}
