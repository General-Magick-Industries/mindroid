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

/// Bounds on auth requests.
///
/// Token acquisition happens while holding the `state` write lock — that
/// serialisation is deliberate (it collapses a thundering herd into a single
/// login), but it means an unbounded request parks every other caller behind
/// it. A gateway that accepts the connection and then stalls (half-open
/// through a load balancer, blackholed route, overloaded IdP) would otherwise
/// hang the process with no error and no recovery short of a restart.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// `base + secs`, or `None` if a server-supplied lifetime overflows the clock.
fn checked_deadline(base: Instant, secs: u64) -> Option<Instant> {
    base.checked_add(Duration::from_secs(secs))
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
        // Only timeouts are configured, so this build cannot realistically
        // fail; fall back to a default client rather than panicking in a
        // constructor that callers expect to be infallible.
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            email: email.to_string(),
            password: password.to_string(),
            client,
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
            // Both lifetimes are server-controlled and `Instant + Duration`
            // panics on overflow, so a hostile value would unwind inside
            // whichever task was acquiring a token — while holding the write
            // lock. Saturate to "already expired" instead: failing closed
            // costs a re-login, failing open hands out a token forever.
            expires_at: checked_deadline(now, auth.expires_in).unwrap_or(now),
            refresh_expires_at: auth
                .refresh_expires_in
                .map(|s| checked_deadline(now, s).unwrap_or(now)),
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
            // Same skew as get_token: reporting "authenticated" for a token
            // get_token would discard makes this predicate fail optimistically,
            // which is the wrong direction for anything gating on it.
            return s.access_valid_at(Instant::now());
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
        let now = Instant::now();
        let s = TokenState {
            access_token: "access".into(),
            refresh_token: "refresh".into(),
            expires_at: now + Duration::from_secs(300),
            refresh_expires_at: None,
        };
        assert!(s.access_valid_at(now));
        // A token expiring inside the skew window must be treated as dead, so
        // we never hand out one that dies mid-request.
        assert!(!s.access_valid_at(now + Duration::from_secs(295)));
        // Pin the boundary itself: 300s token, 10s skew, so t+289 is the last
        // instant it is usable and t+290 is not. Probing only well inside the
        // window would let the comparison flip from `>` to `>=` unnoticed.
        assert!(s.access_valid_at(now + Duration::from_secs(289)));
        assert!(!s.access_valid_at(now + Duration::from_secs(290)));
    }

    #[test]
    fn hostile_lifetimes_expire_immediately_instead_of_panicking() {
        // `Instant + Duration` panics on overflow, and this runs while the
        // write lock is held, so it would unwind inside a token acquisition.
        let st = ApiKeyAuth::store_auth_response(AuthResponse {
            access_token: "a".into(),
            refresh_token: "r".into(),
            expires_in: u64::MAX,
            refresh_expires_in: Some(u64::MAX),
        });
        // Fail closed: an unrepresentable deadline is treated as already past,
        // costing a re-login rather than pinning a token as valid forever.
        assert!(!st.access_valid_at(Instant::now()));
        assert!(!st.refresh_worth_trying_at(Instant::now()));
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

    /// Canned auth gateway.
    ///
    /// Counts hits on *both* endpoints. Counting only logins can't tell
    /// "skipped the refresh round-trip" apart from "made it and recovered
    /// via the fallback" — both end at login — so the refresh counter is what
    /// makes the skip-a-dead-refresh path actually observable.
    struct Gateway {
        base_url: String,
        login_hits: Arc<AtomicUsize>,
        refresh_hits: Arc<AtomicUsize>,
        server: tokio::task::JoinHandle<()>,
    }

    /// Stop the listener with the test that owns it, so sockets and tasks
    /// don't accumulate for the life of the test binary.
    impl Drop for Gateway {
        fn drop(&mut self) {
            self.server.abort();
        }
    }

    struct GatewayConfig {
        refresh_status: u16,
        login_status: u16,
        /// Body returned on a 200 refresh — lets a test exercise the
        /// `refresh_expires_in` wire contract in both its present and absent
        /// forms.
        refresh_body: &'static str,
        login_body: &'static str,
    }

    const DEFAULT_REFRESH_BODY: &str =
        r#"{"access_token":"from_refresh","refresh_token":"r2","expires_in":300}"#;
    const DEFAULT_LOGIN_BODY: &str =
        r#"{"access_token":"from_login","refresh_token":"r1","expires_in":300}"#;

    impl GatewayConfig {
        fn new(refresh_status: u16, login_status: u16) -> Self {
            Self {
                refresh_status,
                login_status,
                refresh_body: DEFAULT_REFRESH_BODY,
                login_body: DEFAULT_LOGIN_BODY,
            }
        }
    }

    async fn spawn_gateway(cfg: GatewayConfig) -> Gateway {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let login_hits = Arc::new(AtomicUsize::new(0));
        let refresh_hits = Arc::new(AtomicUsize::new(0));

        let (lh, rh) = (login_hits.clone(), refresh_hits.clone());
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let (lh, rh) = (lh.clone(), rh.clone());
                tokio::spawn(async move {
                    // Single read: the request is small enough that hyper
                    // coalesces headers and body into one segment. Fine while
                    // this gateway never asserts on request bodies; a read
                    // loop becomes necessary if that changes.
                    let mut buf = vec![0u8; 4096];
                    let Ok(n) = sock.read(&mut buf).await else {
                        return;
                    };
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();

                    let (status, body) = if req.contains("/v1/auth/refresh") {
                        rh.fetch_add(1, Ordering::SeqCst);
                        if cfg.refresh_status == 200 {
                            (200, cfg.refresh_body.to_string())
                        } else {
                            (
                                cfg.refresh_status,
                                r#"{"error":{"detail":"Token is not active"}}"#.to_string(),
                            )
                        }
                    } else if req.contains("/v1/auth/login") {
                        lh.fetch_add(1, Ordering::SeqCst);
                        if cfg.login_status == 200 {
                            (200, cfg.login_body.to_string())
                        } else {
                            (cfg.login_status, r#"{"error":"bad creds"}"#.to_string())
                        }
                    } else {
                        (404, "{}".to_string())
                    };

                    // Empty reason phrase: valid HTTP/1.1, and avoids the
                    // nonsense of "400 OK".
                    let resp = format!(
                        "HTTP/1.1 {status} \r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                    let _ = sock.flush().await;
                });
            }
        });

        Gateway {
            base_url: format!("http://{addr}"),
            login_hits,
            refresh_hits,
            server,
        }
    }

    /// Every network assertion runs under this so a stalled handler fails
    /// legibly instead of hanging CI until the workflow-level kill.
    async fn with_deadline<F: std::future::Future>(f: F) -> F::Output {
        tokio::time::timeout(Duration::from_secs(5), f)
            .await
            .expect("gateway interaction timed out")
    }

    fn seed(auth: &ApiKeyAuth, refresh_token: &str, refresh_expires_at: Option<Instant>) {
        // Access token already dead, so get_token is forced down the
        // refresh-or-login path.
        let mut st = auth.state.try_write().unwrap();
        *st = Some(TokenState {
            access_token: "stale".into(),
            refresh_token: refresh_token.into(),
            expires_at: Instant::now(),
            refresh_expires_at,
        });
    }

    /// The regression this whole change exists for: a refresh that fails with
    /// Keycloak's "Token is not active" must recover via a full login instead
    /// of bricking auth until the process restarts.
    #[tokio::test]
    async fn failed_refresh_falls_back_to_login() {
        let gw = spawn_gateway(GatewayConfig::new(400, 200)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");
        seed(&auth, "dead_refresh", None);

        let token = with_deadline(auth.get_token())
            .await
            .expect("must recover via login");
        assert_eq!(token, "from_login");
        assert_eq!(gw.refresh_hits.load(Ordering::SeqCst), 1, "refresh tried");
        assert_eq!(gw.login_hits.load(Ordering::SeqCst), 1, "login attempted");
    }

    /// A refresh token we already know is expired shouldn't cost a round-trip.
    ///
    /// The refresh counter is load-bearing here: without it, deleting the
    /// early return still passes, because refresh would 500 and the fallback
    /// would reach login anyway — the same observable outcome.
    #[tokio::test]
    async fn known_expired_refresh_goes_straight_to_login() {
        let gw = spawn_gateway(GatewayConfig::new(500, 200)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");
        seed(&auth, "expired", Some(Instant::now()));

        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "from_login");
        assert_eq!(
            gw.refresh_hits.load(Ordering::SeqCst),
            0,
            "a known-dead refresh token must not be sent"
        );
        assert_eq!(gw.login_hits.load(Ordering::SeqCst), 1);
    }

    /// When recovery is genuinely impossible, don't keep a token we know is
    /// dead — clearing lets the next call start from a clean login.
    #[tokio::test]
    async fn state_is_cleared_when_refresh_and_login_both_fail() {
        let gw = spawn_gateway(GatewayConfig::new(400, 401)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");
        seed(&auth, "dead", None);

        assert!(with_deadline(auth.get_token()).await.is_err());
        assert!(
            auth.state.read().await.is_none(),
            "unusable state must not persist"
        );
    }

    #[tokio::test]
    async fn first_call_with_no_state_logs_in() {
        let gw = spawn_gateway(GatewayConfig::new(400, 200)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");

        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "from_login");
        assert_eq!(gw.login_hits.load(Ordering::SeqCst), 1);
        assert_eq!(
            gw.refresh_hits.load(Ordering::SeqCst),
            0,
            "nothing to refresh from"
        );
    }

    #[tokio::test]
    async fn healthy_refresh_is_still_preferred_over_login() {
        let gw = spawn_gateway(GatewayConfig::new(200, 200)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");
        seed(&auth, "good", None);

        assert_eq!(
            with_deadline(auth.get_token()).await.unwrap(),
            "from_refresh"
        );
        assert_eq!(gw.refresh_hits.load(Ordering::SeqCst), 1);
        assert_eq!(gw.login_hits.load(Ordering::SeqCst), 0, "login must not run");

        // Rotation end-to-end: the token from the refresh response has to
        // reach the cached state, not just the returned access token.
        let st = auth.state.read().await;
        assert_eq!(
            st.as_ref().unwrap().refresh_token,
            "r2",
            "rotated refresh token must be persisted"
        );
    }

    /// The `refresh_expires_in` wire contract, present form. Nothing else
    /// parses this field from JSON — the struct-level test bypasses serde.
    #[tokio::test]
    async fn refresh_expiry_is_parsed_from_the_wire_when_present() {
        let cfg = GatewayConfig {
            login_body: r#"{"access_token":"from_login","refresh_token":"r1","expires_in":300,"refresh_expires_in":1800}"#,
            ..GatewayConfig::new(400, 200)
        };
        let gw = spawn_gateway(cfg).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");

        with_deadline(auth.get_token()).await.unwrap();
        let st = auth.state.read().await;
        assert!(
            st.as_ref().unwrap().refresh_expires_at.is_some(),
            "refresh_expires_in must be read off the wire"
        );
    }

    /// Absent form: a gateway that omits the field leaves expiry unknown,
    /// which must not be conflated with expired.
    #[tokio::test]
    async fn refresh_expiry_is_none_when_gateway_omits_it() {
        let gw = spawn_gateway(GatewayConfig::new(400, 200)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");

        with_deadline(auth.get_token()).await.unwrap();
        let st = auth.state.read().await;
        assert!(st.as_ref().unwrap().refresh_expires_at.is_none());
    }

    /// `Auth::refresh()` is public trait surface with the same clear-on-failure
    /// logic as `get_token`, and every other network test goes through
    /// `get_token` — so without this, a refactor could reintroduce the bricked
    /// session on the one path nothing guards.
    #[tokio::test]
    async fn trait_refresh_also_falls_back_to_login() {
        let gw = spawn_gateway(GatewayConfig::new(400, 200)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");
        seed(&auth, "dead_refresh", None);

        with_deadline(auth.refresh())
            .await
            .expect("refresh() must recover via login");
        assert_eq!(gw.login_hits.load(Ordering::SeqCst), 1);
        let st = auth.state.read().await;
        assert_eq!(st.as_ref().unwrap().access_token, "from_login");
    }

    #[tokio::test]
    async fn trait_refresh_clears_state_when_recovery_fails() {
        let gw = spawn_gateway(GatewayConfig::new(400, 401)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");
        seed(&auth, "dead", None);

        assert!(with_deadline(auth.refresh()).await.is_err());
        assert!(auth.state.read().await.is_none());
    }

    /// Concurrency is the reason the write lock spans the network call: N
    /// callers must collapse into one login, not N.
    #[tokio::test]
    async fn concurrent_callers_trigger_a_single_login() {
        let gw = spawn_gateway(GatewayConfig::new(400, 200)).await;
        let auth = Arc::new(ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw"));

        let tokens = with_deadline(futures::future::join_all(
            (0..10).map(|_| {
                let a = auth.clone();
                async move { a.get_token().await.unwrap() }
            }),
        ))
        .await;

        assert!(tokens.iter().all(|t| t == "from_login"));
        assert_eq!(
            gw.login_hits.load(Ordering::SeqCst),
            1,
            "single-flight: 10 callers must not cause 10 logins"
        );
    }
}
