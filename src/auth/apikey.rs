use async_trait::async_trait;

use crate::{Auth, MindroidError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn};

// None of the credential-bearing types below derive Debug. Nothing prints them
// today, but a single future `debug!("{body:?}")` would put a password or a
// live token in the logs with no compiler warning and nothing in review to
// catch it. TokenState gets a redacting impl instead, since it is the one a
// person debugging this flow would actually reach for.

#[derive(Serialize)]
struct LoginRequest<'a> {
    email: &'a str,
    password: &'a str,
}

#[derive(Serialize)]
struct RefreshRequest {
    refresh_token: String,
}

#[derive(Deserialize)]
struct AuthResponse {
    access_token: String,
    refresh_token: String,
    expires_in: u64,
    /// Lifetime of the *refresh* token. Keycloak returns this; older gateways
    /// may not, hence the Option — absent means "unknown", not "never expires".
    #[serde(default)]
    refresh_expires_in: Option<u64>,
}

struct TokenState {
    access_token: String,
    refresh_token: String,
    expires_at: Instant,
    /// None when the gateway didn't tell us — we then discover expiry only by
    /// a failed refresh, which the login fallback recovers from.
    refresh_expires_at: Option<Instant>,
}

impl fmt::Debug for TokenState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenState")
            .field("access_token", &"<redacted>")
            .field("refresh_token", &"<redacted>")
            .field("expires_at", &self.expires_at)
            .field("refresh_expires_at", &self.refresh_expires_at)
            .finish()
    }
}

/// An upstream response body, carried as an error `source` rather than
/// interpolated into the message.
///
/// On the fallback branch an unparseable response is returned verbatim — which
/// is what a proxy, WAF, or IdP error page produces — and `MindroidError::Auth`
/// renders its message into logs that may ship somewhere with a much broader
/// audience than the auth path. Keeping the body in `source` leaves it
/// available to a deliberate `{:#}` unwind without riding into every log line.
#[derive(Debug)]
struct UpstreamBody(String);

impl fmt::Display for UpstreamBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UpstreamBody {}

/// Why an auth request failed, at the granularity the retry policy needs.
enum AuthFailure {
    /// The gateway evaluated the credential and refused it (400/401/403).
    /// Retrying the same way is pointless; only a different credential helps.
    Rejected(MindroidError),
    /// Transport failure, 5xx, or anything else that never got as far as
    /// evaluating the credential — so it says nothing about whether the
    /// credential is still good.
    Unavailable(MindroidError),
}

impl AuthFailure {
    fn from_status(status: u16, message: String, body: String) -> Self {
        let err = MindroidError::Auth {
            message,
            source: Some(Box::new(UpstreamBody(body))),
        };
        // 400 is here because that is what Keycloak returns for an expired or
        // already-consumed refresh token ("Token is not active").
        if matches!(status, 400 | 401 | 403) {
            Self::Rejected(err)
        } else {
            Self::Unavailable(err)
        }
    }

    fn is_rejected(&self) -> bool {
        matches!(self, Self::Rejected(_))
    }

    fn into_error(self) -> MindroidError {
        match self {
            Self::Rejected(e) | Self::Unavailable(e) => e,
        }
    }
}

/// A failed acquisition, plus whether it proved the cached session is dead.
struct AcquireFailure {
    error: MindroidError,
    /// True only when the gateway definitively rejected a credential. An
    /// unreachable gateway proves nothing about the tokens we hold, and
    /// discarding them on that basis throws away a refresh token that is very
    /// likely still valid server-side.
    session_dead: bool,
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

/// Overall ceiling for one acquisition (a refresh that may chain into a login).
///
/// `acquire` runs under the state write lock, and on the rejected-refresh path
/// it makes two sequential requests — each bounded by `REQUEST_TIMEOUT`, so the
/// lock could otherwise be held for ~2x that against a half-open gateway,
/// stalling every other caller (including ones holding a still-valid token).
/// A deadline overrun is transient: it proves nothing about the cached token,
/// so the session is retained and backoff bounds the retry rate.
const ACQUIRE_TIMEOUT: Duration = Duration::from_secs(20);

/// Backoff bounds applied after a failed acquisition.
const BACKOFF_BASE: Duration = Duration::from_secs(1);
const BACKOFF_MAX: Duration = Duration::from_secs(60);

/// `base + secs`, or `None` if a server-supplied lifetime overflows the clock.
fn checked_deadline(base: Instant, secs: u64) -> Option<Instant> {
    base.checked_add(Duration::from_secs(secs))
}

/// Exponential backoff, capped.
///
/// Without it, a failing gateway means one full credential login per inbound
/// message for as long as the outage lasts. That is both pointless load on a
/// service already in trouble and exactly the traffic pattern that trips IdP
/// account protections — turning a transient outage into one that needs a
/// human to unlock the account.
fn backoff_delay(consecutive_failures: u32) -> Duration {
    let shift = consecutive_failures.saturating_sub(1).min(6);
    BACKOFF_BASE.saturating_mul(1u32 << shift).min(BACKOFF_MAX)
}

/// Session state plus what we've learned from failures.
///
/// The failure fields deliberately sit alongside the token rather than inside
/// `Option<TokenState>`: clearing a dead session must discard the *token*
/// without also discarding the fact that we are in a failing streak, which is
/// the only thing keeping the retry rate bounded.
#[derive(Default)]
struct AuthState {
    token: Option<TokenState>,
    consecutive_failures: u32,
    retry_after: Option<Instant>,
    last_error: Option<String>,
}

impl AuthState {
    fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.retry_after = None;
        self.last_error = None;
    }

    fn record_failure(&mut self, now: Instant, err: &MindroidError) {
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        self.retry_after = now.checked_add(backoff_delay(self.consecutive_failures));
        self.last_error = Some(err.to_string());
    }

    fn in_backoff_at(&self, now: Instant) -> bool {
        self.retry_after.is_some_and(|t| t > now)
    }

    /// The error returned while backoff is in force — reports the streak and
    /// the cause rather than silently doing nothing.
    fn backoff_error(&self) -> MindroidError {
        MindroidError::Auth {
            message: format!(
                "auth backoff in effect after {} consecutive failures; last error: {}",
                self.consecutive_failures,
                self.last_error.as_deref().unwrap_or("unknown")
            ),
            source: None,
        }
    }
}

pub struct ApiKeyAuth {
    base_url: String,
    email: String,
    password: String,
    client: reqwest::Client,
    state: Arc<RwLock<AuthState>>,
}

impl ApiKeyAuth {
    /// Construct without a transport-security check.
    ///
    /// Prefer [`ApiKeyAuth::try_new`], which refuses to send credentials over
    /// plaintext unless explicitly opted in. This constructor is kept
    /// infallible for backwards compatibility and only warns.
    pub fn new(base_url: &str, email: &str, password: &str) -> Self {
        let base_url = base_url.trim_end_matches('/').to_string();
        if is_plaintext(&base_url) {
            warn!(
                "auth base_url {base_url} is plaintext; credentials will cross \
                 the network unprotected — use https:// (see try_new)"
            );
        }
        Self::build(base_url, email, password)
    }

    /// Construct, refusing plaintext transport unless `allow_insecure` is set.
    ///
    /// Mirrors the guard already applied to the two other modules that send
    /// credentials — `transport::centrifugo` for `ws://` and
    /// `persona::magickmind_stage` for `http://` — using the same
    /// `allow_insecure` opt-in so operators learn one concept. This path
    /// carries the account password rather than a short-lived token, which is
    /// the one credential the agent cannot rotate on its own.
    pub fn try_new(
        base_url: &str,
        email: &str,
        password: &str,
        allow_insecure: bool,
    ) -> Result<Self> {
        let base_url = base_url.trim_end_matches('/').to_string();
        if is_plaintext(&base_url) && !allow_insecure {
            return Err(MindroidError::config(format!(
                "refusing to send credentials over plaintext {base_url}: use https://, \
                 or set auth.allow_insecure = true for local development"
            )));
        }
        Ok(Self::build(base_url, email, password))
    }

    fn build(base_url: String, email: &str, password: &str) -> Self {
        let builder = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT);

        // Tests talk to a loopback gateway on an ephemeral port. reqwest honours
        // HTTP_PROXY/ALL_PROXY and does not exempt loopback, so a proxied dev
        // machine or a CI runner with egress control would route those requests
        // into the void and surface it as an auth failure.
        #[cfg(test)]
        let builder = builder.no_proxy();

        // Only timeouts are configured, so this build cannot realistically
        // fail; fall back to a default client rather than panicking in a
        // constructor callers expect to be infallible.
        let client = builder.build().unwrap_or_else(|_| reqwest::Client::new());

        Self {
            base_url,
            email: email.to_string(),
            password: password.to_string(),
            client,
            state: Arc::new(RwLock::new(AuthState::default())),
        }
    }

    /// POST an auth request and classify the outcome.
    ///
    /// Shared by both endpoints so the status-to-`AuthFailure` mapping exists
    /// in exactly one place — the retry policy is only as sound as this
    /// classification.
    async fn post_auth<B: Serialize>(
        &self,
        path: &str,
        body: &B,
        what: &str,
    ) -> std::result::Result<AuthResponse, AuthFailure> {
        let url = format!("{}{path}", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(body)
            .send()
            .await
            .map_err(|e| {
                // Never reached the point of evaluating the credential.
                AuthFailure::Unavailable(MindroidError::Auth {
                    message: format!("{what} request failed: {e}"),
                    source: Some(Box::new(e)),
                })
            })?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(AuthFailure::from_status(
                status,
                format!("{what} failed (HTTP {status})"),
                body,
            ));
        }

        resp.json::<AuthResponse>().await.map_err(|e| {
            // A success status we can't parse means something stood in for the
            // IdP; treat it as unavailable rather than as a rejection.
            AuthFailure::Unavailable(MindroidError::Auth {
                message: format!("Failed to parse {what} response: {e}"),
                source: Some(Box::new(e)),
            })
        })
    }

    async fn login(&self) -> std::result::Result<AuthResponse, AuthFailure> {
        debug!("Logging in as {}", self.email);
        let body = LoginRequest {
            email: &self.email,
            password: &self.password,
        };
        self.post_auth("/v1/auth/login", &body, "Login").await
    }

    async fn do_refresh(
        &self,
        refresh_token: &str,
    ) -> std::result::Result<AuthResponse, AuthFailure> {
        debug!("Refreshing token");
        let body = RefreshRequest {
            refresh_token: refresh_token.to_string(),
        };
        self.post_auth("/v1/auth/refresh", &body, "Token refresh")
            .await
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
    async fn acquire(
        &self,
        state: Option<&TokenState>,
    ) -> std::result::Result<AuthResponse, AcquireFailure> {
        let Some(s) = state else {
            return self.login_only().await;
        };

        if !s.refresh_worth_trying_at(Instant::now()) {
            debug!("Refresh token expired; logging in instead of refreshing");
            return self.login_only().await;
        }

        match self.do_refresh(&s.refresh_token).await {
            Ok(auth) => Ok(auth),

            // The gateway never evaluated the token, so it may well still be
            // good — and login would hit the same unhealthy gateway. Falling
            // back here would double the load on a service already failing,
            // on what is a per-message hot path.
            Err(AuthFailure::Unavailable(error)) => {
                debug!("Refresh unavailable ({error}); not attempting login");
                Err(AcquireFailure {
                    error,
                    session_dead: false,
                })
            }

            // The refresh token is definitively dead. A full login is the only
            // way back, and this is the case the whole change exists for.
            Err(AuthFailure::Rejected(refresh_err)) => {
                debug!("Refresh rejected ({refresh_err}); falling back to login");
                match self.login().await {
                    Ok(auth) => Ok(auth),
                    Err(login_failure) => Err(AcquireFailure {
                        // Keep the refresh error as the cause: "refresh was
                        // rejected" and "refresh was unreachable" call for
                        // completely different responses, and only one of them
                        // is visible in the login error.
                        error: with_cause(login_failure.into_error(), refresh_err),
                        // The refresh token was proven dead regardless of how
                        // the login went, so the cached session is unusable.
                        session_dead: true,
                    }),
                }
            }
        }
    }

    async fn login_only(&self) -> std::result::Result<AuthResponse, AcquireFailure> {
        self.login().await.map_err(|f| AcquireFailure {
            session_dead: f.is_rejected(),
            error: f.into_error(),
        })
    }

    /// Acquire under the write lock, applying backoff and updating failure
    /// bookkeeping. Shared by `get_token` and `refresh` so both paths agree on
    /// when a session is discarded.
    async fn acquire_into(&self, state: &mut AuthState) -> Result<String> {
        let now = Instant::now();
        if state.in_backoff_at(now) {
            return Err(state.backoff_error());
        }

        let acquired = tokio::time::timeout(ACQUIRE_TIMEOUT, self.acquire(state.token.as_ref()))
            .await
            .unwrap_or_else(|_| {
                // Deadline overrun: treat as transient (session_dead: false) so
                // a still-valid token survives and backoff throttles the retry.
                Err(AcquireFailure {
                    error: MindroidError::Auth {
                        message: format!(
                            "auth acquisition exceeded {}s deadline",
                            ACQUIRE_TIMEOUT.as_secs()
                        ),
                        source: None,
                    },
                    session_dead: false,
                })
            });

        match acquired {
            Ok(auth) => {
                let token = auth.access_token.clone();
                state.token = Some(Self::store_auth_response(auth));
                state.record_success();
                Ok(token)
            }
            Err(f) => {
                // Only discard the session when something actually proved it
                // dead. A transport failure proves nothing, and throwing away
                // a still-valid refresh token forces every later call down the
                // full password-login path for the rest of the process's life.
                if f.session_dead {
                    state.token = None;
                }
                state.record_failure(now, &f.error);
                Err(f.error)
            }
        }
    }
}

/// Attach `cause` as the source of `err`, preserving the original message.
fn with_cause(err: MindroidError, cause: MindroidError) -> MindroidError {
    match err {
        MindroidError::Auth { message, .. } => MindroidError::Auth {
            message,
            source: Some(Box::new(cause)),
        },
        other => other,
    }
}

fn is_plaintext(base_url: &str) -> bool {
    base_url.starts_with("http://")
}

#[async_trait]
impl Auth for ApiKeyAuth {
    async fn get_token(&self) -> Result<String> {
        // Check if we have a valid token (fast path)
        {
            let state = self.state.read().await;
            if let Some(ref s) = state.token
                && s.access_valid_at(Instant::now())
            {
                return Ok(s.access_token.clone());
            }
        }

        // Need to login or refresh — take write lock
        let mut state = self.state.write().await;

        // Re-check after acquiring write lock (another task may have refreshed)
        if let Some(ref s) = state.token
            && s.access_valid_at(Instant::now())
        {
            return Ok(s.access_token.clone());
        }

        self.acquire_into(&mut state).await
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
            && let Some(ref s) = state.token
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
        self.acquire_into(&mut state).await.map(|_| ())
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
        /// Live, so a test can flip the gateway from failing to healthy and
        /// exercise one session across the recovery.
        refresh_status: Arc<AtomicUsize>,
        server: tokio::task::JoinHandle<()>,
    }

    impl Gateway {
        fn set_refresh_status(&self, status: u16) {
            self.refresh_status.store(status as usize, Ordering::SeqCst);
        }
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
        let refresh_status = Arc::new(AtomicUsize::new(cfg.refresh_status as usize));

        let (lh, rh, rs) = (
            login_hits.clone(),
            refresh_hits.clone(),
            refresh_status.clone(),
        );
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let (lh, rh, rs) = (lh.clone(), rh.clone(), rs.clone());
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
                        let current = rs.load(Ordering::SeqCst) as u16;
                        if current == 200 {
                            (200, cfg.refresh_body.to_string())
                        } else {
                            (
                                current,
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
            refresh_status,
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
        st.token = Some(TokenState {
            access_token: "stale".into(),
            refresh_token: refresh_token.into(),
            expires_at: Instant::now(),
            refresh_expires_at,
        });
    }

    /// Clear backoff so a test can make a second attempt without waiting out
    /// the real delay.
    fn clear_backoff(auth: &ApiKeyAuth) {
        let mut st = auth.state.try_write().unwrap();
        st.retry_after = None;
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
            auth.state.read().await.token.is_none(),
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
        assert_eq!(
            gw.login_hits.load(Ordering::SeqCst),
            0,
            "login must not run"
        );

        // Rotation end-to-end: the token from the refresh response has to
        // reach the cached state, not just the returned access token.
        let st = auth.state.read().await;
        assert_eq!(
            st.token.as_ref().unwrap().refresh_token,
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
            st.token.as_ref().unwrap().refresh_expires_at.is_some(),
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
        assert!(st.token.as_ref().unwrap().refresh_expires_at.is_none());
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
        assert_eq!(st.token.as_ref().unwrap().access_token, "from_login");
    }

    #[tokio::test]
    async fn trait_refresh_clears_state_when_recovery_fails() {
        let gw = spawn_gateway(GatewayConfig::new(400, 401)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");
        seed(&auth, "dead", None);

        assert!(with_deadline(auth.refresh()).await.is_err());
        assert!(auth.state.read().await.token.is_none());
    }

    // ---- error classification (5xx vs credential rejection) ----

    /// A 5xx says nothing about the refresh token, and login would hit the
    /// same unhealthy gateway — so falling back would double the load on a
    /// service already failing, on a per-message hot path.
    #[tokio::test]
    async fn unavailable_refresh_does_not_fall_back_to_login() {
        let gw = spawn_gateway(GatewayConfig::new(503, 200)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");
        seed(&auth, "probably_fine", None);

        assert!(with_deadline(auth.get_token()).await.is_err());
        assert_eq!(gw.refresh_hits.load(Ordering::SeqCst), 1);
        assert_eq!(
            gw.login_hits.load(Ordering::SeqCst),
            0,
            "a 5xx must not escalate into a credential login"
        );
    }

    /// The counterpart: a 400 is Keycloak's "Token is not active", which does
    /// warrant a login. This is the pairing that stops the classification
    /// collapsing into "never fall back".
    #[tokio::test]
    async fn rejected_refresh_does_fall_back_to_login() {
        let gw = spawn_gateway(GatewayConfig::new(400, 200)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");
        seed(&auth, "dead", None);

        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "from_login");
        assert_eq!(gw.login_hits.load(Ordering::SeqCst), 1);
    }

    // ---- session retention on non-definitive failures ----

    /// An unreachable gateway proves nothing about the tokens we hold.
    /// Discarding them would force every later call down the full
    /// password-login path even though the refresh token is still valid
    /// server-side — the mobile case being a device that loses connectivity
    /// for a moment mid-refresh.
    #[tokio::test]
    async fn transient_failure_keeps_the_session() {
        let gw = spawn_gateway(GatewayConfig::new(503, 200)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");
        seed(&auth, "still_good", None);

        assert!(with_deadline(auth.get_token()).await.is_err());
        let st = auth.state.read().await;
        assert_eq!(
            st.token.as_ref().unwrap().refresh_token,
            "still_good",
            "a reachability failure must not discard a usable refresh token"
        );
    }

    /// The retained token is not merely kept but actually used once the
    /// gateway recovers — no password login required. The gateway flips from
    /// failing to healthy mid-test, so this exercises one session across the
    /// recovery rather than a fresh one after it.
    #[tokio::test]
    async fn retained_session_refreshes_once_the_gateway_recovers() {
        let gw = spawn_gateway(GatewayConfig::new(503, 200)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");
        seed(&auth, "still_good", None);

        assert!(with_deadline(auth.get_token()).await.is_err());

        gw.set_refresh_status(200);
        clear_backoff(&auth);

        assert_eq!(
            with_deadline(auth.get_token()).await.unwrap(),
            "from_refresh",
            "the retained refresh token must be what recovers the session"
        );
        assert_eq!(
            gw.login_hits.load(Ordering::SeqCst),
            0,
            "recovery must not need the password"
        );
    }

    // ---- backoff ----

    #[test]
    fn backoff_grows_then_caps() {
        assert_eq!(backoff_delay(1), Duration::from_secs(1));
        assert_eq!(backoff_delay(2), Duration::from_secs(2));
        assert_eq!(backoff_delay(3), Duration::from_secs(4));
        // Capped, and saturating rather than overflowing the shift.
        assert_eq!(backoff_delay(7), BACKOFF_MAX);
        assert_eq!(backoff_delay(u32::MAX), BACKOFF_MAX);
    }

    /// Without backoff, a message-driven agent issues one full password login
    /// per inbound message for the duration of an outage — which is also what
    /// trips IdP account protections and turns an outage into a lockout.
    #[tokio::test]
    async fn repeated_failures_are_throttled() {
        let gw = spawn_gateway(GatewayConfig::new(400, 401)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");

        assert!(with_deadline(auth.get_token()).await.is_err());
        let after_first = gw.login_hits.load(Ordering::SeqCst);
        assert_eq!(after_first, 1);

        // Immediate retries land inside the backoff window.
        for _ in 0..5 {
            assert!(with_deadline(auth.get_token()).await.is_err());
        }
        assert_eq!(
            gw.login_hits.load(Ordering::SeqCst),
            after_first,
            "retries during backoff must not reach the gateway"
        );
    }

    #[tokio::test]
    async fn backoff_error_names_the_streak_and_cause() {
        let gw = spawn_gateway(GatewayConfig::new(400, 401)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");

        assert!(with_deadline(auth.get_token()).await.is_err());
        let err = with_deadline(auth.get_token())
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("backoff"), "got: {err}");
        assert!(
            err.contains("consecutive failures"),
            "backoff error should explain itself: {err}"
        );
    }

    #[tokio::test]
    async fn success_resets_the_failure_streak() {
        let gw = spawn_gateway(GatewayConfig::new(400, 401)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");
        assert!(with_deadline(auth.get_token()).await.is_err());
        assert_eq!(auth.state.read().await.consecutive_failures, 1);
        drop(gw);

        let gw2 = spawn_gateway(GatewayConfig::new(400, 200)).await;
        let auth = ApiKeyAuth::new(&gw2.base_url, "user@example.com", "pw");
        with_deadline(auth.get_token()).await.unwrap();

        let st = auth.state.read().await;
        assert_eq!(st.consecutive_failures, 0);
        assert!(st.retry_after.is_none());
        assert!(st.last_error.is_none());
    }

    /// Backoff must not strand a caller who already has a usable token.
    #[tokio::test]
    async fn backoff_does_not_block_a_still_valid_token() {
        let gw = spawn_gateway(GatewayConfig::new(400, 401)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");
        assert!(with_deadline(auth.get_token()).await.is_err());

        // A healthy token arrives (e.g. a concurrent path succeeded).
        {
            let mut st = auth.state.try_write().unwrap();
            st.token = Some(TokenState {
                access_token: "live".into(),
                refresh_token: "r".into(),
                expires_at: Instant::now() + Duration::from_secs(300),
                refresh_expires_at: None,
            });
        }
        assert_eq!(
            with_deadline(auth.get_token()).await.unwrap(),
            "live",
            "the fast path must not be gated on backoff"
        );
    }

    // ---- transport security ----

    #[test]
    fn try_new_refuses_plaintext_without_opt_in() {
        assert!(ApiKeyAuth::try_new("http://gateway.internal", "u", "p", false).is_err());
    }

    #[test]
    fn try_new_allows_plaintext_when_opted_in() {
        assert!(ApiKeyAuth::try_new("http://localhost:8080", "u", "p", true).is_ok());
    }

    #[test]
    fn try_new_accepts_tls_without_opt_in() {
        assert!(ApiKeyAuth::try_new("https://gateway.example", "u", "p", false).is_ok());
    }

    // ---- secret hygiene ----

    #[test]
    fn token_state_debug_redacts_both_tokens() {
        let s = TokenState {
            access_token: "super-secret-access".into(),
            refresh_token: "super-secret-refresh".into(),
            expires_at: Instant::now(),
            refresh_expires_at: None,
        };
        let rendered = format!("{s:?}");
        assert!(!rendered.contains("super-secret-access"), "{rendered}");
        assert!(!rendered.contains("super-secret-refresh"), "{rendered}");
        assert!(rendered.contains("redacted"));
    }

    /// The upstream body must not ride into the error message (and from there
    /// into every log line) — it belongs on `source`.
    #[tokio::test]
    async fn upstream_body_stays_out_of_the_error_message() {
        let gw = spawn_gateway(GatewayConfig::new(503, 200)).await;
        let auth = ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw");
        seed(&auth, "t", None);

        let err = with_deadline(auth.get_token()).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("503"), "status should survive: {msg}");
        assert!(
            !msg.contains("Token is not active"),
            "body must not be interpolated into the message: {msg}"
        );
        // Still reachable for deliberate diagnosis.
        let chain = format!("{:#}", anyhow::Error::from(err));
        assert!(chain.contains("Token is not active"), "{chain}");
    }

    /// Concurrency is the reason the write lock spans the network call: N
    /// callers must collapse into one login, not N.
    #[tokio::test]
    async fn concurrent_callers_trigger_a_single_login() {
        let gw = spawn_gateway(GatewayConfig::new(400, 200)).await;
        let auth = Arc::new(ApiKeyAuth::new(&gw.base_url, "user@example.com", "pw"));

        let tokens = with_deadline(futures::future::join_all((0..10).map(|_| {
            let a = auth.clone();
            async move { a.get_token().await.unwrap() }
        })))
        .await;

        assert!(tokens.iter().all(|t| t == "from_login"));
        assert_eq!(
            gw.login_hits.load(Ordering::SeqCst),
            1,
            "single-flight: 10 callers must not cause 10 logins"
        );
    }
}
