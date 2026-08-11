//! End-user token auth with self-rotation.
//!
//! Wraps a Magick Mind end-user JWT (`POST /v1/end-users/tokens`) and keeps it
//! alive by rotating through `POST /v1/end-user/tokens/refresh` before expiry.
//! Agent tokens live one hour by default and 24 hours at most, so a process
//! meant to run for weeks cannot hold a static one.
//!
//! # Why this is not `ApiKeyAuth`
//!
//! `ApiKeyAuth` holds an account password, so any failure is recoverable: it
//! logs in again. This type holds no reusable credential — the presented token
//! *is* the credential, and the server revokes it as it issues the successor
//! (RFC 9700 rotation). Three consequences shape everything below:
//!
//! 1. **A rejected rotation is terminal.** There is no fallback to re-mint;
//!    only the spawner that minted the original can do that. So a rejection
//!    latches [`TokenFate::Dead`] rather than triggering a retry.
//! 2. **Concurrent rotations are chain-fatal.** Presenting an already-rotated
//!    jti outside the replay window makes the server revoke the *whole chain*.
//!    Rotation therefore happens under a write lock, single-flight.
//! 3. **A response we fail to store is a lost credential.** The server has
//!    already revoked the old token by the time we parse the reply, so a
//!    dropped response cannot be recovered by asking again.
//!
//! Rotation is proactive: [`get_token`](Auth::get_token) rotates once the
//! remaining lifetime falls under [`ROTATE_BEFORE`], rather than waiting for
//! expiry and failing a live request.
//!
//! # The rotated token is held in memory only
//!
//! Nothing writes the new credential back: `auth.token` is read-only input. So
//! after the first rotation, a token supplied from a config file is stale by
//! construction, and the server has already revoked it. On restart the process
//! presents that dead token and gets a 401.
//!
//! Worse, if the file's `token_expires_at` still shows time remaining,
//! `get_token` takes the cached fast path and never attempts a rotation — the
//! symptom is silent 401s from every call rather than a legible auth error.
//!
//! Restarts are ordinary (pod restart, OOM kill, config reload, node drain), so
//! a token written once into a config file goes stale. Use one of:
//!
//! - `auth.token_file`, which a control plane rewrites and this type re-reads,
//! - a control plane that mints a fresh token per process start, or
//! - `rotate_token = false` (*supervised* mode) with a supervisor redelivering.
//!
//! Supervised mode still uses this type — it holds the credential without
//! rotating it, so expiry, rejection, and terminal state stay visible. A dumb
//! token holder would report healthy while every call 401s.
//!
//! This type never *writes* the file, so a self-refreshed token remains
//! memory-only and a restart still needs a delivered or freshly minted one.
//!
//! # Recovering a chain that hit its absolute cap
//!
//! Rotation extends a chain but cannot outlive its absolute cap (30 days). Past
//! that, only a fresh mint helps, and minting needs the service-user credential
//! the agent deliberately does not hold — so recovery is necessarily the control
//! plane's job, not the agent's.
//!
//! This type makes the death legible: the credential latches terminal, callers
//! see a clear error, and a reconnect loop reading [`Auth::is_terminal`] stops
//! rather than 401-looping forever.
//!
//! Two delivery channels bring a fresh credential back in, both adopted without
//! a restart:
//!
//! - [`replace_token`](EndUserAuth::replace_token) — for a control plane in the
//!   *same process*, which holds this object.
//! - [`with_token_file`](EndUserAuth::with_token_file) — for a *separate*
//!   process. The control plane writes the mint response to a path; this type
//!   re-reads it before rotating and whenever the credential is terminal.
//!
//! Both are scoped per credential rather than read from ambient state. An
//! environment variable would not be: `std::env` is per-process, so a host
//! running several agents could let one adopt another's token — a valid
//! credential under the wrong identity, with no error.
//!
//! Neither is required. With no channel configured, or none written, the
//! credential simply dies on its own terms and the process exits for a
//! supervisor to restart.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

use crate::{Auth, MindroidError, Result};

/// Rotate once the token has less than this left — long enough that a transient
/// outage has room for several retries before the credential is lost.
///
/// A driver that polls `get_token` must tick *inside* this window or the call is
/// a no-op; see [`rotation_deadline`].
pub const ROTATE_BEFORE: Duration = Duration::from_secs(120);

/// When a poll-driven caller should next call `get_token` for a token with
/// `ttl_secs` of life left, so the call lands inside the rotation window.
///
/// Deriving the cadence from a fraction of the TTL does not work: at the
/// server's 3600s default, 80% is 2880s, which leaves 720s remaining — six times
/// [`ROTATE_BEFORE`], so the poll returns the cached token and rotates nothing.
/// The token then expires at 3600s and the next rotation presents an expired JWT,
/// which is rejected as terminal. Only a TTL under 600s would have worked.
///
/// The tick aims at the middle of the window rather than its leading edge, so
/// clock jitter cannot push it outside, and a failed rotation still has half the
/// window left for a backoff retry before the credential is lost.
pub fn rotation_deadline(ttl_secs: u64) -> Duration {
    const MIN_TICK: Duration = Duration::from_secs(30);

    // Fire when `ROTATE_BEFORE / 2` remains: inside the window by construction,
    // for every TTL, which a fraction of the TTL can never guarantee.
    let lead = ROTATE_BEFORE / 2;
    Duration::from_secs(ttl_secs)
        .saturating_sub(lead)
        .max(MIN_TICK)
}

/// Below this a token is unusable. Between here and `ROTATE_BEFORE` we prefer to
/// rotate but will still serve the current token if rotation is failing.
const MIN_USABLE: Duration = Duration::from_secs(10);

const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Ceiling for one rotation. Held under the write lock, so an unbounded request
/// would park every caller behind it.
const ROTATE_TIMEOUT: Duration = Duration::from_secs(20);

/// Ceiling on reading the delivered-token file. Also held under the write lock,
/// and a token file commonly lives on a mount that can hang.
const DELIVERY_READ_TIMEOUT: Duration = Duration::from_secs(5);

const BACKOFF_BASE: Duration = Duration::from_secs(1);

/// Ceiling on retry spacing.
///
/// Deliberately above 60s: the refresh route allows 60 attempts per hour and the
/// server counts rejected ones, so retrying exactly once a minute consumes the
/// entire budget with no headroom — a client that recovers mid-window then finds
/// its legitimate rotation refused. 120s leaves half the budget free.
const BACKOFF_MAX: Duration = Duration::from_secs(120);

/// Exponential backoff, capped. The refresh route is rate-limited per chain
/// (60/hour default); burning that budget on an outage makes it permanent.
fn backoff_delay(consecutive_failures: u32) -> Duration {
    // Shift far enough to actually reach BACKOFF_MAX — capping at 6 would stop
    // growth at 64s and silently pin the ceiling below its stated value.
    let shift = consecutive_failures.saturating_sub(1).min(10);
    BACKOFF_BASE.saturating_mul(1u32 << shift).min(BACKOFF_MAX)
}

/// `base + secs`, or `None` if a server-supplied lifetime overflows the clock.
fn checked_deadline(base: Instant, secs: u64) -> Option<Instant> {
    base.checked_add(Duration::from_secs(secs))
}

/// Audience marking a token the server will not let refresh itself.
const SUPERVISED_AUDIENCE: &str = "supervised";

/// A control plane's delivery channel into a running agent: the mint response,
/// written verbatim to a file this credential re-reads on need.
#[derive(Deserialize)]
struct DeliveredToken {
    token: String,
    #[serde(default)]
    expires_at: Option<String>,
    /// Accepted so the mint response can be written verbatim, but *not*
    /// preferred here: it is relative to when the server issued the token, and a
    /// file is read some unknown time after it was written, so the duration
    /// over-reports by however long it sat there. `expires_at` is absolute and
    /// therefore the safer of the two for delivery — the opposite precedence
    /// from a live response.
    #[serde(default)]
    expires_in: Option<u64>,
}

/// Read a delivered credential, if the file holds one that differs from what we
/// already have.
///
/// Every failure here is a miss rather than an error: an absent file is the
/// normal case, and a partial read is what a non-atomic writer produces. The
/// caller then proceeds with whatever it already held — so a supervisor that
/// writes nothing simply leaves the credential to die on its own terms, which is
/// the correct outcome.
/// Runs on a blocking thread under a deadline: `rotate_into` calls this while
/// holding the async write lock, so a hung path (a stalled network mount) would
/// otherwise park the worker thread and every concurrent `get_token` behind it.
/// `ROTATE_TIMEOUT` covers only the HTTP call, not this.
struct PendingDeliveryRead {
    current: String,
    handle: tokio::task::JoinHandle<Option<DeliveredToken>>,
}

type DeliveryRead = Arc<Mutex<Option<PendingDeliveryRead>>>;

async fn read_delivered_async(
    pending: &DeliveryRead,
    path: &str,
    current: &str,
) -> Option<DeliveredToken> {
    read_delivered_with_timeout(pending, path, current, DELIVERY_READ_TIMEOUT).await
}

async fn read_delivered_with_timeout(
    pending: &DeliveryRead,
    path: &str,
    current: &str,
    timeout: Duration,
) -> Option<DeliveredToken> {
    let mut pending = pending.lock().await;
    if pending.is_none() {
        let (path, current) = (path.to_string(), current.to_string());
        let snapshot = current.clone();
        *pending = Some(PendingDeliveryRead {
            current: snapshot,
            handle: tokio::task::spawn_blocking(move || read_delivered(&path, &current)),
        });
    }
    let read = pending.as_mut()?;
    let same_generation = read.current == current;
    let result = tokio::time::timeout(timeout, &mut read.handle).await;
    match result {
        Ok(Ok(delivered)) => {
            pending.take();
            same_generation.then_some(delivered).flatten()
        }
        Ok(Err(e)) => {
            pending.take();
            warn!("Delivered-token read panicked: {e}");
            None
        }
        Err(_) => {
            warn!(
                "Delivered-token read exceeded {}s; proceeding without it",
                DELIVERY_READ_TIMEOUT.as_secs()
            );
            None
        }
    }
}

fn read_delivered(path: &str, current: &str) -> Option<DeliveredToken> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| debug!("No delivered token at {path}: {e}"))
        .ok()?;

    let delivered: DeliveredToken = serde_json::from_str(&raw)
        .map_err(|e| warn!("Ignoring malformed token file {path}: {e}"))
        .ok()?;

    if delivered.token.is_empty() || delivered.token == current {
        return None;
    }
    Some(delivered)
}

/// Read the `aud` claim without verifying the signature.
///
/// `aud` is either a string or an array of them per RFC 7519, so both are
/// accepted. Unverified is fine here: this only produces a clearer startup error
/// than the 403 the server would return anyway.
fn jwt_audience(token: &str) -> Option<Vec<String>> {
    use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};

    let payload = token.split('.').nth(1)?;
    let decoded = URL_SAFE_NO_PAD.decode(payload).ok()?;
    let json: serde_json::Value = serde_json::from_slice(&decoded).ok()?;

    match json.get("aud")? {
        serde_json::Value::String(s) => Some(vec![s.clone()]),
        serde_json::Value::Array(items) => Some(
            items
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
        ),
        _ => None,
    }
}

#[derive(Serialize)]
struct RefreshRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl_seconds: Option<i64>,
}

/// `MintEndUserTokenResponse`.
#[derive(Deserialize)]
struct TokenResponse {
    token: String,
    #[serde(default)]
    expires_at: String,
    /// Lifetime in seconds, preferred over `expires_at` when the server sends
    /// it. See [`resolve_deadline`].
    #[serde(default)]
    expires_in: Option<u64>,
    #[serde(default)]
    #[allow(dead_code)]
    token_type: String,
}

/// The live credential.
struct TokenState {
    token: String,
    expires_at: Instant,
}

impl fmt::Debug for TokenState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenState")
            .field("token", &"<redacted>")
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

impl TokenState {
    fn remaining_at(&self, now: Instant) -> Duration {
        self.expires_at.saturating_duration_since(now)
    }

    fn usable_at(&self, now: Instant) -> bool {
        self.remaining_at(now) > MIN_USABLE
    }

    fn wants_rotation_at(&self, now: Instant) -> bool {
        self.remaining_at(now) <= ROTATE_BEFORE
    }
}

/// Whether the credential can still be recovered — an unreachable gateway means
/// "retry", a revoked chain means "re-mint or stop".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenFate {
    Live,
    /// Server refused the credential. Only a fresh mint restores service.
    Dead,
}

/// Upstream body, carried as an error `source` rather than interpolated into the
/// message — auth errors reach logs with a broader audience than the auth path.
#[derive(Debug)]
struct UpstreamBody(String);

impl fmt::Display for UpstreamBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UpstreamBody {}

/// Why a rotation failed, at the granularity the retry policy needs.
enum RotateFailure {
    /// Expired, revoked, replayed, or supervised. No retry helps.
    Rejected(MindroidError),
    /// 429. The chain is intact — kept separate so it does not latch `Dead`.
    RateLimited(MindroidError),
    /// Transport failure or 5xx: says nothing about the credential.
    Unavailable(MindroidError),
}

impl RotateFailure {
    fn from_status(status: u16, message: String, body: String) -> Self {
        let err = MindroidError::Auth {
            message,
            source: Some(Box::new(UpstreamBody(body))),
        };
        match status {
            429 => Self::RateLimited(err),
            // 403 is the supervised bar; 401 covers expired/revoked/replayed.
            401 | 403 => Self::Rejected(err),
            // 400 is a malformed request — an over-cap `token_ttl_seconds`, say.
            // The credential may be perfectly good, so latching Dead here turns a
            // config typo into an agent that kills itself an hour after start and
            // never recovers. Retryable, so the operator sees repeated failures
            // with the reason instead of one silent exit.
            _ => Self::Unavailable(err),
        }
    }

    fn into_error(self) -> MindroidError {
        match self {
            Self::Rejected(e) | Self::RateLimited(e) | Self::Unavailable(e) => e,
        }
    }
}

#[derive(Default)]
struct FailureState {
    consecutive: u32,
    retry_after: Option<Instant>,
    last_error: Option<String>,
}

impl FailureState {
    fn record_success(&mut self) {
        self.consecutive = 0;
        self.retry_after = None;
        self.last_error = None;
    }

    fn record_failure(&mut self, now: Instant, err: &MindroidError) {
        self.consecutive = self.consecutive.saturating_add(1);
        self.retry_after = now.checked_add(backoff_delay(self.consecutive));
        self.last_error = Some(err.to_string());
    }

    fn in_backoff_at(&self, now: Instant) -> bool {
        self.retry_after.is_some_and(|t| t > now)
    }
}

struct AuthState {
    token: TokenState,
    fate: TokenFate,
    failure: FailureState,
}

impl AuthState {
    fn dead_error(&self) -> MindroidError {
        MindroidError::Auth {
            message: format!(
                "end-user token is permanently invalid and cannot be rotated \
                 in-process (re-mint required); last error: {}",
                self.failure.last_error.as_deref().unwrap_or("unknown")
            ),
            source: None,
        }
    }
}

/// Parse an RFC 3339 `expires_at` into a monotonic deadline.
///
/// Anchors the deadline to a monotonic `Instant`, so wall-clock drift *after*
/// parsing cannot move it. Skew at parse time is NOT handled: the remaining
/// duration is `expires - Utc::now()`, so a fast host clock rotates early and a
/// slow one leaves an expired token unrotated while `is_authenticated` still
/// reports true. [`resolve_deadline`] prefers a server-supplied `expires_in`
/// precisely to avoid this — but that field is not sent yet, so this path is
/// still the live one.
///
/// `None` on an unparseable or past deadline, which the caller reads as
/// "rotate now".
/// Resolve a token's deadline, preferring a server-supplied lifetime.
///
/// `expires_in` is a *duration*, so it needs no clock reading at all and is
/// immune to host/server skew. `expires_at` is an absolute timestamp, so
/// converting it means subtracting our own wall clock — a slow host then
/// believes the token outlives its real expiry and serves a dead credential
/// while reporting healthy.
///
/// Forward-compatible: the field is optional, so this prefers `expires_in` the
/// moment the server starts sending it and falls back until then. `None` means
/// "rotate now", which is the safe direction to be wrong in.
fn resolve_deadline(expires_in: Option<u64>, expires_at: &str, now: Instant) -> Option<Instant> {
    match expires_in {
        Some(0) => None,
        Some(secs) => checked_deadline(now, secs),
        None => parse_expires_at(expires_at, now),
    }
}

fn parse_expires_at(value: &str, now: Instant) -> Option<Instant> {
    let expires = chrono::DateTime::parse_from_rfc3339(value).ok()?;
    let remaining = expires.timestamp() - chrono::Utc::now().timestamp();
    if remaining <= 0 {
        return None;
    }
    checked_deadline(now, remaining as u64)
}

/// Self-rotating end-user credential. See the module docs for why rejection is
/// terminal rather than retried.
pub struct EndUserAuth {
    base_url: String,
    /// TTL requested on each rotation. `None` takes the server default (1h).
    ttl_seconds: Option<i64>,
    client: reqwest::Client,
    state: Arc<RwLock<AuthState>>,
    /// Something out of band re-mints and redelivers this credential, so this
    /// process must not rotate it.
    supervised: bool,
    /// Path a control plane writes freshly minted credentials to, re-read on
    /// need. `None` disables the channel.
    token_file: Option<String>,
    delivery_read: DeliveryRead,
}

impl EndUserAuth {
    /// Construct from a freshly minted end-user JWT. `expires_at` is the RFC 3339
    /// deadline from the mint response; `None` rotates on first use.
    pub fn try_new(
        base_url: &str,
        token: impl Into<String>,
        expires_at: Option<&str>,
        allow_insecure: bool,
    ) -> Result<Self> {
        let base_url = base_url.trim_end_matches('/').to_string();
        // Parses the URL rather than prefix-matching: schemes are case-insensitive,
        // so `HTTP://host` normalizes to http and a `starts_with("http://")` test
        // would wave it through with the live token attached.
        crate::core::net::require_secure_url(&base_url, allow_insecure, "auth.allow_insecure")?;

        let now = Instant::now();
        // No parseable deadline → already due.
        let deadline = expires_at
            .and_then(|s| parse_expires_at(s, now))
            .unwrap_or(now);

        // No redirects: reqwest compares host and port without the scheme when
        // deciding whether to strip Authorization, so a same-host https -> http
        // redirect would forward the credential in cleartext. If the hardened
        // builder can't be built, fail closed rather than falling back to
        // reqwest's weaker defaults (redirects on, no timeouts).
        let builder = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none());

        // reqwest honours HTTP_PROXY/ALL_PROXY without exempting loopback.
        #[cfg(test)]
        let builder = builder.no_proxy();

        let client = builder.build().map_err(|e| MindroidError::Auth {
            message: format!("failed to build the end-user auth HTTP client: {e}"),
            source: Some(Box::new(e)),
        })?;

        Ok(Self {
            base_url,
            ttl_seconds: None,
            client,
            supervised: false,
            token_file: None,
            delivery_read: Arc::new(Mutex::new(None)),
            state: Arc::new(RwLock::new(AuthState {
                token: TokenState {
                    token: token.into(),
                    expires_at: deadline,
                },
                fate: TokenFate::Live,
                failure: FailureState::default(),
            })),
        })
    }

    /// TTL requested per rotation. The server rejects anything above its own cap
    /// (24h default), so an over-large value makes every rotation fail.
    pub fn with_ttl_seconds(mut self, ttl: i64) -> Self {
        self.ttl_seconds = Some(ttl);
        self
    }

    /// Hold the credential without rotating it — the *supervised* mode, where a
    /// control plane re-mints out of band and redelivers.
    ///
    /// Preferred over swapping in a plain static token holder: this still tracks
    /// expiry, still latches [`Auth::note_rejection`] when the server rejects the
    /// credential, and still reports [`Auth::kind`] and [`Auth::is_terminal`]
    /// honestly. A dumb holder reports healthy forever, so a supervisor that
    /// stops redelivering produces an agent that 401s every call while looking
    /// alive — the exact failure the terminal flag exists to surface.
    pub fn supervised(mut self) -> Self {
        self.supervised = true;
        self
    }

    /// Watch `path` for credentials a control plane delivers.
    ///
    /// The cross-process counterpart to [`replace_token`](Self::replace_token):
    /// that one needs a handle to this object, so it only serves a control plane
    /// in the same process. A pod-per-agent deployment writes the mint response
    /// to a file instead.
    ///
    /// Read on need — before a rotation, and whenever the credential is terminal
    /// — so nothing is stat-ed on the healthy path. The file is never written by
    /// this type.
    pub fn with_token_file(mut self, path: impl Into<String>) -> Self {
        self.token_file = Some(path.into());
        self
    }

    /// Install a freshly minted credential, replacing whatever is held.
    ///
    /// This is the control plane's delivery channel into a *running* agent. The
    /// chain has an absolute cap that no rotation can extend, and only the
    /// service-user credential can mint past it — which the agent deliberately
    /// does not hold. Without this, the only recovery is restarting the process.
    ///
    /// Clears terminal state and any rotation backoff: this is a different
    /// credential, not a retry of the dead one. `expires_at` is the RFC 3339
    /// deadline from the mint response; `None` rotates on first use, which costs
    /// one round-trip and is always safe.
    ///
    /// Scoped to this credential by construction. A process hosting several
    /// agents calls it on each agent's own `EndUserAuth`, so tokens cannot be
    /// crossed — which is why this is a method rather than an environment
    /// variable, `std::env` being per-process rather than per-agent.
    ///
    /// ```ignore
    /// // Control plane, on noticing the agent went terminal:
    /// let minted = mint_end_user_token(&service_auth, agent_id).await?;
    /// agent_auth.replace_token(minted.token, Some(&minted.expires_at));
    /// ```
    pub async fn replace_token(&self, token: impl Into<String>, expires_at: Option<&str>) {
        let now = Instant::now();
        let mut state = self.state.write().await;

        state.token = TokenState {
            token: token.into(),
            expires_at: expires_at
                .and_then(|s| parse_expires_at(s, now))
                .unwrap_or(now),
        };
        state.fate = TokenFate::Live;
        state.failure.record_success();

        info!("Adopted a control-plane-supplied end-user token");
    }

    /// Fail now if this token is barred from self-refresh.
    ///
    /// A delegated mint issues a *supervised* token unless the caller opts in to
    /// self-refresh, and the refresh route answers a supervised token with 403,
    /// which is terminal. Without this check the composition of two defensible
    /// defaults — the server minting supervised, mindroid rotating by default —
    /// is an agent that starts cleanly and kills itself at the first rotation,
    /// an hour later.
    ///
    /// Reads the unverified `aud` claim. The server verifies the signature; a
    /// forged claim only produces a spurious startup error, never a false pass.
    pub fn ensure_self_refreshable(&self) -> Result<()> {
        let token = self
            .state
            .try_read()
            .map(|s| s.token.token.clone())
            .unwrap_or_default();

        // No audience claim to inspect — let the server be the judge.
        let Some(aud) = jwt_audience(&token) else {
            return Ok(());
        };

        if aud.iter().any(|a| a == SUPERVISED_AUDIENCE) {
            return Err(MindroidError::config(format!(
                "this end-user token is supervised (aud={SUPERVISED_AUDIENCE}) and \
                 the refresh route rejects it with 403: mint it with self-refresh \
                 enabled, or set auth.rotate_token = false and rotate out of band"
            )));
        }
        Ok(())
    }

    /// Whether the credential is permanently dead. A supervisor polls this to
    /// distinguish "retry" from "stop": once true, waiting cannot help.
    ///
    /// Also exposed as [`Auth::is_terminal`], which is how the transport's
    /// reconnect loop reaches it through `Arc<dyn Auth>`.
    pub fn is_terminal(&self) -> bool {
        self.state
            .try_read()
            .is_ok_and(|s| s.fate == TokenFate::Dead)
    }

    /// POST the rotation and classify the outcome.
    ///
    /// Takes the token by parameter rather than reading it from `self.state`:
    /// the caller holds the write lock for the whole rotation, and
    /// `tokio::sync::RwLock` is write-preferring, so acquiring a read lock here
    /// would deadlock against the caller's own guard.
    async fn post_rotate(&self, token: &str) -> std::result::Result<TokenResponse, RotateFailure> {
        let url = format!("{}/v1/end-user/tokens/refresh", self.base_url);

        let resp = self
            .client
            .post(&url)
            .bearer_auth(token)
            .json(&RefreshRequest {
                ttl_seconds: self.ttl_seconds,
            })
            .send()
            .await
            .map_err(|e| {
                RotateFailure::Unavailable(MindroidError::Auth {
                    message: format!("token rotation request failed: {e}"),
                    source: Some(Box::new(e)),
                })
            })?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(RotateFailure::from_status(
                status,
                format!("token rotation failed (HTTP {status})"),
                body,
            ));
        }

        // A 2xx we cannot parse means the server already revoked the presented
        // token and we lost the successor — retrying would kill the chain.
        resp.json::<TokenResponse>().await.map_err(|e| {
            RotateFailure::Rejected(MindroidError::Auth {
                message: format!(
                    "token rotation returned an unparseable success response \
                     ({e}); the presented token has already been revoked \
                     server-side and the successor is lost — re-mint required"
                ),
                source: Some(Box::new(e)),
            })
        })
    }

    /// Rotate under the write lock. Single-flight by construction: two concurrent
    /// rotations would replay a jti and revoke the whole chain.
    async fn rotate_into(&self, state: &mut AuthState) -> Result<String> {
        let now = Instant::now();

        // Checked BEFORE the terminal guard: a credential past the chain's
        // absolute cap is exactly the one a control plane needs to replace, so
        // refusing to look would make the channel useless in the case it exists
        // for. A miss (no file, unreadable, unchanged) falls through and the
        // credential dies on its own terms — the file is a chance to recover,
        // not a requirement.
        if let Some(path) = &self.token_file
            && let Some(delivered) =
                read_delivered_async(&self.delivery_read, path, &state.token.token).await
        {
            info!("Adopting a delivered end-user token from {path}");
            state.token = TokenState {
                // `expires_at` first here: see DeliveredToken::expires_in.
                expires_at: delivered
                    .expires_at
                    .as_deref()
                    .and_then(|s| parse_expires_at(s, now))
                    .or_else(|| delivered.expires_in.and_then(|s| checked_deadline(now, s)))
                    .unwrap_or(now),
                token: delivered.token,
            };
            state.fate = TokenFate::Live;
            state.failure.record_success();
            return Ok(state.token.token.clone());
        }

        if state.fate == TokenFate::Dead {
            return Err(state.dead_error());
        }

        // Supervised: a control plane owns this credential. Serve what we hold
        // while it is usable, and say plainly when it is not — rotating here
        // would present a token the refresh route bars, killing the chain.
        if self.supervised {
            if state.token.usable_at(now) {
                return Ok(state.token.token.clone());
            }
            return Err(MindroidError::Auth {
                message: "supervised end-user token expired and this process does \
                          not rotate it: the control plane must redeliver a fresh \
                          token (or set auth.rotate_token = true to self-refresh)"
                    .to_string(),
                source: None,
            });
        }
        if state.failure.in_backoff_at(now) {
            if state.token.usable_at(now) {
                return Ok(state.token.token.clone());
            }
            // Backoff outlasted the token. The presented credential can no longer
            // be rotated (the server revokes on expiry), so waiting cannot help —
            // latch Dead rather than reconnect-looping forever holding the
            // runtime open. This is the one path that reached an unrecoverable
            // state without setting the flag that exists to stop it.
            warn!(
                "End-user token expired while rotation was backing off; \
                 the credential can no longer be recovered in-process"
            );
            state.fate = TokenFate::Dead;
            return Err(state.dead_error());
        }

        let presented = state.token.token.clone();
        let rotated = tokio::time::timeout(ROTATE_TIMEOUT, self.post_rotate(&presented))
            .await
            .unwrap_or_else(|_| {
                // Ambiguous: the server may have rotated and we never saw the
                // reply. Treat the presented token as spent — reusing it risks
                // the chain.
                Err(RotateFailure::Rejected(MindroidError::Auth {
                    message: format!(
                        "token rotation exceeded {}s deadline; the rotation may \
                         have succeeded server-side, so the presented token \
                         cannot safely be reused — re-mint required",
                        ROTATE_TIMEOUT.as_secs()
                    ),
                    source: None,
                }))
            });

        match rotated {
            Ok(resp) => {
                let now = Instant::now();
                let expires_at = resolve_deadline(resp.expires_in, &resp.expires_at, now)
                    .unwrap_or_else(|| {
                        warn!(
                            "rotated token has no usable lifetime (expires_in={:?}, \
                             expires_at={:?}); treating it as immediately due",
                            resp.expires_in, resp.expires_at
                        );
                        now
                    });
                debug!(
                    "rotated end-user token, {}s remaining",
                    expires_at.saturating_duration_since(now).as_secs()
                );
                state.token = TokenState {
                    token: resp.token.clone(),
                    expires_at,
                };
                state.failure.record_success();
                Ok(resp.token)
            }
            Err(failure) => {
                let terminal = matches!(failure, RotateFailure::Rejected(_));
                let err = failure.into_error();
                if terminal {
                    // Latch, so later calls fail fast instead of re-presenting a
                    // token the server has already refused.
                    state.fate = TokenFate::Dead;
                    warn!("end-user token is permanently invalid: {err}");
                }
                state.failure.record_failure(now, &err);

                // A rate-limit or outage says nothing about the token we hold.
                // Serve it while it lasts rather than failing a turn the agent
                // could still have completed.
                if !terminal && state.token.usable_at(now) {
                    debug!("rotation failed ({err}); serving the current token");
                    return Ok(state.token.token.clone());
                }
                Err(err)
            }
        }
    }
}

#[async_trait]
impl Auth for EndUserAuth {
    async fn get_token(&self) -> Result<String> {
        {
            let state = self.state.read().await;
            let now = Instant::now();
            if state.fate == TokenFate::Live && !state.token.wants_rotation_at(now) {
                return Ok(state.token.token.clone());
            }
        }

        let mut state = self.state.write().await;

        // Re-check: another task may have rotated while we waited. Without this
        // every blocked caller would rotate again and replay a spent jti.
        let now = Instant::now();
        if state.fate == TokenFate::Live && !state.token.wants_rotation_at(now) {
            return Ok(state.token.token.clone());
        }

        self.rotate_into(&mut state).await
    }

    async fn get_auth_headers(&self) -> Result<Vec<(String, String)>> {
        let token = self.get_token().await?;
        Ok(vec![(
            "Authorization".to_string(),
            format!("Bearer {token}"),
        )])
    }

    fn is_authenticated(&self) -> bool {
        // A contended lock means a rotation is in flight; treat "unknown" as
        // authenticated and let `get_token` produce the real answer.
        match self.state.try_read() {
            Ok(state) => state.fate == TokenFate::Live && state.token.usable_at(Instant::now()),
            Err(_) => true,
        }
    }

    /// Force a rotation. This consumes the current token and spends a slot of the
    /// chain's rate-limit budget — `get_token` already rotates on its own
    /// schedule, so use this only when reacting to a 401 from elsewhere.
    async fn refresh(&self) -> Result<()> {
        let mut state = self.state.write().await;
        self.rotate_into(&mut state).await.map(|_| ())
    }

    fn is_terminal(&self) -> bool {
        EndUserAuth::is_terminal(self)
    }

    fn kind(&self) -> crate::models::CredentialKind {
        crate::models::CredentialKind::EndUser
    }

    /// Latch `Dead` on a rejection any caller observed.
    ///
    /// `rotate_into` is the only other writer of this state, so without this a
    /// revocation discovered through a 401 elsewhere leaves the credential
    /// reporting healthy while every request fails.
    ///
    /// Uses `try_write` and gives up if contended: a rotation holding the lock
    /// is about to classify the outcome itself, so its verdict is the better one.
    fn note_rejection(&self) {
        if let Ok(mut state) = self.state.try_write() {
            if state.fate != TokenFate::Dead {
                tracing::warn!("End-user credential rejected by an upstream call; marking it dead");
            }
            state.fate = TokenFate::Dead;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::net::TcpListener;

    /// A minimal rotation endpoint. Returns the queued status/body per hit and
    /// counts requests so single-flight can be asserted.
    struct MockGateway {
        base_url: String,
        hits: Arc<AtomicUsize>,
        server: tokio::task::JoinHandle<()>,
    }

    /// Stop the listener with the test that owns it, so sockets and tasks don't
    /// accumulate for the life of the test binary.
    impl Drop for MockGateway {
        fn drop(&mut self) {
            self.server.abort();
        }
    }

    async fn spawn_gateway(responses: Vec<(u16, String)>, delay: Option<Duration>) -> MockGateway {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_clone = hits.clone();

        let server = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let responses = responses.clone();
                let hits = hits_clone.clone();
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    // Single read: the request is small enough that hyper
                    // coalesces headers and body into one segment.
                    let mut buf = vec![0u8; 4096];
                    let Ok(_) = stream.read(&mut buf).await else {
                        return;
                    };
                    let n = hits.fetch_add(1, Ordering::SeqCst);
                    if let Some(d) = delay {
                        tokio::time::sleep(d).await;
                    }
                    let (status, body) = responses
                        .get(n)
                        .cloned()
                        .unwrap_or_else(|| (500, "exhausted".into()));
                    // Empty reason phrase: valid HTTP/1.1, and avoids the
                    // nonsense of "403 OK".
                    let resp = format!(
                        "HTTP/1.1 {status} \r\nContent-Type: application/json\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(resp.as_bytes()).await;
                    let _ = stream.flush().await;
                });
            }
        });

        MockGateway {
            base_url: format!("http://{addr}"),
            hits,
            server,
        }
    }

    /// Every network assertion runs under this so a stalled handler fails
    /// legibly instead of burning the full rotation deadline.
    async fn with_deadline<F: std::future::Future>(f: F) -> F::Output {
        tokio::time::timeout(Duration::from_secs(5), f)
            .await
            .expect("gateway interaction timed out")
    }

    fn token_body(token: &str, secs: i64) -> String {
        let expires = chrono::Utc::now() + chrono::Duration::seconds(secs);
        format!(
            r#"{{"token":"{token}","expires_at":"{}","token_type":"Bearer"}}"#,
            expires.to_rfc3339()
        )
    }

    /// A token with plenty of life left is served without touching the network.
    #[tokio::test]
    async fn fresh_token_is_served_without_rotating() {
        let gw = spawn_gateway(vec![], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(3600)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "original", Some(&expires), true).unwrap();

        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "original");
        assert_eq!(gw.hits.load(Ordering::SeqCst), 0, "should not have rotated");
    }

    /// Inside the rotation window the token is exchanged, and the successor is
    /// what later callers receive.
    #[tokio::test]
    async fn rotates_when_close_to_expiry() {
        let gw = spawn_gateway(vec![(200, token_body("rotated", 3600))], None).await;
        // 30s left — inside ROTATE_BEFORE (120s).
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(30)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "original", Some(&expires), true).unwrap();

        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "rotated");
        assert_eq!(gw.hits.load(Ordering::SeqCst), 1);
        // Second call is served from cache — the successor has a full lifetime.
        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "rotated");
        assert_eq!(gw.hits.load(Ordering::SeqCst), 1, "should not re-rotate");
    }

    /// `expires_in` is a duration, so it needs no clock reading and is immune to
    /// host/server skew. Preferred over `expires_at` whenever the server sends
    /// it — which it does not yet, hence the optional field.
    #[test]
    fn a_server_supplied_lifetime_wins_over_an_absolute_timestamp() {
        let now = Instant::now();
        // A timestamp that would resolve to ~1h, against an explicit 60s.
        let far = (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339();

        let deadline = resolve_deadline(Some(60), &far, now).expect("60s is usable");
        let remaining = deadline.saturating_duration_since(now);
        assert!(
            remaining <= Duration::from_secs(61) && remaining >= Duration::from_secs(59),
            "expires_in must win: got {remaining:?}"
        );
    }

    #[test]
    fn an_absent_lifetime_falls_back_to_the_timestamp() {
        let now = Instant::now();
        let in_an_hour = (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339();

        let deadline = resolve_deadline(None, &in_an_hour, now).expect("timestamp is usable");
        assert!(deadline.saturating_duration_since(now) > Duration::from_secs(3500));

        // Neither usable → rotate now.
        assert!(resolve_deadline(None, "not-a-timestamp", now).is_none());
        assert!(resolve_deadline(Some(0), &in_an_hour, now).is_none());
    }

    /// End to end: a server that starts sending `expires_in` is picked up off
    /// the wire with no other change.
    #[tokio::test]
    async fn a_rotation_response_carrying_expires_in_is_honoured() {
        let body = serde_json::json!({
            "token": "rotated",
            "expires_in": 3600,
            "token_type": "Bearer",
        })
        .to_string();
        let gw = spawn_gateway(vec![(200, body)], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(30)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "original", Some(&expires), true).unwrap();

        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "rotated");
        // A full hour of life, from expires_in alone — no expires_at was sent.
        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "rotated");
        assert_eq!(
            gw.hits.load(Ordering::SeqCst),
            1,
            "the successor's lifetime came from expires_in, so no re-rotation"
        );
    }

    /// The chain-fatal case: concurrent callers must produce exactly one
    /// rotation. A second would present a spent jti and revoke the chain.
    #[tokio::test]
    async fn concurrent_callers_rotate_once() {
        let gw = spawn_gateway(
            vec![(200, token_body("rotated", 3600))],
            Some(Duration::from_millis(80)),
        )
        .await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(30)).to_rfc3339();
        let auth =
            Arc::new(EndUserAuth::try_new(&gw.base_url, "original", Some(&expires), true).unwrap());

        let mut handles = Vec::new();
        for _ in 0..8 {
            let a = auth.clone();
            handles.push(tokio::spawn(
                async move { with_deadline(a.get_token()).await },
            ));
        }
        for h in handles {
            assert_eq!(h.await.unwrap().unwrap(), "rotated");
        }
        assert_eq!(
            gw.hits.load(Ordering::SeqCst),
            1,
            "concurrent rotation would replay a spent jti and kill the chain"
        );
    }

    /// A supervised token gets 403 and must latch dead rather than retrying —
    /// retrying cannot help, and it burns the chain's rate-limit budget.
    #[tokio::test]
    async fn supervised_token_is_terminal() {
        let gw = spawn_gateway(
            vec![
                (403, r#"{"msg":"supervisor-managed"}"#.into()),
                (200, token_body("never-reached", 3600)),
            ],
            None,
        )
        .await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(30)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "original", Some(&expires), true).unwrap();

        assert!(with_deadline(auth.get_token()).await.is_err());
        assert!(auth.is_terminal(), "403 must latch as terminal");
        assert!(!auth.is_authenticated());

        // Latched: no further network calls, even though the mock would now 200.
        assert!(with_deadline(auth.get_token()).await.is_err());
        assert_eq!(gw.hits.load(Ordering::SeqCst), 1, "must not retry a 403");
    }

    /// Build an unsigned JWT with the given payload — only the middle segment is
    /// ever inspected here.
    fn jwt_with(payload: serde_json::Value) -> String {
        use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
        format!(
            "eyJhbGciOiJIUzI1NiJ9.{}.sig",
            URL_SAFE_NO_PAD.encode(payload.to_string())
        )
    }

    fn write_token_file(dir: &std::path::Path, token: &str, expires_at: &str) -> String {
        let path = dir.join("token.json");
        std::fs::write(
            &path,
            serde_json::json!({ "token": token, "expires_at": expires_at }).to_string(),
        )
        .unwrap();
        path.to_string_lossy().into_owned()
    }

    /// The cross-process delivery channel: a control plane writes the mint
    /// response, and a terminal credential picks it up without a restart.
    #[tokio::test]
    async fn a_delivered_token_file_revives_a_dead_credential() {
        let tmp = tempfile::tempdir().unwrap();
        let gw = spawn_gateway(vec![(401, r#"{"msg":"chain expired"}"#.into())], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(30)).to_rfc3339();
        let path = tmp.path().join("token.json").to_string_lossy().into_owned();

        let auth = EndUserAuth::try_new(&gw.base_url, "spent", Some(&expires), true)
            .unwrap()
            .with_token_file(&path);

        // No file yet: the credential dies on its own terms.
        assert!(with_deadline(auth.get_token()).await.is_err());
        assert!(auth.is_terminal());

        let fresh = (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339();
        write_token_file(tmp.path(), "delivered", &fresh);

        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "delivered");
        assert!(!auth.is_terminal(), "delivery clears terminal state");
    }

    /// A supervisor that writes nothing must not keep the agent alive: no file
    /// means no rescue, and the process should exit rather than 401-loop.
    #[tokio::test]
    async fn an_absent_token_file_leaves_the_credential_dead() {
        let tmp = tempfile::tempdir().unwrap();
        let gw = spawn_gateway(vec![(401, r#"{"msg":"revoked"}"#.into())], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(30)).to_rfc3339();
        let missing = tmp.path().join("never-written.json");

        let auth = EndUserAuth::try_new(&gw.base_url, "spent", Some(&expires), true)
            .unwrap()
            .with_token_file(missing.to_string_lossy().as_ref());

        assert!(with_deadline(auth.get_token()).await.is_err());
        assert!(auth.is_terminal());
        // Still dead on a later call — nothing revived it.
        assert!(with_deadline(auth.get_token()).await.is_err());
        assert!(auth.is_terminal());
    }

    /// A half-written file is what a non-atomic writer produces. Ignore it and
    /// keep what we hold rather than adopting garbage.
    #[tokio::test]
    async fn a_malformed_token_file_is_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        let gw = spawn_gateway(vec![], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339();
        let path = tmp.path().join("token.json");
        std::fs::write(&path, r#"{"token": "half-writ"#).unwrap();

        let auth = EndUserAuth::try_new(&gw.base_url, "held", Some(&expires), true)
            .unwrap()
            .with_token_file(path.to_string_lossy().as_ref());

        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "held");
    }

    #[tokio::test]
    async fn a_timed_out_delivery_read_is_reused_instead_of_respawned() {
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let worker_barrier = barrier.clone();
        let handle = tokio::task::spawn_blocking(move || {
            worker_barrier.wait();
            None
        });
        let task_id = handle.id();
        let pending: DeliveryRead = Arc::new(Mutex::new(Some(PendingDeliveryRead {
            current: "unused".to_string(),
            handle,
        })));

        assert!(
            read_delivered_with_timeout(&pending, "unused", "unused", Duration::from_millis(10),)
                .await
                .is_none()
        );
        assert_eq!(pending.lock().await.as_ref().unwrap().handle.id(), task_id);

        barrier.wait();
        assert!(
            read_delivered_with_timeout(&pending, "unused", "unused", Duration::from_secs(1))
                .await
                .is_none()
        );
        assert!(pending.lock().await.is_none());
    }

    #[tokio::test]
    async fn a_completed_delivery_read_is_ignored_after_token_generation_changes() {
        let handle = tokio::task::spawn_blocking(|| {
            Some(DeliveredToken {
                token: "delivered-for-old-generation".to_string(),
                expires_at: None,
                expires_in: None,
            })
        });
        let pending: DeliveryRead = Arc::new(Mutex::new(Some(PendingDeliveryRead {
            current: "old".to_string(),
            handle,
        })));

        let delivered =
            read_delivered_with_timeout(&pending, "unused", "new", Duration::from_secs(1)).await;

        assert!(delivered.is_none());
        assert!(pending.lock().await.is_none());
    }

    /// Re-delivering the same token must not look like a new credential.
    #[tokio::test]
    async fn an_unchanged_token_file_is_not_readopted() {
        let tmp = tempfile::tempdir().unwrap();
        let gw = spawn_gateway(vec![], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(60)).to_rfc3339();
        let path = write_token_file(tmp.path(), "same", &expires);

        let auth = EndUserAuth::try_new(&gw.base_url, "same", Some(&expires), true)
            .unwrap()
            .with_token_file(&path)
            .supervised();

        // Inside the rotation window, so rotate_into runs and reads the file.
        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "same");
        assert_eq!(gw.hits.load(Ordering::SeqCst), 0);
    }

    /// Each credential watches its own path, so a host running several agents
    /// cannot cross their tokens.
    #[tokio::test]
    async fn each_credential_watches_its_own_file() {
        let tmp = tempfile::tempdir().unwrap();
        let gw = spawn_gateway(vec![], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(60)).to_rfc3339();

        let dir_a = tmp.path().join("a");
        let dir_b = tmp.path().join("b");
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();

        let path_a = write_token_file(&dir_a, "agent-a-v2", &expires);
        let path_b = write_token_file(&dir_b, "agent-b-v2", &expires);

        let a = EndUserAuth::try_new(&gw.base_url, "agent-a", Some(&expires), true)
            .unwrap()
            .with_token_file(&path_a)
            .supervised();
        let b = EndUserAuth::try_new(&gw.base_url, "agent-b", Some(&expires), true)
            .unwrap()
            .with_token_file(&path_b)
            .supervised();

        assert_eq!(with_deadline(a.get_token()).await.unwrap(), "agent-a-v2");
        assert_eq!(with_deadline(b.get_token()).await.unwrap(), "agent-b-v2");
    }

    /// The chain's absolute cap cannot be extended by rotation, so a terminal
    /// credential is precisely the one a control plane needs to replace. If
    /// `replace_token` could not revive it, the delivery channel would be
    /// useless in the case it exists for.
    #[tokio::test]
    async fn replacing_a_dead_credential_revives_it() {
        let gw = spawn_gateway(vec![(401, r#"{"msg":"chain expired"}"#.into())], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(30)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "spent", Some(&expires), true).unwrap();

        assert!(with_deadline(auth.get_token()).await.is_err());
        assert!(auth.is_terminal(), "a 401 latches dead");

        let fresh = (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339();
        auth.replace_token("minted", Some(&fresh)).await;

        assert!(!auth.is_terminal(), "a replacement clears terminal state");
        assert!(Auth::is_authenticated(&auth));
        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "minted");
        assert_eq!(
            gw.hits.load(Ordering::SeqCst),
            1,
            "the replacement is served without another rotation"
        );
    }

    /// A replacement also clears rotation backoff — it is a different
    /// credential, not a retry of the one that was failing.
    #[tokio::test]
    async fn replacing_clears_rotation_backoff() {
        let gw = spawn_gateway(vec![(503, "down".into())], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(60)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "original", Some(&expires), true).unwrap();

        // Fail once to arm the backoff.
        let _ = with_deadline(auth.get_token()).await;

        let fresh = (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339();
        auth.replace_token("minted", Some(&fresh)).await;

        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "minted");
    }

    /// Each credential owns its own state, so a process hosting several agents
    /// cannot cross their tokens — the reason this is a method and not an
    /// environment variable.
    #[tokio::test]
    async fn replacing_one_credential_does_not_touch_another() {
        let gw = spawn_gateway(vec![], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339();
        let a = EndUserAuth::try_new(&gw.base_url, "agent-a", Some(&expires), true).unwrap();
        let b = EndUserAuth::try_new(&gw.base_url, "agent-b", Some(&expires), true).unwrap();

        a.replace_token("agent-a-v2", Some(&expires)).await;

        assert_eq!(with_deadline(a.get_token()).await.unwrap(), "agent-a-v2");
        assert_eq!(with_deadline(b.get_token()).await.unwrap(), "agent-b");
    }

    /// Supervised mode holds the credential without rotating it, but it is still
    /// an end-user credential — a plain static holder would report `ServiceUser`
    /// and route this token to the wrong surface.
    #[tokio::test]
    async fn a_supervised_credential_still_reports_its_kind_and_serves_its_token() {
        let gw = spawn_gateway(vec![], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "held", Some(&expires), true)
            .unwrap()
            .supervised();

        assert_eq!(Auth::kind(&auth), crate::models::CredentialKind::EndUser);
        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "held");
        assert_eq!(
            gw.hits.load(Ordering::SeqCst),
            0,
            "supervised mode must never call the refresh route"
        );
    }

    /// Inside the rotation window a supervised credential still serves its token
    /// rather than rotating — the control plane owns that.
    #[tokio::test]
    async fn a_supervised_credential_does_not_rotate_near_expiry() {
        let gw = spawn_gateway(vec![], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(60)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "held", Some(&expires), true)
            .unwrap()
            .supervised();

        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "held");
        assert_eq!(gw.hits.load(Ordering::SeqCst), 0);
    }

    /// When the supervisor stops redelivering, say so plainly. A dumb static
    /// holder would keep reporting healthy while every call 401s.
    #[tokio::test]
    async fn a_supervised_credential_reports_a_stale_token_instead_of_serving_it() {
        let gw = spawn_gateway(vec![], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(1)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "stale", Some(&expires), true)
            .unwrap()
            .supervised();

        let err = with_deadline(auth.get_token())
            .await
            .expect_err("an unusable supervised token must not be served");
        let msg = err.to_string();
        assert!(msg.contains("control plane"), "name who must act: {msg}");
        assert_eq!(gw.hits.load(Ordering::SeqCst), 0);
    }

    /// A supervised token is barred from self-refresh, so rotating it yields a
    /// 403 — terminal. Catch it at boot rather than an hour in.
    #[tokio::test]
    async fn a_supervised_token_is_rejected_at_construction() {
        let gw = spawn_gateway(vec![], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339();
        let token = jwt_with(serde_json::json!({ "sub": "u1", "aud": "supervised" }));

        let auth = EndUserAuth::try_new(&gw.base_url, token, Some(&expires), true).unwrap();
        let err = auth
            .ensure_self_refreshable()
            .expect_err("a supervised token must not pass with rotation enabled");

        let msg = err.to_string();
        assert!(
            msg.contains("rotate_token"),
            "error should name the fix: {msg}"
        );
    }

    #[tokio::test]
    async fn a_self_refreshable_token_passes_construction() {
        let gw = spawn_gateway(vec![], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339();

        // Array form of `aud`, and no audience at all, must both be accepted.
        for token in [
            jwt_with(serde_json::json!({ "sub": "u1", "aud": ["end-user"] })),
            jwt_with(serde_json::json!({ "sub": "u1" })),
            "not-a-jwt".to_string(),
        ] {
            let auth = EndUserAuth::try_new(&gw.base_url, token, Some(&expires), true).unwrap();
            assert!(auth.ensure_self_refreshable().is_ok());
        }
    }

    /// 400 means the request was malformed — an over-cap `token_ttl_seconds`,
    /// say. The credential may be fine, so latching Dead would turn a config
    /// typo into an agent that never recovers.
    #[tokio::test]
    async fn a_bad_request_is_not_terminal() {
        let gw = spawn_gateway(vec![(400, r#"{"msg":"ttl exceeds cap"}"#.into())], None).await;
        // Inside ROTATE_BEFORE but above MIN_USABLE: rotation is attempted, and
        // when it fails the still-usable token is served.
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(60)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "original", Some(&expires), true).unwrap();

        let served = with_deadline(auth.get_token()).await;
        assert_eq!(gw.hits.load(Ordering::SeqCst), 1, "rotation was attempted");
        assert_eq!(
            served.expect("a usable token is still served"),
            "original",
            "a failed rotation falls back to the live token"
        );
        assert!(
            !auth.is_terminal(),
            "a request-shape error is not a verdict on the credential"
        );
    }

    /// Backoff outlasting the token is unrecoverable: the server revokes on
    /// expiry, so waiting cannot help. Without latching, the reconnect loop
    /// spins forever holding the runtime open — the exact zombie `is_terminal`
    /// exists to prevent, reached by the one path that never set it.
    #[tokio::test]
    async fn a_token_that_expires_during_backoff_latches_dead() {
        // 503 → Unavailable → backoff, not terminal.
        let gw = spawn_gateway(vec![(503, "down".into()), (503, "down".into())], None).await;
        // Already unusable: below MIN_USABLE.
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(1)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "original", Some(&expires), true).unwrap();

        // First call fails and arms the backoff.
        assert!(with_deadline(auth.get_token()).await.is_err());
        assert!(!auth.is_terminal(), "a 503 alone is not terminal");

        // Second call lands inside the backoff with an unusable token.
        assert!(with_deadline(auth.get_token()).await.is_err());
        assert!(
            auth.is_terminal(),
            "an expired token that cannot be rotated must stop the runtime"
        );
    }

    /// The refresh route allows 60/hour and counts rejected attempts, so a 60s
    /// ceiling would spend the entire budget on retries.
    #[test]
    fn backoff_leaves_rate_limit_headroom() {
        let max = backoff_delay(u32::MAX);
        assert!(
            max > Duration::from_secs(60),
            "retrying once a minute consumes the whole 60/hour budget: {max:?}"
        );
    }

    #[tokio::test]
    async fn revoked_chain_is_terminal() {
        let gw = spawn_gateway(vec![(401, r#"{"msg":"invalid token"}"#.into())], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(30)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "original", Some(&expires), true).unwrap();

        assert!(with_deadline(auth.get_token()).await.is_err());
        assert!(auth.is_terminal());
    }

    /// A revocation discovered by some *other* caller's 401 must latch too.
    /// Rotation is not the only way a credential dies, and until this landed a
    /// revoked chain reported healthy while every request failed.
    #[tokio::test]
    async fn a_rejection_seen_elsewhere_latches_dead() {
        let gw = spawn_gateway(vec![], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "live-token", Some(&expires), true).unwrap();

        assert!(!auth.is_terminal(), "starts healthy");
        assert!(Auth::is_authenticated(&auth));

        // A 401 from a persona/episode/memory call, not from rotation.
        Auth::note_rejection(&auth);

        assert!(
            auth.is_terminal(),
            "a rejection must be visible to supervisors"
        );
        assert!(!Auth::is_authenticated(&auth));
        assert_eq!(
            gw.hits.load(Ordering::SeqCst),
            0,
            "latching must not spend a rotation from the rate-limit budget"
        );
    }

    /// The credential carries its own routing kind, so a caller cannot pair it
    /// with a contradicting one.
    #[tokio::test]
    async fn an_end_user_credential_reports_its_kind() {
        let gw = spawn_gateway(vec![], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "t", Some(&expires), true).unwrap();
        assert_eq!(Auth::kind(&auth), crate::models::CredentialKind::EndUser);
    }

    /// 429 means the chain is intact — back off, do not latch dead. Latching
    /// here would kill an agent for exceeding a per-hour quota.
    #[tokio::test]
    async fn rate_limit_is_not_terminal() {
        let gw = spawn_gateway(vec![(429, r#"{"msg":"slow down"}"#.into())], None).await;
        // Still usable (60s > MIN_USABLE) but inside the rotation window.
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(60)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "original", Some(&expires), true).unwrap();

        // Rotation fails, but the current token still has life — serve it.
        let token = with_deadline(auth.get_token()).await.unwrap();
        assert_eq!(token, "original");
        assert!(!auth.is_terminal(), "429 must not latch dead");
        assert!(auth.is_authenticated());
    }

    /// A 5xx is transient: the server never evaluated the token, so the
    /// credential is still presumed good.
    #[tokio::test]
    async fn server_error_is_not_terminal() {
        let gw = spawn_gateway(vec![(503, "upstream down".into())], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(60)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "original", Some(&expires), true).unwrap();

        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "original");
        assert!(!auth.is_terminal());
    }

    /// Backoff must not strand a caller that still holds a usable token.
    #[tokio::test]
    async fn backoff_still_serves_a_usable_token() {
        let gw = spawn_gateway(vec![(503, "down".into()), (503, "down".into())], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(60)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "original", Some(&expires), true).unwrap();

        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "original");
        // Second call lands inside backoff; the token is still good, so serve it
        // without another round-trip.
        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "original");
        assert_eq!(gw.hits.load(Ordering::SeqCst), 1, "backoff should suppress");
    }

    /// A 2xx we cannot parse means the successor is lost and the presented
    /// token is already revoked server-side — terminal, not retryable.
    #[tokio::test]
    async fn unparseable_success_is_terminal() {
        let gw = spawn_gateway(vec![(200, "<html>proxy error</html>".into())], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(30)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "original", Some(&expires), true).unwrap();

        assert!(with_deadline(auth.get_token()).await.is_err());
        assert!(
            auth.is_terminal(),
            "a lost successor cannot be recovered by retrying"
        );
    }

    /// No `expires_at` means rotate on first use rather than assuming validity.
    #[tokio::test]
    async fn unknown_expiry_rotates_immediately() {
        let gw = spawn_gateway(vec![(200, token_body("rotated", 3600))], None).await;
        let auth = EndUserAuth::try_new(&gw.base_url, "original", None, true).unwrap();

        assert_eq!(with_deadline(auth.get_token()).await.unwrap(), "rotated");
        assert_eq!(gw.hits.load(Ordering::SeqCst), 1);
    }

    /// Plaintext carries a live credential; refuse unless explicitly opted in.
    #[test]
    fn plaintext_requires_opt_in() {
        assert!(EndUserAuth::try_new("http://gw.local", "t", None, false).is_err());
        assert!(EndUserAuth::try_new("http://gw.local", "t", None, true).is_ok());
        assert!(EndUserAuth::try_new("https://gw.local", "t", None, false).is_ok());
    }

    /// The hardened HTTP client builds for a valid base_url, and any rejected
    /// base_url surfaces as a typed, propagated error rather than a panic or a
    /// silently-weakened client. (`ClientBuilder::build()` itself has no
    /// deterministic failure trigger from safe, cross-platform test code, so
    /// this only exercises the error path reachable via `require_secure_url`;
    /// that path and the `builder.build()` path both return through the same
    /// `?` in `try_new`.)
    #[test]
    fn try_new_is_typed_ok_or_err() {
        let ok = EndUserAuth::try_new("https://gw.local", "t", None, false);
        assert!(ok.is_ok());

        match EndUserAuth::try_new("not a url", "t", None, true) {
            Err(MindroidError::Config { .. }) => {}
            Err(other) => panic!("expected a typed Config error, got {other}"),
            Ok(_) => panic!("expected an error for an unparseable base_url"),
        }
    }

    /// Headers carry the rotated token, not the original.
    #[tokio::test]
    async fn auth_headers_use_the_current_token() {
        let gw = spawn_gateway(vec![(200, token_body("rotated", 3600))], None).await;
        let expires = (chrono::Utc::now() + chrono::Duration::seconds(30)).to_rfc3339();
        let auth = EndUserAuth::try_new(&gw.base_url, "original", Some(&expires), true).unwrap();

        let headers = with_deadline(auth.get_auth_headers()).await.unwrap();
        assert_eq!(headers[0].1, "Bearer rotated");
    }

    #[test]
    fn backoff_grows_and_caps() {
        assert_eq!(backoff_delay(1), Duration::from_secs(1));
        assert_eq!(backoff_delay(3), Duration::from_secs(4));
        assert_eq!(backoff_delay(20), BACKOFF_MAX);
    }

    /// Expiry is derived as a duration off the monotonic clock, so wall-clock
    /// skew between the agent host and the server cannot make a good token look
    /// expired.
    #[test]
    fn expires_at_parses_as_a_relative_duration() {
        let now = Instant::now();
        let future = (chrono::Utc::now() + chrono::Duration::seconds(600)).to_rfc3339();
        let parsed = parse_expires_at(&future, now).expect("should parse");
        let remaining = parsed.saturating_duration_since(now).as_secs();
        assert!((595..=600).contains(&remaining), "got {remaining}s");

        let past = (chrono::Utc::now() - chrono::Duration::seconds(60)).to_rfc3339();
        assert!(parse_expires_at(&past, now).is_none());
        assert!(parse_expires_at("not-a-date", now).is_none());
    }
}
