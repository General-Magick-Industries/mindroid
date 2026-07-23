use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::error::{MindroidError, Result};

/// Build an HTTP client for a credential-bearing JSON API.
///
/// Redirects are refused: reqwest strips `Authorization` across hosts but
/// compares host and port without the scheme, so a same-host `https -> http`
/// redirect would forward the bearer token in cleartext. These endpoints have
/// no reason to redirect, so failing loudly is correct.
pub(crate) fn secure_json_client(timeout: Duration) -> reqwest::Client {
    let builder = reqwest::Client::builder()
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none());

    // reqwest honours HTTP_PROXY/ALL_PROXY and does not exempt loopback, so a
    // proxied dev machine or egress-controlled CI runner would route a test's
    // ephemeral-port request into the void.
    #[cfg(test)]
    let builder = builder.no_proxy();

    builder.build().expect("failed to build HTTP client")
}

/// Cache key: the scope being prepared, plus the user it is personalized for.
///
/// The user component is `None` for non-user senders, whose prompt is the
/// generic, non-personalized one. Two real users never share that slot.
pub(crate) type PromptCacheKey = (String, Option<String>);

struct CacheEntry {
    prompt: String,
    fetched_at: Instant,
}

/// TTL cache of server-prepared prompts, with stale-serve degradation.
///
/// Expired entries are kept rather than evicted on read, so a failed re-fetch
/// can fall back to the last-good prompt instead of dropping the message. A
/// zero TTL disables caching entirely: nothing is stored, so there is no stale
/// fallback and a fetch failure propagates.
pub(crate) struct PreparedPromptCache {
    ttl: Duration,
    entries: Mutex<HashMap<PromptCacheKey, CacheEntry>>,
}

impl PreparedPromptCache {
    /// Evict expired entries once the map grows beyond this size.
    const SWEEP_THRESHOLD: usize = 200;

    pub(crate) fn new(ttl: Duration) -> Self {
        Self {
            ttl,
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// The configured TTL. Only read by tests asserting the default and the
    /// `with_ttl` override; the cache itself uses `self.ttl` directly.
    #[cfg(test)]
    pub(crate) fn ttl(&self) -> Duration {
        self.ttl
    }

    pub(crate) fn set_ttl(&mut self, ttl: Duration) {
        self.ttl = ttl;
    }

    /// A cached prompt that is still within TTL. Never evicts — an expired
    /// entry stays available to [`Self::get_any`] as a stale fallback.
    pub(crate) fn get_fresh(&self, key: &PromptCacheKey) -> Option<String> {
        let entries = self.entries.lock().expect("prompt cache mutex poisoned");
        entries
            .get(key)
            .filter(|e| e.fetched_at.elapsed() < self.ttl)
            .map(|e| e.prompt.clone())
    }

    /// A cached prompt regardless of age (the stale-serve fallback).
    pub(crate) fn get_any(&self, key: &PromptCacheKey) -> Option<String> {
        let entries = self.entries.lock().expect("prompt cache mutex poisoned");
        entries.get(key).map(|e| e.prompt.clone())
    }

    /// Store a prompt. No-op when caching is disabled (zero TTL); sweeps
    /// expired entries once the map grows past [`Self::SWEEP_THRESHOLD`].
    pub(crate) fn insert(&self, key: PromptCacheKey, prompt: String) {
        if self.ttl.is_zero() {
            return;
        }

        let mut entries = self.entries.lock().expect("prompt cache mutex poisoned");
        if entries.len() > Self::SWEEP_THRESHOLD {
            let now = Instant::now();
            entries.retain(|_, e| now.duration_since(e.fetched_at) < self.ttl);
        }
        entries.insert(
            key,
            CacheEntry {
                prompt,
                fetched_at: Instant::now(),
            },
        );
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries
            .lock()
            .expect("prompt cache mutex poisoned")
            .len()
    }
}

/// Reject a base URL that would carry auth headers over a non-TLS scheme.
///
/// Auth headers ride on every request to these services, so a plaintext URL
/// leaks the credential to anyone on the path. The check parses the URL and
/// compares the scheme rather than matching a string prefix: URL schemes are
/// case-insensitive, so `"HTTP://host"` normalizes to `http` while a
/// `starts_with("http://")` test would pass it through. Allow-listing `https`
/// also rejects `ftp`, `file`, and anything else that is not TLS.
///
/// `knob` names the config field that opts out, so the error tells the operator
/// exactly what to set (e.g. `"persona.allow_insecure"`).
pub(crate) fn require_secure_url(base_url: &str, allow_insecure: bool, knob: &str) -> Result<()> {
    let url = reqwest::Url::parse(base_url)
        .map_err(|e| MindroidError::config(format!("invalid base_url {base_url}: {e}")))?;

    if url.scheme() != "https" && !allow_insecure {
        return Err(MindroidError::config(format!(
            "base_url {base_url} uses the non-TLS scheme {}:// — use https://, \
             or set {knob} = true for local development",
            url.scheme()
        )));
    }
    Ok(())
}

/// Longest error-body excerpt kept for diagnostics.
const MAX_ERR_BODY: usize = 512;

/// Reduce a server error body to a bounded, single-line excerpt.
///
/// Error bodies are server-controlled and unbounded, and they land in `warn!`
/// output — often once per message. Left raw, a hostile or misbehaving endpoint
/// can flood the log, and control characters can forge extra log lines in
/// line-oriented sinks. Strips control characters (including newlines) and caps
/// the length.
pub(crate) fn error_excerpt(body: &str) -> String {
    let mut out: String = body
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .take(MAX_ERR_BODY)
        .collect();
    if body.chars().count() > MAX_ERR_BODY {
        out.push_str("... (truncated)");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn https_is_allowed() {
        assert!(require_secure_url("https://x.test", false, "k").is_ok());
    }

    #[test]
    fn plaintext_http_is_rejected() {
        let err = require_secure_url("http://x.test", false, "episodes.allow_insecure")
            .unwrap_err()
            .to_string();
        assert!(err.contains("http://"), "got: {err}");
        assert!(err.contains("episodes.allow_insecure"), "got: {err}");
    }

    /// The bug this helper exists to fix: schemes are case-insensitive, so a
    /// prefix test on the raw string lets `HTTP://` through.
    #[test]
    fn uppercase_scheme_is_rejected() {
        assert!(!"HTTP://x.test".starts_with("http://"));
        assert!(require_secure_url("HTTP://x.test", false, "k").is_err());
        assert!(require_secure_url("Http://x.test", false, "k").is_err());
    }

    #[test]
    fn non_tls_schemes_are_rejected() {
        assert!(require_secure_url("ftp://x.test", false, "k").is_err());
        assert!(require_secure_url("file:///tmp/x", false, "k").is_err());
    }

    #[test]
    fn allow_insecure_opts_out() {
        assert!(require_secure_url("http://x.test", true, "k").is_ok());
        assert!(require_secure_url("HTTP://x.test", true, "k").is_ok());
    }

    #[test]
    fn unparseable_url_is_rejected() {
        assert!(require_secure_url("not a url", false, "k").is_err());
    }

    #[test]
    fn short_body_passes_through() {
        assert_eq!(error_excerpt("not found"), "not found");
    }

    #[test]
    fn control_chars_are_neutralized() {
        // Newlines would otherwise forge extra lines in a log sink.
        assert_eq!(error_excerpt("a\nb\tc\r\nd"), "a b c  d");
    }

    #[test]
    fn long_body_is_truncated() {
        let out = error_excerpt(&"x".repeat(MAX_ERR_BODY + 100));
        assert!(out.ends_with("... (truncated)"));
        assert_eq!(out.chars().filter(|c| *c == 'x').count(), MAX_ERR_BODY);
    }

    // -- PreparedPromptCache -------------------------------------------------
    //
    // These cover both prepare stages: each owns a PreparedPromptCache, so a
    // regression here surfaces once rather than needing duplicate suites.

    fn key(scope: &str, user: Option<&str>) -> PromptCacheKey {
        (scope.to_string(), user.map(str::to_string))
    }

    #[test]
    fn fresh_entry_is_served() {
        let c = PreparedPromptCache::new(Duration::from_secs(60));
        c.insert(key("a1", Some("u1")), "prompt".into());
        assert_eq!(
            c.get_fresh(&key("a1", Some("u1"))).as_deref(),
            Some("prompt")
        );
    }

    #[test]
    fn expired_entry_is_not_fresh_but_serves_stale() {
        let c = PreparedPromptCache::new(Duration::from_millis(1));
        c.insert(key("a1", None), "prompt".into());
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(c.get_fresh(&key("a1", None)), None);
        assert_eq!(c.get_any(&key("a1", None)).as_deref(), Some("prompt"));
    }

    #[test]
    fn zero_ttl_never_inserts() {
        let c = PreparedPromptCache::new(Duration::ZERO);
        c.insert(key("a1", Some("u1")), "prompt".into());
        assert_eq!(c.len(), 0);
        // Nothing stored means no stale fallback, so a failure propagates.
        assert_eq!(c.get_any(&key("a1", Some("u1"))), None);
    }

    #[test]
    fn insert_sweeps_expired_entries_beyond_threshold() {
        let c = PreparedPromptCache::new(Duration::from_millis(1));
        for i in 0..=PreparedPromptCache::SWEEP_THRESHOLD {
            c.insert(key(&format!("a{i}"), None), "prompt".into());
        }
        std::thread::sleep(Duration::from_millis(5));
        c.insert(key("fresh", None), "prompt".into());
        assert_eq!(c.len(), 1, "expired entries should have been swept");
    }

    #[test]
    fn distinct_users_do_not_share_an_entry() {
        let c = PreparedPromptCache::new(Duration::from_secs(60));
        c.insert(key("a1", Some("alice")), "for alice".into());
        c.insert(key("a1", Some("bob")), "for bob".into());
        assert_eq!(
            c.get_fresh(&key("a1", Some("alice"))).as_deref(),
            Some("for alice")
        );
        assert_eq!(
            c.get_fresh(&key("a1", Some("bob"))).as_deref(),
            Some("for bob")
        );
        // The non-personalized slot is separate from both.
        assert_eq!(c.get_fresh(&key("a1", None)), None);
    }

    #[test]
    fn distinct_scopes_do_not_share_an_entry() {
        let c = PreparedPromptCache::new(Duration::from_secs(60));
        c.insert(key("agent-1", None), "one".into());
        c.insert(key("agent-2", None), "two".into());
        assert_eq!(c.get_fresh(&key("agent-1", None)).as_deref(), Some("one"));
        assert_eq!(c.get_fresh(&key("agent-2", None)).as_deref(), Some("two"));
    }
}
