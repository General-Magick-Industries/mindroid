use crate::error::{MindroidError, Result};

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
}
