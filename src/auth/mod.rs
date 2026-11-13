#[cfg(feature = "apikey")]
pub mod apikey;
#[cfg(feature = "magickmind")]
pub mod enduser;
pub mod static_id;

use async_trait::async_trait;
use std::sync::Arc;

use crate::error::Result;
use crate::models::CredentialKind;

#[async_trait]
pub trait Auth: Send + Sync + 'static {
    async fn get_token(&self) -> Result<String>;

    async fn get_auth_headers(&self) -> Result<Vec<(String, String)>>;

    fn is_authenticated(&self) -> bool;

    async fn refresh(&self) -> Result<()>;

    /// Whether the credential is permanently dead — waiting cannot help.
    ///
    /// Lets a caller distinguish "retry" from "stop". A reconnect loop that
    /// cannot tell the difference retries a rejected credential forever, which
    /// looks alive to a supervisor while every request 401s.
    ///
    /// Defaults to `false`: a credential with no notion of terminal failure
    /// (a static token, an API key) is never permanently dead in this sense.
    fn is_terminal(&self) -> bool {
        false
    }

    /// Which identity this credential acts as.
    ///
    /// Adapters pick service-user vs end-user routes from this. It belongs to
    /// the credential rather than to config: taking the two from different
    /// sources lets an injected end-user token be presented to service-user
    /// surfaces, which fails at the server as an opaque 401.
    ///
    /// Defaults to [`CredentialKind::ServiceUser`].
    fn kind(&self) -> CredentialKind {
        CredentialKind::ServiceUser
    }

    /// Record that a request bearing this credential was rejected (401/403).
    ///
    /// Rotation classifies its own failures, but a rejection discovered by any
    /// *other* caller never reaches the credential — so a server-side revocation
    /// leaves [`is_terminal`](Self::is_terminal) false and
    /// [`is_authenticated`](Self::is_authenticated) true while every request
    /// fails. Call this on a 401 to let the credential latch that state without
    /// spending a rotation from its rate-limit budget.
    ///
    /// Default: no-op, for credentials with no failure state to track.
    fn note_rejection(&self) {}
}

#[async_trait]
impl<T: Auth> Auth for Arc<T> {
    async fn get_token(&self) -> Result<String> {
        (**self).get_token().await
    }

    async fn get_auth_headers(&self) -> Result<Vec<(String, String)>> {
        (**self).get_auth_headers().await
    }

    fn is_authenticated(&self) -> bool {
        (**self).is_authenticated()
    }

    async fn refresh(&self) -> Result<()> {
        (**self).refresh().await
    }

    fn is_terminal(&self) -> bool {
        (**self).is_terminal()
    }

    fn kind(&self) -> CredentialKind {
        (**self).kind()
    }

    fn note_rejection(&self) {
        (**self).note_rejection()
    }
}

#[cfg(any(
    feature = "apikey",
    feature = "persistence",
    feature = "llm-hosted",
    feature = "persona"
))]
pub async fn build_auth_header_map(auth: &dyn Auth) -> crate::Result<reqwest::header::HeaderMap> {
    use crate::error::MindroidError;
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
    let headers = auth.get_auth_headers().await?;
    let mut map = HeaderMap::new();
    for (k, v) in headers {
        // Dropping a malformed auth header would send the request
        // unauthenticated and surface as a confusing server 401. Fail here
        // instead, where the cause is knowable. The value is never included in
        // the error — it is the credential.
        let name = HeaderName::from_bytes(k.as_bytes()).map_err(|e| MindroidError::Auth {
            message: format!("auth header name {k:?} is not a valid HTTP header name: {e}"),
            source: None,
        })?;
        let value = HeaderValue::from_str(&v).map_err(|_| MindroidError::Auth {
            message: format!(
                "auth header {k:?} has a value that is not valid for an HTTP header \
                 (check the configured token for stray whitespace or non-ASCII bytes)"
            ),
            source: None,
        })?;
        map.insert(name, value);
    }
    Ok(map)
}
