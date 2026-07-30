#[cfg(feature = "apikey")]
pub mod apikey;
pub mod static_id;

use async_trait::async_trait;
use std::sync::Arc;

use crate::error::Result;

/// The kind of identity a credential authenticates as. This is the single axis
/// the magickmind/bifrost API surface splits on: a service user acts on behalf
/// of its tenant; an end user (e.g. an agent's own JWT) acts as itself. Route
/// selection and Centrifugo token placement are both derived from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CredentialKind {
    /// A service-user credential (Keycloak login). Targets the tenant-scoped
    /// `/v1/...` routes and is JWKS-verified on Centrifugo connect. Default.
    #[default]
    ServiceUser,
    /// An end-user credential (e.g. an agent's own bifrost JWT). Targets the
    /// `/v1/end-user/...` routes and is validated by the Centrifugo connect
    /// proxy rather than the JWKS gate.
    EndUser,
}

/// Where a credential's token belongs in a Centrifugo connect frame. Derived
/// from [`CredentialKind`]: a service user's token is JWKS-verified in the
/// top-level `token`; an end user's token rides in `data`, routed to the
/// bifrost connect proxy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenPlacement {
    /// Top-level `connect.token` — verified by Centrifugo (JWKS/HMAC). Default.
    #[default]
    Token,
    /// `connect.data.token` — forwarded to the bifrost connect proxy.
    Data,
}

#[async_trait]
pub trait Auth: Send + Sync + 'static {
    async fn get_token(&self) -> Result<String>;

    async fn get_auth_headers(&self) -> Result<Vec<(String, String)>>;

    fn is_authenticated(&self) -> bool;

    async fn refresh(&self) -> Result<()>;

    /// The identity kind this credential authenticates as. Defaults to
    /// [`CredentialKind::ServiceUser`]; end-user credentials override it.
    fn credential_kind(&self) -> CredentialKind {
        CredentialKind::ServiceUser
    }

    /// Which Centrifugo connect-frame field the token belongs in, derived from
    /// [`Auth::credential_kind`].
    fn connect_token_placement(&self) -> TokenPlacement {
        match self.credential_kind() {
            CredentialKind::EndUser => TokenPlacement::Data,
            CredentialKind::ServiceUser => TokenPlacement::Token,
        }
    }
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

    fn credential_kind(&self) -> CredentialKind {
        (**self).credential_kind()
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
