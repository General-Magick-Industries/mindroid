#[cfg(feature = "apikey")]
pub mod apikey;
pub mod static_id;

use async_trait::async_trait;
use std::sync::Arc;

use crate::error::Result;

#[async_trait]
pub trait Auth: Send + Sync + 'static {
    async fn get_token(&self) -> Result<String>;

    async fn get_auth_headers(&self) -> Result<Vec<(String, String)>>;

    fn is_authenticated(&self) -> bool;

    async fn refresh(&self) -> Result<()>;
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
