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
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
    let headers = auth.get_auth_headers().await?;
    let mut map = HeaderMap::new();
    for (k, v) in headers {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(k.as_bytes()),
            HeaderValue::from_str(&v),
        ) {
            map.insert(name, value);
        }
    }
    Ok(map)
}
