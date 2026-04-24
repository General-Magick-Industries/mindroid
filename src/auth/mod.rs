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

#[cfg(feature = "http-client")]
pub use crate::http::build_auth_header_map;
