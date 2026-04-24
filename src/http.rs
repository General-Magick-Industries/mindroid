//! Shared authenticated HTTP client for Magick Mind API services.
//!
//! Eliminates duplicated `reqwest` + `Auth` + error-mapping boilerplate
//! across `MagickmindClient`, `CorpusClient`, `MagickmindMemory`, and
//! `MagickmindPersonaClient`.
//!
//! # Example
//!
//! ```ignore
//! use mindroid::http::AuthenticatedHttpClient;
//!
//! let client = AuthenticatedHttpClient::new("https://api.example.com", identity)
//!     .with_api_key("sk-...");
//!
//! let resp: MyResponse = client.post_json("/v1/resource", &my_request).await?;
//! ```

use std::sync::Arc;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::auth::Auth;
use crate::error::{MindroidError, Result};

/// Authenticated HTTP client that handles auth headers, API key injection,
/// status checking, and error mapping.
///
/// Domain-specific clients (e.g. `MagickmindClient`, `CorpusClient`) compose
/// over this rather than reimplementing the same boilerplate.
pub struct AuthenticatedHttpClient {
    http: reqwest::Client,
    base_url: String,
    identity: Arc<dyn Auth>,
    api_key: Option<String>,
}

impl AuthenticatedHttpClient {
    /// Create a new client with auth identity.
    pub fn new(base_url: impl Into<String>, identity: Arc<dyn Auth>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            identity,
            api_key: None,
        }
    }

    /// Set an optional API key sent as `x-api-key` header.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// The base URL this client targets.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// POST JSON and deserialize the response.
    pub async fn post_json<Req: Serialize, Resp: DeserializeOwned>(
        &self,
        path: &str,
        body: &Req,
    ) -> Result<Resp> {
        let url = format!("{}{}", self.base_url, path);
        let headers = self.build_headers().await?;

        let resp = self
            .http
            .post(&url)
            .headers(headers)
            .json(body)
            .send()
            .await
            .map_err(|e| MindroidError::Api {
                message: format!("POST {url} failed: {e}"),
                status_code: e.status().map(|s| s.as_u16()),
            })?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(MindroidError::Api {
                message: format!("POST {url} returned {status}: {text}"),
                status_code: Some(status.as_u16()),
            });
        }

        resp.json().await.map_err(|e| MindroidError::Api {
            message: format!("Failed to parse response from POST {url}: {e}"),
            status_code: None,
        })
    }

    /// POST JSON and return the raw response text.
    ///
    /// Useful when you need the raw body for debug logging before parsing.
    pub async fn post_json_text<Req: Serialize>(
        &self,
        path: &str,
        body: &Req,
    ) -> Result<String> {
        let url = format!("{}{}", self.base_url, path);
        let headers = self.build_headers().await?;

        let resp = self
            .http
            .post(&url)
            .headers(headers)
            .json(body)
            .send()
            .await
            .map_err(|e| MindroidError::Api {
                message: format!("POST {url} failed: {e}"),
                status_code: e.status().map(|s| s.as_u16()),
            })?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(MindroidError::Api {
                message: format!("POST {url} returned {status}: {text}"),
                status_code: Some(status.as_u16()),
            });
        }

        resp.text().await.map_err(|e| MindroidError::Api {
            message: format!("Failed to read response body from POST {url}: {e}"),
            status_code: None,
        })
    }

    /// Return a `RequestBuilder` with auth headers pre-attached.
    ///
    /// Use this for requests that need custom URL building, query parameters,
    /// or non-standard response handling.
    pub async fn request(
        &self,
        method: reqwest::Method,
        url: reqwest::Url,
    ) -> Result<reqwest::RequestBuilder> {
        let headers = self.build_headers().await?;
        Ok(self.http.request(method, url).headers(headers))
    }

    /// Send a request and check for success status.
    ///
    /// Returns the response for further processing (`.json()`, `.text()`, etc.).
    pub async fn send_and_check(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<reqwest::Response> {
        let resp = request.send().await.map_err(|e| MindroidError::Api {
            message: format!("HTTP request failed: {e}"),
            status_code: e.status().map(|s| s.as_u16()),
        })?;

        let status = resp.status();
        if !status.is_success() {
            let url = resp.url().to_string();
            let text = resp.text().await.unwrap_or_default();
            return Err(MindroidError::Api {
                message: format!("{url} returned {status}: {text}"),
                status_code: Some(status.as_u16()),
            });
        }

        Ok(resp)
    }

    /// Build auth headers including optional x-api-key.
    async fn build_headers(&self) -> Result<HeaderMap> {
        let mut map = build_auth_header_map(self.identity.as_ref()).await?;

        if let Some(ref key) = self.api_key {
            if let Ok(value) = HeaderValue::from_str(key) {
                map.insert(
                    HeaderName::from_static("x-api-key"),
                    value,
                );
            }
        }

        Ok(map)
    }
}

impl std::fmt::Debug for AuthenticatedHttpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthenticatedHttpClient")
            .field("base_url", &self.base_url)
            .field("has_api_key", &self.api_key.is_some())
            .finish()
    }
}

/// Build a `HeaderMap` from an [`Auth`] provider's key-value pairs.
pub async fn build_auth_header_map(auth: &dyn Auth) -> Result<HeaderMap> {
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
