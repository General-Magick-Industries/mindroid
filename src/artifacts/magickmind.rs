//! [`MagickmindArtifactStore`]: an [`ArtifactStore`] backed by Magick Mind's
//! artifact service, reached through Bifrost.
//!
//! Bytes move through presigned S3 URLs. The scope is a magickspace id (see
//! [`ArtifactStore::scope_for`]); what Bifrost checks per call is in ADR-0009.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use reqwest::{Method, RequestBuilder, Response, StatusCode};
use serde::{Deserialize, Serialize};

use super::{Artifact, ArtifactStore, StoredArtifact};
use crate::core::content::is_artifact_id;
use crate::models::{CredentialKind, Message};
use crate::{Auth, MindroidError, Result};

const MAX_DOWNLOAD_BYTES: usize = 64 * 1024 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// Stores artifacts in a magickspace through Bifrost's artifact routes.
///
/// A service-user credential uses the tenant routes; an end-user credential uses
/// the `/v1/end-user/...` routes, where Bifrost also checks space membership.
pub struct MagickmindArtifactStore {
    http: reqwest::Client,
    base_url: String,
    auth: Arc<dyn Auth>,
    credential_kind: CredentialKind,
}

impl MagickmindArtifactStore {
    /// A store calling Bifrost at `base_url` with `auth`'s credentials.
    ///
    /// # Errors
    ///
    /// Fails when the HTTP client cannot be built (no TLS backend).
    pub fn new(
        base_url: impl Into<String>,
        auth: Arc<dyn Auth>,
        credential_kind: CredentialKind,
    ) -> Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| api_err(format!("artifact HTTP client: {e}"), None))?;
        Ok(Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            auth,
            credential_kind,
        })
    }

    fn end_user(&self) -> bool {
        self.credential_kind == CredentialKind::EndUser
    }

    fn space_url(&self, space: &str, rest: &str) -> Result<String> {
        if !is_artifact_id(space) {
            return Err(MindroidError::artifact(format!(
                "artifact scope '{space}' is not a magickspace id"
            )));
        }
        let prefix = if self.end_user() {
            "/v1/end-user"
        } else {
            "/v1"
        };
        Ok(format!(
            "{}{prefix}/magickspaces/{space}/artifacts/{rest}",
            self.base_url
        ))
    }

    fn owned_url(&self, rest: &str) -> String {
        let prefix = if self.end_user() {
            "/v1/end-user"
        } else {
            "/v1"
        };
        format!("{}{prefix}/artifacts/{rest}", self.base_url)
    }

    async fn bifrost(&self, method: Method, url: &str) -> Result<RequestBuilder> {
        let headers = crate::auth::build_auth_header_map(self.auth.as_ref()).await?;
        Ok(self.http.request(method, url).headers(headers))
    }

    async fn download(&self, url: &str) -> Result<Option<DownloadResponse>> {
        let resp = send(self.bifrost(Method::GET, url).await?, "download").await?;
        if resp.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        Ok(Some(json(checked(resp, "download").await?).await?))
    }
}

#[derive(Serialize)]
struct PresignRequest<'a> {
    content_type: &'a str,
    size_bytes: u64,
}

#[derive(Deserialize)]
struct PresignResponse {
    id: String,
    bucket: String,
    key: String,
    upload_url: String,
    #[serde(default)]
    required_headers: HashMap<String, String>,
}

#[derive(Serialize)]
struct FinalizeRequest<'a> {
    artifact_id: &'a str,
    bucket: &'a str,
    key: &'a str,
}

#[derive(Deserialize)]
struct DownloadResponse {
    download_url: String,
    #[serde(default)]
    content_type: String,
}

fn api_err(message: String, status: Option<StatusCode>) -> MindroidError {
    MindroidError::Api {
        message,
        status_code: status.map(|s| s.as_u16()),
    }
}

async fn send(req: RequestBuilder, step: &str) -> Result<Response> {
    req.send().await.map_err(|e| {
        api_err(
            format!("artifact {step} request failed: {}", e.without_url()),
            None,
        )
    })
}

async fn checked(resp: Response, step: &str) -> Result<Response> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let body = crate::core::net::error_excerpt(&resp.text().await.unwrap_or_default());
    Err(api_err(
        format!("artifact {step} failed: {status} {body}"),
        Some(status),
    ))
}

async fn json<T: serde::de::DeserializeOwned>(resp: Response) -> Result<T> {
    resp.json()
        .await
        .map_err(|e| api_err(format!("artifact response decode failed: {e}"), None))
}

impl MagickmindArtifactStore {
    async fn fetch(&self, scope: &str, id: &str, limit: usize) -> Result<Artifact> {
        let id = path_safe_id(id)?;
        // The owned route also serves this caller's uploads that no message has attached yet.
        let space = if self.end_user() {
            self.download(&self.space_url(scope, &format!("{id}/download"))?)
                .await?
        } else {
            None
        };
        let download = match space {
            Some(d) => d,
            None => self
                .download(&self.owned_url(&format!("{id}/download")))
                .await?
                .ok_or_else(|| MindroidError::artifact(format!("artifact '{id}' not found")))?,
        };

        let mut resp = checked(
            send(self.http.get(&download.download_url), "fetch").await?,
            "fetch",
        )
        .await?;
        let too_large = || super::exceeds(id, limit);
        if resp.content_length().is_some_and(|n| n > limit as u64) {
            return Err(too_large());
        }
        let header_mime = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let mut data = Vec::new();
        while let Some(chunk) = resp.chunk().await.map_err(|e| {
            api_err(
                format!("artifact fetch body failed: {}", e.without_url()),
                None,
            )
        })? {
            if data.len() + chunk.len() > limit {
                return Err(too_large());
            }
            data.extend_from_slice(&chunk);
        }
        let mime_type = Some(download.content_type)
            .filter(|m| !m.is_empty())
            .or(header_mime)
            .unwrap_or_else(|| "application/octet-stream".into());
        Ok(Artifact { data, mime_type })
    }
}

#[async_trait]
impl ArtifactStore for MagickmindArtifactStore {
    async fn save(&self, scope: &str, data: &[u8], mime_type: &str) -> Result<StoredArtifact> {
        let presign: PresignResponse = json(
            checked(
                send(
                    self.bifrost(Method::POST, &self.space_url(scope, "presign")?)
                        .await?
                        .json(&PresignRequest {
                            content_type: mime_type,
                            size_bytes: data.len() as u64,
                        }),
                    "presign",
                )
                .await?,
                "presign",
            )
            .await?,
        )
        .await?;

        // Every required header is bound into the signature: drop one and S3 refuses the PUT.
        let put = presign
            .required_headers
            .iter()
            .fold(self.http.put(&presign.upload_url), |req, (k, v)| {
                req.header(k, v)
            })
            .body(data.to_vec());
        checked(send(put, "upload").await?, "upload").await?;

        let finalize_url = if self.end_user() {
            self.space_url(scope, "finalize")?
        } else {
            self.owned_url("finalize")
        };
        let finalize = self
            .bifrost(Method::POST, &finalize_url)
            .await?
            .json(&FinalizeRequest {
                artifact_id: &presign.id,
                bucket: &presign.bucket,
                key: &presign.key,
            });
        checked(send(finalize, "finalize").await?, "finalize").await?;

        Ok(StoredArtifact::new(presign.id))
    }

    async fn load(&self, scope: &str, id: &str) -> Result<Artifact> {
        self.fetch(scope, id, MAX_DOWNLOAD_BYTES).await
    }

    async fn load_bounded(&self, scope: &str, id: &str, max_bytes: usize) -> Result<Artifact> {
        self.fetch(scope, id, max_bytes.min(MAX_DOWNLOAD_BYTES))
            .await
    }

    async fn delete(&self, _scope: &str, id: &str) -> Result<()> {
        let id = path_safe_id(id)?;
        let resp = send(
            self.bifrost(Method::DELETE, &self.owned_url(id)).await?,
            "delete",
        )
        .await?;
        if resp.status() == StatusCode::NOT_FOUND {
            return Ok(());
        }
        checked(resp, "delete").await?;
        Ok(())
    }

    fn scope_for(&self, message: &Message) -> String {
        message.conversation_id().to_string()
    }
}

/// Ids reach here from the model via `get_artifact`, so only plain tokens go into a URL path.
fn path_safe_id(id: &str) -> Result<&str> {
    if is_artifact_id(id) {
        Ok(id)
    } else {
        Err(MindroidError::artifact(format!(
            "invalid artifact id '{id}'"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::static_id::StaticAuth;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    struct Seen {
        method: String,
        path: String,
        headers: String,
        body: Vec<u8>,
    }

    /// Answers one request per response, in order, recording each; `responses`
    /// gets the server's base URL so a presign can point its upload back here.
    async fn serve(
        responses: impl FnOnce(&str) -> Vec<(u16, String)>,
    ) -> (String, tokio::task::JoinHandle<Vec<Seen>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let responses = responses(&base);
        let handle = tokio::spawn(async move {
            let mut seen = Vec::new();
            for (status, body) in responses {
                let (mut sock, _) = listener.accept().await.unwrap();
                let mut raw = Vec::new();
                let mut buf = [0u8; 8192];
                let head_end = loop {
                    let n = sock.read(&mut buf).await.unwrap();
                    raw.extend_from_slice(&buf[..n]);
                    if let Some(i) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                        break i + 4;
                    }
                };
                let head = String::from_utf8_lossy(&raw[..head_end]).to_ascii_lowercase();
                let length = head
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .map_or(0, |v| v.trim().parse::<usize>().unwrap());
                while raw.len() < head_end + length {
                    let n = sock.read(&mut buf).await.unwrap();
                    raw.extend_from_slice(&buf[..n]);
                }
                let mut line = head.lines().next().unwrap().split(' ');
                seen.push(Seen {
                    method: line.next().unwrap().to_ascii_uppercase(),
                    path: line.next().unwrap().to_string(),
                    headers: head.clone(),
                    body: raw[head_end..head_end + length].to_vec(),
                });
                let reply = format!(
                    "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                sock.write_all(reply.as_bytes()).await.unwrap();
                sock.shutdown().await.ok();
            }
            seen
        });
        (base, handle)
    }

    fn store(base: &str, kind: CredentialKind) -> MagickmindArtifactStore {
        MagickmindArtifactStore::new(base, Arc::new(StaticAuth::new("tok")), kind).unwrap()
    }

    fn presign(base: &str) -> String {
        serde_json::json!({
            "id": "a1", "bucket": "b", "key": "k", "upload_url": format!("{base}/s3-put"),
            "required_headers": { "Content-Type": "image/png", "x-amz-tagging": "artifact-state=pending" },
        })
        .to_string()
    }

    #[tokio::test]
    async fn end_user_save_presigns_uploads_and_finalizes_in_the_space() {
        let (base, server) = serve(|base| {
            vec![
                (200, presign(base)),
                (200, String::new()),
                (200, "{}".into()),
            ]
        })
        .await;

        let stored = store(&base, CredentialKind::EndUser)
            .save("space1", b"png-bytes", "image/png")
            .await
            .unwrap();
        let seen = server.await.unwrap();

        assert_eq!(stored.id, "a1");
        assert_eq!(seen[0].method, "POST");
        assert_eq!(
            seen[0].path,
            "/v1/end-user/magickspaces/space1/artifacts/presign"
        );
        assert!(seen[0].headers.contains("authorization: bearer tok"));
        let body: serde_json::Value = serde_json::from_slice(&seen[0].body).unwrap();
        assert_eq!(
            body,
            serde_json::json!({ "content_type": "image/png", "size_bytes": 9 })
        );

        assert_eq!(
            (seen[1].method.as_str(), seen[1].path.as_str()),
            ("PUT", "/s3-put")
        );
        assert!(
            seen[1]
                .headers
                .contains("x-amz-tagging: artifact-state=pending")
        );
        assert!(seen[1].headers.contains("content-type: image/png"));
        assert!(
            !seen[1].headers.contains("authorization"),
            "the bearer token must never reach S3"
        );
        assert_eq!(seen[1].body, b"png-bytes");

        assert_eq!(
            seen[2].path,
            "/v1/end-user/magickspaces/space1/artifacts/finalize"
        );
        let body: serde_json::Value = serde_json::from_slice(&seen[2].body).unwrap();
        assert_eq!(
            body,
            serde_json::json!({ "artifact_id": "a1", "bucket": "b", "key": "k" })
        );
    }

    #[tokio::test]
    async fn service_user_save_finalizes_on_the_tenant_route() {
        let (base, server) = serve(|base| {
            vec![
                (200, presign(base)),
                (200, String::new()),
                (200, "{}".into()),
            ]
        })
        .await;

        store(&base, CredentialKind::ServiceUser)
            .save("space1", b"x", "image/png")
            .await
            .unwrap();
        let seen = server.await.unwrap();

        assert_eq!(seen[0].path, "/v1/magickspaces/space1/artifacts/presign");
        assert_eq!(seen[2].path, "/v1/artifacts/finalize");
    }

    #[tokio::test]
    async fn a_failed_upload_does_not_finalize() {
        let (base, server) = serve(|base| vec![(200, presign(base)), (403, "denied".into())]).await;

        let err = store(&base, CredentialKind::EndUser)
            .save("space1", b"x", "image/png")
            .await;
        let seen = server.await.unwrap();

        assert!(err.is_err());
        assert_eq!(
            seen.len(),
            2,
            "finalize must not run after S3 refuses the PUT"
        );
    }

    #[tokio::test]
    async fn end_user_load_falls_back_to_the_owned_route_before_attachment() {
        let (base, server) = serve(|base| {
            let download = serde_json::json!({
                "download_url": format!("{base}/s3-get"),
                "content_type": "image/jpeg",
            });
            vec![
                (404, "{}".into()),
                (200, download.to_string()),
                (200, "jpeg-bytes".into()),
            ]
        })
        .await;

        let artifact = store(&base, CredentialKind::EndUser)
            .load("space1", "a1")
            .await
            .unwrap();
        let seen = server.await.unwrap();

        assert_eq!(
            seen[0].path,
            "/v1/end-user/magickspaces/space1/artifacts/a1/download"
        );
        assert_eq!(seen[1].path, "/v1/end-user/artifacts/a1/download");
        assert_eq!(seen[2].path, "/s3-get");
        assert!(!seen[2].headers.contains("authorization"));
        assert_eq!(artifact.data, b"jpeg-bytes");
        assert_eq!(artifact.mime_type, "image/jpeg");
    }

    #[tokio::test]
    async fn a_bounded_load_stops_at_the_limit() {
        let (base, server) = serve(|base| {
            let download = serde_json::json!({ "download_url": format!("{base}/s3-get") });
            vec![(200, download.to_string()), (200, "0123456789".into())]
        })
        .await;

        let err = store(&base, CredentialKind::EndUser)
            .load_bounded("space1", "a1", 4)
            .await
            .unwrap_err();
        server.await.unwrap();

        assert!(err.to_string().contains("exceeds 4 bytes"), "{err}");
    }

    #[tokio::test]
    async fn load_reports_an_artifact_neither_route_serves() {
        let (base, server) = serve(|_| vec![(404, "{}".into()), (404, "{}".into())]).await;

        let err = store(&base, CredentialKind::EndUser)
            .load("space1", "a1")
            .await
            .unwrap_err();
        server.await.unwrap();

        assert!(err.to_string().contains("not found"), "{err}");
    }

    #[tokio::test]
    async fn delete_is_idempotent_on_the_owned_route() {
        let (base, server) = serve(|_| vec![(404, "{}".into())]).await;

        store(&base, CredentialKind::EndUser)
            .delete("space1", "a1")
            .await
            .unwrap();
        let seen = server.await.unwrap();

        assert_eq!(seen[0].method, "DELETE");
        assert_eq!(seen[0].path, "/v1/end-user/artifacts/a1");
    }

    #[tokio::test]
    async fn a_refusal_on_the_space_route_does_not_fall_back() {
        let (base, server) = serve(|_| vec![(403, "{}".into())]).await;

        let err = store(&base, CredentialKind::EndUser)
            .load("space1", "a1")
            .await;
        let seen = server.await.unwrap();

        assert!(err.is_err());
        assert_eq!(seen.len(), 1, "only a 404 may fall back to the owned route");
    }

    #[tokio::test]
    async fn service_user_load_uses_the_tenant_route_only() {
        let (base, server) = serve(|base| {
            let download = serde_json::json!({ "download_url": format!("{base}/s3-get") });
            vec![(200, download.to_string()), (200, "bytes".into())]
        })
        .await;

        let artifact = store(&base, CredentialKind::ServiceUser)
            .load("space1", "a1")
            .await
            .unwrap();
        let seen = server.await.unwrap();

        assert_eq!(seen[0].path, "/v1/artifacts/a1/download");
        assert_eq!(artifact.data, b"bytes");
    }

    #[tokio::test]
    async fn a_scope_that_is_not_a_space_id_never_reaches_a_url() {
        let store = store("http://127.0.0.1:9", CredentialKind::EndUser);
        for scope in ["user:a1#a1", "../v1/admin", "", "space 1"] {
            assert!(
                store.save(scope, b"x", "image/png").await.is_err(),
                "{scope:?}"
            );
            assert!(store.load(scope, "a1").await.is_err(), "{scope:?}");
        }
    }

    #[tokio::test]
    async fn ids_that_are_not_plain_tokens_never_reach_a_url() {
        let store = store("http://127.0.0.1:9", CredentialKind::EndUser);
        for id in ["../a1", "a1/download", "a 1", "", "a1?x=1"] {
            assert!(store.load("space1", id).await.is_err(), "{id:?}");
            assert!(store.delete("space1", id).await.is_err(), "{id:?}");
        }
    }

    #[test]
    fn scope_is_the_conversation_not_the_delivery_channel() {
        let store = store("http://unused", CredentialKind::EndUser);
        let mut message = Message::new("hi", "u1", "user:a1#a1");
        assert_eq!(store.scope_for(&message), "user:a1#a1");
        message
            .metadata
            .insert("magickspace_id".into(), "space1".into());
        assert_eq!(store.scope_for(&message), "space1");
    }
}
