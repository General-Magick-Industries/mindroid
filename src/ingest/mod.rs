//! Pluggable inbound ingest: `Source` + `Encoder` traits over the
//! `ContentPart` boundary.
//!
//! The inbound path that turns a raw incoming attachment into the typed
//! [`ContentPart`]s the model sees is split into two composable, swappable
//! traits — mirroring the `Memory`/`Auth` trait pattern:
//!
//! - [`Source`] resolves a raw input handle (base64 / data-URL / URL / inline
//!   bytes) into a [`ResolvedSource`] (inline bytes or a URI) plus its MIME type.
//!   New *input kind* = new `Source` impl.
//! - [`Encoder`] turns a resolved source + MIME into `Vec<ContentPart>`. New
//!   *modality handling* = new `Encoder` impl.
//!
//! [`IngestStage`] composes a `Source` and an `Encoder` and appends the result
//! to the last user message. The wiring is fixed; behavior lives entirely in the
//! two impls. The default pairing ([`Base64Source`] + [`MediaEncoder`]) reproduces
//! the built-in `AttachMedia` behavior.

use async_trait::async_trait;
use std::sync::Arc;

use crate::core::content::{ContentPart, ContentSource};
use crate::error::Result;

/// A raw, unresolved incoming attachment as produced by a transport or caller.
#[derive(Debug, Clone)]
pub enum RawInput {
    /// Already-decoded inline bytes.
    Bytes { data: Vec<u8>, mime_type: String },
    /// Base64-encoded inline bytes (optionally a full `data:` URL).
    Base64 { b64: String, mime_type: String },
    /// A hosted URI — passed through without fetching.
    Uri { uri: String, mime_type: String },
}

impl RawInput {
    pub fn mime_type(&self) -> &str {
        match self {
            RawInput::Bytes { mime_type, .. }
            | RawInput::Base64 { mime_type, .. }
            | RawInput::Uri { mime_type, .. } => mime_type,
        }
    }
}

/// A resolved source: either inline bytes or a pass-through URI.
#[derive(Debug, Clone)]
pub struct ResolvedSource {
    pub source: ContentSource,
    pub mime_type: String,
}

/// Resolves a [`RawInput`] into a [`ResolvedSource`].
///
/// Implement this to support a new input kind (e.g. fetch a URL into bytes,
/// resolve an upload handle, decrypt). The default [`Base64Source`] handles the
/// base64 / inline / URI forms.
#[async_trait]
pub trait Source: Send + Sync + 'static {
    async fn resolve(&self, input: &RawInput) -> Result<ResolvedSource>;
}

/// Turns a resolved source + MIME into typed [`ContentPart`]s.
#[async_trait]
pub trait Encoder: Send + Sync + 'static {
    async fn encode(&self, resolved: &ResolvedSource) -> Result<Vec<ContentPart>>;
}

// Arc blanket impls — share one Source/Encoder across components (mirror `Auth`).
#[async_trait]
impl<T: Source> Source for Arc<T> {
    async fn resolve(&self, input: &RawInput) -> Result<ResolvedSource> {
        (**self).resolve(input).await
    }
}

#[async_trait]
impl<T: Encoder> Encoder for Arc<T> {
    async fn encode(&self, resolved: &ResolvedSource) -> Result<Vec<ContentPart>> {
        (**self).encode(resolved).await
    }
}

/// Default [`Source`]: decodes base64 to inline bytes, passes inline/URI through.
///
/// Tolerates a full `data:<mime>;base64,<payload>` URL in the `Base64` variant by
/// stripping the prefix.
pub struct Base64Source;

#[async_trait]
impl Source for Base64Source {
    async fn resolve(&self, input: &RawInput) -> Result<ResolvedSource> {
        match input {
            RawInput::Bytes { data, mime_type } => Ok(ResolvedSource {
                source: ContentSource::Inline { data: data.clone() },
                mime_type: mime_type.clone(),
            }),
            RawInput::Uri { uri, mime_type } => Ok(ResolvedSource {
                source: ContentSource::Uri { uri: uri.clone() },
                mime_type: mime_type.clone(),
            }),
            RawInput::Base64 { b64, mime_type } => {
                use base64::{Engine, engine::general_purpose::STANDARD};
                // Strip a `data:...;base64,` prefix if present.
                let payload = b64
                    .rsplit_once("base64,")
                    .map(|(_, p)| p)
                    .unwrap_or(b64.as_str());
                let data = STANDARD.decode(payload.trim()).map_err(|e| {
                    crate::error::MindroidError::config(format!(
                        "Base64Source: failed to decode base64: {e}"
                    ))
                })?;
                Ok(ResolvedSource {
                    source: ContentSource::Inline { data },
                    mime_type: mime_type.clone(),
                })
            }
        }
    }
}

/// Default [`Encoder`]: MIME-dispatches a resolved source to the matching
/// `ContentPart` variant (`image/*` → Image, `audio/*` → Audio, etc.). This is the
/// resolution logic that the built-in `AttachMedia` stage uses, behind the trait.
pub struct MediaEncoder;

#[async_trait]
impl Encoder for MediaEncoder {
    async fn encode(&self, resolved: &ResolvedSource) -> Result<Vec<ContentPart>> {
        let ResolvedSource { source, mime_type } = resolved.clone();
        let part = if mime_type.starts_with("image/") {
            ContentPart::image(source, mime_type)
        } else if mime_type.starts_with("audio/") {
            ContentPart::audio(source, mime_type, None)
        } else if mime_type.starts_with("video/") {
            ContentPart::video(source, mime_type, None)
        } else {
            ContentPart::file(source, mime_type, None)
        };
        Ok(vec![part])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inline_bytes(r: &ResolvedSource) -> &[u8] {
        match &r.source {
            ContentSource::Inline { data } => data,
            ContentSource::Uri { uri } => panic!("expected inline bytes, got uri {uri}"),
        }
    }

    #[tokio::test]
    async fn base64_source_passes_inline_bytes_through_untouched() {
        let out = Base64Source
            .resolve(&RawInput::Bytes {
                data: b"raw".to_vec(),
                mime_type: "image/png".into(),
            })
            .await
            .unwrap();
        assert_eq!(inline_bytes(&out), b"raw");
        assert_eq!(out.mime_type, "image/png");
    }

    #[tokio::test]
    async fn base64_source_passes_uris_through_without_fetching() {
        let out = Base64Source
            .resolve(&RawInput::Uri {
                uri: "https://example.com/a.png".into(),
                mime_type: "image/png".into(),
            })
            .await
            .unwrap();
        match out.source {
            ContentSource::Uri { uri } => assert_eq!(uri, "https://example.com/a.png"),
            ContentSource::Inline { .. } => panic!("a URI must not be fetched into bytes"),
        }
    }

    #[tokio::test]
    async fn base64_source_decodes_bare_payloads() {
        let out = Base64Source
            .resolve(&RawInput::Base64 {
                b64: "aGVsbG8=".into(), // "hello"
                mime_type: "text/plain".into(),
            })
            .await
            .unwrap();
        assert_eq!(inline_bytes(&out), b"hello");
    }

    /// Transports commonly hand over a whole `data:` URL rather than the payload.
    #[tokio::test]
    async fn base64_source_strips_a_data_url_prefix() {
        let out = Base64Source
            .resolve(&RawInput::Base64 {
                b64: "data:image/png;base64,aGVsbG8=".into(),
                mime_type: "image/png".into(),
            })
            .await
            .unwrap();
        assert_eq!(inline_bytes(&out), b"hello");
    }

    #[tokio::test]
    async fn base64_source_reports_undecodable_input() {
        let err = Base64Source
            .resolve(&RawInput::Base64 {
                b64: "!!!not base64!!!".into(),
                mime_type: "image/png".into(),
            })
            .await
            .expect_err("invalid base64 must not silently produce empty bytes");
        assert!(err.to_string().contains("Base64Source"), "{err}");
    }

    #[test]
    fn raw_input_exposes_its_mime_for_every_variant() {
        assert_eq!(
            RawInput::Bytes {
                data: vec![],
                mime_type: "a/b".into()
            }
            .mime_type(),
            "a/b"
        );
        assert_eq!(
            RawInput::Base64 {
                b64: String::new(),
                mime_type: "c/d".into()
            }
            .mime_type(),
            "c/d"
        );
        assert_eq!(
            RawInput::Uri {
                uri: String::new(),
                mime_type: "e/f".into()
            }
            .mime_type(),
            "e/f"
        );
    }

    async fn encode_mime(mime: &str) -> ContentPart {
        let resolved = ResolvedSource {
            source: ContentSource::Inline {
                data: b"x".to_vec(),
            },
            mime_type: mime.into(),
        };
        let mut parts = MediaEncoder.encode(&resolved).await.unwrap();
        assert_eq!(parts.len(), 1, "one source encodes to exactly one part");
        parts.pop().unwrap()
    }

    /// MIME dispatch is the whole behavior of the default encoder; anything
    /// unrecognized must fall back to File rather than being dropped.
    #[tokio::test]
    async fn media_encoder_dispatches_on_mime_prefix() {
        assert!(matches!(
            encode_mime("image/png").await,
            ContentPart::Image { .. }
        ));
        assert!(matches!(
            encode_mime("audio/wav").await,
            ContentPart::Audio { .. }
        ));
        assert!(matches!(
            encode_mime("video/mp4").await,
            ContentPart::Video { .. }
        ));
        assert!(matches!(
            encode_mime("application/pdf").await,
            ContentPart::File { .. }
        ));
        assert!(matches!(encode_mime("").await, ContentPart::File { .. }));
    }
}
