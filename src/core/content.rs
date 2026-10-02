use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Arbitrary metadata attached to a media part — typically whatever an
/// [`ArtifactStore`](crate::artifacts::ArtifactStore) returns alongside the id
/// (e.g. an LLM-generated caption's structured data, backend facts like an S3
/// etag/region, content hashes, or extracted entities). The keys and shapes are
/// entirely store/extractor-defined; the framework treats it as opaque and
/// round-trips it untouched. Empty by default and omitted from serialization, so
/// parts without metadata serialize byte-identically to before.
pub type ContentMetadata = Map<String, Value>;

/// Source of multi-modal content data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContentSource {
    Inline { data: Vec<u8> },
    Uri { uri: String },
}

/// A typed content part for multi-modal LLM messages.
///
/// The media variants are `#[non_exhaustive]`: construct them with
/// [`ContentPart::image`] and friends, and match their fields with `..`. This
/// release added a `metadata` field to all four, and doing so silently broke
/// downstream struct literals and exhaustive field patterns — the attribute
/// makes the next such addition a non-event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContentPart {
    Text {
        text: String,
    },
    #[non_exhaustive]
    Image {
        source: ContentSource,
        mime_type: String,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: ContentMetadata,
    },
    #[non_exhaustive]
    Audio {
        source: ContentSource,
        mime_type: String,
        #[serde(default)]
        sample_rate: Option<u32>,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: ContentMetadata,
    },
    #[non_exhaustive]
    Video {
        source: ContentSource,
        mime_type: String,
        #[serde(default)]
        duration_ms: Option<u64>,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: ContentMetadata,
    },
    #[non_exhaustive]
    File {
        source: ContentSource,
        mime_type: String,
        #[serde(default)]
        filename: Option<String>,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: ContentMetadata,
    },
}

impl ContentPart {
    /// Create a text content part.
    pub fn text(s: impl Into<String>) -> Self {
        ContentPart::Text { text: s.into() }
    }

    /// Create an image part (metadata empty).
    pub fn image(source: ContentSource, mime_type: impl Into<String>) -> Self {
        ContentPart::Image {
            source,
            mime_type: mime_type.into(),
            metadata: ContentMetadata::new(),
        }
    }

    /// Create an audio part (metadata empty).
    pub fn audio(
        source: ContentSource,
        mime_type: impl Into<String>,
        sample_rate: Option<u32>,
    ) -> Self {
        ContentPart::Audio {
            source,
            mime_type: mime_type.into(),
            sample_rate,
            metadata: ContentMetadata::new(),
        }
    }

    /// Create a video part (metadata empty).
    pub fn video(
        source: ContentSource,
        mime_type: impl Into<String>,
        duration_ms: Option<u64>,
    ) -> Self {
        ContentPart::Video {
            source,
            mime_type: mime_type.into(),
            duration_ms,
            metadata: ContentMetadata::new(),
        }
    }

    /// Create a file part (metadata empty).
    pub fn file(
        source: ContentSource,
        mime_type: impl Into<String>,
        filename: Option<String>,
    ) -> Self {
        ContentPart::File {
            source,
            mime_type: mime_type.into(),
            filename,
            metadata: ContentMetadata::new(),
        }
    }

    /// Extract text if this is a Text variant.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentPart::Text { text } => Some(text),
            _ => None,
        }
    }

    /// Check if this is a Text variant.
    pub fn is_text(&self) -> bool {
        matches!(self, ContentPart::Text { .. })
    }

    /// Whether this part carries inline bytes (rather than text or a URI).
    pub fn is_inline(&self) -> bool {
        matches!(
            self,
            ContentPart::Image {
                source: ContentSource::Inline { .. },
                ..
            } | ContentPart::Audio {
                source: ContentSource::Inline { .. },
                ..
            } | ContentPart::Video {
                source: ContentSource::Inline { .. },
                ..
            } | ContentPart::File {
                source: ContentSource::Inline { .. },
                ..
            }
        )
    }

    /// Read the metadata map of a media part (`None` for `Text`).
    pub fn metadata(&self) -> Option<&ContentMetadata> {
        match self {
            ContentPart::Text { .. } => None,
            ContentPart::Image { metadata, .. }
            | ContentPart::Audio { metadata, .. }
            | ContentPart::Video { metadata, .. }
            | ContentPart::File { metadata, .. } => Some(metadata),
        }
    }

    /// Mutably access the metadata map of a media part (`None` for `Text`) so a
    /// stage can enrich it in place without matching every variant.
    pub fn metadata_mut(&mut self) -> Option<&mut ContentMetadata> {
        match self {
            ContentPart::Text { .. } => None,
            ContentPart::Image { metadata, .. }
            | ContentPart::Audio { metadata, .. }
            | ContentPart::Video { metadata, .. }
            | ContentPart::File { metadata, .. } => Some(metadata),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_part_text() {
        let part = ContentPart::text("hello world");
        assert_eq!(part.as_text(), Some("hello world"));
        assert!(part.is_text());

        let image = ContentPart::image(
            ContentSource::Uri {
                uri: "https://example.com/img.png".into(),
            },
            "image/png",
        );
        assert_eq!(image.as_text(), None);
        assert!(!image.is_text());
        // Text has no metadata; media parts start with an empty map.
        assert!(part.metadata().is_none());
        assert_eq!(image.metadata(), Some(&ContentMetadata::new()));
    }

    #[test]
    fn test_content_part_serde_roundtrip() {
        let parts = vec![
            ContentPart::text("hello"),
            ContentPart::image(
                ContentSource::Uri {
                    uri: "https://example.com/img.png".into(),
                },
                "image/png",
            ),
            ContentPart::audio(
                ContentSource::Inline {
                    data: vec![1, 2, 3],
                },
                "audio/wav",
                Some(44100),
            ),
            ContentPart::video(
                ContentSource::Uri {
                    uri: "https://example.com/vid.mp4".into(),
                },
                "video/mp4",
                Some(5000),
            ),
            ContentPart::file(
                ContentSource::Inline {
                    data: vec![0xDE, 0xAD],
                },
                "application/pdf",
                Some("doc.pdf".into()),
            ),
        ];

        for part in &parts {
            let json = serde_json::to_string(part).unwrap();
            let decoded: ContentPart = serde_json::from_str(&json).unwrap();
            assert_eq!(part, &decoded);
        }
    }

    #[test]
    fn test_metadata_roundtrips_and_empty_is_omitted() {
        // A part with metadata round-trips the map intact.
        let mut meta = ContentMetadata::new();
        meta.insert("etag".into(), Value::String("abc123".into()));
        meta.insert("entities".into(), serde_json::json!(["person", "monitor"]));
        let mut part = ContentPart::image(ContentSource::Uri { uri: "id-1".into() }, "image/jpeg");
        *part.metadata_mut().unwrap() = meta.clone();

        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("\"metadata\""));
        assert!(json.contains("\"etag\""));
        let decoded: ContentPart = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.metadata(), Some(&meta));

        // An empty-metadata part omits the key entirely — byte-identical to the
        // pre-metadata format (no stray `"metadata":{}`).
        let bare = ContentPart::image(ContentSource::Uri { uri: "id-2".into() }, "image/png");
        let bare_json = serde_json::to_string(&bare).unwrap();
        assert!(
            !bare_json.contains("metadata"),
            "empty metadata must be omitted: {bare_json}"
        );
    }

    #[test]
    fn test_media_metadata_is_backward_compatible() {
        // Old serialized media parts had no `metadata` field — they must still
        // deserialize, defaulting to an empty map.
        let legacy =
            r#"{"type":"image","source":{"kind":"uri","uri":"abc"},"mime_type":"image/png"}"#;
        let decoded: ContentPart = serde_json::from_str(legacy).unwrap();
        assert_eq!(decoded.metadata(), Some(&ContentMetadata::new()));
    }

    #[test]
    fn test_legacy_file_with_description_still_deserializes() {
        // `description` was removed from File. Rows written earlier that still carry
        // a `description` key must deserialize harmlessly (serde ignores the unknown
        // field), yielding a plain File part.
        let legacy = r#"{"type":"file","source":{"kind":"uri","uri":"abc123"},"mime_type":"image/png","filename":"x.png","description":"a cat"}"#;
        let decoded: ContentPart = serde_json::from_str(legacy).unwrap();
        assert_eq!(
            decoded,
            ContentPart::file(
                ContentSource::Uri {
                    uri: "abc123".into()
                },
                "image/png",
                Some("x.png".into()),
            )
        );
    }

    #[test]
    fn test_content_source_serde() {
        let inline = ContentSource::Inline {
            data: vec![1, 2, 3, 4],
        };
        let json = serde_json::to_string(&inline).unwrap();
        let decoded: ContentSource = serde_json::from_str(&json).unwrap();
        assert_eq!(inline, decoded);

        let uri = ContentSource::Uri {
            uri: "https://example.com/file".into(),
        };
        let json = serde_json::to_string(&uri).unwrap();
        let decoded: ContentSource = serde_json::from_str(&json).unwrap();
        assert_eq!(uri, decoded);
    }
}
