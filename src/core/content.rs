use serde::{Deserialize, Serialize};

/// Source of multi-modal content data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContentSource {
    Inline { data: Vec<u8> },
    Uri { uri: String },
}

/// A typed content part for multi-modal LLM messages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        source: ContentSource,
        mime_type: String,
    },
    Audio {
        source: ContentSource,
        mime_type: String,
        #[serde(default)]
        sample_rate: Option<u32>,
    },
    Video {
        source: ContentSource,
        mime_type: String,
        #[serde(default)]
        duration_ms: Option<u64>,
    },
    File {
        source: ContentSource,
        mime_type: String,
        #[serde(default)]
        filename: Option<String>,
    },
}

impl ContentPart {
    /// Create a text content part.
    pub fn text(s: impl Into<String>) -> Self {
        ContentPart::Text { text: s.into() }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_part_text() {
        let part = ContentPart::text("hello world");
        assert_eq!(part.as_text(), Some("hello world"));
        assert!(part.is_text());

        let image = ContentPart::Image {
            source: ContentSource::Uri {
                uri: "https://example.com/img.png".into(),
            },
            mime_type: "image/png".into(),
        };
        assert_eq!(image.as_text(), None);
        assert!(!image.is_text());
    }

    #[test]
    fn test_content_part_serde_roundtrip() {
        let parts = vec![
            ContentPart::text("hello"),
            ContentPart::Image {
                source: ContentSource::Uri {
                    uri: "https://example.com/img.png".into(),
                },
                mime_type: "image/png".into(),
            },
            ContentPart::Audio {
                source: ContentSource::Inline {
                    data: vec![1, 2, 3],
                },
                mime_type: "audio/wav".into(),
                sample_rate: Some(44100),
            },
            ContentPart::Video {
                source: ContentSource::Uri {
                    uri: "https://example.com/vid.mp4".into(),
                },
                mime_type: "video/mp4".into(),
                duration_ms: Some(5000),
            },
            ContentPart::File {
                source: ContentSource::Inline {
                    data: vec![0xDE, 0xAD],
                },
                mime_type: "application/pdf".into(),
                filename: Some("doc.pdf".into()),
            },
        ];

        for part in &parts {
            let json = serde_json::to_string(part).unwrap();
            let decoded: ContentPart = serde_json::from_str(&json).unwrap();
            assert_eq!(part, &decoded);
        }
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
