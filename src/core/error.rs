#[derive(thiserror::Error, Debug)]
pub enum MindroidError {
    #[error("Auth failed: {message}")]
    Auth {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    #[error("Transport error: {message}")]
    Transport {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    #[error("Pipeline error at stage '{stage}': {message}")]
    Pipeline {
        stage: String,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    #[error("Memory error: {message}")]
    Memory {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    #[error("Artifact error: {message}")]
    Artifact {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    #[error("API error: {message} (HTTP {status_code:?})")]
    Api {
        message: String,
        status_code: Option<u16>,
    },
    #[error("Config error: {message}")]
    Config {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl MindroidError {
    /// Convenience constructor for config errors without a source chain.
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
            source: None,
        }
    }

    /// Convenience constructor for pipeline errors without a source chain.
    pub fn pipeline(message: impl Into<String>) -> Self {
        Self::Pipeline {
            stage: "coordinator".into(),
            message: message.into(),
            source: None,
        }
    }

    /// Convenience constructor for artifact-store errors without a source chain.
    pub fn artifact(message: impl Into<String>) -> Self {
        Self::Artifact {
            message: message.into(),
            source: None,
        }
    }
}

pub type Result<T> = std::result::Result<T, MindroidError>;
