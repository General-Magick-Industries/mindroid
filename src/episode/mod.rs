//! Episodic-memory ingest: send every message — inbound and the agent's own
//! reply — to MagickMind's episode `/process` endpoint.
//!
//! mindroid drops the agent's own inbound echo before the pipeline runs (see
//! [`runtime`](crate::core::runtime)), so a single inbound stage would capture
//! only user turns. [`EpisodeIngestStage`] handles inbound; the agent's reply is
//! captured separately by [`EpisodeReplyIngestStage`] after generation.

mod ingest;

pub use ingest::{EpisodeIngestStage, EpisodeReplyIngestStage};
