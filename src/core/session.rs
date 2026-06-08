use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::core::message::TransportSend;
use crate::error::Result;
use crate::models::Response;

/// A handle to a single bidirectional connection (for example, a voice stream).
///
/// Created by server-style transports and passed into [`PipelineContext`] via
/// its `session` field. Unlike the per-invocation pipeline state, a
/// `SessionHandle` is connection-scoped: it carries the reply channel back to
/// one specific client and a type-keyed extension map whose contents persist
/// across pipeline runs for the same connection.
///
/// The extension map uses interior mutability ([`RwLock`]) so that callers
/// holding `Arc<SessionHandle>` can still store per-session state without
/// requiring exclusive ownership.
///
/// [`PipelineContext`]: crate::pipeline::PipelineContext
pub struct SessionHandle {
    /// Stable identifier for the connection/session.
    pub id: String,
    /// Reply channel back to this specific connection.
    pub sender: Arc<dyn TransportSend>,
    /// Type-keyed per-session state (interior-mutable for use behind `Arc`).
    extensions: RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>,
}

impl SessionHandle {
    /// Create a session handle wrapping a reply sender.
    pub fn new(id: impl Into<String>, sender: impl TransportSend + 'static) -> Self {
        Self {
            id: id.into(),
            sender: Arc::new(sender),
            extensions: RwLock::new(HashMap::new()),
        }
    }

    /// Send a response back to this connection.
    pub async fn send(&self, response: &Response) -> Result<Option<String>> {
        self.sender.send(response).await
    }

    /// Store a typed value in the per-session extension map.
    ///
    /// Uses interior mutability so this works through `Arc<SessionHandle>`.
    pub fn set_ext<T: Send + Sync + 'static>(&self, value: T) {
        self.extensions
            .write()
            .expect("session extensions lock poisoned")
            .insert(TypeId::of::<T>(), Box::new(value));
    }

    /// Retrieve a clone of a typed value from the extension map.
    ///
    /// Returns `None` if the type is not present.
    pub fn get_ext<T: Send + Sync + Clone + 'static>(&self) -> Option<T> {
        self.extensions
            .read()
            .expect("session extensions lock poisoned")
            .get(&TypeId::of::<T>())?
            .downcast_ref::<T>()
            .cloned()
    }

    /// Remove and return a typed value from the extension map.
    pub fn take_ext<T: Send + Sync + 'static>(&self) -> Option<T> {
        self.extensions
            .write()
            .expect("session extensions lock poisoned")
            .remove(&TypeId::of::<T>())
            .and_then(|b| b.downcast().ok().map(|b| *b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct CapturingSender;

    #[async_trait]
    impl TransportSend for CapturingSender {
        async fn send(&self, response: &Response) -> Result<Option<String>> {
            Ok(Some(response.content.clone()))
        }
    }

    #[tokio::test]
    async fn send_delegates_to_sender() {
        let session = SessionHandle::new("sess-1", CapturingSender);
        assert_eq!(session.id, "sess-1");

        let response = Response::new("hello", "chan", "agent");
        let echoed = session.send(&response).await.unwrap();
        assert_eq!(echoed.as_deref(), Some("hello"));
    }

    #[test]
    fn extensions_round_trip() {
        // set_ext/get_ext work through shared references (interior mutability).
        let session = SessionHandle::new("sess-2", CapturingSender);
        assert!(session.get_ext::<u32>().is_none());

        session.set_ext::<u32>(42);
        assert_eq!(session.get_ext::<u32>(), Some(42));
    }

    #[test]
    fn extensions_work_through_arc() {
        let session = Arc::new(SessionHandle::new("sess-3", CapturingSender));
        session.set_ext::<String>("hello".to_string());
        assert_eq!(session.get_ext::<String>().as_deref(), Some("hello"));

        let taken = session.take_ext::<String>();
        assert_eq!(taken.as_deref(), Some("hello"));
        assert!(session.get_ext::<String>().is_none());
    }
}
