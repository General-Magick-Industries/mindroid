use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use futures::Stream;
use tokio::sync::broadcast;
use tokio_stream::StreamExt as TokioStreamExt;
use tokio_stream::wrappers::BroadcastStream;

/// A typed event emitted when a value in a [`SharedExtensionMap`] changes.
#[derive(Debug, Clone)]
pub enum StateEvent {
    Changed {
        type_id: TypeId,
        type_name: &'static str,
    },
}

// ---------------------------------------------------------------------------
// ExtensionMap — single-threaded, owned values
// ---------------------------------------------------------------------------

/// A typed heterogeneous map keyed by `TypeId`. Values are owned and retrieved
/// by borrow. Not thread-safe; for shared use see [`SharedExtensionMap`].
#[derive(Default)]
pub struct ExtensionMap {
    inner: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl ExtensionMap {
    /// Insert (or replace) a value of type `T`.
    pub fn set<T: Send + Sync + 'static>(&mut self, value: T) {
        self.inner.insert(TypeId::of::<T>(), Box::new(value));
    }

    /// Borrow the stored value of type `T`, if present.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.inner
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref::<T>())
    }

    /// Remove and return the value of type `T`, if present.
    pub fn take<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.inner
            .remove(&TypeId::of::<T>())
            .and_then(|b| b.downcast::<T>().ok())
            .map(|b| *b)
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

// ---------------------------------------------------------------------------
// SharedExtensionMap — lock-free, clone-on-read, broadcast notifications
// ---------------------------------------------------------------------------

/// A thread-safe, clone-on-read heterogeneous map backed by a [`DashMap`].
/// Subscribers can watch for changes via a broadcast channel or typed streams.
///
/// Clone yields a new handle to the **same** underlying map with a fresh
/// broadcast receiver.
pub struct SharedExtensionMap {
    map: Arc<DashMap<TypeId, Box<dyn Any + Send + Sync>>>,
    notify: broadcast::Sender<StateEvent>,
}

impl SharedExtensionMap {
    /// Create a new, empty map with a broadcast channel of capacity 64.
    pub fn new() -> Self {
        let (notify, _) = broadcast::channel(64);
        Self {
            map: Arc::new(DashMap::new()),
            notify,
        }
    }

    /// Insert (or replace) a value of type `T` and broadcast a
    /// [`StateEvent::Changed`] notification.
    pub fn set<T: Clone + Send + Sync + 'static>(&self, value: T) {
        self.map.insert(TypeId::of::<T>(), Box::new(value));
        // Ignore send errors — no active receivers is fine.
        let _ = self.notify.send(StateEvent::Changed {
            type_id: TypeId::of::<T>(),
            type_name: std::any::type_name::<T>(),
        });
    }

    /// Return a clone of the stored value of type `T`, if present.
    ///
    /// Returns `Option<T>` (not a reference) because DashMap references must
    /// not be held across `.await` points.
    pub fn get<T: Clone + Send + Sync + 'static>(&self) -> Option<T> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|entry| entry.value().downcast_ref::<T>().cloned())
    }

    /// Subscribe to change notifications. The returned receiver will receive a
    /// [`StateEvent`] every time any value is set.
    pub fn subscribe(&self) -> broadcast::Receiver<StateEvent> {
        self.notify.subscribe()
    }

    /// Return a [`Stream`] that yields a cloned `T` whenever the value of type
    /// `T` changes. Values set before the stream is created are not replayed.
    pub fn watch<T: Clone + Send + Sync + 'static>(&self) -> impl Stream<Item = T> + '_ {
        let rx = self.notify.subscribe();
        let map = self.map.clone();
        BroadcastStream::new(rx).filter_map(move |result| match result {
            Ok(StateEvent::Changed { type_id, .. }) if type_id == TypeId::of::<T>() => map
                .get(&type_id)
                .and_then(|entry| entry.value().downcast_ref::<T>().cloned()),
            _ => None,
        })
    }
}

impl Default for SharedExtensionMap {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SharedExtensionMap {
    /// Clones the handle: shares the same underlying [`DashMap`] but creates a
    /// fresh broadcast receiver.
    fn clone(&self) -> Self {
        Self {
            map: Arc::clone(&self.map),
            notify: self.notify.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_map_set_get_take() {
        let mut map = ExtensionMap::default();

        // set + get
        map.set(42u32);
        assert_eq!(map.get::<u32>(), Some(&42u32));

        // take removes the value
        let taken = map.take::<u32>();
        assert_eq!(taken, Some(42u32));

        // gone after take
        assert_eq!(map.get::<u32>(), None);
        assert_eq!(map.take::<u32>(), None);
    }

    #[test]
    fn test_shared_extension_map_set_get() {
        let map = SharedExtensionMap::new();

        map.set(99u64);
        // get returns a clone
        let val: Option<u64> = map.get::<u64>();
        assert_eq!(val, Some(99u64));

        // setting again overwrites
        map.set(200u64);
        assert_eq!(map.get::<u64>(), Some(200u64));

        // different types are independent
        map.set("hello".to_string());
        assert_eq!(map.get::<String>(), Some("hello".to_string()));
        assert_eq!(map.get::<u64>(), Some(200u64));
    }

    #[tokio::test]
    async fn test_shared_extension_map_watch() {
        use tokio_stream::StreamExt as _;

        let map = SharedExtensionMap::new();
        let mut stream = map.watch::<u32>();

        // Spawn a task that sets the value after a brief yield
        let map2 = map.clone();
        tokio::spawn(async move {
            tokio::task::yield_now().await;
            map2.set(7u32);
        });

        // The stream should yield the value
        let received = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
            .await
            .expect("timed out waiting for stream item");

        assert_eq!(received, Some(7u32));
    }
}
