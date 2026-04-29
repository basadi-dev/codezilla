//! Per-subscriber filtered event bus.
//!
//! Each `subscribe` call creates a dedicated mpsc channel for that subscriber
//! and registers its [`EventFilter`]. On `publish`, the bus walks the
//! subscriber map, evaluates each filter against the event, and routes the
//! event only to matching subscribers — *publish-time* filtering. This means:
//!
//! - A subscriber listening to thread A no longer wakes up for thread B's
//!   events (and doesn't have to re-implement the filter check itself).
//! - A slow subscriber that fills its mpsc buffer drops its own events
//!   (logged as a warning), without poisoning the channel for others. The
//!   previous broadcast-based design had a single shared lag state — one
//!   slow consumer could trigger `Lagged` for everyone.
//!
//! `EventSubscription::receiver` is now `mpsc::Receiver<RuntimeEvent>`, so
//! callers should `match` on `Some(event)` / `None` instead of
//! `Ok` / `Err(RecvError)`.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use tokio::sync::mpsc;

use crate::system::domain::{RuntimeEvent, ThreadId};

/// Per-subscriber buffer size. Each subscriber gets its own bounded mpsc
/// channel; if it can't keep up, the bus drops the event for *that* subscriber
/// only and logs a warning. Other subscribers are unaffected.
const SUBSCRIBER_BUFFER: usize = 1024;

/// Max events retained in the in-memory ring for `subscribe_with_replay`.
/// Sized for a few minutes of activity; older events fall off and become
/// non-replayable. Process restart drops the ring entirely (intentional —
/// replay is a recover-from-disconnect feature, not a durable log).
const REPLAY_RING_CAPACITY: usize = 4096;

#[derive(Debug, Clone, Default)]
pub struct EventFilter {
    /// If `Some(id)`, the subscriber receives only events whose
    /// `event.thread_id` matches `id`. If `None`, all events.
    pub thread_id: Option<ThreadId>,
}

impl EventFilter {
    /// Returns true if this filter matches the given event.
    fn matches(&self, event: &RuntimeEvent) -> bool {
        match (&self.thread_id, &event.thread_id) {
            (None, _) => true,
            (Some(want), Some(have)) => want == have,
            // Filter requires a thread but the event has none — drop.
            (Some(_), None) => false,
        }
    }
}

#[allow(dead_code)]
pub struct EventSubscription {
    pub subscriber_id: String,
    pub filter: EventFilter,
    pub receiver: mpsc::Receiver<RuntimeEvent>,
}

struct Subscriber {
    filter: EventFilter,
    sender: mpsc::Sender<RuntimeEvent>,
}

struct EventBusInner {
    subscribers: HashMap<String, Subscriber>,
    /// Bounded ring of recent events for late-subscriber replay. Events are
    /// pushed in publish order; sequence numbers are assigned upstream.
    recent: VecDeque<RuntimeEvent>,
}

pub struct EventBus {
    inner: Mutex<EventBusInner>,
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(EventBusInner {
                subscribers: HashMap::new(),
                recent: VecDeque::with_capacity(REPLAY_RING_CAPACITY),
            }),
        }
    }

    /// Route `event` to every subscriber whose filter matches and append it
    /// to the replay ring. A subscriber whose channel is full drops the event
    /// (logged) instead of blocking the publisher. A subscriber whose
    /// receiver was dropped without `unsubscribe` is removed lazily.
    pub fn publish(&self, event: RuntimeEvent) {
        let mut inner = self.inner.lock().unwrap();

        // Append to the replay ring first so subscribe_with_replay sees a
        // consistent prefix once we drop the lock.
        if inner.recent.len() == REPLAY_RING_CAPACITY {
            inner.recent.pop_front();
        }
        inner.recent.push_back(event.clone());

        let mut dead_ids: Vec<String> = Vec::new();
        for (id, sub) in inner.subscribers.iter() {
            if !sub.filter.matches(&event) {
                continue;
            }
            match sub.sender.try_send(event.clone()) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    tracing::warn!(
                        subscriber_id = %id,
                        event_kind = ?event.kind,
                        "event_bus: subscriber buffer full — dropping event for this subscriber only"
                    );
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    dead_ids.push(id.clone());
                }
            }
        }
        for id in dead_ids {
            inner.subscribers.remove(&id);
        }
    }

    pub fn subscribe(&self, subscriber_id: String, filter: EventFilter) -> EventSubscription {
        self.subscribe_with_replay(subscriber_id, filter, None)
    }

    /// Subscribe with optional replay of events from the in-memory ring.
    ///
    /// When `after_sequence` is `Some(n)`, every retained event with
    /// `sequence > n` matching `filter` is delivered to the new subscription's
    /// channel *before* any new events are published. Pass `None` for the
    /// classic "live only" subscription.
    ///
    /// The replay slice is bounded by `REPLAY_RING_CAPACITY` and by the per-
    /// subscriber channel capacity. If a caller's `after_sequence` is older
    /// than the oldest retained event, those events are gone — callers should
    /// treat replay as best-effort recovery, not a durable log.
    pub fn subscribe_with_replay(
        &self,
        subscriber_id: String,
        filter: EventFilter,
        after_sequence: Option<i64>,
    ) -> EventSubscription {
        let (sender, receiver) = mpsc::channel(SUBSCRIBER_BUFFER);
        let mut inner = self.inner.lock().unwrap();

        if let Some(cursor) = after_sequence {
            for event in inner.recent.iter() {
                if event.sequence > cursor && filter.matches(event) {
                    // try_send so a small subscriber buffer doesn't deadlock
                    // the bus. If the buffer is too small for the replay
                    // burst, the caller misses events — same contract as
                    // live publish.
                    if let Err(mpsc::error::TrySendError::Full(_)) = sender.try_send(event.clone())
                    {
                        tracing::warn!(
                            subscriber_id = %subscriber_id,
                            "event_bus: replay buffer full — late events dropped"
                        );
                        break;
                    }
                }
            }
        }

        inner.subscribers.insert(
            subscriber_id.clone(),
            Subscriber {
                filter: filter.clone(),
                sender,
            },
        );

        EventSubscription {
            subscriber_id,
            filter,
            receiver,
        }
    }

    pub fn unsubscribe(&self, subscriber_id: &str) {
        self.inner.lock().unwrap().subscribers.remove(subscriber_id);
    }

    /// Returns the highest `sequence` currently in the replay ring for the
    /// given thread, or `None` if no events are retained for that thread.
    /// Useful for callers that want to capture a checkpoint to resume from.
    #[allow(dead_code)]
    pub fn last_sequence_for_thread(&self, thread_id: &str) -> Option<i64> {
        let inner = self.inner.lock().unwrap();
        inner
            .recent
            .iter()
            .rev()
            .find(|e| e.thread_id.as_deref() == Some(thread_id))
            .map(|e| e.sequence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::domain::{now_millis, RuntimeEventKind};
    use serde_json::json;

    fn evt(kind: RuntimeEventKind, thread_id: Option<&str>) -> RuntimeEvent {
        RuntimeEvent {
            event_id: "e".into(),
            kind,
            thread_id: thread_id.map(str::to_string),
            turn_id: None,
            sequence: 0,
            payload: json!({}),
            emitted_at: now_millis(),
        }
    }

    #[tokio::test]
    async fn unfiltered_subscriber_receives_every_event() {
        let bus = EventBus::new();
        let mut sub = bus.subscribe("a".into(), EventFilter::default());
        bus.publish(evt(RuntimeEventKind::TurnStarted, Some("t1")));
        bus.publish(evt(RuntimeEventKind::TurnCompleted, Some("t2")));
        assert_eq!(
            sub.receiver.recv().await.unwrap().kind,
            RuntimeEventKind::TurnStarted
        );
        assert_eq!(
            sub.receiver.recv().await.unwrap().kind,
            RuntimeEventKind::TurnCompleted
        );
    }

    #[tokio::test]
    async fn thread_filter_drops_unrelated_events_at_publish() {
        let bus = EventBus::new();
        let mut sub = bus.subscribe(
            "a".into(),
            EventFilter {
                thread_id: Some("t1".into()),
            },
        );
        bus.publish(evt(RuntimeEventKind::TurnStarted, Some("t2"))); // dropped
        bus.publish(evt(RuntimeEventKind::TurnStarted, Some("t1"))); // delivered
        bus.publish(evt(RuntimeEventKind::TurnCompleted, None)); // dropped (no thread)
        let got = sub.receiver.recv().await.unwrap();
        assert_eq!(got.kind, RuntimeEventKind::TurnStarted);
        assert_eq!(got.thread_id.as_deref(), Some("t1"));
        // No further events should be queued.
        let next =
            tokio::time::timeout(std::time::Duration::from_millis(50), sub.receiver.recv()).await;
        assert!(
            next.is_err(),
            "expected no more events after publish-time filtering, got {next:?}"
        );
    }

    #[tokio::test]
    async fn slow_subscriber_does_not_block_others() {
        let bus = EventBus::new();
        // Two subscribers, both unfiltered. The slow one is never drained and
        // its buffer will fill; the fast one is drained interleaved with each
        // publish so it never falls behind.
        let _slow = bus.subscribe("slow".into(), EventFilter::default());
        let mut fast = bus.subscribe("fast".into(), EventFilter::default());

        // Publish more than the buffer capacity. After event #SUBSCRIBER_BUFFER
        // the slow subscriber starts dropping; the fast subscriber continues
        // to receive every event without loss.
        let total = SUBSCRIBER_BUFFER + 50;
        for i in 0..total {
            let mut e = evt(RuntimeEventKind::Warning, Some("t1"));
            e.sequence = i as i64;
            bus.publish(e);
            let got = fast
                .receiver
                .recv()
                .await
                .expect("fast subscriber missed event");
            assert_eq!(got.sequence, i as i64);
        }
    }

    #[tokio::test]
    async fn unsubscribe_stops_delivery() {
        let bus = EventBus::new();
        let mut sub = bus.subscribe("a".into(), EventFilter::default());
        bus.unsubscribe("a");
        bus.publish(evt(RuntimeEventKind::TurnStarted, Some("t1")));
        let next =
            tokio::time::timeout(std::time::Duration::from_millis(50), sub.receiver.recv()).await;
        assert!(next.is_err() || next.unwrap().is_none());
    }

    fn evt_at(kind: RuntimeEventKind, thread_id: Option<&str>, sequence: i64) -> RuntimeEvent {
        let mut e = evt(kind, thread_id);
        e.sequence = sequence;
        e
    }

    #[tokio::test]
    async fn replay_delivers_events_after_cursor_then_live_events() {
        let bus = EventBus::new();
        // Publish three events to the ring before anyone subscribes.
        bus.publish(evt_at(RuntimeEventKind::TurnStarted, Some("t1"), 1));
        bus.publish(evt_at(RuntimeEventKind::ItemCompleted, Some("t1"), 2));
        bus.publish(evt_at(RuntimeEventKind::ItemCompleted, Some("t1"), 3));

        // Late-subscribe with cursor=1 → expect events 2 and 3 replayed,
        // then any future events live.
        let mut sub = bus.subscribe_with_replay(
            "late".into(),
            EventFilter {
                thread_id: Some("t1".into()),
            },
            Some(1),
        );

        let r1 = sub.receiver.recv().await.unwrap();
        let r2 = sub.receiver.recv().await.unwrap();
        assert_eq!(r1.sequence, 2);
        assert_eq!(r2.sequence, 3);

        // Live event after subscription must arrive next.
        bus.publish(evt_at(RuntimeEventKind::TurnCompleted, Some("t1"), 4));
        let r3 = sub.receiver.recv().await.unwrap();
        assert_eq!(r3.sequence, 4);
    }

    #[tokio::test]
    async fn replay_respects_filter_and_skips_other_threads() {
        let bus = EventBus::new();
        bus.publish(evt_at(RuntimeEventKind::TurnStarted, Some("t1"), 1));
        bus.publish(evt_at(RuntimeEventKind::TurnStarted, Some("t2"), 2));
        bus.publish(evt_at(RuntimeEventKind::TurnCompleted, Some("t1"), 3));

        let mut sub = bus.subscribe_with_replay(
            "late".into(),
            EventFilter {
                thread_id: Some("t1".into()),
            },
            Some(0),
        );

        // Should see t1's events only, in order.
        let r1 = sub.receiver.recv().await.unwrap();
        let r2 = sub.receiver.recv().await.unwrap();
        assert_eq!(r1.sequence, 1);
        assert_eq!(r2.sequence, 3);
        let next =
            tokio::time::timeout(std::time::Duration::from_millis(50), sub.receiver.recv()).await;
        assert!(
            next.is_err(),
            "no more events expected after replay; got {next:?}"
        );
    }

    #[tokio::test]
    async fn last_sequence_for_thread_tracks_ring() {
        let bus = EventBus::new();
        assert_eq!(bus.last_sequence_for_thread("t1"), None);
        bus.publish(evt_at(RuntimeEventKind::TurnStarted, Some("t1"), 1));
        bus.publish(evt_at(RuntimeEventKind::TurnStarted, Some("t2"), 2));
        bus.publish(evt_at(RuntimeEventKind::TurnCompleted, Some("t1"), 3));
        assert_eq!(bus.last_sequence_for_thread("t1"), Some(3));
        assert_eq!(bus.last_sequence_for_thread("t2"), Some(2));
        assert_eq!(bus.last_sequence_for_thread("missing"), None);
    }

    #[tokio::test]
    async fn ring_drops_oldest_when_full() {
        let bus = EventBus::new();
        // Publish more than the ring capacity; oldest events should fall off.
        for i in 0..(REPLAY_RING_CAPACITY + 100) {
            bus.publish(evt_at(RuntimeEventKind::Warning, Some("t1"), i as i64));
        }
        let mut sub = bus.subscribe_with_replay(
            "late".into(),
            EventFilter {
                thread_id: Some("t1".into()),
            },
            Some(0),
        );
        // First replayed event should be one of the most recent
        // REPLAY_RING_CAPACITY events, NOT sequence 1.
        let first = sub.receiver.recv().await.unwrap();
        assert!(
            first.sequence as usize >= 100,
            "expected oldest events to be evicted; got sequence {}",
            first.sequence
        );
    }
}
