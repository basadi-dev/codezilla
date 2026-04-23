use std::collections::HashMap;
use std::sync::RwLock;
use tokio::sync::broadcast;

use crate::system::domain::{RuntimeEvent, ThreadId};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct EventFilter {
    pub thread_id: Option<ThreadId>,
}

#[allow(dead_code)]
pub struct EventSubscription {
    pub subscriber_id: String,
    pub filter: EventFilter,
    pub receiver: broadcast::Receiver<RuntimeEvent>,
}

pub struct EventBus {
    sender: broadcast::Sender<RuntimeEvent>,
    subscribers: RwLock<HashMap<String, EventFilter>>,
}

impl EventBus {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(1024);
        Self {
            sender,
            subscribers: RwLock::new(HashMap::new()),
        }
    }

    pub fn publish(&self, event: RuntimeEvent) {
        let _ = self.sender.send(event);
    }

    pub fn subscribe(&self, subscriber_id: String, filter: EventFilter) -> EventSubscription {
        self.subscribers
            .write()
            .unwrap()
            .insert(subscriber_id.clone(), filter.clone());
        EventSubscription {
            subscriber_id,
            filter,
            receiver: self.sender.subscribe(),
        }
    }

    pub fn unsubscribe(&self, subscriber_id: &str) {
        self.subscribers.write().unwrap().remove(subscriber_id);
    }
}
