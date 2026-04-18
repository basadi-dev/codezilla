package multiagent

import (
	"sync"
)

// EventType categorises the system broadcasts
type EventType string

const (
	EventTaskStarted    EventType = "task_started"
	EventTaskCompleted  EventType = "task_completed"
	EventFileDiscovered EventType = "file_discovered"
	EventFileModified   EventType = "file_modified"
)

// Event represents a broadcast message across the agent framework
type Event struct {
	Type     EventType
	WorkerID string
	Payload  map[string]interface{}
}

// MemoryBus acts as the thread-safe communication layer between parallel agents.
// It provides a shared Key-Value store and a pub/sub mechanism.
type MemoryBus struct {
	store sync.Map

	subsMu sync.RWMutex
	subs   map[EventType][]chan<- Event
}

// NewMemoryBus instantiates a new concurrency-safe communication bus
func NewMemoryBus() *MemoryBus {
	return &MemoryBus{
		subs: make(map[EventType][]chan<- Event),
	}
}

// Get retrieves a key from the shared agent memory
func (b *MemoryBus) Get(key string) (interface{}, bool) {
	return b.store.Load(key)
}

// Set stores a key-value pair in the shared agent memory
func (b *MemoryBus) Set(key string, value interface{}) {
	b.store.Store(key, value)
}

// Delete removes a key from the shared agent memory
func (b *MemoryBus) Delete(key string) {
	b.store.Delete(key)
}

// Subscribe registers a channel to receive events of a specific type
func (b *MemoryBus) Subscribe(eventType EventType, ch chan<- Event) {
	b.subsMu.Lock()
	defer b.subsMu.Unlock()
	b.subs[eventType] = append(b.subs[eventType], ch)
}

// Publish broadcasts an event to all subscribed listeners asynchronously
func (b *MemoryBus) Publish(event Event) {
	b.subsMu.RLock()
	defer b.subsMu.RUnlock()

	listeners := b.subs[event.Type]
	if len(listeners) == 0 {
		return
	}

	for _, ch := range listeners {
		// Non-blocking send
		select {
		case ch <- event:
		default:
			// If channel is full, we drop the event to avoid blocking the bus
		}
	}
}
