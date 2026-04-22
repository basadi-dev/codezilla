package multiagent

import (
	"sync"
)

// EventType categorises the system broadcasts
type EventType string

const (
	EventTaskStarted    EventType = "task_started"
	EventTaskCompleted  EventType = "task_completed"
	EventWorkerThinking EventType = "worker_thinking"
	// EventWorkerToolCall is published when a worker is about to call a tool.
	// Payload: "tool_name" (string), "task_id" (string)
	EventWorkerToolCall EventType = "worker_tool_call"
	EventFileDiscovered EventType = "file_discovered"
	EventFileModified   EventType = "file_modified"
)

// Event represents a broadcast message across the agent framework
type Event struct {
	Type     EventType
	WorkerID string
	Payload  map[string]interface{}
}

// subscriber wraps a channel with an ID for targeted unsubscription
type subscriber struct {
	id uint64
	ch chan<- Event
}

// MemoryBus acts as the thread-safe communication layer between parallel agents.
// It provides a shared Key-Value store and a pub/sub mechanism.
type MemoryBus struct {
	store sync.Map

	subsMu sync.RWMutex
	subs   map[EventType][]subscriber
	nextID uint64 // monotonic subscriber ID, protected by subsMu
}

// NewMemoryBus instantiates a new concurrency-safe communication bus
func NewMemoryBus() *MemoryBus {
	return &MemoryBus{
		subs: make(map[EventType][]subscriber),
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

// Subscribe registers a channel to receive events of a specific type.
// Returns a subscription ID that can be passed to Unsubscribe for cleanup.
func (b *MemoryBus) Subscribe(eventType EventType, ch chan<- Event) uint64 {
	b.subsMu.Lock()
	defer b.subsMu.Unlock()
	b.nextID++
	id := b.nextID
	b.subs[eventType] = append(b.subs[eventType], subscriber{id: id, ch: ch})
	return id
}

// Unsubscribe removes a subscriber by its ID. Safe to call multiple times.
func (b *MemoryBus) Unsubscribe(eventType EventType, subID uint64) {
	b.subsMu.Lock()
	defer b.subsMu.Unlock()

	listeners := b.subs[eventType]
	for i, s := range listeners {
		if s.id == subID {
			b.subs[eventType] = append(listeners[:i], listeners[i+1:]...)
			return
		}
	}
}

// Publish broadcasts an event to all subscribed listeners asynchronously
func (b *MemoryBus) Publish(event Event) {
	b.subsMu.RLock()
	defer b.subsMu.RUnlock()

	listeners := b.subs[event.Type]
	if len(listeners) == 0 {
		return
	}

	for _, s := range listeners {
		// Non-blocking send
		select {
		case s.ch <- event:
		default:
			// If channel is full, we drop the event to avoid blocking the bus
		}
	}
}
