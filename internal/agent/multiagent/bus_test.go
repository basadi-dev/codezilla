package multiagent

import (
	"sync"
	"testing"
)

func TestMemoryBus_PubSub(t *testing.T) {
	bus := NewMemoryBus()

	ch1 := make(chan Event, 5)
	ch2 := make(chan Event, 5)

	bus.Subscribe(EventTaskStarted, ch1)
	bus.Subscribe(EventTaskStarted, ch2)

	bus.Publish(Event{
		Type:     EventTaskStarted,
		WorkerID: "worker-1",
	})

	e1 := <-ch1
	e2 := <-ch2

	if e1.WorkerID != "worker-1" || e2.WorkerID != "worker-1" {
		t.Errorf("Expected event broadcast to reach both channels")
	}
}

func TestMemoryBus_State(t *testing.T) {
	bus := NewMemoryBus()

	bus.Set("key1", "value1")

	val, ok := bus.Get("key1")
	if !ok || val != "value1" {
		t.Errorf("Expected Get to return 'value1', got %v", val)
	}

	bus.Delete("key1")
	_, ok = bus.Get("key1")
	if ok {
		t.Errorf("Expected key1 to be deleted")
	}
}

func TestMemoryBus_Concurrency(t *testing.T) {
	bus := NewMemoryBus()

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			bus.Set(string(rune(idx)), idx)
			bus.Get(string(rune(idx)))
		}(i)
	}
	wg.Wait()
}

func TestMemoryBus_Unsubscribe(t *testing.T) {
	bus := NewMemoryBus()

	ch := make(chan Event, 5)
	subID := bus.Subscribe(EventFileModified, ch)

	// Publish should reach subscriber
	bus.Publish(Event{Type: EventFileModified, WorkerID: "w1"})
	e := <-ch
	if e.WorkerID != "w1" {
		t.Errorf("Expected event from w1, got %s", e.WorkerID)
	}

	// After unsubscribe, channel should not receive events
	bus.Unsubscribe(EventFileModified, subID)
	bus.Publish(Event{Type: EventFileModified, WorkerID: "w2"})

	select {
	case evt := <-ch:
		t.Errorf("Should not have received event after unsubscribe, got %+v", evt)
	default:
		// expected — nothing received
	}
}

func TestMemoryBus_UnsubscribeIdempotent(t *testing.T) {
	bus := NewMemoryBus()
	ch := make(chan Event, 1)
	subID := bus.Subscribe(EventTaskStarted, ch)

	// Calling unsubscribe multiple times should not panic
	bus.Unsubscribe(EventTaskStarted, subID)
	bus.Unsubscribe(EventTaskStarted, subID)
}
