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

	// Add slight delay in test to allow non-blocking sends to buffer
	var e1, e2 Event

	e1 = <-ch1
	e2 = <-ch2

	if e1.WorkerID != "worker-1" || e2.WorkerID != "worker-1" {
		t.Errorf("Expected event broadcast to reach both channels")
	}
}

func TestMemoryBus_State(t *testing.T) {
	bus := NewMemoryBus()

	// Generic KV Test
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
	// Spawn many goroutines reading and writing to ensure sync.Map works
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
