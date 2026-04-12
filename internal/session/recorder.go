package session

import (
	"bufio"
	"encoding/json"
	"os"
	"sync"
	"time"

	"codezilla/pkg/logger"
)

type EventType string

const (
	EventSessionStart EventType = "session_start"
	EventUIInput      EventType = "ui_input"
	EventStateChange  EventType = "state_change"
	EventLLMRequest   EventType = "llm_request"
	EventToken        EventType = "token"
	EventToolStart    EventType = "tool_start"
	EventToolResult   EventType = "tool_result"
	EventError        EventType = "error"
)

type Event struct {
	TimestampNano int64                  `json:"ts"`
	Type          EventType              `json:"type"`
	Data          map[string]interface{} `json:"data,omitempty"`
}

type Recorder struct {
	mu     sync.Mutex
	file   *os.File
	writer *bufio.Writer
	logger *logger.Logger
}

func NewRecorder(filepath string, log *logger.Logger) (*Recorder, error) {
	if filepath == "" {
		return nil, nil // No recorder if empty path
	}

	f, err := os.OpenFile(filepath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		if log != nil {
			log.Error("Failed to open session recorder file", "error", err, "path", filepath)
		}
		return nil, err
	}

	r := &Recorder{
		file:   f,
		writer: bufio.NewWriter(f),
		logger: log,
	}

	r.Record(EventSessionStart, map[string]interface{}{})
	return r, nil
}

func (r *Recorder) Record(evtType EventType, data map[string]interface{}) {
	if r == nil {
		return
	}

	if data == nil {
		data = make(map[string]interface{})
	}

	evt := Event{
		TimestampNano: time.Now().UnixNano(),
		Type:          evtType,
		Data:          data,
	}

	bytes, err := json.Marshal(evt)
	if err == nil {
		r.mu.Lock()
		r.writer.Write(append(bytes, '\n'))
		r.mu.Unlock()
	}
}

func (r *Recorder) Close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.file != nil {
		_ = r.writer.Flush()
		err := r.file.Close()
		r.file = nil
		return err
	}
	return nil
}
