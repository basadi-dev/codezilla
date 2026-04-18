package ui

import (
	"context"
	"fmt"
	"os"
	"time"
)

func TestTick() {
	termUI, err := NewBubbleTeaUI(nil, false, false)
	if err != nil {
		fmt.Printf("Failed: %v\n", err)
		os.Exit(1)
	}
	tui := termUI.(TUIRunner)

	tui.RunTUI(context.Background(), func(ctx context.Context) error {
		termUI.ShowThinking()
		termUI.UpdateThinkingStatus("Calling LLM...")
		time.Sleep(5 * time.Second)
		termUI.HideThinking()
		time.Sleep(1 * time.Second)
		return nil
	})
}
