package ui

import (
	"context"
	"time"
)

func TestTick() {
	termUI, _ := NewBubbleTeaUI("tst.hist", false, false)
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
