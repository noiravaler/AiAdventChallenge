package tools

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"ai-chat-go/api"
)

func TestDefinitions(t *testing.T) {
	defs := Definitions()
	if len(defs) != 2 {
		t.Fatalf("definitions len = %d, want 2", len(defs))
	}
	names := map[string]bool{}
	for _, d := range defs {
		if d.Type != "function" {
			t.Errorf("tool %q type = %q, want function", d.Function.Name, d.Type)
		}
		if d.Function.Name == "" {
			t.Errorf("tool has empty name")
		}
		names[d.Function.Name] = true
	}
	if !names["get_current_time"] || !names["roll_dice"] {
		t.Fatalf("definitions missing expected tools: %v", names)
	}
}

func TestRollDiceRange(t *testing.T) {
	seen := map[int]bool{}
	for i := 0; i < 200; i++ {
		out, err := Execute("roll_dice", json.RawMessage(`{}`))
		if err != nil {
			t.Fatalf("roll_dice: %v", err)
		}
		n, err := strconv.Atoi(out)
		if err != nil {
			t.Fatalf("roll_dice returned non-number %q", out)
		}
		if n < 1 || n > 6 {
			t.Fatalf("roll_dice returned out-of-range %d", n)
		}
		seen[n] = true
	}
	if len(seen) != 6 {
		t.Fatalf("saw %d distinct values over 200 rolls, want all 6", len(seen))
	}
}

func TestCurrentTime(t *testing.T) {
	out, err := Execute("get_current_time", json.RawMessage(`{"region":"Europe/Moscow"}`))
	if err != nil {
		t.Fatalf("get_current_time: %v", err)
	}
	if !strings.Contains(out, "Europe/Moscow") {
		t.Fatalf("result %q does not mention the region", out)
	}
}

func TestCurrentTimeUnknownRegion(t *testing.T) {
	if _, err := Execute("get_current_time", json.RawMessage(`{"region":"Nowhere/Nowhere"}`)); err == nil {
		t.Fatal("expected an error for an unknown region")
	}
}

func TestCurrentTimeMissingRegion(t *testing.T) {
	if _, err := Execute("get_current_time", json.RawMessage(`{}`)); err == nil {
		t.Fatal("expected an error for a missing region")
	}
}

func TestUnknownTool(t *testing.T) {
	if _, err := Execute("no_such_tool", json.RawMessage(`{}`)); err == nil {
		t.Fatal("expected an error for an unknown tool")
	}
}

func TestRunLoopStopsOnContent(t *testing.T) {
	content, steps, err := RunLoop(context.Background(), func(_ context.Context, _ []api.Message) (string, []api.ToolCall, error) {
		return "hi", nil, nil
	}, nil, 5)
	if err != nil {
		t.Fatalf("RunLoop: %v", err)
	}
	if content != "hi" {
		t.Fatalf("content = %q, want hi", content)
	}
	if len(steps) != 1 {
		t.Fatalf("steps len = %d, want 1", len(steps))
	}
	if steps[0].Tool != "" || steps[0].Text == "" {
		t.Fatalf("unexpected step: %+v", steps[0])
	}
}

func TestRunLoopSteps(t *testing.T) {
	calls := []api.ToolCall{{
		ID:       "call_1",
		Type:     "function",
		Function: api.ToolCallFunction{Name: "roll_dice", Arguments: `{}`},
	}}
	rounds := 0
	content, steps, err := RunLoop(context.Background(), func(_ context.Context, _ []api.Message) (string, []api.ToolCall, error) {
		rounds++
		if rounds == 1 {
			return "", []api.ToolCall{calls[0]}, nil
		}
		return "5", nil, nil
	}, nil, 3)
	if err != nil {
		t.Fatalf("RunLoop: %v", err)
	}
	if content != "5" {
		t.Fatalf("content = %q, want 5", content)
	}
	if len(steps) != 3 {
		t.Fatalf("steps len = %d, want 3 (round, tool, round): %+v", len(steps), steps)
	}
	// steps[1] must be the tool execution, recording the function and its args.
	if steps[1].Tool != "roll_dice" || steps[1].Args != `{}` || steps[1].Result == "" {
		t.Fatalf("tool step = %+v", steps[1])
	}
}
