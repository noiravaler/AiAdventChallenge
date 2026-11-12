package server

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"time"

	"ai-chat-go/api"
)

// chatRecord is one persisted history message: the full OpenAI-compatible
// message plus the timestamp it was created. Storing the message as-is keeps
// intermediate tool calls (assistant tool_calls + tool results) intact so the
// conversation can be replayed to the model after a restart.
type chatRecord struct {
	Role       string         `json:"role"`
	Content    string         `json:"content,omitempty"`
	ToolCalls  []api.ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	At         time.Time      `json:"at,omitempty"`
}

// historyState is the on-disk shape of the persisted chat context: the active
// model, the dialog history token size and the conversation turns (system
// prompt is reconstructed on load, so it is not stored).
type historyState struct {
	Model         string       `json:"model,omitempty"`
	HistoryTokens int          `json:"history_tokens,omitempty"`
	Messages      []chatRecord `json:"messages"`
}

// recordOf builds a persisted record from a message and its timestamp.
func recordOf(m api.Message, at time.Time) chatRecord {
	return chatRecord{
		Role:       m.Role,
		Content:    m.Content,
		ToolCalls:  m.ToolCalls,
		ToolCallID: m.ToolCallID,
		At:         at,
	}
}

// message returns the stored record as an api.Message (without its timestamp).
func (r chatRecord) message() api.Message {
	return api.Message{
		Role:       r.Role,
		Content:    r.Content,
		ToolCalls:  r.ToolCalls,
		ToolCallID: r.ToolCallID,
	}
}

// loadState reads a persisted context file. A missing file yields (nil, nil);
// any stored system message is dropped so the current system prompt always
// governs a freshly loaded session.
func loadState(path string) (*historyState, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var st historyState
	if err := json.Unmarshal(data, &st); err != nil {
		return nil, err
	}
	// Filter out any system turn left in the stored file.
	dst := st.Messages[:0]
	for _, m := range st.Messages {
		if m.Role != "system" {
			dst = append(dst, m)
		}
	}
	st.Messages = dst
	return &st, nil
}

// saveState writes the context to disk as indented JSON, creating the parent
// directory if needed. The variadic ctxTokens (first value wins) stores the
// accumulated context token total; omitted keeps the field absent (omitempty).
func saveState(path, model string, msgs []chatRecord, ctxTokens ...int) error {
	total := 0
	if len(ctxTokens) > 0 {
		total = ctxTokens[0]
	}
	data, err := json.MarshalIndent(historyState{Model: model, HistoryTokens: total, Messages: msgs}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// latestAt returns the newest message timestamp in a context; zero if the
// context has no timestamped messages (e.g. an empty or freshly reset one).
func latestAt(msgs []chatRecord) time.Time {
	var latest time.Time
	for _, m := range msgs {
		if m.At.After(latest) {
			latest = m.At
		}
	}
	return latest
}

// sanitizeModelKey turns a model name into a filesystem-safe token (keeps
// alphanumerics, '.', '-', '_'; any other character becomes '_').
func sanitizeModelKey(model string) string {
	var b strings.Builder
	for _, r := range model {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '.', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	s := b.String()
	if s == "" || s == "." || s == ".." {
		s = "model"
	}
	return s
}

// modelHistoryPath derives the per-model context path from the base path
// configured as HistoryFile: "<stem>-<modelKey><ext>". An empty base yields "".
func modelHistoryPath(base, model string) string {
	if base == "" {
		return ""
	}
	dir := filepath.Dir(base)
	name := filepath.Base(base)
	ext := filepath.Ext(name)
	stem := strings.TrimSuffix(name, ext)
	return filepath.Join(dir, stem+"-"+sanitizeModelKey(model)+ext)
}
