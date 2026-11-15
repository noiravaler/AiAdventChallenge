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
// conversation can be replayed to the model after a restart. ID and Parent
// link the record into the conversation tree: Parent points at the message it
// replies to (empty for the tree root), so alternative dialog branches survive
// a restart. Legacy files have no ids — they migrate to one linear branch.
type chatRecord struct {
	ID         string         `json:"id,omitempty"`
	Parent     string         `json:"parent,omitempty"`
	Role       string         `json:"role"`
	Content    string         `json:"content,omitempty"`
	ToolCalls  []api.ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	At         time.Time      `json:"at,omitempty"`
}

// branchState is the persisted state of one dialog branch (a leaf of the
// conversation tree). Compression is a property of the branch, not of the
// tree: CoveredUpTo marks the last message folded away on this branch,
// Summary is the Summary-strategy digest sent to the model as a system
// message, Plaque holds the display-only chat plaque text ("Сжатие
// контекста…"), and Facts/FactsUpTo carry the Sticky Facts block with the
// point it covers. HistoryTokens is the per-branch history size.
//
// ForkUpTo is the fork point this branch grew from (empty for the root
// branch): compression on the branch folds only the branch's own messages —
// strictly after the fork point — while the shared prefix and the fork point
// itself stay visible. Frozen carries the prefix-coverage stages inherited at
// fork time (each stage folds its range and shows its plaque; nodes between
// stages remain displayed), so a nested fork keeps every earlier stage.
type branchState struct {
	Leaf          string      `json:"leaf"`
	Title         string      `json:"title,omitempty"`
	ForkUpTo      string      `json:"fork_up_to,omitempty"`
	Frozen        []foldStage `json:"frozen,omitempty"`
	CoveredUpTo   string      `json:"covered_up_to,omitempty"`
	Summary       string      `json:"summary,omitempty"`
	Plaque        string      `json:"plaque,omitempty"`
	Facts         string      `json:"facts,omitempty"`
	FactsUpTo     string      `json:"facts_up_to,omitempty"`
	HistoryTokens int         `json:"history_tokens,omitempty"`
}

// foldStage is one inherited prefix-coverage stage of a branch (see
// branchState.Frozen): the range covered at fork time plus the digest or
// display plaque that replaces it.
type foldStage struct {
	CoveredUpTo string `json:"covered_up_to,omitempty"`
	Summary     string `json:"summary,omitempty"`
	Plaque      string `json:"plaque,omitempty"`
}

// historyState is the on-disk shape of the persisted chat context: the active
// model, the dialog history token size, the sticky-facts block (Sticky Facts
// compression strategy), the cumulative request-token total and the
// conversation turns (system prompt is reconstructed on load, so it is not
// stored). Branches carry the per-branch compression state and Active names
// the restored branch; legacy files (no branches) migrate to one linear
// branch.
type historyState struct {
	Model             string        `json:"model,omitempty"`
	HistoryTokens     int           `json:"history_tokens,omitempty"`
	Facts             string        `json:"facts,omitempty"`
	TotalPromptTokens int           `json:"total_prompt_tokens,omitempty"`
	Messages          []chatRecord  `json:"messages"`
	Branches          []branchState `json:"branches,omitempty"`
	Switchable        []string      `json:"switchable,omitempty"`
	Active            string        `json:"active,omitempty"`
	NextID            int           `json:"next_id,omitempty"`
}

// recordOf builds a persisted record from a tree entry (message + timestamp
// plus its tree links).
func recordOf(e histEntry) chatRecord {
	return chatRecord{
		ID:         e.ID,
		Parent:     e.Parent,
		Role:       e.Msg.Role,
		Content:    e.Msg.Content,
		ToolCalls:  e.Msg.ToolCalls,
		ToolCallID: e.Msg.ToolCallID,
		At:         e.At,
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
// directory if needed.
func saveState(path string, st historyState) error {
	data, err := json.MarshalIndent(st, "", "  ")
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
