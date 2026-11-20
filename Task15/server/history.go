package server

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"ai-chat-go/storage"
)

// The persisted shapes now live in the storage package (SQLite is the durable
// store); the server keeps type aliases so the tree/branch code is unchanged.
//
// chatRecord is one persisted history message, foldStage one inherited
// prefix-coverage stage of a branch, branchState the per-leaf compression and
// working-memory state (CoveredUpTo/Summary/Plaque/Facts plus Task/Notes),
// historyState the whole per-model session (layers 1–2).
type (
	chatRecord   = storage.ChatRecord
	foldStage    = storage.FoldStage
	branchState  = storage.BranchState
	historyState = storage.HistoryState
)

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

// loadState reads a legacy JSON context file (used to migrate old
// conversations into SQLite). A missing file yields (nil, nil); any stored
// system message is dropped so the current system prompt always governs a
// freshly loaded session.
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

// saveState writes the context to disk as indented JSON (the legacy format,
// kept for fixtures and for migrating old files). The SQLite path does not
// use it.
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

// normalizeLegacyState prepares a legacy JSON context for SQLite: records
// without ids (old flat format) get sequential ids chained linearly and
// duplicate ids are renumbered, so the messages(model, id) unique index
// holds. Well-formed records keep their ids and parents (branches intact).
// Returns st for call chaining; mutates st in place.
func normalizeLegacyState(st *historyState) *historyState {
	seen := make(map[string]bool, len(st.Messages))
	maxID := 0
	for _, r := range st.Messages {
		if n, err := strconv.Atoi(strings.TrimPrefix(r.ID, "m")); err == nil && n > maxID {
			maxID = n
		}
	}
	prev := ""
	for i := range st.Messages {
		r := &st.Messages[i]
		if r.ID == "" || seen[r.ID] {
			// Legacy/duplicate record: number it and chain it linearly.
			for {
				maxID++
				r.ID = "m" + strconv.Itoa(maxID)
				if !seen[r.ID] {
					break
				}
			}
			r.Parent = prev
		}
		seen[r.ID] = true
		prev = r.ID
	}
	if st.NextID <= maxID {
		st.NextID = maxID + 1
	}
	return st
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

// modelHistoryPath derives the per-model legacy JSON path from the base path
// configured as HistoryFile: "<stem>-<modelKey><ext>". An empty base yields "".
// It is still used to find old context files that must migrate into SQLite.
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
