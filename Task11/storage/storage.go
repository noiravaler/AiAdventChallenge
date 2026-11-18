// Package storage persists the chat context (short-term and working memory)
// and the long-term memory in a single SQLite database. The three memory
// layers live in separate table groups keyed by the model name:
//
//	messages / branches / session_state — the dialog tree and per-branch
//	                                      compression state (layers 1–2);
//	memories                            — durable profile, decisions and
//	                                      knowledge facts (layer 3).
//
// The driver is modernc.org/sqlite: pure Go, no cgo.
package storage

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"ai-chat-go/api"

	_ "modernc.org/sqlite"
)

// ChatRecord is one persisted history message: the full OpenAI-compatible
// message plus the timestamp it was created. Storing the message as-is keeps
// intermediate tool calls (assistant tool_calls + tool results) intact so the
// conversation can be replayed to the model after a restart. ID and Parent
// link the record into the conversation tree: Parent points at the message it
// replies to (empty for the tree root).
type ChatRecord struct {
	ID         string         `json:"id,omitempty"`
	Parent     string         `json:"parent,omitempty"`
	Role       string         `json:"role"`
	Content    string         `json:"content,omitempty"`
	ToolCalls  []api.ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	At         time.Time      `json:"at,omitempty"`
}

// Message returns the stored record as an api.Message (without its timestamp).
func (r ChatRecord) Message() api.Message {
	return api.Message{
		Role:       r.Role,
		Content:    r.Content,
		ToolCalls:  r.ToolCalls,
		ToolCallID: r.ToolCallID,
	}
}

// FoldStage is one inherited prefix-coverage stage of a branch: the range
// covered at fork time plus the digest or display plaque that replaces it.
type FoldStage struct {
	CoveredUpTo string `json:"covered_up_to,omitempty"`
	Summary     string `json:"summary,omitempty"`
	Plaque      string `json:"plaque,omitempty"`
}

// BranchState is the persisted state of one dialog branch (a leaf of the
// conversation tree): the compression bookkeeping (CoveredUpTo, Summary,
// Facts and the inherited Frozen stages) plus the working-memory fields Task
// (the current task's goal, rewritten by compression) and Notes (free-form
// artifacts). HistoryTokens is the per-branch history size.
type BranchState struct {
	Leaf          string      `json:"leaf"`
	Title         string      `json:"title,omitempty"`
	ForkUpTo      string      `json:"fork_up_to,omitempty"`
	Frozen        []FoldStage `json:"frozen,omitempty"`
	CoveredUpTo   string      `json:"covered_up_to,omitempty"`
	Summary       string      `json:"summary,omitempty"`
	Plaque        string      `json:"plaque,omitempty"`
	Facts         string      `json:"facts,omitempty"`
	FactsUpTo     string      `json:"facts_up_to,omitempty"`
	Task          string      `json:"task,omitempty"`
	Notes         string      `json:"notes,omitempty"`
	HistoryTokens int         `json:"history_tokens,omitempty"`
}

// HistoryState is the persisted shape of one model's chat context: the model
// name, the dialog history token size, the sticky-facts block, the cumulative
// request-token total and the conversation turns (the system prompt is
// reconstructed on load, so it is not stored). Branches carry the per-branch
// compression state and Active names the restored branch.
type HistoryState struct {
	Model         string        `json:"model,omitempty"`
	HistoryTokens int           `json:"history_tokens,omitempty"`
	Facts         string        `json:"facts,omitempty"`
	Messages      []ChatRecord  `json:"messages"`
	Branches      []BranchState `json:"branches,omitempty"`
	Switchable    []string      `json:"switchable,omitempty"`
	Active        string        `json:"active,omitempty"`
	NextID        int           `json:"next_id,omitempty"`
}

// Memory kinds (the long-term memory layer).
const (
	KindProfile   = "profile"
	KindDecision  = "decision"
	KindKnowledge = "knowledge"
)

// MemorySources — how a memory item came to be.
const (
	SourceCompress = "compress" // extracted by facts compression
	SourceUser     = "user"     // explicit user request
	SourceTool     = "tool"     // model's remember() tool call
)

// MemoryItem is one long-term memory record: a single «ключ: значение» fact
// with its kind, origin and usage statistics. Confirmations counts how often
// the fact survived re-extraction during compression (facts promoted by
// compression start at 1 and stabilize at 2+); explicit user/tool writes start
// at 2. Hits counts prompt injections.
type MemoryItem struct {
	ID            string    `json:"id"`
	Kind          string    `json:"kind"`
	Key           string    `json:"key"`
	Text          string    `json:"text"`
	Source        string    `json:"source"`
	Created       time.Time `json:"created"`
	Updated       time.Time `json:"updated,omitempty"`
	Confirmations int       `json:"confirmations"`
	Hits          int       `json:"hits"`
	LastHit       time.Time `json:"last_hit,omitempty"`
}

// DB wraps the SQLite handle with the memory-oriented API.
type DB struct {
	db *sql.DB
}

// Open creates (if needed) and opens the SQLite database file, enabling WAL
// and a busy timeout, and makes sure the schema exists.
func Open(path string) (*DB, error) {
	if path == "" {
		return nil, errors.New("storage: empty database path")
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("storage: create directory: %w", err)
		}
	}
	sqlDB, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("storage: open %s: %w", path, err)
	}
	for _, pragma := range []string{
		"PRAGMA busy_timeout=5000",
		"PRAGMA foreign_keys=ON",
		"PRAGMA synchronous=NORMAL",
	} {
		if _, err := sqlDB.Exec(pragma); err != nil {
			sqlDB.Close()
			return nil, fmt.Errorf("storage: %s: %w", pragma, err)
		}
	}
	var mode string
	if err := sqlDB.QueryRow("PRAGMA journal_mode=WAL").Scan(&mode); err != nil {
		sqlDB.Close()
		return nil, fmt.Errorf("storage: journal_mode: %w", err)
	}
	d := &DB{db: sqlDB}
	if err := d.createSchema(); err != nil {
		sqlDB.Close()
		return nil, err
	}
	return d, nil
}

// Close closes the database.
func (d *DB) Close() error { return d.db.Close() }

// createSchema creates the three memory layers' tables. Timestamps are stored
// as INTEGER unix nanoseconds (0 = unset) so MAX() comparisons stay correct.
func (d *DB) createSchema() error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS messages (
			model        TEXT NOT NULL,
			id           TEXT NOT NULL,
			parent       TEXT NOT NULL DEFAULT '',
			role         TEXT NOT NULL,
			content      TEXT NOT NULL DEFAULT '',
			tool_calls   TEXT NOT NULL DEFAULT '',
			tool_call_id TEXT NOT NULL DEFAULT '',
			at           INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (model, id)
		)`,
		`CREATE TABLE IF NOT EXISTS branches (
			model          TEXT NOT NULL,
			leaf           TEXT NOT NULL,
			title          TEXT NOT NULL DEFAULT '',
			fork_up_to     TEXT NOT NULL DEFAULT '',
			frozen         TEXT NOT NULL DEFAULT '',
			covered_up_to  TEXT NOT NULL DEFAULT '',
			summary        TEXT NOT NULL DEFAULT '',
			plaque         TEXT NOT NULL DEFAULT '',
			facts          TEXT NOT NULL DEFAULT '',
			facts_up_to    TEXT NOT NULL DEFAULT '',
			task           TEXT NOT NULL DEFAULT '',
			notes          TEXT NOT NULL DEFAULT '',
			history_tokens INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (model, leaf)
		)`,
		`CREATE TABLE IF NOT EXISTS session_state (
			model          TEXT PRIMARY KEY,
			history_tokens INTEGER NOT NULL DEFAULT 0,
			facts          TEXT NOT NULL DEFAULT '',
			active         TEXT NOT NULL DEFAULT '',
			next_id        INTEGER NOT NULL DEFAULT 0,
			switchable     TEXT NOT NULL DEFAULT ''
		)`,
		`CREATE TABLE IF NOT EXISTS memories (
			model         TEXT NOT NULL,
			id            TEXT NOT NULL,
			kind          TEXT NOT NULL CHECK (kind IN ('profile','decision','knowledge')),
			key           TEXT NOT NULL,
			text          TEXT NOT NULL,
			source        TEXT NOT NULL DEFAULT '',
			created       INTEGER NOT NULL DEFAULT 0,
			updated       INTEGER NOT NULL DEFAULT 0,
			confirmations INTEGER NOT NULL DEFAULT 1,
			hits          INTEGER NOT NULL DEFAULT 0,
			last_hit      INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (model, id),
			UNIQUE (model, kind, key)
		)`,
	}
	for _, stmt := range stmts {
		if _, err := d.db.Exec(stmt); err != nil {
			return fmt.Errorf("storage: schema: %w", err)
		}
	}
	return nil
}

// ── Layers 1–2: session persistence ─────────────────────────────────────────

// SaveSession replaces the stored session of st.Model with the given state in
// one transaction: messages, branches and session pointers.
func (d *DB) SaveSession(st HistoryState) error {
	tx, err := d.db.Begin()
	if err != nil {
		return fmt.Errorf("storage: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // read-only after commit
	model := st.Model
	if _, err := tx.Exec(`DELETE FROM messages WHERE model=?`, model); err != nil {
		return fmt.Errorf("storage: clear messages: %w", err)
	}
	if _, err := tx.Exec(`DELETE FROM branches WHERE model=?`, model); err != nil {
		return fmt.Errorf("storage: clear branches: %w", err)
	}
	if _, err := tx.Exec(`DELETE FROM session_state WHERE model=?`, model); err != nil {
		return fmt.Errorf("storage: clear session_state: %w", err)
	}
	for _, m := range st.Messages {
		tc, err := json.Marshal(m.ToolCalls)
		if err != nil {
			return fmt.Errorf("storage: marshal tool_calls: %w", err)
		}
		if _, err := tx.Exec(
			`INSERT INTO messages (model, id, parent, role, content, tool_calls, tool_call_id, at)
			 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
			model, m.ID, m.Parent, m.Role, m.Content, string(tc), m.ToolCallID, m.At.UnixNano(),
		); err != nil {
			return fmt.Errorf("storage: insert message: %w", err)
		}
	}
	for _, b := range st.Branches {
		frozen, err := json.Marshal(b.Frozen)
		if err != nil {
			return fmt.Errorf("storage: marshal frozen: %w", err)
		}
		if _, err := tx.Exec(
			`INSERT INTO branches (model, leaf, title, fork_up_to, frozen, covered_up_to,
			 summary, plaque, facts, facts_up_to, task, notes, history_tokens)
			 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			model, b.Leaf, b.Title, b.ForkUpTo, string(frozen), b.CoveredUpTo,
			b.Summary, b.Plaque, b.Facts, b.FactsUpTo, b.Task, b.Notes, b.HistoryTokens,
		); err != nil {
			return fmt.Errorf("storage: insert branch: %w", err)
		}
	}
	switchable, err := json.Marshal(st.Switchable)
	if err != nil {
		return fmt.Errorf("storage: marshal switchable: %w", err)
	}
	if _, err := tx.Exec(
		`INSERT INTO session_state (model, history_tokens, facts, active, next_id, switchable)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		model, st.HistoryTokens, st.Facts, st.Active, st.NextID, string(switchable),
	); err != nil {
		return fmt.Errorf("storage: insert session_state: %w", err)
	}
	return tx.Commit()
}

// LoadSession reads one model's persisted session. A missing session yields
// (nil, nil).
func (d *DB) LoadSession(model string) (*HistoryState, error) {
	st := &HistoryState{Model: model}
	var switchable string
	err := d.db.QueryRow(
		`SELECT history_tokens, facts, active, next_id, switchable
		 FROM session_state WHERE model=?`, model,
	).Scan(&st.HistoryTokens, &st.Facts, &st.Active, &st.NextID, &switchable)
	switch {
	case errors.Is(err, sql.ErrNoRows):
		return nil, nil
	case err != nil:
		return nil, fmt.Errorf("storage: load session_state: %w", err)
	}
	_ = json.Unmarshal([]byte(switchable), &st.Switchable)

	rows, err := d.db.Query(
		`SELECT id, parent, role, content, tool_calls, tool_call_id, at
		 FROM messages WHERE model=? ORDER BY rowid`, model)
	if err != nil {
		return nil, fmt.Errorf("storage: load messages: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var r ChatRecord
		var tc string
		var at int64
		if err := rows.Scan(&r.ID, &r.Parent, &r.Role, &r.Content, &tc, &r.ToolCallID, &at); err != nil {
			return nil, fmt.Errorf("storage: scan message: %w", err)
		}
		if tc != "" {
			_ = json.Unmarshal([]byte(tc), &r.ToolCalls)
		}
		r.At = nanoToTime(at)
		st.Messages = append(st.Messages, r)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("storage: iterate messages: %w", err)
	}

	brows, err := d.db.Query(
		`SELECT leaf, title, fork_up_to, frozen, covered_up_to, summary, plaque,
		        facts, facts_up_to, task, notes, history_tokens
		 FROM branches WHERE model=? ORDER BY leaf`, model)
	if err != nil {
		return nil, fmt.Errorf("storage: load branches: %w", err)
	}
	defer brows.Close()
	for brows.Next() {
		var b BranchState
		var frozen string
		if err := brows.Scan(&b.Leaf, &b.Title, &b.ForkUpTo, &frozen, &b.CoveredUpTo,
			&b.Summary, &b.Plaque, &b.Facts, &b.FactsUpTo, &b.Task, &b.Notes, &b.HistoryTokens); err != nil {
			return nil, fmt.Errorf("storage: scan branch: %w", err)
		}
		if frozen != "" {
			_ = json.Unmarshal([]byte(frozen), &b.Frozen)
		}
		st.Branches = append(st.Branches, b)
	}
	if err := brows.Err(); err != nil {
		return nil, fmt.Errorf("storage: iterate branches: %w", err)
	}
	return st, nil
}

// LatestSessionModel returns the model whose conversation has the newest
// message (the restore() semantics of the previous JSON storage). Falls back
// to the most recently saved session_state row when no messages exist. An
// empty database yields "".
func (d *DB) LatestSessionModel() (string, error) {
	var model string
	err := d.db.QueryRow(
		`SELECT model FROM messages WHERE role != 'system'
		 GROUP BY model ORDER BY MAX(at) DESC, model DESC LIMIT 1`).Scan(&model)
	switch {
	case err == nil:
		return model, nil
	case errors.Is(err, sql.ErrNoRows):
		// fall through to session_state
	default:
		return "", fmt.Errorf("storage: latest session: %w", err)
	}
	err = d.db.QueryRow(
		`SELECT model FROM session_state ORDER BY rowid DESC LIMIT 1`).Scan(&model)
	if errors.Is(err, sql.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("storage: latest session state: %w", err)
	}
	return model, nil
}

// DeleteSession removes one model's dialog (layers 1–2) while keeping its
// long-term memories (layer 3).
func (d *DB) DeleteSession(model string) error {
	for _, stmt := range []string{
		`DELETE FROM messages WHERE model=?`,
		`DELETE FROM branches WHERE model=?`,
		`DELETE FROM session_state WHERE model=?`,
	} {
		if _, err := d.db.Exec(stmt, model); err != nil {
			return fmt.Errorf("storage: delete session: %w", err)
		}
	}
	return nil
}

// ── Layer 3: long-term memory ───────────────────────────────────────────────

// ListMemories returns all memory items of a model, oldest first.
func (d *DB) ListMemories(model string) ([]MemoryItem, error) {
	rows, err := d.db.Query(
		`SELECT id, kind, key, text, source, created, updated, confirmations, hits, last_hit
		 FROM memories WHERE model=? ORDER BY created, id`, model)
	if err != nil {
		return nil, fmt.Errorf("storage: list memories: %w", err)
	}
	defer rows.Close()
	var out []MemoryItem
	for rows.Next() {
		it, err := scanMemory(rows.Scan)
		if err != nil {
			return nil, err
		}
		out = append(out, it)
	}
	return out, rows.Err()
}

// scanMemory fills a MemoryItem through the given generic column scanner (a
// *sql.Row or *sql.Rows).
func scanMemory(scan func(dest ...any) error) (MemoryItem, error) {
	var it MemoryItem
	var created, updated, lastHit int64
	if err := scan(&it.ID, &it.Kind, &it.Key, &it.Text, &it.Source,
		&created, &updated, &it.Confirmations, &it.Hits, &lastHit); err != nil {
		return MemoryItem{}, err
	}
	it.Created = nanoToTime(created)
	it.Updated = nanoToTime(updated)
	it.LastHit = nanoToTime(lastHit)
	return it, nil
}

// UpsertMemory stores a memory fact with deduplication by (kind, key). When
// the fact already exists its text/value is refreshed and Confirmations is
// incremented (the fact has just been re-confirmed); otherwise it is inserted
// with confirmations = max(1, it.Confirmations) — explicit user/tool writes
// pass 2 and start out durable. The stored item is returned.
func (d *DB) UpsertMemory(model string, it MemoryItem) (MemoryItem, error) {
	var existing MemoryItem
	var created, updated, lastHit int64
	err := d.db.QueryRow(
		`SELECT id, kind, key, text, source, created, updated, confirmations, hits, last_hit
		 FROM memories WHERE model=? AND kind=? AND key=?`,
		model, it.Kind, it.Key,
	).Scan(&existing.ID, &existing.Kind, &existing.Key, &existing.Text, &existing.Source,
		&created, &updated, &existing.Confirmations, &existing.Hits, &lastHit)
	switch {
	case err == nil:
		existing.Created = nanoToTime(created)
		existing.Updated = nanoToTime(updated)
		existing.LastHit = nanoToTime(lastHit)
		existing.Text = it.Text
		existing.Source = it.Source
		existing.Confirmations++
		existing.Updated = time.Now()
		if _, err := d.db.Exec(
			`UPDATE memories SET text=?, source=?, updated=?, confirmations=? WHERE model=? AND id=?`,
			existing.Text, existing.Source, existing.Updated.UnixNano(), existing.Confirmations, model, existing.ID,
		); err != nil {
			return MemoryItem{}, fmt.Errorf("storage: update memory: %w", err)
		}
		return existing, nil
	case errors.Is(err, sql.ErrNoRows):
		// insert below
	default:
		return MemoryItem{}, fmt.Errorf("storage: find memory: %w", err)
	}

	it.Confirmations = max(1, it.Confirmations)
	if it.Created.IsZero() {
		it.Created = time.Now()
	}
	id := it.ID
	if id == "" {
		id = fmt.Sprintf("k%d", time.Now().UnixNano())
	}
	if _, err := d.db.Exec(
		`INSERT INTO memories (model, id, kind, key, text, source, created, updated, confirmations, hits, last_hit)
		 VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 0, 0)`,
		model, id, it.Kind, it.Key, it.Text, it.Source, it.Created.UnixNano(), it.Confirmations,
	); err != nil {
		return MemoryItem{}, fmt.Errorf("storage: insert memory: %w", err)
	}
	it.ID = id
	it.Updated = time.Time{}
	return it, nil
}

// DeleteAllData wipes every model's dialog (messages, branches, session
// pointers) and the whole long-term layer (memories of all models) — the
// "Удалить все" action clears every memory level at once.
func (d *DB) DeleteAllData() error {
	tx, err := d.db.Begin()
	if err != nil {
		return fmt.Errorf("storage: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // read-only after commit
	for _, stmt := range []string{
		`DELETE FROM messages`,
		`DELETE FROM branches`,
		`DELETE FROM session_state`,
		`DELETE FROM memories`,
	} {
		if _, err := tx.Exec(stmt); err != nil {
			return fmt.Errorf("storage: delete all (%s): %w", stmt, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("storage: commit: %w", err)
	}
	return nil
}

// DeleteMemories removes every memory item of a model (the long-term layer),
// keeping messages and session state untouched.
func (d *DB) DeleteMemories(model string) error {
	if _, err := d.db.Exec(`DELETE FROM memories WHERE model=?`, model); err != nil {
		return fmt.Errorf("storage: delete memories: %w", err)
	}
	return nil
}

// DeleteMemory removes one memory item by id.
func (d *DB) DeleteMemory(model, id string) error {
	if _, err := d.db.Exec(`DELETE FROM memories WHERE model=? AND id=?`, model, id); err != nil {
		return fmt.Errorf("storage: delete memory: %w", err)
	}
	return nil
}

// TouchMemories bumps Hits/LastHit of the items injected into a prompt — the
// usage feedback that later drives capacity eviction.
func (d *DB) TouchMemories(model string, ids []string) error {
	for _, id := range ids {
		if _, err := d.db.Exec(
			`UPDATE memories SET hits=hits+1, last_hit=? WHERE model=? AND id=?`,
			time.Now().UnixNano(), model, id,
		); err != nil {
			return fmt.Errorf("storage: touch memory: %w", err)
		}
	}
	return nil
}

// nanoToTime converts a stored unix-nano timestamp (0 = unset) back to time.
func nanoToTime(n int64) time.Time {
	if n == 0 {
		return time.Time{}
	}
	return time.Unix(0, n)
}
