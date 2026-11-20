// Package storage persists the chat context (short-term and working memory)
// and the long-term memory in a single SQLite database. The three memory
// layers live in separate table groups: layers 1–2 are keyed by (model,
// session), layer 3 (long-term memory) is keyed by session alone — every
// chat tab owns its own long-term memory, independent of the model:
//
//	messages / branches / session_state — the dialog tree and per-branch
//	                                      compression state (layers 1–2);
//	memories                            — durable profile, decisions and
//	                                      knowledge facts of one tab (layer 3).
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
// (the current task's goal, rewritten by compression), Notes (free-form
// artifacts) and TaskFSM (the task state machine, see TaskFSM).
// HistoryTokens is the per-branch history size.
type BranchState struct {
	Leaf          string         `json:"leaf"`
	Title         string         `json:"title,omitempty"`
	ForkUpTo      string         `json:"fork_up_to,omitempty"`
	Frozen        []FoldStage    `json:"frozen,omitempty"`
	CoveredUpTo   string         `json:"covered_up_to,omitempty"`
	Summary       string         `json:"summary,omitempty"`
	Plaque        string         `json:"plaque,omitempty"`
	Facts         string         `json:"facts,omitempty"`
	FactsUpTo     string         `json:"facts_up_to,omitempty"`
	Task          string         `json:"task,omitempty"`
	Notes         string         `json:"notes,omitempty"`
	TaskFSM       *TaskFSM       `json:"task_state,omitempty"`
	Onboarding    *OnboardingFSM `json:"onboarding,omitempty"`
	HistoryTokens int            `json:"history_tokens,omitempty"`
}

// Task step statuses (the plan checklist of a TaskFSM).
const (
	StepPending = "pending"
	StepActive  = "active"
	StepDone    = "done"
)

// TaskStep is one line of the task plan checklist: the step title and its
// status (pending / active / done; at most one step is active).
type TaskStep struct {
	Title  string `json:"title"`
	Status string `json:"status"`
}

// TaskFSM is the task state machine of a branch (part of the working-memory
// layer): the current stage of the planning → execution → validation → done
// cycle (see the strict transition table in server/task.go), the current
// step, the expected action and the optional plan checklist. Nil on a branch
// means "no task in progress".
type TaskFSM struct {
	Stage  string     `json:"stage"`
	Step   string     `json:"step,omitempty"`
	Expect string     `json:"expect,omitempty"`
	Plan   []TaskStep `json:"plan,omitempty"`
}

// OnboardingFSM is the new-user questionnaire state machine of a branch: the
// stage of the interview → confirm → done cycle and the index of the current
// question in the script (see the transition table in server/onboarding.go).
// Nil on a branch means "the questionnaire never ran here"; done is kept as
// a persisted marker so a skipped questionnaire does not restart.
type OnboardingFSM struct {
	Stage string `json:"stage"`
	Step  int    `json:"step,omitempty"`
}

// HistoryState is the persisted shape of one model's chat context: the model
// name, the dialog history token size, the sticky-facts block, the cumulative
// request-token total and the conversation turns (the system prompt is
// reconstructed on load, so it is not stored). Branches carry the per-branch
// compression state and Active names the restored branch.
type HistoryState struct {
	Model         string        `json:"model,omitempty"`
	Session       string        `json:"session,omitempty"`
	Title         string        `json:"title,omitempty"`
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
	SourceCompress   = "compress"   // extracted by facts compression
	SourceUser       = "user"       // explicit user request
	SourceTool       = "tool"       // model's remember() tool call
	SourceOnboarding = "onboarding" // answer of the new-user questionnaire
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
			session      TEXT NOT NULL DEFAULT '',
			id           TEXT NOT NULL,
			parent       TEXT NOT NULL DEFAULT '',
			role         TEXT NOT NULL,
			content      TEXT NOT NULL DEFAULT '',
			tool_calls   TEXT NOT NULL DEFAULT '',
			tool_call_id TEXT NOT NULL DEFAULT '',
			at           INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (model, session, id)
		)`,
		`CREATE TABLE IF NOT EXISTS branches (
			model          TEXT NOT NULL,
			session        TEXT NOT NULL DEFAULT '',
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
			task_state     TEXT NOT NULL DEFAULT '',
			onboarding     TEXT NOT NULL DEFAULT '',
			history_tokens INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (model, session, leaf)
		)`,
		`CREATE TABLE IF NOT EXISTS session_state (
			model          TEXT NOT NULL,
			session        TEXT NOT NULL,
			title          TEXT NOT NULL DEFAULT '',
			history_tokens INTEGER NOT NULL DEFAULT 0,
			facts          TEXT NOT NULL DEFAULT '',
			active         TEXT NOT NULL DEFAULT '',
			next_id        INTEGER NOT NULL DEFAULT 0,
			switchable     TEXT NOT NULL DEFAULT '',
			PRIMARY KEY (model, session)
		)`,
		`CREATE TABLE IF NOT EXISTS memories (
			session       TEXT NOT NULL,
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
			PRIMARY KEY (session, id),
			UNIQUE (session, kind, key)
		)`,
		// Инварианты ассистента: жёсткие правила вне диалога и вне слоёв
		// памяти (глобальные, без ключа сессии). Журнал фиксирует каждое
		// изменение: что → на что → когда (см. storage/invariants.go).
		`CREATE TABLE IF NOT EXISTS invariants (
			id       INTEGER PRIMARY KEY AUTOINCREMENT,
			text     TEXT NOT NULL,
			patterns TEXT NOT NULL DEFAULT '',
			enabled  INTEGER NOT NULL DEFAULT 1,
			created  INTEGER NOT NULL DEFAULT 0
		)`,
		`CREATE TABLE IF NOT EXISTS invariants_log (
			id        INTEGER PRIMARY KEY AUTOINCREMENT,
			action    TEXT NOT NULL,
			old_value TEXT NOT NULL DEFAULT '',
			new_value TEXT NOT NULL DEFAULT '',
			source    TEXT NOT NULL DEFAULT '',
			at        INTEGER NOT NULL DEFAULT 0
		)`,
	}
	if err := d.migrateMemories(); err != nil {
		return fmt.Errorf("storage: migrate memories: %w", err)
	}
	if err := d.migrateBranches(); err != nil {
		return fmt.Errorf("storage: migrate branches: %w", err)
	}
	for _, stmt := range stmts {
		if _, err := d.db.Exec(stmt); err != nil {
			return fmt.Errorf("storage: schema: %w", err)
		}
	}
	return nil
}

// migrateMemories drops a pre-refactor memories table (keyed by model) so the
// session-keyed schema is created fresh: the old per-model rows have no
// session to belong to, and the layer is rebuilt as per-tab memory.
func (d *DB) migrateMemories() error {
	var hasTable, hasSession int
	if err := d.db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memories'`).Scan(&hasTable); err != nil {
		return fmt.Errorf("inspect memories: %w", err)
	}
	if hasTable == 0 {
		return nil
	}
	if err := d.db.QueryRow(`SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='session'`).Scan(&hasSession); err != nil {
		return fmt.Errorf("inspect memories columns: %w", err)
	}
	if hasSession > 0 {
		return nil
	}
	if _, err := d.db.Exec(`DROP TABLE memories`); err != nil {
		return fmt.Errorf("drop legacy memories: %w", err)
	}
	return nil
}

// migrateBranches adds the task_state (the task FSM) and onboarding (the
// questionnaire FSM) columns — JSON blobs — to databases created before
// these state machines existed.
func (d *DB) migrateBranches() error {
	var hasTable int
	if err := d.db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='branches'`).Scan(&hasTable); err != nil {
		return fmt.Errorf("inspect branches: %w", err)
	}
	if hasTable == 0 {
		return nil // the schema statement below creates it with the column
	}
	for _, col := range []string{"task_state", "onboarding"} {
		var hasColumn int
		if err := d.db.QueryRow(`SELECT COUNT(*) FROM pragma_table_info('branches') WHERE name=?`, col).Scan(&hasColumn); err != nil {
			return fmt.Errorf("inspect branches columns: %w", err)
		}
		if hasColumn > 0 {
			continue
		}
		if _, err := d.db.Exec(`ALTER TABLE branches ADD COLUMN ` + col + ` TEXT NOT NULL DEFAULT ''`); err != nil {
			return fmt.Errorf("add branches.%s: %w", col, err)
		}
	}
	return nil
}

// ── Layers 1–2: session persistence ─────────────────────────────────────────

// SaveSession replaces the stored session identified by (st.Model, st.Session)
// with the given state in one transaction: messages, branches and session
// pointers. An empty Session maps to the legacy default session, so old
// single-context data is preserved.
func (d *DB) SaveSession(st HistoryState) error {
	tx, err := d.db.Begin()
	if err != nil {
		return fmt.Errorf("storage: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // read-only after commit
	model, session := st.Model, st.Session
	if _, err := tx.Exec(`DELETE FROM messages WHERE model=? AND session=?`, model, session); err != nil {
		return fmt.Errorf("storage: clear messages: %w", err)
	}
	if _, err := tx.Exec(`DELETE FROM branches WHERE model=? AND session=?`, model, session); err != nil {
		return fmt.Errorf("storage: clear branches: %w", err)
	}
	if _, err := tx.Exec(`DELETE FROM session_state WHERE model=? AND session=?`, model, session); err != nil {
		return fmt.Errorf("storage: clear session_state: %w", err)
	}
	for _, m := range st.Messages {
		tc, err := json.Marshal(m.ToolCalls)
		if err != nil {
			return fmt.Errorf("storage: marshal tool_calls: %w", err)
		}
		if _, err := tx.Exec(
			`INSERT INTO messages (model, session, id, parent, role, content, tool_calls, tool_call_id, at)
			 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			model, session, m.ID, m.Parent, m.Role, m.Content, string(tc), m.ToolCallID, m.At.UnixNano(),
		); err != nil {
			return fmt.Errorf("storage: insert message: %w", err)
		}
	}
	for _, b := range st.Branches {
		frozen, err := json.Marshal(b.Frozen)
		if err != nil {
			return fmt.Errorf("storage: marshal frozen: %w", err)
		}
		taskState := ""
		if b.TaskFSM != nil {
			// Пустой FSM ({"stage":""}) в базу не пишем: считаем его
			// отсутствующей задачей.
			if !(b.TaskFSM.Stage == "" && b.TaskFSM.Step == "" &&
				b.TaskFSM.Expect == "" && len(b.TaskFSM.Plan) == 0) {
				tsJSON, err := json.Marshal(b.TaskFSM)
				if err != nil {
					return fmt.Errorf("storage: marshal task_state: %w", err)
				}
				taskState = string(tsJSON)
			}
		}
		onboarding := ""
		if b.Onboarding != nil && b.Onboarding.Stage != "" {
			// Пустой этап в базу не пишем, а done сохраняется: это маркер
			// «анкета пройдена/отклонена», без него skip-all перезапустил
			// бы анкету на следующем сообщении (профиль-то пуст).
			onbJSON, err := json.Marshal(b.Onboarding)
			if err != nil {
				return fmt.Errorf("storage: marshal onboarding: %w", err)
			}
			onboarding = string(onbJSON)
		}
		if _, err := tx.Exec(
			`INSERT INTO branches (model, session, leaf, title, fork_up_to, frozen, covered_up_to,
			 summary, plaque, facts, facts_up_to, task, notes, task_state, onboarding, history_tokens)
			 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			model, session, b.Leaf, b.Title, b.ForkUpTo, string(frozen), b.CoveredUpTo,
			b.Summary, b.Plaque, b.Facts, b.FactsUpTo, b.Task, b.Notes, taskState, onboarding, b.HistoryTokens,
		); err != nil {
			return fmt.Errorf("storage: insert branch: %w", err)
		}
	}
	switchable, err := json.Marshal(st.Switchable)
	if err != nil {
		return fmt.Errorf("storage: marshal switchable: %w", err)
	}
	title := st.Title
	if title == "" {
		title = session
	}
	if _, err := tx.Exec(
		`INSERT INTO session_state (model, session, title, history_tokens, facts, active, next_id, switchable)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		model, session, title, st.HistoryTokens, st.Facts, st.Active, st.NextID, string(switchable),
	); err != nil {
		return fmt.Errorf("storage: insert session_state: %w", err)
	}
	return tx.Commit()
}

// LoadSession reads one (model, session)'s persisted state. A missing session
// yields (nil, nil).
func (d *DB) LoadSession(model, session string) (*HistoryState, error) {
	st := &HistoryState{Model: model, Session: session}
	var switchable string
	err := d.db.QueryRow(
		`SELECT title, history_tokens, facts, active, next_id, switchable
		 FROM session_state WHERE model=? AND session=?`, model, session,
	).Scan(&st.Title, &st.HistoryTokens, &st.Facts, &st.Active, &st.NextID, &switchable)
	switch {
	case errors.Is(err, sql.ErrNoRows):
		return nil, nil
	case err != nil:
		return nil, fmt.Errorf("storage: load session_state: %w", err)
	}
	_ = json.Unmarshal([]byte(switchable), &st.Switchable)

	rows, err := d.db.Query(
		`SELECT id, parent, role, content, tool_calls, tool_call_id, at
		 FROM messages WHERE model=? AND session=? ORDER BY rowid`, model, session)
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
		        facts, facts_up_to, task, notes, task_state, onboarding, history_tokens
		 FROM branches WHERE model=? AND session=? ORDER BY leaf`, model, session)
	if err != nil {
		return nil, fmt.Errorf("storage: load branches: %w", err)
	}
	defer brows.Close()
	for brows.Next() {
		var b BranchState
		var frozen, taskState, onboarding string
		if err := brows.Scan(&b.Leaf, &b.Title, &b.ForkUpTo, &frozen, &b.CoveredUpTo,
			&b.Summary, &b.Plaque, &b.Facts, &b.FactsUpTo, &b.Task, &b.Notes, &taskState, &onboarding, &b.HistoryTokens); err != nil {
			return nil, fmt.Errorf("storage: scan branch: %w", err)
		}
		if frozen != "" {
			_ = json.Unmarshal([]byte(frozen), &b.Frozen)
		}
		if taskState != "" {
			_ = json.Unmarshal([]byte(taskState), &b.TaskFSM)
			// Пустой FSM ({"stage":""} из старых версий) считаем отсутствующим.
			if b.TaskFSM != nil && b.TaskFSM.Stage == "" && b.TaskFSM.Step == "" &&
				b.TaskFSM.Expect == "" && len(b.TaskFSM.Plan) == 0 {
				b.TaskFSM = nil
			}
		}
		if onboarding != "" {
			_ = json.Unmarshal([]byte(onboarding), &b.Onboarding)
			// Пустой этап ({"stage":""}) считаем отсутствующей анкетой.
			if b.Onboarding != nil && b.Onboarding.Stage == "" {
				b.Onboarding = nil
			}
		}
		st.Branches = append(st.Branches, b)
	}
	if err := brows.Err(); err != nil {
		return nil, fmt.Errorf("storage: iterate branches: %w", err)
	}
	return st, nil
}

// LatestSession returns the (model, session) pair of the conversation with the
// newest non-system message (the restore() semantics of the previous storage).
// Falls back to the most recently saved session_state row when no messages
// exist. An empty database yields ("", "").
func (d *DB) LatestSession() (model, session string, err error) {
	err = d.db.QueryRow(
		`SELECT model, session FROM messages
		 WHERE role != 'system' AND session != ''
		 GROUP BY model, session ORDER BY MAX(at) DESC LIMIT 1`).Scan(&model, &session)
	switch {
	case err == nil:
		return model, session, nil
	case errors.Is(err, sql.ErrNoRows):
		// fall through to session_state
	default:
		return "", "", fmt.Errorf("storage: latest session: %w", err)
	}
	err = d.db.QueryRow(
		`SELECT model, session FROM session_state ORDER BY rowid DESC LIMIT 1`).Scan(&model, &session)
	if errors.Is(err, sql.ErrNoRows) {
		return "", "", nil
	}
	if err != nil {
		return "", "", fmt.Errorf("storage: latest session state: %w", err)
	}
	return model, session, nil
}

// SessionMeta is the lightweight, tab-bar description of one persisted chat
// session (the full conversation lives in messages/branches).
type SessionMeta struct {
	Model   string `json:"model"`
	Session string `json:"session"`
	Title   string `json:"title"`
}

// ListSessions returns every persisted chat session's metadata, newest first
// (by session_state insertion order — the order tabs were created).
func (d *DB) ListSessions() ([]SessionMeta, error) {
	rows, err := d.db.Query(
		`SELECT model, session, title FROM session_state ORDER BY rowid DESC`)
	if err != nil {
		return nil, fmt.Errorf("storage: list sessions: %w", err)
	}
	defer rows.Close()
	var out []SessionMeta
	for rows.Next() {
		var m SessionMeta
		if err := rows.Scan(&m.Model, &m.Session, &m.Title); err != nil {
			return nil, fmt.Errorf("storage: scan session: %w", err)
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// DeleteSession removes one (model, session) dialog (layers 1–2) while keeping
// its long-term memories (layer 3).
func (d *DB) DeleteSession(model, session string) error {
	for _, stmt := range []string{
		`DELETE FROM messages WHERE model=? AND session=?`,
		`DELETE FROM branches WHERE model=? AND session=?`,
		`DELETE FROM session_state WHERE model=? AND session=?`,
	} {
		if _, err := d.db.Exec(stmt, model, session); err != nil {
			return fmt.Errorf("storage: delete session: %w", err)
		}
	}
	return nil
}

// DeleteSessionAll removes one (model, session) dialog together with its
// long-term memories (all three layers in one transaction) — closing a chat
// tab drops its whole context, not just the conversation.
func (d *DB) DeleteSessionAll(model, session string) error {
	tx, err := d.db.Begin()
	if err != nil {
		return fmt.Errorf("storage: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // read-only after commit
	for _, stmt := range []string{
		`DELETE FROM messages WHERE model=? AND session=?`,
		`DELETE FROM branches WHERE model=? AND session=?`,
		`DELETE FROM session_state WHERE model=? AND session=?`,
	} {
		if _, err := tx.Exec(stmt, model, session); err != nil {
			return fmt.Errorf("storage: delete session: %w", err)
		}
	}
	if _, err := tx.Exec(`DELETE FROM memories WHERE session=?`, session); err != nil {
		return fmt.Errorf("storage: delete session memories: %w", err)
	}
	return tx.Commit()
}

// RenameSession updates the tab-bar title of one persisted chat session.
func (d *DB) RenameSession(model, session, title string) error {
	if _, err := d.db.Exec(
		`UPDATE session_state SET title=? WHERE model=? AND session=?`,
		title, model, session,
	); err != nil {
		return fmt.Errorf("storage: rename session: %w", err)
	}
	return nil
}

// ── Layer 3: long-term memory ───────────────────────────────────────────────

// ListMemories returns all memory items of one session (tab), oldest first.
func (d *DB) ListMemories(session string) ([]MemoryItem, error) {
	rows, err := d.db.Query(
		`SELECT id, kind, key, text, source, created, updated, confirmations, hits, last_hit
		 FROM memories WHERE session=? ORDER BY created, id`, session)
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
func (d *DB) UpsertMemory(session string, it MemoryItem) (MemoryItem, error) {
	var existing MemoryItem
	var created, updated, lastHit int64
	err := d.db.QueryRow(
		`SELECT id, kind, key, text, source, created, updated, confirmations, hits, last_hit
		 FROM memories WHERE session=? AND kind=? AND key=?`,
		session, it.Kind, it.Key,
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
			`UPDATE memories SET text=?, source=?, updated=?, confirmations=? WHERE session=? AND id=?`,
			existing.Text, existing.Source, existing.Updated.UnixNano(), existing.Confirmations, session, existing.ID,
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
		`INSERT INTO memories (session, id, kind, key, text, source, created, updated, confirmations, hits, last_hit)
		 VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 0, 0)`,
		session, id, it.Kind, it.Key, it.Text, it.Source, it.Created.UnixNano(), it.Confirmations,
	); err != nil {
		return MemoryItem{}, fmt.Errorf("storage: insert memory: %w", err)
	}
	it.ID = id
	it.Updated = time.Time{}
	return it, nil
}

// DeleteAllData wipes every tab's dialog (messages, branches, session
// pointers) and the whole long-term layer (memories of all sessions).
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

// DeleteMemories removes every memory item of one session (tab), keeping
// messages and session state untouched.
func (d *DB) DeleteMemories(session string) error {
	if _, err := d.db.Exec(`DELETE FROM memories WHERE session=?`, session); err != nil {
		return fmt.Errorf("storage: delete memories: %w", err)
	}
	return nil
}

// DeleteMemory removes one memory item of a session by id.
func (d *DB) DeleteMemory(session, id string) error {
	if _, err := d.db.Exec(`DELETE FROM memories WHERE session=? AND id=?`, session, id); err != nil {
		return fmt.Errorf("storage: delete memory: %w", err)
	}
	return nil
}

// TouchMemories bumps Hits/LastHit of the items injected into a prompt — the
// usage feedback that later drives capacity eviction.
func (d *DB) TouchMemories(session string, ids []string) error {
	for _, id := range ids {
		if _, err := d.db.Exec(
			`UPDATE memories SET hits=hits+1, last_hit=? WHERE session=? AND id=?`,
			time.Now().UnixNano(), session, id,
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
