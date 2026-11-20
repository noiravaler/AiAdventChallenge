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

// Memory policies (the life rule of a fact, column policy): a fact either
// keeps one actual value (later writes REPLACE it — срок, статус, версия) or
// accumulates as a list of entries (accepted decisions, change history).
// The policy is normalized at write time — stored rows always carry it.
const (
	PolicyLatest = "latest" // перезапись: хранится одно актуальное значение
	PolicyAppend = "append" // список: записи накапливаются
)

// DefaultMemoryPolicy returns the policy of a fact with an unspecified one:
// profile attributes and knowledge overwrite (a new value supersedes the
// old), decisions accumulate as a list.
func DefaultMemoryPolicy(kind string) string {
	if kind == KindDecision {
		return PolicyAppend
	}
	return PolicyLatest
}

// MemorySources — how a memory item came to be.
const (
	SourceCompress   = "compress"   // extracted by facts compression
	SourceUser       = "user"       // explicit user request
	SourceTool       = "tool"       // model's remember() tool call
	SourceOnboarding = "onboarding" // answer of the new-user questionnaire
	SourceEvict      = "evict"      // dropped by the capacity eviction
	SourceSystem     = "system"     // automatic service action (ttl sweep)
)

// MemoryItem is one long-term memory record: a single «ключ: значение» fact
// with its kind, origin and usage statistics. Confirmations counts how often
// the fact survived re-extraction during compression (facts promoted by
// compression start at 1 and stabilize at 2+); explicit user/tool writes start
// at 2. Hits counts prompt injections. Created is the first-write time,
// Updated — the last write (each fact carries its write time). Policy is the
// life rule (latest — the value is replaced by newer writes, append — entries
// accumulate as a list); Ephemeral marks facts deleted when the tab's task
// completes (ttl = "until the task is done").
type MemoryItem struct {
	ID            string    `json:"id"`
	Kind          string    `json:"kind"`
	Key           string    `json:"key"`
	Text          string    `json:"text"`
	Source        string    `json:"source"`
	Policy        string    `json:"policy,omitempty"`
	Ephemeral     bool      `json:"ephemeral,omitempty"`
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
		// Долговременная память (слой 3). Policy — правило жизни факта
		// (latest — перезапись значения, append — накопление списка);
		// уникальность (session, kind, key) держит частичный индекс только
		// для latest — записи списков (append) свободно накапливаются.
		`CREATE TABLE IF NOT EXISTS memories (
			session       TEXT NOT NULL,
			id            TEXT NOT NULL,
			kind          TEXT NOT NULL CHECK (kind IN ('profile','decision','knowledge')),
			key           TEXT NOT NULL,
			text          TEXT NOT NULL,
			source        TEXT NOT NULL DEFAULT '',
			policy        TEXT NOT NULL DEFAULT '',
			ephemeral     INTEGER NOT NULL DEFAULT 0,
			created       INTEGER NOT NULL DEFAULT 0,
			updated       INTEGER NOT NULL DEFAULT 0,
			confirmations INTEGER NOT NULL DEFAULT 1,
			hits          INTEGER NOT NULL DEFAULT 0,
			last_hit      INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (session, id)
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
		// Журнал изменений долговременной памяти: что было → что стало,
		// когда, источник (см. storage/memorylog.go).
		`CREATE TABLE IF NOT EXISTS memories_log (
			id        INTEGER PRIMARY KEY AUTOINCREMENT,
			session   TEXT NOT NULL,
			memory_id TEXT NOT NULL DEFAULT '',
			action    TEXT NOT NULL,
			kind      TEXT NOT NULL DEFAULT '',
			key       TEXT NOT NULL DEFAULT '',
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
	if err := d.migrateMemoriesPolicy(); err != nil {
		return fmt.Errorf("storage: migrate memory policies: %w", err)
	}
	// Уникальность (session, kind, key) — только для фактов-перезаписей;
	// append-записи (решения, история) накапливаются без ограничений.
	if _, err := d.db.Exec(
		`CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_latest
		 ON memories (session, kind, key) WHERE policy IN ('', 'latest')`,
	); err != nil {
		return fmt.Errorf("storage: memories unique index: %w", err)
	}
	return nil
}

// migrateMemoriesPolicy rebuilds a pre-policy memories table: adds the policy
// and ephemeral columns, drops the table-level UNIQUE (it now lives in the
// partial index) and assigns default policies to the existing rows
// (decision → append, profile/knowledge → latest). Data is copied as-is.
func (d *DB) migrateMemoriesPolicy() error {
	var hasTable int
	if err := d.db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memories'`).Scan(&hasTable); err != nil {
		return fmt.Errorf("inspect memories: %w", err)
	}
	if hasTable == 0 {
		return nil // the schema statement below creates it with the columns
	}
	var hasPolicy int
	if err := d.db.QueryRow(`SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name='policy'`).Scan(&hasPolicy); err != nil {
		return fmt.Errorf("inspect memories columns: %w", err)
	}
	if hasPolicy > 0 {
		return nil // already migrated
	}
	tx, err := d.db.Begin()
	if err != nil {
		return fmt.Errorf("begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // read-only after commit
	for _, stmt := range []string{
		`ALTER TABLE memories RENAME TO memories_legacy`,
		`CREATE TABLE memories (
			session       TEXT NOT NULL,
			id            TEXT NOT NULL,
			kind          TEXT NOT NULL CHECK (kind IN ('profile','decision','knowledge')),
			key           TEXT NOT NULL,
			text          TEXT NOT NULL,
			source        TEXT NOT NULL DEFAULT '',
			policy        TEXT NOT NULL DEFAULT '',
			ephemeral     INTEGER NOT NULL DEFAULT 0,
			created       INTEGER NOT NULL DEFAULT 0,
			updated       INTEGER NOT NULL DEFAULT 0,
			confirmations INTEGER NOT NULL DEFAULT 1,
			hits          INTEGER NOT NULL DEFAULT 0,
			last_hit      INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (session, id)
		)`,
		`INSERT INTO memories
			(session, id, kind, key, text, source, policy, ephemeral,
			 created, updated, confirmations, hits, last_hit)
		 SELECT session, id, kind, key, text, source,
			CASE kind WHEN 'decision' THEN 'append' ELSE 'latest' END,
			0, created, updated, confirmations, hits, last_hit
		 FROM memories_legacy`,
		`DROP TABLE memories_legacy`,
	} {
		if _, err := tx.Exec(stmt); err != nil {
			return fmt.Errorf("storage: rebuild memories: %w", err)
		}
	}
	return tx.Commit()
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
	// Закрытие вкладки удаляет и её журнал памяти: контекст вкладки исчезает
	// целиком, а journal-строки без сессии-владельца были бы мусором.
	if _, err := tx.Exec(`DELETE FROM memories_log WHERE session=?`, session); err != nil {
		return fmt.Errorf("storage: delete session memory log: %w", err)
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
		`SELECT id, kind, key, text, source, policy, ephemeral, created, updated, confirmations, hits, last_hit
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
	var ephemeral int
	var created, updated, lastHit int64
	if err := scan(&it.ID, &it.Kind, &it.Key, &it.Text, &it.Source, &it.Policy, &ephemeral,
		&created, &updated, &it.Confirmations, &it.Hits, &lastHit); err != nil {
		return MemoryItem{}, err
	}
	it.Ephemeral = ephemeral != 0
	it.Created = nanoToTime(created)
	it.Updated = nanoToTime(updated)
	it.LastHit = nanoToTime(lastHit)
	return it, nil
}

// prepareMemoryItem normalizes a new fact before it lands in the table: the
// policy defaults to the kind's rule and is stored explicitly, the id falls
// back to a unique "k<nano>", the first-write time is now, confirmations
// start at max(1, requested) — explicit writes pass 2 and start out durable.
func prepareMemoryItem(it MemoryItem) MemoryItem {
	if it.Policy != PolicyLatest && it.Policy != PolicyAppend {
		it.Policy = DefaultMemoryPolicy(it.Kind)
	}
	if it.ID == "" {
		it.ID = fmt.Sprintf("k%d", time.Now().UnixNano())
	}
	if it.Created.IsZero() {
		it.Created = time.Now()
	}
	it.Confirmations = max(1, it.Confirmations)
	return it
}

// InsertMemory stores a brand-new fact (a first write or an append-list
// entry) and logs it in the same transaction. The action selects the journal
// record (insert or append); the stored item is returned.
func (d *DB) InsertMemory(session string, it MemoryItem, action, source string) (MemoryItem, error) {
	it = prepareMemoryItem(it)
	tx, err := d.db.Begin()
	if err != nil {
		return MemoryItem{}, fmt.Errorf("storage: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // read-only after commit
	if _, err := tx.Exec(
		`INSERT INTO memories (session, id, kind, key, text, source, policy, ephemeral, created, updated, confirmations, hits, last_hit)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 0, 0)`,
		session, it.ID, it.Kind, it.Key, it.Text, it.Source, it.Policy, boolToInt(it.Ephemeral),
		it.Created.UnixNano(), it.Confirmations,
	); err != nil {
		return MemoryItem{}, fmt.Errorf("storage: insert memory: %w", err)
	}
	if action != MemLogInsert && action != MemLogAppend {
		action = MemLogInsert
	}
	if err := insertMemoryLog(tx, session, it.ID, action, it.Kind, it.Key, "", it.Text, source); err != nil {
		return MemoryItem{}, err
	}
	if err := tx.Commit(); err != nil {
		return MemoryItem{}, fmt.Errorf("storage: commit: %w", err)
	}
	return it, nil
}

// ReplaceMemory supersedes the value of a latest-policy fact: the old text is
// logged (what was → what became) in the same transaction, the row keeps its
// id, kind, key, policy, created (the first-write time) and lifecycle flags;
// confirmations restart from the incoming fact (a changed value is a new
// candidate unless written explicitly). The stored item is returned.
func (d *DB) ReplaceMemory(session, id string, it MemoryItem, source string) (MemoryItem, error) {
	tx, err := d.db.Begin()
	if err != nil {
		return MemoryItem{}, fmt.Errorf("storage: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // read-only after commit
	var oldText string
	err = tx.QueryRow(
		`SELECT text FROM memories WHERE session=? AND id=?`, session, id,
	).Scan(&oldText)
	switch {
	case err == nil:
	case errors.Is(err, sql.ErrNoRows):
		return MemoryItem{}, fmt.Errorf("storage: запись %q не найдена", id)
	default:
		return MemoryItem{}, fmt.Errorf("storage: find memory: %w", err)
	}
	updated := time.Now()
	confirmations := max(1, it.Confirmations)
	if _, err := tx.Exec(
		`UPDATE memories SET text=?, source=?, updated=?, confirmations=? WHERE session=? AND id=?`,
		it.Text, it.Source, updated.UnixNano(), confirmations, session, id,
	); err != nil {
		return MemoryItem{}, fmt.Errorf("storage: replace memory: %w", err)
	}
	if err := insertMemoryLog(tx, session, id, MemLogReplace, it.Kind, it.Key, oldText, it.Text, source); err != nil {
		return MemoryItem{}, err
	}
	if err := tx.Commit(); err != nil {
		return MemoryItem{}, fmt.Errorf("storage: commit: %w", err)
	}
	it.ID = id
	it.Created = time.Time{} // not refreshed here; the cache keeps the original
	it.Updated = updated
	it.Confirmations = confirmations
	return it, nil
}

// ReconfirmMemory bumps the confirmations counter of a re-extracted fact (the
// text did not change, so nothing is logged). The stored item is returned.
func (d *DB) ReconfirmMemory(session, id string) (MemoryItem, error) {
	tx, err := d.db.Begin()
	if err != nil {
		return MemoryItem{}, fmt.Errorf("storage: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // read-only after commit
	it, err := scanMemory(tx.QueryRow(
		`SELECT id, kind, key, text, source, policy, ephemeral, created, updated, confirmations, hits, last_hit
		 FROM memories WHERE session=? AND id=?`, session, id).Scan)
	switch {
	case err == nil:
	case errors.Is(err, sql.ErrNoRows):
		return MemoryItem{}, fmt.Errorf("storage: запись %q не найдена", id)
	default:
		return MemoryItem{}, fmt.Errorf("storage: find memory: %w", err)
	}
	it.Confirmations++
	it.Updated = time.Now()
	if _, err := tx.Exec(
		`UPDATE memories SET confirmations=?, updated=? WHERE session=? AND id=?`,
		it.Confirmations, it.Updated.UnixNano(), session, id,
	); err != nil {
		return MemoryItem{}, fmt.Errorf("storage: reconfirm memory: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return MemoryItem{}, fmt.Errorf("storage: commit: %w", err)
	}
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
// messages and session state untouched. The wipe is logged as one reset
// entry (the journal itself survives — it is the record of what was there).
func (d *DB) DeleteMemories(session string) error {
	tx, err := d.db.Begin()
	if err != nil {
		return fmt.Errorf("storage: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // read-only after commit
	var n int
	if err := tx.QueryRow(`SELECT COUNT(*) FROM memories WHERE session=?`, session).Scan(&n); err != nil {
		return fmt.Errorf("storage: count memories: %w", err)
	}
	if _, err := tx.Exec(`DELETE FROM memories WHERE session=?`, session); err != nil {
		return fmt.Errorf("storage: delete memories: %w", err)
	}
	if n > 0 {
		if err := insertMemoryLog(tx, session, "", MemLogReset, "", "", fmt.Sprintf("%d записей", n), "", "ui"); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// DeleteMemory removes one memory item of a session by id and logs the
// deletion (old_value keeps the text) in the same transaction.
func (d *DB) DeleteMemory(session, id, source string) error {
	tx, err := d.db.Begin()
	if err != nil {
		return fmt.Errorf("storage: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // read-only after commit
	var kind, key, text string
	err = tx.QueryRow(
		`SELECT kind, key, text FROM memories WHERE session=? AND id=?`, session, id,
	).Scan(&kind, &key, &text)
	switch {
	case err == nil:
	case errors.Is(err, sql.ErrNoRows):
		return fmt.Errorf("storage: запись %q не найдена", id)
	default:
		return fmt.Errorf("storage: find memory: %w", err)
	}
	if _, err := tx.Exec(`DELETE FROM memories WHERE session=? AND id=?`, session, id); err != nil {
		return fmt.Errorf("storage: delete memory: %w", err)
	}
	if err := insertMemoryLog(tx, session, id, MemLogDelete, kind, key, text, "", source); err != nil {
		return err
	}
	return tx.Commit()
}

// ExpireEphemeralMemories removes every ephemeral fact (ttl = "until the task
// is done") of a session — the task-completion sweep — and logs each removal
// (action=expire) in the same transaction. The removed items are returned so
// the caller can report what has just expired.
func (d *DB) ExpireEphemeralMemories(session, source string) ([]MemoryItem, error) {
	tx, err := d.db.Begin()
	if err != nil {
		return nil, fmt.Errorf("storage: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // read-only after commit
	rows, err := tx.Query(
		`SELECT id, kind, key, text, source, policy, ephemeral, created, updated, confirmations, hits, last_hit
		 FROM memories WHERE session=? AND ephemeral=1`, session)
	if err != nil {
		return nil, fmt.Errorf("storage: select ephemeral memories: %w", err)
	}
	var expired []MemoryItem
	for rows.Next() {
		it, err := scanMemory(rows.Scan)
		if err != nil {
			rows.Close()
			return nil, err
		}
		expired = append(expired, it)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("storage: iterate ephemeral memories: %w", err)
	}
	for _, it := range expired {
		if _, err := tx.Exec(`DELETE FROM memories WHERE session=? AND id=?`, session, it.ID); err != nil {
			return nil, fmt.Errorf("storage: expire memory: %w", err)
		}
		if err := insertMemoryLog(tx, session, it.ID, MemLogExpire, it.Kind, it.Key, it.Text, "", source); err != nil {
			return nil, err
		}
	}
	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("storage: commit: %w", err)
	}
	return expired, nil
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
