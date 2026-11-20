package storage

import (
	"database/sql"
	"path/filepath"
	"testing"
	"time"

	"ai-chat-go/api"

	_ "modernc.org/sqlite"
)

func openTestDB(t *testing.T) *DB {
	t.Helper()
	d, err := Open(filepath.Join(t.TempDir(), "chat.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { d.Close() })
	return d
}

// TestSessionRoundtrip verifies that a saved session (messages with tree
// links and tool calls, branches with all working-memory fields, session
// pointers) restores byte-for-byte, that LatestSessionModel picks the model
// with the newest message and that DeleteSession keeps the memories.
func TestSessionRoundtrip(t *testing.T) {
	d := openTestDB(t)
	at := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	st := HistoryState{
		Model:      "test-model",
		Active:     "m3",
		NextID:     3,
		Switchable: []string{"m2"},
		Messages: []ChatRecord{
			{ID: "m2", Role: "user", Content: "вопрос", At: at},
			{ID: "m3", Parent: "m2", Role: "assistant", Content: "ответ", At: at.Add(time.Second)},
			{ID: "m4", Parent: "m3", Role: "tool", ToolCallID: "call_1", Content: "4", At: at.Add(2 * time.Second),
				ToolCalls: nil},
			{ID: "m5", Parent: "m4", Role: "assistant", Content: "итог", At: at.Add(3 * time.Second),
				ToolCalls: []api.ToolCall{{ID: "call_1", Type: "function",
					Function: api.ToolCallFunction{Name: "roll_dice", Arguments: "{}"}}}},
		},
		Branches: []BranchState{{
			Leaf: "m5", Title: "Тест", ForkUpTo: "m2", CoveredUpTo: "m2",
			Summary: "сводка", Plaque: "плашка", Facts: "имя: Аня", FactsUpTo: "m2",
			Task: "написать код", Notes: "артефакт", HistoryTokens: 7,
			TaskFSM: &TaskFSM{Stage: "execution", Step: "модуль", Expect: "тесты",
				Plan: []TaskStep{{Title: "a", Status: StepDone}, {Title: "b", Status: StepActive}}},
			Frozen: []FoldStage{{CoveredUpTo: "m1", Summary: "старая сводка"}},
		}},
	}
	if err := d.SaveSession(st); err != nil {
		t.Fatalf("SaveSession: %v", err)
	}

	got, err := d.LoadSession("test-model", "")
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if got == nil {
		t.Fatal("LoadSession = nil, want a session")
	}
	if got.Model != "test-model" || got.Active != "m3" || got.NextID != 3 {
		t.Fatalf("session header = %+v", got)
	}
	if len(got.Messages) != 4 || got.Messages[1].Parent != "m2" {
		t.Fatalf("messages = %+v", got.Messages)
	}
	if len(got.Messages[3].ToolCalls) != 1 || got.Messages[3].ToolCalls[0].Function.Name != "roll_dice" {
		t.Fatalf("tool_calls lost: %+v", got.Messages[3].ToolCalls)
	}
	if len(got.Switchable) != 1 || got.Switchable[0] != "m2" {
		t.Fatalf("switchable = %v", got.Switchable)
	}
	b := got.Branches[0]
	if b.Task != "написать код" || b.Notes != "артефакт" || b.Facts != "имя: Аня" || b.HistoryTokens != 7 {
		t.Fatalf("branch state = %+v", b)
	}
	if len(b.Frozen) != 1 || b.Frozen[0].CoveredUpTo != "m1" {
		t.Fatalf("frozen = %+v", b.Frozen)
	}
	if b.TaskFSM == nil || b.TaskFSM.Stage != "execution" || b.TaskFSM.Step != "модуль" ||
		b.TaskFSM.Expect != "тесты" || len(b.TaskFSM.Plan) != 2 ||
		b.TaskFSM.Plan[0].Status != StepDone || b.TaskFSM.Plan[1].Status != StepActive {
		t.Fatalf("task_state = %+v, want the execution FSM with the 2-step plan", b.TaskFSM)
	}

	// Missing session → nil, no error.
	if missing, err := d.LoadSession("other", ""); err != nil || missing != nil {
		t.Fatalf("LoadSession(other) = %v, %v; want nil, nil", missing, err)
	}

	// Newest message wins across models.
	other := HistoryState{Model: "newest", Messages: []ChatRecord{
		{ID: "m9", Role: "user", Content: "новее", At: at.Add(time.Hour)},
	}}
	if err := d.SaveSession(other); err != nil {
		t.Fatalf("SaveSession(other): %v", err)
	}
	model, session, err := d.LatestSession()
	if err != nil {
		t.Fatalf("LatestSession: %v", err)
	}
	if model != "newest" {
		t.Fatalf("LatestSession = %q, want newest", model)
	}
	if session != "" {
		t.Fatalf("LatestSession session = %q, want \"\"", session)
	}

	// DeleteSession wipes layers 1–2 only.
	if err := d.DeleteSession("test-model", ""); err != nil {
		t.Fatalf("DeleteSession: %v", err)
	}
	if gone, err := d.LoadSession("test-model", ""); err != nil || gone != nil {
		t.Fatalf("session after delete = %v, %v; want nil, nil", gone, err)
	}
	if other, err := d.LoadSession("newest", ""); err != nil || other == nil {
		t.Fatalf("other session deleted too: %v, %v", other, err)
	}
}

// TestMemoryUpsertDedup verifies the long-term layer: facts deduplicate by
// (kind, key) with a confirmations bump on re-write, explicit writes start
// durable, TouchMemories counts injections and DeleteMemory removes by id.
func TestMemoryUpsertDedup(t *testing.T) {
	d := openTestDB(t)
	const model = "test-model"

	first, err := d.UpsertMemory(model, MemoryItem{Kind: KindKnowledge, Key: "курс", Text: "курс: начат", Source: SourceCompress})
	if err != nil {
		t.Fatalf("UpsertMemory: %v", err)
	}
	if first.Confirmations != 1 || first.Hits != 0 {
		t.Fatalf("first item = %+v", first)
	}
	// Same key → same item, confirmations bumped.
	second, err := d.UpsertMemory(model, MemoryItem{Kind: KindKnowledge, Key: "курс", Text: "курс: half-way", Source: SourceCompress})
	if err != nil {
		t.Fatalf("UpsertMemory 2: %v", err)
	}
	if second.ID != first.ID || second.Confirmations != 2 || second.Text != "курс: half-way" {
		t.Fatalf("second item = %+v, want id %s + confirmations 2", second, first.ID)
	}
	// Different kind, same key → a separate item.
	profile, err := d.UpsertMemory(model, MemoryItem{Kind: KindProfile, Key: "имя", Text: "имя: Аня", Source: SourceUser, Confirmations: 2})
	if err != nil {
		t.Fatalf("UpsertMemory profile: %v", err)
	}
	if profile.Confirmations != 2 {
		t.Fatalf("explicit write = %+v, want confirmations 2", profile)
	}
	items, err := d.ListMemories(model)
	if err != nil {
		t.Fatalf("ListMemories: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("items = %d, want 2", len(items))
	}

	// Touch bumps hits and records the injection time.
	if err := d.TouchMemories(model, []string{first.ID}); err != nil {
		t.Fatalf("TouchMemories: %v", err)
	}
	items, _ = d.ListMemories(model)
	for _, it := range items {
		if it.ID == first.ID && (it.Hits != 1 || it.LastHit.IsZero()) {
			t.Fatalf("touched item = %+v", it)
		}
	}

	// Delete by id.
	if err := d.DeleteMemory(model, first.ID); err != nil {
		t.Fatalf("DeleteMemory: %v", err)
	}
	items, _ = d.ListMemories(model)
	if len(items) != 1 || items[0].ID == first.ID {
		t.Fatalf("items after delete = %+v", items)
	}
}

// TestOpenEmptyPath verifies that an empty path is rejected.
func TestOpenEmptyPath(t *testing.T) {
	if _, err := Open(""); err == nil {
		t.Fatal("Open(\"\") succeeded, want an error")
	}
}

// TestBranchesTaskStateMigration verifies that a database created before the
// task state machine existed (branches without the task_state column) is
// migrated in place on Open: the column appears, old rows survive and the FSM
// round-trips through the new schema.
func TestBranchesTaskStateMigration(t *testing.T) {
	path := filepath.Join(t.TempDir(), "old.db")

	// Build a pre-task-machine database: the branches table WITHOUT the
	// task_state column, with one legacy branch row.
	raw, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	_, err = raw.Exec(`CREATE TABLE branches (
		model TEXT NOT NULL, session TEXT NOT NULL DEFAULT '',
		leaf TEXT NOT NULL, title TEXT NOT NULL DEFAULT '',
		fork_up_to TEXT NOT NULL DEFAULT '', frozen TEXT NOT NULL DEFAULT '',
		covered_up_to TEXT NOT NULL DEFAULT '', summary TEXT NOT NULL DEFAULT '',
		plaque TEXT NOT NULL DEFAULT '', facts TEXT NOT NULL DEFAULT '',
		facts_up_to TEXT NOT NULL DEFAULT '', task TEXT NOT NULL DEFAULT '',
		notes TEXT NOT NULL DEFAULT '', history_tokens INTEGER NOT NULL DEFAULT 0,
		PRIMARY KEY (model, session, leaf)
	)`)
	if err != nil {
		raw.Close()
		t.Fatalf("create legacy branches: %v", err)
	}
	if _, err = raw.Exec(`INSERT INTO branches (model, session, leaf, task) VALUES ('m', '', 'm1', 'старая задача')`); err != nil {
		raw.Close()
		t.Fatalf("insert legacy row: %v", err)
	}
	if err = raw.Close(); err != nil {
		t.Fatalf("close legacy db: %v", err)
	}

	// Open migrates the schema in place.
	d, err := Open(path)
	if err != nil {
		t.Fatalf("Open (migration): %v", err)
	}
	t.Cleanup(func() { d.Close() })
	var hasColumn int
	if err := d.db.QueryRow(`SELECT COUNT(*) FROM pragma_table_info('branches') WHERE name='task_state'`).Scan(&hasColumn); err != nil {
		t.Fatalf("inspect task_state: %v", err)
	}
	if hasColumn != 1 {
		t.Fatal("task_state column missing after migration")
	}

	// Save a session with the FSM on the migrated schema and read it back.
	st := HistoryState{
		Model: "m", Active: "m1", NextID: 1,
		Messages: []ChatRecord{{ID: "m1", Role: "user", Content: "привет", At: time.Now().UTC()}},
		Branches: []BranchState{{
			Leaf: "m1", Task: "старая задача",
			TaskFSM: &TaskFSM{Stage: "planning", Plan: []TaskStep{{Title: "шаг", Status: StepActive}}},
		}},
	}
	if err := d.SaveSession(st); err != nil {
		t.Fatalf("SaveSession: %v", err)
	}
	got, err := d.LoadSession("m", "")
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if got == nil || len(got.Branches) != 1 || got.Branches[0].TaskFSM == nil ||
		got.Branches[0].TaskFSM.Stage != "planning" || len(got.Branches[0].TaskFSM.Plan) != 1 {
		t.Fatalf("migrated branch = %+v, want the FSM to round-trip", got)
	}

	// An empty ({"stage":""}) FSM is normalized to nil both on save and on
	// load, so a legacy garbage row cannot resurrect a phantom task.
	raw2, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	if _, err = raw2.Exec(`UPDATE branches SET task_state='{"stage":""}' WHERE leaf='m1'`); err != nil {
		raw2.Close()
		t.Fatalf("inject legacy row: %v", err)
	}
	if err = raw2.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	got, err = d.LoadSession("m", "")
	if err != nil {
		t.Fatalf("LoadSession (legacy row): %v", err)
	}
	if got.Branches[0].TaskFSM != nil {
		t.Fatalf("legacy {\"stage\":\"\"} row loaded as %+v, want nil", got.Branches[0].TaskFSM)
	}
	if err := d.SaveSession(HistoryState{
		Model: "m", Active: "m1", NextID: 1,
		Messages: st.Messages,
		Branches: []BranchState{{Leaf: "m1", TaskFSM: &TaskFSM{}}},
	}); err != nil {
		t.Fatalf("SaveSession (empty FSM): %v", err)
	}
	var blob string
	if err := d.db.QueryRow(`SELECT task_state FROM branches WHERE leaf='m1'`).Scan(&blob); err != nil {
		t.Fatalf("scan task_state: %v", err)
	}
	if blob != "" {
		t.Fatalf("empty FSM saved as %q, want empty string", blob)
	}
}

func TestDeleteAllData(t *testing.T) {
	d := openTestDB(t)
	at := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	for _, model := range []string{"model-a", "model-b"} {
		st := HistoryState{
			Model: model, Active: "m2", NextID: 2,
			Messages: []ChatRecord{
				{ID: "m1", Role: "user", Content: "вопрос", At: at},
				{ID: "m2", Parent: "m1", Role: "assistant", Content: "ответ", At: at.Add(time.Second)},
			},
		}
		if err := d.SaveSession(st); err != nil {
			t.Fatalf("SaveSession(%s): %v", model, err)
		}
		if _, err := d.UpsertMemory(model, MemoryItem{Kind: KindProfile, Key: "имя", Text: "имя: Аня", Source: SourceUser}); err != nil {
			t.Fatalf("UpsertMemory(%s): %v", model, err)
		}
	}
	if err := d.DeleteAllData(); err != nil {
		t.Fatalf("DeleteAllData: %v", err)
	}
	for _, model := range []string{"model-a", "model-b"} {
		if got, err := d.LoadSession(model, ""); err != nil || got != nil {
			t.Fatalf("LoadSession(%s) after DeleteAllData = %+v, %v; want nil", model, got, err)
		}
		items, err := d.ListMemories(model)
		if err != nil {
			t.Fatalf("ListMemories(%s): %v", model, err)
		}
		if len(items) != 0 {
			t.Fatalf("ListMemories(%s) after DeleteAllData = %d items, want 0", model, len(items))
		}
	}
}
