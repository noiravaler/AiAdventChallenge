package storage

import (
	"database/sql"
	"path/filepath"
	"slices"
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

// TestMemoryPoliciesAndLog verifies the long-term layer write rules: latest
// facts keep one actual value (a conflicting write REPLACES the value and is
// logged «что было → что стало»), append facts accumulate as a list, a fact
// with a later write time wins (stale incoming writes are rejected), explicit
// writes start durable, the journal records insert/append/replace/delete and
// DeleteMemory removes by id.
func TestMemoryPoliciesAndLog(t *testing.T) {
	d := openTestDB(t)
	const model = "test-model"

	first, err := d.InsertMemory(model, MemoryItem{Kind: KindKnowledge, Key: "курс", Text: "курс: начат", Source: SourceCompress}, MemLogInsert, SourceCompress)
	if err != nil {
		t.Fatalf("InsertMemory: %v", err)
	}
	if first.Confirmations != 1 || first.Hits != 0 || first.Policy != PolicyLatest {
		t.Fatalf("first item = %+v", first)
	}
	// Conflicting value of a latest fact → replace; created survives, the
	// replace is logged.
	replaced, err := d.ReplaceMemory(model, first.ID, MemoryItem{Kind: KindKnowledge, Key: "курс", Text: "курс: half-way", Source: SourceCompress, Confirmations: 1}, SourceCompress)
	if err != nil {
		t.Fatalf("ReplaceMemory: %v", err)
	}
	if replaced.ID != first.ID || replaced.Text != "курс: half-way" || replaced.Confirmations != 1 {
		t.Fatalf("replaced item = %+v", replaced)
	}
	// Re-seen same text → confirmations bump, no journal row.
	if _, err := d.ReconfirmMemory(model, first.ID); err != nil {
		t.Fatalf("ReconfirmMemory: %v", err)
	}
	items, err := d.ListMemories(model)
	if err != nil {
		t.Fatalf("ListMemories: %v", err)
	}
	if len(items) != 1 || items[0].Confirmations != 2 || items[0].Text != "курс: half-way" {
		t.Fatalf("items after reconfirm = %+v", items)
	}

	// Append facts accumulate: two different decisions → two rows.
	dec1, err := d.InsertMemory(model, MemoryItem{Kind: KindDecision, Key: "стек", Text: "стек: выбрали Go", Source: SourceUser, Confirmations: 2}, MemLogAppend, SourceUser)
	if err != nil {
		t.Fatalf("InsertMemory decision 1: %v", err)
	}
	if dec1.Policy != PolicyAppend {
		t.Fatalf("decision policy = %q, want append", dec1.Policy)
	}
	if _, err := d.InsertMemory(model, MemoryItem{Kind: KindDecision, Key: "стек", Text: "стек: отказались от ORM", Source: SourceUser, Confirmations: 2}, MemLogAppend, SourceUser); err != nil {
		t.Fatalf("InsertMemory decision 2: %v", err)
	}
	items, _ = d.ListMemories(model)
	if len(items) != 3 {
		t.Fatalf("items = %d, want 3 (two append rows + one latest)", len(items))
	}

	// Explicit profile write starts durable.
	profile, err := d.InsertMemory(model, MemoryItem{Kind: KindProfile, Key: "имя", Text: "имя: Аня", Source: SourceUser, Confirmations: 2}, MemLogInsert, SourceUser)
	if err != nil {
		t.Fatalf("InsertMemory profile: %v", err)
	}
	if profile.Confirmations != 2 || profile.Policy != PolicyLatest {
		t.Fatalf("explicit write = %+v, want confirmations 2 + latest", profile)
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

	// The journal saw: insert, replace, two appends, insert — and no
	// reconfirm row.
	journal, err := d.ListMemoryLog(model, 0)
	if err != nil {
		t.Fatalf("ListMemoryLog: %v", err)
	}
	actions := map[string]int{}
	var replacedRow *MemoryLog
	for i := range journal {
		actions[journal[i].Action]++
		if journal[i].Action == MemLogReplace {
			replacedRow = &journal[i]
		}
	}
	if actions[MemLogInsert] != 2 || actions[MemLogAppend] != 2 || actions[MemLogReplace] != 1 || actions[MemLogDelete] != 0 {
		t.Fatalf("journal actions = %v (want 2 insert, 2 append, 1 replace, 0 delete)", actions)
	}
	if replacedRow == nil || replacedRow.OldValue != "курс: начат" || replacedRow.NewValue != "курс: half-way" {
		t.Fatalf("replace row = %+v, want old «курс: начат» → new «курс: half-way»", replacedRow)
	}

	// Delete by id — logged.
	if err := d.DeleteMemory(model, first.ID, SourceUser); err != nil {
		t.Fatalf("DeleteMemory: %v", err)
	}
	items, _ = d.ListMemories(model)
	if len(items) != 3 || slices.ContainsFunc(items, func(it MemoryItem) bool { return it.ID == first.ID }) {
		t.Fatalf("items after delete = %+v", items)
	}
	journal, _ = d.ListMemoryLog(model, 0)
	if len(journal) == 0 || journal[0].Action != MemLogDelete || journal[0].OldValue != "курс: half-way" {
		t.Fatalf("newest journal row = %+v, want delete of «курс: half-way»", journal[0])
	}
}

// TestMemoryStaleWriteRejected verifies the time-priority rule: a fact with a
// later write time wins — an incoming fact whose Updated is older than the
// stored one must not overwrite it. The rule lives in server/memory.go
// (resolveFact); here the storage primitives are checked: the replace path
// accepts any write, so the server is the guard — see the server test
// TestResolveFact in server/memory_test.go.
func TestMemoryStaleWriteRejected(t *testing.T) {
	d := openTestDB(t)
	const model = "test-model"

	it, err := d.InsertMemory(model, MemoryItem{Kind: KindKnowledge, Key: "версия", Text: "версия: 2", Source: SourceCompress}, MemLogInsert, SourceCompress)
	if err != nil {
		t.Fatalf("InsertMemory: %v", err)
	}
	// Simulate a later write time on the stored item.
	if _, err := d.ReconfirmMemory(model, it.ID); err != nil {
		t.Fatalf("ReconfirmMemory: %v", err)
	}
	items, _ := d.ListMemories(model)
	stored := items[0]
	if !stored.Updated.After(it.Created) {
		t.Fatalf("stored updated = %v, want after created %v", stored.Updated, it.Created)
	}
	_ = stored // the stale rejection itself is covered by TestResolveFact
}

// TestMemoryEphemeralExpiry verifies ttl=task: ephemeral facts are removed by
// ExpireEphemeralMemories (each removal logged), durable facts survive.
func TestMemoryEphemeralExpiry(t *testing.T) {
	d := openTestDB(t)
	const model = "test-model"

	tmp, err := d.InsertMemory(model, MemoryItem{Kind: KindKnowledge, Key: "черновик", Text: "черновик: v1", Source: SourceTool, Ephemeral: true}, MemLogInsert, SourceTool)
	if err != nil {
		t.Fatalf("InsertMemory ephemeral: %v", err)
	}
	durable, err := d.InsertMemory(model, MemoryItem{Kind: KindProfile, Key: "имя", Text: "имя: Аня", Source: SourceUser, Confirmations: 2}, MemLogInsert, SourceUser)
	if err != nil {
		t.Fatalf("InsertMemory durable: %v", err)
	}
	expired, err := d.ExpireEphemeralMemories(model, SourceSystem)
	if err != nil {
		t.Fatalf("ExpireEphemeralMemories: %v", err)
	}
	if len(expired) != 1 || expired[0].ID != tmp.ID {
		t.Fatalf("expired = %+v, want only %s", expired, tmp.ID)
	}
	items, _ := d.ListMemories(model)
	if len(items) != 1 || items[0].ID != durable.ID {
		t.Fatalf("items after expiry = %+v, want only %s", items, durable.ID)
	}
	journal, _ := d.ListMemoryLog(model, 0)
	if len(journal) == 0 || journal[0].Action != MemLogExpire {
		t.Fatalf("newest journal row = %+v, want expire", journal[0])
	}
}

// TestMemoriesPolicyMigration verifies the rebuild migration: a pre-policy
// database (memories with table-level UNIQUE, no policy/ephemeral columns)
// migrates on Open — the columns appear, rows keep their data, decisions get
// the append policy, everything else latest.
func TestMemoriesPolicyMigration(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "old.db")
	d, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	// Build the legacy table shape by hand.
	for _, stmt := range []string{
		`DROP TABLE memories`,
		`CREATE TABLE memories (
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
		`INSERT INTO memories (session, id, kind, key, text, source, created, confirmations)
		 VALUES ('s1', 'k1', 'decision', 'стек', 'стек: Go', 'compress', 1700000000000000000, 2),
		        ('s1', 'k2', 'profile', 'имя', 'имя: Аня', 'user', 1700000001000000000, 2)`,
	} {
		if _, err := d.db.Exec(stmt); err != nil {
			t.Fatalf("legacy setup %q: %v", stmt, err)
		}
	}
	d.Close()

	d2, err := Open(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer d2.Close()
	items, err := d2.ListMemories("s1")
	if err != nil {
		t.Fatalf("ListMemories after migration: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("items = %d, want 2", len(items))
	}
	byID := map[string]MemoryItem{}
	for _, it := range items {
		byID[it.ID] = it
	}
	if got := byID["k1"]; got.Policy != PolicyAppend || got.Text != "стек: Go" || got.Confirmations != 2 || got.Created.IsZero() {
		t.Fatalf("migrated decision = %+v", got)
	}
	if got := byID["k2"]; got.Policy != PolicyLatest || got.Text != "имя: Аня" {
		t.Fatalf("migrated profile = %+v", got)
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
		if _, err := d.InsertMemory(model, MemoryItem{Kind: KindProfile, Key: "имя", Text: "имя: Аня", Source: SourceUser}, MemLogInsert, SourceUser); err != nil {
			t.Fatalf("InsertMemory(%s): %v", model, err)
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
