package server

// Tests of the task state machine (task.go): the strict stage-transition
// table, plan normalization, the compression-line parsing (ЭТАП/ШАГ/
// ОЖИДАНИЕ), the /api/task HTTP round-trip, the reset scopes ("chat" keeps
// the FSM, "working"/"history"/"all" clear it) and SQLite persistence across
// a server restart.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"ai-chat-go/api"
	"ai-chat-go/config"
	"ai-chat-go/storage"
)

// newTaskTestServer builds a white-box Server on a throwaway SQLite database
// (dbFile may be empty for in-memory-only tests) and an inert upstream: these
// tests drive the FSM directly and over /api/task, never over /api/chat.
func newTaskTestServer(t *testing.T, dbFile string) *Server {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	t.Cleanup(upstream.Close)
	client := api.New(upstream.URL, "test-key", 5*time.Second)
	cfg := config.Config{
		APIKey:  "test-key",
		BaseURL: upstream.URL,
		Model:   "test-model",
		System:  "be brief",
		Timeout: 5 * time.Second,
		DBFile:  dbFile,
	}
	return New(client, cfg)
}

// TestCanTransitionMatrix pins the strict automaton: every (from, to) pair
// against the documented transition table ("" = no task yet, reset target).
func TestCanTransitionMatrix(t *testing.T) {
	cases := []struct {
		from, to string
		want     bool
	}{
		{"", StagePlanning, true},
		{"", StageExecution, false},
		{StagePlanning, StageExecution, true},
		{StagePlanning, "", true}, // reset (отмена)
		{StagePlanning, StageValidation, false},
		{StagePlanning, StageDone, false},
		{StageExecution, StageValidation, true},
		{StageExecution, StagePlanning, false},
		{StageExecution, StageDone, false},
		{StageExecution, "", false},
		{StageValidation, StageDone, true},
		{StageValidation, StageExecution, true}, // откат на проблемы
		{StageValidation, StagePlanning, false},
		{StageValidation, "", false},
		{StageDone, StagePlanning, true}, // новая задача
		{StageDone, StageExecution, false},
		{StageDone, "", false},
		{StagePlanning, "bogus", false},
		{"bogus", StagePlanning, false},
	}
	for _, c := range cases {
		if got := canTransition(c.from, c.to); got != c.want {
			t.Errorf("canTransition(%q, %q) = %v, want %v", c.from, c.to, got, c.want)
		}
	}
}

// TestNextStage checks the advance direction used by task_state(action=advance):
// a missing task ("" ) has no successor — the tool asks to create the task
// with action=set, stage=planning first.
func TestNextStage(t *testing.T) {
	cases := map[string]string{StagePlanning: StageExecution,
		StageExecution: StageValidation, StageValidation: StageDone}
	for from, want := range cases {
		got, ok := nextStage(from)
		if !ok || got != want {
			t.Errorf("nextStage(%q) = %q, %v; want %q, true", from, got, ok, want)
		}
	}
	if got, ok := nextStage(""); ok {
		t.Errorf("nextStage(\"\") = %q, %v; want no successor", got, ok)
	}
	if got, ok := nextStage(StageDone); ok {
		t.Errorf("nextStage(done) = %q, %v; want no successor", got, ok)
	}
	if _, ok := nextStage("bogus"); ok {
		t.Error("nextStage(bogus) = ok; want false")
	}
}

// applyUpdate is the white-box test helper: one strict FSM update under s.mu.
func (s *Server) applyUpdate(t *testing.T, in *storage.TaskFSM) (*storage.TaskFSM, error) {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.applyTaskUpdateLocked(in)
}

// TestApplyTaskUpdateStrict walks the whole automaton from the user/model side:
// the legal cycle planning → execution → validation → done → planning, the
// validation → execution rollback, the reset-from-planning-only rule and the
// rejected jumps.
func TestApplyTaskUpdateStrict(t *testing.T) {
	s := newTaskTestServer(t, "")

	// New task starts from planning.
	if _, err := s.applyUpdate(t, &storage.TaskFSM{Stage: StagePlanning}); err != nil {
		t.Fatalf("new task (planning): %v", err)
	}
	// Illegal jumps are rejected with a reason (the current stage is
	// idempotent: setting it again is a no-op, not an error).
	for _, bad := range []string{StageValidation, StageDone} {
		if _, err := s.applyUpdate(t, &storage.TaskFSM{Stage: bad}); err == nil {
			t.Fatalf("planning → %s succeeded, want an error", bad)
		}
	}
	if _, err := s.applyUpdate(t, &storage.TaskFSM{Stage: "bogus"}); err == nil {
		t.Fatal("unknown stage accepted, want an error")
	}
	// The plan gate: execution cannot start before an approved plan.
	if _, err := s.applyUpdate(t, &storage.TaskFSM{Stage: StageExecution}); err == nil {
		t.Fatal("planning → execution without a plan succeeded, want an error")
	}
	if _, err := s.applyUpdate(t, &storage.TaskFSM{Plan: []storage.TaskStep{{Title: "a"}, {Title: "b"}}}); err != nil {
		t.Fatalf("set plan: %v", err)
	}
	// The legal cycle, with the rollback validation → execution (and back
	// through validation before done). done требует закрытого плана —
	// закрываем оба шага перед циклом (normalizePlan сохраняет all-done).
	if _, err := s.applyUpdate(t, &storage.TaskFSM{Plan: []storage.TaskStep{
		{Title: "a", Status: storage.StepDone}, {Title: "b", Status: storage.StepDone},
	}}); err != nil {
		t.Fatalf("close plan: %v", err)
	}
	for _, next := range []string{StageExecution, StageValidation, StageExecution, StageValidation, StageDone, StagePlanning} {
		if _, err := s.applyUpdate(t, &storage.TaskFSM{Stage: next}); err != nil {
			t.Fatalf("advance to %s: %v", next, err)
		}
	}
	// From planning the task can be reset (cancelled); the FSM is dropped.
	if _, err := s.applyUpdate(t, &storage.TaskFSM{Step: "что-то"}); err != nil {
		t.Fatalf("set step without stage: %v", err)
	}
	if ts, err := s.applyUpdate(t, &storage.TaskFSM{Stage: "reset"}); err != nil || ts != nil {
		t.Fatalf("reset from planning = %v, %v; want nil, nil", ts, err)
	}
	// A second reset without a task is rejected — and a rejected update must
	// not leave an empty (non-nil) FSM behind: the branch stays taskless.
	if _, err := s.applyUpdate(t, &storage.TaskFSM{Stage: "reset"}); err == nil {
		t.Fatal("reset without a task succeeded, want an error")
	}
	s.mu.Lock()
	if ts := s.taskFSMLocked(); ts != nil {
		t.Fatalf("rejected update left an FSM behind: %+v, want nil", ts)
	}
	s.mu.Unlock()
	// A fresh task: step/expect update without a stage, then a full update.
	if _, err := s.applyUpdate(t, &storage.TaskFSM{Stage: StagePlanning}); err != nil {
		t.Fatalf("new task: %v", err)
	}
	if _, err := s.applyUpdate(t, &storage.TaskFSM{Step: "написать тесты"}); err != nil {
		t.Fatalf("set step: %v", err)
	}
	ts, err := s.applyUpdate(t, &storage.TaskFSM{
		Stage: StageExecution,
		Step:  "реализовать модуль", Expect: "запустить тесты",
		Plan: []storage.TaskStep{{Title: "a"}, {Title: "b"}, {Title: "c"}},
	})
	if err != nil {
		t.Fatalf("full update: %v", err)
	}
	if ts.Stage != StageExecution || ts.Step != "реализовать модуль" || ts.Expect != "запустить тесты" {
		t.Fatalf("full update result = %+v", ts)
	}
	if len(ts.Plan) != 3 || ts.Plan[0].Status != storage.StepActive {
		t.Fatalf("plan after full update = %+v, want 3 steps, first active", ts.Plan)
	}
}

// TestNormalizePlan checks the plan-checklist sanitization: empty titles are
// dropped, statuses are normalized, at most one step is active, and the first
// pending step after the done prefix becomes active.
func TestNormalizePlan(t *testing.T) {
	steps := func(titles ...string) []storage.TaskStep {
		out := make([]storage.TaskStep, len(titles))
		for i, tr := range titles {
			out[i] = storage.TaskStep{Title: tr}
		}
		return out
	}
	with := func(statuses ...string) []storage.TaskStep {
		out := make([]storage.TaskStep, len(statuses))
		for i, s := range statuses {
			out[i] = storage.TaskStep{Title: "s" + string(rune('a'+i)), Status: s}
		}
		return out
	}
	// All pending → the first step becomes active.
	got := normalizePlan(steps("a", "b", "c"))
	if len(got) != 3 || got[0].Status != storage.StepActive || got[1].Status != storage.StepPending {
		t.Fatalf("all pending = %+v, want first active", got)
	}
	// Empty titles are dropped; unknown statuses become pending.
	got = normalizePlan([]storage.TaskStep{{Title: " "}, {Title: "a", Status: "BOGUS"}, {Title: "b"}})
	if len(got) != 2 || got[0].Status != storage.StepActive || got[1].Status != storage.StepPending {
		t.Fatalf("empty title = %+v, want [a active, b pending]", got)
	}
	// Several active steps → only the first survives.
	got = normalizePlan(with("active", "active", "pending"))
	if len(got) != 3 || got[0].Status != storage.StepActive || got[1].Status != storage.StepPending {
		t.Fatalf("two active = %+v, want only the first active", got)
	}
	// Done prefix → the first pending after it becomes active.
	got = normalizePlan(with("done", "pending", "pending"))
	if got[0].Status != storage.StepDone || got[1].Status != storage.StepActive || got[2].Status != storage.StepPending {
		t.Fatalf("done prefix = %+v, want [done, active, pending]", got)
	}
	// A fully done plan stays fully done (no auto-activation).
	got = normalizePlan(with("done", "done"))
	if len(got) != 2 || got[0].Status != storage.StepDone || got[1].Status != storage.StepDone {
		t.Fatalf("all done = %+v, want both done", got)
	}
	// An empty plan stays empty.
	if got = normalizePlan(nil); len(got) != 0 {
		t.Fatalf("empty plan = %+v, want empty", got)
	}

	if done, total := planProgress(with("done", "done", "pending")); done != 2 || total != 3 {
		t.Fatalf("planProgress = %d/%d, want 2/3", done, total)
	}
}

// TestSplitTaskStateLines checks the parser of the compression FSM lines
// (ЭТАП:/ШАГ:/ОЖИДАНИЕ:, any letter case) and that the rest of the summary is
// preserved.
func TestSplitTaskStateLines(t *testing.T) {
	ts, rest := splitTaskStateLines("Сводка диалога\nэтап: execution\nшаг: писать код\nОЖИДАНИЕ: запуск тестов\nФАКТ: имя: Аня")
	if ts == nil {
		t.Fatal("no FSM parsed, want stage/step/expect")
	}
	if ts.Stage != StageExecution || ts.Step != "писать код" || ts.Expect != "запуск тестов" {
		t.Fatalf("FSM = %+v", ts)
	}
	if rest != "Сводка диалога\nФАКТ: имя: Аня" {
		t.Fatalf("rest = %q, want the non-FSM lines", rest)
	}

	// Without FSM lines the result is nil and the text survives.
	if ts, rest := splitTaskStateLines("только сводка\nбез строк"); ts != nil || rest != "только сводка\nбез строк" {
		t.Fatalf("no lines = %v, %q", ts, rest)
	}
	// Empty values are ignored; a fully empty FSM becomes nil.
	if ts, _ := splitTaskStateLines("ЭТАП:"); ts != nil {
		t.Fatalf("empty stage value = %+v, want nil", ts)
	}
}

// TestApplyCompressionFSMLenient verifies the lenient merge: a compression
// snapshot refreshes the step/expect text of a running task, but never moves
// the stage and never creates a task (the summary is a retelling of history,
// not an action — the stage moves only through task_state).
func TestApplyCompressionFSMLenient(t *testing.T) {
	b := &branchState{TaskFSM: &storage.TaskFSM{Stage: StageExecution, Step: "старый шаг"}}
	applyCompressionFSM(b, &storage.TaskFSM{Stage: StageDone, Step: "новый шаг", Expect: "проверить результат"})
	if b.TaskFSM.Stage != StageExecution || b.TaskFSM.Step != "новый шаг" || b.TaskFSM.Expect != "проверить результат" {
		t.Fatalf("lenient merge = %+v", b.TaskFSM)
	}
	// Этап не перетаскивается из сводки — ни вперёд, ни назад.
	applyCompressionFSM(b, &storage.TaskFSM{Stage: StagePlanning})
	if b.TaskFSM.Stage != StageExecution {
		t.Fatalf("compression moved the stage: %+v", b.TaskFSM)
	}
	// Сжатие не создаёт задачу из ничего.
	empty := &branchState{}
	applyCompressionFSM(empty, &storage.TaskFSM{Stage: StageDone, Step: "s"})
	if empty.TaskFSM != nil {
		t.Fatalf("compression created a task: %+v", empty.TaskFSM)
	}
	applyCompressionFSM(b, nil)
	if b.TaskFSM.Stage != StageExecution {
		t.Fatalf("nil snapshot changed the state: %+v", b.TaskFSM)
	}
}

// TestCloneTaskFSM checks the deep copy used when a branch inherits (or a
// reset keeps) the FSM: mutating the clone must not touch the original.
func TestCloneTaskFSM(t *testing.T) {
	if got := cloneTaskFSM(nil); got != nil {
		t.Fatalf("cloneTaskFSM(nil) = %v, want nil", got)
	}
	orig := &storage.TaskFSM{Stage: StagePlanning, Step: "s",
		Plan: []storage.TaskStep{{Title: "a"}, {Title: "b"}}}
	cp := cloneTaskFSM(orig)
	if cp == orig {
		t.Fatal("clone shares the FSM with the original")
	}
	if cp.Plan == nil || orig.Plan == nil || &cp.Plan[0] == &orig.Plan[0] {
		t.Fatal("clone shares the plan slice with the original")
	}
	cp.Stage = StageExecution
	cp.Plan[0].Title = "изменено"
	if orig.Stage != StagePlanning || orig.Plan[0].Title != "a" {
		t.Fatalf("mutating the clone touched the original: %+v", orig)
	}
}

// taskGET fetches GET /api/task and returns the task field (nil = no task).
func taskGET(t *testing.T, url string) *storage.TaskFSM {
	t.Helper()
	resp, err := http.Get(url + "/api/task")
	if err != nil {
		t.Fatalf("GET /api/task: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /api/task = %d", resp.StatusCode)
	}
	var out struct {
		OK   bool             `json:"ok"`
		Task *storage.TaskFSM `json:"task"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("GET /api/task decode: %v", err)
	}
	return out.Task
}

// taskPOST sends a /api/task body and returns (status, the task or the error text).
func taskPOST(t *testing.T, url, body string) (int, *storage.TaskFSM, string) {
	t.Helper()
	resp, bodyStr := postBody(t, url+"/api/task", body)
	text := bodyStr
	if resp.StatusCode != http.StatusOK {
		return resp.StatusCode, nil, strings.TrimSpace(text)
	}
	var out struct {
		OK   bool             `json:"ok"`
		Task *storage.TaskFSM `json:"task"`
	}
	if err := json.Unmarshal([]byte(text), &out); err != nil {
		t.Fatalf("POST /api/task decode %q: %v", text, err)
	}
	return resp.StatusCode, out.Task, ""
}

// TestTaskAPIRoundtrip drives the user side (GET/POST /api/task) over HTTP:
// the full legal cycle, a rejected jump, the plan update (statuses
// server-normalized), the reset and the invalid JSON body.
func TestTaskAPIRoundtrip(t *testing.T) {
	s := newTaskTestServer(t, "")
	tsrv := httptest.NewServer(s.Handler())
	defer tsrv.Close()

	if ts := taskGET(t, tsrv.URL); ts != nil {
		t.Fatalf("fresh GET = %+v, want null", ts)
	}
	// New task → planning.
	if _, ts, msg := taskPOST(t, tsrv.URL, `{"stage":"planning"}`); ts == nil || ts.Stage != StagePlanning {
		t.Fatalf("new task = %+v, %q", ts, msg)
	}
	// A jump over execution is rejected with the allowed transitions.
	if code, _, msg := taskPOST(t, tsrv.URL, `{"stage":"validation"}`); code != http.StatusBadRequest {
		t.Fatalf("planning → validation = %d, want 400: %s", code, msg)
	} else if !strings.Contains(msg, "запрещённый переход") {
		t.Fatalf("400 body = %q, want the transition reason", msg)
	}
	// The plan gate: execution before an approved plan is rejected.
	if code, _, msg := taskPOST(t, tsrv.URL, `{"stage":"execution","step":"модуль","expect":"тесты"}`); code != http.StatusBadRequest {
		t.Fatalf("execution without a plan = %d, want 400: %s", code, msg)
	} else if !strings.Contains(msg, "плана") {
		t.Fatalf("400 body = %q, want the plan-gate reason", msg)
	}
	// A plan update (the server normalizes the statuses), then execution.
	_, ts, msg := taskPOST(t, tsrv.URL, `{"plan":[{"title":"a","status":"done"},{"title":"b","status":"pending"},{"title":"c","status":"pending"}]}`)
	if ts == nil || ts.Stage != StagePlanning {
		t.Fatalf("plan update = %+v, %q, want planning untouched", ts, msg)
	}
	if len(ts.Plan) != 3 || ts.Plan[0].Status != storage.StepDone || ts.Plan[1].Status != storage.StepActive || ts.Plan[2].Status != storage.StepPending {
		t.Fatalf("plan statuses = %+v, want [done active pending]", ts.Plan)
	}
	if done, total := planProgress(ts.Plan); done != 1 || total != 3 {
		t.Fatalf("plan progress = %d/%d, want 1/3", done, total)
	}
	if _, ts, msg = taskPOST(t, tsrv.URL, `{"stage":"execution","step":"модуль","expect":"тесты"}`); ts == nil || ts.Stage != StageExecution {
		t.Fatalf("execution after the plan = %+v, %q, want execution", ts, msg)
	}
	// Закрываем план: финальный done требует полностью выполненного плана
	// (стадия при обновлении плана не трогается).
	if _, ts, msg := taskPOST(t, tsrv.URL, `{"plan":[{"title":"a","status":"done"},{"title":"b","status":"done"},{"title":"c","status":"done"}]}`); ts == nil || ts.Stage != StageExecution {
		t.Fatalf("plan close = %+v, %q, want execution untouched", ts, msg)
	}
	// The cycle to done and back to a new task, then the reset.
	if _, ts, _ := taskPOST(t, tsrv.URL, `{"stage":"validation"}`); ts == nil || ts.Stage != StageValidation {
		t.Fatal("validation failed")
	}
	if _, ts, _ := taskPOST(t, tsrv.URL, `{"stage":"done"}`); ts == nil || ts.Stage != StageDone {
		t.Fatal("done failed")
	}
	if _, ts, _ := taskPOST(t, tsrv.URL, `{"stage":"planning"}`); ts == nil || ts.Stage != StagePlanning {
		t.Fatal("new task failed")
	}
	if _, ts, msg := taskPOST(t, tsrv.URL, `{"stage":"reset"}`); ts != nil || msg != "" {
		t.Fatalf("reset = %v, %q, want null", ts, msg)
	}
	if ts := taskGET(t, tsrv.URL); ts != nil {
		t.Fatalf("GET after reset = %+v, want null", ts)
	}
	// Invalid JSON → 400.
	resp, err := http.Post(tsrv.URL+"/api/task", "application/json", strings.NewReader("{не json"))
	if err != nil {
		t.Fatalf("POST invalid: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("invalid JSON = %d, want 400", resp.StatusCode)
	}

	// The memory panel's payload carries the same FSM as "task".
	if code, _, msg := taskPOST(t, tsrv.URL, `{"stage":"planning"}`); code != http.StatusOK {
		t.Fatalf("reseed planning: %d %s", code, msg)
	}
	memResp, err := http.Get(tsrv.URL + "/api/memory")
	if err != nil {
		t.Fatalf("GET /api/memory: %v", err)
	}
	defer memResp.Body.Close()
	var mem struct {
		Task *storage.TaskFSM `json:"task"`
	}
	if err := json.NewDecoder(memResp.Body).Decode(&mem); err != nil {
		t.Fatalf("GET /api/memory decode: %v", err)
	}
	if mem.Task == nil || mem.Task.Stage != StagePlanning {
		t.Fatalf("/api/memory task = %+v, want planning", mem.Task)
	}
}

// TestTaskFSMResetScopes pins the reset semantics of the FSM: «Удалить чат»
// keeps it (it belongs to the task, not the messages), «рабочая»/«история»/
// «все» clear it.
func TestTaskFSMResetScopes(t *testing.T) {
	s := newTaskTestServer(t, "")
	tsrv := httptest.NewServer(s.Handler())
	defer tsrv.Close()

	seed := func() {
		if code, _, msg := taskPOST(t, tsrv.URL, `{"stage":"planning","step":"шаг"}`); code != http.StatusOK {
			t.Fatalf("seed planning: %d %s", code, msg)
		}
	}
	seed()

	// «Удалить чат» — the FSM survives.
	if resp, _ := postBody(t, tsrv.URL+"/api/reset", `{"scope":"chat"}`); resp.StatusCode != http.StatusOK {
		t.Fatal("chat reset failed")
	}
	if ts := taskGET(t, tsrv.URL); ts == nil || ts.Stage != StagePlanning || ts.Step != "шаг" {
		t.Fatalf("FSM after chat reset = %+v, want planning/шаг", ts)
	}

	// «Удалить рабочую» — the FSM is cleared with the layer.
	seed()
	if resp, _ := postBody(t, tsrv.URL+"/api/reset", `{"scope":"working"}`); resp.StatusCode != http.StatusOK {
		t.Fatal("working reset failed")
	}
	if ts := taskGET(t, tsrv.URL); ts != nil {
		t.Fatalf("FSM after working reset = %+v, want null", ts)
	}

	// «Удалить историю» — the FSM is cleared with the branch state.
	seed()
	if resp, _ := postBody(t, tsrv.URL+"/api/reset", `{"scope":"history"}`); resp.StatusCode != http.StatusOK {
		t.Fatal("history reset failed")
	}
	if ts := taskGET(t, tsrv.URL); ts != nil {
		t.Fatalf("FSM after history reset = %+v, want null", ts)
	}

	// «Удалить всё» — the FSM is cleared as well.
	seed()
	if resp, _ := postBody(t, tsrv.URL+"/api/reset", `{"scope":"all"}`); resp.StatusCode != http.StatusOK {
		t.Fatal("all reset failed")
	}
	if ts := taskGET(t, tsrv.URL); ts != nil {
		t.Fatalf("FSM after all reset = %+v, want null", ts)
	}
}

// TestTaskFSMPersistsAcrossRestart verifies the SQLite round-trip: a task FSM
// saved with the session is restored on the next server start.
func TestTaskFSMPersistsAcrossRestart(t *testing.T) {
	dbFile := t.TempDir() + "/chat.db"

	s1 := newTaskTestServer(t, dbFile)
	tsrv1 := httptest.NewServer(s1.Handler())
	if _, ts, msg := taskPOST(t, tsrv1.URL, `{"stage":"planning"}`); ts == nil || ts.Stage != StagePlanning {
		t.Fatalf("seed = %+v, %q", ts, msg)
	}
	if _, ts, _ := taskPOST(t, tsrv1.URL, `{"stage":"execution","step":"модуль","expect":"тесты","plan":[{"title":"a","status":"done"},{"title":"b"}]}`); ts == nil {
		t.Fatal("execution seed failed")
	}
	tsrv1.Close()
	if s1.db != nil {
		s1.db.Close() // освобождает файл для следующего сервера
	}

	// A fresh server on the same database must restore the FSM.
	s2 := newTaskTestServer(t, dbFile)
	tsrv2 := httptest.NewServer(s2.Handler())
	defer tsrv2.Close()
	ts := taskGET(t, tsrv2.URL)
	if ts == nil || ts.Stage != StageExecution || ts.Step != "модуль" || ts.Expect != "тесты" {
		t.Fatalf("restored FSM = %+v, want execution/модуль/тесты", ts)
	}
	if len(ts.Plan) != 2 || ts.Plan[0].Status != storage.StepDone || ts.Plan[1].Status != storage.StepActive {
		t.Fatalf("restored plan = %+v, want [done active]", ts.Plan)
	}
}

// TestOneStageMovePerExchange pins the per-exchange stage-move budget: the
// model cannot walk the whole automaton in one reply (the regression: «задача
// готова» сразу после создания задачи проскакивала planning → … → done
// цепочкой advance в одном обмене). Повторный set того же этапа бюджет не
// тратит; сброс счётчика имитирует начало нового обмена (handleChat).
func TestOneStageMovePerExchange(t *testing.T) {
	s := newTaskTestServer(t, "")
	exec := s.memoryToolExec(s.activeSession)

	// Первый переход за обмен легален.
	if _, err := exec("task_state", json.RawMessage(`{"action":"set","stage":"planning"}`)); err != nil {
		t.Fatalf("set planning: %v", err)
	}
	// План (не переход этапа — бюджет не тратит).
	if _, err := exec("task_state", json.RawMessage(`{"action":"plan","plan":["a","b"]}`)); err != nil {
		t.Fatalf("set plan: %v", err)
	}
	if _, err := exec("task_state", json.RawMessage(`{"action":"advance"}`)); err == nil || !strings.Contains(err.Error(), "уже менялся") {
		t.Fatalf("second stage move in one exchange = %v, want the budget error", err)
	}
	// Тот же этап повторно — бюджет не тратится.
	if _, err := exec("task_state", json.RawMessage(`{"action":"set","stage":"planning"}`)); err != nil {
		t.Fatalf("idempotent set: %v", err)
	}
	// Новый обмен — счётчик сброшен, переход снова легален.
	s.mu.Lock()
	s.stageMoves = 0
	s.mu.Unlock()
	if _, err := exec("task_state", json.RawMessage(`{"action":"advance"}`)); err != nil {
		t.Fatalf("advance after the budget reset: %v", err)
	}
	// И снова только один: следующий advance отклонён.
	if _, err := exec("task_state", json.RawMessage(`{"action":"advance"}`)); err == nil || !strings.Contains(err.Error(), "уже менялся") {
		t.Fatalf("chained advance = %v, want the budget error", err)
	}
}

// TestDoneRequiresCompletePlan pins the done gate: finishing with an
// incomplete plan is rejected (the model and the UI alike); once every step
// is done the transition goes through.
func TestDoneRequiresCompletePlan(t *testing.T) {
	s := newTaskTestServer(t, "")
	if _, err := s.applyUpdate(t, &storage.TaskFSM{
		Stage: StagePlanning,
		Plan:  []storage.TaskStep{{Title: "a"}, {Title: "b"}},
	}); err != nil {
		t.Fatalf("new task with a plan: %v", err)
	}
	for _, next := range []string{StageExecution, StageValidation} {
		if _, err := s.applyUpdate(t, &storage.TaskFSM{Stage: next}); err != nil {
			t.Fatalf("advance to %s: %v", next, err)
		}
	}
	if _, err := s.applyUpdate(t, &storage.TaskFSM{Stage: StageDone}); err == nil || !strings.Contains(err.Error(), "не полностью") {
		t.Fatalf("done with a 0/2 plan = %v, want the gate error", err)
	}
	// Закрываем план — финал становится легальным.
	if _, err := s.applyUpdate(t, &storage.TaskFSM{Plan: []storage.TaskStep{
		{Title: "a", Status: storage.StepDone}, {Title: "b", Status: storage.StepDone},
	}}); err != nil {
		t.Fatalf("close plan: %v", err)
	}
	if ts, err := s.applyUpdate(t, &storage.TaskFSM{Stage: StageDone}); err != nil || ts == nil || ts.Stage != StageDone {
		t.Fatalf("done with a closed plan = %+v, %v", ts, err)
	}
}
