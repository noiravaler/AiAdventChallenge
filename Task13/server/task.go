// The task state machine: a branch's task is tracked as a finite automaton
// (stage + current step + expected action + plan checklist), part of the
// working-memory layer. Transitions are strict (stageNext); the FSM is driven
// by the model's task_state tool, the user's "Память" panel (GET/POST
// /api/task) and the compression prompts (ЭТАП:/ШАГ:/ОЖИДАНИЕ: lines).
package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"ai-chat-go/storage"
)

// Task FSM stages (stageOrder also defines the advance direction).
const (
	StagePlanning   = "planning"
	StageExecution  = "execution"
	StageValidation = "validation"
	StageDone       = "done"
)

// stageOrder — the canonical stage order; task_state(action=advance) moves
// to the next entry, done has no successor (a new task starts from planning).
var stageOrder = []string{StagePlanning, StageExecution, StageValidation, StageDone}

// stageNext — the strict transition table:
//
//	""             → planning           (new task)
//	planning       → execution, reset   (start — only once a plan is set — or cancel)
//	execution      → validation
//	validation     → done, execution    (accept or go back on problems)
//	done           → planning           (new task)
//
// The empty target means "reset" (the FSM is cleared, allowed from planning
// only). The planning → execution transition additionally requires a
// non-empty plan (the plan is approved before the implementation starts);
// the check lives in applyTaskUpdateLocked, not in the prompt.
var stageNext = map[string][]string{
	"":              {StagePlanning},
	StagePlanning:   {StageExecution, ""},
	StageExecution:  {StageValidation},
	StageValidation: {StageDone, StageExecution},
	StageDone:       {StagePlanning},
}

// validStage reports whether the value is one of the four FSM stages.
func validStage(stage string) bool {
	switch stage {
	case StagePlanning, StageExecution, StageValidation, StageDone:
		return true
	}
	return false
}

// canTransition reports whether from→to is allowed by the strict automaton.
// The empty target is the "reset" transition.
func canTransition(from, to string) bool {
	if to != "" && !validStage(to) {
		return false
	}
	for _, next := range stageNext[from] {
		if next == to {
			return true
		}
	}
	return false
}

// nextStage returns the stage after the given one in the advance direction.
func nextStage(stage string) (string, bool) {
	for i, s := range stageOrder {
		if s == stage {
			if i+1 < len(stageOrder) {
				return stageOrder[i+1], true
			}
			return "", false
		}
	}
	return "", false
}

// orStage renders a stage for humans ("нет задачи" when empty).
func orStage(stage string) string {
	if stage == "" {
		return "нет задачи"
	}
	return stage
}

// allowedFrom lists the transitions available from a stage, human-readable.
func allowedFrom(from string) string {
	next := append([]string(nil), stageNext[from]...)
	for i, n := range next {
		if n == "" {
			n = "reset (отмена задачи)"
		}
		next[i] = n
	}
	return strings.Join(next, ", ")
}

// ── Plan checklist ──────────────────────────────────────────────────────────

// normalizePlan sanitizes a plan: empty titles are dropped, statuses are
// normalized, at most one step stays active; when no step is active and the
// plan is not fully done, the first pending step after the done prefix
// becomes active.
func normalizePlan(plan []storage.TaskStep) []storage.TaskStep {
	out := make([]storage.TaskStep, 0, len(plan))
	for _, st := range plan {
		st.Title = strings.TrimSpace(st.Title)
		if st.Title == "" {
			continue
		}
		switch strings.ToLower(st.Status) {
		case storage.StepDone:
			st.Status = storage.StepDone
		case storage.StepActive:
			st.Status = storage.StepActive
		default:
			st.Status = storage.StepPending
		}
		out = append(out, st)
	}
	active := 0
	for _, st := range out {
		if st.Status == storage.StepActive {
			active++
		}
	}
	if active == 0 {
		// The first pending step after the done prefix becomes active.
		i := 0
		for i < len(out) && out[i].Status == storage.StepDone {
			i++
		}
		if i < len(out) && out[i].Status == storage.StepPending {
			out[i].Status = storage.StepActive
		}
	}
	if active > 1 {
		seen := false
		for i := range out {
			if out[i].Status != storage.StepActive {
				continue
			}
			if seen {
				out[i].Status = storage.StepPending
			} else {
				seen = true
			}
		}
	}
	return out
}

// planProgress counts the done steps of a plan ("done/total").
func planProgress(plan []storage.TaskStep) (done, total int) {
	total = len(plan)
	for _, st := range plan {
		if st.Status == storage.StepDone {
			done++
		}
	}
	return
}

// ── Server accessors (callers must hold s.mu unless noted) ─────────────────

// taskFSMLocked returns the active branch's task FSM (nil when no task is in
// progress). Callers must hold s.mu.
func (s *Server) taskFSMLocked() *storage.TaskFSM {
	if b := s.activeBranchLocked(); b != nil {
		return b.TaskFSM
	}
	return nil
}

// cloneTaskFSM deep-copies an FSM: branch-state copies must not share the
// pointer with the ancestor they inherit from.
func cloneTaskFSM(ts *storage.TaskFSM) *storage.TaskFSM {
	if ts == nil {
		return nil
	}
	cp := *ts
	cp.Plan = append([]storage.TaskStep(nil), ts.Plan...)
	return &cp
}

// emptyTaskFSM reports whether an FSM carries no state at all (it is dropped
// in that case, so idle chats keep the chip hidden).
func emptyTaskFSM(ts *storage.TaskFSM) bool {
	return ts.Stage == "" && ts.Step == "" && ts.Expect == "" && len(ts.Plan) == 0
}

// applyTaskUpdateLocked applies a user/model update to the active branch's
// FSM under the strict transition rules. An empty stage leaves the stage
// untouched; "reset" clears the FSM (allowed from planning only); a new task
// starts from planning. Returns the resulting FSM (nil after a reset or when
// the FSM carries no state yet) or a transition error. Callers must hold s.mu.
func (s *Server) applyTaskUpdateLocked(in *storage.TaskFSM) (*storage.TaskFSM, error) {
	b := s.ensureBranchStateLocked()
	// cur живёт локально: при ошибке перехода b.TaskFSM не трогаем, иначе
	// отклонённое обновление оставило бы пустой (но непустой-указатель) FSM,
	// который persist записал бы в базу как {"stage":""}.
	cur := b.TaskFSM
	if cur == nil {
		cur = &storage.TaskFSM{}
	}
	if in != nil {
		switch stage := strings.TrimSpace(in.Stage); {
		case stage == "":
			// untouched
		case stage == "reset":
			if !canTransition(cur.Stage, "") {
				return nil, fmt.Errorf("задачу можно сбросить только из этапа planning (сейчас: %s; допустимые переходы: %s)",
					orStage(cur.Stage), allowedFrom(cur.Stage))
			}
			b.TaskFSM = nil
			return nil, nil
		case validStage(stage):
			if stage != cur.Stage && !canTransition(cur.Stage, stage) {
				return nil, fmt.Errorf("запрещённый переход %s → %s (допустимые: %s)",
					orStage(cur.Stage), stage, allowedFrom(cur.Stage))
			}
			// Гейт плана: реализация (execution) начинается только после
			// утверждённого плана — без плана planning → execution
			// отклоняется кодом, а не промптом. План, задаваемый тем же
			// обновлением, считается утверждённым (инвариант «execution без
			// плана» сохраняется). Откат validation → execution гейтом не
			// трогается (план к тому моменту уже есть).
			plan := cur.Plan
			if in.Plan != nil {
				plan = normalizePlan(in.Plan)
			}
			if stage == StageExecution && cur.Stage == StagePlanning && len(plan) == 0 {
				return nil, fmt.Errorf("реализацию нельзя начать до утверждённого плана: сначала задай план (action=plan или чек-лист в панели «Память»)")
			}
			// Гейт done: финал достижим только с полностью выполненным
			// планом — иначе задача «выполнялась» без реальной работы
			// (действует и для модели, и для UI; пустой план не мешает).
			if stage == StageDone {
				if done, total := planProgress(plan); total > 0 && done < total {
					return nil, fmt.Errorf("завершить задачу нельзя: план выполнен не полностью (%d/%d) — закрой шаги плана (клики по шагам или action=plan с completed) или сократи план", done, total)
				}
			}
			cur.Stage = stage
		default:
			return nil, fmt.Errorf("неизвестный этап %q (допустимы: planning, execution, validation, done, reset)", stage)
		}
		if step := strings.TrimSpace(in.Step); step != "" {
			cur.Step = step
		}
		if expect := strings.TrimSpace(in.Expect); expect != "" {
			cur.Expect = expect
		}
		if in.Plan != nil {
			cur.Plan = normalizePlan(in.Plan)
		}
	}
	if emptyTaskFSM(cur) {
		b.TaskFSM = nil
		return nil, nil
	}
	b.TaskFSM = cur
	return b.TaskFSM, nil
}

// ── Rendering (prompts and tool results) ────────────────────────────────────

// taskStateText renders the FSM for the model (the task_state tool result).
func taskStateText(ts *storage.TaskFSM) string {
	if ts == nil || emptyTaskFSM(ts) {
		return "Задачи нет (состояние задачи пустое)."
	}
	var b strings.Builder
	b.WriteString("Этап: " + orStage(ts.Stage) + "\n")
	if ts.Step != "" {
		b.WriteString("Текущий шаг: " + ts.Step + "\n")
	}
	if ts.Expect != "" {
		b.WriteString("Ожидаемое действие: " + ts.Expect + "\n")
	}
	if len(ts.Plan) > 0 {
		done, total := planProgress(ts.Plan)
		fmt.Fprintf(&b, "План (%d/%d):\n", done, total)
		for i, st := range ts.Plan {
			mark := "○"
			switch st.Status {
			case storage.StepDone:
				mark = "✓"
			case storage.StepActive:
				mark = "●"
			}
			fmt.Fprintf(&b, "%d) %s %s\n", i+1, mark, st.Title)
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

// taskFSMPromptBlock renders the FSM as a working-memory system-prompt block.
func taskFSMPromptBlock(ts *storage.TaskFSM) string {
	if ts == nil || emptyTaskFSM(ts) {
		return ""
	}
	var b strings.Builder
	b.WriteString("Состояние задачи (конечный автомат planning → execution → validation → done):\n")
	b.WriteString("- Этап: " + orStage(ts.Stage) + "\n")
	if ts.Step != "" {
		b.WriteString("- Текущий шаг: " + ts.Step + "\n")
	}
	if ts.Expect != "" {
		b.WriteString("- Ожидаемое действие: " + ts.Expect + "\n")
	}
	if len(ts.Plan) > 0 {
		done, total := planProgress(ts.Plan)
		parts := make([]string, 0, len(ts.Plan))
		for i, st := range ts.Plan {
			p := fmt.Sprintf("%d) %s", i+1, st.Title)
			switch st.Status {
			case storage.StepDone:
				p += " [готово]"
			case storage.StepActive:
				p += " [активен]"
			}
			parts = append(parts, p)
		}
		fmt.Fprintf(&b, "- План (%d/%d): %s\n", done, total, strings.Join(parts, "; "))
	}
	b.WriteString("Смена этапа или шага — только вызовом инструмента task_state; не объявляй смену состояния без вызова инструмента. За один ответ доступен не более одного перехода этапа; done — только когда все шаги плана выполнены.")
	// Этап — отражение фактической фазы работы, а не намерений: иначе задача
	// «зависает» в planning с готовым планом (наблюдаемый дефект).
	b.WriteString("\nЭтап отражает фактическую фазу работы: план составлен и подтверждён пользователем — сразу action=advance в execution и выполняй шаги там; не оставляй задачу в planning с утверждённым планом и не выполняй шаги плана до перехода в execution.")
	if ts.Stage == StagePlanning {
		if len(ts.Plan) == 0 {
			b.WriteString("\nПереход в execution заблокирован до утверждённого плана: сначала action=plan.")
		} else {
			b.WriteString("\nПлан уже составлен: покажи его пользователю; после подтверждения сразу action=advance — реализация идёт в execution, а не в planning.")
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

// ── Compression lines (ЭТАП:/ШАГ:/ОЖИДАНИЕ:) ────────────────────────────────

// titleCase uppercases the first letter of a word ("этап" → "Этап").
func titleCase(w string) string {
	r := []rune(w)
	if len(r) == 0 {
		return w
	}
	return strings.ToUpper(string(r[0])) + string(r[1:])
}

// cutStateLine strips a «WORD: …» prefix in any letter case (WORD — one of
// the compression FSM words); ok=false means the line is not that FSM line.
func cutStateLine(line, word string) (string, bool) {
	caps := strings.ToUpper(word)
	low := strings.ToLower(word)
	for _, p := range []string{caps + ":", titleCase(low) + ":", low + ":"} {
		if v, ok := strings.CutPrefix(line, p); ok {
			return strings.TrimSpace(v), true
		}
	}
	return "", false
}

// splitTaskStateLines extracts the «ЭТАП: …», «ШАГ: …», «ОЖИДАНИЕ: …» lines
// (any letter case) from a compression response — the task FSM state the
// model is asked to carry across compression. Returns the parsed FSM (nil
// when no FSM line is present or all values are empty) and the remaining text.
func splitTaskStateLines(text string) (*storage.TaskFSM, string) {
	var ts *storage.TaskFSM
	apply := func(field string, value string) {
		if value == "" {
			return
		}
		if ts == nil {
			ts = &storage.TaskFSM{}
		}
		switch field {
		case "stage":
			ts.Stage = value
		case "step":
			ts.Step = value
		case "expect":
			ts.Expect = value
		}
	}
	var rest strings.Builder
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if v, ok := cutStateLine(line, "ЭТАП"); ok {
			apply("stage", v)
			continue
		}
		if v, ok := cutStateLine(line, "ШАГ"); ok {
			apply("step", v)
			continue
		}
		if v, ok := cutStateLine(line, "ОЖИДАНИЕ"); ok {
			apply("expect", v)
			continue
		}
		rest.WriteString(line + "\n")
	}
	if ts != nil && emptyTaskFSM(ts) {
		ts = nil
	}
	return ts, strings.TrimRight(rest.String(), "\n")
}

// applyCompressionFSM merges the FSM carried by a compression response into
// the branch state leniently. Only the step/expect text is refreshed, and
// only for an already running task: the summary is a retelling of history,
// not an action — if it could move the stage, the «ЭТАП: done» line from an
// old task's tail would close a freshly started one (seen in practice). The
// stage moves only through task_state, under the strict transition rules and
// the one-move-per-exchange budget.
func applyCompressionFSM(b *branchState, ts *storage.TaskFSM) {
	if ts == nil || b.TaskFSM == nil {
		return
	}
	if ts.Step != "" {
		b.TaskFSM.Step = ts.Step
	}
	if ts.Expect != "" {
		b.TaskFSM.Expect = ts.Expect
	}
}

// ── HTTP API ────────────────────────────────────────────────────────────────

// handleTaskGet serves GET /api/task — the active branch's task FSM
// (task=null when no task is in progress).
func (s *Server) handleTaskGet(w http.ResponseWriter, _ *http.Request) {
	s.mu.Lock()
	ts := s.taskFSMLocked()
	s.mu.Unlock()
	writeJSON(w, map[string]any{"ok": true, "task": ts})
}

// handleTaskUpdate serves POST /api/task with a TaskFSM body (stage, step,
// expect, plan): the user-driven FSM update under the same strict rules as
// the model's tool. Returns the resulting FSM.
func (s *Server) handleTaskUpdate(w http.ResponseWriter, r *http.Request) {
	var in storage.TaskFSM
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(&in); err != nil {
		http.Error(w, "Некорректное тело запроса (ожидается JSON): "+err.Error(), http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	ts, err := s.applyTaskUpdateLocked(&in)
	s.mu.Unlock()
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.persist()
	writeJSON(w, map[string]any{"ok": true, "task": ts})
}
