// The onboarding state machine: a new user is interviewed (stage interview)
// before the assistant answers anything substantial, the answers are stored
// as profile facts of the tab's long-term memory, then confirmed (stage
// confirm) and the FSM ends in done — a persisted marker that keeps a
// skipped questionnaire from restarting. Like the task FSM (task.go) the
// automaton is code-driven: the model only proposes, the code validates
// every transition and gates the tools down to onboarding_* while the
// questionnaire runs. The HTTP API (GET/POST /api/onboarding) drives it from
// the "Память" panel.
package server

import (
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"

	"ai-chat-go/api"
	"ai-chat-go/storage"
)

//go:embed onboarding_questions.txt
var onboardingScriptFS embed.FS

// Onboarding FSM stages: interview — вопросы по одному из скрипта; confirm —
// сводка собранного профиля и правки ответов по ключам; done — финал,
// сохраняется в базе как маркер «анкета пройдена/отклонена» (иначе анкета с
// пропущенными вопросами перезапустилась бы на каждом сообщении).
const (
	onbStageInterview = "interview"
	onbStageConfirm   = "confirm"
	onbStageDone      = "done"
)

// onbQuestion — один вопрос анкеты: ключ записи профиля в долговременной
// памяти, текст вопроса и обязательность (обязательный вопрос нельзя
// пропустить по одному — только ответить или прервать всю анкету).
type onbQuestion struct {
	Key      string
	Text     string
	Required bool
}

// onboardingScript parses the embedded questionnaire once; a broken script
// disables the onboarding (logged), the rest of the server works as before.
var onboardingScript = sync.OnceValue(func() []onbQuestion {
	data, err := onboardingScriptFS.ReadFile("onboarding_questions.txt")
	if err != nil {
		log.Printf("onboarding: скрипт вопросов недоступен: %v; анкетирование отключено", err)
		return nil
	}
	qs, err := parseOnboardingQuestions(string(data))
	if err != nil {
		log.Printf("onboarding: некорректный скрипт вопросов (%v); анкетирование отключено", err)
		return nil
	}
	return qs
})

// parseOnboardingQuestions parses the «Ключ:/Вопрос:/Обязательный:» blocks
// (blank-line separated) validating every block: non-empty key and text,
// unique keys, Обязательный — только «да»/«нет».
func parseOnboardingQuestions(text string) ([]onbQuestion, error) {
	var out []onbQuestion
	seen := map[string]bool{}
	flush := func(cur onbQuestion) error {
		cur.Key = strings.ToLower(strings.TrimSpace(cur.Key))
		cur.Text = strings.TrimSpace(cur.Text)
		switch {
		case cur.Key == "" && cur.Text == "":
			return nil // пустой блок пропускаем
		case cur.Key == "":
			return fmt.Errorf("блок без ключа (вопрос %q)", cur.Text)
		case cur.Text == "":
			return fmt.Errorf("у вопроса %q нет текста", cur.Key)
		case seen[cur.Key]:
			return fmt.Errorf("дублирующийся ключ %q", cur.Key)
		}
		seen[cur.Key] = true
		out = append(out, cur)
		return nil
	}
	var cur onbQuestion
	var started bool
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimRight(line, "\r")
		switch {
		case strings.HasPrefix(line, "Ключ:"):
			if started {
				if err := flush(cur); err != nil {
					return nil, err
				}
				cur = onbQuestion{}
			}
			started = true
			cur.Key = strings.TrimSpace(strings.TrimPrefix(line, "Ключ:"))
		case strings.HasPrefix(line, "Вопрос:"):
			cur.Text = strings.TrimSpace(strings.TrimPrefix(line, "Вопрос:"))
		case strings.HasPrefix(line, "Обязательный:"):
			v := strings.ToLower(strings.TrimSpace(strings.TrimPrefix(line, "Обязательный:")))
			switch v {
			case "да", "yes":
				cur.Required = true
			case "нет", "no":
				cur.Required = false
			default:
				return nil, fmt.Errorf("у вопроса %q неверное значение Обязательный: %q (да/нет)", cur.Key, v)
			}
		}
	}
	if started {
		if err := flush(cur); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// onbStageNext — таблица переходов анкеты (строгий автомат):
//
//	""        → interview, done  (старт анкеты; done ставит UI/инструмент skip)
//	interview → interview (следующий вопрос), confirm (все закрыты), done (skip all)
//	confirm   → confirm (правка ответа по ключу), done (подтверждено)
//	done      → (финал; рестарт — только вручную через POST /api/onboarding)
var onbStageNext = map[string][]string{
	"":                {onbStageInterview, onbStageDone},
	onbStageInterview: {onbStageInterview, onbStageConfirm, onbStageDone},
	onbStageConfirm:   {onbStageConfirm, onbStageDone},
	onbStageDone:      {},
}

// onbCanTransition reports whether from→to is allowed by the automaton.
func onbCanTransition(from, to string) bool {
	for _, next := range onbStageNext[from] {
		if next == to {
			return true
		}
	}
	return false
}

// onbAllowedFrom lists the transitions available from a stage (for errors).
func onbAllowedFrom(from string) string {
	if next := onbStageNext[from]; len(next) == 0 {
		return "финал (перезапуск только через POST /api/onboarding action=restart)"
	} else {
		return strings.Join(next, ", ")
	}
}

// ── Server accessors (callers must hold s.mu unless noted) ─────────────────

// onboardingFSMLocked returns the active branch's onboarding FSM (nil when
// the questionnaire never ran here). Callers must hold s.mu.
func (s *Server) onboardingFSMLocked() *storage.OnboardingFSM {
	if b := s.activeBranchLocked(); b != nil {
		return b.Onboarding
	}
	return nil
}

// cloneOnboardingFSM deep-copies an FSM: branch-state copies must not share
// the pointer with the ancestor they inherit from.
func cloneOnboardingFSM(onb *storage.OnboardingFSM) *storage.OnboardingFSM {
	if onb == nil {
		return nil
	}
	cp := *onb
	return &cp
}

// emptyOnboardingFSM reports whether the FSM carries no state (nil or the
// legacy {"stage":""} blob) — treated exactly like "no questionnaire".
func emptyOnboardingFSM(onb *storage.OnboardingFSM) bool {
	return onb == nil || onb.Stage == ""
}

// onbRunning reports whether the questionnaire is in progress (interview or
// confirm) — the mode that gates the tools and drives the prompt block.
func onbRunning(onb *storage.OnboardingFSM) bool {
	return !emptyOnboardingFSM(onb) && onb.Stage != onbStageDone
}

// setOnboardingLocked stores the FSM on the active branch, clamping the step
// into the script range (an interview step beyond the script — e.g. after a
// shortened questionnaire file — lands in confirm); an empty stage drops the
// FSM. Returns the stored FSM (nil after a drop). Callers must hold s.mu.
func (s *Server) setOnboardingLocked(onb *storage.OnboardingFSM) *storage.OnboardingFSM {
	b := s.ensureBranchStateLocked()
	if onb == nil || onb.Stage == "" {
		b.Onboarding = nil
		return nil
	}
	if onb.Stage == onbStageInterview {
		if onb.Step < 0 {
			onb.Step = 0
		}
		if onb.Step >= len(onboardingScript()) {
			onb.Stage = onbStageConfirm
		}
	}
	b.Onboarding = onb
	return b.Onboarding
}

// maybeStartOnboardingLocked starts the questionnaire on the active branch
// when it has never run here (no FSM) and the tab's profile memory is empty:
// the profile of a tab IS the layer-3 memory, so a tab with a filled profile
// is never re-interviewed, and a done-marker blocks a restart of a skipped
// questionnaire. Callers must hold s.mu.
func (s *Server) maybeStartOnboardingLocked(session string) {
	if len(onboardingScript()) == 0 {
		return
	}
	if s.onboardingFSMLocked() != nil {
		return
	}
	for _, it := range s.memoryCacheLocked(session).items {
		if it.Kind == storage.KindProfile {
			return // профиль уже есть — анкета не нужна
		}
	}
	s.setOnboardingLocked(&storage.OnboardingFSM{Stage: onbStageInterview})
}

// onboardingAnswersLocked maps the script keys onto the collected profile
// facts of the tab (missing keys are absent from the map). Callers must hold
// s.mu.
func (s *Server) onboardingAnswersLocked(session string) map[string]storage.MemoryItem {
	out := map[string]storage.MemoryItem{}
	for _, it := range s.memoryCacheLocked(session).items {
		if it.Kind == storage.KindProfile {
			out[it.Key] = it
		}
	}
	return out
}

// upsertOnboardingAnswerLocked writes one answer as a profile fact (source
// onboarding, durable from the start like an explicit write). Callers must
// hold s.mu.
func (s *Server) upsertOnboardingAnswerLocked(session, key, answer string) {
	s.upsertMemoryLocked(session, storage.MemoryItem{
		Kind:          storage.KindProfile,
		Key:           key,
		Text:          key + ": " + answer,
		Source:        storage.SourceOnboarding,
		Confirmations: 2,
	})
}

// ── Rendering (prompts and tool results) ────────────────────────────────────

// onbStageOrEmpty renders a stage for humans ("нет анкеты" when empty).
func onbStageOrEmpty(onb *storage.OnboardingFSM) string {
	if emptyOnboardingFSM(onb) {
		return "нет анкеты"
	}
	return onb.Stage
}

// onbKeys lists the script keys (for error messages).
func onbKeys(script []onbQuestion) string {
	ks := make([]string, 0, len(script))
	for _, q := range script {
		ks = append(ks, q.Key)
	}
	return strings.Join(ks, ", ")
}

// onbByKey finds a script question by key.
func onbByKey(script []onbQuestion, key string) (onbQuestion, bool) {
	for _, q := range script {
		if q.Key == key {
			return q, true
		}
	}
	return onbQuestion{}, false
}

// onboardingSummaryLocked renders the collected profile (the confirm-mode
// сводка and the tool results). Callers must hold s.mu.
func (s *Server) onboardingSummaryLocked(session string) string {
	script := onboardingScript()
	answers := s.onboardingAnswersLocked(session)
	var lines []string
	for _, q := range script {
		if it, ok := answers[q.Key]; ok {
			lines = append(lines, "- "+it.Text)
			continue
		}
		if q.Required {
			lines = append(lines, "- "+q.Key+": (нет ответа)")
		}
	}
	if len(lines) == 0 {
		return "Профиль пуст (все вопросы пропущены)."
	}
	return "Собранный профиль:\n" + strings.Join(lines, "\n")
}

// onboardingPromptBlockLocked renders the working-memory block that pins the
// model to the questionnaire mode; done (or no FSM) yields an empty block —
// the chat goes on as usual. Callers must hold s.mu.
func (s *Server) onboardingPromptBlockLocked(session string) string {
	onb := s.onboardingFSMLocked()
	if !onbRunning(onb) || len(onboardingScript()) == 0 {
		return ""
	}
	script := onboardingScript()
	var b strings.Builder
	switch onb.Stage {
	case onbStageInterview:
		if onb.Step >= len(script) {
			return "" // не должно случаться (клэмп в setOnboardingLocked)
		}
		q := script[onb.Step]
		fmt.Fprintf(&b, "РЕЖИМ АНКЕТИРОВАНИЯ нового пользователя (этап: interview, вопрос %d/%d).\n", onb.Step+1, len(script))
		fmt.Fprintf(&b, "Задай пользователю вопрос: «%s»\n", q.Text)
		b.WriteString("Правила:\n")
		b.WriteString("- на посторонние вопросы не отвечай, мягко возвращайся к анкете;\n")
		b.WriteString("- ответ пользователя сохраняй инструментом onboarding_answer (answer — суть ответа одной фразой);\n")
		b.WriteString("- если в одном сообщении ответы на несколько вопросов — несколько вызовов onboarding_answer подряд;\n")
		b.WriteString("- необязательный вопрос можно пропустить инструментом onboarding_skip; all=true прерывает всю анкету;\n")
		b.WriteString("Пока анкета идёт, остальные инструменты недоступны.")
	case onbStageConfirm:
		b.WriteString("РЕЖИМ АНКЕТИРОВАНИЯ нового пользователя (этап: confirm — проверка собранного профиля).\n")
		b.WriteString(s.onboardingSummaryLocked(session) + "\n")
		b.WriteString("Правила:\n")
		b.WriteString("- покажи сводку пользователю и спроси, всё ли верно;\n")
		b.WriteString("- поправить ответ: onboarding_answer с answer и key (ключ вопроса); анкета остаётся в confirm;\n")
		b.WriteString("- завершить анкету: onboarding_confirm (или onboarding_skip);\n")
		b.WriteString("Пока анкета идёт, остальные инструменты недоступны.")
	}
	return strings.TrimRight(b.String(), "\n")
}

// ── Tool gate ───────────────────────────────────────────────────────────────

// onboardingToolGate blocks every tool except the onboarding_* ones while the
// questionnaire runs (interview/confirm): the model proposes — the code
// disposes. done (or no FSM) leaves all tools open.
func (s *Server) onboardingToolGate(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if onb := s.onboardingFSMLocked(); onbRunning(onb) {
		return fmt.Errorf("инструмент %s недоступен: идёт анкетирование нового пользователя (этап %s); используй onboarding_answer / onboarding_skip / onboarding_confirm", name, onb.Stage)
	}
	return nil
}

// ── Tools ───────────────────────────────────────────────────────────────────

// onboardingDefinitions returns the tool definitions of the questionnaire;
// they are exposed to the model only while the FSM runs (see toolDefinitions).
func onboardingDefinitions() []api.Tool {
	return []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "onboarding_answer",
				Description: "Сохраняет ответ на вопрос анкеты нового пользователя. В режиме interview key не нужен — сохраняется ответ на текущий вопрос; в режиме confirm key обязателен (ключ вопроса из сводки).",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"answer": map[string]any{
							"type":        "string",
							"description": "Ответ пользователя одной фразой, например «Аня».",
						},
						"key": map[string]any{
							"type":        "string",
							"description": "Ключ вопроса (только для правки ответа в режиме confirm).",
						},
					},
					"required": []string{"answer"},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "onboarding_skip",
				Description: "Пропускает текущий необязательный вопрос анкеты или всю анкету (all=true). В режиме confirm завершает анкету без явного подтверждения.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"all": map[string]any{
							"type":        "boolean",
							"description": "true — прервать всю анкету (этап done); false — пропустить только текущий вопрос.",
						},
					},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "onboarding_confirm",
				Description: "Завершает анкету из режима confirm: профиль подтверждён и остаётся в долговременной памяти.",
				Parameters:  map[string]any{"type": "object", "properties": map[string]any{}},
			},
		},
	}
}

// toolOnboardingAnswer stores one questionnaire answer as a profile fact of
// the tab's long-term memory and advances the FSM: interview → the next
// question (confirm when the script is over); confirm → the answer is
// rewritten by key, the FSM stays. The result names the next step, so the
// model sees the effect of its call.
func (s *Server) toolOnboardingAnswer(session string, args json.RawMessage) (string, error) {
	var in struct {
		Answer string `json:"answer"`
		Key    string `json:"key"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", fmt.Errorf("некорректные аргументы: %w", err)
	}
	answer := strings.TrimSpace(in.Answer)
	if answer == "" {
		return "", fmt.Errorf("пустой ответ")
	}
	script := onboardingScript()
	if len(script) == 0 {
		return "", fmt.Errorf("анкета не настроена")
	}
	s.mu.Lock()
	// persist() сам берёт s.mu, поэтому вызывается в defer уже после
	// разблокировки — и только когда ход действительно состоялся.
	changed := false
	defer func() {
		s.mu.Unlock()
		if changed {
			s.persist()
		}
	}()
	onb := s.onboardingFSMLocked()
	if !onbRunning(onb) {
		return "", fmt.Errorf("анкета не активна (состояние: %s)", onbStageOrEmpty(onb))
	}
	switch onb.Stage {
	case onbStageInterview:
		q := script[onb.Step]
		s.upsertOnboardingAnswerLocked(session, q.Key, answer)
		onb.Step++
		if onb.Step >= len(script) {
			onb.Stage = onbStageConfirm
			s.setOnboardingLocked(onb)
			changed = true
			return "ответ принят. Все вопросы закрыты — переход к подтверждению профиля.\n" +
				s.onboardingSummaryLocked(session), nil
		}
		changed = true
		return fmt.Sprintf("ответ принят. Следующий вопрос (%d/%d): %s",
			onb.Step+1, len(script), script[onb.Step].Text), nil
	case onbStageConfirm:
		key := strings.ToLower(strings.TrimSpace(in.Key))
		if key == "" {
			return "", fmt.Errorf("в режиме confirm нужен key вопроса (один из: %s)", onbKeys(script))
		}
		if _, ok := onbByKey(script, key); !ok {
			return "", fmt.Errorf("неизвестный ключ %q (возможные: %s)", key, onbKeys(script))
		}
		s.upsertOnboardingAnswerLocked(session, key, answer)
		changed = true
		return "ответ обновлён, анкета остаётся в режиме подтверждения.\n" +
			s.onboardingSummaryLocked(session), nil
	}
	return "", fmt.Errorf("неизвестный этап анкеты %q", onb.Stage)
}

// toolOnboardingSkip skips the current question (interview; only an optional
// one — a required question is passed with an answer or the whole
// questionnaire is abandoned via all=true), or finishes the questionnaire.
func (s *Server) toolOnboardingSkip(session string, args json.RawMessage) (string, error) {
	var in struct {
		All bool `json:"all"`
	}
	if len(args) > 0 {
		if err := json.Unmarshal(args, &in); err != nil {
			return "", fmt.Errorf("некорректные аргументы: %w", err)
		}
	}
	script := onboardingScript()
	if len(script) == 0 {
		return "", fmt.Errorf("анкета не настроена")
	}
	s.mu.Lock()
	// persist() сам берёт s.mu — см. toolOnboardingAnswer.
	changed := false
	defer func() {
		s.mu.Unlock()
		if changed {
			s.persist()
		}
	}()
	onb := s.onboardingFSMLocked()
	if !onbRunning(onb) {
		return "", fmt.Errorf("анкета не активна (состояние: %s)", onbStageOrEmpty(onb))
	}
	switch onb.Stage {
	case onbStageInterview:
		if in.All {
			s.setOnboardingLocked(&storage.OnboardingFSM{Stage: onbStageDone, Step: onb.Step})
			changed = true
			return "анкета прервана: профиль не собран, вопросы больше не задаются.", nil
		}
		q := script[onb.Step]
		if q.Required {
			return "", fmt.Errorf("вопрос «%s» обязательный — пропустить его по одному нельзя; ответь через onboarding_answer или прерви всю анкету (onboarding_skip, all=true)", q.Text)
		}
		onb.Step++
		if onb.Step >= len(script) {
			onb.Stage = onbStageConfirm
			s.setOnboardingLocked(onb)
			changed = true
			return "вопрос пропущен. Все вопросы закрыты — переход к подтверждению профиля.\n" +
				s.onboardingSummaryLocked(session), nil
		}
		changed = true
		return fmt.Sprintf("вопрос пропущен. Следующий вопрос (%d/%d): %s",
			onb.Step+1, len(script), script[onb.Step].Text), nil
	case onbStageConfirm:
		s.setOnboardingLocked(&storage.OnboardingFSM{Stage: onbStageDone, Step: onb.Step})
		changed = true
		return "анкета завершена без явного подтверждения: собранные ответы остались в памяти профиля.", nil
	}
	return "", fmt.Errorf("неизвестный этап анкеты %q", onb.Stage)
}

// toolOnboardingConfirm finishes the questionnaire from confirm (the strict
// automaton rejects it from interview).
func (s *Server) toolOnboardingConfirm(session string) (string, error) {
	s.mu.Lock()
	// persist() сам берёт s.mu — см. toolOnboardingAnswer.
	changed := false
	defer func() {
		s.mu.Unlock()
		if changed {
			s.persist()
		}
	}()
	onb := s.onboardingFSMLocked()
	if emptyOnboardingFSM(onb) {
		return "", fmt.Errorf("анкета не активна")
	}
	if onb.Stage != onbStageConfirm {
		return "", fmt.Errorf("подтвердить можно только из этапа confirm (сейчас: %s; допустимые переходы: %s)",
			onb.Stage, onbAllowedFrom(onb.Stage))
	}
	s.setOnboardingLocked(&storage.OnboardingFSM{Stage: onbStageDone, Step: onb.Step})
	changed = true
	return "профиль подтверждён, анкета завершена. Теперь отвечай на вопросы пользователя как обычно.", nil
}

// ── HTTP API ────────────────────────────────────────────────────────────────

// handleOnboardingGet serves GET /api/onboarding — the active branch's
// onboarding FSM (null when the questionnaire never ran here) plus the
// script size.
func (s *Server) handleOnboardingGet(w http.ResponseWriter, _ *http.Request) {
	s.mu.Lock()
	onb := s.onboardingFSMLocked()
	total := len(onboardingScript())
	s.mu.Unlock()
	writeJSON(w, map[string]any{"ok": true, "onboarding": onb, "total": total})
}

// handleOnboardingUpdate serves POST /api/onboarding {"action": ...} — the
// user-driven automaton (the "Память" panel buttons): skip finishes the
// questionnaire, confirm accepts the profile, restart starts it over (the
// only way back from done — the model cannot re-open a closed questionnaire).
func (s *Server) handleOnboardingUpdate(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Action string `json:"action"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(&in); err != nil {
		http.Error(w, "Некорректное тело запроса (ожидается JSON): "+err.Error(), http.StatusBadRequest)
		return
	}
	if len(onboardingScript()) == 0 {
		http.Error(w, "Анкета не настроена (пустой скрипт вопросов)", http.StatusConflict)
		return
	}
	s.mu.Lock()
	var onb *storage.OnboardingFSM
	switch in.Action {
	case "skip":
		onb = s.setOnboardingLocked(&storage.OnboardingFSM{Stage: onbStageDone})
	case "confirm":
		cur := s.onboardingFSMLocked()
		if !onbRunning(cur) || cur.Stage != onbStageConfirm {
			s.mu.Unlock()
			http.Error(w, "Подтвердить можно только из этапа confirm", http.StatusBadRequest)
			return
		}
		onb = s.setOnboardingLocked(&storage.OnboardingFSM{Stage: onbStageDone, Step: cur.Step})
	case "restart":
		onb = s.setOnboardingLocked(&storage.OnboardingFSM{Stage: onbStageInterview})
	default:
		s.mu.Unlock()
		http.Error(w, "Неизвестное действие "+in.Action+" (допустимы: skip, confirm, restart)", http.StatusBadRequest)
		return
	}
	s.mu.Unlock()
	s.persist()
	writeJSON(w, map[string]any{"ok": true, "onboarding": onb})
}
