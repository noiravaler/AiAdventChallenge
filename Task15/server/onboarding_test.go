package server

// Tests of the onboarding state machine (onboarding.go): the questionnaire
// script parser, the strict transition table, the interview → confirm → done
// cycle driven by the onboarding_* tools, the tool gate that locks the other
// tools while the questionnaire runs, the HTTP API, the reset scopes and
// SQLite persistence across a restart.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"ai-chat-go/storage"
)

// ── Questionnaire script ────────────────────────────────────────────────────

// TestOnboardingScriptEmbedded pins the shipped questionnaire: five questions
// with the expected keys, four of them required (особенности — optional).
func TestOnboardingScriptEmbedded(t *testing.T) {
	script := onboardingScript()
	if len(script) != 5 {
		t.Fatalf("вопросов в скрипте: %d, want 5", len(script))
	}
	wantKeys := []string{"имя", "роль", "цель", "стиль", "особенности"}
	wantReq := []bool{true, true, true, true, false}
	for i, q := range script {
		if q.Key != wantKeys[i] {
			t.Fatalf("вопрос %d: ключ %q, want %q", i, q.Key, wantKeys[i])
		}
		if q.Text == "" {
			t.Fatalf("вопрос %d (%s): пустой текст", i, q.Key)
		}
		if q.Required != wantReq[i] {
			t.Fatalf("вопрос %d (%s): required = %v, want %v", i, q.Key, q.Required, wantReq[i])
		}
	}
}

// TestParseOnboardingQuestionsErrors covers the strict validation: a keyless
// block, a missing question text, a duplicate key and a wrong Обязательный
// value are all parse errors; a valid file parses.
func TestParseOnboardingQuestionsErrors(t *testing.T) {
	cases := map[string]string{
		"no key":        "Ключ: \nВопрос: Как дела?\nОбязательный: да",
		"no text":       "Ключ: имя\nОбязательный: да",
		"duplicate key": "Ключ: имя\nВопрос: A?\nОбязательный: да\n\nКлюч: имя\nВопрос: B?\nОбязательный: да",
		"bad required":  "Ключ: имя\nВопрос: A?\nОбязательный: возможно",
	}
	for name, text := range cases {
		if qs, err := parseOnboardingQuestions(text); err == nil {
			t.Fatalf("%s: ошибку не вернули (распознано %d вопросов)", name, len(qs))
		}
	}
	qs, err := parseOnboardingQuestions("Ключ: имя\nВопрос: Как к вам обращаться?\nОбязательный: да\n\nКлюч: стиль\nВопрос: Как отвечать?\nОбязательный: нет")
	if err != nil || len(qs) != 2 || !qs[0].Required || qs[1].Required {
		t.Fatalf("valid: %d вопросов, %v", len(qs), err)
	}
}

// ── Transition table ────────────────────────────────────────────────────────

// TestOnbCanTransitionMatrix pins the automaton: "" → interview (done ставит
// UI), interview → interview/confirm/done, confirm → confirm/done, done —
// финал (рестарт только через /api/onboarding).
func TestOnbCanTransitionMatrix(t *testing.T) {
	cases := []struct {
		from, to string
		want     bool
	}{
		{"", onbStageInterview, true},
		{"", onbStageConfirm, false},
		{onbStageInterview, onbStageInterview, true},
		{onbStageInterview, onbStageConfirm, true},
		{onbStageInterview, onbStageDone, true},
		{onbStageInterview, "", false},
		{onbStageConfirm, onbStageConfirm, true},
		{onbStageConfirm, onbStageInterview, false},
		{onbStageConfirm, onbStageDone, true},
		{onbStageDone, onbStageInterview, false}, // рестарт только через UI
		{onbStageDone, onbStageConfirm, false},
		{onbStageDone, onbStageDone, false},
	}
	for _, c := range cases {
		if got := onbCanTransition(c.from, c.to); got != c.want {
			t.Errorf("onbCanTransition(%q, %q) = %v, want %v", c.from, c.to, got, c.want)
		}
	}
}

// ── Tool-driven cycle (white-box) ───────────────────────────────────────────

// startOnboarding starts the questionnaire on the test server (white-box:
// handleChat triggers it only with a live model round-trip). The auto-start
// toggle defaults to off, so the tests switch it on explicitly.
func startOnboarding(t *testing.T, s *Server) string {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.onboardingOn = true
	session := s.activeSession
	s.maybeStartOnboardingLocked(session)
	onb := s.onboardingFSMLocked()
	if onb == nil || onb.Stage != onbStageInterview || onb.Step != 0 {
		t.Fatalf("анкета не началась: %+v", onb)
	}
	return session
}

// TestOnboardingInterviewCycle drives the whole questionnaire through the
// tool executor: four answers reach confirm, a keyed answer edits one entry,
// onboarding_confirm finishes; every answer lands in the profile memory as a
// durable onboarding fact.
func TestOnboardingInterviewCycle(t *testing.T) {
	s := newTaskTestServer(t, "")
	session := startOnboarding(t, s)
	exec := s.memoryToolExec(session)

	// Первый ответ — на вопрос «имя»; результат ведёт к вопросу 2/5.
	res, err := exec("onboarding_answer", json.RawMessage(`{"answer":"Аня"}`))
	if err != nil {
		t.Fatalf("onboarding_answer: %v", err)
	}
	if !strings.Contains(res, "2/5") {
		t.Fatalf("результат не ведёт к вопросу 2/5: %q", res)
	}

	// Гейт: пока анкета идёт, посторонние инструменты закрыты.
	if _, err := exec("remember", json.RawMessage(`{"text":"тест"}`)); err == nil || !strings.Contains(err.Error(), "недоступен") {
		t.Fatalf("гейт пропустил remember: %v", err)
	}

	// confirm из interview невозможен.
	if _, err := exec("onboarding_confirm", nil); err == nil {
		t.Fatal("onboarding_confirm прошёл из interview")
	}

	// Остальные четыре вопроса (всего пять — включая необязательный).
	for i := 0; i < 4; i++ {
		if _, err := exec("onboarding_answer", json.RawMessage(`{"answer":"ответ"}`)); err != nil {
			t.Fatalf("onboarding_answer #%d: %v", i+2, err)
		}
	}
	s.mu.Lock()
	onb := s.onboardingFSMLocked()
	if onb == nil || onb.Stage != onbStageConfirm {
		s.mu.Unlock()
		t.Fatalf("после 5 ответов этап = %+v, want confirm", onb)
	}
	ans := s.onboardingAnswersLocked(session)
	s.mu.Unlock()
	if len(ans) != 5 {
		t.Fatalf("в памяти %d записей, want 5", len(ans))
	}
	if it := ans["имя"]; it.Text != "имя: Аня" || it.Source != storage.SourceOnboarding ||
		it.Kind != storage.KindProfile || it.Confirmations < 2 {
		t.Fatalf("запись «имя» = %+v", it)
	}

	// Правка по ключу остаётся в confirm.
	if res, err := exec("onboarding_answer", json.RawMessage(`{"answer":"Аня Смирнова","key":"имя"}`)); err != nil || !strings.Contains(res, "подтверждения") {
		t.Fatalf("правка: %q, %v", res, err)
	}

	// Завершение.
	if _, err := exec("onboarding_confirm", nil); err != nil {
		t.Fatalf("onboarding_confirm: %v", err)
	}
	s.mu.Lock()
	onb = s.onboardingFSMLocked()
	s.mu.Unlock()
	if onb == nil || onb.Stage != onbStageDone {
		t.Fatalf("после confirm этап = %+v, want done", onb)
	}

	// После done все инструменты открыты, а анкета не перезапускается.
	if _, err := exec("remember", json.RawMessage(`{"text":"проверка"}`)); err != nil {
		t.Fatalf("после done remember заблокирован: %v", err)
	}
	s.mu.Lock()
	s.maybeStartOnboardingLocked(session)
	onb = s.onboardingFSMLocked()
	s.mu.Unlock()
	if onb == nil || onb.Stage != onbStageDone {
		t.Fatalf("maybeStart перезапустил анкету: %+v", onb)
	}
}

// TestOnboardingSkipRules covers the skip semantics: a required question
// cannot be skipped one by one (only all=true abandons the questionnaire),
// an optional one can; from confirm a skip finishes without confirmation.
func TestOnboardingSkipRules(t *testing.T) {
	s := newTaskTestServer(t, "")
	session := startOnboarding(t, s)
	exec := s.memoryToolExec(session)

	// Вопрос 1 (имя) обязательный — по одному не пропускается.
	if _, err := exec("onboarding_skip", json.RawMessage(`{"all":false}`)); err == nil {
		t.Fatal("обязательный вопрос пропустился")
	}

	// Четыре ответа — и необязательный пятый пропускается в confirm.
	for i := 0; i < 4; i++ {
		if _, err := exec("onboarding_answer", json.RawMessage(`{"answer":"ответ"}`)); err != nil {
			t.Fatalf("onboarding_answer #%d: %v", i+1, err)
		}
	}
	if _, err := exec("onboarding_skip", json.RawMessage(`{}`)); err != nil {
		t.Fatalf("skip optional: %v", err)
	}
	s.mu.Lock()
	onb := s.onboardingFSMLocked()
	s.mu.Unlock()
	if onb == nil || onb.Stage != onbStageConfirm {
		t.Fatalf("после skip optional = %+v, want confirm", onb)
	}

	// Skip из confirm завершает анкету без подтверждения.
	if _, err := exec("onboarding_skip", json.RawMessage(`{}`)); err != nil {
		t.Fatalf("skip из confirm: %v", err)
	}

	// Снова interview (white-box) — skip all=true прерывает анкету целиком,
	// и done-маркер не даёт ей перезапуститься.
	s.mu.Lock()
	s.setOnboardingLocked(&storage.OnboardingFSM{Stage: onbStageInterview})
	s.mu.Unlock()
	if _, err := exec("onboarding_skip", json.RawMessage(`{"all":true}`)); err != nil {
		t.Fatalf("skip all: %v", err)
	}
	s.mu.Lock()
	s.maybeStartOnboardingLocked(session)
	onb = s.onboardingFSMLocked()
	s.mu.Unlock()
	if onb == nil || onb.Stage != onbStageDone {
		t.Fatalf("skip all перезапустился: %+v", onb)
	}
}

// ── HTTP API ────────────────────────────────────────────────────────────────

// onbGET reads GET /api/onboarding.
func onbGET(t *testing.T, url string) *storage.OnboardingFSM {
	t.Helper()
	resp, err := http.Get(url + "/api/onboarding")
	if err != nil {
		t.Fatalf("GET /api/onboarding: %v", err)
	}
	defer resp.Body.Close()
	var out struct {
		OK         bool                   `json:"ok"`
		Onboarding *storage.OnboardingFSM `json:"onboarding"`
		Total      int                    `json:"total"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode /api/onboarding: %v", err)
	}
	if !out.OK {
		t.Fatal("ok=false")
	}
	if out.Total != 5 {
		t.Fatalf("total = %d, want 5", out.Total)
	}
	return out.Onboarding
}

// TestOnboardingHTTP covers GET /api/onboarding and the POST actions: null
// before the start, restart → interview, confirm from interview → 400,
// skip → done, unknown action → 400.
func TestOnboardingHTTP(t *testing.T) {
	s := newTaskTestServer(t, "")
	tsrv := httptest.NewServer(s.Handler())
	defer tsrv.Close()

	// До старта анкеты FSM нет.
	if onb := onbGET(t, tsrv.URL); onb != nil {
		t.Fatalf("до старта onboarding = %+v, want null", onb)
	}

	if resp, body := postBody(t, tsrv.URL+"/api/onboarding", `{"action":"restart"}`); resp.StatusCode != 200 {
		t.Fatalf("restart: %d %s", resp.StatusCode, body)
	}
	if onb := onbGET(t, tsrv.URL); onb == nil || onb.Stage != onbStageInterview || onb.Step != 0 {
		t.Fatalf("после restart = %+v, want interview/0", onb)
	}

	// Подтверждение возможно только из confirm.
	if resp, body := postBody(t, tsrv.URL+"/api/onboarding", `{"action":"confirm"}`); resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("confirm из interview: %d %s", resp.StatusCode, body)
	}

	if resp, body := postBody(t, tsrv.URL+"/api/onboarding", `{"action":"skip"}`); resp.StatusCode != 200 {
		t.Fatalf("skip: %d %s", resp.StatusCode, body)
	}
	if onb := onbGET(t, tsrv.URL); onb == nil || onb.Stage != onbStageDone {
		t.Fatalf("после skip = %+v, want done", onb)
	}
	if resp, _ := postBody(t, tsrv.URL+"/api/onboarding", `{"action":"bogus"}`); resp.StatusCode != http.StatusBadRequest {
		t.Fatal("неизвестное действие прошло")
	}
}

// TestOnboardingResetScopes: «Удалить чат» (scope chat) сохраняет анкету,
// «рабочую» (working) — сбрасывает FSM анкеты вместе с задачей.
func TestOnboardingResetScopes(t *testing.T) {
	s := newTaskTestServer(t, "")
	tsrv := httptest.NewServer(s.Handler())
	defer tsrv.Close()
	if resp, body := postBody(t, tsrv.URL+"/api/onboarding", `{"action":"restart"}`); resp.StatusCode != 200 {
		t.Fatalf("restart: %d %s", resp.StatusCode, body)
	}

	if resp, body := postBody(t, tsrv.URL+"/api/reset", `{"scope":"chat"}`); resp.StatusCode != 200 {
		t.Fatalf("reset chat: %d %s", resp.StatusCode, body)
	}
	if onb := onbGET(t, tsrv.URL); onb == nil || onb.Stage != onbStageInterview {
		t.Fatalf("chat потерял анкету: %+v", onb)
	}

	if resp, body := postBody(t, tsrv.URL+"/api/reset", `{"scope":"working"}`); resp.StatusCode != 200 {
		t.Fatalf("reset working: %d %s", resp.StatusCode, body)
	}
	if onb := onbGET(t, tsrv.URL); onb != nil {
		t.Fatalf("working не почистил анкету: %+v", onb)
	}
}

// ── Persistence ─────────────────────────────────────────────────────────────

// TestOnboardingPersistsAcrossRestart: the FSM (interview step and the done
// marker of a skip-all questionnaire) and the collected answers survive a
// restart on the same database.
func TestOnboardingPersistsAcrossRestart(t *testing.T) {
	dbFile := t.TempDir() + "/chat.db"

	s1 := newTaskTestServer(t, dbFile)
	tsrv1 := httptest.NewServer(s1.Handler())
	if resp, body := postBody(t, tsrv1.URL+"/api/onboarding", `{"action":"restart"}`); resp.StatusCode != 200 {
		t.Fatalf("restart: %d %s", resp.StatusCode, body)
	}
	exec := s1.memoryToolExec(s1.activeSession)
	if _, err := exec("onboarding_answer", json.RawMessage(`{"answer":"Аня"}`)); err != nil {
		t.Fatalf("onboarding_answer: %v", err)
	}
	tsrv1.Close()
	if s1.db != nil {
		s1.db.Close() // освобождает файл для следующего сервера
	}

	// Свежий сервер на той же базе: interview/1 + профиль.
	s2 := newTaskTestServer(t, dbFile)
	tsrv2 := httptest.NewServer(s2.Handler())
	onb := onbGET(t, tsrv2.URL)
	if onb == nil || onb.Stage != onbStageInterview || onb.Step != 1 {
		t.Fatalf("восстановлено = %+v, want interview/1", onb)
	}
	s2.mu.Lock()
	it, ok := s2.onboardingAnswersLocked(s2.activeSession)["имя"]
	s2.mu.Unlock()
	if !ok || it.Text != "имя: Аня" {
		t.Fatalf("профиль не восстановлен: %+v", it)
	}

	// done-маркер тоже переживает рестарт — иначе skip-all перезапустил бы
	// анкету на следующем сообщении.
	s2.mu.Lock()
	s2.setOnboardingLocked(&storage.OnboardingFSM{Stage: onbStageDone})
	s2.mu.Unlock()
	s2.persist()
	tsrv2.Close()
	if s2.db != nil {
		s2.db.Close()
	}
	s3 := newTaskTestServer(t, dbFile)
	tsrv3 := httptest.NewServer(s3.Handler())
	defer tsrv3.Close()
	if onb := onbGET(t, tsrv3.URL); onb == nil || onb.Stage != onbStageDone {
		t.Fatalf("done-маркер потерян: %+v", onb)
	}
}
