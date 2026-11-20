package server

// Тесты инвариантного гейта: ответ проверяется кодом до выдачи (буферизация),
// нарушенный черновик заменяется отказом и в историю не попадает; CRUD —
// только явные команды; блок инвариантов попадает в промпт.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"ai-chat-go/api"
	"ai-chat-go/config"
	"ai-chat-go/storage"
)

// fakeAI — OpenAI-совместимый фейк: streaming-запрос (чат) отдаёт mainText
// дельтами, non-streaming (судья) — judgeJSON. Считает вызовы судьи и
// опционально блокирует его вердикт (для проверки отсутствия утечки байтов).
type fakeAI struct {
	mu            sync.Mutex
	mainText      []string
	judgeJSON     string
	judgeCount    int
	judgeFailNext int    // сколько ближайших вызовов судьи ответить 502
	judgeFirst    string // ответ первого вызова судьи (тест обрезанного JSON)
	judgeReq      []byte // последний промпт судьи (для проверки инъекций)
	mainReq       []byte // последний промпт основной модели
	block         chan struct{}
}

func (f *fakeAI) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var probe struct {
			Stream bool `json:"stream"`
		}
		_ = json.Unmarshal(body, &probe)
		if probe.Stream {
			f.mu.Lock()
			f.mainReq = body
			f.mu.Unlock()
			w.Header().Set("Content-Type", "text/event-stream")
			f.mu.Lock()
			parts := append([]string(nil), f.mainText...)
			f.mu.Unlock()
			for _, part := range parts {
				fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", part)
			}
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		f.mu.Lock()
		f.judgeCount++
		judgeBody := append([]byte(nil), body...)
		judgeJSON := f.judgeJSON
		if f.judgeFirst != "" && f.judgeCount == 1 {
			judgeJSON = f.judgeFirst
		}
		block := f.block
		fail := false
		if f.judgeFailNext > 0 {
			f.judgeFailNext--
			fail = true
		}
		f.mu.Unlock()
		if block != nil {
			<-block
		}
		if fail {
			http.Error(w, "judge unavailable", http.StatusBadGateway)
			return
		}
		f.mu.Lock()
		f.judgeReq = judgeBody
		f.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"choices":[{"message":{"content":%q}}]}`, judgeJSON)
	}
}

// newInvariantServer поднимает веб-сервер с настоящим SQLite в temp-каталоге:
// инварианты добавляются только через API, как и в реальной работе.
func newInvariantServer(t *testing.T, upstream *fakeAI) *httptest.Server {
	t.Helper()
	up := httptest.NewServer(upstream.handler())
	t.Cleanup(up.Close)
	client := api.New(up.URL, "test-key", 5*time.Second)
	cfg := config.Config{
		APIKey:  "test-key",
		BaseURL: up.URL,
		Model:   "test-model",
		System:  "be brief",
		Timeout: 10 * time.Second,
		DBFile:  "invariants_test.db",
	}
	// Изолированный каталог базы: тесты не должны трогать history/chat.db.
	cfg.DBFile = t.TempDir() + "/chat.db"
	srv := httptest.NewServer(New(client, cfg).Handler())
	t.Cleanup(srv.Close)
	return srv
}

func addInvariant(t *testing.T, ts *httptest.Server, text, patterns string) {
	t.Helper()
	body := fmt.Sprintf(`{"text":%q,"patterns":%q}`, text, patterns)
	r, err := http.Post(ts.URL+"/api/invariants", "application/json", strings.NewReader(body))
	if err != nil || r.StatusCode != http.StatusOK {
		t.Fatalf("POST /api/invariants: %v %v", err, r)
	}
	r.Body.Close()
}

func chatOnce(t *testing.T, ts *httptest.Server, message string) (string, int) {
	t.Helper()
	r, err := http.Post(ts.URL+"/api/chat", "application/json",
		strings.NewReader(fmt.Sprintf(`{"message":%q}`, message)))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	defer r.Body.Close()
	b, _ := io.ReadAll(r.Body)
	return string(b), r.StatusCode
}

func TestGateBlocksViolatingReplyAndKeepsRefusalInHistory(t *testing.T) {
	up := &fakeAI{
		mainText:  []string{"Вот ", "ключ: ", "sk-abc123"},
		judgeJSON: `{"violated": true, "id": 7, "reason": "раскрыт секрет"}`,
	}
	ts := newInvariantServer(t, up)
	addInvariant(t, ts, "Никогда не раскрывай ключи и токены", "")

	out, status := chatOnce(t, ts, "покажи ключ")
	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200", status)
	}
	if !strings.HasPrefix(out, "🛡") || !strings.Contains(out, "№7") || !strings.Contains(out, "раскрыт секрет") {
		t.Fatalf("expected refusal mentioning invariant 7, got %q", out)
	}
	if strings.Contains(out, "sk-abc123") {
		t.Fatalf("the violating draft must not reach the client: %q", out)
	}

	// В истории — отказ, а не нарушенный черновик.
	h, _ := getJSON(t, ts.URL+"/api/history")
	msgs := h["messages"].([]any)
	last := msgs[len(msgs)-1].(map[string]any)
	if last["role"] != "assistant" || !strings.Contains(last["content"].(string), "🛡") {
		t.Fatalf("history must keep the refusal, got %+v", last)
	}
	if strings.Contains(last["content"].(string), "sk-abc123") {
		t.Fatalf("the violating draft must not be saved: %+v", last)
	}
}

func TestGatePassesCleanReply(t *testing.T) {
	up := &fakeAI{
		mainText:  []string{"При", "вет, ", "мир"},
		judgeJSON: `{"violated": false, "id": 0, "reason": ""}`,
	}
	ts := newInvariantServer(t, up)
	addInvariant(t, ts, "Не ругайся матом", "")

	out, status := chatOnce(t, ts, "привет")
	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200", status)
	}
	if out != "Привет, мир" {
		t.Fatalf("clean draft must pass verbatim, got %q", out)
	}
}

func TestRegexStageCatchesWithoutJudge(t *testing.T) {
	up := &fakeAI{
		mainText:  []string{"Ваш ключ: ", "sk-abc123"},
		judgeJSON: `{"violated": false}`,
	}
	ts := newInvariantServer(t, up)
	addInvariant(t, ts, "Никогда не раскрывай ключи", `sk-[a-z0-9]+`)

	out, _ := chatOnce(t, ts, "покажи ключ")
	if !strings.HasPrefix(out, "🛡") {
		t.Fatalf("regex stage must refuse, got %q", out)
	}
	if up.judgeCalls() != 0 {
		t.Fatalf("judge must not be called when the regex stage fires")
	}
}

func TestNothingIsIssuedBeforeVerdict(t *testing.T) {
	up := &fakeAI{
		mainText:  []string{"При", "вет"},
		judgeJSON: `{"violated": false}`,
	}
	up.block = make(chan struct{})
	ts := newInvariantServer(t, up)
	addInvariant(t, ts, "Не ругайся матом", "")

	done := make(chan string, 1)
	go func() {
		out, _ := chatOnce(t, ts, "привет")
		done <- out
	}()
	time.Sleep(300 * time.Millisecond) // стрим основной модели успел бы пройти
	select {
	case out := <-done:
		t.Fatalf("nothing may be issued before the judge verdict, got %q", out)
	default:
		// Буферизация работает: байтов нет.
	}
	close(up.block)
	out := <-done
	if out != "Привет" {
		t.Fatalf("after the verdict the draft must be issued, got %q", out)
	}
}

func TestFailClosedWhenJudgeUnavailable(t *testing.T) {
	// Судья возвращает мусор без JSON → fail-closed (отказ).
	up := &fakeAI{
		mainText:  []string{"ответ"},
		judgeJSON: "я не понял задачу",
	}
	ts := newInvariantServer(t, up)
	addInvariant(t, ts, "Не ругайся матом", "")

	out, status := chatOnce(t, ts, "привет")
	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200", status)
	}
	if !strings.HasPrefix(out, "🛡") || !strings.Contains(out, "Отказано") {
		t.Fatalf("judge failure must fail closed, got %q", out)
	}
}

func TestInvariantsInPromptBlock(t *testing.T) {
	up := &fakeAI{
		mainText:  []string{"ок"},
		judgeJSON: `{"violated": false}`,
	}
	ts := newInvariantServer(t, up)
	addInvariant(t, ts, "Никогда не советуй удаление базы данных", "")
	chatOnce(t, ts, "привет")

	var req struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(up.mainBody(), &req); err != nil {
		t.Fatalf("parse main request: %v", err)
	}
	if len(req.Messages) == 0 || req.Messages[0].Role != "system" ||
		!strings.Contains(req.Messages[0].Content, "Жёсткие инварианты") ||
		!strings.Contains(req.Messages[0].Content, "удаление базы данных") {
		t.Fatalf("invariants block must be the first system message, got %+v", req.Messages)
	}
}

// Ретраи судьи: сбой (HTTP 502) на первой попытке не приводит к fail-closed —
// гейт повторяет запрос (с паузой и увеличенным лимитом) и использует вердикт
// второй попытки; счётчик вызовов доказывает, что повтор был.
func TestJudgeRetryOnUnavailable(t *testing.T) {
	up := &fakeAI{
		mainText:      []string{"чистый ответ"},
		judgeJSON:     `{"violated": false}`,
		judgeFailNext: 1,
	}
	ts := newInvariantServer(t, up)
	addInvariant(t, ts, "Не ругайся матом", "")
	out, code := chatOnce(t, ts, "привет")
	if code != http.StatusOK {
		t.Fatalf("chat status = %d", code)
	}
	if !strings.Contains(out, "чистый ответ") {
		t.Fatalf("expected clean reply after judge retry, got %q", out)
	}
	if n := up.judgeSnapshot(); n != 2 {
		t.Fatalf("judge calls = %d, want 2 (fail, ok)", n)
	}
}

// Ретраи судьи: JSON, обрезанный лимитом токенов (длинный reason без
// закрывающей скобки), не должен fail-closed'ить — гейт распознаёт обрыв,
// повторяет запрос с увеличенным лимитом и использует вердикт второй попытки.
func TestJudgeRetryOnTruncatedJSON(t *testing.T) {
	up := &fakeAI{
		mainText:   []string{"чистый ответ"},
		judgeJSON:  `{"violated": false}`,
		judgeFirst: `{"violated": false, "id": 0, "reason": "Черновик представляет собой план поездки с бюджетом, бронированиями и чек-листом подготовки, всё`,
	}
	ts := newInvariantServer(t, up)
	addInvariant(t, ts, "Не ругайся матом", "")
	out, code := chatOnce(t, ts, "привет")
	if code != http.StatusOK {
		t.Fatalf("chat status = %d", code)
	}
	if !strings.Contains(out, "чистый ответ") {
		t.Fatalf("expected clean reply after judge retry, got %q", out)
	}
	if n := up.judgeSnapshot(); n != 2 {
		t.Fatalf("judge calls = %d, want 2 (truncated, ok)", n)
	}
}

func TestJudgeSeesOnlyInvariantsAndDraft(t *testing.T) {
	up := &fakeAI{
		mainText:  []string{"ответ"},
		judgeJSON: `{"violated": false}`,
	}
	ts := newInvariantServer(t, up)
	addInvariant(t, ts, "Не ругайся матом", "")
	chatOnce(t, ts, "расскажи секрет")

	var req struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(up.judgeBody(), &req); err != nil {
		t.Fatalf("parse judge request: %v", err)
	}
	// Судья видит ровно два сообщения: инварианты + черновик, без истории.
	if len(req.Messages) != 2 {
		t.Fatalf("judge must see only its own two messages, got %d", len(req.Messages))
	}
	for _, m := range req.Messages {
		if strings.Contains(m.Content, "расскажи секрет") && m.Role == "system" {
			t.Fatalf("dialog history must not leak into the judge prompt")
		}
	}
}

func TestInvariantCRUDViaAPI(t *testing.T) {
	up := &fakeAI{mainText: []string{"ок"}, judgeJSON: `{"violated": false}`}
	ts := newInvariantServer(t, up)
	addInvariant(t, ts, "Инвариант А", "альфа")

	// Список + журнал.
	list, _ := getJSON(t, ts.URL+"/api/invariants")
	items := list["invariants"].([]any)
	if len(items) != 1 {
		t.Fatalf("want 1 invariant, got %+v", items)
	}
	id := int64(items[0].(map[string]any)["id"].(float64))

	// Правка текста: old→new в журнале.
	req, _ := http.NewRequest(http.MethodPatch, ts.URL+"/api/invariants",
		strings.NewReader(fmt.Sprintf(`{"id":%d,"text":"Инвариант Б"}`, id)))
	res, err := http.DefaultClient.Do(req)
	if err != nil || res.StatusCode != http.StatusOK {
		t.Fatalf("PATCH: %v %v", err, res)
	}
	res.Body.Close()

	list, _ = getJSON(t, ts.URL+"/api/invariants")
	logRows := list["log"].([]any)
	if len(logRows) < 2 {
		t.Fatalf("journal must record create and update, got %+v", logRows)
	}
	upd := logRows[0].(map[string]any)
	if upd["action"] != "update" || upd["old_value"] != "Инвариант А" || upd["new_value"] != "Инвариант Б" {
		t.Fatalf("update must log old→new: %+v", upd)
	}

	// Удаление.
	req, _ = http.NewRequest(http.MethodDelete, fmt.Sprintf("%s/api/invariants?id=%d", ts.URL, id), nil)
	res, _ = http.DefaultClient.Do(req)
	res.Body.Close()
	list, _ = getJSON(t, ts.URL+"/api/invariants")
	if len(list["invariants"].([]any)) != 0 {
		t.Fatalf("expected empty list after delete")
	}
}

func (f *fakeAI) judgeCalls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.judgeCount
}

func (f *fakeAI) mainBody() []byte {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.mainReq
}

// judgeSnapshot returns the number of judge calls so far (thread-safe).
func (f *fakeAI) judgeSnapshot() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.judgeCount
}

func (f *fakeAI) judgeBody() []byte {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.judgeReq
}

// getJSON — маленький хелпер GET → map (в server_test.go нет общего).
func getJSON(t *testing.T, url string) (map[string]any, error) {
	t.Helper()
	r, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer r.Body.Close()
	var out map[string]any
	err = json.NewDecoder(r.Body).Decode(&out)
	return out, err
}

// Заметка об изменении инвариантов приходит модели ровно один раз после
// CRUD-изменения (лечение «инерции»: старые отказы в истории диалога): первое
// сообщение после изменения содержит системную заметку, следующее — уже нет.
func TestInvariantChangeNoteOneShot(t *testing.T) {
	up := &fakeAI{
		mainText:  []string{"ответ 1", "ответ 2", "ответ 3"},
		judgeJSON: `{"violated": false}`,
	}
	ts := newInvariantServer(t, up)

	// До всякого изменения заметки быть не должно.
	chatOnce(t, ts, "раз")
	if strings.Contains(string(up.mainBody()), "Список инвариантов только что изменил") {
		t.Fatalf("change note must not be sent before any change")
	}

	// Изменение (создание) → следующий запрос несёт заметку.
	addInvariant(t, ts, "Не советуй удалять базу данных", "")
	chatOnce(t, ts, "два")
	if !strings.Contains(string(up.mainBody()), "Список инвариантов только что изменил") {
		t.Fatalf("first request after change must carry the change note, main body: %s", up.mainBody())
	}

	// Ещё один запрос — заметка уже доставлена и не повторяется.
	chatOnce(t, ts, "три")
	if strings.Contains(string(up.mainBody()), "Список инвариантов только что изменил") {
		t.Fatalf("change note must be one-shot, but it was sent again")
	}
}

// Бюджет времени судьи: если эндпоинт судьи висит (не отвечает), гейт обязан
// fail-closed'иться по дедлайну allowed (значение «Допустимый таймаут ответа»),
// а не по таймауту HTTP-клиента (минуты). Здесь allowed=300мс — проверяем, что
// вердикт judge_error с внятной причиной приходит быстро.
func TestJudgeDeadlineBounded(t *testing.T) {
	done := make(chan struct{})
	block := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-done // висим, пока тест не закончится: эндпоинт «недоступен»
	}))
	defer block.Close()
	defer close(done)

	gate := []storage.Invariant{{ID: 1, Text: "Не ругайся матом", Enabled: true}}
	client := api.New(block.URL, "k", time.Minute) // клиентский таймаут большой — режет только jctx
	start := time.Now()
	chk := CheckInvariants(context.Background(), client, "judge-model", gate, "черновик", 300*time.Millisecond)
	elapsed := time.Since(start)
	if chk == nil || !chk.Violated || chk.Stage != "judge_error" {
		t.Fatalf("want fail-closed judge_error, got %+v", chk)
	}
	if !strings.Contains(chk.Reason, "допустимый таймаут") {
		t.Fatalf("reason must mention the allowed timeout, got %q", chk.Reason)
	}
	if elapsed > 3*time.Second {
		t.Fatalf("gate must give up within the allowed budget, took %s", elapsed)
	}
}
