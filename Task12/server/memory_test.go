package server

// Tests of the three-layer memory: long-term injection rules (profile always,
// confirmed facts by keyword match), the remember tool, promotion of
// compression facts into the long-term layer, reset semantics and the SQLite
// persistence (including the legacy JSON migration).

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"ai-chat-go/api"
	"ai-chat-go/config"
)

func newTestServerWithDB(t *testing.T, upstreamURL, dbFile string) *httptest.Server {
	t.Helper()
	client := api.New(upstreamURL, "test-key", 5*time.Second)
	cfg := config.Config{
		APIKey:      "test-key",
		BaseURL:     upstreamURL,
		Model:       "test-model",
		System:      "be brief",
		Timeout:     5 * time.Second,
		HistoryFile: filepath.Join(filepath.Dir(dbFile), "history.json"),
		DBFile:      dbFile,
	}
	srv := httptest.NewServer(New(client, cfg).Handler())
	t.Cleanup(srv.Close)
	return srv
}

func postBody(t *testing.T, url, body string) (*http.Response, string) {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	data, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	return resp, string(data)
}

func memoryItems(t *testing.T, url string) []struct {
	ID            string `json:"id"`
	Kind          string `json:"kind"`
	Key           string `json:"key"`
	Text          string `json:"text"`
	Confirmations int    `json:"confirmations"`
} {
	t.Helper()
	resp, err := http.Get(url + "/api/memory")
	if err != nil {
		t.Fatalf("GET /api/memory: %v", err)
	}
	defer resp.Body.Close()
	var out struct {
		Items []struct {
			ID            string `json:"id"`
			Kind          string `json:"kind"`
			Key           string `json:"key"`
			Text          string `json:"text"`
			Confirmations int    `json:"confirmations"`
		} `json:"items"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode memory: %v", err)
	}
	return out.Items
}

// TestMemoryProfileAlwaysInjected verifies rule 1 of the long-term layer: a
// profile entry reaches every request, regardless of the user's wording.
func TestMemoryProfileAlwaysInjected(t *testing.T) {
	var mu sync.Mutex
	var chatReqs []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), "Текущий блок фактов") { // compression calls are skipped
			mu.Lock()
			chatReqs = append(chatReqs, string(body))
			mu.Unlock()
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, part := range []string{"При", "вет"} {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", part)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	dbFile := filepath.Join(t.TempDir(), "chat.db")
	ts := newTestServerWithDB(t, upstream.URL, dbFile)

	if resp, _ := postBody(t, ts.URL+"/api/memory", `{"text":"имя: Аня","kind":"profile"}`); resp.StatusCode != http.StatusOK {
		t.Fatalf("POST /api/memory status = %d", resp.StatusCode)
	}

	if _, data := postBody(t, ts.URL+"/api/chat", `{"message":"совсем посторонний вопрос"}`); !strings.Contains(data, "При") {
		t.Fatalf("chat failed: %s", data)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(chatReqs) == 0 {
		t.Fatal("upstream saw no chat request")
	}
	last := chatReqs[len(chatReqs)-1]
	if !strings.Contains(last, "[профиль] имя: Аня") {
		t.Fatalf("profile entry not injected: %s", last)
	}
}

// TestMemoryKeywordGate verifies rule 2: unconfirmed knowledge is never
// injected; confirmed knowledge only when its key matches the user message.
func TestMemoryKeywordGate(t *testing.T) {
	var mu sync.Mutex
	var chatReqs []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), "Текущий блок фактов") {
			mu.Lock()
			chatReqs = append(chatReqs, string(body))
			mu.Unlock()
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ок\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	dbFile := filepath.Join(t.TempDir(), "chat.db")
	ts := newTestServerWithDB(t, upstream.URL, dbFile)

	// Explicit write → durable immediately (Confirmations=2).
	postBody(t, ts.URL+"/api/memory", `{"text":"курс: начат","kind":"knowledge"}`)

	_, data := postBody(t, ts.URL+"/api/chat", `{"message":"расскажи про погоду"}`)
	mu.Lock()
	req1 := chatReqs[len(chatReqs)-1]
	mu.Unlock()
	if strings.Contains(req1, "курс: начат") {
		t.Fatalf("unmatched knowledge must not be injected: %s", req1)
	}
	if strings.Contains(data, "курс") {
		t.Fatalf("unexpected answer mention: %s", data)
	}

	_, _ = postBody(t, ts.URL+"/api/chat", `{"message":"как продвигается курс?"}`)
	mu.Lock()
	req2 := chatReqs[len(chatReqs)-1]
	mu.Unlock()
	if !strings.Contains(req2, "[знание] курс: начат") {
		t.Fatalf("matched knowledge missing from the request: %s", req2)
	}
}

// TestMemoryToolRememberAndRecall verifies the remember tool round-trip: the
// model calls remember, the fact lands in the long-term layer and recall /
// GET /api/memory expose it.
func TestMemoryToolRememberAndRecall(t *testing.T) {
	var mu sync.Mutex
	reqCount := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		reqCount++
		cur := reqCount
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		if cur == 1 {
			fmt.Fprint(w, `data: {"choices":[{"delta":{"role":"assistant"}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"remember","arguments":""}}]}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"text\":\"город: Пермь\",\"kind\":\"knowledge\"}"}}]}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"записал\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	dbFile := filepath.Join(t.TempDir(), "chat.db")
	ts := newTestServerWithDB(t, upstream.URL, dbFile)

	postBody(t, ts.URL+"/api/chat", `{"message":"запомни, что я живу в Перми"}`)

	items := memoryItems(t, ts.URL)
	if len(items) != 1 {
		t.Fatalf("memory items = %d, want 1", len(items))
	}
	it := items[0]
	if it.Kind != "knowledge" || it.Key != "город" || it.Text != "город: Пермь" {
		t.Fatalf("remembered item = %+v", it)
	}
	if it.Confirmations != 2 {
		t.Fatalf("explicit write confirmations = %d, want 2", it.Confirmations)
	}
}

// TestFactsPromotedToLongTerm verifies the core v2 rule: sticky-facts
// compression feeds the long-term layer. A fact survives two compression
// passes → Confirmations=2 → it is injected when the user message matches.
func TestFactsPromotedToLongTerm(t *testing.T) {
	var mu sync.Mutex
	var seen []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		seen = append(seen, string(body))
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		if strings.Contains(string(body), "Текущий блок фактов") {
			// Compression: rewrite the facts block in the documented format.
			for _, part := range []string{"[профиль] имя: аня\n", "[знание] курс: го\n"} {
				fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", part)
			}
		} else {
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ок\"}}]}\n\n")
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	dbFile := filepath.Join(t.TempDir(), "chat.db")
	ts := newTestServerWithDB(t, upstream.URL, dbFile)

	const cs = `"compress":{"enabled":true,"strategy":"sticky","keep":4,"threshold":5}`
	seedTurns(t, ts, 10)
	postBody(t, ts.URL+"/api/chat", `{"message":"привет",`+cs+`}`)          // compression #1
	postBody(t, ts.URL+"/api/chat", `{"message":"вопрос",`+cs+`}`)          // compression #2
	postBody(t, ts.URL+"/api/chat", `{"message":"как продвигается курс?"}`) // injection check

	mu.Lock()
	defer mu.Unlock()
	if len(seen) < 5 {
		t.Fatalf("upstream saw %d requests, want >=5", len(seen))
	}
	// Compression request bodies carry the previous facts.
	compressReqs, chatReqs := 0, 0
	var lastChat string
	for _, req := range seen {
		if strings.Contains(req, "Текущий блок фактов") {
			compressReqs++
		} else if strings.Contains(req, "Сообщение") || strings.Contains(req, "question") || true {
			// chat request: distinguishable by the user content below
		}
		if strings.Contains(req, "как продвигается курс") {
			lastChat = req
			chatReqs++
		}
	}
	if compressReqs < 2 {
		t.Fatalf("compressions = %d, want 2", compressReqs)
	}
	if chatReqs == 0 || lastChat == "" {
		t.Fatal("no request carried the keyword question")
	}
	if !strings.Contains(lastChat, "[профиль] имя: аня") {
		t.Fatalf("profile missing from the request: %s", lastChat)
	}
	if !strings.Contains(lastChat, "[знание] курс: го") {
		t.Fatalf("confirmed fact missing from the request: %s", lastChat)
	}

	items := memoryItems(t, ts.URL)
	if len(items) != 2 {
		t.Fatalf("long-term items = %d, want 2", len(items))
	}
	for _, it := range items {
		if it.Confirmations < 2 {
			t.Fatalf("item %s confirmations = %d, want >=2 after two compressions", it.Key, it.Confirmations)
		}
	}
}

// TestMemorySurvivesReset verifies that /api/reset clears the dialog but
// keeps the long-term memory.
func TestMemorySurvivesReset(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	dbFile := filepath.Join(t.TempDir(), "chat.db")
	ts := newTestServerWithDB(t, upstream.URL, dbFile)

	postBody(t, ts.URL+"/api/memory", `{"text":"имя: Аня","kind":"profile"}`)
	if resp, _ := postBody(t, ts.URL+"/api/reset", `null`); resp.StatusCode != http.StatusOK {
		t.Fatal("reset failed")
	}
	items := memoryItems(t, ts.URL)
	if len(items) != 1 || items[0].Key != "имя" {
		t.Fatalf("memory after reset = %+v, want the profile item kept", items)
	}
}

// TestResetScopes verifies the two delete scopes of /api/reset: "chat"
// (default) clears the messages and the short-term layer but keeps the
// working memory (task, notes, facts) and the long-term items; "all" wipes
// every level including the long-term layer of all models.
func TestResetScopes(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		body, _ := io.ReadAll(r.Body)
		if strings.Contains(string(body), "Текущий блок фактов") {
			// Sticky-facts compression: a task line plus one fact.
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ЗАДАЧА: подобрать стеллажи\\n\"}}]}\n\n")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"[профиль] имя: аня\\n\"}}]}\n\n")
		} else {
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ок\"}}]}\n\n")
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	dbFile := filepath.Join(t.TempDir(), "chat.db")
	ts := newTestServerWithDB(t, upstream.URL, dbFile)

	const cs = `"compress":{"enabled":true,"strategy":"sticky","keep":4,"threshold":5}`
	seedTurns(t, ts, 10)
	postBody(t, ts.URL+"/api/chat", `{"message":"привет",`+cs+`}`)

	// memState reads both layers of /api/memory: working (task/facts) and
	// short (messages in context), plus the long-term item count.
	memState := func() (task, facts string, messages, items int) {
		t.Helper()
		get, err := http.Get(ts.URL + "/api/memory")
		if err != nil {
			t.Fatalf("GET /api/memory: %v", err)
		}
		defer get.Body.Close()
		var out struct {
			Working struct {
				Task  string `json:"task"`
				Facts string `json:"facts"`
			} `json:"working"`
			Short struct {
				Messages int `json:"messages"`
			} `json:"short"`
			Items []struct{} `json:"items"`
		}
		if err := json.NewDecoder(get.Body).Decode(&out); err != nil {
			t.Fatalf("decode /api/memory: %v", err)
		}
		return out.Working.Task, out.Working.Facts, out.Short.Messages, len(out.Items)
	}
	reset := func(scope string) {
		t.Helper()
		if resp, _ := postBody(t, ts.URL+"/api/reset", `{"scope":"`+scope+`"}`); resp.StatusCode != http.StatusOK {
			t.Fatalf("%s reset failed: %d", scope, resp.StatusCode)
		}
	}

	if task, facts, messages, items := memState(); task != "подобрать стеллажи" ||
		!strings.Contains(facts, "имя: аня") || messages == 0 || items != 1 {
		t.Fatalf("state after compression: task=%q facts=%q messages=%d items=%d", task, facts, messages, items)
	}

	// «Долговременная» (корзина блока): записи стираются, сообщения и
	// рабочая память остаются.
	reset("long")
	if _, _, messages, items := memState(); items != 0 || messages == 0 {
		t.Fatalf("after long reset: messages=%d items=%d, want messages kept, items 0", messages, items)
	}
	if task, facts, _, _ := memState(); task != "подобрать стеллажи" || !strings.Contains(facts, "имя: аня") {
		t.Fatalf("working after long reset: task=%q facts=%q, want them kept", task, facts)
	}

	// «Рабочая» (корзина блока): задача/заметки/факты стираются, сообщения
	// чата остаются на месте.
	reset("working")
	if task, facts, messages, items := memState(); task != "" || facts != "" || messages == 0 || items != 0 {
		t.Fatalf("after working reset: task=%q facts=%q messages=%d items=%d, want empty working, kept messages",
			task, facts, messages, items)
	}

	// «Удалить чат»: сообщения и краткосрочный слой стираются, рабочая
	// память (задача, заметки, факты) и долговременные записи остаются.
	// Наполняем рабочий слой заново (сжатие пишет факт и в долговременный).
	postBody(t, ts.URL+"/api/chat", `{"message":"привет ещё раз",`+cs+`}`)
	reset("chat")
	if task, facts, messages, items := memState(); task != "подобрать стеллажи" ||
		!strings.Contains(facts, "имя: аня") || messages != 0 || items != 1 {
		t.Fatalf("after chat reset: task=%q facts=%q messages=%d items=%d, want working kept, chat empty",
			task, facts, messages, items)
	}

	// «Удалить историю»: краткосрочная И рабочая память очищаются,
	// долговременные записи остаются.
	reset("history")
	if task, facts, messages, items := memState(); task != "" || facts != "" || messages != 0 || items != 1 {
		t.Fatalf("after history reset: task=%q facts=%q messages=%d items=%d, want empty short+working, kept items",
			task, facts, messages, items)
	}

	// «Удалить все»: чистятся и рабочая, и долговременная память.
	reset("all")
	if task, facts, messages, items := memState(); task != "" || facts != "" || messages != 0 || items != 0 {
		t.Fatalf("after full reset: task=%q facts=%q messages=%d items=%d, want everything empty",
			task, facts, messages, items)
	}
}

// TestChatResetKeepsWorkingWithoutSystem reproduces the reported bug: with an
// empty system prompt the active leaf after «Удалить чат» is "" (no system
// node is appended), and the working-memory state stored under that root key
// must still be found by /api/memory.
func TestChatResetKeepsWorkingWithoutSystem(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		body, _ := io.ReadAll(r.Body)
		if strings.Contains(string(body), "Текущий блок фактов") {
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ЗАДАЧА: подобрать стеллажи\\n\"}}]}\n\n")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"[профиль] имя: аня\\n\"}}]}\n\n")
		} else {
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ок\"}}]}\n\n")
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	dbFile := filepath.Join(t.TempDir(), "chat.db")
	client := api.New(upstream.URL, "test-key", 5*time.Second)
	cfg := config.Config{
		APIKey: "test-key", BaseURL: upstream.URL, Model: "test-model",
		System: "", Timeout: 5 * time.Second,
		HistoryFile: filepath.Join(filepath.Dir(dbFile), "history.json"), DBFile: dbFile,
	}
	ts := httptest.NewServer(New(client, cfg).Handler())
	t.Cleanup(ts.Close)

	const cs = `"compress":{"enabled":true,"strategy":"sticky","keep":4,"threshold":5}`
	seedTurns(t, ts, 10)
	postBody(t, ts.URL+"/api/chat", `{"message":"привет",`+cs+`}`)

	working := func() (task, facts string) {
		t.Helper()
		get, err := http.Get(ts.URL + "/api/memory")
		if err != nil {
			t.Fatalf("GET /api/memory: %v", err)
		}
		defer get.Body.Close()
		var out struct {
			Working struct {
				Task  string `json:"task"`
				Facts string `json:"facts"`
			} `json:"working"`
		}
		if err := json.NewDecoder(get.Body).Decode(&out); err != nil {
			t.Fatalf("decode /api/memory: %v", err)
		}
		return out.Working.Task, out.Working.Facts
	}

	if task, facts := working(); task != "подобрать стеллажи" || !strings.Contains(facts, "имя: аня") {
		t.Fatalf("working after compression: task=%q facts=%q", task, facts)
	}
	if resp, _ := postBody(t, ts.URL+"/api/reset", `{"scope":"chat"}`); resp.StatusCode != http.StatusOK {
		t.Fatal("chat reset failed")
	}
	if task, facts := working(); task != "подобрать стеллажи" || !strings.Contains(facts, "имя: аня") {
		t.Fatalf("working after chat reset without system prompt: task=%q facts=%q, want them kept", task, facts)
	}
}

// TestSQLitePersistenceRestores verifies the SQLite persistence: the dialog
// and the long-term memory survive a server restart on the same database.
func TestSQLitePersistenceRestores(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	dbFile := filepath.Join(t.TempDir(), "chat.db")
	ts := newTestServerWithDB(t, upstream.URL, dbFile)

	resp, data := postBody(t, ts.URL+"/api/chat", `{"message":"привет"}`)
	if resp.StatusCode != http.StatusOK || !strings.Contains(data, "Привет") {
		t.Fatalf("chat failed: %d %s", resp.StatusCode, data)
	}
	postBody(t, ts.URL+"/api/memory", `{"text":"имя: Аня","kind":"profile"}`)

	// "Restart": a fresh server on the same database file.
	ts2 := newTestServerWithDB(t, upstream.URL, dbFile)
	get, err := http.Get(ts2.URL + "/api/history")
	if err != nil {
		t.Fatalf("GET /api/history: %v", err)
	}
	var hist struct {
		Messages []historyMessage `json:"messages"`
	}
	if err := json.NewDecoder(get.Body).Decode(&hist); err != nil {
		t.Fatalf("decode history: %v", err)
	}
	get.Body.Close()
	if len(hist.Messages) != 2 || hist.Messages[0].Content != "привет" {
		t.Fatalf("restored history = %+v", hist.Messages)
	}
	items := memoryItems(t, ts2.URL)
	if len(items) != 1 || items[0].Key != "имя" {
		t.Fatalf("restored memory = %+v", items)
	}
}

// TestSQLiteLegacyMigration verifies that old per-model JSON contexts are
// imported into SQLite on start (and their files renamed to *.migrated).
func TestSQLiteLegacyMigration(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	dir := t.TempDir()
	base := filepath.Join(dir, "history.json")
	dbFile := filepath.Join(dir, "chat.db")

	legacy := historyState{
		Model:    "test-model",
		Active:   "m2",
		NextID:   2,
		Messages: []chatRecord{{ID: "m2", Role: "user", Content: "старый диалог", At: time.Now()}},
	}
	perModel := modelHistoryPath(base, "test-model")
	if err := saveState(perModel, legacy); err != nil {
		t.Fatalf("saveState: %v", err)
	}

	client := api.New(upstream.URL, "test-key", 5*time.Second)
	cfg := config.Config{
		APIKey: "test-key", BaseURL: upstream.URL, Model: "test-model",
		System: "be brief", Timeout: 5 * time.Second,
		HistoryFile: base, DBFile: dbFile,
	}
	ts := httptest.NewServer(New(client, cfg).Handler())
	defer ts.Close()

	get, err := http.Get(ts.URL + "/api/history")
	if err != nil {
		t.Fatalf("GET /api/history: %v", err)
	}
	var hist struct {
		Messages []historyMessage `json:"messages"`
	}
	if err := json.NewDecoder(get.Body).Decode(&hist); err != nil {
		t.Fatalf("decode history: %v", err)
	}
	get.Body.Close()
	if len(hist.Messages) != 1 || hist.Messages[0].Content != "старый диалог" {
		t.Fatalf("migrated history = %+v", hist.Messages)
	}
	if _, err := os.Stat(perModel + ".migrated"); err != nil {
		t.Fatalf("legacy file not renamed: %v", err)
	}
}

// TestSQLiteLegacyMigrationFlat verifies that old flat JSON contexts without
// message ids (the pre-branching format) migrate into SQLite: the server
// assigns ids linearly so the messages(model, id) unique index holds.
func TestSQLiteLegacyMigrationFlat(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	dir := t.TempDir()
	base := filepath.Join(dir, "history.json")
	dbFile := filepath.Join(dir, "chat.db")

	legacy := historyState{
		Model:    "test-model",
		Active:   "",
		NextID:   0,
		Messages: []chatRecord{{Role: "user", Content: "раз"}, {Role: "assistant", Content: "два"}, {Role: "user", Content: "три"}},
	}
	perModel := modelHistoryPath(base, "test-model")
	if err := saveState(perModel, legacy); err != nil {
		t.Fatalf("saveState: %v", err)
	}

	client := api.New(upstream.URL, "test-key", 5*time.Second)
	cfg := config.Config{
		APIKey: "test-key", BaseURL: upstream.URL, Model: "test-model",
		System: "be brief", Timeout: 5 * time.Second,
		HistoryFile: base, DBFile: dbFile,
	}
	ts := httptest.NewServer(New(client, cfg).Handler())
	defer ts.Close()

	get, err := http.Get(ts.URL + "/api/history")
	if err != nil {
		t.Fatalf("GET /api/history: %v", err)
	}
	var hist struct {
		Messages []historyMessage `json:"messages"`
	}
	if err := json.NewDecoder(get.Body).Decode(&hist); err != nil {
		t.Fatalf("decode history: %v", err)
	}
	get.Body.Close()
	if len(hist.Messages) != 3 || hist.Messages[2].Content != "три" {
		t.Fatalf("migrated flat history = %+v", hist.Messages)
	}
	if hist.Messages[0].ID == "" || hist.Messages[0].ID == hist.Messages[1].ID {
		t.Fatalf("migrated messages have no unique ids: %+v", hist.Messages)
	}
	if _, err := os.Stat(perModel + ".migrated"); err != nil {
		t.Fatalf("legacy file not renamed: %v", err)
	}
}

// TestMemoryLayersEndpoint verifies that GET /api/memory reports all three
// layers: the long-term items plus the current contents of the short-term
// (displayed turns and tokens) and working (task, notes, facts) memory.
func TestMemoryLayersEndpoint(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	dbFile := filepath.Join(t.TempDir(), "chat.db")
	ts := newTestServerWithDB(t, upstream.URL, dbFile)

	postBody(t, ts.URL+"/api/chat", `{"message":"привет"}`)
	postBody(t, ts.URL+"/api/memory", `{"text":"имя: Аня","kind":"profile"}`)

	get, err := http.Get(ts.URL + "/api/memory")
	if err != nil {
		t.Fatalf("GET /api/memory: %v", err)
	}
	defer get.Body.Close()
	var out struct {
		Items []struct {
			ID   string `json:"id"`
			Kind string `json:"kind"`
		} `json:"items"`
		Short struct {
			Messages int `json:"messages"`
			Tokens   int `json:"tokens"`
		} `json:"short"`
		Working struct {
			Task  string `json:"task"`
			Notes string `json:"notes"`
			Facts string `json:"facts"`
		} `json:"working"`
		Long struct {
			Tokens int `json:"tokens"`
		} `json:"long"`
		Tokens struct {
			Short   int `json:"short"`
			Working int `json:"working"`
			Long    int `json:"long"`
		} `json:"tokens"`
	}
	if err := json.NewDecoder(get.Body).Decode(&out); err != nil {
		t.Fatalf("decode /api/memory: %v", err)
	}
	if len(out.Items) != 1 || out.Items[0].Kind != "profile" {
		t.Fatalf("items = %+v, want one profile entry", out.Items)
	}
	if out.Short.Messages < 2 {
		t.Fatalf("short.messages = %d, want >=2 after an exchange", out.Short.Messages)
	}
	if out.Working.Task != "" || out.Working.Notes != "" {
		t.Fatalf("working = %+v, want an empty working memory", out.Working)
	}
	// Токены: краткосрочный слой — фактический размер контекста (может быть
	// 0 без usage от апстрима), рабочий пуст → 0, долговременный — оценка
	// единственной записи.
	if out.Tokens.Working != 0 {
		t.Fatalf("tokens = %+v, want working=0 for an empty layer", out.Tokens)
	}
	if out.Long.Tokens <= 0 || out.Tokens.Long != out.Long.Tokens {
		t.Fatalf("long.tokens=%d tokens.long=%d, want a positive matching estimate",
			out.Long.Tokens, out.Tokens.Long)
	}
}

// TestMemoryDeleteEndpoint verifies DELETE /api/memory.
func TestMemoryDeleteEndpoint(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	dbFile := filepath.Join(t.TempDir(), "chat.db")
	ts := newTestServerWithDB(t, upstream.URL, dbFile)

	_, data := postBody(t, ts.URL+"/api/memory", `{"text":"имя: Аня","kind":"profile"}`)
	var added struct {
		Item struct {
			ID string `json:"id"`
		} `json:"item"`
	}
	if err := json.Unmarshal([]byte(data), &added); err != nil || added.Item.ID == "" {
		t.Fatalf("add response = %s (%v)", data, err)
	}
	req, err := http.NewRequest(http.MethodDelete, ts.URL+"/api/memory?id="+added.Item.ID, nil)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	if resp, err := http.DefaultClient.Do(req); err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("DELETE /api/memory: %v %v", err, resp)
	}
	if items := memoryItems(t, ts.URL); len(items) != 0 {
		t.Fatalf("items after delete = %+v", items)
	}
}
