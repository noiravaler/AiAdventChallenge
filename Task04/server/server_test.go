package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"ai-chat-go/api"
	"ai-chat-go/config"
)

func newTestServer(t *testing.T, upstreamURL string) *httptest.Server {
	t.Helper()
	client := api.New(upstreamURL, "test-key", 5*time.Second)
	cfg := config.Config{
		APIKey:  "test-key",
		BaseURL: upstreamURL,
		Model:   "test-model",
		System:  "be brief",
		Timeout: 5 * time.Second,
	}
	srv := httptest.NewServer(New(client, cfg).Handler())
	t.Cleanup(srv.Close)
	return srv
}

// sseUpstream returns an OpenAI-compatible endpoint that streams three deltas.
func sseUpstream() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, part := range []string{"При", "вет", ", мир"} {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", part)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
}

func TestChatStreamsAndKeepsHistory(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"привет"}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if got := string(body); got != "Привет, мир" {
		t.Fatalf("streamed body = %q, want %q", got, "Привет, мир")
	}

	var hist struct {
		Messages []historyMessage `json:"messages"`
	}
	get, err := http.Get(ts.URL + "/api/history")
	if err != nil {
		t.Fatalf("GET /api/history: %v", err)
	}
	defer get.Body.Close()
	if err := json.NewDecoder(get.Body).Decode(&hist); err != nil {
		t.Fatalf("decode history: %v", err)
	}
	if len(hist.Messages) != 2 {
		t.Fatalf("history len = %d, want 2", len(hist.Messages))
	}
	if hist.Messages[0].Role != "user" || hist.Messages[0].Content != "привет" {
		t.Fatalf("first message = %+v", hist.Messages[0])
	}
	if hist.Messages[1].Role != "assistant" || hist.Messages[1].Content != "Привет, мир" {
		t.Fatalf("second message = %+v", hist.Messages[1])
	}
}

func TestResetClearsHistory(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	if _, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"привет"}`)); err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}

	res, err := http.Post(ts.URL+"/api/reset", "application/json", nil)
	if err != nil {
		t.Fatalf("POST /api/reset: %v", err)
	}
	res.Body.Close()

	get, err := http.Get(ts.URL + "/api/history")
	if err != nil {
		t.Fatalf("GET /api/history: %v", err)
	}
	defer get.Body.Close()
	var hist struct {
		Messages []historyMessage `json:"messages"`
	}
	if err := json.NewDecoder(get.Body).Decode(&hist); err != nil {
		t.Fatalf("decode history: %v", err)
	}
	if len(hist.Messages) != 0 {
		t.Fatalf("history len = %d, want 0 after reset", len(hist.Messages))
	}
}

func TestAPIErrorBeforeStream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"bad key"}}`, http.StatusUnauthorized)
	}))
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"привет"}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "bad key") {
		t.Fatalf("body = %q, want it to contain the upstream error", body)
	}

	get, err := http.Get(ts.URL + "/api/history")
	if err != nil {
		t.Fatalf("GET /api/history: %v", err)
	}
	defer get.Body.Close()
	var hist struct {
		Messages []historyMessage `json:"messages"`
	}
	if err := json.NewDecoder(get.Body).Decode(&hist); err != nil {
		t.Fatalf("decode history: %v", err)
	}
	if len(hist.Messages) != 0 {
		t.Fatalf("failed turn must be dropped, history = %+v", hist.Messages)
	}
}

func TestEmptyMessageRejected(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"   "}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
}

func TestIndexServed(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	get, err := http.Get(ts.URL + "/")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer get.Body.Close()
	if get.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", get.StatusCode)
	}
	body, _ := io.ReadAll(get.Body)
	if !strings.Contains(string(body), "Задача 1") {
		t.Fatalf("index page does not contain task tab 1")
	}
	if !strings.Contains(string(body), "Задача 2") {
		t.Fatalf("index page does not contain task tab 2")
	}
	if !strings.Contains(string(body), "Выполнить") {
		t.Fatalf("index page does not contain the run button")
	}
	if !strings.Contains(string(body), "Температура 0") {
		t.Fatalf("index page does not contain the temperature row")
	}
	if !strings.Contains(string(body), "Ответ:") {
		t.Fatalf("index page does not contain the expected answer field")
	}
	bodyStr := string(body)
	// Отбрасываем содержимое <script>, чтобы проверка тегов не задевала JS-код
	// рендера markdown-таблиц (renderTable легитимно эмитит <thead>/<th>).
	if i := strings.Index(bodyStr, "<script>"); i >= 0 {
		if j := strings.Index(bodyStr[i:], "</script>"); j >= 0 {
			bodyStr = bodyStr[:i] + bodyStr[i+j+len("</script>"):]
		}
	}
	if strings.Contains(bodyStr, "<thead>") {
		t.Fatalf("index page still contains a table header")
	}
	if strings.Contains(bodyStr, "Рассуждение: low") {
		t.Fatalf("index page still contains the removed reasoning tabs")
	}
	if strings.Contains(bodyStr, "Настройки") {
		t.Fatalf("index page still contains the removed sidebar")
	}
}

func TestModelsListed(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	client := api.New(upstream.URL, "test-key", 5*time.Second)
	cfg := config.Config{
		APIKey:  "test-key",
		BaseURL: upstream.URL,
		Model:   "glm-5.3-flash",
		Timeout: 5 * time.Second,
		Models:  []string{"glm-5.3-flash", "deepseek-v4-flash"},
	}
	ts := httptest.NewServer(New(client, cfg).Handler())
	defer ts.Close()

	get, err := http.Get(ts.URL + "/api/models")
	if err != nil {
		t.Fatalf("GET /api/models: %v", err)
	}
	defer get.Body.Close()
	var data struct {
		Models  []string `json:"models"`
		Current string   `json:"current"`
	}
	if err := json.NewDecoder(get.Body).Decode(&data); err != nil {
		t.Fatalf("decode /api/models: %v", err)
	}
	if len(data.Models) != 2 {
		t.Fatalf("models = %v, want 2", data.Models)
	}
	if data.Current != "glm-5.3-flash" {
		t.Fatalf("current = %q, want glm-5.3-flash", data.Current)
	}
}

func TestModelSwitchResetsHistory(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	client := api.New(upstream.URL, "test-key", 5*time.Second)
	cfg := config.Config{
		APIKey:  "test-key",
		BaseURL: upstream.URL,
		Model:   "glm-5.3-flash",
		System:  "be brief",
		Timeout: 5 * time.Second,
		Models:  []string{"glm-5.3-flash", "deepseek-v4-flash"},
	}
	ts := httptest.NewServer(New(client, cfg).Handler())
	defer ts.Close()

	if _, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"привет"}`)); err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}

	res, err := http.Post(ts.URL+"/api/model", "application/json", strings.NewReader(`{"model":"deepseek-v4-flash"}`))
	if err != nil {
		t.Fatalf("POST /api/model: %v", err)
	}
	res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("POST /api/model status = %d, want 200", res.StatusCode)
	}

	var info struct {
		Model string `json:"model"`
	}
	getInfo, err := http.Get(ts.URL + "/api/info")
	if err != nil {
		t.Fatalf("GET /api/info: %v", err)
	}
	defer getInfo.Body.Close()
	if err := json.NewDecoder(getInfo.Body).Decode(&info); err != nil {
		t.Fatalf("decode /api/info: %v", err)
	}
	if info.Model != "deepseek-v4-flash" {
		t.Fatalf("info.model = %q, want deepseek-v4-flash", info.Model)
	}

	getHist, err := http.Get(ts.URL + "/api/history")
	if err != nil {
		t.Fatalf("GET /api/history: %v", err)
	}
	defer getHist.Body.Close()
	var hist struct {
		Messages []historyMessage `json:"messages"`
	}
	if err := json.NewDecoder(getHist.Body).Decode(&hist); err != nil {
		t.Fatalf("decode history: %v", err)
	}
	if len(hist.Messages) != 0 {
		t.Fatalf("history after model switch = %d, want 0 (new session)", len(hist.Messages))
	}
}

func TestNoResponseAborted(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		time.Sleep(500 * time.Millisecond)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"X\"}}]}\n\n")
	}))
	defer upstream.Close()

	client := api.New(upstream.URL, "test-key", 5*time.Second)
	cfg := config.Config{
		APIKey:  "test-key",
		BaseURL: upstream.URL,
		Model:   "m",
		Timeout: 5 * time.Second,
		Models:  []string{"m"},
	}
	s := New(client, cfg)
	s.noReplyTimeout = 100 * time.Millisecond
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"привет"}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusGatewayTimeout {
		t.Fatalf("status = %d, want %d (504)", resp.StatusCode, http.StatusGatewayTimeout)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "не ответила") {
		t.Fatalf("body = %q, want it to mention a no-response timeout", body)
	}
}

// TestHandleModelsDiscoversAndFilters checks that /api/models pulls the list
// from the upstream /models endpoint, drops non-chat models and keeps the
// current selection valid.
func TestHandleModelsDiscoversAndFilters(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"object":"list","data":[
				{"id":"default-chat"},{"id":"default-coding"},
				{"id":"whisper-large-v3-turbo"},{"id":"qwen3.8-27b"},
				{"id":"qwen3-vl-embedding-8b"},{"id":"gpt-4o"},
				{"id":"qwen3-reranker-4b"}]}`)
		case "/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"OK\"}}]}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		default:
			http.NotFound(w, r)
		}
	}))
	defer upstream.Close()

	client := api.New(upstream.URL, "test-key", 5*time.Second)
	cfg := config.Config{
		APIKey:  "test-key",
		BaseURL: upstream.URL,
		Model:   "default-chat",
		Timeout: 5 * time.Second,
		Models:  []string{"fallback-old"}, // discovery must replace this
	}
	ts := httptest.NewServer(New(client, cfg).Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/api/models")
	if err != nil {
		t.Fatalf("GET /api/models: %v", err)
	}
	defer resp.Body.Close()

	var data struct {
		Models  []string `json:"models"`
		Current string   `json:"current"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Fatalf("decode /api/models: %v", err)
	}

	want := []string{"default-chat", "default-coding", "qwen3.8-27b", "gpt-4o"}
	if !reflect.DeepEqual(data.Models, want) {
		t.Fatalf("models = %v, want %v (non-chat models must be filtered)", data.Models, want)
	}
	if data.Current != "default-chat" {
		t.Fatalf("current = %q, want default-chat", data.Current)
	}
}

func TestChatSendsFormatAndMaxTokens(t *testing.T) {
	var (
		mu   sync.Mutex
		body = map[string]json.RawMessage{}
	)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]json.RawMessage
		_ = json.NewDecoder(r.Body).Decode(&req)
		mu.Lock()
		body = req
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ок\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"тест","format":"json","maxTokens":4096}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	mu.Lock()
	req := body
	mu.Unlock()

	var mt int
	if err := json.Unmarshal(req["max_tokens"], &mt); err != nil || mt != 4096 {
		t.Fatalf("max_tokens = %v, want 4096", req["max_tokens"])
	}
	if !strings.Contains(string(req["response_format"]), "json_object") {
		t.Fatalf("response_format = %s, want json_object", req["response_format"])
	}
	var model string
	_ = json.Unmarshal(req["model"], &model)
	if model != "test-model" {
		t.Fatalf("model = %q, want test-model", model)
	}
	var stream bool
	_ = json.Unmarshal(req["stream"], &stream)
	if !stream {
		t.Fatalf("stream = false, want true")
	}
}

func TestDefaultMaxTokensClamp(t *testing.T) {
	s := &Server{}
	if got := s.defaultMaxTokens(); got != 20000 {
		t.Fatalf("0 -> %d, want 20000", got)
	}
	s.cfg.MaxTokens = 250
	if got := s.defaultMaxTokens(); got != 250 {
		t.Fatalf("250 -> %d, want 250", got)
	}
	s.cfg.MaxTokens = 99999
	if got := s.defaultMaxTokens(); got != 20000 {
		t.Fatalf("99999 -> %d, want 20000", got)
	}
}

func TestNormalizeFormat(t *testing.T) {
	cases := map[string]string{
		"json": "json", "JSON": "json",
		"jsonschema": "jsonschema", "json_schema": "jsonschema",
		"jsonschema_structured": "jsonschema_structured", "structured": "jsonschema_structured",
		"text": "text", "": "text", "yaml": "text",
	}
	for in, want := range cases {
		if got := normalizeFormat(in); got != want {
			t.Fatalf("normalizeFormat(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestDetailsEndpoint(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"привет"}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("chat status = %d, want 200", resp.StatusCode)
	}

	get, err := http.Get(ts.URL + "/api/debug")
	if err != nil {
		t.Fatalf("GET /api/debug: %v", err)
	}
	defer get.Body.Close()
	var dbg struct {
		Request  string `json:"request"`
		Response string `json:"response"`
	}
	if err := json.NewDecoder(get.Body).Decode(&dbg); err != nil {
		t.Fatalf("decode /api/debug: %v", err)
	}

	if strings.TrimSpace(dbg.Request) == "" {
		t.Fatalf("request exchange is empty")
	}
	if strings.TrimSpace(dbg.Response) == "" {
		t.Fatalf("response exchange is empty")
	}

	var reqObj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(dbg.Request), &reqObj); err != nil {
		t.Fatalf("request is not valid JSON: %v", err)
	}
	if !strings.Contains(dbg.Request, "test-model") {
		t.Fatalf("request does not contain the model: %s", dbg.Request)
	}

	var respObj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(dbg.Response), &respObj); err != nil {
		t.Fatalf("response is not valid JSON: %v", err)
	}
	if !strings.Contains(dbg.Response, "Привет") {
		t.Fatalf("response does not contain the streamed content: %s", dbg.Response)
	}
}

func TestChatSendsJsonSchema(t *testing.T) {
	var (
		mu   sync.Mutex
		body = map[string]json.RawMessage{}
	)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]json.RawMessage
		_ = json.NewDecoder(r.Body).Decode(&req)
		mu.Lock()
		body = req
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"{\\\"ok\\\":true}\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	schema := `{"type":"object","properties":{"ok":{"type":"boolean"}}}`
	payload, _ := json.Marshal(map[string]any{
		"message":   "тест",
		"format":    "jsonschema_structured",
		"maxTokens": 4096,
		"schema":    schema,
	})
	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(string(payload)))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	mu.Lock()
	req := body
	mu.Unlock()

	rf := string(req["response_format"])
	if !strings.Contains(rf, "json_schema") {
		t.Fatalf("response_format = %s, want json_schema", rf)
	}
	if !strings.Contains(rf, `"strict":true`) {
		t.Fatalf("response_format = %s, want strict=true", rf)
	}
	if !strings.Contains(rf, "properties") || !strings.Contains(rf, `"ok"`) {
		t.Fatalf("response_format = %s, want embedded schema", rf)
	}
}

func TestJsonSchemaRequiresValidSchema(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	// Missing schema must be rejected.
	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"x","format":"jsonschema"}`))
	if err != nil {
		t.Fatalf("POST empty schema: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("empty schema status = %d, want 400", resp.StatusCode)
	}

	// Invalid (non-JSON) schema must be rejected.
	resp, err = http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"x","format":"jsonschema","schema":"not json"}`))
	if err != nil {
		t.Fatalf("POST invalid schema: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("invalid schema status = %d, want 400", resp.StatusCode)
	}
}

func TestTokenLimitDetail(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"Привет"}}]}`+"\n\n")
		fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"length"}]}`+"\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"привет","format":"text","maxTokens":10}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	var dbg struct {
		Request  string `json:"request"`
		Response string `json:"response"`
		Detail   struct {
			Type      string `json:"type"`
			Summary   string `json:"summary"`
			MaxTokens int    `json:"max_tokens"`
		} `json:"detail"`
	}
	get, err := http.Get(ts.URL + "/api/debug")
	if err != nil {
		t.Fatalf("GET /api/debug: %v", err)
	}
	defer get.Body.Close()
	if err := json.NewDecoder(get.Body).Decode(&dbg); err != nil {
		t.Fatalf("decode /api/debug: %v", err)
	}

	if dbg.Detail.Type != "token_limit" {
		t.Fatalf("detail.type = %q, want token_limit", dbg.Detail.Type)
	}
	if !strings.Contains(dbg.Detail.Summary, "лимит") {
		t.Fatalf("detail.summary = %q, want mention of limit", dbg.Detail.Summary)
	}
	if dbg.Detail.MaxTokens != 10 {
		t.Fatalf("detail.max_tokens = %d, want 10", dbg.Detail.MaxTokens)
	}
}

func TestSchemaViolationDetail(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"не json"}}]}`+"\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	schema := `{"type":"object","properties":{"ok":{"type":"boolean"}}}`
	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"x","format":"jsonschema_structured","schema":`+strconvQuote(schema)+`,"maxTokens":100}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	var dbg struct {
		Request  string `json:"request"`
		Response string `json:"response"`
		Detail   struct {
			Type    string `json:"type"`
			Summary string `json:"summary"`
			Content string `json:"content"`
		} `json:"detail"`
	}
	get, err := http.Get(ts.URL + "/api/debug")
	if err != nil {
		t.Fatalf("GET /api/debug: %v", err)
	}
	defer get.Body.Close()
	if err := json.NewDecoder(get.Body).Decode(&dbg); err != nil {
		t.Fatalf("decode /api/debug: %v", err)
	}

	if dbg.Detail.Type != "schema_violation" {
		t.Fatalf("detail.type = %q, want schema_violation", dbg.Detail.Type)
	}
	if !strings.Contains(dbg.Detail.Summary, "схем") {
		t.Fatalf("detail.summary = %q, want mention of schema", dbg.Detail.Summary)
	}
	if !strings.Contains(dbg.Detail.Content, "не json") {
		t.Fatalf("detail.content = %q, want the non-JSON reply", dbg.Detail.Content)
	}
}

func strconvQuote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func TestAnswersMatch(t *testing.T) {
	cases := []struct {
		reply    string
		expected string
		want     bool
	}{
		{"Ответ: 47", "47", true},
		{"Число равно 47.", "47", true},
		{"x = 47", "47", true},
		{"Получается 74.", "47", false},
		{"это не задача", "47", false},
		{"Ответ — сорок семь", "47", false},
		{"итог: 47", "47", true},
		{"Москва", "москва", true},
		{"Москва", "зеленоград", false},
		{"C(7/3; 0)", "C(7/3; 0)", true},
		{"C(7/3;0)", "C(7/3; 0)", true},           // пробел после ;
		{"C(7/3 ; 0)", "C(7/3; 0)", true},         // пробел перед ;
		{"точка C(7/3; 0)", "C(7/3; 0)", true},    // лишние слова
		{"C(\\frac{7}{3}; 0)", "C(7/3; 0)", true}, // \frac раскрывается в 7/3
		{"Все 100 синеглазых уедут", "Все 100 синеглазых покинут остров", true}, // стем-совпадение
		{"x = 3,5/1,5 = 7/3", "C(7/3; 0)", true}, // дробь в тексте ответа и в ожидании
		{"100 человек ушли", "Все 100 синеглазых покинут остров", false},
		{"", "47", false},
		{"47", "", false},
	}
	for _, c := range cases {
		if got := answersMatch(c.reply, c.expected); got != c.want {
			t.Errorf("answersMatch(%q, %q) = %v, want %v", c.reply, c.expected, got, c.want)
		}
	}
}

func TestMethodDisplayName(t *testing.T) {
	cases := []struct {
		method, expert, want string
	}{
		{"direct", "", "Прямой ответ"},
		{"stepwise", "", "Пошаговое решение"},
		{"prompt", "", "Через промпт"},
		{"expert", "mathematician", "Эксперт: Математик"},
		{"expert", "philosopher", "Эксперт: Философ"},
		{"expert", "critic", "Эксперт: Критик"},
		{"expert", "unknown", "Эксперт: Эксперт"},
	}
	for _, c := range cases {
		if got := methodDisplayName(c.method, c.expert); got != c.want {
			t.Errorf("methodDisplayName(%q, %q) = %q, want %q", c.method, c.expert, got, c.want)
		}
	}
}

func TestExpertLabel(t *testing.T) {
	if expertLabel("mathematician") != "Математик" {
		t.Error("expertLabel(mathematician) should be Математик")
	}
	if expertLabel("nope") != "Эксперт" {
		t.Error("expertLabel(nope) should default to Эксперт")
	}
}
