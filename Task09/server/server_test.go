package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
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
	if !strings.Contains(string(body), "Отправить") {
		t.Fatalf("index page does not look like the ICQ chat page")
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
	if got := s.defaultMaxTokens(); got != 1024 {
		t.Fatalf("0 -> %d, want 1024", got)
	}
	s.cfg.MaxTokens = 250
	if got := s.defaultMaxTokens(); got != 250 {
		t.Fatalf("250 -> %d, want 250", got)
	}
	s.cfg.MaxTokens = 99999
	if got := s.defaultMaxTokens(); got != 15000 {
		t.Fatalf("99999 -> %d, want 15000", got)
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

func TestClampRounds(t *testing.T) {
	cases := map[int]int{0: 6, -1: 6, 1: 1, 5: 5, 6: 6, 10: 10, 11: 10, 100: 10}
	for in, want := range cases {
		if got := clampRounds(in); got != want {
			t.Fatalf("clampRounds(%d) = %d, want %d", in, got, want)
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

// TestChatRunsToolLoop checks that the web chat performs a tool-calling round:
// the model first requests roll_dice, the result is fed back, and the streamed
// final answer is returned to the browser and kept in history.
func TestChatRunsToolLoop(t *testing.T) {
	var mu sync.Mutex
	reqCount := 0
	var secondMessages []map[string]json.RawMessage

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]json.RawMessage
		_ = json.NewDecoder(r.Body).Decode(&req)

		mu.Lock()
		reqCount++
		cur := reqCount
		if cur == 2 {
			_ = json.Unmarshal(req["messages"], &secondMessages)
		}
		mu.Unlock()

		w.Header().Set("Content-Type", "text/event-stream")
		if cur == 1 {
			// First round: the model asks for a die roll (tool call).
			fmt.Fprint(w, `data: {"choices":[{"delta":{"role":"assistant"}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"roll_dice","arguments":""}}]}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		} else {
			// Second round: the model answers after receiving the tool result.
			fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"Выпало 4"}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`+"\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		}
	}))
	defer upstream.Close()

	ts := newTestServer(t, upstream.URL)

	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"брось кубик"}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if got := string(body); got != "Выпало 4" {
		t.Fatalf("streamed body = %q, want %q", got, "Выпало 4")
	}

	// The tool result must have been translated into a tool turn sent in the
	// second request, so the model can use it.
	mu.Lock()
	defer mu.Unlock()
	if reqCount < 2 {
		t.Fatalf("expected >=2 upstream requests, got %d", reqCount)
	}
	foundTool := false
	for _, m := range secondMessages {
		var role string
		_ = json.Unmarshal(m["role"], &role)
		if role == "tool" {
			var id string
			_ = json.Unmarshal(m["tool_call_id"], &id)
			if id == "call_1" {
				foundTool = true
			}
		}
	}
	if !foundTool {
		t.Fatal("second request does not carry the tool result for call_1")
	}
}

// TestChatTraceReportsSteps checks that after a tool-calling round the
// /api/debug response exposes the interaction steps and aggregated token usage.
func TestChatTraceReportsSteps(t *testing.T) {
	var mu sync.Mutex
	reqCount := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		reqCount++
		cur := reqCount
		mu.Unlock()

		w.Header().Set("Content-Type", "text/event-stream")
		if cur == 1 {
			fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"roll_dice","arguments":""}}]}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n")
			fmt.Fprint(w, `data: {"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}`+"\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		} else {
			fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"4"}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`+"\n\n")
			fmt.Fprint(w, `data: {"usage":{"prompt_tokens":20,"completion_tokens":6,"total_tokens":26}}`+"\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		}
	}))
	defer upstream.Close()

	ts := newTestServer(t, upstream.URL)
	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"брось кубик","maxRounds":3}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", resp.StatusCode, body)
	}

	get, err := http.Get(ts.URL + "/api/debug")
	if err != nil {
		t.Fatalf("GET /api/debug: %v", err)
	}
	defer get.Body.Close()
	var dbg struct {
		Steps []struct {
			Label            string `json:"text"`
			Tool             string `json:"tool"`
			Args             string `json:"args"`
			Result           string `json:"result"`
			PromptTokens     int    `json:"prompt_tokens"`
			CompletionTokens int    `json:"completion_tokens"`
		}
		Tokens int   `json:"tokens"`
		TimeMS int64 `json:"time_ms"`
	}
	if err := json.NewDecoder(get.Body).Decode(&dbg); err != nil {
		t.Fatalf("decode /api/debug: %v", err)
	}
	if len(dbg.Steps) < 3 {
		t.Fatalf("steps len = %d, want >=3 (round, tool, round)", len(dbg.Steps))
	}
	foundTool := false
	for _, st := range dbg.Steps {
		if st.Tool == "roll_dice" && st.Args == "{}" && st.Result != "" {
			foundTool = true
		}
	}
	if !foundTool {
		t.Fatalf("no tool step with function+args+result: %+v", dbg.Steps)
	}
	// Round steps carry their own API call's usage; tool steps carry none.
	if dbg.Steps[0].PromptTokens != 10 || dbg.Steps[0].CompletionTokens != 4 {
		t.Fatalf("round 1 tokens = %d/%d, want 10/4", dbg.Steps[0].PromptTokens, dbg.Steps[0].CompletionTokens)
	}
	if dbg.Steps[len(dbg.Steps)-1].PromptTokens != 20 || dbg.Steps[len(dbg.Steps)-1].CompletionTokens != 6 {
		t.Fatalf("final round tokens = %d/%d, want 20/6", dbg.Steps[len(dbg.Steps)-1].PromptTokens, dbg.Steps[len(dbg.Steps)-1].CompletionTokens)
	}
	for _, st := range dbg.Steps {
		if st.Tool != "" && (st.PromptTokens != 0 || st.CompletionTokens != 0) {
			t.Fatalf("tool step has tokens: %+v", st)
		}
	}
	if dbg.Tokens != 40 {
		t.Fatalf("total tokens = %d, want 40 (14+26)", dbg.Tokens)
	}
	if dbg.TimeMS < 0 {
		t.Fatalf("time_ms = %d, want >=0", dbg.TimeMS)
	}
}

func strconvQuote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

// readState loads the persisted context file and fails the test on error.
func readState(t *testing.T, path string) historyState {
	t.Helper()
	st, err := loadState(path)
	if err != nil {
		t.Fatalf("loadState(%s): %v", path, err)
	}
	if st == nil {
		t.Fatalf("history file %s is missing or empty", path)
	}
	return *st
}

// newTestServerWithHistory is newTestServer with a conversation persistence
// file enabled.
func newTestServerWithHistory(t *testing.T, upstreamURL, historyFile string) *httptest.Server {
	t.Helper()
	client := api.New(upstreamURL, "test-key", 5*time.Second)
	cfg := config.Config{
		APIKey:      "test-key",
		BaseURL:     upstreamURL,
		Model:       "test-model",
		System:      "be brief",
		Timeout:     5 * time.Second,
		HistoryFile: historyFile,
	}
	srv := httptest.NewServer(New(client, cfg).Handler())
	t.Cleanup(srv.Close)
	return srv
}

func TestPersistencePersistsConversation(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	histFile := filepath.Join(t.TempDir(), "history.json")
	ts := newTestServerWithHistory(t, upstream.URL, histFile)

	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"привет"}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	// Read the full streamed body so the handler has finished before we inspect
	// the persisted file (the response is flushed after the first delta).
	if _, err := io.ReadAll(resp.Body); err != nil {
		t.Fatalf("read chat stream: %v", err)
	}
	resp.Body.Close()

	st := readState(t, modelHistoryPath(histFile, "test-model"))
	if st.Model != "test-model" {
		t.Fatalf("persisted model = %q, want test-model", st.Model)
	}
	if len(st.Messages) != 2 {
		t.Fatalf("persisted messages len = %d, want 2", len(st.Messages))
	}
	if st.Messages[0].Role != "user" || st.Messages[0].Content != "привет" {
		t.Fatalf("persisted[0] = %+v", st.Messages[0])
	}
	if st.Messages[1].Role != "assistant" || st.Messages[1].Content != "Привет, мир" {
		t.Fatalf("persisted[1] = %+v", st.Messages[1])
	}
}

func TestPersistenceClearedOnReset(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	histFile := filepath.Join(t.TempDir(), "history.json")
	ts := newTestServerWithHistory(t, upstream.URL, histFile)

	chatResp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"привет"}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	if _, err := io.ReadAll(chatResp.Body); err != nil {
		t.Fatalf("read chat stream: %v", err)
	}
	chatResp.Body.Close()

	resetResp, err := http.Post(ts.URL+"/api/reset", "application/json", nil)
	if err != nil {
		t.Fatalf("POST /api/reset: %v", err)
	}
	resetResp.Body.Close()

	// После сброса файл контекста текущей модели должен быть удалён.
	st, err := loadState(modelHistoryPath(histFile, "test-model"))
	if err != nil {
		t.Fatalf("loadState: %v", err)
	}
	if st != nil {
		t.Fatalf("expected context removed after reset, got %d messages", len(st.Messages))
	}
}

func TestPersistenceRestoresOnStart(t *testing.T) {
	histFile := filepath.Join(t.TempDir(), "history.json")
	if err := saveState(histFile, "test-model", []chatRecord{
		recordOf(api.UserMessage("привет"), time.Now()),
		recordOf(api.AssistantMessage("Привет, мир"), time.Now()),
	}); err != nil {
		t.Fatalf("saveState: %v", err)
	}

	client := api.New("http://127.0.0.1:1", "test-key", 5*time.Second)
	cfg := config.Config{
		APIKey:      "test-key",
		BaseURL:     "http://127.0.0.1:1",
		Model:       "other-model", // differs from the persisted model
		Models:      []string{"test-model", "other-model"},
		System:      "be brief",
		Timeout:     5 * time.Second,
		HistoryFile: histFile,
	}
	s := New(client, cfg)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	if s.model != "test-model" {
		t.Fatalf("restored model = %q, want test-model", s.model)
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
		t.Fatalf("hist[0] = %+v", hist.Messages[0])
	}
	if hist.Messages[1].Role != "assistant" || hist.Messages[1].Content != "Привет, мир" {
		t.Fatalf("hist[1] = %+v", hist.Messages[1])
	}
}

// TestPersistencePersistsToolCalls checks that the intermediate tool exchange
// (assistant tool_calls + tool result) is written to the persisted context
// together with the final answer, so a restart can replay the full turn.
func TestPersistencePersistsToolCalls(t *testing.T) {
	var mu sync.Mutex
	reqCount := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		reqCount++
		cur := reqCount
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		if cur == 1 {
			fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"roll_dice","arguments":""}}]}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		} else {
			fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"Выпало 4"}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`+"\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		}
	}))
	defer upstream.Close()

	histFile := filepath.Join(t.TempDir(), "history.json")
	ts := newTestServerWithHistory(t, upstream.URL, histFile)

	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"брось кубик"}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	if _, err := io.ReadAll(resp.Body); err != nil {
		t.Fatalf("read stream: %v", err)
	}
	resp.Body.Close()

	st := readState(t, modelHistoryPath(histFile, "test-model"))
	// Expected order: user, assistant(tool_calls), tool(result), assistant(final).
	if len(st.Messages) != 4 {
		t.Fatalf("persisted messages len = %d, want 4: %+v", len(st.Messages), st.Messages)
	}
	if st.Messages[0].Role != "user" || st.Messages[0].Content != "брось кубик" {
		t.Fatalf("persisted[0] = %+v", st.Messages[0])
	}
	if st.Messages[1].Role != "assistant" || len(st.Messages[1].ToolCalls) != 1 || st.Messages[1].ToolCalls[0].ID != "call_1" {
		t.Fatalf("persisted[1] = %+v", st.Messages[1])
	}
	if st.Messages[2].Role != "tool" || st.Messages[2].ToolCallID != "call_1" || st.Messages[2].Content == "" {
		t.Fatalf("persisted[2] = %+v", st.Messages[2])
	}
	if st.Messages[3].Role != "assistant" || st.Messages[3].Content != "Выпало 4" {
		t.Fatalf("persisted[3] = %+v", st.Messages[3])
	}
}

func TestPersistenceDisabledWhenEmpty(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	// HistoryFile empty => no file is written; the chat still works.
	ts := newTestServer(t, upstream.URL)

	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"привет"}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
}

// usageUpstream streams one answer plus a usage block reporting the given
// prompt/completion tokens (the accounting the context-token total is built on).
func usageUpstream(prompt, completion int) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"ок"}}]}`+"\n\n")
		fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`+"\n\n")
		fmt.Fprintf(w, `data: {"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}`+"\n\n", prompt, completion, prompt+completion)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
}

// TestPersistenceHistoryTokens checks the dialog-history token size: after each
// chat it equals the first request's prompt tokens plus the final answer's
// completion tokens (intermediate tool exchanges don't count; not a running
// sum). It is persisted in the per-model history file, restored on start,
// served via /api/debug and reset by /api/reset.
func TestPersistenceHistoryTokens(t *testing.T) {
	upstream := usageUpstream(10, 5)
	defer upstream.Close()
	histFile := filepath.Join(t.TempDir(), "history.json")
	ts := newTestServerWithHistory(t, upstream.URL, histFile)
	path := modelHistoryPath(histFile, "test-model")

	ctxOf := func(base string) int {
		t.Helper()
		get, err := http.Get(base + "/api/debug")
		if err != nil {
			t.Fatalf("GET /api/debug: %v", err)
		}
		defer get.Body.Close()
		var dbg struct {
			HistoryTokens int `json:"history_tokens"`
		}
		if err := json.NewDecoder(get.Body).Decode(&dbg); err != nil {
			t.Fatalf("decode /api/debug: %v", err)
		}
		return dbg.HistoryTokens
	}

	if got := ctxOf(ts.URL); got != 0 {
		t.Fatalf("initial history_tokens = %d, want 0", got)
	}

	// First exchange: prompt 10 + completion 5 -> context size 15, persisted.
	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"привет"}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	if _, err := io.ReadAll(resp.Body); err != nil {
		t.Fatalf("read chat stream: %v", err)
	}
	resp.Body.Close()
	if got := ctxOf(ts.URL); got != 15 {
		t.Fatalf("history_tokens after 1st chat = %d, want 15", got)
	}
	if st := readState(t, path); st.HistoryTokens != 15 {
		t.Fatalf("persisted history_tokens = %d, want 15", st.HistoryTokens)
	}

	// Second exchange replaces (not sums) the size: the new request already
	// carries the whole history, so the size stays prompt+completion.
	resp, err = http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"ещё"}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	if _, err := io.ReadAll(resp.Body); err != nil {
		t.Fatalf("read chat stream: %v", err)
	}
	resp.Body.Close()
	if got := ctxOf(ts.URL); got != 15 {
		t.Fatalf("history_tokens after 2nd chat = %d, want 15 (size, not a running sum)", got)
	}
	if st := readState(t, path); st.HistoryTokens != 15 {
		t.Fatalf("persisted history_tokens = %d, want 15", st.HistoryTokens)
	}

	// A fresh server instance restores the size from the file on start.
	ts2 := newTestServerWithHistory(t, upstream.URL, histFile)
	if got := ctxOf(ts2.URL); got != 15 {
		t.Fatalf("history_tokens after restart = %d, want 15", got)
	}

	// Deleting the dialog resets the total.
	resetResp, err := http.Post(ts2.URL+"/api/reset", "application/json", nil)
	if err != nil {
		t.Fatalf("POST /api/reset: %v", err)
	}
	resetResp.Body.Close()
	if got := ctxOf(ts2.URL); got != 0 {
		t.Fatalf("history_tokens after reset = %d, want 0", got)
	}
}

// toolRoundUpstream streams a two-round exchange: round 1 requests a tool
// (usage prompt 10 / completion 4), round 2 answers (prompt 20 / completion 6).
func toolRoundUpstream() *httptest.Server {
	var mu sync.Mutex
	reqCount := 0
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		reqCount++
		cur := reqCount
		mu.Unlock()

		w.Header().Set("Content-Type", "text/event-stream")
		if cur == 1 {
			fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"roll_dice","arguments":""}}]}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n")
			fmt.Fprint(w, `data: {"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}`+"\n\n")
		} else {
			fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"4"}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`+"\n\n")
			fmt.Fprint(w, `data: {"usage":{"prompt_tokens":20,"completion_tokens":6,"total_tokens":26}}`+"\n\n")
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
}

// TestHistoryTokensWithToolRounds verifies the history size across a tool
// exchange: it is the first round's prompt (10) plus the final answer's
// completion (6) = 16 — the intermediate tool round (prompt 20 / completion 4)
// is not counted.
func TestHistoryTokensWithToolRounds(t *testing.T) {
	upstream := toolRoundUpstream()
	defer upstream.Close()
	histFile := filepath.Join(t.TempDir(), "history.json")
	ts := newTestServerWithHistory(t, upstream.URL, histFile)

	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"брось кубик","maxRounds":3}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	if _, err := io.ReadAll(resp.Body); err != nil {
		t.Fatalf("read chat stream: %v", err)
	}
	resp.Body.Close()

	get, err := http.Get(ts.URL + "/api/debug")
	if err != nil {
		t.Fatalf("GET /api/debug: %v", err)
	}
	defer get.Body.Close()
	var dbg struct {
		HistoryTokens int `json:"history_tokens"`
	}
	if err := json.NewDecoder(get.Body).Decode(&dbg); err != nil {
		t.Fatalf("decode /api/debug: %v", err)
	}
	if dbg.HistoryTokens != 16 {
		t.Fatalf("history_tokens = %d, want 16 (first prompt 10 + final answer 6)", dbg.HistoryTokens)
	}
	if st := readState(t, modelHistoryPath(histFile, "test-model")); st.HistoryTokens != 16 {
		t.Fatalf("persisted history_tokens = %d, want 16", st.HistoryTokens)
	}
}

// TestParseDialog checks that a preset-dialog text is split into alternating
// user/assistant turns: empty lines are dropped and continuation lines are
// joined into the turn opened most recently.
func TestParseDialog(t *testing.T) {
	src := "User: Привет!\nModel: Здравствуйте!\nUser: Два слова\nвторая строка\nModel: Ответ\n\n"
	msgs := parseDialog(src)
	if len(msgs) != 4 {
		t.Fatalf("len = %d, want 4: %+v", len(msgs), msgs)
	}
	if msgs[0].Role != "user" || msgs[0].Content != "Привет!" {
		t.Fatalf("msgs[0] = %+v", msgs[0])
	}
	if msgs[1].Role != "assistant" || msgs[1].Content != "Здравствуйте!" {
		t.Fatalf("msgs[1] = %+v", msgs[1])
	}
	if msgs[2].Role != "user" || msgs[2].Content != "Два слова\nвторая строка" {
		t.Fatalf("msgs[2] = %+v", msgs[2])
	}
	if msgs[3].Role != "assistant" || msgs[3].Content != "Ответ" {
		t.Fatalf("msgs[3] = %+v", msgs[3])
	}
}

// TestHandleDialogLoadsPreset verifies that POST /api/dialog replaces the
// conversation with the preset turns (the system prompt is kept), persists
// them and answers in the /api/history shape. An unknown name yields 404.
func TestHandleDialogLoadsPreset(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	resp, err := http.Post(ts.URL+"/api/dialog", "application/json", strings.NewReader(`{"name":"curtains-kindergarten"}`))
	if err != nil {
		t.Fatalf("POST /api/dialog: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200, body=%s", resp.StatusCode, body)
	}
	var out struct {
		Messages []historyMessage `json:"messages"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode /api/dialog response: %v", err)
	}
	if len(out.Messages) == 0 || out.Messages[0].Role != "user" || !strings.Contains(out.Messages[0].Content, "Анна") {
		t.Fatalf("messages[0] = %+v", out.Messages)
	}
	// The preset is 15 dialog pairs (30 turns). The quiz is NOT loaded into
	// the context: its questions go out only via the "+" menu (/api/quiz).
	if len(out.Messages) != 30 {
		t.Fatalf("messages len = %d, want 30", len(out.Messages))
	}
	if out.Messages[29].Role != "assistant" || !strings.Contains(out.Messages[29].Content, "Спасибо за заказ") {
		t.Fatalf("messages[29] = %+v, want last dialog turn", out.Messages[29])
	}

	// The loaded dialog is the current context: /api/history returns the same
	// turns, so a page reload keeps showing them.
	get, err := http.Get(ts.URL + "/api/history")
	if err != nil {
		t.Fatalf("GET /api/history: %v", err)
	}
	defer get.Body.Close()
	var hist struct {
		Messages []historyMessage `json:"messages"`
	}
	if err := json.NewDecoder(get.Body).Decode(&hist); err != nil {
		t.Fatalf("decode /api/history: %v", err)
	}
	if len(hist.Messages) != len(out.Messages) || !reflect.DeepEqual(hist.Messages, out.Messages) {
		t.Fatalf("history differs from the preset response")
	}

	// Unknown preset names are rejected.
	bad, err := http.Post(ts.URL+"/api/dialog", "application/json", strings.NewReader(`{"name":"nope"}`))
	if err != nil {
		t.Fatalf("POST /api/dialog unknown: %v", err)
	}
	bad.Body.Close()
	if bad.StatusCode != http.StatusNotFound {
		t.Fatalf("unknown preset status = %d, want 404", bad.StatusCode)
	}
}

func TestParseQuiz(t *testing.T) {
	src := "Вопрос: Первый вопрос?\nОтвет: Первый ответ.\n\nВопрос: Второй\nвопрос?\nОтвет: Второй ответ.\n"
	msgs := parseQuiz(src)
	if len(msgs) != 4 {
		t.Fatalf("len = %d, want 4: %+v", len(msgs), msgs)
	}
	if msgs[0].Role != "user" || msgs[0].Content != "Первый вопрос?" {
		t.Fatalf("msgs[0] = %+v", msgs[0])
	}
	if msgs[1].Role != "assistant" || msgs[1].Content != "Первый ответ." {
		t.Fatalf("msgs[1] = %+v", msgs[1])
	}
	if msgs[2].Role != "user" || msgs[2].Content != "Второй\nвопрос?" {
		t.Fatalf("msgs[2] = %+v", msgs[2])
	}
	if msgs[3].Role != "assistant" || msgs[3].Content != "Второй ответ." {
		t.Fatalf("msgs[3] = %+v", msgs[3])
	}
}

// TestHandleQuiz verifies that GET /api/quiz serves the question/answer pairs
// of the preset quiz for the "+" menu.
func TestHandleQuiz(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	get, err := http.Get(ts.URL + "/api/quiz")
	if err != nil {
		t.Fatalf("GET /api/quiz: %v", err)
	}
	defer get.Body.Close()
	if get.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", get.StatusCode)
	}
	var out struct {
		Questions []quizPair `json:"questions"`
	}
	if err := json.NewDecoder(get.Body).Decode(&out); err != nil {
		t.Fatalf("decode /api/quiz: %v", err)
	}
	if len(out.Questions) != 9 {
		t.Fatalf("questions len = %d, want 9", len(out.Questions))
	}
	if !strings.Contains(out.Questions[0].Question, "коллекция") ||
		!strings.Contains(out.Questions[0].Answer, "ТексДизайн") {
		t.Fatalf("questions[0] = %+v", out.Questions[0])
	}
	if !strings.Contains(out.Questions[8].Answer, "42 315") {
		t.Fatalf("questions[8] = %+v", out.Questions[8])
	}
}

// TestHandleCompress verifies the compression: everything older than the last
// "keep" turns is replaced by a model-generated summary stored as a "summary"
// role entry; the response carries the raw model exchange. Compressing with a
// keep window larger than the history is rejected.
func TestHandleCompress(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	// The curtains preset: 30 dialog turns (the quiz is not loaded).
	if _, err := http.Post(ts.URL+"/api/dialog", "application/json", strings.NewReader(`{"name":"curtains-kindergarten"}`)); err != nil {
		t.Fatalf("POST /api/dialog: %v", err)
	}

	resp, err := http.Post(ts.URL+"/api/compress", "application/json", strings.NewReader(`{"keep":6}`))
	if err != nil {
		t.Fatalf("POST /api/compress: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200, body=%s", resp.StatusCode, body)
	}
	var out struct {
		OK         bool   `json:"ok"`
		Compressed int    `json:"compressed"`
		Kept       int    `json:"kept"`
		Summary    string `json:"summary"`
		Request    string `json:"request"`
		Response   string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode /api/compress: %v", err)
	}
	if !out.OK || out.Compressed != 24 || out.Kept != 6 {
		t.Fatalf("ok=%v compressed=%d kept=%d, want true/24/6", out.OK, out.Compressed, out.Kept)
	}
	// sseUpstream streams "Привет, мир" — that is the summary text.
	if out.Summary != "Привет, мир" {
		t.Fatalf("summary = %q, want %q", out.Summary, "Привет, мир")
	}
	if !strings.Contains(out.Request, "сводку") {
		t.Fatalf("model request does not contain the summary instruction: %s", out.Request)
	}

	// History: the summary marker followed by the six kept dialog turns (the
	// last three pairs of the curtains dialog).
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
	if len(hist.Messages) != 7 {
		t.Fatalf("history len = %d, want 7", len(hist.Messages))
	}
	if hist.Messages[0].Role != "summary" || hist.Messages[0].Content != "Привет, мир" {
		t.Fatalf("messages[0] = %+v, want summary marker", hist.Messages[0])
	}
	if hist.Messages[1].Role != "user" {
		t.Fatalf("messages[1] = %+v, want first kept dialog turn (user)", hist.Messages[1])
	}
	if hist.Messages[6].Role != "assistant" || !strings.Contains(hist.Messages[6].Content, "Спасибо за заказ") {
		t.Fatalf("messages[6] = %+v, want last dialog turn", hist.Messages[6])
	}

	// Nothing older than the window: keep larger than the history → not an
	// error, but an informational "ok" result with compressed=0.
	bad, err := http.Post(ts.URL+"/api/compress", "application/json", strings.NewReader(`{"keep":30}`))
	if err != nil {
		t.Fatalf("POST /api/compress second: %v", err)
	}
	defer bad.Body.Close()
	if bad.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(bad.Body)
		t.Fatalf("nothing-to-compress status = %d, want 200, body=%s", bad.StatusCode, body)
	}
	var info struct {
		OK         bool   `json:"ok"`
		Compressed int    `json:"compressed"`
		Kept       int    `json:"kept"`
		Message    string `json:"message"`
	}
	if err := json.NewDecoder(bad.Body).Decode(&info); err != nil {
		t.Fatalf("decode nothing-to-compress: %v", err)
	}
	if !info.OK || info.Compressed != 0 || info.Kept != 6 || info.Message == "" {
		t.Fatalf("ok=%v compressed=%d kept=%d message=%q, want true/0/6/<message>", info.OK, info.Compressed, info.Kept, info.Message)
	}
}

// TestHandleCompressWithoutSystem verifies compression when the config has no
// system prompt: then hist[0] is a conversation turn or a summary, not the
// system entry, and the cut must not skip it.
func TestHandleCompressWithoutSystem(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	client := api.New(upstream.URL, "test-key", 5*time.Second)
	cfg := config.Config{APIKey: "test-key", BaseURL: upstream.URL, Model: "test-model", Timeout: 5 * time.Second}
	ts := httptest.NewServer(New(client, cfg).Handler())
	t.Cleanup(ts.Close)

	// 30 dialog turns, then compress all but the last three (keep is clamped
	// to the 3..30 range).
	if _, err := http.Post(ts.URL+"/api/dialog", "application/json", strings.NewReader(`{"name":"curtains-kindergarten"}`)); err != nil {
		t.Fatalf("POST /api/dialog: %v", err)
	}
	resp, err := http.Post(ts.URL+"/api/compress", "application/json", strings.NewReader(`{"keep":3}`))
	if err != nil {
		t.Fatalf("POST /api/compress: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200, body=%s", resp.StatusCode, body)
	}

	// History: the summary marker followed by the three kept turns.
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
	if len(hist.Messages) != 4 {
		t.Fatalf("history len = %d, want 4", len(hist.Messages))
	}
	if hist.Messages[0].Role != "summary" || hist.Messages[0].Content != "Привет, мир" {
		t.Fatalf("messages[0] = %+v, want summary marker", hist.Messages[0])
	}
	if hist.Messages[3].Role != "assistant" || !strings.Contains(hist.Messages[3].Content, "Спасибо за заказ") {
		t.Fatalf("messages[3] = %+v, want last dialog turn", hist.Messages[3])
	}
}
