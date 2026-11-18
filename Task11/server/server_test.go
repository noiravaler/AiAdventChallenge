package server

import (
	"bufio"
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
	"ai-chat-go/storage"
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

// loadLongContext fills the context with 30 displayed turns: the memory-check
// preset (10 user turns) plus five plain exchanges (20 turns total).
// preset used to provide this history; the compression tests keep their
// original arithmetic with this helper.
func loadLongContext(t *testing.T, ts *httptest.Server) {
	t.Helper()
	resp, err := http.Post(ts.URL+"/api/dialog", "application/json", strings.NewReader(`{"name":"memory-check"}`))
	if err != nil || resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		t.Fatalf("POST /api/dialog: %v", err)
	}
	resp.Body.Close()
	for i := 0; i < 5; i++ {
		r, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"привет"}`))
		if err != nil || r.StatusCode != http.StatusOK {
			r.Body.Close()
			t.Fatalf("POST /api/chat #%d: %v", i, err)
		}
		r.Body.Close()
	}
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
			Sub              []struct {
				Tool             string `json:"tool"`
				Args             string `json:"args"`
				Result           string `json:"result"`
				PromptTokens     int    `json:"prompt_tokens"`
				CompletionTokens int    `json:"completion_tokens"`
			} `json:"sub"`
		}
		Tokens int   `json:"tokens"`
		TimeMS int64 `json:"time_ms"`
	}
	if err := json.NewDecoder(get.Body).Decode(&dbg); err != nil {
		t.Fatalf("decode /api/debug: %v", err)
	}
	// Two rounds; tool executions are nested inside the round that requested
	// them.
	if len(dbg.Steps) != 2 {
		t.Fatalf("steps len = %d, want 2 (round, round): %+v", len(dbg.Steps), dbg.Steps)
	}
	foundTool := false
	for _, st := range dbg.Steps {
		for _, sub := range st.Sub {
			if sub.Tool == "roll_dice" && sub.Args == "{}" && sub.Result != "" {
				foundTool = true
			}
		}
	}
	if !foundTool {
		t.Fatalf("no nested tool step with function+args+result: %+v", dbg.Steps)
	}
	// Round steps carry their own API call's usage; nested tool steps none.
	if dbg.Steps[0].PromptTokens != 10 || dbg.Steps[0].CompletionTokens != 4 {
		t.Fatalf("round 1 tokens = %d/%d, want 10/4", dbg.Steps[0].PromptTokens, dbg.Steps[0].CompletionTokens)
	}
	if dbg.Steps[1].PromptTokens != 20 || dbg.Steps[1].CompletionTokens != 6 {
		t.Fatalf("final round tokens = %d/%d, want 20/6", dbg.Steps[1].PromptTokens, dbg.Steps[1].CompletionTokens)
	}
	for _, st := range dbg.Steps {
		for _, sub := range st.Sub {
			if sub.Tool != "" && (sub.PromptTokens != 0 || sub.CompletionTokens != 0) {
				t.Fatalf("nested tool step has tokens: %+v", sub)
			}
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
	if err := saveState(histFile, historyState{Model: "test-model", Messages: []chatRecord{
		recordOf(histEntry{At: time.Now(), Msg: api.UserMessage("привет")}),
		recordOf(histEntry{At: time.Now(), Msg: api.AssistantMessage("Привет, мир")}),
	}}); err != nil {
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

	resp, err := http.Post(ts.URL+"/api/dialog", "application/json", strings.NewReader(`{"name":"memory-check"}`))
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
	// The preset is 10 user turns (the file has no model answers: the run
	// keeps the model's real replies). The quiz is NOT loaded into the
	// context: its questions go out only via the "+" menu (/api/quiz).
	if len(out.Messages) != 10 {
		t.Fatalf("messages len = %d, want 10", len(out.Messages))
	}
	if out.Messages[9].Role != "user" || !strings.Contains(out.Messages[9].Content, "Подытожьте") {
		t.Fatalf("messages[9] = %+v, want last user turn", out.Messages[9])
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

// TestMemoryCheckReplayRunsAllPairs verifies the replay of the memory-check
// preset: loading puts the 10 scenario user turns into the context, the
// replay runs all 10 of them through the model (start/total=10 … done) and
// the model's real answers are appended to the dialog.
func TestMemoryCheckReplayRunsAllPairs(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	// Loading the preset as the current context: 20 messages.
	resp, err := http.Post(ts.URL+"/api/dialog", "application/json", strings.NewReader(`{"name":"memory-check"}`))
	if err != nil {
		t.Fatalf("POST /api/dialog: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	// The replay: all ten pairs are sent through the model.
	run, err := http.Post(ts.URL+"/api/dialog/run", "application/json",
		strings.NewReader(`{"name":"memory-check"}`))
	if err != nil {
		t.Fatalf("POST /api/dialog/run: %v", err)
	}
	defer run.Body.Close()
	var events []map[string]any
	sc := bufio.NewScanner(run.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 4<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var ev map[string]any
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("decode event %q: %v", line, err)
		}
		events = append(events, ev)
	}
	if len(events) == 0 || events[0]["type"] != "start" || events[0]["total"] != float64(10) {
		t.Fatalf("first event = %+v, want start/10", events[0])
	}
	if events[len(events)-1]["type"] != "done" {
		t.Fatalf("last event = %+v, want done", events[len(events)-1])
	}

	// Прогон продолжает диалог: 20 загруженных реплик + 10 пар прогона.
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
	if len(hist.Messages) != 30 {
		t.Fatalf("history len = %d, want 30 (load + replay)", len(hist.Messages))
	}
	if !strings.Contains(hist.Messages[29].Content, "Привет, мир") {
		t.Fatalf("messages[29] = %+v, want the model's last answer", hist.Messages[29])
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

// TestMemoryCheckPreset verifies that the "memory-check" scenario asset is
// well-formed: 10 user turns (no model answers — the run keeps the model's
// real replies) and 6 quiz questions — the shapes the dialog replay and the
// "+" menu rely on.
func TestMemoryCheckPreset(t *testing.T) {
	msgs := parseDialog(presetDialogs["memory-check"].dialog)
	if len(msgs) != 10 {
		t.Fatalf("memory-check dialog turns = %d, want 10", len(msgs))
	}
	for i, m := range msgs {
		if m.Role != "user" {
			t.Fatalf("memory-check turn %d role = %s, want user", i, m.Role)
		}
	}
	q := parseQuiz(presetDialogs["memory-check"].quiz)
	if len(q) != 12 {
		t.Fatalf("memory-check quiz turns = %d, want 12", len(q))
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
	// 6 вопросов викторины «Проверка памяти».
	if len(out.Questions) != 6 {
		t.Fatalf("questions len = %d, want 6", len(out.Questions))
	}
	if !strings.Contains(out.Questions[0].Question, "Как зовут собеседницу") ||
		!strings.Contains(out.Questions[0].Answer, "Анна") {
		t.Fatalf("questions[0] = %+v", out.Questions[0])
	}
	if !strings.Contains(out.Questions[5].Answer, "Восемь") {
		t.Fatalf("questions[5] = %+v", out.Questions[5])
	}
}

// TestAutoCompressSummary verifies on-the-fly summary compression: after a
// chat exchange that pushes the displayed turn count past the configured
// context size, everything older than the kept window is replaced by a
// model-generated summary stored as a "summary" role entry.
// TestCompressSettingsEndpoint verifies POST /api/compress: the compression
// settings from the panel apply immediately — an existing dialog that is
// over the new threshold is compressed right away, without waiting for the
// next message.
func TestCompressSettingsEndpoint(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	// 20 dialog turns via the preset (no compression during the load).
	resp, err := http.Post(ts.URL+"/api/dialog", "application/json", strings.NewReader(`{"name":"memory-check"}`))
	if err != nil || resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		t.Fatalf("POST /api/dialog: %v", err)
	}
	resp.Body.Close()

	histLen := func() int {
		t.Helper()
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
		return len(hist.Messages)
	}
	if n := histLen(); n != 10 {
		t.Fatalf("history len = %d, want 10 after the preset load", n)
	}

	// Sliding window with keep=4/threshold=5: the 10-turn dialog is over the
	// threshold, so the settings endpoint compresses immediately.
	post, err := http.Post(ts.URL+"/api/compress", "application/json",
		strings.NewReader(`{"enabled":true,"strategy":"sliding","keep":4,"threshold":5}`))
	if err != nil {
		t.Fatalf("POST /api/compress: %v", err)
	}
	var out struct {
		OK         bool           `json:"ok"`
		Compressed bool           `json:"compressed"`
		Info       map[string]any `json:"info"`
	}
	if err := json.NewDecoder(post.Body).Decode(&out); err != nil {
		t.Fatalf("decode /api/compress: %v", err)
	}
	post.Body.Close()
	if !out.OK || !out.Compressed {
		t.Fatalf("ok=%v compressed=%v, want true/true", out.OK, out.Compressed)
	}
	if n := histLen(); n != 5 {
		t.Fatalf("history len = %d, want 5 (fold marker + 4 kept) after the immediate compression", n)
	}

	// The dialog is now within the threshold: nothing to compress.
	post, err = http.Post(ts.URL+"/api/compress", "application/json",
		strings.NewReader(`{"enabled":true,"strategy":"sliding","keep":4,"threshold":5}`))
	if err != nil {
		t.Fatalf("POST /api/compress #2: %v", err)
	}
	if err := json.NewDecoder(post.Body).Decode(&out); err != nil {
		t.Fatalf("decode /api/compress #2: %v", err)
	}
	post.Body.Close()
	if !out.OK || out.Compressed {
		t.Fatalf("ok=%v compressed=%v, want true/false", out.OK, out.Compressed)
	}
	if n := histLen(); n != 5 {
		t.Fatalf("history len = %d after the no-op pass, want 5 (unchanged)", n)
	}
}

// TestSummaryExtractsTask verifies that the summary compression maintains the
// working-memory task: the model appends a «ЗАДАЧА: …» line to the summary,
// the line moves into the working layer and is not shown in the summary text.
func TestSummaryExtractsTask(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		body, _ := io.ReadAll(r.Body)
		if strings.Contains(string(body), "сжать историю диалога") {
			for _, part := range []string{"Сводка переписки.\n", "ЗАДАЧА: подобрать стеллажи"} {
				fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", part)
			}
		} else {
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Привет, мир\"}}]}\n\n")
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	// 30 dialog turns so the next exchange is over the default threshold.
	loadLongContext(t, ts)

	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(
		`{"message":"привет","compress":{"enabled":true,"strategy":"summary","keep":6}}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	get, err := http.Get(ts.URL + "/api/memory")
	if err != nil {
		t.Fatalf("GET /api/memory: %v", err)
	}
	defer get.Body.Close()
	var mem struct {
		Working struct {
			Task string `json:"task"`
		} `json:"working"`
	}
	if err := json.NewDecoder(get.Body).Decode(&mem); err != nil {
		t.Fatalf("decode /api/memory: %v", err)
	}
	if mem.Working.Task != "подобрать стеллажи" {
		t.Fatalf("working.task = %q, want «подобрать стеллажи»", mem.Working.Task)
	}

	get, err = http.Get(ts.URL + "/api/history")
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
	if len(hist.Messages) == 0 || hist.Messages[0].Role != "summary" {
		t.Fatalf("messages[0] = %+v, want the summary marker", hist.Messages)
	}
	if strings.Contains(hist.Messages[0].Content, "ЗАДАЧА") {
		t.Fatalf("summary text = %q, want the task line stripped", hist.Messages[0].Content)
	}
}

func TestAutoCompressSummary(t *testing.T) {
	// The upstream reports usage so the dialog-wide prompt total can be
	// checked; every call (chat rounds and the summary call) sees the same
	// usage block.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, part := range []string{"При", "вет", ", мир"} {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", part)
		}
		fmt.Fprint(w, "data: {\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":5,\"total_tokens\":16}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	// 30 dialog turns via the helper (see loadLongContext).
	loadLongContext(t, ts)

	// One more exchange with on-the-fly compression enabled (keep 6): the
	// history grows to 32 displayed turns, so 26 are folded into a summary.
	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(
		`{"message":"привет","compress":{"enabled":true,"strategy":"summary","keep":6}}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	// History: the summary marker followed by the six kept turns (the new
	// pair and the last dialog pairs).
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
	if hist.Messages[6].Role != "assistant" || hist.Messages[6].Content != "Привет, мир" {
		t.Fatalf("messages[6] = %+v, want the new assistant answer", hist.Messages[6])
	}

	// Details: the compression is reported with its counters, the nested step
	// carries the summary and the dialog-wide prompt-token total includes the
	// compression call.
	dbg, err := http.Get(ts.URL + "/api/debug")
	if err != nil {
		t.Fatalf("GET /api/debug: %v", err)
	}
	defer dbg.Body.Close()
	var d struct {
		CompressSeq  int            `json:"compress_seq"`
		LastCompress map[string]any `json:"last_compress"`
		Steps        []struct {
			Text string `json:"text"`
			Sub  []struct {
				Text   string `json:"text"`
				Result string `json:"result"`
			} `json:"sub"`
		} `json:"steps"`
	}
	if err := json.NewDecoder(dbg.Body).Decode(&d); err != nil {
		t.Fatalf("decode debug: %v", err)
	}
	if d.CompressSeq < 1 || d.LastCompress == nil || d.LastCompress["strategy"] != "summary" {
		t.Fatalf("compress_seq=%d last_compress=%v, want summary compression", d.CompressSeq, d.LastCompress)
	}
	found := false
	for _, st := range d.Steps {
		for _, sub := range st.Sub {
			if strings.Contains(sub.Text, "(Summary)") && sub.Result == "Привет, мир" {
				found = true
			}
		}
	}
	if !found {
		t.Fatalf("no nested summary compression step: %+v", d.Steps)
	}
}

// TestAutoCompressWithoutSystem verifies compression when the config has no
// system prompt: then hist[0] is a conversation turn or a summary, not the
// system entry, and the cut must not skip it.
func TestAutoCompressWithoutSystem(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	client := api.New(upstream.URL, "test-key", 5*time.Second)
	cfg := config.Config{APIKey: "test-key", BaseURL: upstream.URL, Model: "test-model", Timeout: 5 * time.Second}
	ts := httptest.NewServer(New(client, cfg).Handler())
	t.Cleanup(ts.Close)

	// 30 dialog turns, then one more exchange with keep=3: 29 older turns are
	// folded into the summary.
	loadLongContext(t, ts)
	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(
		`{"message":"привет","compress":{"enabled":true,"strategy":"summary","keep":3}}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
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
	if hist.Messages[3].Role != "assistant" || hist.Messages[3].Content != "Привет, мир" {
		t.Fatalf("messages[3] = %+v, want the new assistant answer", hist.Messages[3])
	}
}

// TestAutoCompressSliding verifies the Sliding Window strategy: turns older
// than the kept window are dropped without a model call and a display-only
// marker is stored in the history.
func TestAutoCompressSliding(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	loadLongContext(t, ts)
	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(
		`{"message":"привет","compress":{"enabled":true,"strategy":"sliding","keep":3}}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

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
		t.Fatalf("history len = %d, want 4 (marker + 3 kept)", len(hist.Messages))
	}
	if hist.Messages[0].Role != "marker" || !strings.Contains(hist.Messages[0].Content, "Sliding Window") {
		t.Fatalf("messages[0] = %+v, want sliding marker", hist.Messages[0])
	}
	if hist.Messages[3].Role != "assistant" || hist.Messages[3].Content != "Привет, мир" {
		t.Fatalf("messages[3] = %+v, want the new assistant answer", hist.Messages[3])
	}

	// No model call was needed for the cut: the summary text never appears as
	// a summary entry, and the compression info reports the strategy.
	dbg, err := http.Get(ts.URL + "/api/debug")
	if err != nil {
		t.Fatalf("GET /api/debug: %v", err)
	}
	defer dbg.Body.Close()
	var d struct {
		LastCompress map[string]any `json:"last_compress"`
	}
	if err := json.NewDecoder(dbg.Body).Decode(&d); err != nil {
		t.Fatalf("decode debug: %v", err)
	}
	if d.LastCompress == nil || d.LastCompress["strategy"] != "sliding" || d.LastCompress["compressed"] != float64(19) {
		t.Fatalf("last_compress = %v, want sliding/19", d.LastCompress)
	}
}

// TestAutoCompressStickyFacts verifies the Sticky Facts strategy: the model
// maintains a "ключ: значение" facts block, old turns are dropped and every
// subsequent request receives the facts block plus the kept messages.
func TestAutoCompressStickyFacts(t *testing.T) {
	var mu sync.Mutex
	var seen []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		seen = append(seen, string(body))
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		for _, part := range []string{"При", "вет", ", мир"} {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", part)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	loadLongContext(t, ts)
	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(
		`{"message":"привет","compress":{"enabled":true,"strategy":"sticky","keep":4,"threshold":5}}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	// History: marker + 4 kept turns.
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
	if len(hist.Messages) != 5 {
		t.Fatalf("history len = %d, want 5 (marker + 4 kept)", len(hist.Messages))
	}
	if hist.Messages[0].Role != "marker" || !strings.Contains(hist.Messages[0].Content, "Sticky Facts") {
		t.Fatalf("messages[0] = %+v, want sticky marker", hist.Messages[0])
	}

	// The next exchange must receive the facts block (the model's "Привет,
	// мир" answer became the facts text) ahead of the conversation; with 6
	// displayed turns past the threshold (5) it triggers a second facts
	// extraction.
	resp2, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(
		`{"message":"ещё вопрос","compress":{"enabled":true,"strategy":"sticky","keep":4,"threshold":5}}`))
	if err != nil {
		t.Fatalf("POST /api/chat second: %v", err)
	}
	io.Copy(io.Discard, resp2.Body)
	resp2.Body.Close()

	mu.Lock()
	defer mu.Unlock()
	if len(seen) < 4 {
		t.Fatalf("upstream saw %d requests, want >=4", len(seen))
	}
	// seen: chat 1 round 1, facts extraction, chat 2 round 1, facts extraction.
	facts := seen[1]
	if !strings.Contains(facts, "ключ") {
		t.Fatalf("facts request misses the instruction: %s", facts)
	}
	var chat2 string
	for _, req := range seen {
		if strings.Contains(req, "ещё вопрос") {
			chat2 = req
		}
	}
	if !strings.Contains(chat2, "Блок важных фактов") || !strings.Contains(chat2, "Привет, мир") {
		t.Fatalf("next chat request misses the facts block: %s", chat2)
	}
}

// TestCompressThresholdGates verifies the two compression sliders: the
// trigger ("Порог контекста") and the kept window ("Количество сообщений").
// A threshold above the conversation size prevents compression entirely; a
// low threshold with a smaller keep window compresses down to the window.
func TestCompressThresholdGates(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	loadLongContext(t, ts)
	// 20 preset/chat turns + one exchange = 22 displayed turns; threshold 40
	// is never reached, so nothing is compressed.
	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(
		`{"message":"привет","compress":{"enabled":true,"strategy":"sliding","keep":3,"threshold":40}}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

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
	if len(hist.Messages) != 22 {
		t.Fatalf("history len = %d, want 22 (threshold 40 not reached)", len(hist.Messages))
	}

	// Lower the threshold to 5 with keep=3: the next exchange (23 turns)
	// compresses down to the 3-message window.
	resp2, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(
		`{"message":"привет","compress":{"enabled":true,"strategy":"sliding","keep":3,"threshold":5}}`))
	if err != nil {
		t.Fatalf("POST /api/chat second: %v", err)
	}
	io.Copy(io.Discard, resp2.Body)
	resp2.Body.Close()

	get2, err := http.Get(ts.URL + "/api/history")
	if err != nil {
		t.Fatalf("GET /api/history: %v", err)
	}
	defer get2.Body.Close()
	var hist2 struct {
		Messages []historyMessage `json:"messages"`
	}
	if err := json.NewDecoder(get2.Body).Decode(&hist2); err != nil {
		t.Fatalf("decode history: %v", err)
	}
	if len(hist2.Messages) != 4 {
		t.Fatalf("history len = %d, want 4 (marker + 3 kept)", len(hist2.Messages))
	}
	if hist2.Messages[0].Role != "marker" {
		t.Fatalf("messages[0] = %+v, want a compression marker", hist2.Messages[0])
	}
}

// TestValidateEndpoint verifies POST /api/validate: the model judges the
// answer against the expected one, the verdict reaches the UI and the call's
// tokens join the dialog-wide total.
func TestValidateEndpoint(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"{\"correct\": true, \"comment\": \"совпадает\"}"}}],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}`)
	}))
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	resp, err := http.Post(ts.URL+"/api/validate", "application/json", strings.NewReader(
		`{"answer":"Коллекция Сон","expected":"Коллекция «Сон» от фабрики «ТексДизайн»"}`))
	if err != nil {
		t.Fatalf("POST /api/validate: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200, body=%s", resp.StatusCode, body)
	}
	var out struct {
		Correct bool   `json:"correct"`
		Comment string `json:"comment"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode validate: %v", err)
	}
	if !out.Correct || out.Comment != "совпадает" {
		t.Fatalf("correct=%v comment=%q, want true/совпадает", out.Correct, out.Comment)
	}

	dbg, err := http.Get(ts.URL + "/api/debug")
	if err != nil {
		t.Fatalf("GET /api/debug: %v", err)
	}
	defer dbg.Body.Close()
	var d struct {
		Steps []struct {
			Text string `json:"text"`
			Sub  []struct {
				Text string `json:"text"`
			} `json:"sub"`
		} `json:"steps"`
	}
	if err := json.NewDecoder(dbg.Body).Decode(&d); err != nil {
		t.Fatalf("decode debug: %v", err)
	}
	found := false
	for _, st := range d.Steps {
		if strings.Contains(st.Text, "Проверка ожидаемого ответа: верно") {
			found = true
		}
		for _, sub := range st.Sub {
			if strings.Contains(sub.Text, "Проверка ожидаемого ответа: верно") {
				found = true
			}
		}
	}
	if !found {
		t.Fatalf("no validation step in steps: %+v", d.Steps)
	}
}

// TestDialogRunReplays verifies POST /api/dialog/run: every user turn of the
// preset is sent to the model as its own request, the replay is streamed as
// NDJSON events, and the stored history ends up equal to the scenario file
// The replay sends each user turn to the model and keeps the model's real
// answers: the stored history ends with the upstream's own replies.
func TestDialogRunReplays(t *testing.T) {
	var mu sync.Mutex
	var chatRequests int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if strings.Contains(string(body), `"tools":[`) {
			mu.Lock()
			chatRequests++
			mu.Unlock()
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"заглушка\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":5,\"total_tokens\":16}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	resp, err := http.Post(ts.URL+"/api/dialog/run", "application/json",
		strings.NewReader(`{"name":"memory-check","compress":{"enabled":false}}`))
	if err != nil {
		t.Fatalf("POST /api/dialog/run: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200, body=%s", resp.StatusCode, body)
	}

	var events []map[string]any
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 4<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var ev map[string]any
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("decode event %q: %v", line, err)
		}
		events = append(events, ev)
	}
	if len(events) == 0 || events[0]["type"] != "start" || events[0]["total"] != float64(10) {
		t.Fatalf("first event = %+v, want start/10", events[0])
	}
	last := events[len(events)-1]
	if last["type"] != "done" {
		t.Fatalf("last event = %+v, want done", last)
	}
	users, turns, deltas := 0, 0, 0
	for _, ev := range events {
		switch ev["type"] {
		case "user":
			users++
		case "turn":
			turns++
		case "delta":
			deltas++
		}
	}
	if users != 10 || turns != 10 || deltas == 0 {
		t.Fatalf("users=%d turns=%d deltas=%d, want 10/10/>0", users, turns, deltas)
	}
	// The final turn event carries the model's real answer.
	for _, ev := range events {
		if ev["type"] == "turn" && ev["index"] == float64(9) {
			if !strings.Contains(fmt.Sprint(ev["content"]), "заглушка") {
				t.Fatalf("last turn content = %v, want the model's answer", ev["content"])
			}
		}
	}

	if chatRequests != 10 {
		t.Fatalf("model chat requests = %d, want 10", chatRequests)
	}

	// The done event carries the run's wall time.
	done := events[len(events)-1]
	if ms, ok := done["time_ms"].(float64); !ok || ms < 0 {
		t.Fatalf("done.time_ms = %v, want a non-negative number", done["time_ms"])
	}

	// The stored history equals the scenario: 30 turns, no markers.
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
	if len(hist.Messages) != 20 {
		t.Fatalf("history len = %d, want 20", len(hist.Messages))
	}
	if !strings.Contains(hist.Messages[19].Content, "заглушка") {
		t.Fatalf("messages[19] = %+v, want the model's last answer", hist.Messages[19])
	}
}

// TestCompressForceFoldOnKeepReduce verifies that reducing «Количество
// сообщений» (the kept window) folds the chat immediately — even below the
// compression threshold — instead of waiting for new messages to cross it.
func TestCompressForceFoldOnKeepReduce(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	resp, err := http.Post(ts.URL+"/api/dialog", "application/json", strings.NewReader(`{"name":"memory-check"}`))
	if err != nil || resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		t.Fatalf("POST /api/dialog: %v", err)
	}
	resp.Body.Close()

	// keep=4 but threshold=40: without force nothing would fold (10 turns
	// are far below the threshold); the explicit settings apply must fold
	// the dialog down to the kept window right away.
	post, err := http.Post(ts.URL+"/api/compress", "application/json",
		strings.NewReader(`{"enabled":true,"strategy":"sliding","keep":4,"threshold":40}`))
	if err != nil {
		t.Fatalf("POST /api/compress: %v", err)
	}
	var out struct {
		OK         bool           `json:"ok"`
		Compressed bool           `json:"compressed"`
		Info       map[string]any `json:"info"`
	}
	if err := json.NewDecoder(post.Body).Decode(&out); err != nil {
		t.Fatalf("decode /api/compress: %v", err)
	}
	post.Body.Close()
	if !out.OK || !out.Compressed {
		t.Fatalf("compressed = %v (ok=%v), want an immediate fold on keep reduce", out.Compressed, out.OK)
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
	// 1 fold marker + 4 kept turns.
	if len(hist.Messages) != 5 {
		t.Fatalf("history len = %d, want 5 (marker + 4 kept)", len(hist.Messages))
	}
	if hist.Messages[0].Role != "marker" {
		t.Fatalf("messages[0].Role = %q, want the fold marker", hist.Messages[0].Role)
	}
}

// TestShortTokensEstimatedAfterCompression verifies that the short-term
// layer reports a non-zero approximate token volume right after a
// compression (which resets the API-reported branch token count to zero).
func TestShortTokensEstimatedAfterCompression(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	resp, err := http.Post(ts.URL+"/api/dialog", "application/json", strings.NewReader(`{"name":"memory-check"}`))
	if err != nil || resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		t.Fatalf("POST /api/dialog: %v", err)
	}
	resp.Body.Close()

	post, err := http.Post(ts.URL+"/api/compress", "application/json",
		strings.NewReader(`{"enabled":true,"strategy":"sliding","keep":4,"threshold":40}`))
	if err != nil {
		t.Fatalf("POST /api/compress: %v", err)
	}
	post.Body.Close()

	get, err := http.Get(ts.URL + "/api/memory")
	if err != nil {
		t.Fatalf("GET /api/memory: %v", err)
	}
	defer get.Body.Close()
	var mem struct {
		Short struct {
			Messages int `json:"messages"`
			Tokens   int `json:"tokens"`
		} `json:"short"`
	}
	if err := json.NewDecoder(get.Body).Decode(&mem); err != nil {
		t.Fatalf("decode /api/memory: %v", err)
	}
	if mem.Short.Messages != 4 {
		t.Fatalf("short.messages = %d, want 4 kept turns", mem.Short.Messages)
	}
	if mem.Short.Tokens <= 0 {
		t.Fatalf("short.tokens = %d, want an estimate > 0 after compression", mem.Short.Tokens)
	}
}

// sseEmptyOnceUpstream replies with an empty assistant content for the first
// chat completion and with a real text afterwards — to check that the
// scenario run retries an unanswered replica instead of moving on silently.
func sseEmptyOnceUpstream() *httptest.Server {
	var mu sync.Mutex
	calls := 0
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls++
		first := calls == 1
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		var payload string
		if !first {
			payload = "Ответ модели"
		}
		b, _ := json.Marshal(map[string]any{
			"id": "chatcmpl-empty-once", "object": "chat.completion.chunk", "created": 1, "model": "test",
			"choices": []map[string]any{{"index": 0, "delta": map[string]any{"content": payload}}},
		})
		fmt.Fprintf(w, "data: %s\n\n", b)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
}

// TestDialogRunRetriesEmptyAnswer verifies that a model turn returning an
// empty answer is retried (up to two more attempts): the replica gets a real
// answer, and the run proceeds instead of silently leaving the user message
// unanswered.
func TestDialogRunRetriesEmptyAnswer(t *testing.T) {
	upstream := sseEmptyOnceUpstream()
	defer upstream.Close()
	dbFile := filepath.Join(t.TempDir(), "chat.db")
	ts := newTestServerWithDB(t, upstream.URL, dbFile)

	var events []map[string]any
	run, err := http.Post(ts.URL+"/api/dialog/run", "application/json",
		strings.NewReader(`{"name":"memory-check"}`))
	if err != nil {
		t.Fatalf("POST /api/dialog/run: %v", err)
	}
	sc := bufio.NewScanner(run.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 4<<20)
	for sc.Scan() {
		var ev map[string]any
		if json.Unmarshal(sc.Bytes(), &ev) == nil {
			events = append(events, ev)
		}
	}
	run.Body.Close()

	last := events[len(events)-1]
	if last["type"] != "done" {
		t.Fatalf("last event = %v, want done (an empty answer must not break the run)", last)
	}
	turns := 0
	for _, ev := range events {
		if ev["type"] == "turn" {
			turns++
			if strings.TrimSpace(fmt.Sprint(ev["content"])) == "" {
				t.Fatalf("turn %v left without an answer", ev["index"])
			}
		}
	}
	if turns != 10 {
		t.Fatalf("turns = %d, want 10", turns)
	}

	// The stored history has real answers for every replica.
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
	if len(hist.Messages) != 20 {
		t.Fatalf("history len = %d, want 20", len(hist.Messages))
	}
	for _, m := range hist.Messages {
		if m.Role == "assistant" && strings.TrimSpace(m.Content) == "" {
			t.Fatalf("assistant message left empty: %+v", m)
		}
	}
}

// TestHistoryHidesToolCallNodes verifies that intermediate tool exchanges
// (the assistant node carrying tool_calls and the tool result) never reach
// /api/history: the model's answer shows up right after the user question,
// with no «пустой ответ модели» rows in between.
func TestHistoryHidesToolCallNodes(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		var req map[string]any
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)
		hasToolResult := false
		if msgs, ok := req["messages"].([]any); ok {
			for _, m := range msgs {
				if mm, ok := m.(map[string]any); ok && mm["role"] == "tool" {
					hasToolResult = true
				}
			}
		}
		if !hasToolResult {
			fmt.Fprint(w, `data: {"choices":[{"delta":{"role":"assistant"}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"task_note","arguments":`+
				`"{\"text\":\"заметка\"}"`+`}}]}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n")
		} else {
			fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"Финальный ответ"}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`+"\n\n")
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"вопрос"}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

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
	if len(hist.Messages) != 2 {
		for i, m := range hist.Messages {
			t.Logf("msg[%d] role=%q content=%q", i, m.Role, m.Content)
		}
		t.Fatalf("history len = %d, want 2 (question + answer, no tool-exchange rows)", len(hist.Messages))
	}
	if hist.Messages[0].Role != "user" || hist.Messages[1].Role != "assistant" {
		t.Fatalf("roles = [%s, %s], want [user, assistant]", hist.Messages[0].Role, hist.Messages[1].Role)
	}
	if hist.Messages[1].Content != "Финальный ответ" {
		t.Fatalf("answer = %q, want «Финальный ответ»", hist.Messages[1].Content)
	}
}

// sseCountingUpstream answers each chat completion with its own 1-based text
// ("Ответ 1", "Ответ 2", …) so branch tests can tell alternative replies
// apart.
func sseCountingUpstream() *httptest.Server {
	var mu sync.Mutex
	n := 0
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		n++
		cur := n
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, `data: {"choices":[{"delta":{"content":"Ответ %d"}}]}`+"\n\n", cur)
		fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`+"\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
}

// postJSON posts a JSON body and returns the response (caller closes it).
func postJSON(t *testing.T, url string, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	return resp
}

func chatHist(t *testing.T, url string) []historyMessage {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("GET /api/history: %v", err)
	}
	defer resp.Body.Close()
	var out struct {
		Messages []historyMessage `json:"messages"`
		Active   string           `json:"active"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode /api/history: %v", err)
	}
	return out.Messages
}

// TestBranchForkSwitchDelete walks the core branch scenario: chat, fork at
// the question, regenerate into an alternative answer, switch back to the
// original branch, delete it.
// TestChatRetriesEmptyAnswer verifies that a regular chat exchange retries
// an empty model answer with a doubled token budget (reasoning models can
// spend the whole limit on reasoning) instead of storing an empty reply.
func TestChatRetriesEmptyAnswer(t *testing.T) {
	upstream := sseEmptyOnceUpstream()
	defer upstream.Close()
	dbFile := filepath.Join(t.TempDir(), "chat.db")
	ts := newTestServerWithDB(t, upstream.URL, dbFile)

	resp, body := postBody(t, ts.URL+"/api/chat", `{"message":"привет"}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("chat status = %d, body = %q", resp.StatusCode, body)
	}
	if strings.TrimSpace(body) != "Ответ модели" {
		t.Fatalf("answer after retry = %q, want real content", body)
	}
	hist := chatHist(t, ts.URL+"/api/history")
	if len(hist) != 2 || strings.TrimSpace(hist[1].Content) == "" {
		t.Fatalf("hist = %+v, want user + non-empty assistant", hist)
	}
}

// TestBranchVersionsPillUnderEmptyAnswer verifies the «версии ответа» pill
// placement when the ACTIVE answer version is an empty assistant node
// (legacy history from before the empty-answer retry existed): the pill
// must still sit under the answer, not under the user's question, and the
// empty answer must not be invisible in the chat.
func TestBranchVersionsPillUnderEmptyAnswer(t *testing.T) {
	upstream := sseUpstream()
	defer upstream.Close()
	dbFile := filepath.Join(t.TempDir(), "chat.db")
	at := time.Now()
	// Seed the tree the way the pre-retry app left it: a question with two
	// answer versions, the active one EMPTY (a regen that returned nothing).
	// The user's .env has no system prompt, so the tree root is the question
	// itself (no system node) — same as the live app.
	st := storage.HistoryState{
		Model: "test-model",
		Messages: []storage.ChatRecord{
			{ID: "m1", Parent: "", Role: "user", Content: "вопрос", At: at},
			{ID: "m2", Parent: "m1", Role: "assistant", Content: "Ответ 1", At: at},
			{ID: "m3", Parent: "m1", Role: "assistant", Content: "", At: at},
		},
		Active: "m3",
		NextID: 4,
		// m2 was registered switchable when the regen created m3 (the app
		// registers the abandoned leaf at every regenerate/fork); switchable
		// leaves restore only with a branch state, like the live app saves.
		Switchable: []string{"m2"},
		Branches:   []storage.BranchState{{Leaf: "m2"}},
	}
	db, err := storage.Open(dbFile)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	if err := db.SaveSession(st); err != nil {
		t.Fatalf("save session: %v", err)
	}
	db.Close()

	client := api.New(upstream.URL, "test-key", 5*time.Second)
	cfg := config.Config{
		APIKey:  "test-key",
		BaseURL: upstream.URL,
		Model:   "test-model",
		System:  "", // как в .env пользователя
		Timeout: 5 * time.Second,
		DBFile:  dbFile,
	}
	srv := httptest.NewServer(New(client, cfg).Handler())
	t.Cleanup(srv.Close)
	hist := chatHist(t, srv.URL+"/api/history")
	if len(hist) != 2 {
		t.Fatalf("hist len = %d, want question + active (empty) answer", len(hist))
	}
	if hist[0].Content != "вопрос" || hist[0].Alts != nil {
		t.Fatalf("question row = %+v, want no alts on the question", hist[0])
	}
	if strings.TrimSpace(hist[1].Content) != "" {
		t.Fatalf("active row = %+v, want the EMPTY answer version", hist[1])
	}
	if len(hist[1].Alts) != 2 {
		t.Fatalf("alts = %+v, want 2 answer leaves on the empty answer row", hist[1].Alts)
	}
}

// TestBranchVersionsPillUnderAnswerWithForkStub verifies that answer
// versions (переспросить) keep their switcher under the active model answer
// even when the question also has an alternative-question branch (fork from
// the question + a typed follow-up): the alternative leaf stays in the
// switcher list, so nothing becomes unreachable, but the pill is not glued
// to the user's question.
func TestBranchVersionsPillUnderAnswerWithForkStub(t *testing.T) {
	upstream := sseCountingUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	// Обмен: вопрос → Ответ 1.
	resp := postJSON(t, ts.URL+"/api/chat", `{"message":"вопрос"}`)
	resp.Body.Close()

	hist := chatHist(t, ts.URL+"/api/history")
	questionID, answerID := hist[0].ID, hist[1].ID

	// Ветвление от вопроса и альтернативный вопрос там: вопрос → q2 → Ответ 3.
	resp = postJSON(t, ts.URL+"/api/branch", `{"after":"`+questionID+`"}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("fork status = %d", resp.StatusCode)
	}
	resp.Body.Close()
	resp = postJSON(t, ts.URL+"/api/chat", `{"message":"другой вопрос"}`)
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	// sseCountingUpstream нумерует ответы сквозным счётчиком запросов.
	if strings.TrimSpace(string(body)) != "Ответ 2" {
		t.Fatalf("fork-branch answer = %q", body)
	}

	// Вернуться на исходную ветку и переспросить: второй ответ на тот же вопрос.
	resp = postJSON(t, ts.URL+"/api/branch/switch", `{"id":"`+answerID+`"}`)
	resp.Body.Close()
	resp = postJSON(t, ts.URL+"/api/chat", `{"regenerate":true}`)
	body, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if strings.TrimSpace(string(body)) != "Ответ 3" {
		t.Fatalf("regenerated answer = %q", body)
	}

	hist = chatHist(t, ts.URL+"/api/history")
	if len(hist) != 2 || hist[1].Content != "Ответ 3" {
		t.Fatalf("after regenerate hist = %+v", hist)
	}
	// Пилюля — под активным ответом модели, не под вопросом.
	if hist[0].Alts != nil {
		t.Fatalf("question must have no alts, got %+v", hist[0].Alts)
	}
	if len(hist[1].Alts) != 3 {
		t.Fatalf("alts = %+v, want 3 leaves (old answer, fork branch, new answer)", hist[1].Alts)
	}
}

func TestBranchForkSwitchDelete(t *testing.T) {
	upstream := sseCountingUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	// Обмен: вопрос → Ответ 1.
	resp := postJSON(t, ts.URL+"/api/chat", `{"message":"вопрос"}`)
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if strings.TrimSpace(string(body)) != "Ответ 1" {
		t.Fatalf("first answer = %q", body)
	}

	hist := chatHist(t, ts.URL+"/api/history")
	if len(hist) != 2 || hist[1].Content != "Ответ 1" {
		t.Fatalf("hist = %+v", hist)
	}
	questionID := hist[0].ID
	answerID := hist[1].ID
	if questionID == "" || answerID == "" {
		t.Fatalf("no ids: %+v", hist)
	}
	// Линейный диалог: переключателей нет.
	if hist[0].Alts != nil {
		t.Fatalf("linear dialog must have no alts, got %+v", hist[0].Alts)
	}

	// Ветвление от вопроса: прежний ответ остаётся отдельной веткой.
	resp = postJSON(t, ts.URL+"/api/branch", `{"after":"`+questionID+`"}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("fork status = %d", resp.StatusCode)
	}
	resp.Body.Close()

	// Переспросить: альтернативный ответ на тот же вопрос.
	resp = postJSON(t, ts.URL+"/api/chat", `{"regenerate":true}`)
	body, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if strings.TrimSpace(string(body)) != "Ответ 2" {
		t.Fatalf("regenerated answer = %q", body)
	}

	hist = chatHist(t, ts.URL+"/api/history")
	if len(hist) != 2 || hist[1].Content != "Ответ 2" {
		t.Fatalf("after regenerate hist = %+v", hist)
	}
	// Пилюля версий ответа стоит под активным ответом модели, а не под вопросом.
	if hist[0].Alts != nil {
		t.Fatalf("question must have no alts, got %+v", hist[0].Alts)
	}
	if len(hist[1].Alts) != 2 || hist[1].Alts[0] != answerID {
		t.Fatalf("alts = %+v, want the old answer leaf first", hist[1].Alts)
	}

	// Список веток: старая и активная.
	_, branches, _ := getBranches(t, ts.URL)
	if len(branches) != 2 {
		t.Fatalf("branches = %+v", branches)
	}

	// Переключение на прежнюю ветку — тот же вопрос, старый ответ.
	resp = postJSON(t, ts.URL+"/api/branch/switch", `{"id":"`+answerID+`"}`)
	resp.Body.Close()
	hist = chatHist(t, ts.URL+"/api/history")
	if len(hist) != 2 || hist[1].Content != "Ответ 1" {
		t.Fatalf("after switch hist = %+v", hist)
	}
	if hist[0].Alts != nil || len(hist[1].Alts) != 2 {
		t.Fatalf("alts after switch = q:%+v a:%+v", hist[0].Alts, hist[1].Alts)
	}

	// Удалить активную ветку нельзя.
	resp = postJSON(t, ts.URL+"/api/branch/delete", `{"id":"`+answerID+`"}`) // wrong verb, must 405
	resp.Body.Close()

	del, err := http.NewRequest(http.MethodDelete, ts.URL+"/api/branch", strings.NewReader(`{"id":"`+answerID+`"}`))
	if err != nil {
		t.Fatalf("new delete: %v", err)
	}
	delResp, err := http.DefaultClient.Do(del)
	if err != nil {
		t.Fatalf("DELETE /api/branch: %v", err)
	}
	delResp.Body.Close()
	if delResp.StatusCode != http.StatusConflict {
		t.Fatalf("delete active branch status = %d, want 409", delResp.StatusCode)
	}

	// Переключиться на вторую ветку и удалить прежнюю.
	resp = postJSON(t, ts.URL+"/api/branch/switch", `{"id":"`+hist[1].Alts[1]+`"}`)
	resp.Body.Close()
	del, err = http.NewRequest(http.MethodDelete, ts.URL+"/api/branch", strings.NewReader(`{"id":"`+answerID+`"}`))
	if err != nil {
		t.Fatalf("new delete: %v", err)
	}
	delResp, err = http.DefaultClient.Do(del)
	if err != nil {
		t.Fatalf("DELETE /api/branch: %v", err)
	}
	delResp.Body.Close()
	if delResp.StatusCode != http.StatusOK {
		t.Fatalf("delete branch status = %d, want 200", delResp.StatusCode)
	}
	hist = chatHist(t, ts.URL+"/api/history")
	if len(hist[0].Alts) != 0 {
		t.Fatalf("alts after delete = %+v, want none", hist[0].Alts)
	}
}

func getBranches(t *testing.T, url string) (*http.Response, []map[string]any, error) {
	t.Helper()
	resp, err := http.Get(url + "/api/branches")
	if err != nil {
		t.Fatalf("GET /api/branches: %v", err)
	}
	var out struct {
		Branches []map[string]any `json:"branches"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode /api/branches: %v", err)
	}
	return resp, out.Branches, nil
}

// TestBranchForkContinuesDialog checks that a fork keeps the old continuation
// on its own branch and the new message extends the fork point.
func TestBranchForkContinuesDialog(t *testing.T) {
	upstream := sseCountingUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	postJSON(t, ts.URL+"/api/chat", `{"message":"первый"}`).Body.Close()
	hist := chatHist(t, ts.URL+"/api/history")
	questionID := hist[0].ID // m2

	// Ветвление от первого вопроса и продолжение новым сообщением.
	postJSON(t, ts.URL+"/api/branch", `{"after":"`+questionID+`"}`).Body.Close()
	postJSON(t, ts.URL+"/api/chat", `{"message":"второй"}`).Body.Close()

	hist = chatHist(t, ts.URL+"/api/history")
	if len(hist) != 3 {
		t.Fatalf("after fork+chat hist = %+v", hist)
	}
	// Первый вопрос ведёт к двум продолжениям: старый ответ и новая ветка.
	if len(hist[0].Alts) != 2 {
		t.Fatalf("alts at question = %+v, want 2", hist[0].Alts)
	}
	// Старый ответ — переключаемая ветка; на нём тоже есть выбор.
	oldAnswer := hist[0].Alts[0]
	postJSON(t, ts.URL+"/api/branch/switch", `{"id":"`+oldAnswer+`"}`).Body.Close()
	hist = chatHist(t, ts.URL+"/api/history")
	if len(hist) != 2 {
		t.Fatalf("old branch hist = %+v, want 2 messages", hist)
	}
}

// TestBranchPersistenceRestoresActive checks that branches and the active one
// survive a restart.
func TestBranchPersistenceRestoresActive(t *testing.T) {
	upstream := sseCountingUpstream()
	defer upstream.Close()
	histFile := filepath.Join(t.TempDir(), "history.json")

	newSrv := func() *httptest.Server {
		client := api.New(upstream.URL, "test-key", 5*time.Second)
		cfg := config.Config{
			APIKey:      "test-key",
			BaseURL:     upstream.URL,
			Model:       "test-model",
			System:      "be brief",
			Timeout:     5 * time.Second,
			HistoryFile: histFile,
		}
		srv := httptest.NewServer(New(client, cfg).Handler())
		t.Cleanup(srv.Close)
		return srv
	}

	ts := newSrv()
	postJSON(t, ts.URL+"/api/chat", `{"message":"вопрос"}`).Body.Close()
	hist := chatHist(t, ts.URL+"/api/history")
	questionID, firstAnswer := hist[0].ID, hist[1].ID
	postJSON(t, ts.URL+"/api/branch", `{"after":"`+questionID+`"}`).Body.Close()
	postJSON(t, ts.URL+"/api/chat", `{"regenerate":true}`).Body.Close()
	ts.Close()

	// После перезапуска активна последняя ветка (Ответ 2).
	ts = newSrv()
	hist = chatHist(t, ts.URL+"/api/history")
	if len(hist) != 2 || !strings.Contains(hist[1].Content, "Ответ 2") {
		t.Fatalf("restored hist = %+v", hist)
	}
	// Пилюля версий ответа — под активным ответом, не под вопросом.
	if hist[0].Alts != nil || len(hist[1].Alts) != 2 || hist[1].Alts[0] != firstAnswer {
		t.Fatalf("restored alts = q:%+v a:%+v", hist[0].Alts, hist[1].Alts)
	}
	// Переключение на старую ветку работает после перезапуска.
	postJSON(t, ts.URL+"/api/branch/switch", `{"id":"`+firstAnswer+`"}`).Body.Close()
	hist = chatHist(t, ts.URL+"/api/history")
	if len(hist) != 2 || !strings.Contains(hist[1].Content, "Ответ 1") {
		t.Fatalf("switched hist = %+v", hist)
	}
}

// TestBranchDuringGenerationConflicts checks the 409 guard: branch operations
// refuse to run while the model is generating (genMu busy).
func TestBranchDuringGenerationConflicts(t *testing.T) {
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		<-release
		fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"поздно"}}]}`+"\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	done := make(chan *http.Response, 1)
	go func() { done <- postJSON(t, ts.URL+"/api/chat", `{"message":"вопрос"}`) }()

	// Дождаться, пока генерация реально начнётся (сервер держит genMu).
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		resp := postJSON(t, ts.URL+"/api/branch", `{"after":"m2"}`)
		resp.Body.Close()
		if resp.StatusCode == http.StatusConflict {
			close(release)
			chat := <-done
			chat.Body.Close()
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	close(release)
	chat := <-done
	chat.Body.Close()
	t.Fatalf("branch op never got 409 during generation")
}

// TestBranchNoPhantomAfterSwitchAway checks that a leaf left via a switch and
// later continued does not linger in the switcher as an empty phantom branch:
// a message sent in that branch extends it (and the branch list stays at two)
// instead of creating yet another branch.
func TestBranchNoPhantomAfterSwitchAway(t *testing.T) {
	upstream := sseCountingUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	postJSON(t, ts.URL+"/api/chat", `{"message":"вопрос"}`).Body.Close()
	hist := chatHist(t, ts.URL+"/api/history")
	questionID, answerID := hist[0].ID, hist[1].ID

	// Ветвление от вопроса (остаёмся на вопросе), затем переход на старую
	// ветку и обратно: лист-вопрос попадает в переключатель.
	postJSON(t, ts.URL+"/api/branch", `{"after":"`+questionID+`"}`).Body.Close()
	postJSON(t, ts.URL+"/api/branch/switch", `{"id":"`+answerID+`"}`).Body.Close()
	postJSON(t, ts.URL+"/api/branch/switch", `{"id":"`+questionID+`"}`).Body.Close()

	// Сообщение в новой ветке продолжает её: веток остаётся две, а вопрос
	// (у которого появился потомок) из переключателя исчезает.
	postJSON(t, ts.URL+"/api/chat", `{"message":"второй"}`).Body.Close()

	_, branches, _ := getBranches(t, ts.URL)
	if len(branches) != 2 {
		t.Fatalf("branches = %+v, want 2 (no phantom leaf)", branches)
	}
	hist = chatHist(t, ts.URL+"/api/history")
	if len(hist) != 3 {
		t.Fatalf("hist = %+v", hist)
	}
	// Один переключатель — под вопросом, от которого ответвились.
	for i, m := range hist {
		want := i == 0
		if want != (m.Alts != nil) {
			t.Fatalf("msg %d alts = %+v, want the switcher only under the fork point", i, m.Alts)
		}
	}
	// В старой ветке переключатель тоже один — под тем же вопросом.
	postJSON(t, ts.URL+"/api/branch/switch", `{"id":"`+answerID+`"}`).Body.Close()
	hist = chatHist(t, ts.URL+"/api/history")
	if len(hist) != 2 || hist[1].Content != "Ответ 1" {
		t.Fatalf("old branch hist = %+v", hist)
	}
	if len(hist[0].Alts) != 2 {
		t.Fatalf("old branch alts = %+v, want 2 under the question", hist[0].Alts)
	}
	if hist[1].Alts != nil {
		t.Fatalf("old answer alts = %+v, want none", hist[1].Alts)
	}
}

// TestBranchCompressionKeepsOtherBranches checks that compressing context in
// one branch does not touch the other branches: the old branch must keep the
// exact same materialized history it had before the fork-branch compression.
func TestBranchCompressionKeepsOtherBranches(t *testing.T) {
	upstream := sseCountingUpstream()
	defer upstream.Close()
	client := api.New(upstream.URL, "test-key", 5*time.Second)
	cfg := config.Config{
		APIKey:  "test-key",
		BaseURL: upstream.URL,
		Model:   "test-model",
		System:  "be brief",
		Timeout: 5 * time.Second,
	}
	srvImpl := New(client, cfg)
	ts := httptest.NewServer(srvImpl.Handler())
	t.Cleanup(ts.Close)

	// Три обмена со сжатием (keep=3, threshold=5): после третьего обмена
	// в линейном диалоге 6 ходов — срабатывает сжатие (сводка + 3 хода).
	// Тело ответа дочитываем до конца: сжатие выполняется в хендлере после
	// последнего токена, до закрытия потока.
	for i := 1; i <= 3; i++ {
		resp := postJSON(t, ts.URL+"/api/chat",
			`{"message":"вопрос `+fmt.Sprint(i)+`","compress":{"enabled":true,"strategy":"summary","keep":3,"threshold":5}}`)
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
	before := chatHist(t, ts.URL+"/api/history")
	if len(before) != 4 || before[0].Role != "summary" {
		t.Fatalf("linear hist = %+v, want summary + 3 turns", before)
	}

	// Форк от вопроса третьего обмена; в новой ветке — ещё три обмена,
	// после чего в ней срабатывает сжатие.
	question3 := before[2].ID
	resp := postJSON(t, ts.URL+"/api/branch", `{"after":"`+question3+`"}`)
	resp.Body.Close()
	for i := 4; i <= 6; i++ {
		resp = postJSON(t, ts.URL+"/api/chat",
			`{"message":"вопрос `+fmt.Sprint(i)+`","compress":{"enabled":true,"strategy":"summary","keep":3,"threshold":5}}`)
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
	branch := chatHist(t, ts.URL+"/api/history")
	// Сжатие в ветке сворачивает только её собственные сообщения: общий
	// префикс (унаследованная сводка + её ходы) и точка ветвления остаются
	// видимыми, далее — своя сводка ветки и её последние ходы.
	if len(branch) != 7 || branch[0].Role != "summary" || branch[3].Role != "summary" {
		t.Fatalf("branch hist after compression = %+v, want prefix summary + fork point + own summary + 3 turns", branch)
	}
	if branch[2].ID != question3 || len(branch[2].Alts) < 2 {
		t.Fatalf("fork point must stay visible with its pill: %+v", branch[2])
	}

	// Возврат в прежнюю ветку: её история должна совпасть с той, что была
	// до сжатия в новой ветке (своя сводка + свои три реплики). Пилюли (alts)
	// не сравниваем: после форка у вопроса ветвления закономерно появляются.
	// Время у служебной сводки не сравниваем: она пересоздаётся при каждом
	// рендере истории.
	strip := func(h []historyMessage) []historyMessage {
		out := make([]historyMessage, len(h))
		for i, m := range h {
			m.Alts = nil
			if m.ID == "" {
				m.At = time.Time{}
			}
			out[i] = m
		}
		return out
	}
	postJSON(t, ts.URL+"/api/branch/switch", `{"id":"`+before[3].ID+`"}`).Body.Close()
	after := chatHist(t, ts.URL+"/api/history")
	if !reflect.DeepEqual(strip(before), strip(after)) {
		t.Fatalf("old branch changed after branch compression:\nbefore = %+v\nafter  = %+v", before, after)
	}
}

// TestBranchForkAfterCompressionShowsTail проверяет «Кейс 1»: сжатие в
// линейном диалоге, затем форк от последнего сообщения пользователя —
// в новой ветке должны быть видны сводка и всё, что осталось после сжатия
// (включая само сообщение форка).
func TestBranchForkAfterCompressionShowsTail(t *testing.T) {
	upstream := sseCountingUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	// Три обмена: сжатие сворачивает первые ходы (сводка + 3 хода).
	for i := 1; i <= 3; i++ {
		resp := postJSON(t, ts.URL+"/api/chat",
			`{"message":"вопрос `+fmt.Sprint(i)+`","compress":{"enabled":true,"strategy":"summary","keep":3,"threshold":5}}`)
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
	before := chatHist(t, ts.URL+"/api/history")
	if len(before) != 4 || before[0].Role != "summary" {
		t.Fatalf("linear hist = %+v, want summary + 3 turns", before)
	}

	// Форк от последнего сообщения пользователя (вопрос 3, в «хвосте»).
	lastUser := before[2].ID
	resp := postJSON(t, ts.URL+"/api/branch", `{"after":"`+lastUser+`"}`)
	resp.Body.Close()

	// Один обмен в новой ветке: её история должна показать сводку,
	// хвост линейной ветки и новую пару вопрос-ответ.
	resp = postJSON(t, ts.URL+"/api/chat",
		`{"message":"вопрос 4","compress":{"enabled":true,"strategy":"summary","keep":3,"threshold":5}}`)
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	branch := chatHist(t, ts.URL+"/api/history")
	if len(branch) != 5 || branch[0].Role != "summary" {
		t.Fatalf("branch hist = %+v, want summary + tail + вопрос 4 + ответ", branch)
	}
	if branch[1].ID == "" || branch[1].Role != "assistant" {
		t.Fatalf("branch hist misses the kept tail: %+v", branch[1])
	}
	foundFork, foundNew := false, false
	for _, m := range branch {
		if m.ID == lastUser && m.Content == "вопрос 3" {
			foundFork = true
		}
		if m.Content == "вопрос 4" {
			foundNew = true
		}
	}
	if !foundFork || !foundNew {
		t.Fatalf("branch hist misses fork/new message: %+v", branch)
	}

	// Ещё три обмена в ветке: сжатие срабатывает уже в ней — своя сводка
	// и свой хвост, прежняя ветка не меняется.
	for i := 5; i <= 7; i++ {
		resp = postJSON(t, ts.URL+"/api/chat",
			`{"message":"вопрос `+fmt.Sprint(i)+`","compress":{"enabled":true,"strategy":"summary","keep":3,"threshold":5}}`)
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
	branch = chatHist(t, ts.URL+"/api/history")
	// Сжатие в ветке сворачивает только её собственные сообщения:
	// унаследованная сводка, общий хвост и точка ветвления остаются
	// видимыми, затем — своя сводка и свой хвост.
	if len(branch) != 7 || branch[0].Role != "summary" || branch[3].Role != "summary" {
		t.Fatalf("branch hist after own compression = %+v, want prefix summary + fork point + own summary + 3 turns", branch)
	}
	// Точка ветвления остаётся видимой и несёт свой переключатель.
	if branch[2].ID != lastUser || len(branch[2].Alts) < 2 {
		t.Fatalf("branching point must stay visible with its pill, hist = %+v", branch)
	}

	// Переключение между ветками: у прежней — своя сводка и свой хвост,
	// в новой ветке префикс, точка ветвления и своя сводка сохраняются.
	oldLeaf := before[3].ID
	postJSON(t, ts.URL+"/api/branch/switch", `{"id":"`+oldLeaf+`"}`).Body.Close()
	old := chatHist(t, ts.URL+"/api/history")
	if len(old) != 4 || old[0].Role != "summary" {
		t.Fatalf("old branch hist = %+v, want summary + 3 turns", old)
	}
	postJSON(t, ts.URL+"/api/branch/switch", `{"id":"`+branch[len(branch)-1].ID+`"}`).Body.Close()
	back := chatHist(t, ts.URL+"/api/history")
	if len(back) != 7 || back[0].Role != "summary" || back[3].Role != "summary" {
		t.Fatalf("branch hist after switching back = %+v, want prefix summary + fork point + own summary + 3 turns", back)
	}
}

// TestEmptyForkBranchStaysSwitchable проверяет: пустая форк-ветка («Ветвление
// отсюда» без новых сообщений) остаётся в переключателе и после ухода с неё —
// пилюля под точкой ветвления не исчезает («дерево не стирается»).
func TestEmptyForkBranchStaysSwitchable(t *testing.T) {
	upstream := sseCountingUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	for i := 1; i <= 2; i++ {
		resp := postJSON(t, ts.URL+"/api/chat", `{"message":"вопрос `+fmt.Sprint(i)+`"}`)
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
	hist := chatHist(t, ts.URL+"/api/history")
	lastUser, oldLeaf := "", ""
	for _, m := range hist {
		if m.Role == "user" && m.ID != "" {
			lastUser = m.ID
		}
		if m.Role == "assistant" && m.ID != "" {
			oldLeaf = m.ID
		}
	}
	if lastUser == "" || oldLeaf == "" {
		t.Fatalf("bad hist: %+v", hist)
	}

	// Форк от последнего вопроса: активной становится пустая ветка.
	postJSON(t, ts.URL+"/api/branch", `{"after":"`+lastUser+`"}`).Body.Close()

	// Пилюля обязана быть уже в пустой форк-ветке: через точку проходят
	// два листа — сам вопрос (пустое продолжение) и прежняя ветка.
	hist = chatHist(t, ts.URL+"/api/history")
	for _, m := range hist {
		if m.ID == lastUser && len(m.Alts) < 2 {
			t.Fatalf("empty-fork pill missing in fork branch, alts=%v, hist=%+v", m.Alts, hist)
		}
	}

	// Уход на прежнюю ветку (последний ответ модели).
	postJSON(t, ts.URL+"/api/branch/switch", `{"id":"`+oldLeaf+`"}`).Body.Close()

	// Пилюля под вопросом обязана остаться: через точку проходят два листа —
	// пустая форк-ветка (сам вопрос) и прежняя ветка с ответом.
	hist = chatHist(t, ts.URL+"/api/history")
	found := false
	for _, m := range hist {
		if m.ID == lastUser {
			found = true
			if len(m.Alts) < 2 {
				t.Fatalf("empty-fork pill lost after switch, alts=%v, hist=%+v", m.Alts, hist)
			}
		}
	}
	if !found {
		t.Fatalf("fork question not in history: %+v", hist)
	}

	// Переключение обратно в пустую форк-ветку работает.
	postJSON(t, ts.URL+"/api/branch/switch", `{"id":"`+lastUser+`"}`).Body.Close()
	hist = chatHist(t, ts.URL+"/api/history")
	if n := len(hist); n == 0 || hist[n-1].ID != lastUser {
		t.Fatalf("switch to empty fork branch failed: %+v", hist)
	}
}

// TestDialogRunKeepsBranches проверяет: прогон пресета продолжает активную
// ветку и не стирает дерево — прежняя ветка остаётся доступной через пилюлю,
// а сжатие по ходу прогона происходит только в текущей ветке.
func TestDialogRunKeepsBranches(t *testing.T) {
	upstream := sseCountingUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	// Один обмен, форк от вопроса — активной становится пустая ветка.
	resp := postJSON(t, ts.URL+"/api/chat", `{"message":"вопрос 1"}`)
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	hist := chatHist(t, ts.URL+"/api/history")
	lastUser, oldLeaf := "", ""
	for _, m := range hist {
		if m.Role == "user" && m.ID != "" {
			lastUser = m.ID
		}
		if m.Role == "assistant" && m.ID != "" {
			oldLeaf = m.ID
		}
	}
	if lastUser == "" || oldLeaf == "" {
		t.Fatalf("bad hist: %+v", hist)
	}
	postJSON(t, ts.URL+"/api/branch", `{"after":"`+lastUser+`"}`).Body.Close()

	// Прогон пресета — в новой ветке.
	run, err := http.Post(ts.URL+"/api/dialog/run", "application/json",
		strings.NewReader(`{"name":"memory-check"}`))
	if err != nil {
		t.Fatalf("POST /api/dialog/run: %v", err)
	}
	sc := bufio.NewScanner(run.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 4<<20)
	for sc.Scan() {
	}
	run.Body.Close()

	// Прежняя ветка жива: переключение работает, её история цела,
	// пилюля под точкой ветвления на месте.
	postJSON(t, ts.URL+"/api/branch/switch", `{"id":"`+oldLeaf+`"}`).Body.Close()
	hist = chatHist(t, ts.URL+"/api/history")
	if len(hist) != 2 || hist[0].ID != lastUser {
		t.Fatalf("old branch hist = %+v, want [вопрос 1, ответ]", hist)
	}
	if len(hist[0].Alts) < 2 {
		t.Fatalf("branching pill lost after replay, alts=%v", hist[0].Alts)
	}

	// Обратное переключение в прогонную ветку тоже работает.
	hist2 := chatHist(t, ts.URL+"/api/history")
	var branchLeaf string
	for _, m := range hist2 {
		if m.Role == "assistant" && m.ID != "" && m.ID != oldLeaf {
			branchLeaf = m.ID
		}
	}
	postJSON(t, ts.URL+"/api/branch/switch", `{"id":"`+branchLeaf+`"}`).Body.Close()
}

// TestNestedForkKeepsPrefixVisible проверяет вложенный форк: ветка B, растущая
// из ветки A, наследует замороженные стадии префикса A (сводку линейного
// диалога и собственную сводку A); собственное сжатие B сворачивает только
// сообщения после точки ветвления B — префикс, общий хвост и обе точки
// ветвления остаются видимыми.
func TestNestedForkKeepsPrefixVisible(t *testing.T) {
	upstream := sseCountingUpstream()
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)
	compress := `{"message":"вопрос ` + "%s" + `","compress":{"enabled":true,"strategy":"summary","keep":3,"threshold":5}}`
	exchange := func(i int) {
		resp := postJSON(t, ts.URL+"/api/chat", fmt.Sprintf(compress, i))
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}

	// Линейный диалог: сводка + 3 хода.
	for i := 1; i <= 3; i++ {
		exchange(i)
	}
	before := chatHist(t, ts.URL+"/api/history")
	if len(before) != 4 || before[0].Role != "summary" {
		t.Fatalf("linear hist = %+v, want summary + 3 turns", before)
	}

	// Ветка A от вопроса 3: три обмена, своё сжатие.
	forkA := before[2].ID
	postJSON(t, ts.URL+"/api/branch", `{"after":"`+forkA+`"}`).Body.Close()
	for i := 4; i <= 6; i++ {
		exchange(i)
	}
	a := chatHist(t, ts.URL+"/api/history")
	if len(a) != 7 || a[0].Role != "summary" || a[3].Role != "summary" {
		t.Fatalf("branch A hist = %+v, want prefix summary + fork + own summary + 3 turns", a)
	}
	forkB := a[5].ID // вопрос 6 — точка вложенного форка
	if forkB == "" {
		t.Fatalf("branch A hist has no user message: %+v", a)
	}

	// Ветка B от вопроса 6: наследует обе замороженные стадии A.
	postJSON(t, ts.URL+"/api/branch", `{"after":"`+forkB+`"}`).Body.Close()
	exchange(7)
	b := chatHist(t, ts.URL+"/api/history")
	// [сводка S1, сводка A, ход A, вопрос 6 (форк), вопрос 7, ответ]
	if len(b) != 6 || b[0].Role != "summary" || b[1].Role != "summary" || b[2].Role != "assistant" {
		t.Fatalf("branch B hist = %+v, want two inherited summaries + A tail + fork point + exchange 7", b)
	}
	if b[3].ID != forkB || len(b[3].Alts) < 2 {
		t.Fatalf("nested fork point must stay visible with its pill: %+v", b[3])
	}

	// Собственное сжатие B: сворачивает только ходы после вопроса 6.
	for i := 8; i <= 9; i++ {
		exchange(i)
	}
	b = chatHist(t, ts.URL+"/api/history")
	// [S1, сводка A, ход A, вопрос 6, своя сводка B, ответ 8, вопрос 9, ответ 9]
	if len(b) != 8 || b[0].Role != "summary" || b[1].Role != "summary" || b[4].Role != "summary" {
		t.Fatalf("branch B hist after own compression = %+v, want inherited summaries + fork point + own summary + 3 turns", b)
	}
	if b[3].ID != forkB || len(b[3].Alts) < 2 {
		t.Fatalf("nested fork point must survive B's own compression: %+v", b[3])
	}

	// Ветка A не изменилась сжатием B: сводка префикса, ходы, своя сводка.
	postJSON(t, ts.URL+"/api/branch/switch", `{"id":"`+a[6].ID+`"}`).Body.Close()
	a2 := chatHist(t, ts.URL+"/api/history")
	if len(a2) != 7 || a2[0].Role != "summary" || a2[3].Role != "summary" {
		t.Fatalf("branch A hist after B compression = %+v, want prefix summary + fork + own summary + 3 turns", a2)
	}
}

// TestTaskNoteUpdatesTaskField verifies that a task_note whose text formulates
// the task itself («Задача: …») rewrites the working-memory task field
// instead of appending another «Задача» line to the notes list.
func TestTaskNoteUpdatesTaskField(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		var req map[string]any
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)
		hasToolResult := false
		if msgs, ok := req["messages"].([]any); ok {
			for _, m := range msgs {
				if mm, ok := m.(map[string]any); ok && mm["role"] == "tool" {
					hasToolResult = true
				}
			}
		}
		if !hasToolResult {
			// The model calls task_note with a task-formulating note.
			fmt.Fprint(w, `data: {"choices":[{"delta":{"role":"assistant"}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"task_note","arguments":`+
				`"{\"text\":\"Задача: подобрать стеллажи для библиотеки\"}"`+`}}]}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n")
		} else {
			fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"Записал задачу"}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`+"\n\n")
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(`{"message":"мы подбираем стеллажи"}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	get, err := http.Get(ts.URL + "/api/memory")
	if err != nil {
		t.Fatalf("GET /api/memory: %v", err)
	}
	defer get.Body.Close()
	var mem struct {
		Working struct {
			Task  string `json:"task"`
			Notes string `json:"notes"`
		} `json:"working"`
	}
	if err := json.NewDecoder(get.Body).Decode(&mem); err != nil {
		t.Fatalf("decode /api/memory: %v", err)
	}
	if mem.Working.Task != "подобрать стеллажи для библиотеки" {
		t.Fatalf("working.task = %q, want «подобрать стеллажи для библиотеки»", mem.Working.Task)
	}
	if strings.Contains(mem.Working.Notes, "Задача:") {
		t.Fatalf("working.notes = %q, want no duplicate «Задача» lines", mem.Working.Notes)
	}
}

// TestSummaryExtractsFacts verifies that the summary compression also fills
// the facts block: «ФАКТ: …» lines of the summary response go to the
// working-memory facts and are stripped from the displayed summary text.
func TestSummaryExtractsFacts(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		body, _ := io.ReadAll(r.Body)
		if strings.Contains(string(body), "сжать историю диалога") {
			for _, part := range []string{
				"Сводка переписки.\n",
				"ФАКТ: стеллажи: 8 по 1,5 м\n",
				"ЗАДАЧА: подобрать стеллажи",
			} {
				fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", part)
			}
		} else {
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Привет, мир\"}}]}\n\n")
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	ts := newTestServer(t, upstream.URL)

	loadLongContext(t, ts)

	resp, err := http.Post(ts.URL+"/api/chat", "application/json", strings.NewReader(
		`{"message":"привет","compress":{"enabled":true,"strategy":"summary","keep":6}}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	get, err := http.Get(ts.URL + "/api/memory")
	if err != nil {
		t.Fatalf("GET /api/memory: %v", err)
	}
	defer get.Body.Close()
	var mem struct {
		Working struct {
			Task  string `json:"task"`
			Facts string `json:"facts"`
		} `json:"working"`
	}
	if err := json.NewDecoder(get.Body).Decode(&mem); err != nil {
		t.Fatalf("decode /api/memory: %v", err)
	}
	if mem.Working.Task != "подобрать стеллажи" {
		t.Fatalf("working.task = %q, want «подобрать стеллажи»", mem.Working.Task)
	}
	if !strings.Contains(mem.Working.Facts, "стеллажи: 8 по 1,5 м") {
		t.Fatalf("working.facts = %q, want the extracted fact line", mem.Working.Facts)
	}

	get, err = http.Get(ts.URL + "/api/history")
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
	if len(hist.Messages) == 0 || hist.Messages[0].Role != "summary" {
		t.Fatalf("messages[0] = %+v, want the summary marker", hist.Messages)
	}
	if strings.Contains(hist.Messages[0].Content, "ФАКТ") || strings.Contains(hist.Messages[0].Content, "ЗАДАЧА") {
		t.Fatalf("summary text = %q, want fact and task lines stripped", hist.Messages[0].Content)
	}
}
