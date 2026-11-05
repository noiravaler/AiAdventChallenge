package web

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
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
