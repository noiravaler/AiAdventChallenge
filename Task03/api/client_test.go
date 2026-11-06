package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestStreamChat(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("unexpected Authorization header: %q", got)
		}

		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if req["stream"] != true {
			t.Errorf("expected stream=true, got %v", req["stream"])
		}
		if req["model"] != "test-model" {
			t.Errorf("unexpected model: %v", req["model"])
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Error("response writer does not support flushing")
			return
		}

		for _, chunk := range []string{
			`{"choices":[{"delta":{"role":"assistant"}}]}`,
			`{"choices":[{"delta":{"content":"Hel"}}]}`,
			`{"choices":[{"delta":{"content":"lo"}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		} {
			fmt.Fprintf(w, "data: %s\n\n", chunk)
			flusher.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	client := New(srv.URL, "test-key", 5*time.Second)

	var deltas strings.Builder
	answer, err := client.StreamChat(
		context.Background(),
		"test-model",
		[]Message{UserMessage("hi")},
		0,
		func(s string) { deltas.WriteString(s) },
	)
	if err != nil {
		t.Fatalf("StreamChat: %v", err)
	}
	if answer != "Hello" {
		t.Errorf("answer = %q, want %q", answer, "Hello")
	}
	if deltas.String() != "Hello" {
		t.Errorf("deltas = %q, want %q", deltas.String(), "Hello")
	}
}

func TestChat(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"Hello there"}}]}`)
	}))
	defer srv.Close()

	client := New(srv.URL, "test-key", 5*time.Second)
	answer, err := client.Chat(context.Background(), "test-model", []Message{UserMessage("hi")}, 100)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if answer != "Hello there" {
		t.Errorf("answer = %q, want %q", answer, "Hello there")
	}
}

func TestAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"error":{"message":"invalid api key","type":"auth_error"}}`)
	}))
	defer srv.Close()

	client := New(srv.URL, "bad-key", 5*time.Second)
	_, err := client.Chat(context.Background(), "test-model", []Message{UserMessage("hi")}, 0)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "invalid api key") {
		t.Errorf("error %q does not contain the API message", err)
	}
	if !strings.Contains(err.Error(), "401") {
		t.Errorf("error %q does not contain the status code", err)
	}
}

func TestEmptyChoices(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"choices":[]}`)
	}))
	defer srv.Close()

	client := New(srv.URL, "test-key", 5*time.Second)
	if _, err := client.Chat(context.Background(), "test-model", nil, 0); err == nil {
		t.Fatal("expected an error for an empty choices list")
	}
}

func TestDebugLog(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Привет\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	var logBuf bytes.Buffer
	client := New(srv.URL, "test-key", 5*time.Second)
	client.DebugLog = &logBuf

	if _, err := client.StreamChat(context.Background(), "test-model", []Message{UserMessage("привет")}, 0, nil); err != nil {
		t.Fatalf("StreamChat: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(logBuf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("got %d log lines, want 2:\n%s", len(lines), logBuf.String())
	}

	var req struct {
		Type    string `json:"type"`
		URL     string `json:"url"`
		Payload struct {
			Model    string    `json:"model"`
			Messages []Message `json:"messages"`
			Stream   bool      `json:"stream"`
		} `json:"payload"`
	}
	if err := json.Unmarshal([]byte(lines[0]), &req); err != nil {
		t.Fatalf("request line is not valid JSON: %v\nline: %s", err, lines[0])
	}
	if req.Type != "request" || req.URL != srv.URL+"/chat/completions" {
		t.Fatalf("unexpected request event: %+v", req)
	}
	if req.Payload.Model != "test-model" || !req.Payload.Stream || len(req.Payload.Messages) != 1 {
		t.Fatalf("unexpected request payload: %+v", req.Payload)
	}

	var resp struct {
		Type    string `json:"type"`
		Content string `json:"content"`
		Error   string `json:"error"`
	}
	if err := json.Unmarshal([]byte(lines[1]), &resp); err != nil {
		t.Fatalf("response line is not valid JSON: %v\nline: %s", err, lines[1])
	}
	if resp.Type != "response" || resp.Content != "Привет" || resp.Error != "" {
		t.Fatalf("unexpected response event: %+v", resp)
	}

	// Failed requests are logged with the error instead of the content.
	errSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"bad key"}}`, http.StatusUnauthorized)
	}))
	defer errSrv.Close()

	var errLog bytes.Buffer
	client2 := New(errSrv.URL, "bad-key", 5*time.Second)
	client2.DebugLog = &errLog
	if _, err := client2.Chat(context.Background(), "test-model", []Message{UserMessage("hi")}, 0); err == nil {
		t.Fatal("expected an error")
	}

	var respErr struct {
		Type  string `json:"type"`
		Error string `json:"error"`
	}
	errLines := strings.Split(strings.TrimSpace(errLog.String()), "\n")
	if len(errLines) != 2 {
		t.Fatalf("got %d error-log lines, want 2:\n%s", len(errLines), errLog.String())
	}
	if err := json.Unmarshal([]byte(errLines[1]), &respErr); err != nil {
		t.Fatalf("error line is not valid JSON: %v\nline: %s", err, errLines[1])
	}
	if respErr.Type != "response" || respErr.Error == "" {
		t.Fatalf("unexpected error event: %+v", respErr)
	}
}

func TestListModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("unexpected Authorization header: %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"object":"list","data":[{"id":"default-chat"},{"id":"glm-5.3-flash"}]}`)
	}))
	defer srv.Close()

	client := New(srv.URL, "test-key", 5*time.Second)
	ids, err := client.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(ids) != 2 || ids[0] != "default-chat" || ids[1] != "glm-5.3-flash" {
		t.Fatalf("ids = %v, want [default-chat glm-5.3-flash]", ids)
	}
}

func TestListModelsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"nope"}}`, http.StatusUnauthorized)
	}))
	defer srv.Close()

	client := New(srv.URL, "bad-key", 5*time.Second)
	if _, err := client.ListModels(context.Background()); err == nil {
		t.Fatal("expected an error for a non-200 /models response")
	}
}
