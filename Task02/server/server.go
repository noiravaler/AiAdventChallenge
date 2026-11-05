// Package server serves the browser chat interface.
package server

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"ai-chat-go/api"
	"ai-chat-go/config"
)

//go:embed static/index.html
var indexHTML []byte

// defaultNoReplyTimeout aborts a chat request whose model produces no output
// within this duration, and logs the error.
const defaultNoReplyTimeout = 7 * time.Second

// Server keeps the conversation state for the browser chat.
type Server struct {
	client *api.Client
	cfg    config.Config

	genMu sync.Mutex // serializes chat generations
	mu    sync.Mutex // guards history and model selection
	hist  []api.Message

	model          string        // currently selected model
	models         []string      // models offered in the contact sidebar
	noReplyTimeout time.Duration // abort when the model stays silent this long

	lastDetail string // compact JSON of the last error/limit detail (Детали panel)
}

// New creates a web chat server with the system prompt from cfg.
func New(client *api.Client, cfg config.Config) *Server {
	s := &Server{client: client, cfg: cfg, noReplyTimeout: defaultNoReplyTimeout}
	s.models = append([]string(nil), cfg.Models...)
	if len(s.models) == 0 {
		s.models = append([]string(nil), config.DefaultModels...)
	}
	s.model = cfg.Model
	if s.model == "" && len(s.models) > 0 {
		s.model = s.models[0]
	}
	s.hist = make([]api.Message, 0, 16)
	if cfg.System != "" {
		s.hist = append(s.hist, api.SystemMessage(cfg.System))
	}
	return s
}

// Handler returns the HTTP routes of the web chat.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", s.handleIndex)
	mux.HandleFunc("GET /api/info", s.handleInfo)
	mux.HandleFunc("GET /api/models", s.handleModels)
	mux.HandleFunc("GET /api/history", s.handleHistory)
	mux.HandleFunc("POST /api/chat", s.handleChat)
	mux.HandleFunc("POST /api/model", s.handleModel)
	mux.HandleFunc("POST /api/reset", s.handleReset)
	mux.HandleFunc("GET /api/debug", s.handleDebug)
	return mux
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(indexHTML)
}

func (s *Server) handleInfo(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]any{
		"model":         s.currentModel(),
		"default_model": s.cfg.Model,
		"baseURL":       s.cfg.BaseURL,
		"max_tokens":    s.defaultMaxTokens(),
	})
}

// defaultMaxTokens is the max_tokens value offered to the client before a user
// changes it with the slider, clamped to the 1..15000 UI range.
func (s *Server) defaultMaxTokens() int {
	v := s.cfg.MaxTokens
	if v <= 0 {
		v = 1024
	}
	if v > 15000 {
		v = 15000
	}
	return v
}

// normalizeFormat maps the browser's format selector to a value the API client
// understands: "json" requests a JSON object, "jsonschema" and
// "jsonschema_structured" request a JSON Schema, anything else plain text.
func normalizeFormat(f string) string {
	switch strings.ToLower(strings.TrimSpace(f)) {
	case "json":
		return "json"
	case "jsonschema", "json_schema":
		return "jsonschema"
	case "jsonschema_structured", "structured":
		return "jsonschema_structured"
	default:
		return "text"
	}
}

// currentModel returns the currently selected model.
func (s *Server) currentModel() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.model
}

// handleModels lists the models available in the contact sidebar.
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	models := append([]string(nil), s.models...)
	current := s.model
	s.mu.Unlock()
	writeJSON(w, map[string]any{"models": models, "current": current})
}

// handleModel switches the active model and opens a fresh session.
func (s *Server) handleModel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	req.Model = strings.TrimSpace(req.Model)
	if req.Model == "" {
		http.Error(w, "model is empty", http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	valid := false
	for _, m := range s.models {
		if m == req.Model {
			valid = true
			break
		}
	}
	if !valid {
		s.mu.Unlock()
		http.Error(w, "unknown model", http.StatusBadRequest)
		return
	}
	s.model = req.Model
	s.hist = make([]api.Message, 0, 16)
	if s.cfg.System != "" {
		s.hist = append(s.hist, api.SystemMessage(s.cfg.System))
	}
	s.mu.Unlock()

	log.Printf("chat: switched to model %s (new session)", req.Model)
	writeJSON(w, map[string]bool{"ok": true})
}

type historyMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (s *Server) handleHistory(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	msgs := make([]historyMessage, 0, len(s.hist))
	for _, m := range s.hist {
		if m.Role == "system" {
			continue
		}
		msgs = append(msgs, historyMessage{Role: m.Role, Content: m.Content})
	}
	s.mu.Unlock()
	writeJSON(w, map[string]any{"messages": msgs})
}

// handleDebug returns the JSON of the last API request and response for the
// "Детали" side panel.
func (s *Server) handleDebug(w http.ResponseWriter, r *http.Request) {
	request, response := s.client.LastExchange()
	s.mu.Lock()
	detail := s.lastDetail
	s.mu.Unlock()

	out := map[string]any{"request": request, "response": response}
	if detail != "" {
		out["detail"] = json.RawMessage(detail)
	}
	writeJSON(w, out)
}

// setDetail stores the last error/limit detail as compact JSON for the
// "Детали" panel. Must be called without holding s.mu (it acquires it).
func (s *Server) setDetail(d map[string]any) {
	if d == nil {
		return
	}
	b, err := json.Marshal(d)
	if err != nil {
		return
	}
	s.mu.Lock()
	s.lastDetail = string(b)
	s.mu.Unlock()
}

// clearDetail resets the stored detail at the start of a chat request.
func (s *Server) clearDetail() {
	s.mu.Lock()
	s.lastDetail = ""
	s.mu.Unlock()
}

func (s *Server) handleReset(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	s.hist = make([]api.Message, 0, 16)
	if s.cfg.System != "" {
		s.hist = append(s.hist, api.SystemMessage(s.cfg.System))
	}
	s.mu.Unlock()
	writeJSON(w, map[string]bool{"ok": true})
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Message   string `json:"message"`
		Format    string `json:"format"`
		MaxTokens int    `json:"maxTokens"`
		Schema    string `json:"schema"`
		Stop      string `json:"stop"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	req.Message = strings.TrimSpace(req.Message)
	if req.Message == "" {
		http.Error(w, "message is empty", http.StatusBadRequest)
		return
	}

	format := normalizeFormat(req.Format)
	maxTokens := s.defaultMaxTokens()
	if req.MaxTokens > 0 {
		maxTokens = req.MaxTokens
	}
	if maxTokens < 1 {
		maxTokens = 1
	}
	if maxTokens > 15000 {
		maxTokens = 15000
	}

	// JSON Schema formats require a non-empty, valid JSON Schema.
	req.Schema = strings.TrimSpace(req.Schema)
	if format == "jsonschema" || format == "jsonschema_structured" {
		if req.Schema == "" {
			http.Error(w, "Для формата JSON Schema нужна схема", http.StatusBadRequest)
			return
		}
		if !json.Valid([]byte(req.Schema)) {
			http.Error(w, "Некорректный JSON Schema", http.StatusBadRequest)
			return
		}
	}

	s.clearDetail()

	model := s.currentModel()

	s.mu.Lock()
	s.hist = append(s.hist, api.UserMessage(req.Message))
	messages := append([]api.Message(nil), s.hist...)
	s.mu.Unlock()

	s.genMu.Lock()
	defer s.genMu.Unlock()

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout)
	defer cancel()

	// Abort the request if the model produces no output within noReplyTimeout.
	first := make(chan struct{})
	aborted := make(chan struct{})
	go func() {
		select {
		case <-first:
			// First delta arrived; the model is alive.
		case <-time.After(s.noReplyTimeout):
			close(aborted)
			log.Printf("chat: no response from model %q within %s; aborted", model, s.noReplyTimeout)
			cancel()
		case <-ctx.Done():
		}
	}()

	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	flusher, canFlush := w.(http.Flusher)

	// The status is sent lazily: if the API fails before the first token,
	// we can still answer with a real error code.
	var answer strings.Builder
	wrote := false
	firstWritten := false
	writeDelta := func(delta string) {
		if !firstWritten {
			firstWritten = true
			close(first)
		}
		answer.WriteString(delta)
		if !wrote {
			w.WriteHeader(http.StatusOK)
			wrote = true
		}
		_, _ = io.WriteString(w, delta)
		if canFlush {
			flusher.Flush()
		}
	}

	_, err := s.client.StreamChat(ctx, model, messages, maxTokens, writeDelta, api.Option{ResponseFormat: format, Schema: req.Schema, Stop: req.Stop})

	isAborted := false
	select {
	case <-aborted:
		isAborted = true
	default:
	}

	s.mu.Lock()
	if err != nil {
		// Drop the failed user turn so it can be retried.
		if n := len(s.hist); n > 0 && s.hist[n-1].Role == "user" {
			s.hist = s.hist[:n-1]
		}
		s.mu.Unlock()

		if isAborted {
			s.setDetail(map[string]any{
				"type":          "no_response",
				"finish_reason": "timeout",
				"max_tokens":    maxTokens,
				"summary":       "Модель не ответила за " + s.noReplyTimeout.String(),
				"detail":        fmt.Sprintf("Модель не вернула ни одного токена за %s — запрос был отменён.", s.noReplyTimeout),
			})
		} else {
			s.setDetail(map[string]any{
				"type":    "api_error",
				"summary": "Ошибка API",
				"detail":  err.Error(),
			})
		}

		if !wrote {
			if isAborted {
				http.Error(w, fmt.Sprintf("Модель не ответила за %s. Запрос отменён.", s.noReplyTimeout), http.StatusGatewayTimeout)
				return
			}
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		fmt.Fprintf(w, "\n\n[ошибка: %v]", err)
		return
	}
	reply := answer.String()
	// If a stop sequence was requested, make sure the chat gets exactly the
	// text generated before it, even if a provider echoes the stop token.
	if req.Stop != "" {
		reply = strings.TrimSuffix(reply, req.Stop)
	}
	if strings.TrimSpace(reply) != "" {
		s.hist = append(s.hist, api.AssistantMessage(reply))
	} else if len(s.hist) > 0 && s.hist[len(s.hist)-1].Role == "user" {
		// The model produced no visible content (for example a stop sequence
		// hit inside the reasoning trace). Drop the just-appended user message
		// so it does not linger as an orphan and bloat later requests.
		s.hist = s.hist[:len(s.hist)-1]
	}
	s.mu.Unlock()

	// Detect schema violations and token-limit truncation so the "Детали"
	// panel can surface an error block and the chat can log a short message.
	finish := s.client.LastFinishReason()
	var detail map[string]any
	switch {
	case finish == "length":
		detail = map[string]any{
			"type":          "token_limit",
			"finish_reason": "length",
			"max_tokens":    maxTokens,
			"summary":       "Ответ превысил лимит токенов",
			"detail":        fmt.Sprintf("Модель остановилась по лимиту в %d токенов, не завершив ответ.", maxTokens),
		}
	case (format == "jsonschema" || format == "jsonschema_structured") && reply != "" && !json.Valid([]byte(reply)):
		detail = map[string]any{
			"type":    "schema_violation",
			"summary": "Ответ не соответствует JSON-схеме",
			"detail":  "Модель вернула не JSON несмотря на требование JSON Schema.",
			"content": reply,
		}
	}
	if detail != nil {
		s.setDetail(detail)
	}

	if !wrote {
		// Empty answer: still finish the stream properly.
		w.WriteHeader(http.StatusOK)
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}
