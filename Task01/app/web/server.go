// Package web serves the browser chat interface.
package web

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"ai-chat-go/api"
	"ai-chat-go/config"
)

//go:embed static/index.html
var indexHTML []byte

// Server keeps the conversation state for the browser chat.
type Server struct {
	client *api.Client
	cfg    config.Config

	genMu sync.Mutex // serializes chat generations
	mu    sync.Mutex // guards history
	hist  []api.Message
}

// New creates a web chat server with the system prompt from cfg.
func New(client *api.Client, cfg config.Config) *Server {
	s := &Server{client: client, cfg: cfg}
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
	mux.HandleFunc("GET /api/history", s.handleHistory)
	mux.HandleFunc("POST /api/chat", s.handleChat)
	mux.HandleFunc("POST /api/reset", s.handleReset)
	return mux
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(indexHTML)
}

func (s *Server) handleInfo(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]string{
		"model":   s.cfg.Model,
		"baseURL": s.cfg.BaseURL,
	})
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
		Message string `json:"message"`
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

	s.mu.Lock()
	s.hist = append(s.hist, api.UserMessage(req.Message))
	messages := append([]api.Message(nil), s.hist...)
	s.mu.Unlock()

	s.genMu.Lock()
	defer s.genMu.Unlock()

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout)
	defer cancel()

	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	flusher, canFlush := w.(http.Flusher)

	// The status is sent lazily: if the API fails before the first token,
	// we can still answer with a real error code.
	var answer strings.Builder
	wrote := false
	writeDelta := func(delta string) {
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

	_, err := s.client.StreamChat(ctx, s.cfg.Model, messages, s.cfg.MaxTokens, writeDelta)

	s.mu.Lock()
	if err != nil {
		// Drop the failed user turn so it can be retried.
		if n := len(s.hist); n > 0 && s.hist[n-1].Role == "user" {
			s.hist = s.hist[:n-1]
		}
		s.mu.Unlock()

		if !wrote {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		fmt.Fprintf(w, "\n\n[ошибка: %v]", err)
		return
	}
	reply := answer.String()
	if strings.TrimSpace(reply) != "" {
		s.hist = append(s.hist, api.AssistantMessage(reply))
	}
	s.mu.Unlock()

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
