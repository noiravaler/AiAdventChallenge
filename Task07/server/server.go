// Package server serves the browser chat interface.
package server

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"ai-chat-go/api"
	"ai-chat-go/config"
	"ai-chat-go/tools"
)

//go:embed static/index.html
var indexHTML []byte

// defaultNoReplyTimeout aborts the request when the model produces no token
// (visible content or reasoning trace) for this long. It can be overridden per
// request by the "timeout" field (5..30 s) from the UI slider.
const defaultNoReplyTimeout = 10 * time.Second

// modelRefreshTimeout caps how long /api/models spends querying the upstream
// /models endpoint before falling back to the last known model list.
const modelRefreshTimeout = 5 * time.Second

// historyRetention is how long a per-model context survives without a message.
// Contexts with no messages newer than this are purged (on startup and on the
// model-list refresh).
const historyRetention = 3 * 24 * time.Hour

// Server keeps the conversation state for the browser chat.
type Server struct {
	client *api.Client
	cfg    config.Config

	genMu sync.Mutex // serializes chat generations
	mu    sync.Mutex // guards history and model selection
	hist  []histEntry

	model          string        // currently selected model
	models         []string      // models offered in the contact sidebar
	noReplyTimeout time.Duration // abort when the model stays silent this long

	reserve      *api.Client // independent fallback provider (not via GPUStack)
	reserveModel string      // model served by the reserve provider

	lastDetail string // compact JSON of the last error/limit detail (Детали panel)

	// lastTrace keeps the last chat run's interaction steps and aggregated
	// token accounting / duration, served to the "Детали" panel.
	lastSteps      []tools.RunStep
	lastTokens     int
	lastPrompt     int
	lastCompletion int
	lastTimeMS     int64

	historyFile string // path to the persisted conversation; empty disables it
}

// histEntry is a conversation turn: the message itself plus the wall-clock time
// it was produced. The timestamp lets the UI group messages by day and show
// each message's time; it is also persisted so it survives a restart.
type histEntry struct {
	At  time.Time
	Msg api.Message
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
	// Independent reserve provider: not related to the main endpoint at all.
	if cfg.ReserveBaseURL != "" {
		s.reserve = api.New(cfg.ReserveBaseURL, cfg.ReserveAPIKey, cfg.Timeout)
		s.reserve.DebugLog = nil // панель «Детали» читает обмены из памяти клиента
		s.reserveModel = cfg.ReserveModel
	}
	s.hist = make([]histEntry, 0, 16)
	if cfg.System != "" {
		s.hist = append(s.hist, histEntry{At: time.Now(), Msg: api.SystemMessage(cfg.System)})
	}
	s.historyFile = cfg.HistoryFile
	s.restore()
	return s
}

// restore reloads a persisted per-model conversation (and the active model) so
// a restart continues where the user left off. It is a no-op when persistence
// is disabled (empty historyFile). Stale contexts (no message within the
// retention window) are purged first; the context with the newest activity is
// chosen as the active model. The system message is re-seeded from the current
// configuration rather than stored.
func (s *Server) restore() {
	if s.historyFile == "" {
		return
	}
	s.migrateLegacy()
	s.purgeOldContexts()

	// Pick the context file with the newest message as the active model.
	bestFile, bestModel := "", ""
	var bestAt time.Time
	for _, f := range s.contextFiles() {
		st, err := loadState(f)
		if err != nil || st == nil {
			continue
		}
		if latest := latestAt(st.Messages); latest.After(bestAt) {
			bestAt = latest
			bestFile, bestModel = f, st.Model
		}
	}
	if bestFile == "" {
		return
	}
	// Restore the active model only if it is still a known choice (or reserve).
	if bestModel != "" {
		for _, m := range s.models {
			if m == bestModel {
				s.model = bestModel
				break
			}
		}
		if s.isReserve(bestModel) {
			s.model = bestModel
		}
	}
	st, err := loadState(bestFile)
	if err != nil || st == nil {
		return
	}
	for _, r := range st.Messages {
		s.hist = append(s.hist, histEntry{At: r.At, Msg: r.message()})
	}
}

// migrateLegacy moves any context files the older implementation left in the
// working directory (the single-file context and any per-model files) into the
// dedicated history folder, so no conversation is lost across this upgrade. It
// is a no-op if there is nothing to migrate or a per-model file already exists.
func (s *Server) migrateLegacy() {
	if s.historyFile == "" {
		return
	}
	name := filepath.Base(s.historyFile)
	ext := filepath.Ext(name)
	stem := strings.TrimSuffix(name, ext)
	targetDir := filepath.Dir(s.historyFile)
	// historyFile itself may be a legacy single-file context (e.g. a custom
	// AI_HISTORY_FILE pointing at one file that predates per-model storage).
	s.migrateOne(s.historyFile, targetDir)
	// Per-model legacy files lived next to the process (the working directory).
	matches, _ := filepath.Glob(filepath.Join(".", stem+"*"+ext))
	for _, f := range matches {
		if filepath.Dir(f) == targetDir {
			continue // already under the dedicated folder
		}
		s.migrateOne(f, targetDir)
	}
}

// migrateOne migrates a single legacy context file (containing a Model) into
// the per-model file for that model, unless a per-model file already exists.
func (s *Server) migrateOne(f, _ string) {
	st, err := loadState(f)
	if err != nil || st == nil || st.Model == "" {
		return
	}
	dst := s.contextFileFor(st.Model)
	if dst == "" || dst == f {
		return
	}
	if _, err := os.Stat(dst); err == nil {
		return // per-model file already present
	}
	if err := saveState(dst, st.Model, st.Messages); err == nil {
		log.Printf("history: migrated legacy %s -> %s", f, dst)
		_ = os.Remove(f)
	}
}

// persist writes the current conversation (non-system turns) and the active
// model to its own per-model file. It is called after every history change and
// is a no-op when persistence is disabled.
func (s *Server) persist() {
	if s.historyFile == "" {
		return
	}
	s.mu.Lock()
	var msgs []chatRecord
	for _, e := range s.hist {
		if e.Msg.Role != "system" {
			msgs = append(msgs, recordOf(e.Msg, e.At))
		}
	}
	model := s.model
	path := s.contextFileFor(model)
	s.mu.Unlock()
	if path == "" {
		return
	}
	if err := saveState(path, model, msgs); err != nil {
		log.Printf("history: failed to save %s: %v", path, err)
	}
}

// contextFileFor returns the per-model context path for a given model.
func (s *Server) contextFileFor(model string) string {
	return modelHistoryPath(s.historyFile, model)
}

// contextFiles returns every per-model context file on disk.
func (s *Server) contextFiles() []string {
	if s.historyFile == "" {
		return nil
	}
	dir := filepath.Dir(s.historyFile)
	name := filepath.Base(s.historyFile)
	ext := filepath.Ext(name)
	stem := strings.TrimSuffix(name, ext)
	matches, _ := filepath.Glob(filepath.Join(dir, stem+"-*"+ext))
	return matches
}

// purgeOldContexts enforces the retention window per context file, at two
// levels:
//  1. Per-message: individual messages older than the window are dropped from a
//     file that still has recent activity (so stale turns don't linger inside an
//     otherwise live conversation).
//  2. Whole-file: a file whose newest remaining activity (message timestamp, or
//     its own modification time when messages carry no timestamp) is older than
//     the window is removed entirely.
//
// Files are never removed because a model is absent from the current list — only
// inactivity matters.
func (s *Server) purgeOldContexts() {
	if s.historyFile == "" {
		return
	}
	now := time.Now()
	for _, f := range s.contextFiles() {
		st, err := loadState(f)
		if err != nil || st == nil {
			continue
		}
		// Per-message retention: keep only messages within the window. Messages
		// with no timestamp are kept (their age is unknown; the file's own mtime
		// below decides whether the whole file is stale).
		kept := make([]chatRecord, 0, len(st.Messages))
		for _, m := range st.Messages {
			if m.At.IsZero() || now.Sub(m.At) <= historyRetention {
				kept = append(kept, m)
			}
		}
		pruned := len(kept) != len(st.Messages)

		if len(kept) == 0 {
			// Every message was stale -> drop the whole (now empty) context.
			if err := os.Remove(f); err == nil || errors.Is(err, os.ErrNotExist) {
				log.Printf("history: removed stale context %s (no messages within window)", f)
			} else {
				log.Printf("history: failed to remove stale %s: %v", f, err)
			}
			continue
		}

		latest := latestAt(kept)
		if latest.IsZero() {
			// No message timestamps (e.g. a context written before the "at"
			// field existed, or an empty context). Fall back to the file's own
			// modification time so such stale files still get cleaned up.
			if info, err := os.Stat(f); err == nil {
				latest = info.ModTime()
			} else {
				continue
			}
		}
		if now.Sub(latest) > historyRetention {
			if err := os.Remove(f); err != nil {
				log.Printf("history: failed to remove stale %s: %v", f, err)
			} else {
				log.Printf("history: removed stale context %s (last activity %s)", f, latest.Format(time.RFC3339))
			}
			continue
		}

		if pruned {
			if err := saveState(f, st.Model, kept); err != nil {
				log.Printf("history: failed to prune %s: %v", f, err)
			} else {
				log.Printf("history: pruned %d stale message(s) from %s", len(st.Messages)-len(kept), f)
			}
		}
	}
}

// isReserve reports whether the model belongs to the reserve provider.
func (s *Server) isReserve(model string) bool {
	return s.reserveModel != "" && model == s.reserveModel
}

// clientFor returns the API client responsible for the given model: the
// reserve client for the reserve model, the main client otherwise.
func (s *Server) clientFor(model string) *api.Client {
	if s.isReserve(model) && s.reserve != nil {
		return s.reserve
	}
	return s.client
}

// clientStream streams a chat completion through the client responsible for
// the given model (main provider or the reserve one). onActivity fires for
// every model token, including reasoning chunks not shown in the chat.
func (s *Server) clientStream(ctx context.Context, model string, messages []api.Message, maxTokens int, onDelta func(string), onActivity func(), opts ...api.Option) (string, error) {
	return s.clientFor(model).StreamChatWithActivity(ctx, model, messages, maxTokens, onDelta, onActivity, opts...)
}

// refreshModels queries the upstream /models endpoint, drops obviously non-chat
// models and updates the sidebar list. On any error — or when nothing chat-like
// is left — the currently known list is kept untouched.
func (s *Server) refreshModels(ctx context.Context) {
	ids, err := s.client.ListModels(ctx)
	if err != nil {
		return
	}
	chat := filterChatModels(ids)
	if len(chat) == 0 {
		return
	}

	s.mu.Lock()
	s.models = chat
	// Keep a valid current selection after the list is replaced.
	found := false
	for _, m := range s.models {
		if m == s.model {
			found = true
			break
		}
	}
	if !found {
		s.model = s.models[0]
	}
	s.mu.Unlock()

	// Purely time-based cleanup: remove contexts with no message in the last
	// few days. Files are never deleted merely because a model is missing from
	// the refreshed list.
	s.purgeOldContexts()
}

// filterChatModels removes model IDs that are clearly not chat completions:
// embeddings, rerankers and speech/audio models, matched by name patterns.
func filterChatModels(ids []string) []string {
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		lower := strings.ToLower(id)
		if strings.Contains(lower, "embedding") ||
			strings.Contains(lower, "rerank") ||
			strings.Contains(lower, "whisper") ||
			strings.Contains(lower, "tts") ||
			strings.Contains(lower, "audio") ||
			strings.Contains(lower, "asr") {
			continue
		}
		out = append(out, id)
	}
	return out
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

// clampRounds keeps the tool-call round budget within the 1..10 range used by
// the UI slider; a non-positive value yields the default of 6 rounds.
func clampRounds(v int) int {
	switch {
	case v <= 0:
		return 6
	case v < 1:
		return 1
	case v > 10:
		return 10
	default:
		return v
	}
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

// handleModels lists the models available in the contact sidebar, refreshing
// them from the upstream /models endpoint so the sidebar always shows current
// chat models. When the refresh fails the last known list is used (seeded from
// AI_MODELS / DefaultModels).
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), modelRefreshTimeout)
	s.refreshModels(ctx)
	cancel()

	s.mu.Lock()
	models := append([]string(nil), s.models...)
	current := s.model
	s.mu.Unlock()
	out := map[string]any{"models": models, "current": current}
	if s.reserveModel != "" {
		out["reserve"] = s.reserveModel
		out["reserve_ready"] = s.cfg.ReserveConfigured()
	}
	writeJSON(w, out)
}

// handleModel switches the active model. The outgoing model's context is saved
// to its own file and the newly-selected model's persisted context is loaded
// (a fresh session if that model has none yet).
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
	if !valid && s.isReserve(req.Model) {
		if !s.cfg.ReserveConfigured() {
			s.mu.Unlock()
			http.Error(w, "Резервный провайдер не настроен: задайте AI_RESERVE_API_KEY (и, при необходимости, AI_RESERVE_BASE_URL/AI_RESERVE_MODEL) в .env", http.StatusServiceUnavailable)
			return
		}
		valid = true
	}
	if !valid {
		s.mu.Unlock()
		http.Error(w, "unknown model", http.StatusBadRequest)
		return
	}
	// Persist the outgoing model's context before switching, then load the
	// newly-selected model's own persisted context (or start fresh if none).
	oldModel := s.model
	oldPath := s.contextFileFor(oldModel)
	var oldMsgs []chatRecord
	for _, e := range s.hist {
		if e.Msg.Role != "system" {
			oldMsgs = append(oldMsgs, recordOf(e.Msg, e.At))
		}
	}

	s.model = req.Model
	s.hist = make([]histEntry, 0, 16)
	if s.cfg.System != "" {
		s.hist = append(s.hist, histEntry{At: time.Now(), Msg: api.SystemMessage(s.cfg.System)})
	}
	newPath := s.contextFileFor(req.Model)
	s.mu.Unlock()

	if oldPath != "" {
		if err := saveState(oldPath, oldModel, oldMsgs); err != nil {
			log.Printf("history: failed to save %s: %v", oldPath, err)
		}
	}
	if newPath != "" {
		if st, err := loadState(newPath); err == nil && st != nil {
			s.mu.Lock()
			for _, r := range st.Messages {
				s.hist = append(s.hist, histEntry{At: r.At, Msg: r.message()})
			}
			s.mu.Unlock()
		}
	}

	log.Printf("chat: switched to model %s (context loaded)", req.Model)
	writeJSON(w, map[string]bool{"ok": true})
}

type historyMessage struct {
	Role    string    `json:"role"`
	Content string    `json:"content"`
	At      time.Time `json:"at"`
}

func (s *Server) handleHistory(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	msgs := make([]historyMessage, 0, len(s.hist))
	for _, e := range s.hist {
		// Только отображаемые сообщения: без системного промпта и без
		// промежуточных tool-сообщений (ассистент с tool_calls и ответы
		// инструментов). Пользовательский ввод и итоговый ответ ассистента.
		if e.Msg.Role == "system" || e.Msg.Role == "tool" {
			continue
		}
		if strings.TrimSpace(e.Msg.Content) == "" {
			continue
		}
		msgs = append(msgs, historyMessage{Role: e.Msg.Role, Content: e.Msg.Content, At: e.At})
	}
	s.mu.Unlock()
	writeJSON(w, map[string]any{"messages": msgs})
}

// handleDebug returns the JSON of the last API request and response for the
// "Детали" side panel.
func (s *Server) handleDebug(w http.ResponseWriter, r *http.Request) {
	request, response := s.clientFor(s.currentModel()).LastExchange()
	s.mu.Lock()
	detail := s.lastDetail
	steps := append([]tools.RunStep(nil), s.lastSteps...)
	tokens := s.lastTokens
	prompt := s.lastPrompt
	completion := s.lastCompletion
	timeMS := s.lastTimeMS
	s.mu.Unlock()

	out := map[string]any{
		"request":           request,
		"response":          response,
		"steps":             steps,
		"tokens":            tokens,
		"prompt_tokens":     prompt,
		"completion_tokens": completion,
		"time_ms":           timeMS,
	}
	if detail != "" {
		out["detail"] = json.RawMessage(detail)
	}
	writeJSON(w, out)
}

// setTrace stores the interaction steps and token/duration accounting of the
// last chat run for the "Детали" panel.
func (s *Server) setTrace(steps []tools.RunStep, prompt, completion, tokens int, timeMS int64) {
	s.mu.Lock()
	s.lastSteps = append([]tools.RunStep(nil), steps...)
	s.lastPrompt = prompt
	s.lastCompletion = completion
	s.lastTokens = tokens
	s.lastTimeMS = timeMS
	s.mu.Unlock()
}

// clearTrace resets the stored trace at the start of a chat request.
func (s *Server) clearTrace() {
	s.mu.Lock()
	s.lastSteps = nil
	s.lastTokens = 0
	s.lastPrompt = 0
	s.lastCompletion = 0
	s.lastTimeMS = 0
	s.mu.Unlock()
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
	s.hist = make([]histEntry, 0, 16)
	if s.cfg.System != "" {
		s.hist = append(s.hist, histEntry{At: time.Now(), Msg: api.SystemMessage(s.cfg.System)})
	}
	model := s.model
	path := s.contextFileFor(model)
	s.mu.Unlock()
	// Кнопка удаления чистит только контекст текущей выбранной модели: её файл
	// истории удаляется с диска, остальные модели не затрагиваются.
	if path != "" {
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			log.Printf("history: failed to remove %s: %v", path, err)
		}
	}
	writeJSON(w, map[string]bool{"ok": true})
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Message   string `json:"message"`
		Format    string `json:"format"`
		MaxTokens int    `json:"maxTokens"`
		Timeout   int    `json:"timeout"` // допустимый таймаут ответа, секунды (5..30)
		MaxRounds int    `json:"maxRounds"`
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

	// Максимальное количество циклов вызова инструментов: клиент задаёт 1..10
	// (ползунок), по умолчанию 6.
	maxRounds := clampRounds(req.MaxRounds)

	// Допустимый таймаут ответа: клиент задаёт 5..100 секунд (по умолчанию 90).
	timeout := s.noReplyTimeout
	if req.Timeout > 0 {
		secs := req.Timeout
		if secs < 5 {
			secs = 5
		}
		if secs > 30 {
			secs = 30
		}
		timeout = time.Duration(secs) * time.Second
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
	s.clearTrace()

	model := s.currentModel()

	s.mu.Lock()
	s.hist = append(s.hist, histEntry{At: time.Now(), Msg: api.UserMessage(req.Message)})
	messages := make([]api.Message, 0, len(s.hist))
	for _, e := range s.hist {
		messages = append(messages, e.Msg)
	}
	s.mu.Unlock()

	s.genMu.Lock()
	defer s.genMu.Unlock()

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout)
	defer cancel()

	// Abort the request if the model produces no output within the allowed timeout.
	// Any generation counts as alive: visible content or the reasoning trace.
	first := make(chan struct{})
	var firstOnce sync.Once
	aborted := make(chan struct{})
	activity := func() { firstOnce.Do(func() { close(first) }) }
	go func() {
		select {
		case <-first:
			// First delta arrived; the model is alive.
		case <-time.After(timeout):
			close(aborted)
			log.Printf("chat: no response from model %q within %s; aborted", model, timeout)
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
	writeDelta := func(delta string) {
		firstOnce.Do(func() { close(first) })
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

	// Run the tool-calling loop: the model may call get_current_time / roll_dice
	// (and have the results fed back) before delivering a plain answer. The
	// final text is streamed to the browser through writeDelta; intermediate
	// tool exchanges stay internal. Each round's token usage is accumulated so
	// the "Детали" panel can show the total.
	start := time.Now()
	var used api.Usage
	round := func(ctx context.Context, conv []api.Message) (string, []api.ToolCall, error) {
		content, calls, err := s.clientFor(model).StreamChatToolCalls(ctx, model, conv, maxTokens, writeDelta, activity, api.Option{
			ResponseFormat: format,
			Schema:         req.Schema,
			Stop:           req.Stop,
			Tools:          tools.Definitions(),
		})
		u := s.clientFor(model).LastUsage()
		used.PromptTokens += u.PromptTokens
		used.CompletionTokens += u.CompletionTokens
		used.TotalTokens += u.TotalTokens
		return content, calls, err
	}
	reply, steps, added, err := tools.RunLoop(ctx, round, messages, maxRounds)
	s.setTrace(steps, used.PromptTokens, used.CompletionTokens, used.TotalTokens, time.Since(start).Milliseconds())

	isAborted := false
	select {
	case <-aborted:
		isAborted = true
	default:
	}

	s.mu.Lock()
	if err != nil {
		// Drop the failed user turn so it can be retried.
		if n := len(s.hist); n > 0 && s.hist[n-1].Msg.Role == "user" {
			s.hist = s.hist[:n-1]
		}
		s.mu.Unlock()
		s.persist()

		if isAborted {
			s.setDetail(map[string]any{
				"type":          "no_response",
				"finish_reason": "timeout",
				"max_tokens":    maxTokens,
				"summary":       "Модель не ответила за " + timeout.String(),
				"detail":        fmt.Sprintf("Модель не вернула ни одного токена за %s — запрос был отменён.", timeout),
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
				http.Error(w, fmt.Sprintf("Модель не ответила за %s. Запрос отменён.", timeout), http.StatusGatewayTimeout)
				return
			}
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		fmt.Fprintf(w, "\n\n[ошибка: %v]", err)
		return
	}
	// If a stop sequence was requested, make sure the chat gets exactly the
	// text generated before it, even if a provider echoes the stop token.
	if req.Stop != "" {
		reply = strings.TrimSuffix(reply, req.Stop)
	}
	now := time.Now()
	if strings.TrimSpace(reply) != "" {
		// Persist the intermediate tool exchanges (assistant tool_calls + tool
		// results) and then the final answer, so a restart replays the same
		// conversation to the model.
		for _, am := range added {
			s.hist = append(s.hist, histEntry{At: now, Msg: am})
		}
		s.hist = append(s.hist, histEntry{At: now, Msg: api.AssistantMessage(reply)})
	} else if len(s.hist) > 0 && s.hist[len(s.hist)-1].Msg.Role == "user" {
		// The model produced no visible content (for example a stop sequence
		// hit inside the reasoning trace). Drop the just-appended user message
		// so it does not linger as an orphan and bloat later requests.
		s.hist = s.hist[:len(s.hist)-1]
	}
	s.mu.Unlock()
	s.persist()

	// Detect schema violations and token-limit truncation so the "Детали"
	// panel can surface an error block and the chat can log a short message.
	finish := s.clientFor(model).LastFinishReason()
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
