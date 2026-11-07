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
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"ai-chat-go/api"
	"ai-chat-go/config"
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

	reserve      *api.Client // independent fallback provider (not via GPUStack)
	reserveModel string      // model served by the reserve provider

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
	// Independent reserve provider: not related to the main endpoint at all.
	if cfg.ReserveBaseURL != "" {
		s.reserve = api.New(cfg.ReserveBaseURL, cfg.ReserveAPIKey, cfg.Timeout)
		s.reserve.DebugLog = nil // панель «Детали» читает обмены из памяти клиента
		s.reserveModel = cfg.ReserveModel
	}
	s.hist = make([]api.Message, 0, 16)
	if cfg.System != "" {
		s.hist = append(s.hist, api.SystemMessage(cfg.System))
	}
	return s
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
	mux.HandleFunc("POST /api/solve", s.handleSolve)
	mux.HandleFunc("POST /api/run", s.handleRun)
	mux.HandleFunc("POST /api/run-conclusion", s.handleRunConclusion)

	mux.HandleFunc("POST /api/general-conclusion", s.handleGeneralConclusion)
	mux.HandleFunc("POST /api/conclusion", s.handleConclusion)
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
// changes it with the slider, clamped to the 1..20000 UI range.
func (s *Server) defaultMaxTokens() int {
	v := s.cfg.MaxTokens
	if v <= 0 {
		v = 20000
	}
	if v > 20000 {
		v = 20000
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
	request, response := s.clientFor(s.currentModel()).LastExchange()
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
		Timeout   int    `json:"timeout"` // допустимый таймаут ответа, секунды (5..30)
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
	if maxTokens > 20000 {
		maxTokens = 20000
	}

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

	model := s.currentModel()

	s.mu.Lock()
	s.hist = append(s.hist, api.UserMessage(req.Message))
	messages := append([]api.Message(nil), s.hist...)
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

	_, err := s.clientStream(ctx, model, messages, maxTokens, writeDelta, activity, api.Option{ResponseFormat: format, Schema: req.Schema, Stop: req.Stop})

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

// solutionRequest describes one "Решения" run: a task condition solved with a
// chosen method (and, for the expert method, an expert persona).
type solutionRequest struct {
	Method      string `json:"method"`    // direct | stepwise | prompt | expert
	Expert      string `json:"expert"`    // mathematician | philosopher | critic (for method=expert)
	Condition   string `json:"condition"` // the task text to solve
	ExpectedAns string `json:"expected"`  // expected final answer, used to judge success
	Timeout     int    `json:"timeout"`   // allowed solve time in seconds (10..60; default 10)
	Reasoning   string `json:"reasoning"` // reasoning effort: "low" | "high" (from "Настройка рассуждения")
}

// solutionResult is the outcome of a single solution run, shown as one row in
// the "Решения" table.
type solutionResult struct {
	Method   string `json:"method"`
	Expert   string `json:"expert,omitempty"`
	Name     string `json:"name"`                // display label, e.g. "Прямой ответ", "Эксперт: Математик"
	Tokens   int    `json:"tokens"`              // total tokens consumed by the run
	Duration int64  `json:"duration"`            // wall-clock milliseconds
	Success  bool   `json:"success"`             // model answer matched the expected one
	TimedOut bool   `json:"timed_out,omitempty"` // run aborted because it exceeded the timeout
	Solution string `json:"solution"`            // model's working
	Prompt   string `json:"prompt"`              // composed prompt (only for method=prompt)
	Reason   string `json:"reason,omitempty"`    // why the run produced no answer
	Error    string `json:"error,omitempty"`
}

// methodDisplayName builds the human-readable label of a run.
func methodDisplayName(method, expert string) string {
	switch method {
	case "stepwise":
		return "Пошаговое решение"
	case "prompt":
		return "Через промпт"
	case "expert":
		return "Эксперт: " + expertLabel(expert)
	default:
		return "Прямой ответ"
	}
}

// expertLabel maps an expert key to its display name.
func expertLabel(expert string) string {
	switch expert {
	case "mathematician":
		return "Математик"
	case "philosopher":
		return "Философ"
	case "critic":
		return "Критик"
	default:
		return "Эксперт"
	}
}

// reasoningDirective returns an instruction injected into the system prompt so
// the selected "Настройка рассуждения" clearly affects the model's output.
// Many self-hosted models ignore the reasoning_effort API field, so we encode
// the choice into the prompt itself. Empty string means "no change".
func reasoningDirective(effort string) string {
	switch effort {
	case "low":
		return "Действуй лаконично: без подробного рассуждения дай краткий ход решения и сразу окончательный ответ."
	case "high":
		return "Рассуждай максимально подробно: распиши каждый шаг, промежуточные выкладки и проверку, прежде чем назвать окончательный ответ."
	default:
		return ""
	}
}

// expertPersona returns the system overrides that channel the model into an
// expert role.
func expertPersona(expert string) string {
	switch expert {
	case "mathematician":
		return "Ты — математик. Решай задачу строго и приводи обоснованные выкладки."
	case "philosopher":
		return "Ты — философ. Рассмотри задачу, рассуждай логически и приведи решение."
	case "critic":
		return "Ты — критик. Внимательно проверь задачу, найди возможные подводные камни и приведи решение."
	default:
		return ""
	}
}

// numberRe matches a standalone integer in free text.
var numberRe = regexp.MustCompile(`-?\d+`)

// fractionRe matches a simple "a/b" fraction in free text.
var fractionRe = regexp.MustCompile(`-?\d+(?:[.,]\d+)?\s*/\s*-?\d+(?:[.,]\d+)?`)

// fracExprRe matches a LaTeX \frac{a}{b} command.
var fracExprRe = regexp.MustCompile(`\\frac\{([^{}]*)\}\{([^{}]*)\}`)

// latexCmdRe strips LaTeX wrappers (\left, \right) and ordinary command words.
var latexCmdRe = regexp.MustCompile(`\\(?:left|right|[a-zA-Z]+)`)

// answerSepRe normalizes punctuation separators into a single space.
var answerSepRe = regexp.MustCompile(`[;,:]+`)

// answerSpacesRe collapses runs of whitespace.
var answerSpacesRe = regexp.MustCompile(`\s+`)

// answerWordRe finds words (including Cyrillic) and integers, used for token matching.
var answerWordRe = regexp.MustCompile(`[\p{L}]+|\d+`)

// stopTokens are common Russian filler words ignored when comparing long answers.
var stopTokens = map[string]bool{
	"это": true, "как": true, "что": true, "который": true, "которая": true,
	"которое": true, "которые": true, "при": true, "для": true, "затем": true,
	"также": true, "если": true, "то": true, "же": true, "бы": true, "или": true,
	"но": true, "а": true, "и": true, "не": true, "на": true, "в": true, "с": true,
	"по": true, "из": true, "у": true, "о": true, "об": true, "от": true, "до": true,
	"все": true, "всех": true, "всем": true, "они": true, "его": true, "её": true,
	"ее": true, "их": true, "этот": true, "эта": true, "будет": true, "стало": true,
	"когда": true,
}

// normalizeAnswer lowercases and flattens LaTeX/typographic variations so that
// different spellings of the same answer compare equal.
func normalizeAnswer(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	// \frac{a}{b} -> a/b
	s = fracExprRe.ReplaceAllString(s, "$1/$2")
	// strip \left,\right and other LaTeX command words
	s = latexCmdRe.ReplaceAllString(s, "")
	// punctuation separators -> space
	s = answerSepRe.ReplaceAllString(s, " ")
	// collapse whitespace
	s = answerSpacesRe.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}

// toNumber parses an integer, decimal or simple a/b fraction into a float.
func toNumber(s string) (float64, bool) {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '/'); i > 0 {
		a, e1 := strconv.ParseFloat(strings.TrimSpace(s[:i]), 64)
		b, e2 := strconv.ParseFloat(strings.TrimSpace(s[i+1:]), 64)
		if e1 == nil && e2 == nil && b != 0 {
			return a / b, true
		}
		return 0, false
	}
	v, err := strconv.ParseFloat(s, 64)
	return v, err == nil
}

// extractNumbers returns every integer/decimal/fraction value found in s.
func extractNumbers(s string) []float64 {
	var out []float64
	for _, m := range fractionRe.FindAllString(s, -1) {
		if v, ok := toNumber(m); ok {
			out = append(out, v)
		}
	}
	for _, m := range numberRe.FindAllString(s, -1) {
		if v, err := strconv.ParseFloat(m, 64); err == nil {
			out = append(out, v)
		}
	}
	return out
}

// answerMatchesNumber reports whether any numeric value in the reply equals any
// numeric value in the expected answer (which may be a bare number, a fraction
// like 7/3, or embedded in text such as "C(7/3; 0)").
func answerMatchesNumber(reply, target string) bool {
	expectedNumbers := extractNumbers(target)
	if len(expectedNumbers) == 0 {
		return false
	}
	for _, tv := range extractNumbers(reply) {
		for _, ev := range expectedNumbers {
			diff := tv - ev
			if diff < 0 {
				diff = -diff
			}
			if diff < 1e-6 {
				return true
			}
		}
	}
	return false
}

// stemMatch reports whether two tokens share a common prefix (len >= 4) or one is
// a prefix of the other, so "синеглазых" matches "синеглазые".
func stemMatch(a, b string) bool {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	p := 0
	for p < n && a[p] == b[p] {
		p++
	}
	return p >= 4 || strings.HasPrefix(a, b) || strings.HasPrefix(b, a)
}

// answerTokensMatch reports whether a reasonable share of the significant words of
// the expected answer appear in the reply (order-insensitive, morphology-tolerant).
func answerTokensMatch(reply, expected string) bool {
	reply = normalizeAnswer(reply)
	expected = normalizeAnswer(expected)

	var req []string
	for _, tk := range answerWordRe.FindAllString(expected, -1) {
		if len(tk) >= 3 && !stopTokens[tk] {
			req = append(req, tk)
		}
	}
	if len(req) == 0 {
		return false
	}

	words := answerWordRe.FindAllString(reply, -1)
	if len(words) == 0 {
		return false
	}

	matched := 0
	for _, want := range req {
		for _, w := range words {
			if stemMatch(w, want) {
				matched++
				break
			}
		}
	}

	// Require all tokens when few, otherwise allow a majority (lenient).
	if len(req) <= 1 {
		return matched == len(req)
	}
	return matched*2 >= len(req)
}

// isMostlyNumeric reports whether the expected answer is predominantly a number
// or a coordinate (e.g. "47", "7/3", "C(7/3; 0)"), as opposed to a prose sentence
// that merely happens to contain numbers. Numeric equivalence should only be
// attempted for such answers.
func isMostlyNumeric(expected string) bool {
	normalized := normalizeAnswer(expected)
	words := answerWordRe.FindAllString(normalized, -1)
	letters := 0
	numbers := 0
	for _, w := range words {
		if w[0] >= '0' && w[0] <= '9' {
			numbers++
		} else if len(w) >= 3 && !stopTokens[w] {
			letters++
		}
	}
	// A coordinate like "C(7/3; 0)" has one meaningful letter (C); fractions
	// "7/3" and the bare number "47" have none. Allow at most one letter.
	return numbers > 0 && letters <= 1
}

// answersMatch reports whether the model's reply conveys the expected answer,
// tolerating different spellings, LaTeX/typographic symbols and extra explanation.
func answersMatch(reply, expected string) bool {
	rn := normalizeAnswer(reply)
	en := normalizeAnswer(expected)
	if rn == "" || en == "" {
		return false
	}
	// Numeric (integer, decimal, fraction) equivalence — only for predominantly
	// numeric expected answers, to avoid false positives inside prose.
	if isMostlyNumeric(expected) && answerMatchesNumber(reply, expected) {
		return true
	}
	// Direct containment after normalization (punctuation/space/symbol variants).
	if strings.Contains(rn, en) {
		return true
	}
	// Sentence / coordinate: require a share of significant words to be present.
	return answerTokensMatch(reply, expected)
}

// handleSolve runs one solution of the selected task using the chosen method
// and returns the outcome for the "Решения" table.
func (s *Server) handleSolve(w http.ResponseWriter, r *http.Request) {
	var req solutionRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	req.Condition = strings.TrimSpace(req.Condition)
	req.ExpectedAns = strings.TrimSpace(req.ExpectedAns)
	if req.Condition == "" {
		http.Error(w, "condition is empty", http.StatusBadRequest)
		return
	}

	method := strings.TrimSpace(req.Method)
	if method == "" {
		method = "direct"
	}
	if method != "direct" && method != "stepwise" && method != "prompt" && method != "expert" {
		http.Error(w, "unknown method", http.StatusBadRequest)
		return
	}
	if method == "expert" && strings.TrimSpace(req.Expert) == "" {
		http.Error(w, "expert is empty", http.StatusBadRequest)
		return
	}

	s.genMu.Lock()
	defer s.genMu.Unlock()

	model := s.currentModel()
	maxTokens := s.defaultMaxTokens()

	// Reasoning effort from the "Настройка рассуждения" toggle (min→low,
	// max→high). Passed to the model when non-empty; models that do not
	// support reasoning_effort ignore the field.
	reasoningEffort := strings.TrimSpace(req.Reasoning)
	var effOpt api.Option
	switch reasoningEffort {
	case "low", "high":
		effOpt.ReasoningEffort = reasoningEffort
	}
	opts := []api.Option{}
	if effOpt.ReasoningEffort != "" {
		opts = append(opts, effOpt)
	}

	// Allowed solve time: 10..60 seconds, default 10. This replaces the global
	// cfg.Timeout so each "Решения" run is bounded by the slider in the UI.
	timeoutSecs := req.Timeout
	if timeoutSecs == 0 {
		timeoutSecs = 10
	}
	if timeoutSecs < 10 {
		timeoutSecs = 10
	}
	if timeoutSecs > 60 {
		timeoutSecs = 60
	}
	ctx, cancel := context.WithTimeout(r.Context(), time.Duration(timeoutSecs)*time.Second)
	defer cancel()

	start := time.Now()
	var answer string
	var total api.Usage
	var composedPrompt string

	run := func(messages []api.Message) (string, error) {
		res, usage, err := s.clientFor(model).ChatWithUsage(ctx, model, messages, maxTokens, opts...)
		total.TotalTokens += usage.TotalTokens
		total.PromptTokens += usage.PromptTokens
		total.CompletionTokens += usage.CompletionTokens
		return res, err
	}

	baseSystem := "Ты — помощник по решению задач. Приведи подробный ход решения и окончательный ответ."

	// Усиление влияния «Настройки рассуждения»: помимо поля reasoning_effort,
	// явно направляем модель через системный промпт, т.к. многие self-hosted
	// модели (deepseek, qwen и др.) игнорируют этот API-параметр.
	if dir := reasoningDirective(reasoningEffort); dir != "" {
		baseSystem += " " + dir
	}

	build := func(user string) []api.Message {
		msgs := []api.Message{api.SystemMessage(baseSystem)}
		if method == "expert" {
			if p := expertPersona(req.Expert); p != "" {
				if dir := reasoningDirective(reasoningEffort); dir != "" {
					p += " " + dir
				}
				msgs[0] = api.SystemMessage(p)
			}
		}
		if method == "prompt" && composedPrompt != "" {
			user = composedPrompt + "\n\n" + user
		}
		return append(msgs, api.UserMessage(user))
	}

	var err error
	switch method {
	case "stepwise":
		answer, err = run(build(req.Condition + "\n\nРешай задачу пошагово."))
	case "prompt":
		// First the model composes a prompt for solving the task...
		composedPrompt, err = run([]api.Message{
			api.SystemMessage("Ты — методист. Составь подробный промпт, по которому можно решить следующую задачу. Верни только текст промпта."),
			api.UserMessage(req.Condition),
		})
		if err == nil {
			// ...then it uses that prompt to solve the original condition.
			answer, err = run(build(req.Condition))
		}
	default: // direct, expert
		answer, err = run(build(req.Condition))
	}

	res := solutionResult{
		Method:   method,
		Expert:   req.Expert,
		Name:     methodDisplayName(method, req.Expert),
		Tokens:   total.TotalTokens,
		Duration: time.Since(start).Milliseconds(),
		Solution: strings.TrimSpace(answer),
		Prompt:   composedPrompt,
	}
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			res.TimedOut = true
			res.Error = fmt.Sprintf("Превышен таймаут %d с — решение остановлено", timeoutSecs)
		} else {
			res.Error = err.Error()
		}
	} else {
		res.Success = answersMatch(answer, req.ExpectedAns)
		if strings.TrimSpace(answer) == "" {
			res.Success = false
			res.Reason = "Модель вернула пустой ответ"
		}
	}

	writeJSON(w, res)
}

// runRequest is a single model call at a given temperature for the "Результаты"
// table in Task04.
type runRequest struct {
	Condition   string  `json:"condition"`
	Expected    string  `json:"expected"`
	Temperature float64 `json:"temperature"`
	Timeout     int     `json:"timeout"`
}

// runResult is one cell of the temperature table.
type runResult struct {
	Answer   string `json:"answer"`
	Success  bool   `json:"success"`
	Tokens   int    `json:"tokens"`
	Duration int64  `json:"duration"`
	Reason   string `json:"reason,omitempty"`
}

// handleRun makes one request to the model with the given temperature and judges
// whether the answer matches the expected one.
func (s *Server) handleRun(w http.ResponseWriter, r *http.Request) {
	var req runRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	req.Condition = strings.TrimSpace(req.Condition)
	req.Expected = strings.TrimSpace(req.Expected)
	if req.Condition == "" {
		http.Error(w, "condition is empty", http.StatusBadRequest)
		return
	}

	s.genMu.Lock()
	defer s.genMu.Unlock()

	model := s.currentModel()
	// Task04 uses a token limit of 15000 for the run table.
	maxTokens := s.defaultMaxTokens()
	if maxTokens > 15000 {
		maxTokens = 15000
	}

	// Default timeout 30s (clamp 1..60).
	timeoutSecs := req.Timeout
	if timeoutSecs == 0 {
		timeoutSecs = 30
	}
	if timeoutSecs < 1 {
		timeoutSecs = 1
	}
	if timeoutSecs > 60 {
		timeoutSecs = 60
	}
	ctx, cancel := context.WithTimeout(r.Context(), time.Duration(timeoutSecs)*time.Second)
	defer cancel()

	baseSystem := "Ты — помощник по решению задач. Отвечай кратко: назови только итоговый ответ."
	messages := []api.Message{api.SystemMessage(baseSystem), api.UserMessage(req.Condition)}

	opts := []api.Option{}
	// Передаём значение температуры всегда, когда клиент его прислал (в т.ч. 0),
	// чтобы temperature=0 применялась явно, а не подменялась дефолтом провайдера.
	opts = append(opts, api.Option{Temperature: &req.Temperature})

	start := time.Now()
	res := runResult{}
	answer, usage, err := s.clientFor(model).ChatWithUsage(ctx, model, messages, maxTokens, opts...)
	res.Duration = time.Since(start).Milliseconds()
	res.Success = answersMatch(answer, req.Expected)
	res.Tokens = usage.TotalTokens
	res.Answer = strings.TrimSpace(answer)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			res.Reason = fmt.Sprintf("Превышен таймаут %d с — остановлено", timeoutSecs)
			res.Success = false
			res.Answer = ""
		} else {
			res.Reason = err.Error()
		}
	}
	writeJSON(w, res)
}

// runConclusionRun is a single repeated run captured for the "Вывод" analysis.
type runConclusionRun struct {
	Temperature float64 `json:"temperature"`
	Answer      string  `json:"answer"`
	Success     bool    `json:"success"`
	Reason      string  `json:"reason"`
}

// runConclusionRequest collects the repeated runs of one task for analysis.
type runConclusionRequest struct {
	Condition string             `json:"condition"`
	Expected  string             `json:"expected"`
	Results   []runConclusionRun `json:"results"`
}

// generalTask is one task's full set of runs, passed to the overall analysis.
type generalTask struct {
	Title     string             `json:"title"`
	Condition string             `json:"condition"`
	Expected  string             `json:"expected"`
	Results   []runConclusionRun `json:"results"`
}

// generalConclusionRequest collects several tasks for the overall analysis.
type generalConclusionRequest struct {
	Tasks []generalTask `json:"tasks"`
}

// handleRunConclusion analyzes the repeated runs at different temperatures:
// how answers differ across runs at the same parameters, how often the format
// broke or the answer was wrong, and where variation helps or hurts.
func (s *Server) handleRunConclusion(w http.ResponseWriter, r *http.Request) {
	var req runConclusionRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 2<<20)).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if len(req.Results) == 0 {
		http.Error(w, "no results to analyze", http.StatusBadRequest)
		return
	}

	s.genMu.Lock()
	defer s.genMu.Unlock()

	model := s.currentModel()
	// Give the analysis enough budget so it isn't truncated.
	maxTokens := s.defaultMaxTokens()
	if maxTokens < 3000 {
		maxTokens = 3000
	}
	timeout := s.cfg.Timeout
	ctx, cancel := context.WithTimeout(r.Context(), timeout)
	defer cancel()

	var b strings.Builder
	b.WriteString("Проанализируй повторные запуски модели при разных температурах для одной и той же задачи.\n\n")
	b.WriteString("Условие задачи: " + strings.TrimSpace(req.Condition) + "\n")
	if strings.TrimSpace(req.Expected) != "" {
		b.WriteString("Ожидаемый ответ: " + strings.TrimSpace(req.Expected) + "\n")
	}
	b.WriteString("\nПолученные ответы по прогонам:\n")
	for i, res := range req.Results {
		fmt.Fprintf(&b, "%d. Температура: %.2f", i+1, res.Temperature)
		if res.Success {
			b.WriteString("; успех: да")
		} else {
			b.WriteString("; успех: нет")
		}
		if res.Reason != "" {
			fmt.Fprintf(&b, "; причина: %s", res.Reason)
		}
		if res.Answer != "" {
			fmt.Fprintf(&b, "; ответ: %s", strings.TrimSpace(res.Answer))
		}
		b.WriteString("\n")
	}
	b.WriteString("\nВ выводе ответь на следующие три вопроса:")
	b.WriteString("\n- Насколько ответы отличаются между прогонами при одинаковых параметрах.")
	b.WriteString("\n- Сколько раз сломался формат или ответ оказался неверным.")
	b.WriteString("\n- Где разнообразие полезно, а где вредит.")
	b.WriteString("\n\nОформи ответ компактно: короткие абзацы и маркированные списки.")

	prompt := b.String()
	messages := []api.Message{
		api.SystemMessage("Ты — аналитик. Проанализируй повторные запуски модели при разных температурах и сделай содержательный вывод."),
		api.UserMessage(prompt),
	}

	answer, _, err := s.clientFor(model).ChatWithUsage(ctx, model, messages, maxTokens)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, map[string]string{"analysis": strings.TrimSpace(answer)})
}

// yesNo renders a boolean as "да"/"нет" for the analysis prompts.
func yesNo(b bool) string {
	if b {
		return "да"
	}
	return "нет"
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

// conclusionRun is one solved method (with its reasoning level) fed into the
// "Вывод" analysis.
type conclusionRun struct {
	Name      string `json:"name"`
	Method    string `json:"method"`
	Expert    string `json:"expert"`
	Reasoning string `json:"reasoning"`
	Tokens    int    `json:"tokens"`
	Success   bool   `json:"success"`
	TimedOut  bool   `json:"timed_out"`
	Error     string `json:"error"`
	Solution  string `json:"solution"`
}

// conclusionRequest describes the "Вывод" analysis request.
type conclusionRequest struct {
	Condition string          `json:"condition"`
	Expected  string          `json:"expected"`
	Results   []conclusionRun `json:"results"`
}

// handleConclusion asks the model to analyze all collected solutions (across
// reasoning levels and methods) and produce a comparative conclusion.
func (s *Server) handleConclusion(w http.ResponseWriter, r *http.Request) {
	var req conclusionRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 2<<20)).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if len(req.Results) == 0 {
		http.Error(w, "no results to analyze", http.StatusBadRequest)
		return
	}

	s.genMu.Lock()
	defer s.genMu.Unlock()

	model := s.currentModel()
	// Вывод должен быть развёрнутым сравнительным анализом, поэтому даём ему
	// заметно больше токенов, чем обычному решению, чтобы он не обрезался.
	maxTokens := s.defaultMaxTokens()
	if maxTokens < 3000 {
		maxTokens = 3000
	}
	timeout := s.cfg.Timeout
	ctx, cancel := context.WithTimeout(r.Context(), timeout)
	defer cancel()

	var b strings.Builder
	b.WriteString("Сделай итоговый вывод по решению задачи на основе полученных результатов.\n\n")
	b.WriteString("Условие задачи: " + strings.TrimSpace(req.Condition) + "\n")
	if strings.TrimSpace(req.Expected) != "" {
		b.WriteString("Ожидаемый ответ: " + strings.TrimSpace(req.Expected) + "\n")
	}
	b.WriteString("\nПолученные результаты по способам решения и уровням рассуждения:\n")
	for i, res := range req.Results {
		fmt.Fprintf(&b, "%d. Способ: %s", i+1, res.Name)
		if res.Reasoning != "" {
			fmt.Fprintf(&b, " (рассуждение: %s)", res.Reasoning)
		}
		fmt.Fprintf(&b, "; токены: %d", res.Tokens)
		if res.TimedOut {
			b.WriteString("; таймаут")
		} else if res.Error != "" {
			fmt.Fprintf(&b, "; ошибка: %s", res.Error)
		} else if res.Success {
			b.WriteString("; успех: да")
		} else {
			b.WriteString("; успех: нет")
		}
		if res.Solution != "" {
			fmt.Fprintf(&b, "; решение: %s", strings.TrimSpace(res.Solution))
		}
		b.WriteString("\n")
	}
	b.WriteString("\nВ выводе обязательно отрази:")
	b.WriteString("\n1) Суммарные затраченные ресурсы по всем решениям и по каждому уровню рассуждения (общее и по типам).")
	b.WriteString("\n2) Сравнение способов решения друг с другом (а не только уровней рассуждения): Прямой ответ, Пошаговое решение, Через промпт, Эксперт — какой способ дал наиболее корректный, полный и качественный результат, сколько токенов и времени занял каждый, в чём их сильные и слабые стороны для данной задачи.")
	b.WriteString("\n3) Оправданность/неоправданность этих затрат относительно итогового результата: окупились ли вложенные токены — дали ли большие затраты более верный или более качественный ответ, или были избыточны.")
	b.WriteString("\n4) Рекомендованный итоговый способ и уровень рассуждения для этой конкретной задачи, с обоснованием выбора.")
	b.WriteString("\n\nОформи ответ компактно: короткие абзацы и нумерованные/маркированные списки, без широких таблиц.")

	prompt := b.String()
	messages := []api.Message{
		api.SystemMessage("Ты — аналитик. На основе результатов решения задачи сделай содержательный итоговый вывод, опираясь на затраченные ресурсы и их оправданность."),
		api.UserMessage(prompt),
	}

	answer, _, err := s.clientFor(model).ChatWithUsage(ctx, model, messages, maxTokens)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, map[string]string{"analysis": strings.TrimSpace(answer)})
}

// handleGeneralConclusion analyzes all runs across several tasks and answers:
// (1) which setting suits which task, and (2) whether temperature = 0 gives the
// same answer every time.
func (s *Server) handleGeneralConclusion(w http.ResponseWriter, r *http.Request) {
	var req generalConclusionRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 4<<20)).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if len(req.Tasks) == 0 {
		http.Error(w, "no tasks to analyze", http.StatusBadRequest)
		return
	}

	s.genMu.Lock()
	defer s.genMu.Unlock()

	model := s.currentModel()
	maxTokens := s.defaultMaxTokens()
	if maxTokens < 3000 {
		maxTokens = 3000
	}
	timeout := s.cfg.Timeout
	ctx, cancel := context.WithTimeout(r.Context(), timeout)
	defer cancel()

	var b strings.Builder
	b.WriteString("Проанализируй все ответы, полученные при запуске нескольких задач на разных температурах, и сделай общий вывод.\n\n")
	for _, t := range req.Tasks {
		fmt.Fprintf(&b, "== %s ==\n", t.Title)
		b.WriteString("Условие: " + strings.TrimSpace(t.Condition) + "\n")
		if strings.TrimSpace(t.Expected) != "" {
			b.WriteString("Ожидаемый ответ: " + strings.TrimSpace(t.Expected) + "\n")
		}
		b.WriteString("Ответы по прогонам:\n")
		for i, res := range t.Results {
			fmt.Fprintf(&b, "%d. Температура: %.2f; успех: %s", i+1, res.Temperature, yesNo(res.Success))
			if res.Reason != "" {
				fmt.Fprintf(&b, "; причина: %s", res.Reason)
			}
			if res.Answer != "" {
				fmt.Fprintf(&b, "; ответ: %s", strings.TrimSpace(res.Answer))
			}
			b.WriteString("\n")
		}
		b.WriteString("\n")
	}

	b.WriteString("Сделай общий вывод и обязательно ответь на два вопроса:\n")
	b.WriteString("1. Для каких задач лучше подходит каждая из настроек (температура и т.п.), а для каких — нет. Сопоставь задачи и температуры, объясни выбор.\n")
	b.WriteString("2. Даёт ли temperature = 0 одинаковый ответ каждый раз? Опиши, насколько стабильны ответы при temperature = 0 и как они соотносятся с другими температурами.\n")
	b.WriteString("Если ответов мало или данных не хватает, так и напиши — не придумывай.\n\n")
	b.WriteString("Оформи ответ компактно: короткие абзацы и нумерованные/маркированные списки.")

	prompt := b.String()
	messages := []api.Message{
		api.SystemMessage("Ты — аналитик. Сопоставь результаты запусков модели, сравни настройки и сделай объективный общий вывод."),
		api.UserMessage(prompt),
	}

	answer, _, err := s.clientFor(model).ChatWithUsage(ctx, model, messages, maxTokens)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, map[string]string{"analysis": strings.TrimSpace(answer)})
}
