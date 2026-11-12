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
	"sort"
	"strings"
	"sync"
	"time"

	testassets "ai-chat-go"
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

	// historyTokens is the size of the dialog history: the first request's
	// prompt tokens (whole history plus the new user message) plus the final
	// answer's completion tokens. Intermediate tool exchanges are transient
	// machinery and are not counted. It is persisted in the per-model history
	// file and restored on start.
	historyTokens int

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
// is disabled (empty historyFile). The context with the newest activity is
// chosen as the active model. The system message is re-seeded from the current
// configuration rather than stored.
func (s *Server) restore() {
	if s.historyFile == "" {
		return
	}
	s.migrateLegacy()

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
	s.historyTokens = st.HistoryTokens
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
	historyTokens := s.historyTokens
	path := s.contextFileFor(model)
	s.mu.Unlock()
	if path == "" {
		return
	}
	if err := saveState(path, model, msgs, historyTokens); err != nil {
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
	mux.HandleFunc("POST /api/model", s.handleModel)
	mux.HandleFunc("POST /api/reset", s.handleReset)
	mux.HandleFunc("GET /api/debug", s.handleDebug)
	mux.HandleFunc("GET /api/quiz", s.handleQuiz)
	mux.HandleFunc("POST /api/dialog", s.handleDialog)
	mux.HandleFunc("POST /api/compress", s.handleCompress)
	return mux
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(indexHTML)
}

// preset combines a preset dialog (testdata/dialogs) with an optional quiz
// (testdata/quizzes). Only the dialog turns are loaded into the context; the
// quiz is served to the UI "+" menu (GET /api/quiz) so its questions can be
// sent as regular messages on demand.
type preset struct {
	dialog string
	quiz   string
}

// presetDialogs maps preset dialog names to their embedded texts. Each
// chat-scenario blob in the UI ("Диалог про шторы"…) loads one of these as
// the whole conversation context.
var presetDialogs = map[string]preset{
	"curtains-kindergarten": {dialog: testassets.DialogCurtains, quiz: testassets.QuizCurtains},
}

// parseDialog turns a preset-dialog text into conversation turns. A line
// starting with "User:" opens a user turn, "Model:" — an assistant turn;
// any further non-empty line continues the turn opened most recently.
func parseDialog(text string) []api.Message {
	var out []api.Message
	var role, body string
	flush := func() {
		if body == "" {
			return
		}
		if role == "user" {
			out = append(out, api.UserMessage(body))
		} else {
			out = append(out, api.AssistantMessage(body))
		}
	}
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimRight(line, "\r")
		switch {
		case strings.HasPrefix(line, "User:"):
			flush()
			role, body = "user", strings.TrimSpace(strings.TrimPrefix(line, "User:"))
		case strings.HasPrefix(line, "Model:"):
			flush()
			role, body = "assistant", strings.TrimSpace(strings.TrimPrefix(line, "Model:"))
		default:
			if strings.TrimSpace(line) == "" {
				continue
			}
			if body == "" {
				role, body = "assistant", strings.TrimSpace(line)
			} else {
				body += "\n" + strings.TrimSpace(line)
			}
		}
	}
	flush()
	return out
}

// parseQuiz turns a quiz text ("Вопрос:" / "Ответ:" pairs, one per line) into
// alternating messages: a question becomes a user turn, an answer — an
// assistant turn; any further non-empty line continues the turn opened most
// recently. The "Вопрос:"/"Ответ:" prefixes are stripped from the content.
func parseQuiz(text string) []api.Message {
	var out []api.Message
	var role, body string
	flush := func() {
		if body == "" {
			return
		}
		if role == "user" {
			out = append(out, api.UserMessage(body))
		} else {
			out = append(out, api.AssistantMessage(body))
		}
	}
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimRight(line, "\r")
		switch {
		case strings.HasPrefix(line, "Вопрос:"):
			flush()
			role, body = "user", strings.TrimSpace(strings.TrimPrefix(line, "Вопрос:"))
		case strings.HasPrefix(line, "Ответ:"):
			flush()
			role, body = "assistant", strings.TrimSpace(strings.TrimPrefix(line, "Ответ:"))
		default:
			if strings.TrimSpace(line) == "" {
				continue
			}
			if body == "" {
				role, body = "user", strings.TrimSpace(line)
			} else {
				body += "\n" + strings.TrimSpace(line)
			}
		}
	}
	flush()
	return out
}

// quizPair is one question/answer pair of a preset quiz, served to the UI
// "+" menu by GET /api/quiz. The answer never goes to the model: it is shown
// in the chat as an expected-answer note next to the sent question.
type quizPair struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

// quizPairs extracts the question/answer pairs from every preset quiz, in
// preset-name order for a deterministic listing.
func quizPairs() []quizPair {
	names := make([]string, 0, len(presetDialogs))
	for name := range presetDialogs {
		names = append(names, name)
	}
	sort.Strings(names)
	var out []quizPair
	for _, name := range names {
		msgs := parseQuiz(presetDialogs[name].quiz)
		for i := 0; i+1 < len(msgs); i += 2 {
			if msgs[i].Role == "user" && msgs[i+1].Role == "assistant" {
				out = append(out, quizPair{Question: msgs[i].Content, Answer: msgs[i+1].Content})
			}
		}
	}
	return out
}

// handleQuiz serves the questions of the preset quizzes for the "+" menu.
func (s *Server) handleQuiz(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]any{"questions": quizPairs()})
}

// handleDialog loads a preset dialog from testdata as the current conversation
// context: previous turns are replaced, the system prompt is kept, the result
// is persisted like a normal dialog and returned in the /api/history shape so
// the UI can render every message in the chat log.
func (s *Server) handleDialog(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name string `json:"name"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "Некорректное тело запроса (ожидается JSON): "+err.Error(), http.StatusBadRequest)
		return
	}
	req.Name = strings.TrimSpace(req.Name)
	p, ok := presetDialogs[req.Name]
	if !ok {
		http.Error(w, "Неизвестный сценарий диалога: "+req.Name, http.StatusNotFound)
		return
	}
	msgs := parseDialog(p.dialog)
	// Викторина сценария в контекст не добавляется: её вопросы отправляются
	// только по выбору из меню «+» (GET /api/quiz для списка).
	if len(msgs) == 0 {
		http.Error(w, "Сценарий диалога пуст: "+req.Name, http.StatusInternalServerError)
		return
	}

	now := time.Now()
	s.mu.Lock()
	s.hist = make([]histEntry, 0, len(msgs)+1)
	if s.cfg.System != "" {
		s.hist = append(s.hist, histEntry{At: now, Msg: api.SystemMessage(s.cfg.System)})
	}
	for _, m := range msgs {
		s.hist = append(s.hist, histEntry{At: now, Msg: m})
	}
	s.historyTokens = 0
	s.mu.Unlock()
	s.persist()
	log.Printf("chat: preset dialog %q loaded (%d turns)", req.Name, len(msgs))

	out := make([]historyMessage, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, historyMessage{Role: m.Role, Content: m.Content, At: now})
	}
	writeJSON(w, map[string]any{"messages": out})
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
	oldHistoryTokens := s.historyTokens
	var oldMsgs []chatRecord
	for _, e := range s.hist {
		if e.Msg.Role != "system" {
			oldMsgs = append(oldMsgs, recordOf(e.Msg, e.At))
		}
	}

	s.model = req.Model
	s.hist = make([]histEntry, 0, 16)
	s.historyTokens = 0
	if s.cfg.System != "" {
		s.hist = append(s.hist, histEntry{At: time.Now(), Msg: api.SystemMessage(s.cfg.System)})
	}
	newPath := s.contextFileFor(req.Model)
	s.mu.Unlock()

	if oldPath != "" {
		if err := saveState(oldPath, oldModel, oldMsgs, oldHistoryTokens); err != nil {
			log.Printf("history: failed to save %s: %v", oldPath, err)
		}
	}
	if newPath != "" {
		if st, err := loadState(newPath); err == nil && st != nil {
			s.mu.Lock()
			s.historyTokens = st.HistoryTokens
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
	historyTokens := s.historyTokens
	s.mu.Unlock()

	out := map[string]any{
		"request":           request,
		"response":          response,
		"steps":             steps,
		"tokens":            tokens,
		"prompt_tokens":     prompt,
		"completion_tokens": completion,
		"time_ms":           timeMS,
		"history_tokens":    historyTokens,
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
	s.historyTokens = 0
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
	// Large pastes are legitimate: a 300k-token text JSON-escapes to several
	// megabytes, so allow up to 64 MiB and report a clear error beyond that.
	const maxBody = 64 << 20
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBody+1))
	if err != nil {
		s.setDetail(map[string]any{
			"type":    "bad_request",
			"summary": "Не удалось прочитать запрос",
			"detail":  "Тело POST /api/chat не прочитано: " + err.Error(),
		})
		http.Error(w, "Не удалось прочитать тело запроса: "+err.Error(), http.StatusBadRequest)
		return
	}
	if len(body) > maxBody {
		s.setDetail(map[string]any{
			"type":    "request_too_large",
			"summary": "Сообщение слишком большое",
			"detail":  fmt.Sprintf("Тело запроса %d МБ превышает лимит 64 МБ; сократите вставленный текст.", len(body)>>20),
		})
		http.Error(w, "Сообщение слишком большое: тело запроса превышает лимит 64 МБ", http.StatusRequestEntityTooLarge)
		return
	}
	var req struct {
		Message   string `json:"message"`
		Format    string `json:"format"`
		MaxTokens int    `json:"maxTokens"`
		Timeout   int    `json:"timeout"` // допустимый таймаут ответа, секунды (5..30)
		MaxRounds int    `json:"maxRounds"`
		Schema    string `json:"schema"`
		Stop      string `json:"stop"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		s.setDetail(map[string]any{
			"type":    "bad_request",
			"summary": "Некорректное тело запроса",
			"detail":  "POST /api/chat не разобран как JSON: " + err.Error(),
		})
		http.Error(w, "Некорректное тело запроса (ожидается JSON): "+err.Error(), http.StatusBadRequest)
		return
	}
	req.Message = strings.TrimSpace(req.Message)
	if req.Message == "" {
		http.Error(w, "Сообщение пустое", http.StatusBadRequest)
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
		msg := e.Msg
		if msg.Role == "summary" {
			// Сводка сжатого контекста уходит модели как системное сообщение.
			msg = api.Message{Role: "system", Content: msg.Content}
		}
		messages = append(messages, msg)
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
	// firstRound is the usage of the very first request (whole history plus the
	// new user message); lastRound that of the final one. The history size
	// after the exchange is firstRound.PromptTokens + lastRound.CompletionTokens:
	// intermediate tool exchanges are transient machinery, not part of history.
	var firstRound, lastRound api.Usage
	var roundUsage []api.Usage
	isFirstRound := true
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
		if isFirstRound {
			firstRound = u
			isFirstRound = false
		}
		lastRound = u
		roundUsage = append(roundUsage, u)
		return content, calls, err
	}
	reply, steps, added, err := tools.RunLoop(ctx, round, messages, maxRounds)
	// Attach each round's usage to its "Раунд" step so the "Шаги" panel shows
	// per-call prompt/completion tokens; tool-execution steps stay tokenless.
	ri := 0
	for i := range steps {
		if steps[i].Tool == "" && steps[i].Text != "" {
			if ri < len(roundUsage) {
				steps[i].PromptTokens = roundUsage[ri].PromptTokens
				steps[i].CompletionTokens = roundUsage[ri].CompletionTokens
			}
			ri++
		}
	}
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
		// History tokens: the first request carried the whole history plus the
		// new user message, and the final answer joins it. Intermediate tool
		// exchanges are transient and do not count toward the history.
		prompt := firstRound.PromptTokens
		if prompt == 0 {
			prompt = lastRound.PromptTokens
		}
		completion := lastRound.CompletionTokens
		if completion == 0 {
			completion = firstRound.CompletionTokens
		}
		s.historyTokens = prompt + completion
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

// compressMaxTokens caps the summary generation during context compression.
const compressMaxTokens = 2000

// compressPrompt instructs the model how to summarize the outgoing part of
// the conversation during context compression.
const compressPrompt = "Ты помогаешь сжать историю диалога. Ниже идёт переписка, которую заменяют краткой сводкой. Составь сводку: сохрани ключевые факты, имена, цифры, договорённости и открытые вопросы. Пиши сжато, без приветствий и комментариев — только текст сводки."

// clampKeep keeps the compression window (how many last messages survive the
// compression) within the 3..30 range of the UI slider; non-positive yields
// the default of 10.
func clampKeep(v int) int {
	switch {
	case v <= 0:
		return 10
	case v < 3:
		return 3
	case v > 30:
		return 30
	default:
		return v
	}
}

// handleCompress replaces all conversation turns older than the last "keep"
// messages (the "Размер контекста" slider) with a single model-generated
// summary. The summary is stored as a "summary" role entry: the UI renders it
// as a compression marker at the top of the chat, persistence keeps it across
// restarts, and the model receives it as a system message. The response also
// carries the raw model exchange for the "Сжатие" block of the details panel.
func (s *Server) handleCompress(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Keep int `json:"keep"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "Некорректное тело запроса (ожидается JSON): "+err.Error(), http.StatusBadRequest)
		return
	}
	keep := clampKeep(req.Keep)

	// Serialize with chat generations: the history must not change mid-cut.
	s.genMu.Lock()
	defer s.genMu.Unlock()

	model := s.currentModel()

	// turns lists the displayed conversation turns (user turns and assistant
	// answers). Previous summaries do not count toward the keep window, so a
	// window that is already full cannot be compressed "again": repeated
	// compress with the same setting yields an informational result, not a
	// re-summarization of the summary itself.
	s.mu.Lock()
	var turns []int
	for i, e := range s.hist {
		switch e.Msg.Role {
		case "user":
			turns = append(turns, i)
		case "assistant":
			if strings.TrimSpace(e.Msg.Content) != "" {
				turns = append(turns, i)
			}
		}
	}
	if len(turns) <= keep {
		s.mu.Unlock()
		// Нет сообщений для сжатия — это не ошибка, а нормальный результат.
		writeJSON(w, map[string]any{
			"ok":         true,
			"compressed": 0,
			"kept":       len(turns),
			"message":    fmt.Sprintf("Сжимать нечего: сообщений в контексте (%d) не больше размера контекста (%d).", len(turns), keep),
		})
		return
	}
	cut := turns[len(turns)-keep] // index of the first kept turn
	compressed := 0
	for _, i := range turns {
		if i < cut {
			compressed++
		}
	}
	old := make([]api.Message, 0, cut)
	for i, e := range s.hist {
		// The leading system prompt (when present) and the kept tail are not
		// summarized; a previous summary is re-summarized together with the
		// turns it replaces.
		if i >= cut || e.Msg.Role == "system" {
			continue
		}
		switch e.Msg.Role {
		case "user":
			old = append(old, e.Msg)
		case "assistant":
			if strings.TrimSpace(e.Msg.Content) != "" {
				old = append(old, e.Msg)
			}
		case "summary":
			old = append(old, api.SystemMessage(e.Msg.Content))
		}
	}
	s.mu.Unlock()

	if len(old) == 0 {
		s.mu.Unlock()
		// Сжимаемый набор пуст — тоже нормальный результат, не ошибка.
		writeJSON(w, map[string]any{
			"ok":         true,
			"compressed": 0,
			"kept":       len(turns),
			"message":    "Сжимать нечего: старых сообщений нет.",
		})
		return
	}

	// Ask the model for a compact summary of the outgoing turns.
	sumReq := make([]api.Message, 0, len(old)+2)
	sumReq = append(sumReq, api.SystemMessage(compressPrompt))
	sumReq = append(sumReq, old...)
	sumReq = append(sumReq, api.UserMessage("Составь краткую сводку переписки выше."))

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout)
	defer cancel()

	start := time.Now()
	var summary strings.Builder
	_, err := s.clientFor(model).StreamChatWithActivity(ctx, model, sumReq, compressMaxTokens,
		func(delta string) { summary.WriteString(delta) }, func() {})
	timeMS := time.Since(start).Milliseconds()
	if err != nil {
		log.Printf("chat: compression failed: %v", err)
		http.Error(w, "Сжатие не удалось: "+err.Error(), http.StatusBadGateway)
		return
	}
	sumText := strings.TrimSpace(summary.String())
	if sumText == "" {
		http.Error(w, "Сжатие не удалось: модель вернула пустую сводку.", http.StatusBadGateway)
		return
	}
	// The raw model exchange goes to the "Сжатие" block of the details panel.
	request, response := s.clientFor(model).LastExchange()
	usage := s.clientFor(model).LastUsage()

	// Splice the history: system prompt + summary + the kept tail. Intermediate
	// tool exchanges before the cut disappear together with their turns.
	now := time.Now()
	s.mu.Lock()
	newHist := make([]histEntry, 0, len(s.hist)-cut+1)
	if len(s.hist) > 0 && s.hist[0].Msg.Role == "system" {
		newHist = append(newHist, s.hist[0])
	}
	newHist = append(newHist, histEntry{At: now, Msg: api.Message{Role: "summary", Content: sumText}})
	newHist = append(newHist, s.hist[cut:]...)
	s.hist = newHist
	// Токены истории переоцениваются после следующего обмена с моделью.
	s.historyTokens = 0
	s.mu.Unlock()
	s.persist()

	s.setTrace(nil, usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens, timeMS)
	log.Printf("chat: context compressed: %d turns -> summary (%d tokens), %d kept", compressed, usage.TotalTokens, keep)

	writeJSON(w, map[string]any{
		"ok":                true,
		"compressed":        compressed,
		"kept":              keep,
		"summary":           sumText,
		"request":           request,
		"response":          response,
		"prompt_tokens":     usage.PromptTokens,
		"completion_tokens": usage.CompletionTokens,
		"total_tokens":      usage.TotalTokens,
		"time_ms":           timeMS,
	})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}
