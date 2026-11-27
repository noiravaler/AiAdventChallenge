// Package mcpclient is a minimal MCP (Model Context Protocol) client hub: it
// connects to several remote MCP servers over Streamable HTTP, lists their
// tools and proxies tool calls to them. The MCP tools are bridged into the
// chat's function-calling loop, so the model sees them as ordinary
// OpenAI-compatible function tools.
//
// The hub is fault-tolerant: if a server is unreachable at startup the chat
// keeps working, a background loop retries the connection, and every call is
// bounded by a timeout so a hanging server cannot stall the chat. Tool names
// that occur on more than one server are exposed to the model with a
// "slug__name" prefix, so routing is always unambiguous.
package mcpclient

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
	"time"

	"ai-chat-go/api"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const (
	// connectTimeout bounds the dial + initialize handshake + tool list.
	connectTimeout = 15 * time.Second
	// RetryInterval is the default period between background reconnect tries.
	RetryInterval = time.Minute
	// failFastInterval: after a failed dial, Execute fails fast (instead of
	// dialing again) for this long; manual reconnect bypasses it.
	failFastInterval = 30 * time.Second
)

// ServerDef identifies one MCP endpoint in the configuration.
type ServerDef struct {
	Slug string // латинский идентификатор (префикс при коллизиях имён)
	Name string // отображаемое название
	URL  string // endpoint (Streamable HTTP)
}

// ToolInfo describes a single MCP tool for the UI (GET /api/mcp).
type ToolInfo struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// ServerInfo is the per-server connection status served at GET /api/mcp.
type ServerInfo struct {
	Slug      string     `json:"slug"`
	Name      string     `json:"name"`
	URL       string     `json:"url"`
	Enabled   bool       `json:"enabled"`
	Connected bool       `json:"connected"`
	Stopped   bool       `json:"stopped"` // остановлен кнопкой «стоп» (симуляция отказа сервиса)
	Server    string     `json:"server"`
	Version   string     `json:"version"`
	Protocol  string     `json:"protocol"`
	Count     int        `json:"count"`
	Tools     []ToolInfo `json:"tools"`
	Error     string     `json:"error,omitempty"`
}

// client is the live state of one connected MCP server.
type client struct {
	def           ServerDef
	session       *mcp.ClientSession
	name          string
	version       string
	proto         string
	tools         []api.Tool // оригинальные имена (для UI)
	infos         []ToolInfo
	exposedByOrig map[string]string // orig → имя, выданное модели
	lastFail      time.Time
	lastErr       string
	stopped       bool // «стоп» кнопкой: процесс убит, но инструменты передаются модели
}

// Hub manages several MCP endpoints: sessions, tool lists (with
// collision-safe naming), tool-call routing and background reconnection.
// All methods are safe for concurrent use.
type Hub struct {
	timeout time.Duration
	onTools func([]api.Tool)

	stopCh   chan struct{}
	stopOnce sync.Once

	mu        sync.Mutex
	clients   map[string]*client // по slug
	enabled   map[string]bool    // тумблер вкл/выкл на каждый сервер
	exposed   []api.Tool         // объединённый список для модели
	byExposed map[string]struct {
		slug string
		orig string
	}
}

// NewHub creates a hub for the given endpoints without connecting yet.
func NewHub(defs []ServerDef, timeout time.Duration) *Hub {
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	h := &Hub{
		timeout: timeout,
		stopCh:  make(chan struct{}),
		clients: make(map[string]*client, len(defs)),
		enabled: make(map[string]bool, len(defs)),
		byExposed: make(map[string]struct {
			slug string
			orig string
		}),
	}
	for _, d := range defs {
		h.clients[d.Slug] = &client{def: d, exposedByOrig: map[string]string{}}
		h.enabled[d.Slug] = true
	}
	return h
}

// SetOnToolsChange installs a callback fired after every tool-list rebuild
// with the merged function definitions (used to update the chat tools).
func (h *Hub) SetOnToolsChange(fn func([]api.Tool)) {
	h.mu.Lock()
	h.onTools = fn
	h.mu.Unlock()
}

// DialAll connects to every configured server in parallel and rebuilds the
// tool list. It returns per-slug errors; servers that failed stay
// disconnected and the background loop keeps retrying them.
func (h *Hub) DialAll(ctx context.Context) []struct {
	Slug string
	Err  error
} {
	type res struct {
		slug string
		err  error
	}
	h.mu.Lock()
	slugs := make([]string, 0, len(h.clients))
	for slug := range h.clients {
		slugs = append(slugs, slug)
	}
	h.mu.Unlock()
	sort.Strings(slugs)

	out := make(chan res, len(slugs))
	var wg sync.WaitGroup
	for _, slug := range slugs {
		wg.Add(1)
		go func(slug string) {
			defer wg.Done()
			out <- res{slug, h.Dial(ctx, slug)}
		}(slug)
	}
	wg.Wait()
	close(out)

	var errs []struct {
		Slug string
		Err  error
	}
	for r := range out {
		if r.err != nil {
			errs = append(errs, struct {
				Slug string
				Err  error
			}{r.slug, r.err})
		}
	}
	return errs
}

// Dial connects (or reconnects) one server by slug, refreshes its tool list
// and rebuilds the merged model-facing definitions. It bypasses the fail-fast
// backoff, so it doubles as the manual "reconnect" action. Any previous
// session of that server is closed — no connection is left hanging.
func (h *Hub) Dial(ctx context.Context, slug string) error {
	h.mu.Lock()
	cl, ok := h.clients[slug]
	h.mu.Unlock()
	if !ok {
		return fmt.Errorf("неизвестный MCP-сервер: %s", slug)
	}

	dialCtx, cancel := context.WithTimeout(ctx, connectTimeout)
	defer cancel()

	mc := mcp.NewClient(&mcp.Implementation{Name: "ai-chat", Version: "0.1.0"}, nil)
	transport := &mcp.StreamableClientTransport{
		Endpoint: cl.def.URL,
		// The chat only needs request/response exchanges; the standalone SSE
		// stream is optional per the MCP spec and not every server handles it.
		DisableStandaloneSSE: true,
		// A retry storm against an unreachable server only prolongs the hang;
		// one reconnect attempt is enough, the rest is up to the caller.
		MaxRetries: 1,
	}
	session, err := mc.Connect(dialCtx, transport, nil)
	if err != nil {
		h.failLocked(slug, cl, err)
		return err
	}

	tools, infos, name, version, proto, lerr := describe(dialCtx, session)
	if lerr != nil {
		_ = session.Close()
		h.failLocked(slug, cl, lerr)
		return lerr
	}

	h.mu.Lock()
	old := cl.session
	cl.session = session
	cl.stopped = false // успешное соединение снимает состояние «остановлен кнопкой»
	cl.name, cl.version, cl.proto = name, version, proto
	cl.tools, cl.infos = tools, infos
	cl.lastErr = ""
	h.rebuildLocked()
	onTools := h.onTools
	merged := append([]api.Tool(nil), h.exposed...)
	h.mu.Unlock()
	if old != nil {
		go func() { _ = old.Close() }() // best-effort, DELETE is capped at 5s
	}
	if onTools != nil {
		onTools(merged)
	}
	return nil
}

// failLocked records a failed dial: it stamps the error (for fail-fast and
// the UI) and rebuilds the model-facing tool list so a disconnected or
// switched-off server stops contributing tools to the model. The last known
// tool list stays on the client for the UI, but is not exposed.
func (h *Hub) failLocked(slug string, cl *client, err error) {
	h.mu.Lock()
	cl.lastFail, cl.lastErr = time.Now(), err.Error()
	h.rebuildLocked()
	onTools := h.onTools
	merged := append([]api.Tool(nil), h.exposed...)
	h.mu.Unlock()
	if onTools != nil {
		onTools(merged)
	}
}

// SetEnabled switches one MCP server on or off (the sidebar toggle). Disabled
// servers are disconnected immediately, their tools leave the model-facing
// list and the background loop stops retrying them. Enabling dials the server
// again, bounded by connectTimeout; on success the tools rejoin the model list.
func (h *Hub) SetEnabled(ctx context.Context, slug string, on bool) error {
	h.mu.Lock()
	cl, ok := h.clients[slug]
	if ok {
		h.enabled[slug] = on
	}
	h.mu.Unlock()
	if !ok {
		return fmt.Errorf("неизвестный MCP-сервер: %s", slug)
	}
	if on {
		return h.Dial(ctx, slug)
	}
	h.mu.Lock()
	old := cl.session
	cl.session = nil
	cl.lastErr = "" // не ошибка: выключено пользователем, UI покажет «выключен»
	h.rebuildLocked()
	onTools := h.onTools
	merged := append([]api.Tool(nil), h.exposed...)
	h.mu.Unlock()
	if old != nil {
		go func() { _ = old.Close() }()
	}
	if onTools != nil {
		onTools(merged)
	}
	return nil
}

// Stop "роняет" сервер кнопкой (симуляция отказа сервиса): MCP-процесс
// останавливает вызывающий код (mcpboot), здесь мы закрываем сессию, но
// инструменты остаются в списке, передаваемом модели — модель должна видеть
// инструменты и получать ошибку при вызове, как при реальном сбое сервиса.
// Отличается от тумблера: тот убирает инструменты из списка целиком.
func (h *Hub) Stop(slug string) error {
	h.mu.Lock()
	cl, ok := h.clients[slug]
	if ok {
		cl.stopped = true
	}
	h.mu.Unlock()
	if !ok {
		return fmt.Errorf("неизвестный MCP-сервер: %s", slug)
	}
	h.mu.Lock()
	old := cl.session
	cl.session = nil
	cl.lastErr = "" // не ошибка: остановлено пользователем, UI покажет «остановлен»
	h.rebuildLocked()
	onTools := h.onTools
	merged := append([]api.Tool(nil), h.exposed...)
	h.mu.Unlock()
	if old != nil {
		go func() { _ = old.Close() }()
	}
	if onTools != nil {
		onTools(merged)
	}
	return nil
}

// Resume снимает состояние «остановлен кнопкой» (кнопка «поднять»); соединение
// восстанавливает вызывающий код через Dial.
func (h *Hub) Resume(slug string) error {
	h.mu.Lock()
	cl, ok := h.clients[slug]
	if ok {
		cl.stopped = false
	}
	h.mu.Unlock()
	if !ok {
		return fmt.Errorf("неизвестный MCP-сервер: %s", slug)
	}
	return nil
}

// rebuildLocked recomputes the merged model-facing tool list. Only enabled
// servers with a live session contribute: a server that is switched off or
// disconnected (down, dial failed, toggled off) must not pass its tools to
// the model. The one exception is the "stopped" state (process killed via the
// stop button): its tools stay exposed so the model faces the service failure
// on call. Tool names that occur on more than one contributing server are
// prefixed "slug__name"; unique names keep their original form. Caller must
// hold h.mu.
func (h *Hub) rebuildLocked() {
	active := func(slug string, cl *client) bool {
		return h.enabled[slug] && (cl.session != nil || cl.stopped)
	}
	occurrences := map[string]int{}
	for slug, cl := range h.clients {
		if !active(slug, cl) {
			continue
		}
		for _, t := range cl.tools {
			occurrences[t.Function.Name]++
		}
	}
	h.exposed = h.exposed[:0]
	h.byExposed = make(map[string]struct {
		slug string
		orig string
	})
	// Stable order: the hub's client map is iterated by slug for determinism.
	slugs := make([]string, 0, len(h.clients))
	for slug := range h.clients {
		slugs = append(slugs, slug)
	}
	sort.Strings(slugs)
	for _, slug := range slugs {
		cl := h.clients[slug]
		cl.exposedByOrig = make(map[string]string, len(cl.tools))
		if !active(slug, cl) {
			continue
		}
		for _, t := range cl.tools {
			orig := t.Function.Name
			exposed := orig
			if occurrences[orig] > 1 {
				exposed = slug + "__" + orig
			}
			cl.exposedByOrig[orig] = exposed
			t.Function.Name = exposed
			h.exposed = append(h.exposed, t)
			h.byExposed[exposed] = struct {
				slug string
				orig string
			}{slug, orig}
		}
	}
}

// ensure makes sure the given server has a live session before a tool call:
// it dials on demand, failing fast while the server has recently proven
// unreachable (the background loop keeps trying; manual reconnect bypasses).
func (h *Hub) ensure(slug string) error {
	h.mu.Lock()
	cl := h.clients[slug]
	var lastErr string
	sinceFail := time.Duration(0)
	if cl != nil {
		lastErr, sinceFail = cl.lastErr, time.Since(cl.lastFail)
	}
	h.mu.Unlock()
	if cl == nil {
		return fmt.Errorf("неизвестный MCP-сервер: %s", slug)
	}
	h.mu.Lock()
	on := h.enabled[slug]
	h.mu.Unlock()
	if !on {
		return fmt.Errorf("MCP-сервер выключен тумблером")
	}
	h.mu.Lock()
	stopped := cl.stopped
	h.mu.Unlock()
	if stopped {
		return fmt.Errorf("MCP-сервер остановлен кнопкой «стоп» (симуляция отказа сервиса) — вызов невозможен")
	}
	if cl.session != nil {
		return nil
	}
	if sinceFail < failFastInterval {
		if lastErr != "" {
			return fmt.Errorf("недоступен (%s); подождите или нажмите «Переподключить»", lastErr)
		}
		return errors.New("нет соединения")
	}
	return h.Dial(context.Background(), slug)
}

// Execute proxies a tool call to the MCP server that owns the exposed name.
// It matches the tools.ExternalFunc signature: result text, source label
// ("mcp:<slug>"), latency in ms. Every attempt is bounded by the client
// timeout, so an unreachable or hanging server stalls the chat at most once
// per call.
func (h *Hub) Execute(ctx context.Context, name string, args json.RawMessage) (string, string, int64, []string, error) {
	h.mu.Lock()
	route, ok := h.byExposed[name]
	h.mu.Unlock()
	if !ok {
		return "", "mcp", 0, nil, fmt.Errorf("инструмент %s не относится ни к одному MCP-серверу", name)
	}
	slug, orig := route.slug, route.orig

	if err := h.ensure(slug); err != nil {
		return "", "mcp:" + slug, 0, nil, fmt.Errorf("MCP[%s] %s: %w", slug, orig, err)
	}

	var arguments any
	if len(args) > 0 {
		var m map[string]any
		if err := json.Unmarshal(args, &m); err == nil {
			arguments = m
		} else {
			arguments = string(args)
		}
	}

	start := time.Now()
	res, err := h.call(ctx, slug, orig, arguments)
	ms := time.Since(start).Milliseconds()
	if err != nil {
		// The session may have gone stale (server restart); dial once and
		// retry — but only for quick failures: after a timeout the server is
		// unreachable and a reconnect would just double the wait.
		if !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled) {
			if derr := h.Dial(ctx, slug); derr == nil {
				if res2, err2 := h.call(ctx, slug, orig, arguments); err2 == nil {
					res, err, ms = res2, nil, time.Since(start).Milliseconds()
				}
			}
		}
	}
	if err != nil {
		return "", "mcp:" + slug, ms, nil, fmt.Errorf("MCP[%s] %s: %w", slug, orig, err)
	}
	text := contentText(res)
	if res.IsError {
		return text, "mcp:" + slug, ms, nil, errors.New(text)
	}
	// Изображения из результата (например, chart_render) идут пользователю:
	// сохраняем и отдаём веб-пути для показа в чате, в модель они не попадают.
	return text, "mcp:" + slug, ms, collectImages(res), nil
}

// call performs one CallTool request bounded by the client timeout.
func (h *Hub) call(ctx context.Context, slug, orig string, arguments any) (*mcp.CallToolResult, error) {
	h.mu.Lock()
	cl := h.clients[slug]
	h.mu.Unlock()
	if cl == nil || cl.session == nil {
		return nil, errors.New("сессия MCP закрыта")
	}
	callCtx, cancel := context.WithTimeout(ctx, h.timeout)
	defer cancel()
	return cl.session.CallTool(callCtx, &mcp.CallToolParams{Name: orig, Arguments: arguments})
}

// StartAutoReconnect launches a background loop that dials every disconnected
// server each interval (0 = RetryInterval) until Close. It returns immediately.
func (h *Hub) StartAutoReconnect(interval time.Duration) {
	if interval <= 0 {
		interval = RetryInterval
	}
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-h.stopCh:
				return
			case <-ticker.C:
				h.mu.Lock()
				type todo struct{ slug, url string }
				var todos []todo
				for slug, cl := range h.clients {
					if cl.session == nil && h.enabled[slug] && !cl.stopped {
						todos = append(todos, todo{slug, cl.def.URL})
					}
				}
				h.mu.Unlock()
				for _, t := range todos {
					if err := h.Dial(context.Background(), t.slug); err == nil {
						h.mu.Lock()
						n := len(h.clients[t.slug].infos)
						h.mu.Unlock()
						log.Printf("MCP[%s] %s: соединение восстановлено, инструментов: %d", t.slug, t.url, n)
					}
				}
			}
		}
	}()
}

// Infos returns a snapshot of per-server status and tool lists.
func (h *Hub) Infos() []ServerInfo {
	h.mu.Lock()
	defer h.mu.Unlock()
	slugs := make([]string, 0, len(h.clients))
	for slug := range h.clients {
		slugs = append(slugs, slug)
	}
	sort.Strings(slugs)
	out := make([]ServerInfo, 0, len(slugs))
	for _, slug := range slugs {
		cl := h.clients[slug]
		out = append(out, ServerInfo{
			Slug:      slug,
			Name:      cl.def.Name,
			URL:       cl.def.URL,
			Enabled:   h.enabled[slug],
			Connected: cl.session != nil,
			Stopped:   cl.stopped,
			Server:    cl.name,
			Version:   cl.version,
			Protocol:  cl.proto,
			Count:     len(cl.infos),
			Tools:     append([]ToolInfo(nil), cl.infos...),
			Error:     cl.lastErr,
		})
	}
	return out
}

// InfoCount returns how many servers are currently connected.
func (h *Hub) InfoCount() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	n := 0
	for _, cl := range h.clients {
		if cl.session != nil {
			n++
		}
	}
	return n
}

// Close stops the background reconnect loop and shuts every session down
// (DELETE /mcp server-side, capped at 5s). Safe to call multiple times.
func (h *Hub) Close() error {
	h.stopOnce.Do(func() { close(h.stopCh) })
	h.mu.Lock()
	cls := make([]*client, 0, len(h.clients))
	for _, cl := range h.clients {
		cls = append(cls, cl)
	}
	for _, cl := range h.clients {
		cl.session = nil
	}
	h.mu.Unlock()
	for _, cl := range cls {
		if cl.session != nil {
			_ = cl.session.Close()
		}
	}
	return nil
}

// describe runs ListTools and converts the result to OpenAI function
// definitions plus UI descriptions.
func describe(ctx context.Context, session *mcp.ClientSession) (tools []api.Tool, infos []ToolInfo, name, version, proto string, err error) {
	res, err := session.ListTools(ctx, nil)
	if err != nil {
		return nil, nil, "", "", "", err
	}
	if si := session.InitializeResult(); si != nil {
		name, version, proto = si.ServerInfo.Name, si.ServerInfo.Version, si.ProtocolVersion
	}
	tools = make([]api.Tool, 0, len(res.Tools))
	infos = make([]ToolInfo, 0, len(res.Tools))
	for _, t := range res.Tools {
		params, perr := schemaToMap(t.InputSchema)
		if perr != nil {
			params = map[string]any{"type": "object"}
		}
		desc := strings.TrimSpace(t.Description)
		tools = append(tools, api.Tool{
			Type: "function",
			Function: api.ToolFunction{
				Name:        t.Name,
				Description: desc,
				Parameters:  params,
			},
		})
		infos = append(infos, ToolInfo{Name: t.Name, Description: firstLine(desc)})
	}
	return tools, infos, name, version, proto, nil
}

// schemaToMap turns the MCP tool input schema (any value marshaling to JSON
// Schema) into the map form used by api.ToolFunction.Parameters.
func schemaToMap(schema any) (map[string]any, error) {
	if schema == nil {
		return map[string]any{"type": "object"}, nil
	}
	b, err := json.Marshal(schema)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	if m == nil {
		m = map[string]any{"type": "object"}
	}
	return m, nil
}

// firstLine returns the first non-empty line of a multi-line description.
func firstLine(s string) string {
	for _, line := range strings.Split(s, "\n") {
		if line = strings.TrimSpace(line); line != "" {
			return line
		}
	}
	return ""
}

// contentText flattens a CallToolResult into the textual form fed back to the
// model: text content items joined by newlines, falling back to the
// structured content JSON.
func contentText(res *mcp.CallToolResult) string {
	var lines []string
	for _, item := range res.Content {
		if tc, ok := item.(*mcp.TextContent); ok && strings.TrimSpace(tc.Text) != "" {
			lines = append(lines, tc.Text)
		}
	}
	if len(lines) > 0 {
		return strings.Join(lines, "\n")
	}
	if res.StructuredContent != nil {
		if b, err := json.Marshal(res.StructuredContent); err == nil {
			return string(b)
		}
	}
	return ""
}

// Default returns the process-wide hub installed by the application (used by
// the web server's /api/mcp), or nil when MCP is disabled.
func Default() *Hub {
	defMu.RLock()
	defer defMu.RUnlock()
	return def
}

// Install sets the process-wide MCP hub.
func Install(h *Hub) {
	defMu.Lock()
	def = h
	defMu.Unlock()
}

// CloseDefault closes the process-wide MCP hub, if any. Safe to call multiple
// times (e.g. from a signal handler and a deferred call).
func CloseDefault() {
	defMu.Lock()
	h := def
	def = nil
	defMu.Unlock()
	if h != nil {
		_ = h.Close()
	}
}

var (
	defMu sync.RWMutex
	def   *Hub
)
